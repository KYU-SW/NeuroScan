"""Audit records, hashing, original-ID guard and no-overwrite JSON writing."""

from __future__ import annotations

import datetime
import getpass
import hashlib
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neuroscan_ai.brain_extraction.interfaces import BrainExtractionError

_ID_LIKE = re.compile(r"(?<![A-Za-z0-9])sub-[A-Za-z0-9]+|strokecase\d+", re.I)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_identifiers(text: str, forbidden: Iterable[str] = ()) -> list[str]:
    """Original-ID-like strings and any explicitly forbidden ID found in ``text``."""
    hits = set(_ID_LIKE.findall(text))
    for s in forbidden:
        if s and re.search(rf"(?<![\w-]){re.escape(s)}(?![\w-])", text):
            hits.add(s)
    return sorted(hits)


# Host-path patterns that must never reach a public audit / log (drive letters, user homes, UNC).
_HOST_PATH = re.compile(
    r"(?<![A-Za-z])[A-Za-z]:(?:\\|/)|(?:\\|/)(?:Users|home)(?:\\|/)|\\\\\w", re.I
)
WORKDIR_TOKEN = "<WORKDIR>"
REDACTED = "[REDACTED]"


def _user_names() -> list[str]:
    names = {os.environ.get(k, "") for k in ("USERNAME", "USER", "LOGNAME")}
    try:
        names.add(getpass.getuser())
    except Exception:  # noqa: BLE001
        pass
    names.add(Path.home().name)
    return sorted(n for n in names if n)


def find_host_paths(text: str) -> list[str]:
    """Host-path fragments and local account names found in ``text``."""
    hits = [m.group(0) for m in _HOST_PATH.finditer(text)]
    hits += [
        u
        for u in _user_names()
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(u)}(?![A-Za-z0-9])", text)
    ]
    return hits


def sanitize_paths(obj: Any, workdirs: Iterable[Path | str] = ()) -> Any:
    """Copy of ``obj`` with work-folder paths replaced by ``<WORKDIR>`` and any remaining host
    path replaced by ``[REDACTED]`` (container paths such as ``/data/...`` are kept)."""
    variants: set[str] = set()
    for w in workdirs:
        w = Path(w)
        for v in (str(w), str(w.resolve()), w.as_posix(), w.resolve().as_posix()):
            variants.add(v)
    ordered = sorted(variants, key=len, reverse=True)

    def clean(s: str) -> str:
        for v in ordered:
            s = s.replace(v, WORKDIR_TOKEN)
        s = s.replace(WORKDIR_TOKEN + "\\", WORKDIR_TOKEN + "/")
        s = re.sub(r"(?<![A-Za-z])[A-Za-z]:(?:\\|/)[^\s\"']*", REDACTED, s)
        s = re.sub(r"(?:\\|/)(?:Users|home)(?:\\|/)[^\s\"']*", REDACTED, s, flags=re.I)
        s = re.sub(r"\\\\\w[^\s\"']*", REDACTED, s)  # UNC paths
        for u in _user_names():
            s = re.sub(rf"(?<![A-Za-z0-9]){re.escape(u)}(?![A-Za-z0-9])", REDACTED, s)
        return s

    def walk(x: Any) -> Any:
        if isinstance(x, str):
            return clean(x)
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list | tuple):
            return [walk(v) for v in x]
        return x

    return walk(obj)


def guard_public(obj: Any, forbidden: Iterable[str] = ()) -> None:
    """Refuse any public output that would contain an original ID or a host path."""
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    hits = find_identifiers(text, forbidden)
    if hits:
        raise BrainExtractionError(f"output would contain {len(hits)} original identifier(s)")
    if find_host_paths(text):
        raise BrainExtractionError("output would contain a host path or account name")


def write_json_new(path: Path, obj: Any) -> None:
    if path.exists():
        raise BrainExtractionError(f"{path.name} exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def build_audit(
    *,
    alias: str,
    tool: str,
    tool_version: str,
    spec: dict[str, Any],
    command: list[str] | None,
    executed_external: bool,
    mock: bool,
    transforms: list[dict[str, Any]],
    input_sha256: dict[str, str],
    output_sha256: dict[str, str],
    status: str,
    execution_error: str | None,
) -> dict[str, Any]:
    return {
        "alias": alias,
        "tool": tool,
        "tool_version": tool_version,
        "policy_input": "Policy A: TRACE/DWI only; the same mask is applied to the ADC",
        "spec": spec,
        "command": command,
        "external_tool_executed": executed_external,
        "mock_runner": mock,
        "orientation_transforms": transforms,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "status": status,
        "execution_error": execution_error,
        "not_performed": [
            "mask_resampling",
            "mask_dilation_or_repair",
            "manual_mask_editing",
            "adc_value_change",
            "case_specific_parameters",
            "model_inference",
        ],
        "software": {"nibabel": nib.__version__, "numpy": np.__version__},
        "completed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
