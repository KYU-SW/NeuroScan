"""Public (alias-only) case / component records and no-overwrite writing.

Records never contain an original subject ID, a host path or an account name; ``guard_public``
refuses them. Patient-level records are local outputs under ``results/component_analysis/``,
which is git-ignored.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from neuroscan_ai.brain_extraction.audit import find_host_paths, find_identifiers
from neuroscan_ai.component_analysis.extraction import ComponentAnalysisError

SCHEMA = "component-safety-v1"
RECORD_NAME = "component_record.json"
_ALIAS = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def check_alias(alias: str, forbidden: tuple[str, ...] = ()) -> str:
    if not isinstance(alias, str) or not _ALIAS.match(alias):
        raise ComponentAnalysisError("alias must be lower-case [a-z0-9._-], 1-64 chars")
    if find_identifiers(alias, forbidden):
        raise ComponentAnalysisError("alias looks like an original subject ID")
    return alias


def guard_public(obj: Any, forbidden: tuple[str, ...] = ()) -> None:
    text = json.dumps(obj, default=str)
    if find_identifiers(text, forbidden):
        raise ComponentAnalysisError("record would contain an original identifier")
    if find_host_paths(text):
        raise ComponentAnalysisError("record would contain a host path or account name")


def component_record(
    ident: dict[str, Any],
    dwi: dict[str, Any],
    adc: dict[str, Any],
    adc_status: dict[str, Any],
    boundary: dict[str, Any],
    agreement: dict[str, Any],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        **ident,
        "dwi": dwi,
        "adc": {**adc, **{k: v for k, v in adc_status.items() if k != "adc_status_unsettled"}},
        "brain_mask_boundary": boundary,
        "model_agreement": agreement,
        "warning_codes": sorted({w["code"] for w in warnings}),
        "warnings": warnings,
        "review_state": "unreviewed",
        "reviewer_note": None,
    }


def write_record(out_dir: Path, record: dict[str, Any], forbidden: tuple[str, ...] = ()) -> Path:
    """Write ``<out_dir>/<alias>/component_record.json``; refuses to overwrite."""
    alias = check_alias(record["case"]["alias"], forbidden)
    guard_public(record, forbidden)
    path = Path(out_dir) / alias / RECORD_NAME
    if path.exists():
        raise ComponentAnalysisError("record exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def is_git_ignored(path: Path, repo_root: Path) -> bool:
    rel = Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    p = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "-q", rel], capture_output=True
    )
    return p.returncode == 0
