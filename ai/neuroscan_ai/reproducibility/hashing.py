"""Hashes and canonical forms used by run manifests.

- ``sha256_file``: raw bytes (binary files, checkpoints, NIfTI).
- ``sha256_text_lf``: text with CRLF normalised to LF, i.e. the bytes Git stores, so a Windows
  checkout and a Linux checkout of the same config / split give the same hash.
- ``canonical_json_sha256``: sorted keys, no whitespace, UTF-8; used to link manifests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text_lf(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def file_hashes(path: str | Path) -> dict[str, str]:
    """Raw and LF-canonical SHA-256 of one file."""
    return {"sha256": sha256_file(path), "sha256_lf": sha256_text_lf(path)}


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


def canonical_json_sha256(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def alias_list_sha256(aliases: Iterable[str], policy: str) -> str:
    """Hash of the case alias list in the order defined by ``policy`` (the case key policy).

    The policy name is part of the hashed text, so the same alias set under a different ordering
    rule gives a different hash; case keys are only comparable when this hash is equal.
    """
    if policy != "sorted_alias_position_v1":
        raise ValueError(f"unsupported case key policy '{policy}'")
    ordered = sorted(set(aliases))
    return sha256_bytes((policy + "\n" + "\n".join(ordered)).encode("utf-8"))
