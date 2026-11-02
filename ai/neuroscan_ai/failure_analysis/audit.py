"""Access guards and audit helpers of the metrics-only stage.

``no_image_access`` replaces ``nibabel.load`` for the duration of the run: any attempt to open a
NIfTI (DWI, ADC, GT or prediction) is counted and refused. ``snapshot`` hashes every file of the
Final Test folder so the run can prove afterwards that nothing there changed.
"""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class MetricsOnlyViolation(RuntimeError):
    pass


@contextlib.contextmanager
def no_image_access(counter: dict[str, int]) -> Iterator[None]:
    import nibabel

    real = nibabel.load

    def refuse(*_a: Any, **_k: Any):
        counter["nifti_open_attempts"] = counter.get("nifti_open_attempts", 0) + 1
        raise MetricsOnlyViolation("metrics-only mode: opening a NIfTI file is not allowed")

    nibabel.load = refuse
    try:
        yield
    finally:
        nibabel.load = real


def snapshot(folder: Path) -> dict[str, str]:
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            out[p.relative_to(folder).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def torch_loaded() -> bool:
    return "torch" in sys.modules


def find_ids(folder: Path, ids: Iterable[str], skip: str = "_internal") -> list[str]:
    """Public files (everything outside ``skip``) that contain any original case ID."""
    ids = [i for i in ids if i]
    hits = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or skip in p.relative_to(folder).parts:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if any(i in text for i in ids) or any(i in p.name for i in ids):
            hits.append(p.relative_to(folder).as_posix())
    return hits


def git(root: Path, *args: str) -> str | None:
    p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def git_ignored(root: Path, path: Path) -> bool | None:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None  # outside the repository
    p = subprocess.run(["git", "check-ignore", "-q", rel.as_posix() + "/probe"], cwd=root)
    return p.returncode == 0
