"""API storage root (storage/): uploads/, runs/, temp/. Only storage-relative paths are stored in
the database; every path is re-resolved and confined to its area before use, and symbolic links
are refused anywhere between the area root and the file."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path, PurePosixPath

from neuroscan_api.errors import ApiError

AREAS = ("uploads", "runs", "temp", "research")  # research: Phase 7-B bundles


class StorageError(Exception):
    """A stored path is unsafe or missing; callers map it to a 404 without details."""


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        for a in AREAS:
            (self.root / a).mkdir(parents=True, exist_ok=True)

    def area(self, name: str) -> Path:
        if name not in AREAS:
            raise StorageError("unknown storage area")
        return self.root / name

    @property
    def uploads(self) -> Path:
        return self.root / "uploads"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def temp(self) -> Path:
        return self.root / "temp"

    def resolve(self, area: str, rel: str, *, must_exist: bool = True) -> Path:
        base = self.area(area)
        pure = PurePosixPath(rel)
        if not rel or pure.is_absolute() or ".." in pure.parts or "\\" in rel or ":" in rel:
            raise StorageError("unsafe relative path")
        candidate = base.joinpath(*pure.parts)
        current = base
        for part in pure.parts:  # refuse symlinks / junctions on the way down
            current = current / part
            if os.path.islink(current) or (current.exists() and _is_junction(current)):
                raise StorageError("symbolic link refused")
        if must_exist:
            try:
                real = candidate.resolve(strict=True)
            except (FileNotFoundError, OSError):
                raise StorageError("missing") from None
        else:
            real = candidate.resolve()
        if not real.is_relative_to(base.resolve()):
            raise StorageError("outside storage area")
        return real

    def relative(self, area: str, path: Path) -> str:
        base = self.area(area).resolve()
        real = Path(path).resolve()
        if not real.is_relative_to(base):
            raise StorageError("outside storage area")
        return real.relative_to(base).as_posix()

    def ensure_free_space(self, min_free: int) -> None:
        if shutil.disk_usage(self.root).free < min_free:
            raise insufficient_storage()

    def remove_tree(self, area: str, rel: str) -> int:
        """Delete a directory inside an area (never follows symlinks); returns files removed."""
        try:
            target = self.resolve(area, rel)
        except StorageError:
            return 0
        if target == self.area(area).resolve() or not target.is_dir():
            return 0
        count = sum(1 for p in target.rglob("*") if p.is_file())
        shutil.rmtree(target)
        return count


def _is_junction(p: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction and is_junction(p))


def insufficient_storage() -> ApiError:
    return ApiError(507, "INSUFFICIENT_STORAGE", "Not enough storage space.", retryable=True)


def is_no_space(exc: OSError) -> bool:
    return exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1))
