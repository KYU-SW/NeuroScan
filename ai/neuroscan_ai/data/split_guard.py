"""Single entry point for split membership, with the test split sealed.

Training, validation, normalisation analysis and tuning code obtain case IDs only through
``SplitGuard``. Asking for the test split raises unless the guard was explicitly constructed with
``allow_test=True`` (reserved for the final Phase 5 comparison). Every case handed out is recorded
so a run can prove afterwards which patients it touched.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

SEALED_SPLIT = "test"


class SealedSplitAccessError(RuntimeError):
    """Raised when sealed test patients are requested or detected outside the final evaluation."""


class SplitGuard:
    def __init__(self, split_file: str | Path, allow_test: bool = False) -> None:
        self.split_file = Path(split_file)
        data = json.loads(self.split_file.read_text(encoding="utf-8"))
        self.version = data.get("version", self.split_file.stem)
        self._splits: dict[str, list[str]] = {k: list(v) for k, v in data["splits"].items()}
        self._test_ids = frozenset(self._splits.get(SEALED_SPLIT, []))
        self.allow_test = allow_test
        self.accessed: dict[str, set[str]] = {}

    def cases(self, split: str) -> list[str]:
        if split == SEALED_SPLIT and not self.allow_test:
            raise SealedSplitAccessError(
                "test split is sealed until the final 2D vs 2.5D comparison"
            )
        if split not in self._splits:
            raise KeyError(f"unknown split '{split}', available: {sorted(self._splits)}")
        ids = list(self._splits[split])
        self.accessed.setdefault(split, set()).update(ids)
        return ids

    def assert_no_test(self, case_ids: Iterable[str], context: str = "") -> None:
        if self.allow_test:
            return
        leaked = sorted(set(case_ids) & self._test_ids)
        if leaked:
            raise SealedSplitAccessError(f"test patients used in {context or 'run'}: {leaked}")

    def access_report(self) -> dict[str, object]:
        touched = set().union(*self.accessed.values()) if self.accessed else set()
        return {
            "split_file": self.split_file.as_posix(),
            "split_version": self.version,
            "splits_accessed": {k: len(v) for k, v in sorted(self.accessed.items())},
            "test_cases_accessed": sorted(touched & self._test_ids),
        }
