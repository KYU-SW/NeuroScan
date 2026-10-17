"""Dataset adapters: dataset-specific file discovery only.

Everything downstream (QA, preprocessing, evaluation) works on ``CaseRecord`` and is
dataset-agnostic, so an external dataset under ``data/raw/external/<dataset>/`` only needs
a new adapter registered here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from neuroscan_ai.datasets.base import CaseRecord
from neuroscan_ai.datasets.isles2022 import discover_isles2022

ADAPTERS: dict[str, Callable[[Path], list[CaseRecord]]] = {
    "isles2022": discover_isles2022,
}


def discover(dataset: str, root: str | Path) -> list[CaseRecord]:
    if dataset not in ADAPTERS:
        raise KeyError(f"unknown dataset '{dataset}', registered: {sorted(ADAPTERS)}")
    return ADAPTERS[dataset](Path(root))


__all__ = ["ADAPTERS", "CaseRecord", "discover"]
