"""Research source adapters: where a dataset keeps a case's images, ground truth and a stored
reference prediction. Dataset-specific file rules live only here; difference maps, metrics,
matching and the research API are dataset independent.

Every adapter must check split membership on the case *key string* before any path is built
or any file is opened: only the validation split is accepted, the sealed test split is refused.
The case key never leaves the adapter / the local alias manifest (public outputs use the alias).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from neuroscan_ai.data.split_guard import SealedSplitAccessError, SplitGuard

ALIAS = re.compile(r"^val-\d{3}$")
RESEARCH_SPLIT = "val"


class ResearchSourceError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ResearchSource:
    dataset_id: str
    split: str
    alias: str
    base_image: Path  # DWI (model input 1, viewer base)
    secondary_image: Path | None  # ADC
    ground_truth: Path
    reference_prediction: Path | None  # stored prediction to reproduce (e.g. Phase 4)
    source_case_key: str  # internal only: alias manifest, never in public outputs


class ResearchSourceAdapter(Protocol):
    dataset_id: str

    def resolve(self, case_key: str, alias: str) -> ResearchSource: ...


def check_alias(alias: str) -> str:
    if not ALIAS.fullmatch(alias):
        raise ResearchSourceError("INVALID_ALIAS", "Alias must look like val-001.")
    return alias


class Isles2022ResearchAdapter:
    """ISLES 2022 BIDS layout (same naming as ``datasets.isles2022``)."""

    dataset_id = "isles2022"
    SESSION = "ses-0001"
    CASE_KEY = re.compile(r"^sub-strokecase\d{4}$")

    def __init__(
        self, data_root: Path, split_file: Path, reference_predictions: Path | None = None
    ) -> None:
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.reference_predictions = reference_predictions

    def resolve(self, case_key: str, alias: str) -> ResearchSource:
        check_alias(alias)
        if not self.CASE_KEY.fullmatch(case_key):
            raise ResearchSourceError("INVALID_CASE", "Unknown case key format.")
        guard = SplitGuard(self.split_file)
        try:  # string-only membership checks: nothing has been opened yet
            guard.assert_no_test([case_key], "research case registration")
        except SealedSplitAccessError:
            raise ResearchSourceError(
                "SEALED_TEST_CASE", "Sealed evaluation (test split) data cannot be used."
            ) from None
        if case_key not in guard.cases(RESEARCH_SPLIT):
            raise ResearchSourceError(
                "NOT_VALIDATION", "Only validation-split cases can be research cases."
            )
        prefix = f"{case_key}_{self.SESSION}"
        sub = self.data_root / case_key / self.SESSION
        ref = None
        if self.reference_predictions is not None:
            ref = self.reference_predictions / f"{case_key}_pred.nii.gz"
        return ResearchSource(
            dataset_id=self.dataset_id,
            split=RESEARCH_SPLIT,
            alias=alias,
            base_image=sub / "dwi" / f"{prefix}_dwi.nii.gz",
            secondary_image=sub / "dwi" / f"{prefix}_adc.nii.gz",
            ground_truth=self.data_root
            / "derivatives"
            / case_key
            / self.SESSION
            / f"{prefix}_msk.nii.gz",
            reference_prediction=ref,
            source_case_key=case_key,
        )
