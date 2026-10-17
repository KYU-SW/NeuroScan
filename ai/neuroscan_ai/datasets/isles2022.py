"""ISLES 2022 (BIDS) file discovery. The only ISLES-specific code in the pipeline."""

from __future__ import annotations

from pathlib import Path

from neuroscan_ai.datasets.base import CaseRecord

SESSION = "ses-0001"


def discover_isles2022(root: Path) -> list[CaseRecord]:
    """Return one record per ``sub-strokecase*`` folder, sorted by case ID.

    Paths are built from the BIDS naming convention and are not checked for existence here;
    missing files are reported by dataset QA instead of silently dropping the case.
    """
    records = []
    for sub in sorted(p for p in root.glob("sub-strokecase*") if p.is_dir()):
        case = sub.name
        prefix = f"{case}_{SESSION}"
        records.append(
            CaseRecord(
                dataset="isles2022",
                case_id=case,
                images={
                    "dwi": sub / SESSION / "dwi" / f"{prefix}_dwi.nii.gz",
                    "adc": sub / SESSION / "dwi" / f"{prefix}_adc.nii.gz",
                },
                label=root / "derivatives" / case / SESSION / f"{prefix}_msk.nii.gz",
                extra={"flair": sub / SESSION / "anat" / f"{prefix}_FLAIR.nii.gz"},
            )
        )
    return records
