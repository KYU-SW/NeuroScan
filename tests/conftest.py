from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroscan_ai.datasets.base import CaseRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
ISLES_ROOT = REPO_ROOT / "data" / "raw" / "ISLES2022"


def las_affine(spacing=(2.0, 2.0, 2.0), origin=(90.0, -120.0, -60.0)) -> np.ndarray:
    """Affine whose first array axis points Left (like ISLES'22), i.e. orientation LAS."""
    aff = np.diag([-spacing[0], spacing[1], spacing[2], 1.0])
    aff[:3, 3] = origin
    return aff


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray, dtype=None) -> Path:
    img = nib.Nifti1Image(data if dtype is None else data.astype(dtype), affine)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def make_synthetic_case(
    root: Path, case_id: str = "case001", spacing=(1.5, 1.5, 4.0), shape=(20, 18, 6), lesion=True
) -> CaseRecord:
    rng = np.random.default_rng(0)
    affine = las_affine(spacing)
    dwi = rng.random(shape) * 100
    adc = rng.random(shape) * 1000
    mask = np.zeros(shape)
    if lesion:
        mask[5:12, 6:11, 2:4] = 0.9999999997671694  # mimics ISLES scaling artefact
    d = root / case_id
    return CaseRecord(
        dataset="synthetic",
        case_id=case_id,
        images={
            "dwi": save_nifti(d / "dwi.nii.gz", dwi, affine),
            "adc": save_nifti(d / "adc.nii.gz", adc, affine, np.int16),
        },
        label=save_nifti(d / "msk.nii.gz", mask, affine),
    )


@pytest.fixture
def isles_root() -> Path:
    if not (ISLES_ROOT / "dataset_description.json").exists():
        pytest.skip("ISLES 2022 raw data not available")
    return ISLES_ROOT
