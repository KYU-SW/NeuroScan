"""v2 smoke data access (policy ``preprocess_v2_isles_smoke_v1``), independent of v1 code.

- Original grid only: DWI, ADC and GT are read as stored and must share shape and affine
  (1e-3 mm); nothing is resampled or reoriented. Slices are taken along the last array axis.
- ADC values are never converted: model inputs are per-case z-scores computed over the brain mask
  (an input normalisation only; stored values are untouched).
- Brain mask ``skull_stripped_background_v2_smoke``: voxels whose DWI differs from the constant
  background value (the most common of the 8 corner values), with a 5-60 % fraction guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

PREPROCESS_POLICY = "preprocess_v2_isles_smoke_v1"
BRAIN_MASK_POLICY = "skull_stripped_background_v2_smoke"
BRAIN_FRACTION_RANGE = (0.05, 0.60)
AFFINE_ATOL_MM = 1e-3


class CaseDataError(ValueError):
    pass


@dataclass(frozen=True)
class CaseFiles:
    dwi: Path
    adc: Path
    label: Path


@dataclass
class CaseArrays:
    image: np.ndarray  # (2, H, W, Z) float32: normalised DWI, ADC
    label: np.ndarray  # (H, W, Z) bool
    brain: np.ndarray  # (H, W, Z) bool
    affine: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.label.shape)


def isles_case_files(raw_root: Path, case_id: str) -> CaseFiles:
    """ISLES 2022 raw layout (BIDS): dwi / adc under the case, mask under derivatives."""
    base = Path(raw_root) / case_id / "ses-0001" / "dwi"
    files = CaseFiles(
        base / f"{case_id}_ses-0001_dwi.nii.gz",
        base / f"{case_id}_ses-0001_adc.nii.gz",
        Path(raw_root) / "derivatives" / case_id / "ses-0001" / f"{case_id}_ses-0001_msk.nii.gz",
    )
    for p in (files.dwi, files.adc, files.label):
        if not p.is_file():
            raise CaseDataError("case file missing")
    return files


def brain_mask(dwi: np.ndarray) -> np.ndarray:
    corners = dwi[np.ix_(*[[0, s - 1] for s in dwi.shape[:3]])].ravel()
    values, counts = np.unique(corners, return_counts=True)
    bg = float(values[np.argmax(counts)])
    mask = ~np.isclose(dwi, bg, rtol=0.0, atol=1e-6 * max(1.0, abs(bg)))
    frac = float(mask.mean())
    lo, hi = BRAIN_FRACTION_RANGE
    if not lo <= frac <= hi:
        raise CaseDataError(f"brain mask covers {frac:.1%} of the volume (expected 5-60%)")
    return mask


def _zscore(x: np.ndarray, brain: np.ndarray) -> np.ndarray:
    v = x[brain]
    sd = float(v.std())
    return ((x - float(v.mean())) / (sd if sd > 1e-8 else 1.0)).astype(np.float32)


def load_case(files: CaseFiles) -> CaseArrays:
    imgs = {k: nib.load(str(getattr(files, k))) for k in ("dwi", "adc", "label")}
    ref = imgs["dwi"]
    for k, im in imgs.items():
        if len(im.shape) != 3:
            raise CaseDataError(f"{k} is not 3D")
        if tuple(im.shape) != tuple(ref.shape) or not np.allclose(
            im.affine, ref.affine, atol=AFFINE_ATOL_MM
        ):
            raise CaseDataError(f"{k} is not on the DWI grid")
    dwi = ref.get_fdata(dtype=np.float32)
    adc = imgs["adc"].get_fdata(dtype=np.float32)
    if not (np.isfinite(dwi).all() and np.isfinite(adc).all()):
        raise CaseDataError("non-finite DWI / ADC values")
    label = imgs["label"].get_fdata(dtype=np.float32) > 0.5
    brain = brain_mask(dwi)
    image = np.stack([_zscore(dwi, brain), _zscore(adc, brain)])
    return CaseArrays(image, label, brain, ref.affine.copy())


def pad_to_multiple(x: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Zero-pad the last two axes (H, W) at the end up to a multiple; returns the original size."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    pad = [(0, 0)] * (x.ndim - 2) + [(0, ph), (0, pw)]
    return np.pad(x, pad), (h, w)
