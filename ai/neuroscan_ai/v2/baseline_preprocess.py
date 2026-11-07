"""v2 baseline preprocessing ``preprocess_v2_isles_1mm_inplane_v1`` (independent of v1 code).

1. Validate the original grid: DWI, ADC and GT are 3D, share shape and affine (1e-3 mm), finite,
   GT contains only labels 0/1 (anything else is refused).
2. Brain mask ``skull_stripped_background_isles_v2`` on the ORIGINAL grid: DWI voxels that differ
   from the constant background value (most common of the 8 corner values), 5-60 % guard.
   Used only for input normalisation, QA, boundary distance and the gated comparison.
3. One 1 mm in-plane target grid per case (``geometry.inplane_target_grid``); slice axis kept.
4. DWI and ADC: linear; GT and brain mask: nearest. All four on the same target grid.
5. Model input: per-case z-score of DWI and ADC over the model-grid brain mask (normalisation only;
   stored ADC values are never converted).

Original arrays and files are never modified; the original grid, target grid, spacings and
affines are recorded so that outputs can be restored exactly onto the original grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from neuroscan_ai.v2.geometry import (
    TARGET_INPLANE_MM,
    GeometryError,
    Grid,
    as_binary_mask,
    inplane_target_grid,
    resample,
    resample_mask,
    same_grid,
    validate_grid,
)

PREPROCESS_POLICY = "preprocess_v2_isles_1mm_inplane_v1"
BRAIN_MASK_POLICY = "skull_stripped_background_isles_v2"
BRAIN_FRACTION_RANGE = (0.05, 0.60)
IMAGE_INTERPOLATION = "linear"
LABEL_INTERPOLATION = "nearest"


class PreprocessError(ValueError):
    pass


@dataclass
class PreparedCase:
    image: np.ndarray  # (2, X, Y, Z) float32 on the model grid: dwi_z, adc_z
    label: np.ndarray  # (X, Y, Z) bool on the model grid
    brain: np.ndarray  # (X, Y, Z) bool on the model grid
    model_grid: Grid
    original_grid: Grid
    original_brain: np.ndarray  # (x, y, z) bool on the original grid
    record: dict = field(default_factory=dict)


def background_brain_mask(dwi: np.ndarray) -> np.ndarray:
    corners = dwi[np.ix_(*[[0, s - 1] for s in dwi.shape[:3]])].ravel()
    values, counts = np.unique(corners, return_counts=True)
    bg = float(values[np.argmax(counts)])
    mask = ~np.isclose(dwi, bg, rtol=0.0, atol=1e-6 * max(1.0, abs(bg)))
    frac = float(mask.mean())
    lo, hi = BRAIN_FRACTION_RANGE
    if not lo <= frac <= hi:
        raise PreprocessError(f"brain mask covers {frac:.1%} of the volume (expected 5-60%)")
    return mask


def _zscore(x: np.ndarray, brain: np.ndarray) -> np.ndarray:
    v = x[brain]
    if v.size == 0:
        raise PreprocessError("empty brain mask on the model grid")
    sd = float(v.std())
    return ((x - float(v.mean())) / (sd if sd > 1e-8 else 1.0)).astype(np.float32)


def prepare_arrays(
    dwi: np.ndarray,
    adc: np.ndarray,
    gt: np.ndarray,
    affines: dict[str, np.ndarray],
    target_mm: float = TARGET_INPLANE_MM,
) -> PreparedCase:
    """Prepare one case from in-memory arrays (``affines`` keys: dwi, adc, gt)."""
    arrays = {"dwi": np.asarray(dwi), "adc": np.asarray(adc), "gt": np.asarray(gt)}
    try:
        grids = {k: validate_grid(a.shape, affines[k]) for k, a in arrays.items()}
    except (GeometryError, KeyError) as exc:
        raise PreprocessError(f"invalid original grid: {exc}") from exc
    original = grids["dwi"]
    for k in ("adc", "gt"):
        if not same_grid(grids[k], original):
            raise PreprocessError(f"{k} is not on the DWI grid")
    if not (np.isfinite(arrays["dwi"]).all() and np.isfinite(arrays["adc"]).all()):
        raise PreprocessError("non-finite DWI / ADC values")
    try:
        gt_bin = as_binary_mask(arrays["gt"], "GT")
    except GeometryError as exc:
        raise PreprocessError(str(exc)) from exc

    original_brain = background_brain_mask(arrays["dwi"])
    target = inplane_target_grid(original, target_mm)
    dwi_t = resample(arrays["dwi"], original, target, order=1)
    adc_t = resample(arrays["adc"], original, target, order=1)
    gt_t = resample_mask(gt_bin, original, target)
    brain_t = resample_mask(original_brain, original, target)
    image = np.stack([_zscore(dwi_t, brain_t), _zscore(adc_t, brain_t)])
    record = {
        "preprocess_policy": PREPROCESS_POLICY,
        "brain_mask_policy": BRAIN_MASK_POLICY,
        "brain_mask_grid": "original",
        "target_inplane_mm": float(target_mm),
        "slice_axis": 2,
        "slice_axis_resampled": False,
        "interpolation": {
            "dwi": IMAGE_INTERPOLATION,
            "adc": IMAGE_INTERPOLATION,
            "gt": LABEL_INTERPOLATION,
            "brain_mask": LABEL_INTERPOLATION,
        },
        "resampled": not same_grid(original, target, atol=1e-6),
        "original_grid": original.record(),
        "model_grid": target.record(),
        "brain_fraction_original": round(float(original_brain.mean()), 6),
        "gt_voxels_original": int(gt_bin.sum()),
        "gt_voxels_model_grid": int(gt_t.sum()),
        "normalisation": "per-case z-score over the model-grid brain mask (inputs only)",
        "adc_values_converted": False,
    }
    return PreparedCase(image, gt_t, brain_t, target, original, original_brain, record)


def load_and_prepare(dwi_path: Path, adc_path: Path, gt_path: Path) -> PreparedCase:
    """Read three NIfTI files (read-only) and prepare them. Files are never written."""
    imgs = {k: nib.load(str(p)) for k, p in (("dwi", dwi_path), ("adc", adc_path), ("gt", gt_path))}
    for k, im in imgs.items():
        if len(im.shape) != 3:
            raise PreprocessError(f"{k} is not 3D")
    return prepare_arrays(
        imgs["dwi"].get_fdata(dtype=np.float32),
        imgs["adc"].get_fdata(dtype=np.float32),
        np.asanyarray(imgs["gt"].dataobj),
        {k: im.affine for k, im in imgs.items()},
    )
