"""v2 baseline preprocessing ``preprocess_v2_isles_1mm_inplane_v1`` (independent of v1 code).

1. Validate the original grid: DWI, ADC and GT are 3D, share shape and affine (1e-3 mm), finite,
   GT is decoded with ``binary_mask_decode_v1_atol_1e_6``: NIfTI-scaled (decoded) values within
   1e-6 of 0 or 1 are snapped, anything else is refused (no > 0.5 threshold, no get_unscaled()).
2. Brain mask ``skull_stripped_background_isles_v2`` on the ORIGINAL grid: DWI voxels that differ
   from the constant background value (most common of the 8 corner values), 5-60 % guard.
   Used only for input normalisation, QA, boundary distance and the gated comparison.
3. One 1 mm in-plane target grid per case (``geometry.inplane_target_grid``); slice axis kept;
   per-axis size with the integer-snap rule recorded (``geometry.inplane_target_sizes``).
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

from neuroscan_ai.component_analysis.warnings import (
    THICK_SLICE_THICKNESS_MM,
    THICK_Z_SPACING_MM,
)
from neuroscan_ai.v2.geometry import (
    BINARY_DECODE_ATOL,
    BINARY_DECODE_POLICY,
    TARGET_INPLANE_MM,
    TARGET_SIZE_SNAP_ATOL,
    TARGET_SIZE_SNAP_RTOL,
    GeometryError,
    Grid,
    decode_binary_mask,
    inplane_target_grid,
    inplane_target_sizes,
    resample,
    resample_mask,
    same_grid,
    validate_grid,
)
from neuroscan_ai.v2.safety_metrics import (
    GT_CASE_STATUS,
    SLICE_SUBGROUP_POLICY,
    THICK_SUBGROUP_ABOVE_MM,
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
        gt_bin, gt_decode = decode_binary_mask(arrays["gt"], "GT")
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
        "target_size": inplane_target_sizes(original, target_mm),
        "gt_decode": gt_decode,
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
        np.asanyarray(imgs["gt"].dataobj),  # decoded values (NIfTI scaling applied)
        {k: im.affine for k, im in imgs.items()},
    )


# ---------------------------------------------------------------- config / code policy check
class PolicyConfigError(ValueError):
    pass


SIZE_POLICY = ["integer_snap_if_close", "otherwise_ceil"]
_ISLES_SUBGROUP_KEY = "isles_subgroup"


def policy_config_errors(cfg: dict) -> list[str]:
    """Differences between a v2 baseline config and the policies implemented in code.

    Checks the GT decode policy, the target-grid size policy and the two SEPARATE thick-slice
    rules (ISLES analysis subgroup > 3.0 mm; provisional component-safety warning > 5.0 mm or
    thickness >= 5.0 mm). An empty list means config and code agree."""
    errs: list[str] = []

    def expect(path: str, actual, expected) -> None:
        if actual != expected:
            errs.append(f"{path}: config {actual!r} != code {expected!r}")

    try:
        pre = cfg["preprocessing"]
        gd, tg = pre["gt_decode"], pre["target_grid"]
        thick = cfg["safety_metrics"]["thick_slice_policies"]
    except (KeyError, TypeError) as exc:
        return [f"missing policy block: {exc}"]
    expect("gt_decode.policy_id", gd.get("policy_id"), BINARY_DECODE_POLICY)
    expect("gt_decode.decoded_values", gd.get("decoded_values"), True)
    expect("gt_decode.use_get_unscaled", gd.get("use_get_unscaled"), False)
    expect("gt_decode.use_threshold_binarization", gd.get("use_threshold_binarization"), False)
    expect("gt_decode.zero_atol", gd.get("zero_atol"), BINARY_DECODE_ATOL)
    expect("gt_decode.one_atol", gd.get("one_atol"), BINARY_DECODE_ATOL)
    expect("gt_decode.reject_other_values", gd.get("reject_other_values"), True)
    expect("gt_decode.reject_nan_inf", gd.get("reject_nan_inf"), True)
    expect("target_grid.inplane_spacing_mm", tg.get("inplane_spacing_mm"), [TARGET_INPLANE_MM] * 2)
    expect("target_grid.preserve_z_spacing", tg.get("preserve_z_spacing"), True)
    expect("target_grid.size_policy", tg.get("size_policy"), SIZE_POLICY)
    expect("target_grid.integer_snap_rtol", tg.get("integer_snap_rtol"), TARGET_SIZE_SNAP_RTOL)
    expect("target_grid.integer_snap_atol", tg.get("integer_snap_atol"), TARGET_SIZE_SNAP_ATOL)

    if cfg.get("gt_case_status") != GT_CASE_STATUS:
        errs.append(
            f"gt_case_status: config {cfg.get('gt_case_status')!r} != code {GT_CASE_STATUS!r}"
        )
    isles = thick.get(_ISLES_SUBGROUP_KEY)
    others = {k: v for k, v in thick.items() if k != _ISLES_SUBGROUP_KEY}
    if not isinstance(isles, dict) or len(others) != 1:
        return errs + ["thick_slice_policies: need the ISLES subgroup and one separate warning"]
    ((wkey, warn),) = others.items()
    expect("isles_subgroup.policy_id", isles.get("policy_id"), SLICE_SUBGROUP_POLICY)
    expect(
        "isles_subgroup.effective_z_spacing_mm_gt",
        isles.get("effective_z_spacing_mm_gt"),
        THICK_SUBGROUP_ABOVE_MM,
    )
    expect("isles_subgroup.purpose", isles.get("purpose"), "analysis_subgroup")
    if {"slice_thickness_mm_gte", "provisional"} & set(isles):
        errs.append("isles_subgroup must not carry warning fields (rules are separate)")
    expect(
        f"{wkey}.effective_z_spacing_mm_gt",
        warn.get("effective_z_spacing_mm_gt"),
        THICK_Z_SPACING_MM,
    )
    expect(
        f"{wkey}.slice_thickness_mm_gte",
        warn.get("slice_thickness_mm_gte"),
        THICK_SLICE_THICKNESS_MM,
    )
    expect(f"{wkey}.provisional", warn.get("provisional"), True)
    wid = warn.get("policy_id")
    if not isinstance(wid, str) or not wid or wid == isles.get("policy_id"):
        errs.append(f"{wkey}.policy_id must be its own non-empty policy id")
    if "purpose" in warn:
        errs.append(f"{wkey} must not reuse the analysis-subgroup purpose field")
    return errs


def validate_policy_config(cfg: dict) -> None:
    """Refuse a config whose preprocessing / thick-slice policies differ from the code."""
    errs = policy_config_errors(cfg)
    if errs:
        raise PolicyConfigError("; ".join(errs))
