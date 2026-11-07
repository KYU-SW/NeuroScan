"""Restore model-grid outputs onto the original grid (v2 baseline output policy).

- Probability: model grid -> original grid by linear interpolation (values stay in [0, 1]).
- Threshold is applied on the ORIGINAL grid: ``ungated = probability >= threshold``.
- Brain mask roles (fixed):
  * ORIGINAL-grid mask (built directly on the original grid): final gated prediction, brain-mask
    boundary analysis and original-space QA. It is the only mask used in original space.
  * MODEL-grid mask (1 mm): input normalisation and model-input QA only.
  * Nearest-neighbour ROUND-TRIP mask (model grid -> original grid): XOR comparison with the
    original mask and round-trip interpolation QA only. It never replaces or overwrites the
    original mask, and a round-trip mismatch is recorded, never silently corrected.
- ``gated = ungated AND brain_mask``. Ungated is the official output; gated is a comparison only.
  Predictions outside the brain mask are counted and kept, never removed from the ungated output.
- Case-level brain-mask boundary record on the ungated output (original grid, reusing
  ``component_analysis.boundary``): ``outside_brain_mask_voxels``, ``outside_brain_mask_ratio``
  (outside / ungated voxels), ``touches_image_edge``, ``min_distance_to_brain_boundary_mm`` and
  ``boundary_warnings`` (BOUNDARY_CONTACT, EXTENDS_OUTSIDE_BRAINMASK; provisional, not clinical).
- Every restored array is checked against the original shape, affine, orientation and spacing;
  masks must be exactly 0/1. Nothing is repaired to pass a check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuroscan_ai.component_analysis.boundary import BrainBoundary
from neuroscan_ai.component_analysis.warnings import (
    BOUNDARY_CONTACT,
    EXTENDS_OUTSIDE_BRAINMASK,
)
from neuroscan_ai.v2.baseline_preprocess import PreparedCase
from neuroscan_ai.v2.geometry import GeometryError, Grid, check_on_grid, resample, resample_mask

PROBABILITY_INTERPOLATION = "linear"
MASK_INTERPOLATION = "nearest"
OFFICIAL_OUTPUT = "ungated"
_ROUND_TOL = 1e-9


BRAIN_MASK_ROLES = {
    "original_grid": ["final_gating", "boundary_analysis", "original_space_qa"],
    "model_grid": ["input_normalisation", "model_input_qa"],
    "nearest_roundtrip": ["xor_vs_original", "roundtrip_interpolation_qa"],
}


class OutputGridError(ValueError):
    pass


def roundtrip_mask_qa(original: np.ndarray, roundtrip: np.ndarray) -> dict:
    """XOR QA between the original-grid brain mask and its nearest round trip (read-only).
    ``roundtrip_mask_xor_ratio`` = XOR voxels / original-mask voxels."""
    a, b = np.asarray(original, dtype=bool), np.asarray(roundtrip, dtype=bool)
    if a.shape != b.shape:
        raise OutputGridError("round-trip mask is not on the original grid")
    xor = int((a ^ b).sum())
    na, nb = int(a.sum()), int(b.sum())
    return {
        "roundtrip_mask_xor_voxels": xor,
        "roundtrip_mask_xor_ratio": round(xor / na, 8) if na else None,
        "roundtrip_mask_dice": round(2.0 * int((a & b).sum()) / (na + nb), 8) if na + nb else None,
        "roundtrip_mask_exact": xor == 0,
    }


def boundary_record(ungated: np.ndarray, brain: np.ndarray, grid: Grid) -> dict:
    """Case-level brain-mask boundary record of the ungated output (measured, never cut)."""
    u = np.asarray(ungated, dtype=bool)
    if not u.any():
        return {
            "outside_brain_mask_voxels": 0,
            "outside_brain_mask_ratio": None,
            "touches_image_edge": False,
            "min_distance_to_brain_boundary_mm": None,
            "boundary_warnings": [],
        }
    m = BrainBoundary(brain, grid.spacing).metrics(u)
    warnings = []
    if m["touches_boundary_1_inplane_voxel"] or m["touches_boundary_3mm"]:
        warnings.append(BOUNDARY_CONTACT)
    if m["voxels_outside_brain_mask"] > 0:
        warnings.append(EXTENDS_OUTSIDE_BRAINMASK)
    return {
        "outside_brain_mask_voxels": m["voxels_outside_brain_mask"],
        "outside_brain_mask_ratio": m["outside_brain_mask_ratio"],
        "touches_image_edge": m["touches_image_edge"],
        "min_distance_to_brain_boundary_mm": m["min_distance_to_brain_boundary_mm"],
        "boundary_warnings": warnings,
    }


@dataclass
class OriginalGridOutputs:
    probability: np.ndarray  # float32 in [0, 1]
    ungated: np.ndarray  # uint8 0/1 (official)
    brain_mask: np.ndarray  # uint8 0/1
    gated: np.ndarray  # uint8 0/1 (comparison only)
    grid: Grid
    record: dict


def restore_probability(
    prob_model: np.ndarray, model_grid: Grid, original_grid: Grid
) -> np.ndarray:
    p = np.asarray(prob_model, dtype=np.float64)
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise OutputGridError("model-grid probability must be finite and within [0, 1]")
    out = resample(p, model_grid, original_grid, order=1)
    if out.min() < -_ROUND_TOL or out.max() > 1 + _ROUND_TOL:
        raise OutputGridError("restored probability left [0, 1]")
    return np.clip(out, 0.0, 1.0).astype(np.float32)  # removes only float rounding (<1e-9)


def restore_mask(mask_model: np.ndarray, model_grid: Grid, original_grid: Grid) -> np.ndarray:
    try:
        return resample_mask(mask_model, model_grid, original_grid)
    except GeometryError as exc:
        raise OutputGridError(str(exc)) from exc


def build_outputs(
    prob_model: np.ndarray, case: PreparedCase, threshold: float
) -> OriginalGridOutputs:
    if not 0.0 < float(threshold) < 1.0:
        raise OutputGridError("threshold must be within (0, 1)")
    if tuple(np.asarray(prob_model).shape) != case.model_grid.shape:
        raise OutputGridError("probability is not on the model grid")
    grid = case.original_grid
    prob = restore_probability(prob_model, case.model_grid, grid)
    ungated = (prob >= np.float32(threshold)).astype(np.uint8)
    brain = case.original_brain.astype(np.uint8)
    gated = (ungated & brain).astype(np.uint8)
    errors = []
    for name, arr in (
        ("probability", prob),
        ("ungated", ungated),
        ("brain", brain),
        ("gated", gated),
    ):
        errors += [f"{name}:{e}" for e in check_on_grid(arr, grid, case.original_grid)]
    for name, arr in (("ungated", ungated), ("brain", brain), ("gated", gated)):
        if not set(np.unique(arr).tolist()) <= {0, 1}:
            errors.append(f"{name}:not_binary")
    if errors:
        raise OutputGridError(";".join(errors))
    qa = roundtrip_mask_qa(case.original_brain, restore_mask(case.brain, case.model_grid, grid))
    record = {
        "output_grid": "original",
        "probability_interpolation": PROBABILITY_INTERPOLATION,
        "mask_interpolation": MASK_INTERPOLATION,
        "threshold": float(threshold),
        "threshold_applied_on": "original_grid",
        "official_output": OFFICIAL_OUTPUT,
        "gated_role": "comparison_only",
        "ungated_voxels": int(ungated.sum()),
        "gated_voxels": int(gated.sum()),
        **boundary_record(ungated, brain, grid),
        "brain_mask_source": "original_grid_mask",
        "brain_mask_roles": BRAIN_MASK_ROLES,
        **qa,
    }
    return OriginalGridOutputs(prob, ungated, brain, gated, grid, record)
