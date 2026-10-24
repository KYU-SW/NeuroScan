"""Brain mask policy and brain-mask gating (official inference step, Phase 3+).

Gating is an *anatomical ROI restriction*: probabilities outside the brain are set to 0 because
the input contains no image there (ISLES'22 is skull-stripped, and every train GT lesion voxel
lies inside the brain mask). It is not lesion-size post-processing: every prediction inside the
brain is kept, however small.

The brain mask rule is dataset-specific (``data.brain_mask_policy``). Only skull-stripped data
may use ``skull_stripped_background``; an external dataset that is not skull-stripped must
register its own policy (e.g. a brain-extraction step) and be validated before gating is used.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from neuroscan_ai.training.normalization import brain_mask

BRAIN_MASK_POLICIES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # Phase 3 rule: DWI differs from its constant background (handles sub-strokecase0046's
    # non-zero background) with the brain-fraction guard.
    "skull_stripped_background": brain_mask,
}


class BrainMaskError(ValueError):
    pass


def compute_brain_mask(policy: str, dwi: np.ndarray) -> np.ndarray:
    if policy not in BRAIN_MASK_POLICIES:
        raise BrainMaskError(f"unknown brain mask policy '{policy}': {sorted(BRAIN_MASK_POLICIES)}")
    try:
        mask = BRAIN_MASK_POLICIES[policy](dwi)
    except ValueError as exc:  # fraction guard of the policy
        raise BrainMaskError(str(exc)) from exc
    if not mask.any():
        raise BrainMaskError("brain mask is empty")
    return mask


def gate_probability(probability: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Probability with every voxel outside ``brain`` set to 0. Works for any array shape
    (single slice, batch of slices or volume), so 2D and 2.5D share it unchanged."""
    probability = np.asarray(probability)
    brain = np.asarray(brain, dtype=bool)
    if probability.shape != brain.shape:
        raise BrainMaskError(
            f"shape mismatch: probability {probability.shape} vs mask {brain.shape}"
        )
    if not np.isfinite(probability).all():
        raise BrainMaskError("probability contains NaN/inf")
    if probability.size and (probability.min() < 0 or probability.max() > 1):
        raise BrainMaskError("probability outside [0, 1]")
    return np.where(brain, probability, 0).astype(probability.dtype, copy=False)


def check_gt_inside_brain(label: np.ndarray, brain: np.ndarray, case_id: str) -> None:
    """Every train GT lesion voxel must lie inside the brain mask, else gating is invalid."""
    outside = int(np.count_nonzero((label > 0) & ~brain))
    if outside:
        raise BrainMaskError(f"{case_id}: {outside} GT lesion voxels outside the brain mask")
