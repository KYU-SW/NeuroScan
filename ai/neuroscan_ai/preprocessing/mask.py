"""Binary lesion masks and lesion volume."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Some ISLES'22 masks store 1 as 0.99999999977 through NIfTI scaling, so never compare to == 1.
MASK_THRESHOLD = 0.5

# A {0,1} mask that was linearly interpolated is re-binarised with >= 0.5 (ties kept, which
# favours small lesions); see preprocessing and reconstruction.restore.
RESAMPLED_MASK_THRESHOLD = 0.5


def binarize(mask: np.ndarray, threshold: float = MASK_THRESHOLD) -> np.ndarray:
    """uint8 mask with values exactly {0, 1}: voxel = 1 where mask > threshold."""
    return (np.asarray(mask) > threshold).astype(np.uint8)


def voxel_volume_mm3(spacing: Sequence[float]) -> float:
    return float(np.prod(spacing[:3]))


def lesion_volume_ml(binary_mask: np.ndarray, spacing: Sequence[float]) -> float:
    """Lesion volume in mL (1 mL = 1000 mm^3)."""
    return int(np.count_nonzero(binary_mask)) * voxel_volume_mm3(spacing) / 1000.0
