"""Patient-level lesion quantification for a binary mask on a known grid."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from neuroscan_ai.analysis.components import components
from neuroscan_ai.preprocessing.mask import lesion_volume_ml


def lesion_summary(
    mask: np.ndarray,
    spacing: Sequence[float],
    affine: np.ndarray | None = None,
    slice_axis: int = 2,
    with_components: bool = True,
) -> dict[str, Any]:
    """Voxel count, volume (mL), lesion-positive slices and (optionally) all components.

    Volume uses the spacing of the grid the mask lives on; the official patient volume is
    computed on the original grid (see docs/phase2_evaluation_reconstruction.md).
    """
    binary = np.asarray(mask) > 0
    other_axes = tuple(a for a in range(binary.ndim) if a != slice_axis)
    positive_slices = np.nonzero(binary.any(axis=other_axes))[0]
    comps = components(binary, spacing, affine) if with_components else None
    summary: dict[str, Any] = {
        "voxel_count": int(binary.sum()),
        "volume_ml": lesion_volume_ml(binary, spacing),
        "positive_slice_count": int(positive_slices.size),
        "first_slice": int(positive_slices[0]) if positive_slices.size else None,
        "last_slice": int(positive_slices[-1]) if positive_slices.size else None,
    }
    if comps is not None:
        summary["component_count"] = len(comps)
        summary["components"] = comps
    return summary
