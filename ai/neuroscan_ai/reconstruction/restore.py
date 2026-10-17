"""Bring a processed-space prediction back onto the original patient grid.

Reuses ``preprocessing.geometry.resample_to_grid`` (world-coordinate mapping), so the LAS<->RAS
flip, the XY spacing change and any oblique rotation are all undone by the affines alone.

Interpolation policy
- binary:      linear interpolation of the {0,1} mask, then >= 0.5. Output is exactly {0,1}.
               Equivalent to a local partial-volume majority vote; same rule as preprocessing,
               so restoration is the exact inverse path measured in Phase 1.
- probability: linear interpolation of the continuous map; values stay in [0, 1] (a convex
               combination of inputs). No threshold is applied here.
"""

from __future__ import annotations

import numpy as np

from neuroscan_ai.preprocessing.geometry import resample_to_grid
from neuroscan_ai.preprocessing.mask import RESAMPLED_MASK_THRESHOLD
from neuroscan_ai.reconstruction.volume import PredictionKind, PredictionVolume, VolumeGeometry

LINEAR = 1


def restore_to_original(volume: PredictionVolume, original: VolumeGeometry) -> PredictionVolume:
    resampled = resample_to_grid(
        volume.data, volume.geometry.affine, original.affine, original.shape, order=LINEAR
    )
    if volume.kind is PredictionKind.BINARY:
        data = (resampled >= RESAMPLED_MASK_THRESHOLD).astype(np.uint8)
    else:
        # linear interpolation cannot leave [0, 1]; clip only float32 rounding noise
        data = np.clip(resampled.astype(np.float32, copy=False), 0.0, 1.0)
    return PredictionVolume(data, original, volume.kind)
