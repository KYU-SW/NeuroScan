"""Brain-mask boundary metrics in mm. The brain mask is only measured against: the prediction is
never multiplied by, cut by or restricted to the brain mask, and outside voxels stay in the
component.

Distance of a voxel to the brain-mask boundary = physical distance (voxel centres, spacing from
the affine) to the nearest voxel of the opposite class (inside -> nearest outside voxel, outside ->
nearest inside voxel). A voxel next to the boundary therefore has a distance of one voxel step.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from neuroscan_ai.component_analysis.extraction import ComponentAnalysisError

CONTACT_MM = 3.0  # provisional (design candidate, not validated)


class BrainBoundary:
    def __init__(self, brain: np.ndarray, spacing: tuple[float, float, float]):
        self.brain = np.asarray(brain, dtype=bool)
        if not self.brain.any():
            raise ComponentAnalysisError("brain mask is empty")
        self.spacing = spacing
        self.in_plane_voxel_mm = float(min(spacing[0], spacing[1]))
        self.has_boundary = not self.brain.all()
        if self.has_boundary:
            d_in = ndimage.distance_transform_edt(self.brain, sampling=spacing)
            d_out = ndimage.distance_transform_edt(~self.brain, sampling=spacing)
            self.distance = np.where(self.brain, d_in, d_out)
        else:
            self.distance = None

    def metrics(self, comp: np.ndarray) -> dict[str, Any]:
        n = int(comp.sum())
        outside = int((comp & ~self.brain).sum())
        out: dict[str, Any] = {
            "voxels_outside_brain_mask": outside,
            "outside_brain_mask_ratio": round(outside / n, 6) if n else None,
            "touches_image_edge": touches_image_edge(comp),
            "in_plane_voxel_mm": round(self.in_plane_voxel_mm, 6),
            "contact_rule_mm": CONTACT_MM,
        }
        if self.distance is None:  # brain mask covers the whole array: no boundary inside it
            out.update(
                min_distance_to_brain_boundary_mm=None,
                median_distance_to_brain_boundary_mm=None,
                touches_boundary_1_inplane_voxel=False,
                touches_boundary_3mm=False,
                brain_mask_has_boundary=False,
            )
            return out
        d = self.distance[comp]
        dmin = float(d.min())
        out.update(
            min_distance_to_brain_boundary_mm=round(dmin, 6),
            median_distance_to_brain_boundary_mm=round(float(np.median(d)), 6),
            touches_boundary_1_inplane_voxel=bool(dmin <= self.in_plane_voxel_mm + 1e-6),
            touches_boundary_3mm=bool(dmin <= CONTACT_MM + 1e-6),
            brain_mask_has_boundary=True,
        )
        return out


def touches_image_edge(comp: np.ndarray) -> bool:
    return bool(
        comp[0].any()
        or comp[-1].any()
        or comp[:, 0].any()
        or comp[:, -1].any()
        or comp[:, :, 0].any()
        or comp[:, :, -1].any()
    )
