"""Connected-component analysis of binary masks. Reports every component; removes nothing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import ndimage

from neuroscan_ai.preprocessing.mask import voxel_volume_mm3

# 26-connectivity (faces, edges and corners) is used everywhere in the project.
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n = ndimage.label(np.asarray(mask) > 0, structure=CONNECTIVITY_26)
    return labels, int(n)


def count_components(mask: np.ndarray) -> int:
    return label_components(mask)[1]


def components(
    mask: np.ndarray, spacing: Sequence[float], affine: np.ndarray | None = None
) -> list[dict[str, Any]]:
    """Per-component voxel count, volume, bounding box and centroid, largest first.

    bbox_min/bbox_max are inclusive voxel indices. centroid_voxel is the mean voxel index;
    centroid_world_mm is that point mapped through ``affine`` when given.
    """
    labels, n = label_components(mask)
    if n == 0:
        return []
    index = np.arange(1, n + 1)
    counts = ndimage.sum_labels(np.ones_like(labels), labels, index)
    centroids = ndimage.center_of_mass(np.ones_like(labels), labels, index)
    slices = ndimage.find_objects(labels)
    vox_mm3 = voxel_volume_mm3(spacing)
    out = []
    for i in range(n):
        centroid = [float(c) for c in centroids[i]]
        item = {
            "label": i + 1,
            "voxel_count": int(counts[i]),
            "volume_ml": float(counts[i]) * vox_mm3 / 1000.0,
            "bbox_min": [int(s.start) for s in slices[i]],
            "bbox_max": [int(s.stop) - 1 for s in slices[i]],
            "centroid_voxel": centroid,
        }
        if affine is not None:
            item["centroid_world_mm"] = (affine @ np.array([*centroid, 1.0]))[:3].tolist()
        out.append(item)
    out.sort(key=lambda c: (-c["voxel_count"], c["label"]))
    return out
