"""Component extraction on the original grid (26-connectivity, no resampling, nothing removed).

Component IDs are deterministic: volume descending, then the smallest k, j and i coordinate of
the component (each taken independently), then the lexicographically first voxel in (k, j, i)
order, which is unique because components are disjoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


class ComponentAnalysisError(ValueError):
    """Invalid input for the component safety analysis."""


def as_binary(mask: np.ndarray, name: str) -> np.ndarray:
    """Boolean view of a mask that must contain only 0 and 1 (refuses anything else)."""
    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ComponentAnalysisError(f"{name} must be 3D")
    if arr.dtype == bool:
        return arr
    if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
        raise ComponentAnalysisError(f"{name} contains non-finite values")
    values = np.unique(arr)
    if not set(values.tolist()) <= {0, 1}:
        raise ComponentAnalysisError(f"{name} must be binary (0/1)")
    return arr == 1


def spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    aff = np.asarray(affine, dtype=float)
    if aff.shape != (4, 4) or not np.isfinite(aff).all():
        raise ComponentAnalysisError("affine must be a finite 4x4 matrix")
    sp = np.linalg.norm(aff[:3, :3], axis=0)
    if (sp <= 0).any():
        raise ComponentAnalysisError("affine has a zero-length axis")
    return float(sp[0]), float(sp[1]), float(sp[2])


def extract_components(prediction: np.ndarray) -> tuple[np.ndarray, int]:
    """Label map (uint32, 0 = background, 1..n = component_id) with deterministic IDs."""
    pred = as_binary(prediction, "prediction")
    raw, n = ndimage.label(pred, structure=CONNECTIVITY_26)
    if n == 0:
        return np.zeros(pred.shape, dtype=np.uint32), 0
    keys = []
    for lab, sl in enumerate(ndimage.find_objects(raw), start=1):
        sub = raw[sl] == lab
        idx = np.argwhere(sub) + np.array([s.start for s in sl])  # (i, j, k)
        kji = idx[:, ::-1]
        first = tuple(kji[np.lexsort((kji[:, 2], kji[:, 1], kji[:, 0]))[0]].tolist())
        keys.append(
            (
                -len(idx),
                int(idx[:, 2].min()),
                int(idx[:, 1].min()),
                int(idx[:, 0].min()),
                first,
                lab,
            )
        )
    keys.sort()
    remap = np.zeros(n + 1, dtype=np.uint32)
    for new_id, key in enumerate(keys, start=1):
        remap[key[-1]] = new_id
    return remap[raw], n


def identity(
    labels: np.ndarray, component_id: int, affine: np.ndarray, spacing: tuple[float, float, float]
) -> dict[str, Any]:
    comp = labels == component_id
    idx = np.argwhere(comp)
    n = int(len(idx))
    centroid = idx.mean(axis=0)
    world = np.asarray(affine, dtype=float) @ np.append(centroid, 1.0)
    voxel_mm3 = float(np.prod(spacing))
    return {
        "component_id": int(component_id),
        "voxel_count": n,
        "voxel_volume_mm3": round(voxel_mm3, 6),
        "volume_ml": round(n * voxel_mm3 / 1000.0, 6),
        "bbox_ijk": [idx.min(axis=0).tolist(), idx.max(axis=0).tolist()],
        "slice_min": int(idx[:, 2].min()),
        "slice_max": int(idx[:, 2].max()),
        "n_slices": int(len(np.unique(idx[:, 2]))),
        "centroid_ijk": [round(float(c), 4) for c in centroid],
        "centroid_world_mm": [round(float(c), 4) for c in world[:3]],
    }


def reconstructs(labels: np.ndarray, prediction: np.ndarray) -> bool:
    """True when the union of all components equals the original prediction voxel for voxel."""
    return bool(np.array_equal(labels > 0, as_binary(prediction, "prediction")))
