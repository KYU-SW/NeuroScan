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
    mask: np.ndarray,
    spacing: Sequence[float],
    affine: np.ndarray | None = None,
    labelled: tuple[np.ndarray, int] | None = None,
) -> list[dict[str, Any]]:
    """Per-component voxel count, volume, bounding box and centroid, largest first.

    bbox_min/bbox_max are inclusive voxel indices. centroid_voxel is the mean voxel index;
    centroid_world_mm is that point mapped through ``affine`` when given. ``labelled`` reuses a
    ``label_components(mask)`` result (same mask) instead of labelling again.
    """
    labels, n = labelled if labelled is not None else label_components(mask)
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


# --- Phase 7-A: component labelmap for review (derived display product) ----------------------

MAX_DISPLAY_LABEL = 254  # labels 1..254 = components by size rank
OTHER_DISPLAY_LABEL = 255  # all components ranked after 254 share this "other" label


def _representative_voxel(
    labels: np.ndarray, label: int, region: tuple[slice, ...], centroid, spacing
) -> list[int]:
    """Voxel of the component closest (in mm) to its centroid — always inside the component,
    also when the centroid itself is not (rings, curved or split shapes). Ties: first voxel in
    C order."""
    idx = np.argwhere(labels[region] == label) + np.array([s.start for s in region])
    d2 = (((idx - np.asarray(centroid)) * np.asarray(spacing[:3], dtype=float)) ** 2).sum(axis=1)
    return [int(v) for v in idx[int(np.argmin(d2))]]


def component_analysis(
    mask: np.ndarray,
    spacing: Sequence[float],
    affine: np.ndarray | None,
    slice_axis: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """One labelling of ``mask`` -> (uint8 display labelmap, component list largest first).

    The list is ``components()`` plus, per component: ``size_rank`` (1 = largest),
    ``display_label`` (= size_rank up to 254, then 255 "other"), ``representative_voxel``
    (+ ``representative_world_mm`` when ``affine`` is given) and ``slice_range`` along
    ``slice_axis`` (inclusive voxel indices). The labelmap has the mask's shape, 0 = background,
    and the same foreground as the mask. Nothing is removed or merged in the list; ``mask`` is
    not modified.
    """
    labelled = label_components(mask)
    labels, n = labelled
    comps = components(mask, spacing, affine, labelled=labelled)
    lut = np.zeros(n + 1, dtype=np.uint8)
    regions = ndimage.find_objects(labels)
    for rank, c in enumerate(comps, start=1):
        display = rank if rank <= MAX_DISPLAY_LABEL else OTHER_DISPLAY_LABEL
        lut[c["label"]] = display
        region = regions[c["label"] - 1]
        rep = _representative_voxel(labels, c["label"], region, c["centroid_voxel"], spacing)
        c["size_rank"] = rank
        c["display_label"] = display
        c["representative_voxel"] = rep
        if affine is not None:
            c["representative_world_mm"] = (affine @ np.array([*rep, 1.0]))[:3].tolist()
        c["slice_range"] = {
            "axis": int(slice_axis),
            "first": int(c["bbox_min"][slice_axis]),
            "last": int(c["bbox_max"][slice_axis]),
        }
    return lut[labels], comps


def check_labelmap(
    labelmap: np.ndarray, mask: np.ndarray, comps: list[dict[str, Any]]
) -> list[str]:
    """Consistency of a display labelmap with its mask and component list (empty = OK)."""
    out = []
    if labelmap.shape != mask.shape or labelmap.dtype != np.uint8:
        return ["labelmap shape/dtype"]
    if not np.array_equal(labelmap > 0, np.asarray(mask) > 0):
        out.append("labelmap foreground differs from mask")
    expected: dict[int, int] = {}
    for c in comps:
        expected[c["display_label"]] = expected.get(c["display_label"], 0) + c["voxel_count"]
    counts = np.bincount(labelmap.ravel(), minlength=256)
    for label in range(1, 256):
        if int(counts[label]) != expected.get(label, 0):
            out.append(f"label {label} voxel count")
            break
    for c in comps:
        if labelmap[tuple(c["representative_voxel"])] != c["display_label"]:
            out.append("representative voxel outside its component")
            break
    return out
