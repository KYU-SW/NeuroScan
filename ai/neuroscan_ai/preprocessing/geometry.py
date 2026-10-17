"""Orientation and spacing transforms that keep an exact, invertible voxel-to-world mapping."""

from __future__ import annotations

from collections.abc import Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage

from neuroscan_ai.io.nifti import spacing_from_affine

# Spacing differences at or below this (mm) are treated as "already at target": no interpolation.
SPACING_TOL_MM = 1e-3


def reorient(
    data: np.ndarray, affine: np.ndarray, target: str = "RAS"
) -> tuple[np.ndarray, np.ndarray]:
    """Lossless axis flip/permutation so array axes follow ``target`` (e.g. LAS -> RAS)."""
    current = nib.orientations.io_orientation(affine)
    wanted = nib.orientations.axcodes2ornt(tuple(target))
    transform = nib.orientations.ornt_transform(current, wanted)
    out = nib.orientations.apply_orientation(data, transform)
    out_affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape[:3])
    return np.ascontiguousarray(out), out_affine


def resample_to_spacing(
    data: np.ndarray,
    affine: np.ndarray,
    target_spacing: Sequence[float | None],
    order: int,
    tol: float = SPACING_TOL_MM,
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """Resample along the array axes to ``target_spacing`` (mm); ``None`` keeps that axis.

    Voxel corners of the first voxel stay aligned with the input, so the field of view starts at
    the same physical position. Axes that are kept (``None`` or already within ``tol``) are copied
    voxel-for-voxel, which guarantees e.g. that slice index z maps to the original slice z.

    Returns (data, affine, resampled_axes).
    """
    spacing = spacing_from_affine(affine)
    shape = data.shape[:3]
    ratios, out_shape, resampled = [], [], []
    for n, s, t in zip(shape, spacing, target_spacing, strict=True):
        if t is None or abs(t - s) <= tol:
            ratios.append(1.0)
            out_shape.append(n)
            resampled.append(False)
        else:
            ratios.append(t / s)
            out_shape.append(max(1, int(round(n * s / t))))
            resampled.append(True)

    # Output voxel i samples input voxel coordinate r*i + (r-1)/2 (corner-aligned grids).
    ratios_arr = np.array(ratios)
    offset = (ratios_arr - 1.0) / 2.0
    index_map = np.eye(4)
    index_map[:3, :3] = np.diag(ratios_arr)
    index_map[:3, 3] = offset
    out_affine = affine @ index_map

    if not any(resampled):
        return data.copy(), out_affine, resampled
    out = ndimage.affine_transform(
        data.astype(np.float32, copy=False),
        ratios_arr,
        offset=offset,
        output_shape=tuple(out_shape),
        order=order,
        mode="nearest",
    )
    return out, out_affine, resampled


def resample_to_grid(
    data: np.ndarray,
    src_affine: np.ndarray,
    dst_affine: np.ndarray,
    dst_shape: Sequence[int],
    order: int,
) -> np.ndarray:
    """Resample ``data`` (on ``src_affine``) onto an arbitrary target grid via world coordinates.

    Used to bring processed-space arrays back to the original image grid. When both grids are
    identical this is an exact copy for any ``order``.
    """
    if tuple(dst_shape) == data.shape[:3] and np.allclose(src_affine, dst_affine, atol=1e-6):
        return data.copy()
    dst_to_src = np.linalg.inv(src_affine) @ dst_affine
    return ndimage.affine_transform(
        data.astype(np.float32, copy=False),
        dst_to_src[:3, :3],
        offset=dst_to_src[:3, 3],
        output_shape=tuple(dst_shape),
        order=order,
        mode="nearest",
    )
