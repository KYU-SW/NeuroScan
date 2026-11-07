"""1 mm in-plane model grid and its inverse mapping (``preprocess_v2_isles_1mm_inplane_v1``).

Array axes 0 and 1 are in-plane; axis 2 is the slice axis (kept unchanged).

Target grid (one grid per case, shared by DWI, ADC, GT and brain mask):
- direction cosines of every axis are kept (oblique affines stay oblique; no reorientation);
- in-plane spacing becomes ``target_mm`` (1.0); new in-plane size = ceil(size * spacing / target);
- slice axis: spacing, size and position unchanged;
- corner alignment: the outer edge of voxel 0 coincides with the original grid, so the model grid
  covers the original field of view (it may extend by < 1 target voxel at the far edge).

Resampling maps every destination voxel centre through world coordinates into the source grid
(``inv(src_affine) @ dst_affine``). Order 1 (linear) for images and probabilities, order 0
(nearest) for labels and masks. Samples outside the source are taken from the nearest edge voxel.
When the target grid equals the source grid (already 1 mm in-plane) the data are copied exactly.
Inputs are never modified in place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from nibabel.orientations import aff2axcodes
from scipy import ndimage

TARGET_INPLANE_MM = 1.0
AFFINE_ATOL_MM = 1e-3
IDENTITY_ATOL = 1e-6


class GeometryError(ValueError):
    """Invalid or inconsistent grid geometry."""


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, int, int]
    affine: np.ndarray  # 4x4 voxel -> world (mm)

    @property
    def spacing(self) -> tuple[float, float, float]:
        sp = np.linalg.norm(self.affine[:3, :3], axis=0)
        return float(sp[0]), float(sp[1]), float(sp[2])

    @property
    def axcodes(self) -> tuple[str, str, str]:
        return tuple(aff2axcodes(self.affine))

    def record(self) -> dict:
        return {
            "shape": list(self.shape),
            "affine": [[round(float(v), 8) for v in row] for row in self.affine],
            "spacing_mm": [round(v, 8) for v in self.spacing],
            "axcodes": list(self.axcodes),
        }


def validate_grid(shape: tuple[int, ...], affine: np.ndarray) -> Grid:
    aff = np.asarray(affine, dtype=np.float64)
    if len(shape) != 3 or any(int(s) < 1 for s in shape):
        raise GeometryError("grid must be 3D with positive sizes")
    if aff.shape != (4, 4) or not np.isfinite(aff).all():
        raise GeometryError("affine must be a finite 4x4 matrix")
    if not np.allclose(aff[3], [0, 0, 0, 1], atol=1e-9):
        raise GeometryError("affine last row must be [0, 0, 0, 1]")
    if abs(np.linalg.det(aff[:3, :3])) < 1e-9:
        raise GeometryError("affine is singular")
    if (np.linalg.norm(aff[:3, :3], axis=0) <= 1e-6).any():
        raise GeometryError("affine has a zero-length axis")
    return Grid(tuple(int(s) for s in shape), aff.copy())


def same_grid(a: Grid, b: Grid, atol: float = AFFINE_ATOL_MM) -> bool:
    return a.shape == b.shape and bool(np.allclose(a.affine, b.affine, atol=atol))


def inplane_target_grid(grid: Grid, target_mm: float = TARGET_INPLANE_MM) -> Grid:
    if target_mm <= 0:
        raise GeometryError("target spacing must be positive")
    m = grid.affine[:3, :3]
    sp = np.asarray(grid.spacing)
    new_m = m.copy()
    shape = list(grid.shape)
    origin = grid.affine[:3, 3].copy()
    for ax in (0, 1):
        direction = m[:, ax] / sp[ax]
        new_m[:, ax] = direction * target_mm
        # ceil with a small tolerance so 0.5 mm x 2N stays exactly N voxels
        shape[ax] = max(1, math.ceil(grid.shape[ax] * sp[ax] / target_mm - 1e-6))
        # voxel-0 outer edge stays fixed: centre moves by half the spacing difference
        origin = origin + 0.5 * (target_mm - sp[ax]) * direction
    aff = np.eye(4)
    aff[:3, :3] = new_m
    aff[:3, 3] = origin
    out = Grid(tuple(shape), aff)
    if same_grid(out, grid, atol=IDENTITY_ATOL):
        return Grid(grid.shape, grid.affine.copy())
    return out


def resample(data: np.ndarray, src: Grid, dst: Grid, order: int) -> np.ndarray:
    """Resample a 3D array from ``src`` to ``dst`` (order 1 linear, 0 nearest); new array."""
    if order not in (0, 1):
        raise GeometryError("only order 0 (nearest) or 1 (linear) is allowed")
    arr = np.asarray(data)
    if tuple(arr.shape) != src.shape:
        raise GeometryError("data shape does not match its grid")
    if same_grid(src, dst, atol=IDENTITY_ATOL):
        return np.array(arr, dtype=np.float64 if order == 1 else arr.dtype, copy=True)
    mapping = np.linalg.inv(src.affine) @ dst.affine
    out = ndimage.affine_transform(
        arr.astype(np.float64),
        mapping[:3, :3],
        offset=mapping[:3, 3],
        output_shape=dst.shape,
        order=order,
        mode="nearest",
        prefilter=False,
    )
    return out if order == 1 else out.astype(arr.dtype)


def as_binary_mask(mask: np.ndarray, name: str) -> np.ndarray:
    """Boolean mask from an array that must contain only 0 and 1 (other labels are refused)."""
    arr = np.asarray(mask)
    if arr.dtype == bool:
        return arr.copy()
    if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
        raise GeometryError(f"{name} contains non-finite values")
    if not set(np.unique(arr).tolist()) <= {0, 1}:
        raise GeometryError(f"{name} must contain only labels 0 and 1")
    return arr == 1


def resample_mask(mask: np.ndarray, src: Grid, dst: Grid) -> np.ndarray:
    """Nearest-neighbour resampling of a binary mask; the result is exactly 0/1 (bool)."""
    m = as_binary_mask(mask, "mask").astype(np.uint8)
    out = resample(m, src, dst, order=0)
    if not set(np.unique(out).tolist()) <= {0, 1}:
        raise GeometryError("resampled mask is not binary")
    return out == 1


def check_on_grid(data: np.ndarray, grid: Grid, expected: Grid) -> list[str]:
    """Error codes when an output is not on the expected grid (empty list = consistent)."""
    errors = []
    if tuple(np.asarray(data).shape) != expected.shape or grid.shape != expected.shape:
        errors.append("shape_mismatch")
    if not np.allclose(grid.affine, expected.affine, atol=AFFINE_ATOL_MM):
        errors.append("affine_mismatch")
    if grid.axcodes != expected.axcodes:
        errors.append("orientation_mismatch")
    if not np.allclose(grid.spacing, expected.spacing, atol=AFFINE_ATOL_MM):
        errors.append("spacing_mismatch")
    return errors


def pad_to_multiple(x: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Zero-pad the last two axes at the end up to a multiple; returns the padded copy and size."""
    if multiple < 1:
        raise GeometryError("multiple must be >= 1")
    h, w = x.shape[-2:]
    pad = [(0, 0)] * (x.ndim - 2) + [(0, (-h) % multiple), (0, (-w) % multiple)]
    return np.pad(x, pad), (int(h), int(w))


def crop_to(x: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    if x.shape[-2] < h or x.shape[-1] < w:
        raise GeometryError("cannot crop to a larger size")
    return x[..., :h, :w].copy()
