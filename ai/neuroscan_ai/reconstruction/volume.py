"""Stack per-slice predictions back into a patient volume on a known grid.

The engine only places slices; it never thresholds. A probability map stays a probability map
until an evaluation/inference policy binarises it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from neuroscan_ai.io import nifti


class PredictionKind(StrEnum):
    BINARY = "binary"
    PROBABILITY = "probability"


@dataclass(frozen=True)
class VolumeGeometry:
    """Grid a volume lives on: array shape + voxel-to-world affine."""

    shape: tuple[int, int, int]
    affine: np.ndarray

    @classmethod
    def from_meta(cls, meta: dict[str, Any], space: str) -> VolumeGeometry:
        """``space`` is "processed" or "original" (keys of a Phase 1 meta.json)."""
        block = meta[space]
        return cls(tuple(int(s) for s in block["shape"]), np.asarray(block["affine"], float))

    @property
    def spacing(self) -> tuple[float, ...]:
        return nifti.spacing_from_affine(self.affine)

    @property
    def orientation(self) -> str:
        return nifti.orientation(self.affine)


@dataclass(frozen=True)
class PredictionVolume:
    data: np.ndarray
    geometry: VolumeGeometry
    kind: PredictionKind

    def save(self, path) -> None:
        nifti.save(self.data, self.geometry.affine, path)


class ReconstructionError(ValueError):
    pass


def _validate_values(arr: np.ndarray, kind: PredictionKind, z: int) -> None:
    if kind is PredictionKind.BINARY:
        if not set(np.unique(arr).tolist()) <= {0, 1}:
            raise ReconstructionError(f"slice {z}: binary slice has values other than 0/1")
    else:
        if not np.isfinite(arr).all():
            raise ReconstructionError(f"slice {z}: probability slice has NaN/inf")
        if arr.size and (arr.min() < 0.0 or arr.max() > 1.0):
            raise ReconstructionError(f"slice {z}: probability outside [0, 1]")


def reconstruct_volume(
    slices: Iterable[tuple[int, np.ndarray]],
    geometry: VolumeGeometry,
    kind: PredictionKind | str,
    slice_axis: int = 2,
) -> PredictionVolume:
    """Place ``(z_index, 2D prediction)`` pairs at their index along ``slice_axis``.

    Raises ReconstructionError for out-of-range, duplicate or missing indices, wrong slice
    shapes, or values invalid for ``kind``. Input order does not matter; index decides position.
    """
    kind = PredictionKind(kind)
    n_slices = geometry.shape[slice_axis]
    slice_shape = tuple(s for a, s in enumerate(geometry.shape) if a != slice_axis)
    dtype = np.uint8 if kind is PredictionKind.BINARY else np.float32
    out = np.zeros(geometry.shape, dtype=dtype)
    seen: set[int] = set()

    for z, arr in slices:
        if isinstance(z, bool) or not isinstance(z, int | np.integer):
            raise ReconstructionError(f"slice index must be an integer, got {z!r}")
        z = int(z)
        if not 0 <= z < n_slices:
            raise ReconstructionError(f"slice index {z} outside [0, {n_slices - 1}]")
        if z in seen:
            raise ReconstructionError(f"duplicate slice index {z}")
        arr = np.asarray(arr)
        if arr.shape != slice_shape:
            raise ReconstructionError(f"slice {z}: shape {arr.shape}, expected {slice_shape}")
        _validate_values(arr, kind, z)
        index = [slice(None)] * 3
        index[slice_axis] = z
        out[tuple(index)] = arr.astype(dtype, copy=False)
        seen.add(z)

    missing = sorted(set(range(n_slices)) - seen)
    if missing:
        raise ReconstructionError(f"missing slice indices: {missing}")
    return PredictionVolume(out, geometry, kind)


def decompose_volume(volume: np.ndarray, slice_axis: int = 2) -> list[tuple[int, np.ndarray]]:
    """Split a volume into ``(z_index, 2D slice)`` pairs (the inverse of reconstruction)."""
    return [
        (z, np.take(volume, z, axis=slice_axis).copy()) for z in range(volume.shape[slice_axis])
    ]
