"""Dataset-agnostic NIfTI loading, geometry inspection and saving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

# Two images are on the "same grid" when shapes match and affines agree within this tolerance (mm).
AFFINE_ATOL_MM = 1e-3


def load(path: str | Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def spacing_from_affine(affine: np.ndarray) -> tuple[float, ...]:
    """Voxel size (mm) along each array axis, from the affine column norms."""
    return tuple(float(v) for v in np.linalg.norm(affine[:3, :3], axis=0))


def orientation(affine: np.ndarray) -> str:
    """Orientation code such as 'LAS' (direction each array axis increases toward)."""
    return "".join(nib.aff2axcodes(affine))


def geometry(img: nib.Nifti1Image) -> dict[str, Any]:
    """JSON-serialisable geometry/dtype summary of a NIfTI image (header only, no voxel data)."""
    slope, inter = img.header.get_slope_inter()
    return {
        "shape": [int(s) for s in img.shape[:3]],
        "spacing": [round(s, 6) for s in spacing_from_affine(img.affine)],
        "header_zooms": [round(float(z), 6) for z in img.header.get_zooms()[:3]],
        "affine": np.round(img.affine, 6).tolist(),
        "orientation": orientation(img.affine),
        "dtype": str(img.get_data_dtype()),
        "scl_slope": None if slope is None else float(slope),
        "scl_inter": None if inter is None else float(inter),
    }


def same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image, atol: float = AFFINE_ATOL_MM) -> bool:
    return a.shape[:3] == b.shape[:3] and bool(np.allclose(a.affine, b.affine, atol=atol))


def save(data: np.ndarray, affine: np.ndarray, path: str | Path) -> None:
    """Save array with the given affine; dtype is taken from the array as-is."""
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
