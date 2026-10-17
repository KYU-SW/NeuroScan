"""Dataset-agnostic per-case preprocessing: CaseRecord -> processed NIfTI files + meta.json.

Geometry only (orientation, spacing, dtype). Intensity normalisation is deliberately left to
the model input pipeline so that one processed version can serve several experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import scipy

import neuroscan_ai
from neuroscan_ai.datasets.base import CaseRecord
from neuroscan_ai.io import nifti
from neuroscan_ai.preprocessing.geometry import SPACING_TOL_MM, reorient, resample_to_spacing
from neuroscan_ai.preprocessing.mask import MASK_THRESHOLD, binarize, lesion_volume_ml

# After linear interpolation of a {0,1} mask keep voxels >= 0.5 (ties kept: favours small lesions).
RESAMPLED_MASK_THRESHOLD = 0.5


@dataclass(frozen=True)
class PreprocessConfig:
    version: str
    target_spacing: tuple[float | None, float | None, float | None]
    orientation: str = "RAS"
    image_interp_order: int = 1
    spacing_tol_mm: float = SPACING_TOL_MM


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def preprocess_case(record: CaseRecord, cfg: PreprocessConfig, out_root: Path) -> dict[str, Any]:
    """Write ``out_root/<case_id>/{<modality>.nii.gz, mask.nii.gz, meta.json}``; return meta."""
    imgs = {k: nifti.load(p) for k, p in record.images.items()}
    ref_key = next(iter(imgs))
    ref = imgs[ref_key]
    label_img = nifti.load(record.label) if record.label is not None else None

    others = [*imgs.values(), *([label_img] if label_img is not None else [])]
    for img in others:
        if not nifti.same_grid(img, ref):
            raise ValueError(f"{record.case_id}: inputs are not on the same grid as {ref_key}")

    orig_geom = nifti.geometry(ref)
    orig_spacing = nifti.spacing_from_affine(ref.affine)
    transforms: list[dict[str, Any]] = []

    # 1) Binary mask in the original space (raw GT file itself is never modified).
    mask = None
    lesion: dict[str, Any] = {}
    if label_img is not None:
        mask = binarize(label_img.get_fdata(), MASK_THRESHOLD)
        lesion = {
            "original_voxel_count": int(mask.sum()),
            "original_volume_ml": round(lesion_volume_ml(mask, orig_spacing), 6),
            "has_lesion": bool(mask.any()),
        }
        transforms.append(
            {"name": "binarize_mask", "rule": f"mask > {MASK_THRESHOLD}", "space": "original"}
        )

    # 2) Lossless reorientation.
    arrays = {k: img.get_fdata(dtype=np.float32) for k, img in imgs.items()}
    for k in arrays:
        arrays[k], affine = reorient(arrays[k], ref.affine, cfg.orientation)
    if mask is not None:
        mask, _ = reorient(mask, ref.affine, cfg.orientation)
    transforms.append(
        {
            "name": "reorient",
            "from": orig_geom["orientation"],
            "to": cfg.orientation,
            "lossless": True,
        }
    )

    # 3) Spacing.
    resampled_axes: list[bool] = [False, False, False]
    out_affine = affine
    for k in arrays:
        arrays[k], out_affine, resampled_axes = resample_to_spacing(
            arrays[k], affine, cfg.target_spacing, cfg.image_interp_order, cfg.spacing_tol_mm
        )
    if mask is not None:
        m, _, _ = resample_to_spacing(mask, affine, cfg.target_spacing, 1, cfg.spacing_tol_mm)
        mask = (m >= RESAMPLED_MASK_THRESHOLD).astype(np.uint8) if any(resampled_axes) else m
    transforms.append(
        {
            "name": "resample_to_spacing",
            "target_spacing_mm": list(cfg.target_spacing),
            "resampled_axes": resampled_axes,
            "applied": any(resampled_axes),
            "image_interpolation": "linear"
            if cfg.image_interp_order == 1
            else cfg.image_interp_order,
            "mask_interpolation": f"linear, >= {RESAMPLED_MASK_THRESHOLD}",
            "grid_alignment": "first voxel corner aligned; kept axes copied voxel-for-voxel",
            "spacing_tolerance_mm": cfg.spacing_tol_mm,
        }
    )

    # 4) Dtypes.
    arrays = {k: a.astype(np.float32, copy=False) for k, a in arrays.items()}
    if mask is not None:
        mask = mask.astype(np.uint8, copy=False)
    transforms.append({"name": "cast", "images": "float32", "mask": "uint8"})

    case_dir = out_root / record.case_id
    files = {}
    for k, a in arrays.items():
        nifti.save(a, out_affine, case_dir / f"{k}.nii.gz")
        files[k] = f"{k}.nii.gz"
    if mask is not None:
        nifti.save(mask, out_affine, case_dir / "mask.nii.gz")
        files["mask"] = "mask.nii.gz"

    proc_spacing = nifti.spacing_from_affine(out_affine)
    if mask is not None:
        lesion["processed_voxel_count"] = int(mask.sum())
        lesion["processed_volume_ml"] = round(lesion_volume_ml(mask, proc_spacing), 6)
        lesion["lost_in_processing"] = bool(lesion["has_lesion"] and not mask.any())

    meta = {
        "case_id": record.case_id,
        "dataset": record.dataset,
        "preprocessing_version": cfg.version,
        "source": {k: _rel(p) for k, p in record.all_paths().items() if k != "flair"},
        "original": {
            "shape": orig_geom["shape"],
            "spacing": orig_geom["spacing"],
            "affine": orig_geom["affine"],
            "orientation": orig_geom["orientation"],
            "dtype": {
                **{k: str(img.get_data_dtype()) for k, img in imgs.items()},
                **({"mask": str(label_img.get_data_dtype())} if label_img is not None else {}),
            },
        },
        "lesion": lesion,
        "processed": {
            "shape": [int(s) for s in next(iter(arrays.values())).shape],
            "spacing": [round(s, 6) for s in proc_spacing],
            "affine": np.round(out_affine, 6).tolist(),
            "orientation": nifti.orientation(out_affine),
            "dtype": {
                **{k: "float32" for k in arrays},
                **({"mask": "uint8"} if mask is not None else {}),
            },
            "files": files,
        },
        "transforms": transforms,
        "intensity_normalization": "none (applied later in the model input pipeline)",
        "software": {
            "neuroscan_ai": neuroscan_ai.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nibabel": nib.__version__,
        },
    }
    (case_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
