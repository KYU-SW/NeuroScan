"""In-memory processed/v1 preprocessing + model input preparation for one unlabeled case.

Composes exactly the Phase 1 geometry steps (``preprocessing.pipeline.preprocess_case``) and the
Phase 3 input steps (``training.dataset.load_case``) without writing any file:

  original float32 arrays -> reorient (RAS, lossless) -> resample_to_spacing (XY 1 mm, Z kept)
  -> float32 -> brain mask (skull_stripped_background) -> per-case clip_zscore per modality

Equivalence with the on-disk path is enforced by tests (synthetic + real validation cases).

D1: ``predict_case`` needs a ``CaseVolumes`` with a label array. A zero-filled placeholder of the
processed shape is created in memory only; it is never saved, never evaluated and never reported.
The ``meta`` passed along carries geometry only — no lesion / ground-truth fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neuroscan_ai.io import nifti
from neuroscan_ai.preprocessing.geometry import reorient, resample_to_spacing
from neuroscan_ai.preprocessing.pipeline import PreprocessConfig
from neuroscan_ai.reconstruction.volume import VolumeGeometry
from neuroscan_ai.training.brain_gating import compute_brain_mask
from neuroscan_ai.training.dataset import CaseVolumes, norm_specs
from neuroscan_ai.training.normalization import normalize


@dataclass(frozen=True)
class PreparedCase:
    case: CaseVolumes  # model input (normalised image, brain mask, in-memory placeholder label)
    original: VolumeGeometry  # original DWI grid, exact affine (prediction is restored onto it)
    processed: VolumeGeometry  # processed grid as recorded by Phase 1 (affine rounded to 1e-6)
    processed_affine_exact: np.ndarray
    transforms: list[dict[str, Any]]


def load_preprocess_config(path: str | Path) -> PreprocessConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PreprocessConfig(
        version=raw["version"],
        target_spacing=tuple(raw["target_spacing"]),
        orientation=raw.get("orientation", "RAS"),
        image_interp_order=raw.get("image_interp_order", 1),
    )


def preprocess_arrays(
    arrays: dict[str, np.ndarray], affine: np.ndarray, cfg: PreprocessConfig
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    """Same steps, order and parameters as ``preprocess_case`` (images only, no label)."""
    out: dict[str, np.ndarray] = {}
    reoriented_affine = affine
    for k, a in arrays.items():
        out[k], reoriented_affine = reorient(a, affine, cfg.orientation)
    resampled_axes: list[bool] = [False, False, False]
    out_affine = reoriented_affine
    for k in out:
        out[k], out_affine, resampled_axes = resample_to_spacing(
            out[k],
            reoriented_affine,
            cfg.target_spacing,
            cfg.image_interp_order,
            cfg.spacing_tol_mm,
        )
    out = {k: a.astype(np.float32, copy=False) for k, a in out.items()}
    transforms = [
        {
            "name": "reorient",
            "from": nifti.orientation(affine),
            "to": cfg.orientation,
            "lossless": True,
        },
        {
            "name": "resample_to_spacing",
            "target_spacing_mm": list(cfg.target_spacing),
            "resampled_axes": resampled_axes,
            "applied": any(resampled_axes),
            "image_interpolation": "linear"
            if cfg.image_interp_order == 1
            else cfg.image_interp_order,
        },
        {"name": "cast", "images": "float32"},
    ]
    return out, out_affine, transforms


def prepare_case(
    case_id: str,
    dwi: np.ndarray,
    adc: np.ndarray,
    affine: np.ndarray,
    preprocess_cfg: PreprocessConfig,
    training_cfg: dict[str, Any],
) -> PreparedCase:
    data = training_cfg["data"]
    modalities = list(data["modalities"])
    if modalities != ["dwi", "adc"]:
        raise ValueError(f"unsupported modalities {modalities}")
    arrays, out_affine, transforms = preprocess_arrays(
        {"dwi": dwi, "adc": adc}, affine, preprocess_cfg
    )
    specs = norm_specs(data["normalization"])
    brain = compute_brain_mask(data["brain_mask_policy"], arrays["dwi"])
    image = np.stack([normalize(arrays[m], brain, specs[m]) for m in modalities])
    transforms.append(
        {
            "name": "normalize",
            "method": {m: data["normalization"][m]["method"] for m in modalities},
            "brain_mask_policy": data["brain_mask_policy"],
        }
    )
    shape = tuple(int(s) for s in arrays["dwi"].shape)
    # Phase 1 meta.json stores the processed affine rounded to 1e-6 and Phase 3/4 inference used
    # that recorded grid; reuse the same convention so restoration is bit-identical.
    processed_affine = np.round(out_affine, 6)
    meta = {
        "case_id": case_id,
        "original": {
            "shape": [int(s) for s in dwi.shape],
            "spacing": list(nifti.spacing_from_affine(affine)),
            "affine": np.asarray(affine, float).tolist(),
        },
        "processed": {"shape": list(shape), "affine": processed_affine.tolist()},
    }
    placeholder_label = np.zeros(shape, np.uint8)  # D1: memory only, never saved or evaluated
    case = CaseVolumes(case_id, image, placeholder_label, meta, brain)
    return PreparedCase(
        case=case,
        original=VolumeGeometry(tuple(int(s) for s in dwi.shape), np.asarray(affine, float)),
        processed=VolumeGeometry(shape, processed_affine),
        processed_affine_exact=out_affine,
        transforms=transforms,
    )
