"""Build one verified research case bundle (Phase 7-B) in a staging directory.

Steps (any failure raises ``ResearchBuildError`` and nothing is registered):
 1. split check (done by the adapter before this module sees any path)
 2. frozen Phase 5A inference on the case (case id = alias; threshold, model, preprocessing,
    gating and post-processing exactly as configured — nothing is overridden here)
 3. reproducibility: voxel-wise comparison with the stored reference prediction (Phase 4)
 4. ground truth: finite, binary {0, 1}; prediction / GT on the same grid; valid affine
 5. metrics: ``evaluate_case`` (definitions unchanged; empty GT -> undefined overlap metrics)
 6. components (prediction and GT), TP/FP/FN difference labelmap, overlap-graph matching
 7. cross-checks (difference vs metrics, matching totals, prediction labelmap), file writes with
    reload verification, and a scan of every public JSON for the case key and absolute paths

Public bundle files carry the alias only; the case key goes back to the caller for the local
alias manifest and is never written into the bundle.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neuroscan_ai.analysis.components import check_labelmap, component_analysis
from neuroscan_ai.evaluation.segmentation import evaluate_case
from neuroscan_ai.io import nifti
from neuroscan_ai.preprocessing.mask import MASK_THRESHOLD
from neuroscan_ai.research.compare import LEGEND, check_difference, difference_labelmap
from neuroscan_ai.research.matching import check_matching, match_components
from neuroscan_ai.research.sources import ResearchSource

SCHEMA = "neuroscan.research_case/1"
AFFINE_ATOL = 1e-4
VOLUME_FILES = {  # bundle file -> (viewer id, role)
    "dwi.nii.gz": ("dwi", "image"),
    "adc.nii.gz": ("adc", "image"),
    "prediction_mask.nii.gz": ("prediction_mask", "label"),
    "prediction_components.nii.gz": ("prediction_components", "label"),
    "difference.nii.gz": ("difference", "label"),
}
JSON_FILES = ("metrics.json", "components.json", "matching.json", "research_manifest.json")


class ResearchBuildError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_affine(a: np.ndarray) -> bool:
    return a.shape == (4, 4) and bool(np.isfinite(a).all()) and abs(np.linalg.det(a[:3, :3])) > 1e-6


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float64)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ResearchBuildError("GT_INVALID", "Ground truth must be a finite 3-D image.")
    values = np.unique(data)
    if not np.all(np.isin(values, [0.0, 1.0])):
        raise ResearchBuildError(
            "GT_NOT_BINARY", "Ground truth must contain only 0 and 1.", {"values": values.size}
        )
    if not _valid_affine(img.affine):
        raise ResearchBuildError("INVALID_AFFINE", "Ground truth affine is not valid.")
    return (data > MASK_THRESHOLD).astype(np.uint8), img.affine


def compare_reference(mask: np.ndarray, affine: np.ndarray, reference: Path) -> dict[str, Any]:
    ref_img = nib.load(str(reference))
    ref = np.asanyarray(ref_img.dataobj)
    out: dict[str, Any] = {
        "reference": "stored validation prediction (Phase 4)",
        "shape_equal": ref.shape == mask.shape,
        "affine_equal": bool(np.allclose(ref_img.affine, affine, atol=AFFINE_ATOL)),
    }
    out["mismatched_voxels"] = int(((ref > 0) != (mask > 0)).sum()) if out["shape_equal"] else None
    out["foreground_voxels"] = int((mask > 0).sum())
    out["reference_foreground_voxels"] = int((ref > 0).sum())
    out["identical"] = bool(
        out["shape_equal"] and out["affine_equal"] and out["mismatched_voxels"] == 0
    )
    return out


def _write_labelmap(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    nifti.save(data, affine, path)
    back = nib.load(str(path))
    if not (
        np.array_equal(np.asanyarray(back.dataobj), data)
        and np.asanyarray(back.dataobj).dtype == data.dtype
        and np.allclose(back.affine, affine, atol=1e-5)
    ):
        raise ResearchBuildError("OUTPUT_VERIFY", "Saved research image failed verification.")


def _scan_public(obj: Any, forbidden: list[str]) -> list[str]:
    from neuroscan_ai.inference.schemas import is_absolute_path_text

    bad = []
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
            stack.extend(x.keys())
        elif isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, str):
            if is_absolute_path_text(x):
                bad.append("absolute path")
            if any(f and f in x for f in forbidden):
                bad.append("case key or source name")
    return sorted(set(bad))


def build_research_bundle(
    source: ResearchSource,
    staging: Path,
    *,
    device: str = "auto",
    require_reference: bool = True,
    inference_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and verify the bundle in ``staging`` (must not exist). Returns the manifest."""
    from neuroscan_ai.inference import run_independent_inference  # lazy: imports torch

    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=False)
    engine_root = staging / "_engine"
    r = run_independent_inference(
        source.base_image,
        source.secondary_image,
        engine_root,
        case_id=source.alias,
        device=device,
        **(inference_kwargs or {}),
    )
    if r.status != "succeeded" or not r.run_dir:
        raise ResearchBuildError(
            "INFERENCE_FAILED", "Frozen Phase 5A inference failed.", {"error": r.error}
        )
    run_dir = Path(r.run_dir)
    mask_img = nib.load(str(run_dir / "prediction_mask.nii.gz"))
    mask = np.asanyarray(mask_img.dataobj).astype(np.uint8)
    affine = mask_img.affine
    if not _valid_affine(affine):
        raise ResearchBuildError("INVALID_AFFINE", "Prediction affine is not valid.")
    quant = json.loads((run_dir / "quantification.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "inference_metadata.json").read_text(encoding="utf-8"))
    engine_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    # 3. reproducibility against the stored prediction
    repro: dict[str, Any] = {"reference": None, "identical": None}
    if source.reference_prediction is not None and source.reference_prediction.is_file():
        repro = compare_reference(mask, affine, source.reference_prediction)
        if not repro["identical"]:
            raise ResearchBuildError(
                "PREDICTION_MISMATCH",
                "Re-run prediction differs from the stored validation prediction.",
                repro,
            )
    elif require_reference:
        raise ResearchBuildError("REFERENCE_MISSING", "Stored reference prediction not found.")

    # 4. ground truth and grid
    gt, gt_affine = load_ground_truth(source.ground_truth)
    if gt.shape != mask.shape or not np.allclose(gt_affine, affine, atol=AFFINE_ATOL):
        raise ResearchBuildError(
            "GRID_MISMATCH",
            "Prediction and ground truth are not on the same grid.",
            {"prediction_shape": list(mask.shape), "gt_shape": list(gt.shape)},
        )

    # 5. metrics (definitions of Phase 2)
    spacing = nifti.spacing_from_affine(affine)
    metrics = evaluate_case(mask, gt, spacing, case_id=None)
    metrics.pop("case_id", None)

    # 6. components, difference, matching
    orient = nifti.orientation(affine)
    slice_axis = next(i for i, c in enumerate(orient) if c in "SI")
    pred_map, pred_comps = component_analysis(mask, spacing, affine, slice_axis)
    _, gt_comps = component_analysis(gt, spacing, affine, slice_axis)
    stored_map = np.asanyarray(nib.load(str(run_dir / "prediction_components.nii.gz")).dataobj)
    diff = difference_labelmap(mask, gt)
    matching = match_components(mask, gt, spacing, affine, slice_axis)

    # 7. cross-checks
    problems = check_difference(diff, mask, gt, metrics)
    problems += check_matching(matching, metrics)
    problems += check_labelmap(stored_map, mask, quant["components"])
    if not np.array_equal(stored_map, pred_map):
        problems.append("prediction labelmap differs from a fresh component analysis")
    if [c["size_rank"] for c in quant["components"]] != [c["size_rank"] for c in pred_comps]:
        problems.append("prediction component ranks differ")
    if problems:
        raise ResearchBuildError(
            "CONSISTENCY", "Research case failed consistency checks.", {"problems": problems}
        )

    for name in ("prediction_mask.nii.gz", "prediction_components.nii.gz"):
        shutil.copy2(run_dir / name, staging / name)
    for role in ("dwi", "adc"):
        shutil.copy2(run_dir / "inputs" / f"{role}.nii.gz", staging / f"{role}.nii.gz")
    _write_labelmap(staging / "difference.nii.gz", diff, affine)
    shutil.rmtree(engine_root)  # engine logs / metadata are not part of the public bundle

    gt_public = [
        {
            "size_rank": c["size_rank"],
            "voxel_count": c["voxel_count"],
            "volume_ml": c["volume_ml"],
            "centroid_world_mm": c["centroid_world_mm"],
            "representative_voxel": c["representative_voxel"],
            "representative_world_mm": c["representative_world_mm"],
            "slice_range": c["slice_range"],
        }
        for c in gt_comps
    ]
    model = meta["model"]
    files: dict[str, Any] = {}
    for name, (vid, role) in VOLUME_FILES.items():
        img = nib.load(str(staging / name))
        files[vid] = {
            "file": name,
            "role": role,
            "sha256": _sha(staging / name),
            "size_bytes": (staging / name).stat().st_size,
            "shape": [int(s) for s in img.shape[:3]],
            "spacing_mm": [round(float(s), 6) for s in nifti.spacing_from_affine(img.affine)],
            "affine": np.round(img.affine, 6).tolist(),
            "orientation": nifti.orientation(img.affine),
            "dtype": str(img.get_data_dtype()),
        }
    documents = {
        "metrics.json": {"schema_version": SCHEMA, "alias": source.alias, "metrics": metrics},
        "components.json": {
            "schema_version": SCHEMA,
            "alias": source.alias,
            "prediction_components": quant["components"],
            "ground_truth_components": gt_public,
        },
        "matching.json": {"schema_version": SCHEMA, "alias": source.alias, **matching},
    }
    manifest = {
        "schema_version": SCHEMA,
        "alias": source.alias,
        "dataset_id": source.dataset_id,
        "split": source.split,
        "has_ground_truth": True,
        "created_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "model_name": model.get("model_name"),
            "checkpoint_sha256": model.get("checkpoint_sha256"),
            "config_hash": model.get("config_hash"),
            "threshold": float(quant["prediction_threshold"]),
            "preprocessing_version": (meta.get("preprocessing") or {}).get("version"),
        },
        "reproducibility": repro,
        "views": engine_manifest["views"],
        "volumes": files,
        "documents": {},
        "legend": list(LEGEND),
    }
    forbidden = [source.source_case_key, source.base_image.name, source.ground_truth.name]
    for name, doc in documents.items():
        doc = _json_ready(doc)
        bad = _scan_public(doc, forbidden)
        if bad:
            raise ResearchBuildError(
                "PRIVACY", "Public research file failed privacy checks.", {"problems": bad}
            )
        (staging / name).write_text(json.dumps(doc, indent=1), encoding="utf-8")
        manifest["documents"][name] = _sha(staging / name)
    manifest = _json_ready(manifest)
    bad = _scan_public(manifest, forbidden)
    if bad:
        raise ResearchBuildError(
            "PRIVACY", "Research manifest failed privacy checks.", {"problems": bad}
        )
    (staging / "research_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    return manifest
