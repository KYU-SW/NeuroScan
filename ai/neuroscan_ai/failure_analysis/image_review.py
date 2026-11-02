"""Image-based review objects for selected failure cases (pure functions on arrays).

Everything here works on arrays of one original patient grid (GT grid; the stored predictions
share it). Distances are in mm via the voxel spacing (Euclidean distance transform). Nothing is
removed, thresholded or re-evaluated: the review thresholds below only describe what is seen and
are never used for metrics, post-processing, component removal, model selection or diagnosis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import ndimage

from neuroscan_ai.analysis.components import component_analysis, label_components
from neuroscan_ai.failure_analysis import schemas as sc
from neuroscan_ai.research.compare import FN, FP, TP

SLICE_AXIS = 2  # acquisition slice axis of the original grid (k)
BOUNDARY_MM = 2.0
BOUNDARY_MAJORITY = 0.5
MAX_CENTER_SLICES = 5

# model difference labels
MD_BOTH, MD_2D_ONLY, MD_25D_ONLY = 1, 2, 3


def model_difference_map(pred_2d: np.ndarray, pred_25d: np.ndarray) -> np.ndarray:
    a, b = np.asarray(pred_2d) > 0, np.asarray(pred_25d) > 0
    if a.shape != b.shape:
        raise ValueError("prediction grids differ")
    out = np.zeros(a.shape, np.uint8)
    out[a & b] = MD_BOTH
    out[a & ~b] = MD_2D_ONLY
    out[~a & b] = MD_25D_ONLY
    return out


def distance_to(mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray | None:
    """Per-voxel distance (mm) to the nearest voxel of ``mask``; None when ``mask`` is empty."""
    m = np.asarray(mask) > 0
    if not m.any():
        return None
    return ndimage.distance_transform_edt(~m, sampling=spacing[:3])


def depth_inside(mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    """Per-voxel distance (mm) from a voxel inside ``mask`` to the nearest voxel outside it."""
    m = np.asarray(mask) > 0
    padded = np.pad(m, 1, constant_values=False)  # the grid edge counts as outside
    return ndimage.distance_transform_edt(padded, sampling=spacing[:3])[1:-1, 1:-1, 1:-1]


def component_table(
    mask: np.ndarray,
    spacing: Sequence[float],
    affine: np.ndarray,
    other: np.ndarray | None,
    brain: np.ndarray,
) -> list[dict[str, Any]]:
    """26-connectivity components of ``mask`` (largest first, nothing dropped) with distances:
    ``nearest_other_mm`` = shortest distance from the component to ``other`` (0 when touching;
    None when ``other`` is empty) and ``brain_boundary_mm`` = shortest distance from the
    component to the outside of the brain mask."""
    _, comps = component_analysis(mask, spacing, affine, SLICE_AXIS)
    if not comps:
        return []
    labels, _ = label_components(mask)
    d_other = distance_to(other, spacing) if other is not None else None
    d_edge = depth_inside(brain, spacing)
    rows = []
    for c in comps:
        sel = labels == c["label"]
        rows.append(
            {
                "component_id": c["size_rank"],
                "voxel_count": c["voxel_count"],
                "volume_ml": c["volume_ml"],
                "slice_first": c["slice_range"]["first"],
                "slice_last": c["slice_range"]["last"],
                "representative_voxel": c["representative_voxel"],
                "representative_world_mm": [round(v, 2) for v in c["representative_world_mm"]],
                "bbox_min": c["bbox_min"],
                "bbox_max": c["bbox_max"],
                "nearest_other_mm": None if d_other is None else float(d_other[sel].min()),
                "brain_boundary_mm": float(d_edge[sel].min()),
            }
        )
    return rows


def boundary_share(
    pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], tol_mm: float = BOUNDARY_MM
) -> dict[str, Any]:
    """Share of FP + FN voxels lying within ``tol_mm`` of the GT boundary."""
    p, g = np.asarray(pred) > 0, np.asarray(gt) > 0
    fp, fn = p & ~g, ~p & g
    n = int(fp.sum() + fn.sum())
    if not g.any() or n == 0:
        return {"error_voxels": n, "within_tol": None, "share": None, "tol_mm": tol_mm}
    d_out = distance_to(g, spacing)  # FP voxels: distance to the GT
    d_in = depth_inside(g, spacing)  # FN voxels: distance to the GT edge
    near = int((d_out[fp] <= tol_mm).sum() + (d_in[fn] <= tol_mm).sum())
    return {"error_voxels": n, "within_tol": near, "share": near / n, "tol_mm": tol_mm}


def region_stats(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"voxel_count": 0, "median": None, "q1": None, "q3": None}
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"voxel_count": int(v.size), "median": float(med), "q1": float(q1), "q3": float(q3)}


def intensity_summary(
    norm: dict[str, np.ndarray], diff: np.ndarray, brain: np.ndarray, exclude: np.ndarray
) -> dict[str, Any]:
    """DWI / ADC (case-wise brain z-score) in TP, FP, FN and a brain reference (brain voxels
    outside the GT and outside both predictions). Empty regions stay undefined (None)."""
    regions = {
        "tp": diff == TP,
        "fp": diff == FP,
        "fn": diff == FN,
        "brain_reference": (np.asarray(brain) > 0) & ~exclude,
    }
    return {mod: {r: region_stats(arr[m]) for r, m in regions.items()} for mod, arr in norm.items()}


# --- slices -------------------------------------------------------------------------------------


def _argmax_slice(mask: np.ndarray) -> int | None:
    a = np.moveaxis(np.asarray(mask) > 0, SLICE_AXIS, -1).reshape(-1, mask.shape[SLICE_AXIS])
    s = a.sum(axis=0)
    return int(np.argmax(s)) if s.max() > 0 else None  # first slice on ties


def select_slices(
    gt: np.ndarray,
    diff_2d: np.ndarray,
    diff_25d: np.ndarray,
    mdiff: np.ndarray,
    representative: Sequence[int],
    max_n: int = MAX_CENTER_SLICES,
) -> list[dict[str, Any]]:
    """Centre slices in priority order, duplicates merged (reasons kept), at most ``max_n``."""
    cands = [
        ("gt_area_max", _argmax_slice(gt)),
        ("fn_area_max", _argmax_slice((diff_2d == FN) | (diff_25d == FN))),
        ("fp_2d_area_max", _argmax_slice(diff_2d == FP)),
        ("fp_25d_area_max", _argmax_slice(diff_25d == FP)),
        ("model_difference_max", _argmax_slice((mdiff == MD_2D_ONLY) | (mdiff == MD_25D_ONLY))),
        *[(f"component_representative_{i}", int(z)) for i, z in enumerate(representative, 1)],
    ]
    out: list[dict[str, Any]] = []
    for reason, z in cands:
        if z is None:
            continue
        hit = next((s for s in out if s["z"] == z), None)
        if hit:
            hit["reasons"].append(reason)
        elif len(out) < max_n:
            out.append({"z": z, "reasons": [reason]})
    return out


def neighbours(z: int, n_slices: int) -> list[int | None]:
    """[z-1, z, z+1]; positions outside the volume are None (shown as 'out of range')."""
    return [k if 0 <= k < n_slices else None for k in (z - 1, z, z + 1)]


# --- taxonomy -----------------------------------------------------------------------------------


def model_flags(
    metrics: dict[str, Any],
    matching: dict[str, Any] | None,
    pred_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    boundary: dict[str, Any],
    has_lesion: bool,
) -> dict[str, Any]:
    """Per-model T1-T7 from measured quantities (no cause is assigned)."""
    if not has_lesion:
        fp_far = [r for r in pred_rows if r["nearest_other_mm"] is None]
        return {
            "T1": sc.NOT_APPLICABLE,
            "T2": sc.NOT_APPLICABLE,
            "T3": sc.NOT_APPLICABLE,
            "T4": sc.NOT_APPLICABLE,
            "T5": sc.NOT_APPLICABLE,
            "T6": sc.NOT_APPLICABLE,
            "T7": sc.NOT_APPLICABLE,
            "negative_prediction_components": len(fp_far),
        }
    sens, prec = _f(metrics.get("sensitivity")), _f(metrics.get("precision"))
    tp, fp, fn = (int(metrics[k]) for k in ("tp_voxels", "fp_voxels", "fn_voxels"))
    pred, gt = int(metrics["prediction_voxels"]), int(metrics["gt_voxels"])
    overlap_ratio = tp / gt if gt else None
    t3 = pred > 0 and (
        tp == 0 or (overlap_ratio is not None and overlap_ratio < sc.LOCATION_MISMATCH_OVERLAP)
    )
    groups = (matching or {}).get("groups", [])
    gt_types = {r["component_id"]: None for r in gt_rows}
    for g in (matching or {}).get("ground_truth_components", []):
        gt_types[g["size_rank"]] = g["group_type"]
    small_missed = [
        r
        for r in gt_rows
        if r["volume_ml"] < sc.SMALL_GT_COMPONENT_ML
        and gt_types.get(r["component_id"]) == "ground_truth_only"
    ]
    p_types = {
        p["size_rank"]: p["group_type"] for p in (matching or {}).get("prediction_components", [])
    }
    far_fp = [
        r
        for r in pred_rows
        if p_types.get(r["component_id"]) == "prediction_only"
        and r["nearest_other_mm"] is not None
        and r["nearest_other_mm"] > sc.DISTANT_FP_MM
    ]
    sm = [g for g in groups if g["group_type"] in ("split", "merge", "complex")]
    t6 = tp > 0 and boundary["share"] is not None and boundary["share"] > BOUNDARY_MAJORITY
    return {
        "T1": sc.FLAGGED if (sens < 0.5 and fn > fp) else sc.NOT_FLAGGED,
        "T2": sc.FLAGGED if (prec < 0.5 and fp > fn) else sc.NOT_FLAGGED,
        "T3": sc.FLAGGED if t3 else sc.NOT_FLAGGED,
        "T4": sc.FLAGGED if small_missed else sc.NOT_FLAGGED,
        "T5": sc.FLAGGED if far_fp else sc.NOT_FLAGGED,
        "T6": sc.REVIEW_CANDIDATE if t6 else sc.NOT_FLAGGED,
        "T7": sc.FLAGGED if sm else sc.NOT_FLAGGED,
        "gt_overlap_ratio": overlap_ratio,
        "small_gt_components_missed": len(small_missed),
        "distant_fp_components": len(far_fp),
        "split_merge_complex_groups": {
            t: sum(g["group_type"] == t for g in sm) for t in ("split", "merge", "complex")
        },
        "boundary_error_share_within_2mm": boundary["share"],
    }


def t9_candidate(intensity: dict[str, Any]) -> dict[str, Any]:
    """Signal pattern worth a closer look (not a statement about the GT): FP region with DWI
    median above and ADC median below the brain reference IQR, or FN region whose DWI median
    lies inside the brain reference IQR."""
    dwi, adc = intensity["dwi"], intensity["adc"]
    ref_d, ref_a = dwi["brain_reference"], adc["brain_reference"]
    reasons = []
    if (
        dwi["fp"]["median"] is not None
        and ref_d["q3"] is not None
        and adc["fp"]["median"] is not None
        and ref_a["q1"] is not None
        and dwi["fp"]["median"] > ref_d["q3"]
        and adc["fp"]["median"] < ref_a["q1"]
    ):
        reasons.append("fp_region_dwi_above_and_adc_below_brain_reference_iqr")
    if (
        dwi["fn"]["median"] is not None
        and ref_d["q1"] is not None
        and ref_d["q1"] <= dwi["fn"]["median"] <= ref_d["q3"]
    ):
        reasons.append("fn_region_dwi_within_brain_reference_iqr")
    return {"flag": sc.REVIEW_CANDIDATE if reasons else sc.NOT_FLAGGED, "reasons": reasons}


def _f(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x
