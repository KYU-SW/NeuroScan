"""v2 baseline evaluation and lesion-wise safety metrics, computed on the ORIGINAL grid.

Official evaluation uses the UNGATED prediction; gated metrics are a comparison only.

Component matching (26-connectivity, IDs from ``component_analysis.extraction``):
a predicted and a GT component that share >= 1 voxel form an EXPLORATORY match candidate. This is
not a clinical true positive. Every pair records overlap voxels, prediction / GT overlap ratios,
Dice and IoU. GT components with no overlapping prediction are recorded as missed.

Four separate false-positive-type measures (never interchangeable; none is a clinical FP count):
- ``false_positive_voxel_volume_ml``: voxel-level overprediction, ``prediction AND NOT GT`` (mL);
  used as the checkpoint-selection tie-break;
- ``no_gt_overlap_component_volume_ml`` / ``no_gt_overlap_component_count``: component-level
  candidates, predicted components sharing no voxel with GT;
- ``oversegmentation_volume_ml``: voxels outside GT inside predicted components that DO overlap GT.
The first equals the sum of the other two volumes. No FP threshold is optimised here.

Slice-thickness subgroup (analysis only, ``isles_v1_qa_thick_subgroup_3mm``): thin = effective z
spacing <= 3.0 mm, thick = > 3.0 mm. This is NOT the provisional component-safety (external
dataset) thick-slice warning (effective z > 5.0 mm or slice thickness >= 5.0 mm,
``component_analysis.warnings``); the two
rules are never used interchangeably and neither is a clinical threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neuroscan_ai.component_analysis.extraction import extract_components
from neuroscan_ai.component_analysis.warnings import (
    THICK_SLICE_THICKNESS_MM,
    THICK_Z_SPACING_MM,
)
from neuroscan_ai.v2.geometry import as_binary_mask, decode_binary_mask, validate_grid

MATCH_RULE = "overlap_ge_1_voxel_exploratory_not_clinical_tp"
SMALL_LESION_BINS_ML = (0.1, 0.5, 1.0)  # recall of GT components with volume < bin
SLICE_SUBGROUP_POLICY = "isles_v1_qa_thick_subgroup_3mm"
THICK_SUBGROUP_ABOVE_MM = 3.0  # ISLES analysis subgroup (not a clinical threshold)


def slice_subgroup(z_spacing_mm: float, slice_thickness_mm: float | None = None) -> dict:
    """ISLES thin / thick subgroup plus the SEPARATE provisional component-safety warning flag."""
    z = float(z_spacing_mm)
    warn = z > THICK_Z_SPACING_MM or (
        slice_thickness_mm is not None and float(slice_thickness_mm) >= THICK_SLICE_THICKNESS_MM
    )
    return {
        "policy": SLICE_SUBGROUP_POLICY,
        "effective_z_spacing_mm": round(z, 6),
        "slice_group": "thick" if z > THICK_SUBGROUP_ABOVE_MM else "thin",
        "component_safety_provisional_thick_warning": bool(warn),
        "component_safety_provisional_rule": (
            "effective z > 5.0 mm or slice thickness >= 5.0 mm (provisional, separate policy)"
        ),
        "slice_thickness_mm": slice_thickness_mm,
    }


def voxel_volume_ml(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(np.asarray(affine, dtype=float)[:3, :3]))) / 1000.0


def dice(pred: np.ndarray, gt: np.ndarray) -> float | None:
    p, g = pred.astype(bool), gt.astype(bool)
    denom = int(p.sum()) + int(g.sum())
    return None if denom == 0 else 2.0 * int((p & g).sum()) / denom


def match_components(pred: np.ndarray, gt: np.ndarray, vml: float) -> dict[str, Any]:
    plab, npred = extract_components(pred.astype(np.uint8))
    glab, ngt = extract_components(gt.astype(np.uint8))
    pvol = np.bincount(plab.ravel(), minlength=npred + 1)
    gvol = np.bincount(glab.ravel(), minlength=ngt + 1)
    both = (plab > 0) & (glab > 0)
    pairs_raw: dict[tuple[int, int], int] = {}
    if both.any():
        keys, counts = np.unique(
            np.stack([plab[both], glab[both]]).astype(np.int64), axis=1, return_counts=True
        )
        pairs_raw = {(int(a), int(b)): int(c) for (a, b), c in zip(keys.T, counts, strict=True)}
    pairs = []
    for (pi, gi), ov in sorted(pairs_raw.items()):
        p, g = int(pvol[pi]), int(gvol[gi])
        pairs.append(
            {
                "pred_component": pi,
                "gt_component": gi,
                "overlap_voxels": ov,
                "pred_overlap_ratio": round(ov / p, 6),
                "gt_overlap_ratio": round(ov / g, 6),
                "dice": round(2.0 * ov / (p + g), 6),
                "iou": round(ov / (p + g - ov), 6),
            }
        )
    gt_detected = {gi for (_, gi) in pairs_raw}
    pred_matched = {pi for (pi, _) in pairs_raw}
    gt_components = [
        {
            "gt_component": gi,
            "voxels": int(gvol[gi]),
            "volume_ml": round(int(gvol[gi]) * vml, 6),
            "detected_exploratory": gi in gt_detected,
        }
        for gi in range(1, ngt + 1)
    ]
    pred_components = [
        {
            "pred_component": pi,
            "voxels": int(pvol[pi]),
            "volume_ml": round(int(pvol[pi]) * vml, 6),
            "overlaps_gt": pi in pred_matched,
        }
        for pi in range(1, npred + 1)
    ]
    return {
        "match_rule": MATCH_RULE,
        "pairs": pairs,
        "gt_components": gt_components,
        "pred_components": pred_components,
    }


def case_metrics(
    pred: np.ndarray, gt: np.ndarray, affine: np.ndarray, output: str = "ungated"
) -> dict[str, Any]:
    """Metrics for one case on the original grid (``pred`` and ``gt`` must be 0/1)."""
    p, g = as_binary_mask(pred, "prediction"), decode_binary_mask(gt, "GT")[0]
    if p.shape != g.shape:
        raise ValueError("prediction and GT shapes differ")
    grid = validate_grid(p.shape, affine)
    vml = voxel_volume_ml(grid.affine)
    m = match_components(p, g, vml)
    missed = [c for c in m["gt_components"] if not c["detected_exploratory"]]
    fp_comp = [c for c in m["pred_components"] if not c["overlaps_gt"]]
    plab, _ = extract_components(p.astype(np.uint8))
    overlapping = [c["pred_component"] for c in m["pred_components"] if c["overlaps_gt"]]
    overseg = int((np.isin(plab, overlapping) & ~g).sum()) if overlapping else 0
    z = grid.spacing[2]
    return {
        "output": output,
        "gt_positive": bool(g.any()),
        "dice": dice(p, g),
        "pred_volume_ml": round(int(p.sum()) * vml, 6),
        "gt_volume_ml": round(int(g.sum()) * vml, 6),
        "false_positive_voxel_volume_ml": round(int((p & ~g).sum()) * vml, 6),
        "no_gt_overlap_component_volume_ml": round(sum(c["voxels"] for c in fp_comp) * vml, 6),
        "no_gt_overlap_component_count": len(fp_comp),
        "oversegmentation_volume_ml": round(overseg * vml, 6),
        "missed_gt_count": len(missed),
        "missed_gt_volumes_ml": [c["volume_ml"] for c in missed],
        "z_spacing_mm": round(float(z), 6),
        "slice_group": slice_subgroup(z)["slice_group"],
        "slice_subgroup_policy": SLICE_SUBGROUP_POLICY,
        "components": m,
    }


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "min": round(float(a.min()), 6),
        "median": round(float(np.median(a)), 6),
        "max": round(float(a.max()), 6),
    }


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else round(num / den, 6)


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate safety metrics over cases that share one output type."""
    outputs = {c["output"] for c in cases}
    if len(outputs) > 1:
        raise ValueError("cannot mix ungated and gated case metrics in one summary")
    pos = [c for c in cases if c["gt_positive"]]
    neg = [c for c in cases if not c["gt_positive"]]
    gt_comps = [g for c in pos for g in c["components"]["gt_components"]]
    detected = [g for g in gt_comps if g["detected_exploratory"]]
    small = {}
    for b in SMALL_LESION_BINS_ML:
        in_bin = [g for g in gt_comps if g["volume_ml"] < b]
        small[f"lt_{b}_ml"] = {
            "gt_components": len(in_bin),
            "detected": sum(g["detected_exploratory"] for g in in_bin),
            "recall": _ratio(sum(g["detected_exploratory"] for g in in_bin), len(in_bin)),
        }
    missed_vols = [v for c in pos for v in c["missed_gt_volumes_ml"]]
    dices = [c["dice"] for c in pos if c["dice"] is not None]
    groups = {}
    for name in ("thin", "thick"):
        sub = [c for c in cases if c["slice_group"] == name]
        groups[name] = (
            {"cases": len(sub), **{k: v for k, v in summarize(sub).items() if k != "slice_groups"}}
            if sub and len(sub) != len(cases)
            else {"cases": len(sub)}
        )
    return {
        "output": next(iter(outputs)) if outputs else None,
        "match_rule": MATCH_RULE,
        "cases": len(cases),
        "positive_cases": len(pos),
        "no_lesion_cases": len(neg),
        "mean_dice_positive": round(float(np.mean(dices)), 6) if dices else None,
        "lesion_recall_exploratory": _ratio(len(detected), len(gt_comps)),
        "gt_components": len(gt_comps),
        "missed_gt_count": len(missed_vols),
        "missed_gt_volume_ml": _quantiles(missed_vols),
        "small_lesion_recall": small,
        "case_sensitivity_exploratory": _ratio(
            sum(
                any(g["detected_exploratory"] for g in c["components"]["gt_components"])
                for c in pos
            ),
            len(pos),
        ),
        "no_lesion_fp_volume_ml": round(sum(c["pred_volume_ml"] for c in neg), 6),
        "no_lesion_cases_with_prediction": sum(c["pred_volume_ml"] > 0 for c in neg),
        "false_positive_voxel_volume_ml": round(
            sum(c["false_positive_voxel_volume_ml"] for c in cases), 6
        ),
        "no_gt_overlap_component_volume_ml": round(
            sum(c["no_gt_overlap_component_volume_ml"] for c in cases), 6
        ),
        "no_gt_overlap_component_count": sum(c["no_gt_overlap_component_count"] for c in cases),
        "oversegmentation_volume_ml": round(sum(c["oversegmentation_volume_ml"] for c in cases), 6),
        "slice_groups": groups,
    }


def ungated_vs_gated(ungated: dict[str, Any], gated: dict[str, Any]) -> dict[str, Any]:
    """Gated minus ungated for the numeric summary fields (comparison only)."""
    if ungated.get("output") != "ungated" or gated.get("output") != "gated":
        raise ValueError("expected an ungated and a gated summary")
    keys = (
        "mean_dice_positive",
        "lesion_recall_exploratory",
        "missed_gt_count",
        "case_sensitivity_exploratory",
        "no_lesion_fp_volume_ml",
        "false_positive_voxel_volume_ml",
        "no_gt_overlap_component_volume_ml",
        "no_gt_overlap_component_count",
        "oversegmentation_volume_ml",
    )
    diff = {}
    for k in keys:
        a, b = ungated.get(k), gated.get(k)
        diff[k] = None if a is None or b is None else round(b - a, 6)
    return {"official": "ungated", "difference_gated_minus_ungated": diff}
