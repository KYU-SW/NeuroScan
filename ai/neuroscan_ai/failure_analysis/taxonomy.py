"""Metrics-only taxonomy flags and per-case summaries (observations kept apart from hypotheses).

Only rules that the result tables can decide are applied (T1, T2, T3, T8 candidate); the rest
are ``pending_image_review``. A flag describes a measurable pattern, never a cause, a diagnosis
or a statement that the ground truth is wrong. ``hypotheses`` is always left empty here.
"""

from __future__ import annotations

import math
from typing import Any

from neuroscan_ai.failure_analysis import schemas as sc
from neuroscan_ai.failure_analysis.selection import CaseRecord, _num


def _clean(v: Any) -> Any:
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def model_flags(row: dict[str, Any], has_lesion: bool) -> dict[str, str]:
    if not has_lesion:
        return {c: sc.NOT_APPLICABLE for c in ("T1", "T2", "T3")}
    sens, prec = _num(row.get("sensitivity")), _num(row.get("precision"))
    fn, fp, tp = _num(row.get("fn_voxels")), _num(row.get("fp_voxels")), _num(row.get("tp_voxels"))
    pred = _num(row.get("prediction_voxels"))
    t1 = sens < 0.5 and fn > fp
    t2 = prec < 0.5 and fp > fn  # undefined precision (empty prediction) never flags T2
    t3 = pred > 0 and tp == 0
    return {
        "T1": sc.FLAGGED if t1 else sc.NOT_FLAGGED,
        "T2": sc.FLAGGED if t2 else sc.NOT_FLAGGED,
        "T3": sc.FLAGGED if t3 else sc.NOT_FLAGGED,
    }


def t8_flag(rec: CaseRecord) -> str:
    if not rec.has_lesion:
        return sc.NOT_APPLICABLE
    thick = str(rec.meta("spacing_group")) == "thick"
    worse = math.isfinite(rec.delta) and rec.delta < -sc.TIE_TOLERANCE
    return sc.REVIEW_CANDIDATE if thick and worse else sc.NOT_FLAGGED


def summarize(rec: CaseRecord) -> dict[str, Any]:
    obs_models = {}
    for m in sc.MODELS:
        row = rec.rows[m]
        obs = {k: _clean(row.get(k)) for k in sc.METRIC_FIELDS}
        if not rec.has_lesion:
            obs.update({k: _clean(row.get(k)) for k in sc.NEGATIVE_FIELDS})
        obs_models[m] = obs
    delta = rec.delta if math.isfinite(rec.delta) else None
    comp = {
        m: (
            _num(rec.rows[m].get("component_count_prediction"))
            - _num(rec.rows[m].get("component_count_gt"))
        )
        for m in sc.MODELS
    }
    flags = {m: model_flags(rec.rows[m], rec.has_lesion) for m in sc.MODELS}
    pending = [{"code": c, "rule": sc.TAXONOMY[c]} for c in sc.PENDING_CODES]
    if not rec.has_lesion:
        pending = [p for p in pending if p["code"] not in ("T4", "T6")]
    return {
        "alias": rec.alias,
        "groups": list(rec.groups),
        "selection_reasons": list(rec.reasons),
        "has_lesion": rec.has_lesion,
        "observations": {
            "metrics": obs_models,
            "dice_difference_25d_minus_2d": delta,
            "component_count_difference_prediction_minus_gt": {
                m: (None if math.isnan(v) else v) for m, v in comp.items()
            },
            "metadata": {k: _clean(rec.meta(k)) for k in sc.META_FIELDS},
        },
        "taxonomy_flags": {**flags, "T8": t8_flag(rec)},
        "pending_image_review": pending,
        "hypotheses": [],
        "unknowns": [
            "spatial location of FP / FN voxels (no image or map was opened)",
            "component matching type and distances",
            "DWI / ADC signal in TP / FP / FN regions",
        ],
        "suggested_experiments": [],
        "note": "Metric-based observation only; not a diagnosis, not a statement on ground-truth "
        "correctness, not a reason to change the v1 model or policy.",
    }


def aggregate(summaries: list[dict[str, Any]], groups: dict[str, list[str]]) -> dict[str, Any]:
    def count(model: str, code: str, value: str) -> int:
        return sum(s["taxonomy_flags"][model][code] == value for s in summaries)

    tax = {
        m: {
            c: {v: count(m, c, v) for v in (sc.FLAGGED, sc.NOT_FLAGGED, sc.NOT_APPLICABLE)}
            for c in ("T1", "T2", "T3")
        }
        for m in sc.MODELS
    }
    t8 = {
        v: sum(s["taxonomy_flags"]["T8"] == v for s in summaries)
        for v in (sc.REVIEW_CANDIDATE, sc.NOT_FLAGGED, sc.NOT_APPLICABLE)
    }
    pend: dict[str, int] = {}
    for s in summaries:
        for p in s["pending_image_review"]:
            pend[p["code"]] = pend.get(p["code"], 0) + 1
    by_meta = {
        k: {} for k in ("spacing_group", "adc_unit_group", "volume_group", "obliquity_group")
    }
    for s in summaries:
        for k in by_meta:
            v = str(s["observations"]["metadata"].get(k))
            by_meta[k][v] = by_meta[k].get(v, 0) + 1
    return {
        "mode": sc.MODE_METRICS_ONLY,
        "unique_aliases": len(summaries),
        "group_counts": {g: len(groups.get(g, [])) for g in sc.GROUP_ORDER},
        "group_rules": sc.GROUP_RULES,
        "taxonomy_rules": sc.TAXONOMY,
        "taxonomy_counts": tax,
        "t8_counts": t8,
        "pending_image_review_counts": pend,
        "selected_by_metadata": by_meta,
        "thresholds": {
            "fail_dice": sc.FAIL_DICE,
            "good_dice": sc.GOOD_DICE,
            "small_gt_component_ml": sc.SMALL_GT_COMPONENT_ML,
            "distant_fp_mm": sc.DISTANT_FP_MM,
            "location_mismatch_overlap": sc.LOCATION_MISMATCH_OVERLAP,
        },
        "note": "Review-support counts only; no cause, diagnosis or ground-truth judgement.",
    }


def overview_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for s in summaries:
        o = s["observations"]
        rows.append(
            {
                "alias": s["alias"],
                "groups": "+".join(s["groups"]),
                "has_lesion": s["has_lesion"],
                "dice_2d": o["metrics"]["2d"]["dice"],
                "dice_25d": o["metrics"]["25d"]["dice"],
                "dice_difference": o["dice_difference_25d_minus_2d"],
                **{k: o["metadata"][k] for k in sc.META_FIELDS},
                **{
                    f"{c}_{m}": s["taxonomy_flags"][m][c]
                    for m in sc.MODELS
                    for c in ("T1", "T2", "T3")
                },
                "T8": s["taxonomy_flags"]["T8"],
                "pending_image_review": "+".join(p["code"] for p in s["pending_image_review"]),
            }
        )
    return rows
