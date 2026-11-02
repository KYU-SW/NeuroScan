"""Aggregation, paired statistics, subgroups, negative summary and QA selection (pure functions).

Inputs are the patient rows of ``evaluation.segmentation.evaluate_case`` for the 2D and the 2.5D
model on the same patients. Positive patients (non-empty GT) carry every overlap statistic;
negative patients are summarised separately and never enter a positive mean. Undefined values
(None / NaN) are excluded per metric and counted; they are never replaced by 0 or 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from neuroscan_ai.evaluation.comparison import (
    bootstrap_mean_ci,
    paired_summary,
    paired_table,
    wilcoxon_p,
)

PRIMARY = "dice"
SECONDARY = (
    "iou",
    "sensitivity",
    "precision",
    "volume_error_ml",
    "absolute_volume_error_ml",
    "relative_volume_error_percent",
    "component_count_prediction",
    "component_count_gt",
)
NEGATIVE_FIELDS = (
    "negative_outcome",
    "prediction_voxels",
    "false_positive_volume_ml",
    "false_positive_component_count",
)


def _num(v: Any) -> float:
    try:
        return math.nan if v is None else float(v)
    except (TypeError, ValueError):
        return math.nan


def describe(values: Sequence[Any]) -> dict[str, Any]:
    x = np.array([_num(v) for v in values], dtype=float)
    ok = x[np.isfinite(x)]
    if ok.size == 0:
        return {"n": 0, "n_undefined": int(x.size)}
    return {
        "n": int(ok.size),
        "n_undefined": int(x.size - ok.size),
        "mean": float(ok.mean()),
        "median": float(np.median(ok)),
        "sd": float(ok.std(ddof=1)) if ok.size > 1 else None,
        "min": float(ok.min()),
        "q1": float(np.quantile(ok, 0.25)),
        "q3": float(np.quantile(ok, 0.75)),
        "max": float(ok.max()),
    }


def positives(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if bool(r["has_lesion"])]


def negatives(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not bool(r["has_lesion"])]


def aggregate(rows_by_model: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, rows in rows_by_model.items():
        pos, neg = positives(rows), negatives(rows)
        out[name] = {
            "n_patients": len(rows),
            "n_positive": len(pos),
            "n_negative": len(neg),
            "positive": {m: describe([r.get(m) for r in pos]) for m in (PRIMARY, *SECONDARY)},
        }
    return out


def paired_positive_table(
    rows_2d: Sequence[dict[str, Any]],
    rows_25d: Sequence[dict[str, Any]],
    subgroup_meta: dict[str, dict[str, Any]] | None = None,
    subgroup_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """One row per positive patient: <metric>_base (2D), _cand (2.5D), _delta (2.5D - 2D)."""
    pos_2d, pos_25d = positives(rows_2d), positives(rows_25d)
    if {r["case_id"] for r in pos_2d} != {r["case_id"] for r in pos_25d}:
        raise ValueError("2D and 2.5D positive patient sets differ")
    if subgroup_meta is not None:
        for r in pos_2d:
            meta = subgroup_meta.get(r["case_id"], {})
            for c in subgroup_columns:
                r.setdefault(c, meta.get(c))
    return paired_table(pos_2d, pos_25d, (PRIMARY, *SECONDARY), keep=tuple(subgroup_columns))


def statistical_comparison(table: pd.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    def summary(metric: str) -> dict[str, Any]:
        s = paired_summary(table, metric, n_boot=n_boot, seed=seed)
        d = table[f"{metric}_delta"].to_numpy(float)
        d = d[np.isfinite(d)]
        s["sd_delta"] = float(d.std(ddof=1)) if d.size > 1 else None
        for side, col in (("base", f"{metric}_base"), ("cand", f"{metric}_cand")):
            v = table[col].to_numpy(float)
            v = v[np.isfinite(table[f"{metric}_delta"].to_numpy(float))]
            s[f"sd_{side}"] = float(v.std(ddof=1)) if v.size > 1 else None
        return s

    return {
        "naming": {"base": "2D", "cand": "2.5D", "delta": "2.5D - 2D (per patient)"},
        "bootstrap": {
            "method": "percentile",
            "statistic": "mean paired difference",
            "n_boot": n_boot,
            "seed": seed,
            "ci": 0.95,
        },
        "wilcoxon": "two-sided signed-rank on paired differences",
        "primary": {"endpoint": "positive-patient mean Dice", **summary(PRIMARY)},
        "secondary_exploratory": {m: summary(m) for m in SECONDARY},
        "note": "Only the primary endpoint is confirmatory; secondary p-values are exploratory.",
    }


def subgroup_summary(
    table: pd.DataFrame, columns: Sequence[str], min_n: int, n_boot: int, seed: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        groups: dict[str, Any] = {}
        for name, g in table.groupby(table[col].astype(str), sort=True):
            d = g[f"{PRIMARY}_delta"].to_numpy(float)
            d = d[np.isfinite(d)]
            entry = {
                "n": int(len(g)),
                "dice_2d": describe(g[f"{PRIMARY}_base"].tolist()),
                "dice_25d": describe(g[f"{PRIMARY}_cand"].tolist()),
                "dice_delta": describe(d.tolist()),
                "n_improved": int((d > 1e-3).sum()),
                "n_worse": int((d < -1e-3).sum()),
            }
            if len(d) >= min_n:
                lo, hi = bootstrap_mean_ci(d, n_boot, seed)
                entry["exploratory"] = {"ci95": [lo, hi], "wilcoxon_p": wilcoxon_p(d)}
            else:
                entry["descriptive_only"] = f"n < {min_n}: no interval or p-value"
            groups[str(name)] = entry
        out[col] = groups
    return {
        "primary_metric": PRIMARY,
        "min_n_for_inference": min_n,
        "subgroups": out,
        "note": "Subgroups are descriptive/exploratory; no significance claim.",
    }


def negative_summary(
    rows_2d: Sequence[dict[str, Any]], rows_25d: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    def per_model(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"case_id": r["case_id"], **{f: r.get(f) for f in NEGATIVE_FIELDS}}
            for r in sorted(negatives(rows), key=lambda r: r["case_id"])
        ]

    a, b = per_model(rows_2d), per_model(rows_25d)
    return {
        "n_negative": len(a),
        "policy": "excluded from positive means; overlap metrics undefined (not 0 or 1)",
        "2d": {
            "correct_negative": sum(r["negative_outcome"] == "correct_negative" for r in a),
            "cases": a,
        },
        "25d": {
            "correct_negative": sum(r["negative_outcome"] == "correct_negative" for r in b),
            "cases": b,
        },
    }


def failure_cases(table: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    bad = table[(table[f"{PRIMARY}_base"] < threshold) | (table[f"{PRIMARY}_cand"] < threshold)]
    return [
        {"case_id": r.case_id, "dice_2d": r.dice_base, "dice_25d": r.dice_cand}
        for r in bad.sort_values("case_id").itertuples()
    ]


def qa_selection(
    table: pd.DataFrame, neg: dict[str, Any], n_each: int = 3, threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Pre-registered QA picks; a case may carry several reasons. Deterministic ordering."""
    reasons: dict[str, list[str]] = {}
    t = table[np.isfinite(table[f"{PRIMARY}_delta"])].sort_values(
        [f"{PRIMARY}_delta", "case_id"], ascending=[False, True]
    )
    for cid in t.head(n_each)["case_id"]:
        reasons.setdefault(cid, []).append("largest_improvement")
    for cid in t.sort_values([f"{PRIMARY}_delta", "case_id"]).head(n_each)["case_id"]:
        reasons.setdefault(cid, []).append("largest_deterioration")
    for f in failure_cases(table, threshold):
        reasons.setdefault(f["case_id"], []).append(f"dice_below_{threshold:.2f}")
    for model in ("2d", "25d"):
        for c in neg[model]["cases"]:
            if c["negative_outcome"] != "correct_negative":
                reasons.setdefault(c["case_id"], []).append(f"negative_false_positive_{model}")
    return [{"case_id": cid, "reasons": sorted(set(r))} for cid, r in sorted(reasons.items())]
