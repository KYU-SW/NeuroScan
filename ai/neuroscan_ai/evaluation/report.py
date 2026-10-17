"""Aggregate per-case evaluation rows into a report.

Aggregation policy
- Positive-patient statistics (Dice etc.) use only cases with ``has_lesion = True``.
- Negative patients are summarised separately (correct negatives, false-positive cases and
  volumes). They are never mixed into the positive means, and no empty-GT Dice is invented.
- Undefined values (None, e.g. precision for an empty prediction) are excluded from a metric's
  statistics and counted in ``n_undefined``.
- With only a handful of negatives, no specificity-style rate is reported as a headline number.

Output location is the caller's choice, e.g. results/phase2/, results/internal_test/<run>/,
results/external_validation/<dataset>/<run>/.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

POSITIVE_METRICS = (
    "dice",
    "iou",
    "sensitivity",
    "precision",
    "volume_error_ml",
    "absolute_volume_error_ml",
    "relative_volume_error_percent",
)

RESULT_SCOPES = ("internal_test", "external_validation")


def results_dir(root: Path, scope: str, *parts: str) -> Path:
    """results/<scope>/<parts...>; keeps internal test and external validation apart."""
    if scope not in RESULT_SCOPES:
        raise ValueError(f"scope must be one of {RESULT_SCOPES}")
    return Path(root, scope, *parts)


def _defined(values: Iterable[Any]) -> list[float]:
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out.append(float(v))
    return out


def metric_stats(values: Sequence[Any]) -> dict[str, Any]:
    vals = np.array(_defined(values))
    stats: dict[str, Any] = {"n": int(vals.size), "n_undefined": len(values) - int(vals.size)}
    if vals.size:
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        stats.update(
            mean=float(vals.mean()),
            std=float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            median=float(med),
            q1=float(q1),
            q3=float(q3),
            min=float(vals.min()),
            max=float(vals.max()),
        )
    return stats


def summarize_positive(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in rows if r["has_lesion"]]
    return {"n_cases": len(pos), **{m: metric_stats([r[m] for r in pos]) for m in POSITIVE_METRICS}}


def summarize_negative(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    neg = [r for r in rows if not r["has_lesion"]]
    return {
        "n_cases": len(neg),
        "correct_negative": sum(r["negative_outcome"] == "correct_negative" for r in neg),
        "false_positive_cases": sum(bool(r["any_false_positive"]) for r in neg),
        "cases": [
            {
                "case_id": r["case_id"],
                "outcome": r["negative_outcome"],
                "predicted_lesion_voxels": r["predicted_lesion_voxels"],
                "false_positive_volume_ml": r["false_positive_volume_ml"],
            }
            for r in neg
        ],
    }


def summarize(rows: Sequence[dict[str, Any]], subgroup_keys: Sequence[str] = ()) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_cases": len(rows),
        "positive": summarize_positive(rows),
        "negative": summarize_negative(rows),
        "subgroups": {},
    }
    for key in subgroup_keys:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(str(r.get(key)), []).append(r)
        summary["subgroups"][key] = {
            value: {"positive": summarize_positive(g), "negative": summarize_negative(g)}
            for value, g in sorted(groups.items())
        }
    return summary


def write_report(
    rows: Sequence[dict[str, Any]], summary: dict[str, Any], out_dir: Path, name: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
