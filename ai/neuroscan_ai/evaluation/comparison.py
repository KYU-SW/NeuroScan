"""Paired patient-wise comparison of two runs evaluated on the same patients (e.g. 2D vs 2.5D).

Policy
- Both runs must cover exactly the same cases; a mismatch is an error, never silently dropped.
- Statistics are on per-patient differences (candidate - baseline) over positive patients.
- A pair with an undefined value (e.g. precision of an empty prediction) is excluded from that
  metric only and counted in ``n_undefined``.
- The bootstrap is a seeded percentile bootstrap of the mean difference; the Wilcoxon
  signed-rank test is two-sided. Both are reported, neither is used to tune anything.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# The only keys a 2.5D run may change relative to its 2D baseline (Phase 3 doc, section 19).
ALLOWED_CONFIG_DIFFERENCES = frozenset(
    {"run_name", "data.context_slices", "checkpoint.dir", "output.results_dir"}
)


def flatten_config(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_config(v, f"{key}."))
        else:
            out[key] = v
    return out


def config_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Dotted keys whose values differ (missing on one side counts as different)."""
    fa, fb = flatten_config(a), flatten_config(b)
    return {
        k: (fa.get(k), fb.get(k)) for k in sorted(fa.keys() | fb.keys()) if fa.get(k) != fb.get(k)
    }


def _number(v: Any) -> float:
    if v is None:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def paired_table(
    baseline: Iterable[dict[str, Any]],
    candidate: Iterable[dict[str, Any]],
    metrics: Sequence[str],
    keep: Sequence[str] = (),
) -> pd.DataFrame:
    """One row per case: ``<metric>_base``, ``<metric>_cand``, ``<metric>_delta`` + ``keep``
    columns taken from the baseline rows."""
    base = {r["case_id"]: r for r in baseline}
    cand = {r["case_id"]: r for r in candidate}
    if base.keys() != cand.keys():
        only_b = sorted(base.keys() - cand.keys())
        only_c = sorted(cand.keys() - base.keys())
        raise ValueError(f"case sets differ: only baseline {only_b}, only candidate {only_c}")
    rows = []
    for cid in sorted(base):
        row: dict[str, Any] = {"case_id": cid}
        for k in keep:
            row[k] = base[cid].get(k)
        for m in metrics:
            b, c = _number(base[cid].get(m)), _number(cand[cid].get(m))
            row[f"{m}_base"], row[f"{m}_cand"], row[f"{m}_delta"] = b, c, c - b
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: Sequence[float], n_boot: int = 10_000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def wilcoxon_p(values: Sequence[float]) -> float | None:
    x = np.asarray(values, dtype=float)
    if x.size == 0 or np.all(x == 0):
        return None
    return float(wilcoxon(x, alternative="two-sided").pvalue)


def paired_summary(
    table: pd.DataFrame,
    metric: str,
    tie_tolerance: float = 1e-3,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    b, c, d = (table[f"{metric}_{s}"].to_numpy(float) for s in ("base", "cand", "delta"))
    ok = np.isfinite(d)
    b, c, d = b[ok], c[ok], d[ok]
    lo, hi = bootstrap_mean_ci(d, n_boot, seed)
    return {
        "metric": metric,
        "n": int(d.size),
        "n_undefined": int((~ok).sum()),
        "mean_base": float(b.mean()) if d.size else math.nan,
        "mean_cand": float(c.mean()) if d.size else math.nan,
        "median_base": float(np.median(b)) if d.size else math.nan,
        "median_cand": float(np.median(c)) if d.size else math.nan,
        "mean_delta": float(d.mean()) if d.size else math.nan,
        "median_delta": float(np.median(d)) if d.size else math.nan,
        "ci95": [lo, hi],
        "wilcoxon_p": wilcoxon_p(d),
        "n_improved": int((d > tie_tolerance).sum()),
        "n_worse": int((d < -tie_tolerance).sum()),
        "n_tied": int((np.abs(d) <= tie_tolerance).sum()),
        "tie_tolerance": tie_tolerance,
    }
