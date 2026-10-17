"""Deterministic, stratified patient-wise train/val/test split.

Strategy
1. Negative cases (no GT lesion) are distributed round-robin over test, val, train so they never
   all land in one split.
2. Positive cases are stratified by acquisition group (thin vs thick slices, a proxy for
   centre/scanner) x lesion-volume quartile.
3. Within the concatenated, per-stratum shuffled list, each case goes to the split with the
   largest remaining deficit against its target count. This yields exact split sizes while
   spreading every stratum proportionally across splits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitCase:
    case_id: str
    has_lesion: bool
    lesion_volume_ml: float
    thick_slice: bool


def split_sizes(n: int, ratios: Sequence[float]) -> dict[str, int]:
    """val/test = floor(n * ratio); train takes the remainder."""
    val = math.floor(n * ratios[1])
    test = math.floor(n * ratios[2])
    return {"train": n - val - test, "val": val, "test": test}


def make_patient_split(
    cases: Sequence[SplitCase],
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    n_volume_bins: int = 4,
) -> dict[str, Any]:
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    ids = [c.case_id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case IDs")

    rng = np.random.default_rng(seed)
    ordered = sorted(cases, key=lambda c: c.case_id)  # input order must not matter
    targets = split_sizes(len(ordered), ratios)
    assigned: dict[str, list[str]] = {s: [] for s in SPLITS}

    negatives = [c for c in ordered if not c.has_lesion]
    for i, idx in enumerate(rng.permutation(len(negatives))):
        assigned[("test", "val", "train")[i % 3]].append(negatives[idx].case_id)

    positives = [c for c in ordered if c.has_lesion]
    edges = np.quantile(
        [c.lesion_volume_ml for c in positives], np.linspace(0, 1, n_volume_bins + 1)
    )
    inner_edges = edges[1:-1]

    def stratum(c: SplitCase) -> tuple[str, int]:
        return (
            "thick" if c.thick_slice else "thin",
            int(np.searchsorted(inner_edges, c.lesion_volume_ml, "right")),
        )

    strata: dict[tuple[str, int], list[SplitCase]] = {}
    for c in positives:
        strata.setdefault(stratum(c), []).append(c)
    sequence: list[SplitCase] = []
    for key in sorted(strata):
        members = strata[key]
        sequence.extend(members[i] for i in rng.permutation(len(members)))

    # Positive slots left per split after negatives; give each case to the most under-filled split.
    quota = {s: targets[s] - len(assigned[s]) for s in SPLITS}
    taken = dict.fromkeys(SPLITS, 0)
    for pos, c in enumerate(sequence, start=1):
        deficit = {s: quota[s] * pos / len(sequence) - taken[s] for s in SPLITS}
        best = max(SPLITS, key=lambda s: (deficit[s], -SPLITS.index(s)))
        assigned[best].append(c.case_id)
        taken[best] += 1

    by_id = {c.case_id: c for c in ordered}
    stats = {}
    for s in SPLITS:
        members = [by_id[i] for i in assigned[s]]
        vols = np.array([m.lesion_volume_ml for m in members])
        strata_counts: dict[str, int] = {}
        for m in members:
            key = "negative" if not m.has_lesion else "{}_q{}".format(*stratum(m))
            strata_counts[key] = strata_counts.get(key, 0) + 1
        stats[s] = {
            "n": len(members),
            "negatives": [m.case_id for m in members if not m.has_lesion],
            "thick_slice": sum(m.thick_slice for m in members),
            "lesion_volume_ml": {
                "median": round(float(np.median(vols)), 4),
                "mean": round(float(vols.mean()), 4),
                "min": float(vols.min()),
                "max": float(vols.max()),
            },
            "lt_1ml": int(((vols > 0) & (vols < 1)).sum()),
            "strata": dict(sorted(strata_counts.items())),
        }

    return {
        "seed": seed,
        "ratios": list(ratios),
        "n_cases": len(ordered),
        "strategy": __doc__.strip(),
        "volume_bin_edges_ml": [round(float(e), 4) for e in edges],
        "splits": {s: sorted(assigned[s]) for s in SPLITS},
        "stats": stats,
    }
