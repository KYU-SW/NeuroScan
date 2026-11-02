"""Deterministic case selection (groups F, Z, I, W, N, G) and local alias assignment.

Input: the per-patient rows of the Final Test tables (no image is read). Ordering is fully
deterministic: every sort ends on the case ID, so ties never depend on file order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from neuroscan_ai.failure_analysis import schemas as sc


@dataclass
class CaseRecord:
    case_id: str
    has_lesion: bool
    rows: dict[str, dict[str, Any]]  # model key -> patient metrics row
    groups: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    alias: str | None = None

    def dice(self, model: str) -> float:
        return _num(self.rows[model].get("dice"))

    @property
    def delta(self) -> float:
        return self.dice("25d") - self.dice("2d")

    def meta(self, key: str) -> Any:
        return self.rows["2d"].get(key)


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def _is_true(v: Any) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1")


def build_records(
    rows_2d: Sequence[dict[str, Any]], rows_25d: Sequence[dict[str, Any]]
) -> dict[str, CaseRecord]:
    a = {r["case_id"]: r for r in rows_2d}
    b = {r["case_id"]: r for r in rows_25d}
    if a.keys() != b.keys():
        raise ValueError("2D and 2.5D patient sets differ")
    return {
        cid: CaseRecord(cid, _is_true(a[cid]["has_lesion"]), {"2d": a[cid], "25d": b[cid]})
        for cid in sorted(a)
    }


def select(records: dict[str, CaseRecord]) -> dict[str, list[str]]:
    """Group -> ordered case IDs."""
    pos = [r for r in records.values() if r.has_lesion]
    neg = [r for r in records.values() if not r.has_lesion]
    finite = [r for r in pos if math.isfinite(r.delta)]
    groups: dict[str, list[str]] = {}

    f = [r for r in pos if r.dice("2d") < sc.FAIL_DICE or r.dice("25d") < sc.FAIL_DICE]
    groups["F"] = [
        r.case_id for r in sorted(f, key=lambda r: (min(r.dice("2d"), r.dice("25d")), r.case_id))
    ]
    groups["Z"] = sorted(r.case_id for r in pos if r.dice("25d") == 0.0)
    groups["I"] = [
        r.case_id for r in sorted(finite, key=lambda r: (-r.delta, r.case_id))[: sc.N_EXTREME]
    ]
    groups["W"] = [
        r.case_id for r in sorted(finite, key=lambda r: (r.delta, r.case_id))[: sc.N_EXTREME]
    ]
    groups["N"] = sorted(r.case_id for r in neg)

    good = [r for r in pos if r.dice("2d") >= sc.GOOD_DICE and r.dice("25d") >= sc.GOOD_DICE]
    g: list[str] = []
    for spacing in ("thin", "thick"):
        cand = [r for r in good if str(r.meta("spacing_group")) == spacing]
        if not cand:
            continue
        means = sorted((r.dice("2d") + r.dice("25d")) / 2 for r in cand)
        mid = (means[(len(means) - 1) // 2] + means[len(means) // 2]) / 2
        best = min(
            cand,
            key=lambda r: (abs((r.dice("2d") + r.dice("25d")) / 2 - mid), abs(r.delta), r.case_id),
        )
        g.append(best.case_id)
    groups["G"] = g
    return groups


def assign(records: dict[str, CaseRecord], groups: dict[str, list[str]]) -> list[CaseRecord]:
    """Aliases fa-01.. in group priority F -> Z -> I -> W -> N -> G; a repeat keeps its alias."""
    ordered: list[CaseRecord] = []
    for g in sc.GROUP_ORDER:
        for cid in groups.get(g, []):
            rec = records[cid]
            if rec.alias is None:
                rec.alias = f"fa-{len(ordered) + 1:02d}"
                ordered.append(rec)
            if g not in rec.groups:
                rec.groups.append(g)
                rec.reasons.append(_reason(g, rec))
    return ordered


def _reason(group: str, rec: CaseRecord) -> str:
    if group == "G":
        return f"{sc.GROUP_RULES['G']} ({rec.meta('spacing_group')} representative)"
    return sc.GROUP_RULES[group]
