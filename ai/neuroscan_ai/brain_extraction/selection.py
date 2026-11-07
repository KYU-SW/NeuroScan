"""Pilot case selection (deterministic) and the tool-selection rule.

Nothing here opens an image. ISLES selection works on a metadata table that the caller builds
from the Train split only; any non-Train row is refused. SOOP uses the fixed 5-case pilot.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from neuroscan_ai.brain_extraction.interfaces import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_REVIEW,
    TOOLS,
    BrainExtractionError,
)

SOOP_PILOT_ALIASES = tuple(f"soop-pilot-{i:02d}" for i in range(1, 6))
ISLES_SALT = "neuroscan-bet-isles-v1"
CATEGORY_ORDER = ("thick_slice", "high_obliquity", "small_lesion", "large_lesion")

# Ordered tool-selection criteria (documented; GT preservation and brain coverage are gates).
SELECTION_CRITERIA = (
    "gt_lesion_preservation",
    "no_brain_tissue_loss",
    "scalp_eye_external_tissue_removed",
    "dwi_adc_grid_preserved",
    "consistent_on_soop_and_isles",
    "stable_automatic_processing",
    "offline_execution",
    "installation_and_operation_complexity",
    "runtime_and_gpu_need",
    "license_fit",
)


def soop_pilot_plan() -> list[dict[str, Any]]:
    """The fixed SOOP pilot: aliases only, permanently excluded from every split / evaluation."""
    return [
        {"alias": a, "dataset": "soop", "use": "brain_mask_qa_only"} for a in SOOP_PILOT_ALIASES
    ]


@dataclass(frozen=True)
class IslesCriteria:
    small_max_ml: float = 1.0  # small: 0 < volume < 1 mL (v1 volume group "(0,1) mL")
    large_min_ml: float = 50.0  # large: volume >= 50 mL (v1 top volume group)
    thick_min_mm: float = 3.0  # thick: slice spacing > 3 mm (v1 THICK_SLICE_MM)
    oblique_min_deg: float = 10.0  # high obliquity: >= 10 deg (v1 group "[10,20) deg" and up)
    per_category: int = 2
    max_cases: int = 6


def categories(row: Mapping[str, Any], c: IslesCriteria) -> set[str]:
    v, sp, ob = row["lesion_volume_ml"], row["slice_spacing_mm"], row["obliquity_deg"]
    out = set()
    if 0 < v < c.small_max_ml:
        out.add("small_lesion")
    if v >= c.large_min_ml:
        out.add("large_lesion")
    if sp > c.thick_min_mm:
        out.add("thick_slice")
    if ob >= c.oblique_min_deg:
        out.add("high_obliquity")
    return out


def _key(case_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{case_id}".encode()).hexdigest()


def select_isles_cases(
    rows: Iterable[Mapping[str, Any]], criteria: IslesCriteria | None = None, salt: str = ISLES_SALT
) -> dict[str, Any]:
    """Deterministic greedy selection: categories in CATEGORY_ORDER, each needs ``per_category``.

    For the first unmet category, take the case (in SHA-256 order) of that category that covers
    the most still-unmet categories. Stop at ``max_cases`` or when every category is met or empty.
    Returns the internal mapping (with case IDs; keep local) and a public summary (aliases only).
    """
    c = criteria or IslesCriteria()
    rows = list(rows)
    if any(r.get("split") != "train" for r in rows):
        raise BrainExtractionError("ISLES selection accepts Train rows only")
    cand = sorted(((r["case_id"], categories(r, c)) for r in rows), key=lambda x: _key(x[0], salt))
    need = dict.fromkeys(CATEGORY_ORDER, c.per_category)
    chosen: list[tuple[str, set[str]]] = []
    while len(chosen) < c.max_cases:
        open_cats = [k for k in CATEGORY_ORDER if need[k] > 0]
        picked = None
        for cat in open_cats:
            pool = [(cid, cs) for cid, cs in cand if cat in cs and cid not in dict(chosen)]
            if pool:
                picked = max(pool, key=lambda x: sum(need[k] > 0 for k in x[1]))  # first max wins
                break
        if picked is None:
            break
        chosen.append(picked)
        for k in picked[1]:
            need[k] = max(0, need[k] - 1)
    mapping = [
        {"alias": f"isles-bet-{i:02d}", "case_id": cid, "categories": sorted(cs)}
        for i, (cid, cs) in enumerate(chosen, 1)
    ]
    covered = {k: sum(k in m["categories"] for m in mapping) for k in CATEGORY_ORDER}
    return {
        "internal_mapping": mapping,
        "public_summary": {
            "salt": salt,
            "criteria": asdict(c),
            "cases": [{"alias": m["alias"], "categories": m["categories"]} for m in mapping],
            "category_coverage": covered,
            "categories_short": [k for k in CATEGORY_ORDER if covered[k] < c.per_category],
        },
    }


def summarize_tools(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-tool status counts and the gate decision. The final choice also needs the user's
    visual QA; a tool with any ``fail`` (incl. GT preservation failure) is not eligible.

    record keys: tool, alias, status, gt_preservation_failure,
    visual_qa ('approved' | 'rejected' | None)
    """
    per: dict[str, dict[str, Any]] = {}
    for r in records:
        t = r["tool"]
        if t not in TOOLS:
            raise BrainExtractionError(f"unknown tool '{t}'")
        s = per.setdefault(
            t,
            {STATUS_PASS: 0, STATUS_REVIEW: 0, STATUS_FAIL: 0, "gt_failures": 0, "cases": 0,
             "visual_approved": 0, "visual_rejected": 0, "visual_missing": 0},
        )  # fmt: skip
        s["cases"] += 1
        s[r["status"]] += 1
        s["gt_failures"] += bool(r.get("gt_preservation_failure"))
        v = r.get("visual_qa")
        s[
            "visual_approved" if v == "approved" else "visual_rejected" if v else "visual_missing"
        ] += 1
    eligible = [
        t
        for t, s in per.items()
        if t != "v1_background"
        and s[STATUS_FAIL] == 0
        and s["gt_failures"] == 0
        and s["visual_approved"] == s["cases"]
    ]
    decision = (
        "no tool passed: brain-mask policy is not fixed"
        if not eligible
        else "eligible candidates; apply SELECTION_CRITERIA in order and record the choice"
    )
    return {"per_tool": per, "eligible": sorted(eligible), "decision": decision,
            "criteria_order": list(SELECTION_CRITERIA)}  # fmt: skip
