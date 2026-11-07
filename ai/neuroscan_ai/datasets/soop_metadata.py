"""SOOP public-sidecar aggregation (K4): counts and distributions only, never a subject ID.

Inputs are BIDS JSON sidecars (text) of eligible subjects and, optionally, the dataset-level
``dwi.bval``. No image is read. Every output is an aggregate over subjects.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np

from neuroscan_ai.datasets.soop import (
    ADC_CONTENT_ELIGIBLE,
    ADC_CONTENT_POOL_POLICY,
    ADC_CONTENT_RULE_VERSION,
    ADC_CONTENT_STATUSES,
    ADC_GROUP_RULE_VERSION,
    POOL_CANDIDATE,
    POOL_EXCLUDED,
    POOL_REVIEW,
    REQUIRED,
    adc_content_status,
    adc_storage_group,
    eligibility,
)

FIELDS = (
    "SeriesDescription",
    "ProtocolName",
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturersModelName",
    "SoftwareVersions",
    "ConversionSoftwareVersion",
    "ImageType",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PhilipsScaleSlope",
    "PhilipsRescaleSlope",
    "PhilipsRescaleIntercept",
    "PhilipsRWVSlope",
    "PhilipsRWVIntercept",
    "UsePhilipsFloatNotDisplayScaling",
    "Units",
)
NUMERIC = (
    "MagneticFieldStrength",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PhilipsScaleSlope",
    "PhilipsRescaleSlope",
    "PhilipsRescaleIntercept",
    "PhilipsRWVSlope",
    "PhilipsRWVIntercept",
)
BVALUE_HINTS = ("bval", "bvalue", "b_value", "diffusionbvalue")
TOP_N = 25


def eligible_subjects(paths: Iterable[str]) -> tuple[list[str], Counter]:
    """Eligible subject IDs from a BIDS path list and a count of missing required kinds."""
    available = set(paths)
    subjects = sorted({p.split("/")[0] for p in available if p.startswith("sub-") and "/" in p})
    eligible, missing = [], Counter()
    for s in subjects:
        ok, miss = eligibility(available, s)
        if ok:
            eligible.append(s)
        missing.update(miss)
    return eligible, missing


def _key(v: Any) -> str:
    if isinstance(v, list):
        return "\\".join(str(x) for x in v)
    return "<missing>" if v is None else str(v)


def _numeric(values: list[Any]) -> dict[str, Any]:
    x = np.array([float(v) for v in values if isinstance(v, int | float)], dtype=float)
    if x.size == 0:
        return {"n": 0}
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    return {
        "n": int(x.size),
        "min": float(q[0]),
        "p5": float(q[1]),
        "q1": float(q[2]),
        "median": float(q[3]),
        "q3": float(q[4]),
        "p95": float(q[5]),
        "max": float(q[6]),
        "distinct": int(np.unique(x).size),
    }


def _field_summary(sidecars: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    n = len(sidecars)
    for f in FIELDS:
        vals = [s.get(f) for s in sidecars]
        missing = sum(v is None for v in vals)
        counts = Counter(_key(v) for v in vals if v is not None)
        entry = {
            "missing": missing,
            "missing_rate": round(missing / n, 4) if n else None,
            "distinct": len(counts),
            "top_values": counts.most_common(TOP_N),
        }
        if f in NUMERIC:
            entry["numeric"] = _numeric(vals)
        out[f] = entry
    hints = Counter(k for s in sidecars for k in s if k.lower().replace("_", "") in BVALUE_HINTS)
    out["_bvalue_like_fields"] = dict(hints)
    return out


def _combo(counter: Counter) -> list[dict[str, Any]]:
    return [{"combination": list(k), "count": c} for k, c in counter.most_common()]


def aggregate(
    pairs: Iterable[tuple[dict[str, Any], dict[str, Any]]], metadata_incomplete: int = 0
) -> dict[str, Any]:
    """``pairs`` = (trace_sidecar, adc_sidecar) per eligible subject. Returns aggregates only."""
    traces, adcs = [], []
    groups: Counter = Counter()
    content: Counter = Counter()
    group_x_content: Counter = Counter()
    content_x_series: Counter = Counter()
    scaling_by_sw: Counter = Counter()
    group_by_sw: Counter = Counter()
    scanner: Counter = Counter()
    thickness: Counter = Counter()
    trace_series: Counter = Counter()
    for t, a in pairs:
        traces.append(t)
        adcs.append(a)
        g, _ = adc_storage_group(a)
        groups[g] += 1
        cs, _ = adc_content_status(a)
        content[cs] += 1
        group_x_content[(g, cs)] += 1
        content_x_series[(cs, _key(a.get("SeriesDescription")), _key(a.get("ImageType")))] += 1
        sw = _key(a.get("SoftwareVersions"))
        scaling_by_sw[
            (
                _key(a.get("SeriesDescription")),
                _key(a.get("PhilipsRescaleSlope")),
                _key(a.get("PhilipsRescaleIntercept")),
                _key(a.get("PhilipsRWVSlope")),
                sw,
            )
        ] += 1
        group_by_sw[(g, sw)] += 1
        scanner[
            (
                _key(t.get("Manufacturer")),
                _key(t.get("ManufacturersModelName")),
                _key(t.get("MagneticFieldStrength")),
                _key(t.get("SoftwareVersions")),
            )
        ] += 1
        thickness[(_key(t.get("SliceThickness")), _key(t.get("SpacingBetweenSlices")))] += 1
        trace_series[_key(t.get("SeriesDescription"))] += 1
    n = len(traces)
    same_sw = sum(
        _key(t.get("SoftwareVersions")) == _key(a.get("SoftwareVersions"))
        for t, a in zip(traces, adcs, strict=True)
    )
    return {
        "n_subjects": n,
        "metadata_incomplete": metadata_incomplete,
        "adc_storage_groups": {k: groups.get(k, 0) for k in ("P", "S", "N", "unknown")},
        "adc_storage_rule": f"{ADC_GROUP_RULE_VERSION} (storage label, not a physical unit)",
        "adc_content_status": {k: content.get(k, 0) for k in ADC_CONTENT_STATUSES},
        "adc_content_rule": ADC_CONTENT_RULE_VERSION,
        "eligible_by_content": sum(content.get(k, 0) for k in ADC_CONTENT_ELIGIBLE),
        "pool_policy_by_content": {
            p: sum(n for k, n in content.items() if ADC_CONTENT_POOL_POLICY[k] == p)
            for p in (POOL_CANDIDATE, POOL_REVIEW, POOL_EXCLUDED)
        },
        "pool_policy_note": "pilot cases are additionally excluded from every split",
        "adc_group_x_content": _combo(group_x_content),
        "adc_content_x_series_x_imagetype": _combo(content_x_series),
        "trace_fields": _field_summary(traces),
        "adc_fields": _field_summary(adcs),
        "trace_series_description": dict(trace_series),
        "adc_scaling_combination_x_software": _combo(scaling_by_sw),
        "adc_group_x_software": _combo(group_by_sw),
        "scanner_combination": _combo(scanner),
        "slice_thickness_x_spacing": _combo(thickness),
        "trace_adc_same_software_version": same_sw,
    }


def parse_bval(text: str) -> dict[str, Any]:
    """Unique b-values of a BIDS ``.bval`` text (rows of whitespace-separated numbers)."""
    rows = [r.split() for r in text.strip().splitlines() if r.strip()]
    vals = [float(v) for r in rows for v in r]
    return {
        "rows": len(rows),
        "n_values": len(vals),
        "unique_b_values": sorted(set(vals)),
    }


__all__ = ["REQUIRED", "aggregate", "eligible_subjects", "parse_bval"]
