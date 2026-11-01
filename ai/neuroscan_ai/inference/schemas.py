"""Output JSON contracts (version strings, required keys) and privacy / wording checks.

No external schema library: required keys and simple types are checked directly. Every JSON is
checked before it is written: no absolute local path, no input file name (unless explicitly
allowed), no ground-truth or "disease probability" fields, mandatory research disclaimer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import PureWindowsPath
from typing import Any

QUANTIFICATION_SCHEMA = "neuroscan.quantification/1"
METADATA_SCHEMA = "neuroscan.inference_metadata/1"
MANIFEST_SCHEMA = "neuroscan.manifest/1"

DISCLAIMER = {
    "en": (
        "Research prototype output for medical image review support. It does not establish a "
        "diagnosis; final interpretation must be made by a qualified clinician."
    ),
    "ko": (
        "연구 및 의료영상 검토 보조용 프로토타입 결과입니다. 진단을 확정하지 않으며, "
        "최종 판단은 의료진이 해야 합니다."
    ),
}

REQUIRED = {
    QUANTIFICATION_SCHEMA: {
        "schema_version": str,
        "case_id": str,
        "run_id": str,
        "space": str,
        "prediction_threshold": float,
        "lesion_detected": bool,
        "total_lesion_volume_ml": float,
        "component_count": int,
        "components": list,
        "hemisphere": dict,
        "research_score_summary": dict,
        "flags": list,
        "disclaimer": dict,
    },
    METADATA_SCHEMA: {
        "schema_version": str,
        "status": str,
        "case_id": str,
        "run_id": str,
        "inference_started_at": str,
        "model": dict,
        "inputs": dict,
        "validation": dict,
        "outputs": dict,
        "intended_use": dict,
    },
    MANIFEST_SCHEMA: {
        "schema_version": str,
        "case_id": str,
        "run_id": str,
        "coordinate_system": dict,
        "volumes": list,
        "views": dict,
        "render_3d": dict,
        "results": dict,
        "lesion_summary": dict,
        "sharing": dict,
        "disclaimer": dict,
    },
}

# Keys that must never appear (no ground truth, no disease-probability semantics).
FORBIDDEN_KEYS = frozenset(
    {
        "ground_truth",
        "gt",
        "gt_volume_ml",
        "dice",
        "iou",
        "sensitivity",
        "stroke_probability",
        "probability_of_stroke",
        "lesion_probability",
        "confidence_percent",
        "diagnosis",
    }
)
FORBIDDEN_PHRASES = (
    "stroke probability",
    "probability of stroke",
    "뇌졸중 확률",
    "진단 확정",
    "확진",
    "mca",
    "aca",
    "pca",
)
STANDARD_FILE_NAMES = frozenset(
    {
        "dwi.nii.gz",
        "adc.nii.gz",
        "dwi.nii",
        "adc.nii",
        "prediction_mask.nii.gz",
        "prediction_components.nii.gz",
    }
)
_DRIVE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")


def _walk(obj: Any, path: str = "$") -> Iterable[tuple[str, Any, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def _strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def is_absolute_path_text(s: str) -> bool:
    return bool(_DRIVE.search(s)) or s.startswith(("/", "\\\\")) or PureWindowsPath(s).is_absolute()


def problems(obj: dict[str, Any], schema: str, input_names: Iterable[str] = ()) -> list[str]:
    out = []
    if obj.get("schema_version") != schema:
        out.append(f"schema_version != {schema}")
    for key, typ in REQUIRED[schema].items():
        if key not in obj:
            out.append(f"missing key {key}")
        elif typ is float and isinstance(obj[key], int | float) and not isinstance(obj[key], bool):
            continue
        elif not isinstance(obj[key], typ):
            out.append(f"{key} is not {typ.__name__}")
    for path, key, _ in _walk(obj):
        if str(key).lower() in FORBIDDEN_KEYS:
            out.append(f"forbidden key {path}.{key}")
    # an input literally named like a standard output file carries no identifying information
    names = [n for n in input_names if n and n.lower() not in STANDARD_FILE_NAMES]
    for s in _strings(obj):
        if is_absolute_path_text(s):
            out.append("absolute path in output")
        low = s.lower()
        for phrase in FORBIDDEN_PHRASES:
            if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", low):
                out.append(f"forbidden phrase '{phrase}'")
        for n in names:
            if n in s:
                out.append("input file name in output")
    return sorted(set(out))
