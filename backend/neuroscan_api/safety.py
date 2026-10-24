"""Response safety checks: no absolute paths, file names, header text, tracebacks or clinical
probability / diagnosis wording in anything the API returns.

The rules mirror ``neuroscan_ai.inference.schemas`` (Phase 5A). They are restated here because
importing that package would import torch into the API process (the inference package
``__init__`` loads the model loader), which the API must never do.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import PureWindowsPath
from typing import Any

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
NO_LESION_TEXT = (
    "The model did not mark any lesion region. This does not mean that no lesion is present."
)

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
        "traceback",
        "checkpoint_path",
        "file_name",
        "filename",
        "rel_path",
        "run_dir",
    }
)
FORBIDDEN_PHRASES = (
    "stroke probability",
    "probability of stroke",
    "뇌졸중 확률",
    "진단 확정",
    "확진",
    "traceback (most recent call last)",
    "mca",
    "aca",
    "pca",
)
_DRIVE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")


def is_absolute_path_text(s: str) -> bool:
    return (
        bool(_DRIVE.search(s))
        or s.startswith(("\\\\",))
        or PureWindowsPath(s).is_absolute()
        or (s.startswith("/") and not s.startswith("/api/"))
    )


def _walk(obj: Any, path: str = "$") -> Iterable[tuple[str, str | None, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, str(k), v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, None, obj


def violations(obj: Any, extra_strings: Iterable[str] = ()) -> list[str]:
    """List of rule violations in a JSON-like object (empty list = safe)."""
    out: set[str] = set()
    extras = [s for s in extra_strings if s]
    for path, key, value in _walk(obj):
        if key is not None and key.lower() in FORBIDDEN_KEYS:
            out.add(f"forbidden key {path}.{key}")
        if isinstance(value, str):
            if is_absolute_path_text(value):
                out.add(f"absolute path at {path}")
            low = value.lower()
            for phrase in FORBIDDEN_PHRASES:
                if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", low):
                    out.add(f"forbidden phrase '{phrase}' at {path}")
            for s in extras:
                if s in value:
                    out.add(f"sensitive string at {path}")
    return sorted(out)
