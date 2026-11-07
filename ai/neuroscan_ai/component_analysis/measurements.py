"""DWI / ADC statistics inside one component. Values are read as given: no unit conversion, no
clipping, no abs, no offset, negatives kept. ADC units are not decided here."""

from __future__ import annotations

from typing import Any

import numpy as np

_UNCERTAIN_TOKENS = ("unknown", "probable", "suspicious")


def summary(values: np.ndarray) -> dict[str, float | int | None]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def dwi_stats(dwi: np.ndarray, comp: np.ndarray) -> dict[str, Any]:
    vals = np.asarray(dwi)[comp]
    finite = np.isfinite(vals)
    out = summary(vals[finite])
    out["nonfinite_voxel_count"] = int((~finite).sum())
    return out


def adc_stats(
    adc: np.ndarray, comp: np.ndarray, background_code: float | None = None
) -> dict[str, Any]:
    """Statistics over valid voxels (finite and not the background code); negatives stay valid."""
    vals = np.asarray(adc)[comp]
    finite = np.isfinite(vals)
    bg = (
        finite & (vals == background_code) if background_code is not None else np.zeros_like(finite)
    )
    valid = finite & ~bg
    out = summary(vals[valid])
    out.update(
        negative_voxel_count=int((valid & (vals < 0)).sum()),
        background_code=background_code,
        background_code_voxel_count=int(bg.sum()) if background_code is not None else None,
        nonfinite_voxel_count=int((~finite).sum()),
        valid_voxel_count=int(valid.sum()),
    )
    return out


def adc_status(storage: str | None, content: str | None, unit: str | None) -> dict[str, Any]:
    """The three ADC status fields as given, plus whether any of them is not settled."""

    def unsettled(s: str | None) -> bool:
        return s is None or any(t in str(s).lower() for t in _UNCERTAIN_TOKENS)

    return {
        "adc_storage_status": storage,
        "adc_content_status": content,
        "adc_unit_status": unit,
        "adc_status_unsettled": unsettled(storage) or unsettled(content) or unsettled(unit),
    }
