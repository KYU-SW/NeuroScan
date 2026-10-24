"""Per-case intensity normalisation applied in the dataset transform (processed files untouched).

All methods are *per case*: statistics come from that case's own brain voxels only, so no
statistic is shared across patients or splits, and a per-case affine rescaling of the stored
values (e.g. ADC stored in mm^2/s vs 1e-6 mm^2/s) cancels out exactly. Background (outside the
brain mask) is set to 0 after normalisation.

Methods
- zscore:            (x - mean) / std over brain voxels
- clip_zscore:       clip to brain [p_low, p_high] percentiles, then z-score
- robust:            clip to [p_low, p_high], then (x - median) / IQR
- log_clip_zscore:   log(x) on positive brain voxels, then clip_zscore (skew check for ADC)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EPS = 1e-8


@dataclass(frozen=True)
class NormalizationSpec:
    method: str
    clip_percentiles: tuple[float, float] = (0.5, 99.5)


# A skull-stripped brain fills roughly 15-27% of an ISLES'22 volume (train split analysis);
# anything outside this window means the background rule failed and must not pass silently.
BRAIN_FRACTION_RANGE = (0.05, 0.60)


def background_value(volume: np.ndarray) -> float:
    """Constant background of a skull-stripped volume: the most common of the 8 corner values.

    Usually 0, but NIfTI scaling can store it as e.g. -0.000992 (sub-strokecase0046 DWI).
    """
    corners = volume[np.ix_(*[[0, s - 1] for s in volume.shape[:3]])].ravel()
    values, counts = np.unique(corners, return_counts=True)
    return float(values[np.argmax(counts)])


def brain_mask(dwi: np.ndarray) -> np.ndarray:
    """Brain voxels = DWI differs from its constant background value.

    ADC is not used: it can be exactly 0 inside the brain (fit failures, up to ~3% of brain
    voxels in the train split).
    """
    bg = background_value(dwi)
    mask = ~np.isclose(dwi, bg, rtol=0.0, atol=1e-6 * max(1.0, abs(bg)))
    frac = float(mask.mean())
    lo, hi = BRAIN_FRACTION_RANGE
    if not lo <= frac <= hi:
        raise ValueError(f"brain mask covers {frac:.1%} of the volume (expected {lo:.0%}-{hi:.0%})")
    return mask


def _clip(
    x: np.ndarray, values: np.ndarray, pct: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.percentile(values, pct)
    return np.clip(x, lo, hi), np.clip(values, lo, hi)


def _zscore(x, values, spec):
    return (x - values.mean()) / max(values.std(), EPS)


def _clip_zscore(x, values, spec):
    x, values = _clip(x, values, spec.clip_percentiles)
    return (x - values.mean()) / max(values.std(), EPS)


def _robust(x, values, spec):
    x, values = _clip(x, values, spec.clip_percentiles)
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return (x - med) / max(q3 - q1, EPS)


def _log_clip_zscore(x, values, spec):
    floor = max(np.percentile(values[values > 0], 0.5), EPS) if (values > 0).any() else EPS
    return _clip_zscore(np.log(np.maximum(x, floor)), np.log(np.maximum(values, floor)), spec)


METHODS: dict[str, Callable] = {
    "zscore": _zscore,
    "clip_zscore": _clip_zscore,
    "robust": _robust,
    "log_clip_zscore": _log_clip_zscore,
}


def normalize(volume: np.ndarray, mask: np.ndarray, spec: NormalizationSpec) -> np.ndarray:
    """Return a float32 normalised copy; voxels outside ``mask`` are 0."""
    if spec.method not in METHODS:
        raise ValueError(f"unknown normalisation '{spec.method}', choose from {sorted(METHODS)}")
    x = volume.astype(np.float64, copy=False)
    values = x[mask]
    if values.size == 0:
        raise ValueError("empty brain mask")
    out = METHODS[spec.method](x, values, spec)
    out = np.where(mask, out, 0.0).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("normalisation produced non-finite values")
    return out
