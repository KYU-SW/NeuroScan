"""Training/inference numeric precision policy (``training.precision`` in the config).

- fp32 (official for Phase 3 and Phase 4): no autocast, no GradScaler.
- bf16: autocast(bfloat16); converged like fp32 in the Phase 3 diagnostics; documented
  alternative, not the official MVP setting.
- fp16: forbidden for this model. Gradient norms reach ~1.5e5 (> fp16 max 65504), GradScaler
  keeps skipping steps until its scale collapses and training silently diverges
  (docs/phase3_2d_baseline.md, troubleshooting).
"""

from __future__ import annotations

import contextlib

import torch

AUTOCAST_DTYPES = {"fp32": None, "bf16": torch.bfloat16}
FORBIDDEN = {"fp16": "fp16 AMP is forbidden: gradient overflow collapses GradScaler (Phase 3)"}


def check_precision(precision: str) -> str:
    if precision in FORBIDDEN:
        raise ValueError(FORBIDDEN[precision])
    if precision not in AUTOCAST_DTYPES:
        raise ValueError(f"unknown precision '{precision}', allowed: {sorted(AUTOCAST_DTYPES)}")
    return precision


def autocast(device: torch.device, precision: str) -> contextlib.AbstractContextManager:
    dtype = AUTOCAST_DTYPES[check_precision(precision)]
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=dtype)
