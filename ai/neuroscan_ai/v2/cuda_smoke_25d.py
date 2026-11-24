"""Synthetic CUDA memory smoke of the 2.5D baseline (batch 32 candidate). No MRI, no cache.

Two independent passes with the same seed and the same synthetic input [32, 6, 264, 264]; each pass
builds the model and runs 3 consecutive real training steps (``baseline_training.train_one_epoch``:
forward, masked Dice + BCE, backward, clipping, AdamW step, non-finite checks). Passing needs: both
passes complete, identical per-step losses / output hash / parameter hash,
``max_memory_reserved`` < 14 GiB and system available RAM >= 4 GiB. A missing reading fails.

A failure is recorded and the run ends: no batch reduction, no batch-16 run, no retry. The result
only supports a separate approval; it never changes the config. An output folder is never reused.
Torch determinism switches and RNG states are restored afterwards.
"""

from __future__ import annotations

import gc
import hashlib
import json
import random
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuroscan_ai.v2 import baseline_training as bt
from neuroscan_ai.v2 import resource_monitor as rm
from neuroscan_ai.v2.baseline_progress import check_public_text, utc_now
from neuroscan_ai.v2.unet_25d import IN_CHANNELS, MODEL_ID, build_model

SCHEMA = "neuroscan-v2-25d-cuda-memory-smoke-v1"
BATCH_SHAPE = (32, IN_CHANNELS, 264, 264)
MIN_STEPS = 3
SMOKE_SEED = 42
MAX_RESERVED_GIB = 14.0
MIN_AVAILABLE_RAM_GIB = 4.0
RESULT_NAME, FAILURE_NAME = "smoke_result.json", "failure_record.json"


@contextmanager
def preserved_torch_state() -> Iterator[None]:
    """Restore determinism switches and RNG states changed by a pass."""
    det = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    cud = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    tf = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    cpu, npy, py = torch.get_rng_state(), np.random.get_state(), random.getstate()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(det, warn_only=warn)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cud
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf
        torch.set_rng_state(cpu)
        np.random.set_state(npy)
        random.setstate(py)


def synthetic_batch(shape: tuple[int, int, int, int], seed: int, step: int) -> dict[str, Any]:
    """Synthetic input, disk-shaped target and full valid region (CPU tensors)."""
    b, c, h, w = shape
    rng = np.random.default_rng([int(seed), int(step)])
    yy, xx = np.mgrid[:h, :w]
    disk = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= (min(h, w) / 8) ** 2
    target = np.broadcast_to(disk[None, None], (b, 1, h, w)).astype(np.float32).copy()
    image = rng.standard_normal((b, c, h, w), dtype=np.float32) + target
    return {
        "image": torch.from_numpy(image),
        "target": torch.from_numpy(target),
        "valid_region_mask": torch.ones((b, 1, h, w), dtype=torch.float32),
    }


def _sha_arrays(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(a.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_pass(
    device: torch.device,
    *,
    shape: tuple[int, int, int, int] = BATCH_SHAPE,
    steps: int = MIN_STEPS,
    seed: int = SMOKE_SEED,
) -> dict[str, Any]:
    """One pass: model, ``steps`` consecutive optimiser steps, readings and hashes."""
    if steps < MIN_STEPS:
        raise ValueError(f"at least {MIN_STEPS} consecutive batches are required")
    bt.apply_determinism(seed)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    rm.reset_cuda_peak(device)
    model = build_model(seed).to(device)
    opt = bt.build_optimizer(model)
    losses, norms, ram = [], [], []
    for i in range(steps):
        batch = synthetic_batch(shape, seed, i)
        m = bt.train_one_epoch(model, [batch], opt, device)
        losses.append(m["train_loss_total"])
        norms.append(m["gradient_norm"]["mean_pre_clip"])
        ram.append(rm.system_available_memory_bytes())
    model.eval()
    with torch.no_grad():
        out = model(synthetic_batch((1, *shape[1:]), seed, 0)["image"].to(device))
    if not torch.isfinite(out).all():
        raise bt.TrainingStop("non-finite model output")
    res = {
        "steps": steps,
        "losses": losses,
        "gradient_norms": norms,
        "output_shape": list(out.shape),
        "output_sha256": _sha_arrays([out]),
        "parameter_sha256": _sha_arrays(model.state_dict().values()),
        "min_available_ram_bytes": None if None in ram else min(ram),
        **rm.cuda_memory(device),
    }
    del model, opt, out
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    return res


def evaluate_passes(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, bool]:
    """Pass conditions (every one must be true)."""
    reserved = [p.get("cuda_max_memory_reserved_bytes") for p in (a, b)]
    ram = [p.get("min_available_ram_bytes") for p in (a, b)]
    finite = all(np.isfinite(p["losses"] + p["gradient_norms"]).all() for p in (a, b))
    return {
        "at_least_three_consecutive_batches": min(a["steps"], b["steps"]) >= MIN_STEPS,
        "all_values_finite": bool(finite),
        "same_losses": a["losses"] == b["losses"],
        "same_output": a["output_sha256"] == b["output_sha256"],
        "same_parameters": a["parameter_sha256"] == b["parameter_sha256"],
        "max_memory_reserved_below_14_gib": None not in reserved
        and max(reserved) < MAX_RESERVED_GIB * rm.GIB,
        "available_ram_at_least_4_gib": None not in ram
        and min(ram) >= MIN_AVAILABLE_RAM_GIB * rm.GIB,
    }


def _write_new(path: Path, obj: Mapping[str, Any]) -> None:
    text = json.dumps(obj, indent=2, allow_nan=False)
    check_public_text(text, ())
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


def run_smoke(
    out_dir: Path,
    *,
    device: torch.device,
    context: Mapping[str, Any],
    pass_fn: Callable[..., dict[str, Any]] = run_pass,
    shape: tuple[int, int, int, int] = BATCH_SHAPE,
) -> dict[str, Any]:
    """Two passes, one record. ``pass_fn`` / ``shape`` are replaced in unit tests only; a
    record of another shape is marked ``formal_shape: false`` and never passes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)  # never overwritten
    base = {
        "schema": SCHEMA,
        "model_id": MODEL_ID,
        "input_shape": list(shape),
        "formal_shape": tuple(shape) == BATCH_SHAPE,
        "device_type": device.type,
        "seed": SMOKE_SEED,
        "precision": "fp32",
        "amp": False,
        "tf32": False,
        "synthetic_input_only": True,
        "retried": False,
        "batch_size_changed": False,
        **dict(context),
        "created_at_utc": utc_now(),
    }
    done: list[dict[str, Any]] = []
    try:
        with preserved_torch_state():
            for _ in range(2):
                done.append(pass_fn(device, shape=shape))
    except Exception as e:  # recorded, never retried (a forced stop writes nothing)
        oom = isinstance(e, torch.cuda.OutOfMemoryError)
        _write_new(
            out_dir / FAILURE_NAME,
            {
                **base,
                "status": "failed",
                "error_type": type(e).__name__,
                "cuda_oom": oom,
                "completed_passes": len(done),
                "next_step": "separate approval required (no automatic batch 16)",
            },
        )
        return {"status": "failed", "error_type": type(e).__name__, "cuda_oom": oom}
    checks = evaluate_passes(*done)
    checks["formal_shape_on_cuda"] = base["formal_shape"] and device.type == "cuda"
    status = "passed" if all(checks.values()) else "failed"
    _write_new(
        out_dir / (RESULT_NAME if status == "passed" else FAILURE_NAME),
        {**base, "status": status, "checks": checks, "passes": done},
    )
    return {"status": status, "checks": checks}
