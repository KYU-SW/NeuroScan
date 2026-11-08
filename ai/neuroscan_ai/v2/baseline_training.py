"""v2 2D baseline training core: fixed hyperparameters, their validation, loss and one epoch.

Every hyperparameter is fixed by approval and checked against ``configs/v2/2d_baseline.yaml``
before the gate. Nothing here adjusts itself: no batch-size reduction, learning-rate change,
AMP / TF32 switch, worker change, architecture / threshold / loss-weight / Dice-smooth change and
no retry after an out-of-memory error. Any such error propagates and the run fails (the
lifecycle writes the failed post-run manifest).
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from neuroscan_ai.v2.baseline_slices import (
    DICE_SMOOTH,
    masked_bce_loss,
    masked_soft_dice_loss,
)
from neuroscan_ai.v2.unet_2d import CHANNELS, EXPECTED_PARAMETERS, IN_CHANNELS, MODEL_ID

FIXED_TRAINING: dict[str, Any] = {
    "max_epochs": 100,
    "early_stopping_patience": 20,
    "batch_size": 32,
    "num_workers": 0,
    "seed": 42,
    "precision": "fp32",
    "amp": False,
    "tf32_matmul": False,
    "tf32_convolution": False,
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "augmentation": "none",
    "optimizer": {"name": "adamw", "learning_rate": 1.0e-3, "weight_decay": 1.0e-4},
    "scheduler": {
        "name": "cosine_annealing",
        "minimum_learning_rate": 1.0e-6,
        "step": "epoch",
        "t_max_epochs": 100,
    },
    "loss": {
        "dice_weight": 1.0,
        "bce_weight": 1.0,
        "dice_smooth": DICE_SMOOTH,
        "region": "valid_region_mask",
        "reduction": "per_sample_mean_then_batch_mean",
    },
    "gradient_clipping": {"norm_type": 2, "max_norm": 1.0},
    "sampling": {
        "candidates": "brain_mask_slices",
        "positive_fraction": 0.5,
        "order": "deterministic_run_seed_and_epoch",
        "keep_negative_slices": True,
    },
    "threshold": {"value": 0.5, "source": "baseline_checkpoint_fixed_0_5", "optimized": False},
    "automatic_adjustment": "forbidden",
}
DEVICES = ("cpu", "cuda", "cuda_if_available")  # execution device; not a tuned hyperparameter
FIXED_DETERMINISM = {
    "precision": "fp32",
    "automatic_mixed_precision": False,
    "tf32_matmul": False,
    "tf32_convolution": False,
    "torch_deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}
THRESHOLD = FIXED_TRAINING["threshold"]["value"]
THRESHOLD_SOURCE = FIXED_TRAINING["threshold"]["source"]


class ConfigValidationError(ValueError):
    """Config differs from the approved architecture or hyperparameters (refused before gate)."""


class TrainingStop(RuntimeError):
    """A run-stopping condition (non-finite values, collapse, range violation, missing result)."""

    stage = "training"


class DeviceUnavailableError(RuntimeError):
    """An explicitly requested CUDA device is not available (never falls back to CPU)."""

    stage = "device"


class NonFiniteLossError(TrainingStop):
    pass


class NonFiniteGradientError(TrainingStop):
    pass


class NonFiniteParameterError(TrainingStop):
    pass


# ---------------------------------------------------------------- validation before the gate
def expected_parameter_count(channels=CHANNELS, in_channels: int = IN_CHANNELS) -> int:
    """Trainable parameters of the approved U-Net, computed without building a model."""

    def block(ci: int, co: int) -> int:  # (conv3x3 + bias, InstanceNorm affine) x 2
        return (9 * ci * co + co) + 2 * co + (9 * co * co + co) + 2 * co

    c = list(channels)
    n = block(in_channels, c[0]) + sum(block(c[i - 1], c[i]) for i in range(1, len(c)))
    for i in range(len(c) - 1, 0, -1):  # transposed conv 2x2 + decoder block on the concat
        n += 4 * c[i] * c[i - 1] + c[i - 1] + block(2 * c[i - 1], c[i - 1])
    return n + c[0] + 1  # 1x1 head


def validate_architecture(cfg: Mapping[str, Any]) -> dict[str, Any]:
    m = cfg.get("model", {})
    errs = []
    if m.get("id") != MODEL_ID:
        errs.append("model.id")
    if list(m.get("channels", [])) != list(CHANNELS):
        errs.append("model.channels")
    if m.get("in_channels") != IN_CHANNELS or m.get("out_channels") != 1:
        errs.append("model.in/out_channels")
    if m.get("pretrained_weights") is not None:
        errs.append("model.pretrained_weights")
    n = expected_parameter_count(m.get("channels", CHANNELS), m.get("in_channels", IN_CHANNELS))
    if not (n == EXPECTED_PARAMETERS == m.get("trainable_parameters")):
        errs.append("model.trainable_parameters")
    if errs:
        raise ConfigValidationError("architecture differs from approval: " + ", ".join(errs))
    return {"model_id": MODEL_ID, "channels": list(CHANNELS), "trainable_parameters": n}


def _diff(prefix: str, actual: Any, expected: Any, out: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping):
            out.append(prefix)
            return
        for k in sorted(set(expected) | set(actual)):
            if k not in expected or k not in actual:
                out.append(f"{prefix}.{k}")
            else:
                _diff(f"{prefix}.{k}", actual[k], expected[k], out)
    elif type(actual) is not type(expected) or actual != expected:
        out.append(prefix)


def validate_hyperparameters(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Every training hyperparameter must equal the approved value (exact types too)."""
    tr = dict(cfg.get("training") or {})
    device = tr.pop("device", None)
    errs: list[str] = []
    _diff("training", tr, FIXED_TRAINING, errs)
    if device not in DEVICES:
        errs.append("training.device")
    _diff("determinism", dict(cfg.get("determinism") or {}), FIXED_DETERMINISM, errs)
    if (cfg.get("run_control") or {}).get("training_output_profile") != "baseline_v1":
        errs.append("run_control.training_output_profile")
    if errs:
        raise ConfigValidationError("hyperparameters differ from approval: " + ", ".join(errs))
    return copy.deepcopy(FIXED_TRAINING)


# ---------------------------------------------------------------- runtime pieces
def apply_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_device(name: str | None) -> None:
    if name not in DEVICES:
        raise ConfigValidationError(f"unknown device '{name}' (allowed: {', '.join(DEVICES)})")


def resolve_device(name: str) -> torch.device:
    """``cuda`` / ``cpu`` are used exactly as requested; an unavailable CUDA device is an error
    (no silent CPU fallback). ``cuda_if_available`` is a development / synthetic convenience only;
    formal reproducibility runs name ``cuda`` or ``cpu`` explicitly."""
    validate_device(name)
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError("cuda was requested but is not available; no CPU fallback")
    return torch.device(name)


def build_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    o = FIXED_TRAINING["optimizer"]
    return torch.optim.AdamW(
        model.parameters(), lr=o["learning_rate"], weight_decay=o["weight_decay"]
    )


def build_scheduler(opt: torch.optim.Optimizer) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    s = FIXED_TRAINING["scheduler"]
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=s["t_max_epochs"], eta_min=s["minimum_learning_rate"]
    )


def combined_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """total = 1.0 * masked Dice + 1.0 * masked BCE (both per-sample over valid voxels)."""
    lw = FIXED_TRAINING["loss"]
    dice = masked_soft_dice_loss(logits, target, valid, smooth=lw["dice_smooth"])
    bce = masked_bce_loss(logits, target, valid)
    return lw["dice_weight"] * dice + lw["bce_weight"] * bce, dice, bce


def train_one_epoch(
    model: torch.nn.Module,
    batches,
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    """One pass over ``batches`` (iterable of dicts from ``baseline_dataset.collate``)."""
    model.train()
    clip = FIXED_TRAINING["gradient_clipping"]
    tot = dice_s = bce_s = 0.0
    norms: list[float] = []
    n = 0
    for b in batches:
        x = b["image"].to(device)
        y = b["target"].to(device)
        v = b["valid_region_mask"].to(device)
        opt.zero_grad(set_to_none=True)
        loss, dl, bl = combined_loss(model(x), y, v)
        if not torch.isfinite(loss):
            raise NonFiniteLossError("non-finite training loss")
        loss.backward()
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise NonFiniteGradientError("non-finite gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip["max_norm"], norm_type=clip["norm_type"]
        )
        opt.step()
        for p in model.parameters():
            if not torch.isfinite(p).all():
                raise NonFiniteParameterError("non-finite parameter after the optimiser step")
        k = int(x.shape[0])
        tot += float(loss.detach()) * k
        dice_s += float(dl.detach()) * k
        bce_s += float(bl.detach()) * k
        norms.append(float(norm))
        n += k
    if n == 0:
        raise TrainingStop("empty training epoch")
    return {
        "train_loss_total": tot / n,
        "train_loss_dice": dice_s / n,
        "train_loss_bce": bce_s / n,
        "gradient_norm": {
            "mean_pre_clip": float(np.mean(norms)),
            "max_pre_clip": float(np.max(norms)),
            "steps": len(norms),
            "clip_max_norm": clip["max_norm"],
        },
        "train_samples": n,
    }
