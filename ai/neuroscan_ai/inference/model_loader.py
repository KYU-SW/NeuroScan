"""Verified model loading for independent inference.

Differences from ``training.trainer.load_model_from_checkpoint`` (kept unchanged for Phase 3/4):
- ``torch.load(weights_only=True)``: a checkpoint path given by a user must not run pickled code.
- the checkpoint's ``info.config_hash`` must equal the hash of the declared training config and
  the expected hash of the inference config; context width, input channels and preprocessing
  version must match the 2.5D contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neuroscan_ai.inference.errors import Code, InferenceError, Issue
from neuroscan_ai.inference.validation import sha256_file
from neuroscan_ai.training.config import config_hash, in_channels, load_config
from neuroscan_ai.training.model import build_model


@dataclass
class LoadedModel:
    model: torch.nn.Module
    training_cfg: dict[str, Any]
    info: dict[str, Any]
    device: torch.device
    checkpoint_path: Path
    config_path: Path
    checkpoint_sha256: str
    config_sha256: str
    config_hash: str


def resolve_device(requested: str) -> tuple[torch.device, list[Issue]]:
    """D9: auto -> CUDA if available, else CPU with W_CPU_FALLBACK; explicit cuda must exist."""
    if requested not in ("auto", "cuda", "cpu"):
        raise InferenceError(Code.INVALID_ARGUMENT, "device must be auto, cuda or cpu.")
    if requested == "cpu":
        return torch.device("cpu"), []
    if torch.cuda.is_available():
        return torch.device("cuda"), []
    if requested == "cuda":
        raise InferenceError(Code.DEVICE, "CUDA was requested but no GPU is available.")
    return torch.device("cpu"), [
        Issue(Code.W_CPU_FALLBACK, "No GPU available; running on CPU (slower).")
    ]


def load_inference_model(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: torch.device,
    expected: dict[str, Any],
) -> LoadedModel:
    """``expected``: the ``model`` block of configs/inference_v1.yaml."""
    checkpoint_path, config_path = Path(checkpoint_path), Path(config_path)
    if not checkpoint_path.is_file():
        raise InferenceError(Code.CHECKPOINT, "Model file could not be loaded.", "not found")
    if not config_path.is_file():
        raise InferenceError(Code.CONFIG_MISMATCH, "Model configuration file is missing.")
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        raise InferenceError(
            Code.CONFIG_MISMATCH, "Model configuration could not be read.", type(exc).__name__
        ) from None
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise InferenceError(
            Code.CHECKPOINT, "Model file could not be loaded.", type(exc).__name__
        ) from None
    if not isinstance(ckpt, dict) or "model" not in ckpt or "info" not in ckpt:
        raise InferenceError(Code.CHECKPOINT, "Model file has an unexpected structure.")
    info = dict(ckpt["info"])

    h = config_hash(cfg)
    checks = {
        "checkpoint config_hash == hash(training config)": info.get("config_hash") == h,
        "hash(training config) == expected_config_hash": h == expected["expected_config_hash"],
        "context_slices": int(cfg["data"]["context_slices"]) == expected["expected_context_slices"],
        "in_channels(config)": in_channels(cfg) == expected["expected_in_channels"],
        "in_channels(checkpoint)": info.get("in_channels") == expected["expected_in_channels"],
        "preprocessing_version": info.get("preprocessing_version")
        == expected["expected_preprocessing_version"]
        == cfg["data"]["preprocessing_version"],
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise InferenceError(
            Code.CONFIG_MISMATCH, "Model and configuration do not match.", "; ".join(failed)
        )
    try:
        model = build_model(cfg)
        model.load_state_dict(ckpt["model"], strict=True)
    except Exception as exc:
        raise InferenceError(
            Code.CHECKPOINT,
            "Model weights do not fit the declared architecture.",
            type(exc).__name__,
        ) from None
    model.to(device).eval()
    return LoadedModel(
        model=model,
        training_cfg=cfg,
        info=info,
        device=device,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        config_sha256=sha256_file(config_path),
        config_hash=h,
    )
