"""v2 run outputs, written into the run-control layout (``ai/neuroscan_ai/run_control/outputs.py``)
under position-based case keys. Nothing is overwritten."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuroscan_ai.run_control.outputs import (
    METRIC_KIND,
    aggregate_path,
    case_path,
    run_output_path,
    save_image,
)


def write_json_new(path: Path, obj: Any) -> None:
    if path.exists():
        raise FileExistsError("output exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def save_checkpoint(run_dir: Path, model: torch.nn.Module, meta: dict[str, Any]) -> Path:
    path = run_output_path(run_dir, "checkpoint")
    if path.exists():
        raise FileExistsError("checkpoint exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "meta": meta}, path)
    return path


def write_training_record(run_dir: Path, label: str, obj: Any) -> None:
    write_json_new(run_output_path(run_dir, label), obj)


def write_case_outputs(
    run_dir: Path,
    case_key: str,
    *,
    probability: np.ndarray,
    ungated: np.ndarray,
    brain: np.ndarray,
    gated: np.ndarray,
    affine: np.ndarray,
    metric: dict[str, Any],
) -> None:
    save_image(
        case_path(run_dir, "probability_maps", case_key), probability.astype(np.float32), affine
    )
    save_image(
        case_path(run_dir, "ungated_predictions", case_key), ungated.astype(np.uint8), affine
    )
    save_image(case_path(run_dir, "brain_masks", case_key), brain.astype(np.uint8), affine)
    save_image(case_path(run_dir, "gated_predictions", case_key), gated.astype(np.uint8), affine)
    write_json_new(case_path(run_dir, METRIC_KIND, case_key), metric)


def write_aggregate(run_dir: Path, obj: dict[str, Any]) -> None:
    write_json_new(aggregate_path(run_dir), obj)
