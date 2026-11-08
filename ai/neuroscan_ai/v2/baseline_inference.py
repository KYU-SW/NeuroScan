"""v2 2D baseline validation: model-grid probability -> original grid -> metrics, collapse checks.

Per validation case: slice-wise padded inference (``baseline_slices.infer_volume``, padding
cropped), linear restore to the ORIGINAL grid and threshold 0.5 there (``build_outputs``).
The UNGATED prediction is the official output; the gated prediction (AND original-grid brain mask)
is a comparison only. Metrics use the decoded original-grid GT (``safety_metrics.case_metrics``).

Run-stopping conditions (raise ``TrainingStop`` subclasses): probability outside [0, 1] or
non-finite, non-finite validation loss, any validation case predicted entirely 1, a required
validation result missing. An all-empty validation prediction is recorded, not stopped on (the
consecutive-collapse rule is a separate approval).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from neuroscan_ai.v2.baseline_dataset import BaselineCase, batches
from neuroscan_ai.v2.baseline_outputs import OriginalGridOutputs, build_outputs
from neuroscan_ai.v2.baseline_slices import infer_volume
from neuroscan_ai.v2.baseline_training import (
    THRESHOLD,
    NonFiniteLossError,
    TrainingStop,
    combined_loss,
)
from neuroscan_ai.v2.safety_metrics import case_metrics, summarize, ungated_vs_gated


class ProbabilityRangeError(TrainingStop):
    stage = "validation"


class PredictionCollapseError(TrainingStop):
    stage = "validation"


class MissingValidationResultError(TrainingStop):
    stage = "validation"


def predict_model_grid(
    model: torch.nn.Module, case: BaselineCase, device: torch.device, batch_size: int
) -> np.ndarray:
    prob = infer_volume(model, case.prepared.image, batch_size=batch_size, device=str(device))
    if not np.isfinite(prob).all() or prob.min() < 0.0 or prob.max() > 1.0:
        raise ProbabilityRangeError("model probability non-finite or outside [0, 1]")
    return prob


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module, cases: list[BaselineCase], device: torch.device, batch_size: int
) -> float:
    """Same masked loss as training, over every brain slice of the validation cases (no
    sampling), sample-weighted mean."""
    model.eval()
    order = [(ci, z) for ci, c in enumerate(cases) for z in c.brain_slices]
    tot, n = 0.0, 0
    for b in batches(cases, order, batch_size):
        loss, _, _ = combined_loss(
            model(b["image"].to(device)),
            b["target"].to(device),
            b["valid_region_mask"].to(device),
        )
        k = int(b["image"].shape[0])
        tot += float(loss) * k
        n += k
    if n == 0 or not np.isfinite(tot):
        raise NonFiniteLossError("validation loss non-finite or empty")
    return tot / n


def evaluate(
    model: torch.nn.Module, cases: list[BaselineCase], device: torch.device, batch_size: int
) -> dict[str, Any]:
    """Validation at threshold 0.5 on the original grid. Returns metrics, collapse record and the
    per-case original-grid outputs (kept in memory; written only for the final model)."""
    if not any(c.original_gt.any() for c in cases):
        raise MissingValidationResultError("all validation cases are lesion-free; refused")
    model.eval()
    outputs: dict[str, OriginalGridOutputs] = {}
    ung_m, gat_m = [], []
    for c in cases:
        res = build_outputs(predict_model_grid(model, c, device, batch_size), c.prepared, THRESHOLD)
        outputs[c.case_key] = res
        aff = res.grid.affine
        ung_m.append(case_metrics(res.ungated, c.original_gt, aff, output="ungated"))
        gat_m.append(case_metrics(res.gated, c.original_gt, aff, output="gated"))
    ung, gat = summarize(ung_m), summarize(gat_m)
    total_vox = sum(int(o.ungated.size) for o in outputs.values())
    pos_vox = sum(int(o.ungated.sum()) for o in outputs.values())
    collapse = {
        "all_predictions_empty": pos_vox == 0,
        "cases_predicted_all_one": sum(bool(o.ungated.all()) for o in outputs.values()),
        "no_lesion_fp_volume_ml": ung["no_lesion_fp_volume_ml"],
        "positive_voxel_ratio": round(pos_vox / total_vox, 8) if total_vox else None,
    }
    if collapse["cases_predicted_all_one"]:
        raise PredictionCollapseError("a validation prediction is entirely 1")
    required = (
        "mean_dice_positive",
        "false_positive_voxel_volume_ml",
        "lesion_recall_exploratory",
        "missed_gt_count",
    )
    if any(ung.get(k) is None for k in required) or len(ung_m) != len(cases):
        raise MissingValidationResultError("required validation result missing")
    return {
        "ungated_summary": ung,
        "gated_summary": gat,
        "ungated_vs_gated": ungated_vs_gated(ung, gat),
        "per_case_ungated": ung_m,
        "per_case_gated": gat_m,
        "collapse": collapse,
        "outputs": outputs,
    }
