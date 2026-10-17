"""Patient-level binary segmentation metrics.

Inputs are binary arrays on the same grid. Evaluation is meant to run on the original patient
grid (see reconstruction.restore); the function itself only needs arrays + spacing.

Definitions (TP/FP/FN are voxel counts):
  dice        = 2TP / (2TP + FP + FN)
  iou         = TP / (TP + FP + FN)
  sensitivity = TP / (TP + FN)
  precision   = TP / (TP + FP)        -> None (undefined) when the prediction is empty
  volume_error_ml          = pred_volume - gt_volume        (signed; >0 over-segmentation)
  absolute_volume_error_ml = |volume_error_ml|
  relative_volume_error_percent = 100 * volume_error_ml / gt_volume
For an empty GT no overlap metric is defined: those fields are None and the negative-case
fields from ``negative_cases`` are filled instead. None is never converted to 0 or 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from neuroscan_ai.analysis.components import count_components
from neuroscan_ai.evaluation.negative_cases import evaluate_negative_case
from neuroscan_ai.preprocessing.mask import lesion_volume_ml

# Default probability -> binary rule (prob >= threshold). Model phases override it via config.
DEFAULT_PROBABILITY_THRESHOLD = 0.5

NEGATIVE_FIELDS = (
    "negative_outcome",
    "predicted_lesion_voxels",
    "predicted_lesion_volume_ml",
    "any_false_positive",
    "false_positive_volume_ml",
    "false_positive_component_count",
)


def binarize_prediction(
    probability: np.ndarray, threshold: float = DEFAULT_PROBABILITY_THRESHOLD
) -> np.ndarray:
    return (np.asarray(probability) >= threshold).astype(np.uint8)


def _as_binary(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if not set(np.unique(arr).tolist()) <= {0, 1}:
        raise ValueError(f"{name} must be binary {{0, 1}}; binarize probabilities first")
    return arr.astype(bool)


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def evaluate_case(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spacing: Sequence[float],
    case_id: str | None = None,
) -> dict[str, Any]:
    pred = _as_binary(prediction, "prediction")
    gt = _as_binary(ground_truth, "ground_truth")
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape} vs ground truth {gt.shape}")
    if len(spacing) < 3 or any(s <= 0 for s in spacing[:3]):
        raise ValueError(f"invalid spacing {spacing}")

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    gt_vol = lesion_volume_ml(gt, spacing)
    pred_vol = lesion_volume_ml(pred, spacing)
    has_lesion = bool(gt.any())

    row: dict[str, Any] = {
        "case_id": case_id,
        "has_lesion": has_lesion,
        "gt_voxels": int(gt.sum()),
        "prediction_voxels": int(pred.sum()),
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "dice": None,
        "iou": None,
        "sensitivity": None,
        "precision": None,
        "gt_volume_ml": gt_vol,
        "prediction_volume_ml": pred_vol,
        "volume_error_ml": pred_vol - gt_vol,
        "absolute_volume_error_ml": abs(pred_vol - gt_vol),
        "relative_volume_error_percent": None,
        "component_count_gt": count_components(gt),
        "component_count_prediction": count_components(pred),
        **dict.fromkeys(NEGATIVE_FIELDS),
    }
    if has_lesion:
        row["dice"] = 2 * tp / (2 * tp + fp + fn)
        row["iou"] = tp / (tp + fp + fn)
        row["sensitivity"] = tp / (tp + fn)
        row["precision"] = _ratio(tp, tp + fp)
        row["relative_volume_error_percent"] = 100.0 * (pred_vol - gt_vol) / gt_vol
    else:
        row.update(evaluate_negative_case(pred, spacing))
    return row
