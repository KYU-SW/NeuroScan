"""AI prediction vs ground truth on one grid: the TP / FP / FN difference labelmap.

Labels (uint8): 0 = background, 1 = TP (both), 2 = FP (prediction only), 3 = FN (GT only).
The labelmap is a display product for research review; the metrics stay those of
``evaluation.segmentation.evaluate_case`` and must agree with it voxel for voxel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

BACKGROUND, TP, FP, FN = 0, 1, 2, 3
LEGEND = (
    {"label": TP, "key": "tp", "color": "#009E73", "ko": "일치 영역 (TP)", "en": "Overlap (TP)"},
    {
        "label": FP,
        "key": "fp",
        "color": "#E69F00",
        "ko": "AI만 표시한 영역 (FP)",
        "en": "AI only (FP)",
    },
    {
        "label": FN,
        "key": "fn",
        "color": "#56B4E9",
        "ko": "GT에 있으나 AI가 놓친 영역 (FN)",
        "en": "Ground truth only (FN)",
    },
)
GT_ONLY_COLOR = "#CC79A7"


def _binary(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a)
    if not set(np.unique(a).tolist()) <= {0, 1}:
        raise ValueError(f"{name} must be binary 0/1")
    return a.astype(bool)


def difference_labelmap(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    pred = _binary(prediction, "prediction")
    gt = _binary(ground_truth, "ground truth")
    if pred.shape != gt.shape:
        raise ValueError("prediction and ground truth grids differ")
    out = np.zeros(pred.shape, dtype=np.uint8)
    out[pred & gt] = TP
    out[pred & ~gt] = FP
    out[~pred & gt] = FN
    return out


def check_difference(
    diff: np.ndarray, prediction: np.ndarray, ground_truth: np.ndarray, metrics: dict[str, Any]
) -> list[str]:
    """Consistency of the labelmap with both masks and with ``evaluate_case`` (empty = OK)."""
    if diff.dtype != np.uint8 or diff.shape != np.shape(prediction):
        return ["difference labelmap dtype/shape"]
    counts = np.bincount(diff.ravel(), minlength=4)
    if counts.size > 4:
        return ["difference labelmap has values above 3"]
    tp, fp, fn = int(counts[TP]), int(counts[FP]), int(counts[FN])
    pred_n = int(np.count_nonzero(prediction))
    gt_n = int(np.count_nonzero(ground_truth))
    out = []
    if (tp, fp, fn) != (metrics["tp_voxels"], metrics["fp_voxels"], metrics["fn_voxels"]):
        out.append("TP/FP/FN voxel counts differ from evaluate_case")
    if tp + fp != pred_n or pred_n != metrics["prediction_voxels"]:
        out.append("TP + FP != prediction foreground")
    if tp + fn != gt_n or gt_n != metrics["gt_voxels"]:
        out.append("TP + FN != ground-truth foreground")
    return out
