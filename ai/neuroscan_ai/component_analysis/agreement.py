"""2D / 2.5D agreement for one final component (only when both model outputs are given).

Comparison region R = the final component plus every 26-connected component of the 2D or 2.5D
mask that overlaps it, so each model's full extent around the lesion is compared; Dice / IoU are
computed between (2D within R) and (2.5D within R).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from neuroscan_ai.component_analysis.extraction import CONNECTIVITY_26

LOW_OVERLAP_DICE = 0.5  # provisional (design candidate, not validated)


class ModelMasks:
    def __init__(
        self,
        pred_2d: np.ndarray | None,
        pred_25d: np.ndarray | None,
        prob_2d: np.ndarray | None,
        prob_25d: np.ndarray | None,
    ):
        self.pred_2d, self.pred_25d = pred_2d, pred_25d
        self.prob_2d, self.prob_25d = prob_2d, prob_25d
        self.available = pred_2d is not None and pred_25d is not None
        self.lab_2d = (
            ndimage.label(pred_2d, structure=CONNECTIVITY_26)[0] if pred_2d is not None else None
        )
        self.lab_25d = (
            ndimage.label(pred_25d, structure=CONNECTIVITY_26)[0] if pred_25d is not None else None
        )


def _touching(labels: np.ndarray, comp: np.ndarray) -> np.ndarray:
    ids = np.unique(labels[comp])
    ids = ids[ids > 0]
    return np.isin(labels, ids) if ids.size else np.zeros(labels.shape, dtype=bool)


def _conf(prob: np.ndarray | None, comp: np.ndarray) -> tuple[float | None, float | None]:
    if prob is None:
        return None, None
    v = np.asarray(prob, dtype=np.float64)[comp]
    return float(v.mean()), float(v.max())


def component_agreement(comp: np.ndarray, models: ModelMasks) -> dict[str, Any]:
    c2m, c2x = _conf(models.prob_2d, comp)
    c25m, c25x = _conf(models.prob_25d, comp)
    out: dict[str, Any] = {
        "available": models.available,
        "conf_2d_mean": c2m,
        "conf_2d_max": c2x,
        "conf_25d_mean": c25m,
        "conf_25d_max": c25x,
    }
    if not models.available:
        out.update(
            status="neither_or_unavailable",
            overlap_2d_voxels=None,
            overlap_25d_voxels=None,
            overlap_2d_ratio=None,
            overlap_25d_ratio=None,
            dice_2d_25d=None,
            iou_2d_25d=None,
        )
        return out
    n = int(comp.sum())
    o2 = int((comp & models.pred_2d).sum())
    o25 = int((comp & models.pred_25d).sum())
    region = comp | _touching(models.lab_2d, comp) | _touching(models.lab_25d, comp)
    a, b = models.pred_2d & region, models.pred_25d & region
    inter, sa, sb = int((a & b).sum()), int(a.sum()), int(b.sum())
    union = int((a | b).sum())
    status = (
        "both"
        if o2 and o25
        else "2d_only"
        if o2
        else "25d_only"
        if o25
        else "neither_or_unavailable"
    )
    out.update(
        status=status,
        overlap_2d_voxels=o2,
        overlap_25d_voxels=o25,
        overlap_2d_ratio=round(o2 / n, 6),
        overlap_25d_ratio=round(o25 / n, 6),
        dice_2d_25d=round(2 * inter / (sa + sb), 6) if sa + sb else None,
        iou_2d_25d=round(inter / union, 6) if union else None,
        comparison_region_voxels=int(region.sum()),
    )
    return out
