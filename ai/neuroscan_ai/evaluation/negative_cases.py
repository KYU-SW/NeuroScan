"""Evaluation of cases whose ground truth is empty (no lesion).

Overlap metrics (Dice, IoU, sensitivity, precision) are undefined when GT is empty, so they are
not computed for these cases. Instead the prediction itself is reported:
- prediction empty     -> "correct_negative"
- prediction non-empty -> "false_positive_case", with the false-positive volume.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import numpy as np

from neuroscan_ai.analysis.components import count_components
from neuroscan_ai.preprocessing.mask import lesion_volume_ml


class NegativeOutcome(StrEnum):
    CORRECT_NEGATIVE = "correct_negative"
    FALSE_POSITIVE = "false_positive_case"


def evaluate_negative_case(prediction: np.ndarray, spacing: Sequence[float]) -> dict[str, Any]:
    pred = np.asarray(prediction) > 0
    voxels = int(pred.sum())
    volume = lesion_volume_ml(pred, spacing)
    outcome = NegativeOutcome.CORRECT_NEGATIVE if voxels == 0 else NegativeOutcome.FALSE_POSITIVE
    return {
        "negative_outcome": outcome.value,
        "predicted_lesion_voxels": voxels,
        "predicted_lesion_volume_ml": volume,
        "any_false_positive": voxels > 0,
        "false_positive_volume_ml": volume,  # every predicted voxel is a false positive here
        "false_positive_component_count": count_components(pred),
    }
