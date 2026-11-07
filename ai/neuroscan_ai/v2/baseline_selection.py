"""v2 baseline checkpoint selection and the single post-selection threshold search.

Checkpoint selection (validation split only, threshold fixed at 0.5, UNGATED output):
1. primary: patient-level mean Dice over GT-positive validation cases;
2. among epochs within 1e-4 of the best primary value: lower validation loss;
3. then lower ``false_positive_voxel_volume_ml`` (voxel-level overprediction, all validation
   cases; not a count of false-positive lesions);
4. then the earlier epoch.

Threshold search (after the checkpoint is fixed, once, validation only): candidates 0.10-0.90 in
steps of 0.05; primary = mean ungated Dice over positive cases; exact ties -> closest to 0.5.
Recall, missed small lesions, FP volume and no-lesion FP volume are recorded for every candidate
but are not optimised. The chosen value is optimistically biased (selected on the same validation
cases it is reported on). Test and Final Test are never used.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

SELECTION_THRESHOLD = 0.5
SELECTION_OUTPUT = "ungated"
DICE_TIE_TOLERANCE = 1e-4
THRESHOLD_CANDIDATES = tuple(float(Decimal("0.10") + Decimal("0.05") * i) for i in range(17))
THRESHOLD_TIE_TOLERANCE = 1e-12
THRESHOLD_SOURCE = "validation_single_search_after_checkpoint_fixed"
OPTIMISTIC_BIAS_NOTE = (
    "threshold chosen on the same validation cases it is evaluated on; validation metrics at this "
    "threshold are optimistically biased and are not an unbiased performance estimate"
)
_RECORD_KEYS = ("epoch", "mean_dice_positive", "val_loss", "false_positive_voxel_volume_ml")


class SelectionError(ValueError):
    pass


def select_checkpoint(epochs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the epoch per the approved rule. Each record: epoch, mean_dice_positive, val_loss,
    false_positive_voxel_volume_ml, threshold (must be 0.5), output (must be ``ungated``)."""
    if not epochs:
        raise SelectionError("no epoch records")
    for r in epochs:
        if any(r.get(k) is None for k in _RECORD_KEYS):
            raise SelectionError("epoch record is missing a selection field")
        if float(r.get("threshold", -1)) != SELECTION_THRESHOLD:
            raise SelectionError("selection metrics must be computed at threshold 0.5")
        if r.get("output") != SELECTION_OUTPUT:
            raise SelectionError("selection metrics must use the ungated output")
    if len({r["epoch"] for r in epochs}) != len(epochs):
        raise SelectionError("duplicate epoch numbers")
    best = max(float(r["mean_dice_positive"]) for r in epochs)
    tied = [r for r in epochs if best - float(r["mean_dice_positive"]) <= DICE_TIE_TOLERANCE]
    chosen = min(
        tied,
        key=lambda r: (
            float(r["val_loss"]),
            float(r["false_positive_voxel_volume_ml"]),
            int(r["epoch"]),
        ),
    )
    return {
        "selected_epoch": int(chosen["epoch"]),
        "rule": [
            "max mean ungated Dice of positive validation cases at threshold 0.5",
            f"within {DICE_TIE_TOLERANCE}: lower validation loss",
            "then lower false_positive_voxel_volume_ml (voxel-level, not a lesion count)",
            "then earlier epoch",
        ],
        "best_mean_dice_positive": best,
        "tied_epochs": sorted(int(r["epoch"]) for r in tied),
        "selected_record": {k: chosen[k] for k in _RECORD_KEYS},
        "threshold": SELECTION_THRESHOLD,
        "output": SELECTION_OUTPUT,
        "split": "val",
    }


def search_threshold(
    evaluate: Callable[[float], dict[str, Any]], checkpoint_sha256: str, split: str = "val"
) -> dict[str, Any]:
    """Evaluate every candidate once with ``evaluate(threshold)`` (an ungated validation summary
    from ``safety_metrics.summarize``) and choose the threshold. Never touches other splits."""
    if split != "val":
        raise SelectionError("threshold search is allowed on the validation split only")
    if not (isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64):
        raise SelectionError("the checkpoint must be fixed (SHA-256) before the threshold search")
    rows = []
    for t in THRESHOLD_CANDIDATES:
        s = evaluate(t)
        if s.get("output") != "ungated":
            raise SelectionError("threshold search uses the ungated output")
        small = s.get("small_lesion_recall", {})
        rows.append(
            {
                "threshold": t,
                "mean_dice_positive": s.get("mean_dice_positive"),
                "lesion_recall_exploratory": s.get("lesion_recall_exploratory"),
                "missed_gt_count": s.get("missed_gt_count"),
                "missed_small_lesions": {
                    k: v["gt_components"] - v["detected"] for k, v in small.items()
                },
                "false_positive_voxel_volume_ml": s.get("false_positive_voxel_volume_ml"),
                "no_lesion_fp_volume_ml": s.get("no_lesion_fp_volume_ml"),
            }
        )
    scored = [r for r in rows if r["mean_dice_positive"] is not None]
    if not scored:
        raise SelectionError("no positive validation cases to score")
    best = max(r["mean_dice_positive"] for r in scored)
    tied = [r for r in scored if best - r["mean_dice_positive"] <= THRESHOLD_TIE_TOLERANCE]
    chosen = min(tied, key=lambda r: (abs(r["threshold"] - SELECTION_THRESHOLD), r["threshold"]))
    return {
        "threshold_source": THRESHOLD_SOURCE,
        "candidates": list(THRESHOLD_CANDIDATES),
        "primary": "mean ungated Dice of positive validation cases",
        "tie_break": "closest to 0.5",
        "selected_threshold": chosen["threshold"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_fixed_before_search": True,
        "searches_performed": 1,
        "split": "val",
        "held_out_splits_used": False,  # Test / Final Test never touched
        "optimistic_bias": OPTIMISTIC_BIAS_NOTE,
        "fp_threshold_optimised": False,
        "rows": rows,
    }


def write_once(path: Path, record: dict[str, Any]) -> Path:
    """Write a selection / threshold record; refuses to overwrite (one search per checkpoint)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("record exists; the threshold search is performed only once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    return path
