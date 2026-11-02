"""Fixed rules, thresholds and output layout of the metrics-only failure analysis (v1)."""

from __future__ import annotations

MODE_METRICS_ONLY = "metrics-only"
MODELS = ("2d", "25d")

# Group order = alias priority. A case selected by several groups keeps its first alias.
GROUP_ORDER = ("F", "Z", "I", "W", "N", "G")
GROUP_RULES = {
    "F": "positive case with 2D or 2.5D Dice < 0.50",
    "Z": "positive case with 2.5D Dice == 0",
    "I": "top 3 Dice difference (2.5D - 2D)",
    "W": "bottom 3 Dice difference (2.5D - 2D)",
    "N": "negative case (empty ground truth)",
    "G": "both models Dice >= 0.85: one thin-slice and one thick-slice representative",
}
FAIL_DICE = 0.50
GOOD_DICE = 0.85
N_EXTREME = 3
TIE_TOLERANCE = 1e-3  # same as the Final Test paired comparison (improved / worse / tied)

# Qualitative-review thresholds (never used for metrics, post-processing, removal or diagnosis).
SMALL_GT_COMPONENT_ML = 0.1
DISTANT_FP_MM = 10.0
LOCATION_MISMATCH_OVERLAP = 0.10

# Flag values
FLAGGED = "flagged"
NOT_FLAGGED = "not_flagged"
NOT_APPLICABLE = "not_applicable"
PENDING = "pending_image_review"
REVIEW_CANDIDATE = "review_candidate"

TAXONOMY = {
    "T1": "under-segmentation: sensitivity < 0.5 and FN voxels > FP voxels",
    "T2": "over-segmentation: precision < 0.5 and FP voxels > FN voxels",
    "T3": "location mismatch candidate: prediction voxels > 0 and TP = 0",
    "T4": f"missed small GT component (< {SMALL_GT_COMPONENT_ML} mL): needs component maps",
    "T5": f"distant FP component (> {DISTANT_FP_MM} mm from nearest GT): needs component maps",
    "T6": "boundary mismatch: needs spatial FP/FN distribution",
    "T7": "split / merge difference: component counts recorded, matching type needs images",
    "T8": "inter-model context difference candidate: thick slice and 2.5D worse than 2D",
    "T9": "image signal vs GT inconsistency candidate: needs DWI / ADC",
}
PENDING_CODES = ("T4", "T5", "T6", "T7", "T9")

# Inputs read from the completed Final Test folder (read-only)
REQUIRED_FILES = (
    "COMPLETED",
    "patient_metrics_2d.csv",
    "patient_metrics_25d.csv",
    "paired_metrics.csv",
    "aggregate_metrics.json",
    "subgroup_metrics.json",
    "negative_case_summary.json",
    "qa_manifest.json",
    "execution_audit.json",
)
FORBIDDEN_FILES = ("FAILED",)

METRIC_FIELDS = (
    "dice",
    "iou",
    "sensitivity",
    "precision",
    "tp_voxels",
    "fp_voxels",
    "fn_voxels",
    "gt_voxels",
    "prediction_voxels",
    "gt_volume_ml",
    "prediction_volume_ml",
    "volume_error_ml",
    "absolute_volume_error_ml",
    "relative_volume_error_percent",
    "component_count_gt",
    "component_count_prediction",
)
NEGATIVE_FIELDS = (
    "negative_outcome",
    "false_positive_volume_ml",
    "false_positive_component_count",
)
META_FIELDS = ("volume_group", "spacing_group", "adc_unit_group", "obliquity_group", "center")
