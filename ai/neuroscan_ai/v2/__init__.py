"""NeuroScan v2 (smoke stage): a 2D training runner fully separate from v1.

It does not import or modify any v1 module (training, inference, preprocessing, analysis,
final test). Every run goes through ``neuroscan_ai.run_control`` (manifest gate, required
outputs, post-run manifest, verification). Smoke runs are not performance experiments.
"""

from neuroscan_ai.v2.dataset import BRAIN_MASK_POLICY, PREPROCESS_POLICY, CaseFiles
from neuroscan_ai.v2.model_2d import MODEL_ID
from neuroscan_ai.v2.runner import THRESHOLD_SOURCE, V2TwoDRunner

__all__ = [
    "BRAIN_MASK_POLICY",
    "MODEL_ID",
    "PREPROCESS_POLICY",
    "THRESHOLD_SOURCE",
    "CaseFiles",
    "V2TwoDRunner",
]
