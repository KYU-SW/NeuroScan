"""Phase 5A independent single-patient inference (research prototype, not for clinical use).

Public API: ``run_independent_inference`` (one DWI + ADC pair -> outputs/<case_id>/<run_id>/),
``load_inference_model`` / ``resolve_device`` (load once, reuse across calls), ``InferenceResult``.
This package has no web, database or authentication dependencies.
"""

from neuroscan_ai.inference.model_loader import LoadedModel, load_inference_model, resolve_device
from neuroscan_ai.inference.service import InferenceResult, run_independent_inference

__all__ = [
    "InferenceResult",
    "LoadedModel",
    "load_inference_model",
    "resolve_device",
    "run_independent_inference",
]
