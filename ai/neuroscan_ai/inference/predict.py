"""2.5D prediction for one prepared case, restored onto the original DWI grid.

Reuses the Phase 3/4 official order unchanged (``training.inference.predict_case``):
  logits -> sigmoid -> brain-mask gating -> threshold -> binary slices -> 3D reconstruction
followed by ``reconstruction.restore_to_original`` (binary: linear + >= 0.5).
``loss_fn`` is always None: there is no ground truth and nothing is evaluated.

CUDA out-of-memory: retry with smaller inference batches (results do not depend on the batch
size, tests/test_phase4.py), then, only for device=auto, once on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from neuroscan_ai.inference.errors import Code, InferenceError, Issue
from neuroscan_ai.inference.model_loader import LoadedModel
from neuroscan_ai.inference.preprocess import PreparedCase
from neuroscan_ai.reconstruction.restore import restore_to_original
from neuroscan_ai.reconstruction.volume import PredictionKind, PredictionVolume
from neuroscan_ai.training.inference import predict_case

RETRY_BATCH_SIZES = (16, 4)


@dataclass
class Prediction:
    probability_processed: PredictionVolume  # gated, processed grid (never written to disk, D6)
    binary_processed: PredictionVolume
    binary_original: PredictionVolume  # the official output mask
    device: str
    batch_size: int
    attempts: list[dict] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)


def _run(loaded: LoadedModel, prepared: PreparedCase, device: torch.device, batch_size: int):
    cfg = loaded.training_cfg
    return predict_case(
        loaded.model,
        prepared.case,
        int(cfg["data"]["context_slices"]),
        device,
        size_multiple=cfg["inference"]["size_multiple"],
        batch_size=batch_size,
        precision=cfg["training"]["precision"],
        loss_fn=None,
        threshold=cfg["inference"]["threshold"],
        brain_mask_gating=cfg["inference"]["brain_mask_gating"],
    )


def predict(loaded: LoadedModel, prepared: PreparedCase, allow_cpu_fallback: bool) -> Prediction:
    cfg = loaded.training_cfg
    if not cfg["inference"]["brain_mask_gating"]:
        raise InferenceError(Code.CONFIG_MISMATCH, "Brain-mask gating must be enabled.")
    device = loaded.device
    sizes = [cfg["inference"]["batch_size"], *RETRY_BATCH_SIZES]
    attempts: list[dict] = []
    warnings: list[Issue] = []
    result = None
    for bs in sizes:
        try:
            result = _run(loaded, prepared, device, bs)
            attempts.append({"device": device.type, "batch_size": bs, "ok": True})
            break
        except torch.cuda.OutOfMemoryError:
            attempts.append({"device": device.type, "batch_size": bs, "ok": False, "error": "oom"})
            torch.cuda.empty_cache()
    if result is None and device.type == "cuda" and allow_cpu_fallback:
        warnings.append(Issue(Code.W_CPU_FALLBACK, "GPU memory was insufficient; ran on CPU."))
        device = torch.device("cpu")
        loaded.model.to(device)
        bs = sizes[0]
        result = _run(loaded, prepared, device, bs)
        attempts.append({"device": "cpu", "batch_size": bs, "ok": True})
    if result is None:
        raise InferenceError(Code.INFERENCE_OOM, "GPU memory was insufficient for inference.")
    if len(attempts) > 1:
        warnings.append(Issue(Code.W_OOM_RETRY, "Inference was retried after a memory error."))
    probability, binary, _ = result
    restored = restore_to_original(binary, prepared.original)
    _check_restored(restored, prepared)
    return Prediction(probability, binary, restored, device.type, bs, attempts, warnings)


def _check_restored(vol: PredictionVolume, prepared: PreparedCase) -> None:
    ok = (
        vol.kind is PredictionKind.BINARY
        and vol.data.shape == prepared.original.shape
        and np.array_equal(vol.geometry.affine, prepared.original.affine)
        and vol.data.dtype == np.uint8
        and set(np.unique(vol.data).tolist()) <= {0, 1}
    )
    if not ok:
        raise InferenceError(
            Code.OUTPUT_VERIFY, "Prediction could not be restored to the input image grid."
        )
