"""Model -> per-slice probabilities -> Phase 2 engines (reconstruct, restore, evaluate).

Official inference order (2D and 2.5D alike):
  logits -> sigmoid -> brain-mask gating -> threshold -> binary slices
  -> 3D reconstruction -> original-space restoration -> evaluation

No metric, reconstruction or resampling logic lives here; it only adapts model output to the
Phase 2 interfaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from neuroscan_ai.evaluation.segmentation import binarize_prediction, evaluate_case
from neuroscan_ai.preprocessing.mask import binarize
from neuroscan_ai.reconstruction.restore import restore_to_original
from neuroscan_ai.reconstruction.volume import (
    PredictionKind,
    PredictionVolume,
    VolumeGeometry,
    reconstruct_volume,
)
from neuroscan_ai.training.brain_gating import BrainMaskError, gate_probability
from neuroscan_ai.training.dataset import CaseVolumes, pad_or_crop, stack_context
from neuroscan_ai.training.precision import autocast


def _padded_size(shape: tuple[int, int], multiple: int) -> tuple[int, int]:
    return tuple(int(np.ceil(s / multiple) * multiple) for s in shape)


@torch.no_grad()
def predict_case(
    model: torch.nn.Module,
    case: CaseVolumes,
    context: int,
    device: torch.device,
    size_multiple: int = 16,
    batch_size: int = 64,
    precision: str = "fp32",
    loss_fn: torch.nn.Module | None = None,
    threshold: float = 0.5,
    brain_mask_gating: bool = True,
) -> tuple[PredictionVolume, PredictionVolume, float | None]:
    """(gated probability volume, binary volume) on the processed grid, + mean slice loss.

    The loss (when ``loss_fn`` is given) is the training objective on the raw logits, i.e.
    before gating.
    """
    model.eval()
    if brain_mask_gating and case.brain is None:
        raise BrainMaskError(f"{case.case_id}: gating requested but no brain mask loaded")
    xy = case.label.shape[:2]
    padded = _padded_size(xy, size_multiple)
    slices: list[tuple[int, np.ndarray]] = []
    binary_slices: list[tuple[int, np.ndarray]] = []
    loss_sum, n = 0.0, 0
    for start in range(0, case.n_slices, batch_size):
        zs = list(range(start, min(start + batch_size, case.n_slices)))
        x = np.stack([pad_or_crop(stack_context(case.image, z, context), padded) for z in zs])
        x = torch.from_numpy(x).to(device)
        with autocast(device, precision):
            logits_t = model(x).float()
        # the single sigmoid of the pipeline (numerically stable; no exp overflow)
        prob_t = torch.sigmoid(logits_t)
        logits = pad_or_crop(logits_t.cpu().numpy(), xy)  # back to the processed grid
        probs = pad_or_crop(prob_t.cpu().numpy(), xy)
        if loss_fn is not None:
            y = torch.from_numpy(
                case.label[None, :, :, zs].transpose(3, 0, 1, 2).astype(np.float32)
            )
            loss_sum += float(loss_fn(torch.from_numpy(logits), y)) * len(zs)
            n += len(zs)
        probs = probs[:, 0].astype(np.float32)  # [B, H, W]
        if brain_mask_gating:
            probs = gate_probability(probs, case.brain[:, :, zs].transpose(2, 0, 1))
        binary = binarize_prediction(probs, threshold)
        slices.extend(zip(zs, probs, strict=True))
        binary_slices.extend(zip(zs, binary, strict=True))
    geom = VolumeGeometry.from_meta(case.meta, "processed")
    probability = reconstruct_volume(slices, geom, PredictionKind.PROBABILITY)
    binary_volume = reconstruct_volume(binary_slices, geom, PredictionKind.BINARY)
    return probability, binary_volume, (loss_sum / n if n else None)


def load_original_gt(meta: dict[str, Any], repo_root: Path) -> tuple[np.ndarray, VolumeGeometry]:
    img = nib.load(repo_root / meta["source"]["label"])
    return binarize(img.get_fdata()), VolumeGeometry(tuple(img.shape[:3]), img.affine)


def restore_and_evaluate(
    binary: PredictionVolume,
    original_gt: np.ndarray,
    original: VolumeGeometry,
    case_id: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Original-space evaluation of a binary processed-space prediction (Phase 2 engines)."""
    if binary.kind is not PredictionKind.BINARY:
        raise ValueError("threshold before restoration: official order needs a binary volume")
    restored = restore_to_original(binary, original)
    return evaluate_case(restored.data, original_gt, original.spacing, case_id), restored.data
