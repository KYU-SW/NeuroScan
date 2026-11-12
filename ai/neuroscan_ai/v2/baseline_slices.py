"""v2 baseline slice padding, valid-region masked losses and padded inference (code only).

Padding policy (to make H and W multiples of 8 for the 4-level U-Net):
- zero padding at the end of each in-plane axis: normalised DWI 0.0, normalised ADC 0.0
  (0.0 = the per-case normalisation mean), GT 0, brain mask 0;
- ``valid_region_mask``: 1 on the real image region, 0 on padding.

Losses are computed only inside ``valid_region_mask``:
- BCE: per-sample mean over valid voxels, then mean over the batch;
- soft Dice: per-sample Dice over valid voxels, then mean over the batch.
Padding voxels are never background training voxels, and the padding area of a sample does not
change its weight. Arbitrary logits in padding have no effect on the loss or its gradient.

Inference pads each slice, runs the model, applies sigmoid and CROPS the padding away before the
volume is assembled on the model grid; padding is never stored. The model-grid volume is then
restored to the original grid by ``baseline_outputs.build_outputs``.
The weighting / combination of the two losses is a training hyperparameter (pending approval).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from neuroscan_ai.v2.geometry import crop_to, pad_to_multiple
from neuroscan_ai.v2.unet_2d import INPUT_MULTIPLE

IMAGE_PAD_VALUE = 0.0
MASK_PAD_VALUE = 0
DICE_SMOOTH = 1.0  # provisional (part of the pending loss hyperparameters)


def pad_training_slice(
    image: np.ndarray,
    label: np.ndarray,
    brain: np.ndarray,
    multiple: int = INPUT_MULTIPLE,
    valid: np.ndarray | None = None,
) -> dict:
    """Pad one slice: image (C, H, W), label / brain (H, W). Returns padded copies plus
    ``valid_region_mask`` (uint8) and the original size. ``valid`` is the unpadded valid-region
    slice (a cached ``valid_region_mask``; it must be all 1: the real image region); default 1."""
    if image.ndim != 3 or label.shape != image.shape[1:] or brain.shape != image.shape[1:]:
        raise ValueError("image must be (C, H, W) and masks (H, W) on the same grid")
    img, size = pad_to_multiple(np.asarray(image, dtype=np.float32), multiple)
    if valid is None:
        valid = np.ones(size, np.uint8)
    else:
        valid = np.asarray(valid, dtype=np.uint8)
        if valid.shape != size or not (valid == 1).all():
            raise ValueError("valid_region_mask of an unpadded slice must be all 1")
    out = {
        "image": img,
        "label": pad_to_multiple(np.asarray(label).astype(np.uint8), multiple)[0],
        "brain": pad_to_multiple(np.asarray(brain).astype(np.uint8), multiple)[0],
        "valid_region_mask": pad_to_multiple(valid, multiple)[0],
        "size": size,
    }
    return out


def _check(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != valid.shape or logits.dim() != 4:
        raise ValueError("logits, target and valid mask must all be (B, 1, H, W)")
    v = valid.bool()
    if (v.flatten(1).sum(1) == 0).any():
        raise ValueError("a sample has no valid voxels")
    return v


def masked_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    v = _check(logits, target, valid)
    safe = torch.where(v, logits, torch.zeros_like(logits))
    bce = F.binary_cross_entropy_with_logits(safe, target.float(), reduction="none")
    bce = torch.where(v, bce, torch.zeros_like(bce))
    per_sample = bce.flatten(1).sum(1) / v.flatten(1).sum(1)
    return per_sample.mean()


def masked_soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, smooth: float = DICE_SMOOTH
) -> torch.Tensor:
    v = _check(logits, target, valid)
    p = torch.where(v, torch.sigmoid(torch.where(v, logits, torch.zeros_like(logits))), 0.0)
    g = torch.where(v, target.float(), 0.0)
    inter = (p * g).flatten(1).sum(1)
    denom = p.flatten(1).sum(1) + g.flatten(1).sum(1)
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


@torch.no_grad()
def infer_volume(
    model: torch.nn.Module, image: np.ndarray, batch_size: int = 8, device: str = "cpu"
) -> np.ndarray:
    """Model-grid probability (X, Y, Z) from a prepared image (2, X, Y, Z); padding cropped."""
    if image.ndim != 4:
        raise ValueError("image must be (C, X, Y, Z)")
    model.eval()
    nz = image.shape[3]
    out = np.zeros(image.shape[1:], np.float32)
    for start in range(0, nz, batch_size):
        zs = range(start, min(nz, start + batch_size))
        padded = [pad_to_multiple(image[:, :, :, z], INPUT_MULTIPLE) for z in zs]
        x = torch.from_numpy(np.stack([p for p, _ in padded])).to(device)
        prob = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
        for z, pr, (_, size) in zip(zs, prob, padded, strict=True):
            out[:, :, z] = crop_to(pr, size)
    return out
