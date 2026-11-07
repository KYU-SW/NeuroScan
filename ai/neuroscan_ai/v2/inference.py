"""v2 smoke inference: per-slice sigmoid probabilities on the original grid (padding cropped back).

The probability map is the model output before any brain mask or threshold. Ungated and gated
binaries are derived from it by the caller and both are kept.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from neuroscan_ai.v2.dataset import CaseArrays, pad_to_multiple


@torch.no_grad()
def predict_probability(
    model: nn.Module, case: CaseArrays, *, device: torch.device, pad_multiple: int, batch_size: int
) -> np.ndarray:
    model.to(device).eval()
    h, w, nz = case.shape
    prob = np.zeros((h, w, nz), dtype=np.float32)
    for start in range(0, nz, batch_size):
        zs = list(range(start, min(start + batch_size, nz)))
        x = np.stack([pad_to_multiple(case.image[:, :, :, z], pad_multiple)[0] for z in zs])
        p = torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()[:, 0, :h, :w]
        prob[:, :, zs] = np.moveaxis(p, 0, -1)
    return prob


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    s = int(a.sum()) + int(b.sum())
    return None if s == 0 else 2.0 * int((a & b).sum()) / s
