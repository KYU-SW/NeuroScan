"""v2 2D smoke model ``neuroscan_v2_2d_smoke_v1``: a small two-level U-Net (DWI + ADC -> 1 logit).

Randomly initialised from an explicit seed; no pretrained weights. Upsampling uses a transposed
convolution (deterministic under ``torch.use_deterministic_algorithms``).
"""

from __future__ import annotations

import torch
from torch import nn

MODEL_ID = "neuroscan_v2_2d_smoke_v1"


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class SmokeUNet2D(nn.Module):
    def __init__(self, in_channels: int = 2, base: int = 8):
        super().__init__()
        self.enc1 = _block(in_channels, base)
        self.enc2 = _block(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 2, H, W), H and W even
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.dec1(torch.cat([self.up(e2), e1], dim=1))
        return self.head(d1)  # logits (B, 1, H, W)


def build_model(seed: int, base: int) -> SmokeUNet2D:
    torch.manual_seed(seed)
    return SmokeUNet2D(2, base)


def describe(model: nn.Module, seed: int, base: int) -> dict:
    return {
        "model_id": MODEL_ID,
        "architecture": "two-level U-Net: conv3x3x2 blocks, maxpool, transposed-conv up, 1x1 head",
        "in_channels": 2,
        "input_channels": ["dwi_z", "adc_z"],
        "base_channels": base,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "init_seed": seed,
        "pretrained_weights": None,
        "output": "1 logit per pixel; sigmoid -> lesion probability",
    }
