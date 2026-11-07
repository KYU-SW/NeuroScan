"""v2 2D baseline model ``neuroscan_v2_2d_baseline_v1``: a 4-level U-Net (DWI + ADC -> 1 logit).

Channels [36, 72, 144, 288]. Block = (Conv3x3 -> InstanceNorm2d(affine) -> ReLU) x 2; encoder
MaxPool2d(2); decoder ConvTranspose2d(2, stride 2) -> concat skip -> block; 1x1 head.
Trainable parameters: 2,439,577 (fixed; checked at build time). Three poolings, so H and W must be
multiples of 8. Randomly initialised from an explicit seed; no pretrained weights.
"""

from __future__ import annotations

import torch
from torch import nn

MODEL_ID = "neuroscan_v2_2d_baseline_v1"
CHANNELS = (36, 72, 144, 288)
IN_CHANNELS = 2
EXPECTED_PARAMETERS = 2_439_577
INPUT_MULTIPLE = 2 ** (len(CHANNELS) - 1)  # 8


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.InstanceNorm2d(cout, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.InstanceNorm2d(cout, affine=True),
        nn.ReLU(inplace=True),
    )


class BaselineUNet2D(nn.Module):
    def __init__(self, channels: tuple[int, ...] = CHANNELS, in_channels: int = IN_CHANNELS):
        super().__init__()
        self.channels = tuple(channels)
        self.enc = nn.ModuleList()
        prev = in_channels
        for c in self.channels:
            self.enc.append(_block(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(self.channels) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(self.channels[i], self.channels[i - 1], 2, stride=2))
            self.dec.append(_block(self.channels[i - 1] * 2, self.channels[i - 1]))
        self.head = nn.Conv2d(self.channels[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 2, H, W) -> logits (B, 1, H, W)
        h, w = x.shape[-2:]
        if h % INPUT_MULTIPLE or w % INPUT_MULTIPLE:
            raise ValueError(f"input H and W must be multiples of {INPUT_MULTIPLE}")
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = self.pool(x)
        for up, dec in zip(self.up, self.dec, strict=True):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_model(seed: int) -> BaselineUNet2D:
    torch.manual_seed(seed)
    model = BaselineUNet2D()
    n = count_parameters(model)
    if n != EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameter count {n} != approved {EXPECTED_PARAMETERS}")
    return model


def describe(model: nn.Module, seed: int) -> dict:
    return {
        "model_id": MODEL_ID,
        "architecture": "4-level U-Net: (conv3x3-instancenorm-relu)x2 blocks, maxpool, "
        "transposed-conv up, 1x1 head",
        "channels": list(CHANNELS),
        "in_channels": IN_CHANNELS,
        "input_channels": ["dwi_z", "adc_z"],
        "out_channels": 1,
        "input_multiple": INPUT_MULTIPLE,
        "trainable_parameters": count_parameters(model),
        "init_seed": seed,
        "pretrained_weights": None,
        "output": "1 logit per pixel; sigmoid -> lesion probability",
    }
