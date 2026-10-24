"""Conservative train-only 2D augmentation (MONAI dictionary transforms).

One RandAffined call transforms image (all channels: DWI and ADC, every context slice) and label
together, so they share exactly the same spatial transform; label uses nearest interpolation.
Left-right flip is off by default: lesion side is meaningful for later location analysis.
"""

from __future__ import annotations

import math
from typing import Any

from monai.transforms import (
    Compose,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


def build_train_transform(cfg: dict[str, Any], seed: int) -> Compose | None:
    if not cfg.get("enabled", False):
        return None
    t = [
        RandAffined(
            keys=("image", "label"),
            prob=cfg["affine_prob"],
            rotate_range=(math.radians(cfg["rotate_deg"]),),
            scale_range=(cfg["scale"], cfg["scale"]),
            translate_range=(cfg["translate_px"], cfg["translate_px"]),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        )
    ]
    if cfg.get("left_right_flip", False):
        t.append(RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0))
    t += [
        RandScaleIntensityd(
            keys="image",
            factors=cfg["intensity_scale"],
            prob=cfg["intensity_prob"],
            channel_wise=True,
        ),
        RandShiftIntensityd(
            keys="image",
            offsets=cfg["intensity_shift"],
            prob=cfg["intensity_prob"],
            channel_wise=True,
        ),
        RandGaussianNoised(keys="image", std=cfg["noise_std"], prob=cfg["noise_prob"]),
    ]
    compose = Compose(t)
    compose.set_random_state(seed=seed)
    return compose
