"""Model and loss factories. Output contract: the network returns raw logits [B, 1, H, W].

logits --(loss: DiceCELoss sigmoid=True / BCEWithLogits)--> training signal
logits --sigmoid (once, in inference.predict)--> probability --threshold (evaluation)--> mask
"""

from __future__ import annotations

from typing import Any

import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet

from neuroscan_ai.training.config import in_channels


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    m = cfg["model"]
    if m["name"] != "monai_unet":
        raise ValueError(f"unsupported model {m['name']}")
    return UNet(
        spatial_dims=m["spatial_dims"],
        in_channels=in_channels(cfg),
        out_channels=m["out_channels"],
        channels=tuple(m["channels"]),
        strides=tuple(m["strides"]),
        num_res_units=m["num_res_units"],
        norm=m["norm"],
        dropout=m["dropout"],
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_loss(cfg: dict[str, Any]) -> torch.nn.Module:
    loss = cfg["loss"]
    if loss["name"] != "dice_ce":
        raise ValueError(f"unsupported loss {loss['name']}")
    # sigmoid=True: Dice term applies sigmoid internally; the 1-channel CE term is
    # BCEWithLogitsLoss on the raw logits. So the model must NOT apply sigmoid itself.
    return DiceCELoss(
        sigmoid=True,
        batch=loss["batch_dice"],
        lambda_dice=loss["lambda_dice"],
        lambda_ce=loss["lambda_ce"],
        smooth_nr=loss["smooth_nr"],
        smooth_dr=loss["smooth_dr"],
    )
