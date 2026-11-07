"""Deterministic training for the v2 smoke model (FP32, no AMP, no TF32, deterministic kernels)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch import nn

from neuroscan_ai.v2.dataset import CaseArrays, pad_to_multiple


def set_determinism(seed: int) -> dict[str, Any]:
    """Apply the smoke determinism settings and return them (for the manifest)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"seed": seed, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def resolve_device(name: str) -> torch.device:
    if name == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def brain_slices(cases: list[CaseArrays]) -> list[tuple[int, int]]:
    """(case index, z) for every slice that contains brain, in a fixed order."""
    return [
        (i, z) for i, c in enumerate(cases) for z in range(c.shape[2]) if c.brain[:, :, z].any()
    ]


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + 1.0) / (p.sum() + target.sum() + 1.0)


def train_epochs(
    model: nn.Module,
    cases: list[CaseArrays],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    pad_multiple: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    index = brain_slices(cases)
    gen = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        order = torch.randperm(len(index), generator=gen).tolist()
        losses = []
        for start in range(0, len(order), batch_size):
            batch = [index[k] for k in order[start : start + batch_size]]
            # pad each slice to a common size (cases can differ in-plane)
            xs, ys = [], []
            for ci, z in batch:
                x, _ = pad_to_multiple(cases[ci].image[:, :, :, z], pad_multiple)
                y, _ = pad_to_multiple(
                    cases[ci].label[:, :, z][None].astype(np.float32), pad_multiple
                )
                xs.append(x)
                ys.append(y)
            hmax = max(a.shape[-2] for a in xs)
            wmax = max(a.shape[-1] for a in xs)
            x = np.stack(
                [np.pad(a, [(0, 0), (0, hmax - a.shape[-2]), (0, wmax - a.shape[-1])]) for a in xs]
            )
            y = np.stack(
                [np.pad(a, [(0, 0), (0, hmax - a.shape[-2]), (0, wmax - a.shape[-1])]) for a in ys]
            )
            xt = torch.from_numpy(x).to(device)
            yt = torch.from_numpy(y).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xt)
            loss = bce(logits, yt) + _soft_dice_loss(logits, yt)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch + 1,
                "batches": len(losses),
                "slices": len(order),
                "loss_mean": float(np.mean(losses)) if losses else None,
                "loss_per_batch": losses,
            }
        )
    return history
