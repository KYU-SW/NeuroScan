"""Reproducibility settings.

Bitwise-identical GPU training is not guaranteed even with these settings: some CUDA kernels
(e.g. atomic-add based backward passes) have no deterministic implementation, and
``warn_only=True`` lets them run with a warning instead of failing.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from monai.utils import set_determinism


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "python_random": seed,
        "numpy": seed,
        "torch": seed,
        "monai_set_determinism": seed,
        "cudnn_deterministic": deterministic,
        "cudnn_benchmark": not deterministic,
        "torch_deterministic_algorithms": "warn_only" if deterministic else False,
        "dataloader_workers": 0,
        "sampler_seed": "seed + epoch",
        "augmentation_seed": seed,
    }
