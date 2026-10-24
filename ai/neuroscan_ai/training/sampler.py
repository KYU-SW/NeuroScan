"""Positive/negative slice ratio sampler (with replacement).

Each draw is a lesion-positive slice with probability ``positive_fraction`` (uniform among
positive slices), otherwise a lesion-negative slice (uniform among negative slices, which include
every slice of negative patients). Background slices are never excluded.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class PositiveRatioSampler(Sampler[int]):
    def __init__(
        self,
        slice_positive: np.ndarray,
        positive_fraction: float,
        num_samples: int,
        seed: int,
    ) -> None:
        if not 0.0 < positive_fraction < 1.0:
            raise ValueError("positive_fraction must be in (0, 1): never only one class")
        self.pos = np.flatnonzero(slice_positive)
        self.neg = np.flatnonzero(~np.asarray(slice_positive, bool))
        if self.pos.size == 0 or self.neg.size == 0:
            raise ValueError("need both positive and negative slices")
        self.positive_fraction = positive_fraction
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0
        self.last_positive_fraction: float | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        is_pos = torch.rand(self.num_samples, generator=g) < self.positive_fraction
        n_pos = int(is_pos.sum())
        pos = self.pos[torch.randint(len(self.pos), (n_pos,), generator=g).numpy()]
        neg = self.neg[
            torch.randint(len(self.neg), (self.num_samples - n_pos,), generator=g).numpy()
        ]
        out = np.empty(self.num_samples, dtype=np.int64)
        out[is_pos.numpy()] = pos
        out[~is_pos.numpy()] = neg
        self.last_positive_fraction = n_pos / self.num_samples
        return iter(out.tolist())
