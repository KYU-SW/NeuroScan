"""In-memory slice dataset over processed/v1 patient volumes.

Each patient is loaded once, normalised per case (``training.normalization``) and kept in RAM.
A sample is one centre slice z:

  image [C, H, W]  C = modalities x (2*context + 1), modality-major:
                   context 0 (2D):   [DWI z, ADC z]
                   context 1 (2.5D): [DWI z-1, DWI z, DWI z+1, ADC z-1, ADC z, ADC z+1]
                   neighbours outside the volume repeat the edge slice
  label [1, H, W]  GT slice z (processed space)

Axes follow the processed RAS arrays: H = x, W = y, slice axis = z (index 2), matching
``reconstruction.reconstruct_volume``. Case IDs are always checked against the sealed test split.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from neuroscan_ai.data.split_guard import SplitGuard
from neuroscan_ai.training.brain_gating import compute_brain_mask
from neuroscan_ai.training.normalization import NormalizationSpec, normalize


@dataclass
class CaseVolumes:
    case_id: str
    image: np.ndarray  # [M, X, Y, Z] float32, normalised
    label: np.ndarray  # [X, Y, Z] uint8
    meta: dict[str, Any]
    brain: np.ndarray | None = None  # [X, Y, Z] bool, processed grid

    @property
    def n_slices(self) -> int:
        return self.label.shape[2]

    @property
    def z_spacing(self) -> float:
        return float(self.meta["original"]["spacing"][2])

    @property
    def has_lesion(self) -> bool:
        return bool(self.meta["lesion"]["has_lesion"])


def norm_specs(cfg_norm: dict[str, Any]) -> dict[str, NormalizationSpec]:
    return {
        m: NormalizationSpec(v["method"], tuple(v.get("clip_percentiles", (0.5, 99.5))))
        for m, v in cfg_norm.items()
    }


def load_case(
    processed_root: Path,
    case_id: str,
    modalities: Sequence[str],
    specs: dict[str, NormalizationSpec],
    brain_policy: str = "skull_stripped_background",
) -> CaseVolumes:
    d = Path(processed_root) / case_id
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    raw = {m: nib.load(d / f"{m}.nii.gz").get_fdata(dtype=np.float32) for m in modalities}
    brain = compute_brain_mask(brain_policy, raw["dwi"])
    image = np.stack([normalize(raw[m], brain, specs[m]) for m in modalities])
    label = np.asanyarray(nib.load(d / "mask.nii.gz").dataobj).astype(np.uint8)
    if label.shape != image.shape[1:]:
        raise ValueError(f"{case_id}: image {image.shape[1:]} vs label {label.shape}")
    return CaseVolumes(case_id, image, label, meta, brain)


def load_split_cases(
    guard: SplitGuard,
    split: str,
    processed_root: Path,
    modalities: Sequence[str],
    specs: dict[str, NormalizationSpec],
    case_ids: Sequence[str] | None = None,
    brain_policy: str = "skull_stripped_background",
) -> list[CaseVolumes]:
    """Load ``split`` (or a subset of it) after the test guard has approved every ID."""
    members = guard.cases(split)
    ids = list(members if case_ids is None else case_ids)
    unknown = sorted(set(ids) - set(members))
    if unknown:
        raise ValueError(f"cases not in split '{split}': {unknown}")
    guard.assert_no_test(ids, f"{split} loading")
    return [load_case(processed_root, cid, modalities, specs, brain_policy) for cid in ids]


def stack_context(image: np.ndarray, z: int, context: int) -> np.ndarray:
    """[M, X, Y, Z] -> [M*(2c+1), X, Y] around slice z, modality-major, edge-replicated."""
    n = image.shape[3]
    zs = [min(max(z + dz, 0), n - 1) for dz in range(-context, context + 1)]
    return np.concatenate([image[m][:, :, zs].transpose(2, 0, 1) for m in range(image.shape[0])])


def pad_or_crop(arr: np.ndarray, size: Sequence[int]) -> np.ndarray:
    """Centre pad (zeros) / crop the last two axes to ``size``."""
    out = arr
    for axis, target in zip((-2, -1), size, strict=True):
        n = out.shape[axis]
        if n < target:
            before = (target - n) // 2
            pad = [(0, 0)] * out.ndim
            pad[axis] = (before, target - n - before)
            out = np.pad(out, pad)
        elif n > target:
            start = (n - target) // 2
            out = np.take(out, range(start, start + target), axis=axis)
    return out


class SliceDataset(Dataset):
    def __init__(
        self,
        cases: Sequence[CaseVolumes],
        context: int = 0,
        spatial_size: Sequence[int] | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        exclude_brain_free_slices: bool = False,
    ) -> None:
        """``exclude_brain_free_slices`` (training only): drop slices whose brain mask has zero
        voxels. They carry no image information, are always gated to 0 at inference and caused
        the gradient spikes in Phase 3. Brain-positive lesion-negative slices are always kept."""
        self.cases = list(cases)
        self.context = context
        self.spatial_size = tuple(spatial_size) if spatial_size else None
        self.transform = transform
        self.exclude_brain_free_slices = exclude_brain_free_slices
        all_slices = [(ci, z) for ci, c in enumerate(self.cases) for z in range(c.n_slices)]
        if exclude_brain_free_slices and any(c.brain is None for c in self.cases):
            raise ValueError("excluding brain-free slices needs the brain mask of every case")
        self.n_slices_total = len(all_slices)
        self.index = [
            (ci, z)
            for ci, z in all_slices
            if not exclude_brain_free_slices or self.cases[ci].brain[:, :, z].any()
        ]
        self.n_brain_free_excluded = self.n_slices_total - len(self.index)
        self.slice_positive = np.array(
            [bool(self.cases[ci].label[:, :, z].any()) for ci, z in self.index]
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ci, z = self.index[i]
        case = self.cases[ci]
        image = stack_context(case.image, z, self.context)
        label = case.label[None, :, :, z].astype(np.float32)
        if self.spatial_size:
            image = pad_or_crop(image, self.spatial_size)
            label = pad_or_crop(label, self.spatial_size)
        sample = {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "label": torch.from_numpy(np.ascontiguousarray(label)),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        sample.update(
            case_id=case.case_id,
            slice_index=z,
            z_spacing=case.z_spacing,
            has_lesion=case.has_lesion,
            slice_has_lesion=bool(self.slice_positive[i]),
            brain_voxels=-1 if case.brain is None else int(case.brain[:, :, z].sum()),
        )
        return sample

    def stats(self) -> dict[str, Any]:
        per_case = {}
        for (ci, _), pos in zip(self.index, self.slice_positive, strict=True):
            per_case.setdefault(self.cases[ci].case_id, [0, 0])[0 if pos else 1] += 1
        pos_counts = np.array([v[0] for v in per_case.values()])
        return {
            "n_cases": len(self.cases),
            "n_slices_total": self.n_slices_total,
            "brain_free_slices_excluded": self.n_brain_free_excluded,
            "brain_free_excluded_fraction": self.n_brain_free_excluded
            / max(self.n_slices_total, 1),
            "n_slices": len(self),
            "positive_slices": int(self.slice_positive.sum()),
            "negative_slices": int((~self.slice_positive).sum()),
            "brain_positive_negative_slices": int(
                sum(
                    1
                    for (ci, z), pos in zip(self.index, self.slice_positive, strict=True)
                    if not pos
                    and self.cases[ci].brain is not None
                    and self.cases[ci].brain[:, :, z].any()
                )
            ),
            "positive_ratio": float(self.slice_positive.mean()) if len(self) else 0.0,
            "positive_slices_per_case": {
                "min": int(pos_counts.min()),
                "median": float(np.median(pos_counts)),
                "mean": float(pos_counts.mean()),
                "max": int(pos_counts.max()),
            },
        }
