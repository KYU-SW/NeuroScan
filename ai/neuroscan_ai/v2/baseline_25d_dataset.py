"""v2 2.5D baseline dataset: 3-slice stacks read from the disk cache, batches and sampling.

Fixed policies (recorded, never tuned):
- input   ``input_25d_slice_major_dwi_adc_pairs_v1``: [DWI_z-1, ADC_z-1, DWI_z, ADC_z, DWI_z+1,
  ADC_z+1]; channels 2 and 3 are exactly the 2D baseline input of slice z.
- neighbour ``neighbor_fixed_adjacent_index_pm1_v1``: source slices z-1, z, z+1 (index based; the
  physical distance therefore differs between cases).
- boundary ``edge_slice_repeat_v1``: a missing neighbour repeats the edge slice. No zero slice, no
  wrap-around, no edge exclusion, no z interpolation.
- normalisation ``normalization_case_volume_zscore_brainmask_no_slice_renorm_v1``: the cached
  case-volume z-score (brain-mask statistics) is used as is; nothing is re-normalised here.

``stack_slices`` is the only stack function: training (``slice_item``) and inference
(``NeighborStackView``) both use it. Slices are read on demand; a stack is never stored.

Target, sampling class and masks belong to the centre slice only. A slice whose neighbour holds a
lesion but whose centre does not stays a negative sample (``neighbor_only_positive`` is recorded).
The epoch order depends only on (seed, epoch) and the centre slices.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch

from neuroscan_ai.v2.baseline_cache import CachedCase, ChannelView
from neuroscan_ai.v2.baseline_slices import pad_training_slice
from neuroscan_ai.v2.safety_metrics import slice_subgroup
from neuroscan_ai.v2.unet_25d import IN_CHANNELS, INPUT_CHANNEL_NAMES, INPUT_MULTIPLE

INPUT_CHANNEL_POLICY = "input_25d_slice_major_dwi_adc_pairs_v1"
NEIGHBOR_POLICY = "neighbor_fixed_adjacent_index_pm1_v1"
BOUNDARY_POLICY = "edge_slice_repeat_v1"
NORMALIZATION_POLICY = "normalization_case_volume_zscore_brainmask_no_slice_renorm_v1"
SAMPLING_POLICY = "brain_slices_positive_fraction_0_5_seed_epoch_v1"  # same rule as the 2D baseline
METADATA_FIELDS = (
    "case_key",
    "center_slice_index",
    "source_slice_indices",
    "edge_repeat_applied",
    "z_spacing_mm",
    "thick_slice_subgroup",
    "input_channel_policy_id",
    "center_positive",
    "neighbor_only_positive",
    "positive_neighbor_count",
)


class SamplingError(ValueError):
    pass


class StackError(ValueError):
    """An input stack that does not follow the fixed policies."""


# ---------------------------------------------------------------- stack
def source_slice_indices(z: int, n_slices: int) -> list[int]:
    """[max(z-1, 0), z, min(z+1, Z-1)]; one list for DWI and ADC."""
    if isinstance(z, bool) or not isinstance(z, (int, np.integer)) or n_slices < 1:
        raise StackError("slice index must be an integer and the volume must have slices")
    z = int(z)
    if not 0 <= z < n_slices:
        raise StackError("centre slice outside the volume")
    return [max(z - 1, 0), z, min(z + 1, int(n_slices) - 1)]


def image_view(case: Any):
    """(2, X, Y, Z) DWI / ADC view of a case; ``view[:, :, :, z]`` is one slice pair."""
    if isinstance(case, CachedCase):
        return ChannelView(case.store, case.case_key, case.model_grid_shape)
    return case.prepared.image


def stack_slices(view: Any, z: int) -> np.ndarray:
    """(6, X, Y) float32 stack of slice ``z``, fixed channel order (a copy; values unchanged)."""
    if view.ndim != 4 or view.shape[0] != 2:
        raise StackError("source image must be (2, X, Y, Z): dwi, adc")
    pairs = [np.asarray(view[:, :, :, s]) for s in source_slice_indices(z, view.shape[3])]
    out = np.concatenate(pairs, axis=0)
    if out.shape[0] != IN_CHANNELS or out.dtype != np.float32:
        raise StackError("stack must hold 6 float32 channels")
    return out


class NeighborStackView:
    """Read-only (6, X, Y, Z) view for slice-wise inference (``baseline_slices.infer_volume``)."""

    ndim = 4

    def __init__(self, view: Any):
        self._view = view
        self.shape = (IN_CHANNELS, *view.shape[1:])

    def __getitem__(self, idx):
        if (
            not isinstance(idx, tuple)
            or len(idx) != 4
            or any(s != slice(None) for s in idx[:3])
            or not isinstance(idx[3], (int, np.integer))
        ):
            raise TypeError("NeighborStackView supports [:, :, :, z] only")
        return stack_slices(self._view, int(idx[3]))


# ---------------------------------------------------------------- items and batches
def sample_metadata(case: Any, z: int) -> dict[str, Any]:
    """Audit fields of one sample (no alias, no subject ID, no path)."""
    n = int(case.model_grid_shape[2])
    src = source_slice_indices(z, n)
    pos = set(case.positive_slices)
    center = int(z) in pos
    neighbors = {s for s in (src[0], src[2]) if s != int(z)}  # a repeated edge is no neighbour
    n_pos = len(neighbors & pos)
    spacing = float(case.model_grid_spacing[2])
    return {
        "case_key": case.case_key,
        "center_slice_index": int(z),
        "source_slice_indices": src,
        "edge_repeat_applied": len(set(src)) < 3,
        "z_spacing_mm": round(spacing, 6),
        "thick_slice_subgroup": slice_subgroup(spacing)["slice_group"],
        "input_channel_policy_id": INPUT_CHANNEL_POLICY,
        "center_positive": center,
        "neighbor_only_positive": (not center) and n_pos > 0,
        "positive_neighbor_count": n_pos,
    }


def slice_item(case: Any, z: int) -> dict[str, Any]:
    """One padded sample: 6-channel stack; target, brain mask and valid region of slice ``z``."""
    _, label, brain, valid = case.slice_arrays(z)
    padded = pad_training_slice(stack_slices(image_view(case), z), label, brain, valid=valid)
    return {
        "image": padded["image"],  # (6, H, W) float32
        "target": padded["label"][None].astype(np.float32),  # (1, H, W)
        "brain_mask": padded["brain"][None].astype(np.float32),
        "valid_region_mask": padded["valid_region_mask"][None].astype(np.float32),
        "case_key": case.case_key,
        "slice_index": int(z),
        "metadata": sample_metadata(case, z),
        "geometry": {
            "unpadded_hw": list(padded["size"]),
            "padded_hw": list(padded["image"].shape[1:]),
        },
    }


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack items; smaller slices are zero-padded at the end to the batch maximum (still multiples
    of 8) with ``valid_region_mask`` 0 there."""
    h = max(i["image"].shape[1] for i in items)
    w = max(i["image"].shape[2] for i in items)
    if h % INPUT_MULTIPLE or w % INPUT_MULTIPLE:
        raise StackError("batch size in H / W must be a multiple of 8")
    out: dict[str, Any] = {}
    for k in ("image", "target", "brain_mask", "valid_region_mask"):
        arrs = [
            np.pad(i[k], [(0, 0), (0, h - i[k].shape[1]), (0, w - i[k].shape[2])]) for i in items
        ]
        out[k] = torch.from_numpy(np.stack(arrs))
    out["case_key"] = [i["case_key"] for i in items]
    out["slice_index"] = [i["slice_index"] for i in items]
    out["metadata"] = [i["metadata"] for i in items]
    return out


def batches(cases: list, order: list[tuple[int, int]], batch_size: int):
    for start in range(0, len(order), batch_size):
        yield collate([slice_item(cases[ci], z) for ci, z in order[start : start + batch_size]])


# ---------------------------------------------------------------- sampling
def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode("utf-8")).hexdigest()


def slice_index(cases: list) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(positive, negative) centre slices among the brain slices; neighbours are not considered."""
    pos, neg = [], []
    for ci, c in enumerate(cases):
        ps = set(c.positive_slices)
        for z in c.brain_slices:
            (pos if z in ps else neg).append((ci, z))
    return pos, neg


def _stream(items: list, n: int, rng: np.random.Generator) -> list:
    out: list = []
    while len(out) < n:
        out += [items[k] for k in rng.permutation(len(items))]
    return out[:n]


def slice_table(cases: list) -> list[dict[str, Any]]:
    """Metadata of every candidate centre slice (case order, then slice order)."""
    return [sample_metadata(c, z) for c in cases for z in c.brain_slices]


def epoch_sampling(
    cases: list, seed: int, epoch: int, positive_fraction: float = 0.5
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Centre-slice order of one epoch; a function of (seed, epoch) only."""
    pos, neg = slice_index(cases)
    total = len(pos) + len(neg)
    if total == 0:
        raise SamplingError("no brain slices to sample")
    if not pos:
        raise SamplingError("no lesion-positive training slice; positive fraction impossible")
    n_pos = int(round(positive_fraction * total)) if neg else total
    rng = np.random.default_rng([int(seed), int(epoch)])
    order = _stream(pos, n_pos, rng) + (_stream(neg, total - n_pos, rng) if neg else [])
    order = [order[k] for k in rng.permutation(len(order))]
    listing = [[cases[ci].case_key, int(z)] for ci, z in order]
    meta = {}
    for ci, z in order:
        if (ci, z) not in meta:
            meta[(ci, z)] = sample_metadata(cases[ci], z)
    sources = [[cases[ci].case_key, meta[(ci, z)]["source_slice_indices"]] for ci, z in order]
    record = {
        "policy": SAMPLING_POLICY,
        "epoch": int(epoch),
        "sampling_seed": [int(seed), int(epoch)],
        "brain_slices": total,
        "positive_slices": len(pos),
        "negative_slices": len(neg),
        "sampled": len(order),
        "sampled_positive": n_pos,
        "sampled_positive_ratio": round(n_pos / len(order), 6),
        "expected_positive_fraction": positive_fraction,
        "slice_list_sha256": _sha(listing),
        "source_list_sha256": _sha(sources),
        "input_channel_policy_id": INPUT_CHANNEL_POLICY,
        "sampled_neighbor_only_positive": sum(meta[k]["neighbor_only_positive"] for k in order),
        "sampled_edge_repeat": sum(meta[k]["edge_repeat_applied"] for k in order),
    }
    return order, record


assert len(INPUT_CHANNEL_NAMES) == IN_CHANNELS
