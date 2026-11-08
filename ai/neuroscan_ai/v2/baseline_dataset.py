"""v2 2D baseline dataset: prepared cases, brain-slice index, padded slice items and sampling.

Cases are read with ``baseline_preprocess.load_and_prepare`` (read-only):
``preprocess_v2_isles_1mm_inplane_v1`` (1 mm in-plane, z kept, DWI / ADC linear, GT / brain mask
nearest, GT decode ``binary_mask_decode_v1_atol_1e_6``, z-score over the model-grid brain mask).
Slices are zero-padded to multiples of 8 with ``valid_region_mask`` (``baseline_slices``).

Items and batches carry the position-based ``case_key`` (``case_NNNN``) and the slice index only;
aliases, subject IDs and file paths never enter a batch or an output.

GT case status (``gt_case_status_v1``): an empty original GT is a valid negative case and is kept
(all its brain slices are lesion-negative, ``negative_only_case``); a positive GT that becomes empty
or loses a 26-connected component on the model grid is refused (``PreprocessError``).

Sampling (per epoch, deterministic from the run seed and the epoch number): candidates are the
slices whose model-grid brain mask is non-empty. The epoch length equals the number of candidate
slices; round(0.5 x length) draws come from lesion-positive slices and the rest from
lesion-negative slices, each class walked through in seeded permutations (repeated when a class is
shorter), then the combined list is shuffled. No slice or case is removed: every negative slice
stays a candidate and is visited across epochs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from neuroscan_ai.v2.baseline_preprocess import PreparedCase, PreprocessError, load_and_prepare
from neuroscan_ai.v2.baseline_slices import pad_training_slice
from neuroscan_ai.v2.dataset import CaseFiles
from neuroscan_ai.v2.geometry import decode_binary_mask, resample_mask
from neuroscan_ai.v2.safety_metrics import GT_CASE_STATUS, GT_LOST, NEGATIVE_CASE, gt_case_status
from neuroscan_ai.v2.unet_2d import INPUT_MULTIPLE

SAMPLING_POLICY = "brain_slices_positive_fraction_0_5_seed_epoch_v1"


class SamplingError(ValueError):
    pass


@dataclass
class BaselineCase:
    case_key: str
    prepared: PreparedCase
    original_gt: np.ndarray  # bool, original grid, decoded with binary_mask_decode_v1_atol_1e_6
    brain_slices: list[int] = field(default_factory=list)
    positive_slices: list[int] = field(default_factory=list)
    ground_truth_status: str = ""
    gt_components_lost: int = 0

    @property
    def negative_only_case(self) -> bool:
        return self.ground_truth_status == NEGATIVE_CASE

    @property
    def record(self) -> dict[str, Any]:
        """Preprocessing audit (no paths, no IDs)."""
        r = self.prepared.record
        keys = (
            "preprocess_policy",
            "brain_mask_policy",
            "target_inplane_mm",
            "slice_axis",
            "slice_axis_resampled",
            "interpolation",
            "resampled",
            "original_grid",
            "model_grid",
            "target_size",
            "gt_decode",
            "brain_fraction_original",
            "gt_voxels_original",
            "gt_voxels_model_grid",
            "normalisation",
            "adc_values_converted",
        )
        return {
            "case_key": self.case_key,
            **{k: r[k] for k in keys},
            "ground_truth_status": self.ground_truth_status,
            "gt_case_status_policy": GT_CASE_STATUS["policy_id"],
            "gt_empty_expected": self.negative_only_case,
            "negative_only_case": self.negative_only_case,
            "gt_components_lost": self.gt_components_lost,
        }


def components_lost(original_gt: np.ndarray, prepared: PreparedCase) -> int:
    """Original 26-connected GT components with no voxel left after the nearest round trip."""
    if not original_gt.any():
        return 0
    back = resample_mask(prepared.label, prepared.model_grid, prepared.original_grid)
    lab, n = ndimage.label(original_gt, np.ones((3, 3, 3), bool))
    return sum(not back[lab == i].any() for i in range(1, n + 1))


def load_case(case_key: str, files: CaseFiles) -> BaselineCase:
    prepared = load_and_prepare(files.dwi, files.adc, files.label)
    gt, _ = decode_binary_mask(np.asanyarray(nib.load(str(files.label)).dataobj))
    brain = prepared.brain
    zs = [z for z in range(brain.shape[2]) if brain[:, :, z].any()]
    pos = [z for z in zs if prepared.label[:, :, z].any()]
    lost = components_lost(gt, prepared)
    status = gt_case_status(int(gt.sum()), int(prepared.label.sum()), lost)
    if status == GT_LOST:
        raise PreprocessError(f"{case_key}: positive GT lost during preprocessing (blocked)")
    return BaselineCase(case_key, prepared, gt, zs, pos, status, lost)


def load_cases(files: Mapping[str, CaseFiles], keys: Mapping[str, str]) -> list[BaselineCase]:
    """Cases in case-key order; ``files`` and ``keys`` are keyed by alias (never stored)."""
    return [load_case(keys[a], files[a]) for a in sorted(files, key=lambda a: keys[a])]


def slice_item(case: BaselineCase, z: int) -> dict[str, Any]:
    p = case.prepared
    padded = pad_training_slice(p.image[:, :, :, z], p.label[:, :, z], p.brain[:, :, z])
    return {
        "image": padded["image"],  # (2, H, W) float32
        "target": padded["label"][None].astype(np.float32),  # (1, H, W)
        "brain_mask": padded["brain"][None].astype(np.float32),
        "valid_region_mask": padded["valid_region_mask"][None].astype(np.float32),
        "case_key": case.case_key,
        "slice_index": int(z),
        "geometry": {
            "model_grid_shape": list(p.model_grid.shape),
            "model_grid_spacing_mm": [round(s, 8) for s in p.model_grid.spacing],
            "unpadded_hw": list(padded["size"]),
            "padded_hw": list(padded["image"].shape[1:]),
        },
    }


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack items; slices of different size are zero-padded at the end to the batch maximum
    (still multiples of 8) with ``valid_region_mask`` 0 there."""
    h = max(i["image"].shape[1] for i in items)
    w = max(i["image"].shape[2] for i in items)
    out: dict[str, Any] = {}
    for k in ("image", "target", "brain_mask", "valid_region_mask"):
        arrs = []
        for i in items:
            a = i[k]
            arrs.append(np.pad(a, [(0, 0), (0, h - a.shape[1]), (0, w - a.shape[2])]))
        out[k] = torch.from_numpy(np.stack(arrs))
    assert h % INPUT_MULTIPLE == 0 and w % INPUT_MULTIPLE == 0
    out["case_key"] = [i["case_key"] for i in items]
    out["slice_index"] = [i["slice_index"] for i in items]
    return out


def batches(cases: list[BaselineCase], order: list[tuple[int, int]], batch_size: int):
    for start in range(0, len(order), batch_size):
        yield collate([slice_item(cases[ci], z) for ci, z in order[start : start + batch_size]])


# ---------------------------------------------------------------- sampling
def slice_index(cases: list[BaselineCase]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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


def epoch_sampling(
    cases: list[BaselineCase], seed: int, epoch: int, positive_fraction: float = 0.5
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
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
        "slice_list_sha256": hashlib.sha256(
            json.dumps(listing, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    return order, record
