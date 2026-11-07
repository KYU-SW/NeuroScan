"""v2 2D baseline policy fixes: brain-mask roles, four separate FP measures, zero padding with
valid_region_mask and masked losses. Synthetic arrays only, CPU only; no MRI, no training."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
import torch

from neuroscan_ai.v2.baseline_outputs import BRAIN_MASK_ROLES, build_outputs, roundtrip_mask_qa
from neuroscan_ai.v2.baseline_preprocess import prepare_arrays
from neuroscan_ai.v2.baseline_selection import THRESHOLD_CANDIDATES
from neuroscan_ai.v2.baseline_slices import (
    infer_volume,
    masked_bce_loss,
    masked_soft_dice_loss,
    pad_training_slice,
)
from neuroscan_ai.v2.geometry import check_on_grid
from neuroscan_ai.v2.safety_metrics import case_metrics, summarize
from neuroscan_ai.v2.unet_2d import BaselineUNet2D, build_model, count_parameters
from tests.test_v2_baseline_geometry import _affine, _volume

AFF = np.diag([1.0, 1.0, 2.0, 1.0])  # 0.002 mL voxels
FP_KEYS = (
    "false_positive_voxel_volume_ml",
    "no_gt_overlap_component_volume_ml",
    "no_gt_overlap_component_count",
    "oversegmentation_volume_ml",
)


def _case(spacing=(0.8, 0.8, 5.0), rot=10.0, seed=0):
    dwi, adc, gt = _volume((34, 30, 4), seed)
    aff = _affine(spacing, rot)
    return prepare_arrays(dwi, adc, gt, {"dwi": aff, "adc": aff, "gt": aff}), aff


# ---------------------------------------------------------------- 1-3 brain mask roles
def test_01_original_brain_mask_used_for_final_gating():
    case, _ = _case()
    res = build_outputs(np.full(case.model_grid.shape, 0.9), case, 0.5)
    assert np.array_equal(res.brain_mask, case.original_brain.astype(np.uint8))
    assert np.array_equal(res.gated, res.ungated & case.original_brain.astype(np.uint8))
    assert res.record["brain_mask_source"] == "original_grid_mask"
    assert BRAIN_MASK_ROLES["original_grid"] == [
        "final_gating",
        "boundary_analysis",
        "original_space_qa",
    ]
    assert BRAIN_MASK_ROLES["model_grid"] == ["input_normalisation", "model_input_qa"]


def test_02_roundtrip_mask_never_replaces_original():
    case, _ = _case(spacing=(0.7, 0.9, 5.0))
    original = case.original_brain.copy()
    res = build_outputs(np.full(case.model_grid.shape, 0.9), case, 0.5)
    assert res.record["roundtrip_mask_exact"] is False  # sub-mm: the round trip differs here
    assert np.array_equal(case.original_brain, original)  # not overwritten
    assert np.array_equal(res.brain_mask.astype(bool), original)  # not replaced
    assert BRAIN_MASK_ROLES["nearest_roundtrip"] == [
        "xor_vs_original",
        "roundtrip_interpolation_qa",
    ]


def test_03_roundtrip_xor_metrics_recorded():
    a = np.zeros((6, 6, 2), bool)
    a[1:5, 1:5, :] = True  # 32 voxels
    b = a.copy()
    b[1, 1, 0] = False
    b[5, 5, 1] = True  # xor 2
    qa = roundtrip_mask_qa(a, b)
    assert qa == {
        "roundtrip_mask_xor_voxels": 2,
        "roundtrip_mask_xor_ratio": round(2 / 32, 8),
        "roundtrip_mask_dice": round(2 * 31 / 64, 8),
        "roundtrip_mask_exact": False,
    }
    assert roundtrip_mask_qa(a, a)["roundtrip_mask_exact"] is True
    case, _ = _case(spacing=(1.0, 1.0, 5.0))  # 1 mm: identity grid -> exact
    rec = build_outputs(np.full(case.model_grid.shape, 0.2), case, 0.5).record
    assert rec["roundtrip_mask_exact"] is True and rec["roundtrip_mask_xor_voxels"] == 0
    assert rec["roundtrip_mask_dice"] == 1.0 and rec["roundtrip_mask_xor_ratio"] == 0.0


# ---------------------------------------------------------------- 4-8 four FP measures
def _fp_scene():
    gt = np.zeros((24, 24, 4), np.uint8)
    gt[2:6, 2:6, 1] = 1  # GT A, 16 voxels
    pred = np.zeros_like(gt)
    pred[2:6, 2:8, 1] = 1  # overlaps A: 16 inside + 8 outside GT (oversegmentation)
    pred[15:17, 15:17, 2] = 1  # no GT overlap, 4 voxels
    pred[20, 20, 3] = 1  # no GT overlap, 1 voxel
    return pred, gt


def test_04_false_positive_voxel_volume():
    m = case_metrics(*_fp_scene(), AFF)
    assert m["false_positive_voxel_volume_ml"] == pytest.approx((8 + 4 + 1) * 0.002)


def test_05_no_gt_overlap_component_volume():
    m = case_metrics(*_fp_scene(), AFF)
    assert m["no_gt_overlap_component_volume_ml"] == pytest.approx(5 * 0.002)


def test_06_no_gt_overlap_component_count():
    m = case_metrics(*_fp_scene(), AFF)
    assert m["no_gt_overlap_component_count"] == 2


def test_07_oversegmentation_of_gt_overlapping_component():
    m = case_metrics(*_fp_scene(), AFF)
    assert m["oversegmentation_volume_ml"] == pytest.approx(8 * 0.002)
    pred, gt = _fp_scene()
    pred[:] = 0
    pred[15:17, 15:17, 2] = 1  # only a non-overlapping component -> no oversegmentation
    assert case_metrics(pred, gt, AFF)["oversegmentation_volume_ml"] == 0.0


def test_08_four_measures_distinct_and_consistent():
    m = case_metrics(*_fp_scene(), AFF)
    vals = [m[k] for k in FP_KEYS]
    assert len(set(vals)) == 4  # 0.026, 0.01, 2, 0.016
    assert m["false_positive_voxel_volume_ml"] == pytest.approx(
        m["no_gt_overlap_component_volume_ml"] + m["oversegmentation_volume_ml"]
    )
    s = summarize([m, case_metrics(*_fp_scene(), AFF)])
    assert s["no_gt_overlap_component_count"] == 4
    assert s["false_positive_voxel_volume_ml"] == pytest.approx(0.052)
    assert s["oversegmentation_volume_ml"] == pytest.approx(0.032)
    for old in ("fp_volume_ml", "total_fp_volume_ml", "fp_component_count"):
        assert old not in m and old not in s


# ---------------------------------------------------------------- 9-14 padding and losses
def _slice(h=29, w=21, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(2, h, w)).astype(np.float32)
    label = np.zeros((h, w), bool)
    label[5:12, 4:9] = True
    brain = np.zeros((h, w), bool)
    brain[2:-2, 2:-2] = True
    return img, label, brain


def test_09_zero_padding_values():
    img, label, brain = _slice()
    p = pad_training_slice(img, label, brain)
    assert p["image"].shape == (2, 32, 24) and p["size"] == (29, 21)
    assert (p["image"][:, 29:, :] == 0.0).all() and (p["image"][:, :, 21:] == 0.0).all()
    assert (p["label"][29:, :] == 0).all() and (p["brain"][:, 21:] == 0).all()
    assert np.array_equal(p["image"][:, :29, :21], img)


def test_10_valid_region_mask():
    p = pad_training_slice(*_slice())
    v = p["valid_region_mask"]
    assert v.shape == (32, 24) and set(np.unique(v).tolist()) == {0, 1}
    assert v[:29, :21].all() and not v[29:, :].any() and not v[:, 21:].any()


def _tensors(p, logits_pad_value=None, seed=1):
    h, w = p["valid_region_mask"].shape
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, 1, h, w, generator=g)
    if logits_pad_value is not None:
        v = torch.from_numpy(p["valid_region_mask"]).bool()[None, None]
        logits = torch.where(v, logits, torch.full_like(logits, logits_pad_value))
    target = torch.from_numpy(p["label"]).float()[None, None]
    valid = torch.from_numpy(p["valid_region_mask"])[None, None]
    return logits, target, valid


def _real_region(p, logits):
    h, w = p["size"]
    return logits[..., :h, :w]


def test_11_dice_loss_excludes_padding():
    p = pad_training_slice(*_slice())
    logits, target, valid = _tensors(p)
    masked = masked_soft_dice_loss(logits, target, valid)
    h, w = p["size"]
    real = masked_soft_dice_loss(logits[..., :h, :w], target[..., :h, :w], torch.ones(1, 1, h, w))
    assert torch.allclose(masked, real, atol=1e-6)


def test_12_bce_loss_excludes_padding():
    p = pad_training_slice(*_slice())
    logits, target, valid = _tensors(p)
    masked = masked_bce_loss(logits, target, valid)
    h, w = p["size"]
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[..., :h, :w], target[..., :h, :w]
    )
    assert torch.allclose(masked, expected, atol=1e-6)


def test_13_padding_size_does_not_change_real_region_loss():
    img, label, brain = _slice(h=24, w=16)  # already a multiple of 8: no padding
    results = []
    for multiple in (8, 32, 64):  # 0, then larger padding areas
        p = pad_training_slice(img, label, brain, multiple=multiple)
        h, w = p["valid_region_mask"].shape
        g = torch.Generator().manual_seed(3)
        real = torch.randn(1, 1, 24, 16, generator=g)
        logits = torch.full((1, 1, h, w), -7.0)
        logits[..., :24, :16] = real
        _, target, valid = _tensors(p)
        results.append(
            (masked_bce_loss(logits, target, valid), masked_soft_dice_loss(logits, target, valid))
        )
    for bce, dice in results[1:]:
        assert torch.allclose(bce, results[0][0], atol=1e-6)
        assert torch.allclose(dice, results[0][1], atol=1e-6)
    # per-sample weighting: a heavily padded sample weighs the same in a batch
    pa = pad_training_slice(img, label, brain, multiple=64)
    la, ta, va = _tensors(pa, seed=5)
    lb, tb, vb = _tensors(pad_training_slice(*_slice(h=64, w=64, seed=2), multiple=64), seed=6)
    batch = masked_bce_loss(torch.cat([la, lb]), torch.cat([ta, tb]), torch.cat([va, vb]))
    single = (masked_bce_loss(la, ta, va) + masked_bce_loss(lb, tb, vb)) / 2
    assert torch.allclose(batch, single, atol=1e-6)


def test_14_arbitrary_padding_logits_have_no_effect():
    p = pad_training_slice(*_slice())
    base = _tensors(p, logits_pad_value=0.0)
    for pad_value in (-1e4, 1e4, 37.5):
        alt = _tensors(p, logits_pad_value=pad_value)
        assert torch.equal(masked_bce_loss(*alt), masked_bce_loss(*base))
        assert torch.equal(masked_soft_dice_loss(*alt), masked_soft_dice_loss(*base))
    logits, target, valid = _tensors(p, logits_pad_value=1e4)
    logits.requires_grad_(True)
    (
        masked_bce_loss(logits, target, valid) + masked_soft_dice_loss(logits, target, valid)
    ).backward()
    pad = ~valid.bool()
    assert torch.isfinite(logits.grad).all() and (logits.grad[pad] == 0).all()


# ---------------------------------------------------------------- 15-18 inference, fixed values
def test_15_inference_crops_padding_to_model_grid_shape():
    case, _ = _case(spacing=(0.9, 1.1, 5.0))
    assert any(s % 8 for s in case.model_grid.shape[:2])  # padding is actually needed
    torch.manual_seed(0)
    model = BaselineUNet2D(channels=(4, 8, 16, 32))
    prob = infer_volume(model, case.image, batch_size=3)
    assert prob.shape == case.model_grid.shape and prob.dtype == np.float32
    assert 0.0 <= prob.min() and prob.max() <= 1.0


def test_16_original_grid_shape_affine_restored_after_inference():
    case, aff = _case(spacing=(0.9, 1.1, 5.0), rot=25)
    torch.manual_seed(0)
    prob = infer_volume(BaselineUNet2D(channels=(4, 8, 16, 32)), case.image)
    res = build_outputs(prob, case, 0.5)
    for arr in (res.probability, res.ungated, res.brain_mask, res.gated):
        assert arr.shape == case.original_grid.shape
        assert check_on_grid(arr, res.grid, case.original_grid) == []
    assert np.allclose(res.grid.affine, aff)


def test_17_parameter_count_unchanged():
    assert count_parameters(build_model(seed=0)) == 2_439_577


def test_18_threshold_candidates_unchanged():
    assert list(THRESHOLD_CANDIDATES) == [float(Decimal(f"0.{d:02d}")) for d in range(10, 91, 5)]
    assert len(THRESHOLD_CANDIDATES) == 17
