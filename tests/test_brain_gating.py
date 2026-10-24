"""Brain-mask gating: anatomical ROI restriction in the official inference pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuroscan_ai.reconstruction.volume import PredictionKind, PredictionVolume, VolumeGeometry
from neuroscan_ai.training.brain_gating import (
    BrainMaskError,
    check_gt_inside_brain,
    compute_brain_mask,
    gate_probability,
)
from neuroscan_ai.training.config import load_config
from neuroscan_ai.training.dataset import CaseVolumes
from neuroscan_ai.training.inference import predict_case, restore_and_evaluate
from neuroscan_ai.training.model import build_model
from tests.conftest import REPO_ROOT

POLICY = "skull_stripped_background"


def skull_stripped(shape=(24, 24, 6), bg=0.0):
    vol = np.full(shape, bg, np.float32)
    x, y = shape[0] // 4, shape[1] // 4
    block = (shape[0] - 2 * x, shape[1] - 2 * y, shape[2])
    vol[x : shape[0] - x, y : shape[1] - y, :] = np.random.default_rng(0).normal(200, 40, block)
    return vol


def test_config_declares_official_gating_and_policy():
    cfg = load_config(REPO_ROOT / "configs" / "2d_baseline.yaml")
    assert cfg["inference"]["brain_mask_gating"] is True
    assert cfg["data"]["brain_mask_policy"] == POLICY
    tiny = load_config(REPO_ROOT / "configs" / "2d_tiny_overfit.yaml")
    assert tiny["inference"]["brain_mask_gating"] is True


def test_gating_keeps_brain_and_zeroes_outside():
    brain = compute_brain_mask(POLICY, skull_stripped())
    prob = np.full(brain.shape, 0.8, np.float32)
    out = gate_probability(prob, brain)
    assert (out[brain] == 0.8).all() and (out[~brain] == 0).all()
    assert out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1


def test_tiny_lesion_inside_brain_is_preserved():
    brain = compute_brain_mask(POLICY, skull_stripped())
    prob = np.zeros(brain.shape, np.float32)
    prob[10, 10, 3] = 0.9  # single voxel inside the brain
    out = gate_probability(prob, brain)
    assert out[10, 10, 3] == pytest.approx(0.9) and np.count_nonzero(out) == 1


def test_non_zero_background_case():
    """sub-strokecase0046-like DWI: background stored as -0.000992, not 0."""
    vol = skull_stripped(bg=-0.000992)
    brain = compute_brain_mask(POLICY, vol)
    assert brain.sum() == 12 * 12 * 6
    assert not brain[0, 0, 0]
    naive = vol != 0  # the rejected "== 0" rule would call the whole volume brain
    assert naive.all()


def test_empty_brain_mask_fails():
    with pytest.raises(BrainMaskError):
        compute_brain_mask(POLICY, np.zeros((16, 16, 4), np.float32))


def test_invalid_brain_fraction_fails():
    no_background = np.random.default_rng(1).random((16, 16, 4)).astype(np.float32) + 1
    with pytest.raises(BrainMaskError, match="brain mask covers"):
        compute_brain_mask(POLICY, no_background)


def test_unknown_policy_fails():
    with pytest.raises(BrainMaskError, match="unknown brain mask policy"):
        compute_brain_mask("not_skull_stripped", skull_stripped())


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1.2, -0.1])
def test_gating_rejects_invalid_probability(bad):
    brain = np.ones((4, 4), bool)
    prob = np.full((4, 4), 0.5, np.float32)
    prob[0, 0] = bad
    with pytest.raises(BrainMaskError):
        gate_probability(prob, brain)
    with pytest.raises(BrainMaskError, match="shape"):
        gate_probability(np.zeros((4, 4)), np.ones((4, 5), bool))


def test_gt_outside_brain_fails_for_train_case():
    brain = compute_brain_mask(POLICY, skull_stripped())
    label = np.zeros(brain.shape, np.uint8)
    label[10, 10, 2] = 1
    check_gt_inside_brain(label, brain, "ok")
    label[0, 0, 0] = 1
    with pytest.raises(BrainMaskError, match="outside the brain"):
        check_gt_inside_brain(label, brain, "bad")


@pytest.mark.parametrize("context", [0, 1])  # 2D (2 ch) and 2.5D (6 ch) share the same gating
def test_same_gating_api_for_2d_and_25d(context):
    shape = (16, 12, 5)
    dwi = skull_stripped(shape)
    brain = compute_brain_mask(POLICY, dwi)
    image = np.stack([dwi / 200, dwi / 200]).astype(np.float32)
    meta = {
        "case_id": "c",
        "original": {"spacing": [1, 1, 2]},
        "processed": {"shape": list(shape), "affine": np.eye(4).tolist()},
        "lesion": {"has_lesion": False},
    }
    case = CaseVolumes("c", image, np.zeros(shape, np.uint8), meta, brain)
    cfg = load_config(REPO_ROOT / "configs" / "2d_baseline.yaml")
    cfg["model"].update(channels=[8, 16], strides=[2])
    cfg["data"]["context_slices"] = context
    model = build_model(cfg)
    last = [
        m for m in model.modules() if isinstance(m, torch.nn.Conv2d | torch.nn.ConvTranspose2d)
    ][-1]
    with torch.no_grad():
        last.bias.fill_(30.0)  # every raw probability ~1
    prob, binary, _ = predict_case(model, case, context, torch.device("cpu"), size_multiple=4)
    assert (prob.data[~brain] == 0).all() and (prob.data[brain] > 0.99).all()
    np.testing.assert_array_equal(binary.data.astype(bool), brain)
    assert binary.kind is PredictionKind.BINARY
    _, ungated, _ = predict_case(
        model, case, context, torch.device("cpu"), size_multiple=4, brain_mask_gating=False
    )
    assert ungated.data.all()  # gating is what removed the outside-brain voxels


def test_restoration_requires_thresholded_binary_volume():
    geom = VolumeGeometry((4, 4, 2), np.eye(4))
    prob = PredictionVolume(np.zeros((4, 4, 2), np.float32), geom, PredictionKind.PROBABILITY)
    with pytest.raises(ValueError, match="binary"):
        restore_and_evaluate(prob, np.zeros((4, 4, 2), np.uint8), geom, "c")
