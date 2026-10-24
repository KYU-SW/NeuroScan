"""Phase 4 (2.5D) tests on synthetic data: the 2.5D run must differ from the 2D baseline only in
``data.context_slices``, and every 6-channel code path must keep the Phase 3 contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuroscan_ai.evaluation.comparison import (
    ALLOWED_CONFIG_DIFFERENCES,
    bootstrap_mean_ci,
    config_diff,
    paired_summary,
    paired_table,
)
from neuroscan_ai.training.augmentation import build_train_transform
from neuroscan_ai.training.config import in_channels, load_config
from neuroscan_ai.training.dataset import SliceDataset, stack_context
from neuroscan_ai.training.inference import predict_case
from neuroscan_ai.training.model import build_model
from tests.conftest import REPO_ROOT
from tests.test_training import coded_case, output_conv

CONFIGS = REPO_ROOT / "configs"
BASE_2D = CONFIGS / "2d_baseline.yaml"
BASE_25D = CONFIGS / "25d_baseline.yaml"
TINY_2D = CONFIGS / "2d_tiny_overfit.yaml"
TINY_25D = CONFIGS / "25d_tiny_overfit.yaml"


def small_25d_cfg():
    cfg = load_config(BASE_25D)
    cfg["model"]["channels"] = [8, 16, 32]
    cfg["model"]["strides"] = [2, 2]
    return cfg


# --- fairness: only the context width changes ------------------------------------------------


@pytest.mark.parametrize(("base", "cand"), [(BASE_2D, BASE_25D), (TINY_2D, TINY_25D)])
def test_25d_config_differs_from_2d_only_in_context_and_output_paths(base, cand):
    a, b = load_config(base), load_config(cand)
    assert set(config_diff(a, b)) == ALLOWED_CONFIG_DIFFERENCES
    assert a["data"]["context_slices"] == 0 and b["data"]["context_slices"] == 1
    assert in_channels(a) == 2 and in_channels(b) == 6


def test_25d_outputs_never_point_at_phase3_directories():
    for path in (BASE_25D, TINY_25D):
        cfg = load_config(path)
        assert "phase3" not in cfg["checkpoint"]["dir"]
        assert "phase3" not in cfg["output"]["results_dir"]


def test_config_diff_detects_an_accidental_change():
    a = load_config(BASE_2D)
    b = load_config(BASE_25D)
    b["optimizer"]["lr"] = 5e-4
    assert "optimizer.lr" in set(config_diff(a, b)) - ALLOWED_CONFIG_DIFFERENCES


# --- context stacking / edge replication / dataset shape -------------------------------------


def test_six_channel_order_and_edge_replication_on_every_slice():
    img = coded_case(shape=(16, 12, 5)).image
    n = img.shape[3]
    for z in range(n):
        lo, hi = max(z - 1, 0), min(z + 1, n - 1)
        expected = [lo, z, hi, 100 + lo, 100 + z, 100 + hi]
        np.testing.assert_array_equal(stack_context(img, z, 1)[:, 3, 3], expected)


def test_six_channel_dataset_sample_shape_and_alignment():
    ds = SliceDataset([coded_case(lesion_z=(3,))], context=1, spatial_size=(32, 32))
    for i in range(len(ds)):
        s = ds[i]
        z = s["slice_index"]
        assert s["image"].shape == (6, 32, 32) and s["label"].shape == (1, 32, 32)
        assert s["image"].dtype == torch.float32
        centre = s["image"][[1, 4], 16, 16]  # inside the centred 16x12 field
        torch.testing.assert_close(centre, torch.tensor([z, 100 + z], dtype=torch.float32))
        assert bool(s["label"].any()) == (z == 3)  # label is the centre slice only


def test_brain_free_exclusion_uses_the_centre_slice_only():
    case = coded_case(shape=(16, 12, 5), lesion_z=(2,))
    case.brain[:] = False
    case.brain[:, :, 1:4] = True  # slices 0 and 4 have no brain, but their neighbours do
    ds = SliceDataset([case], context=1, exclude_brain_free_slices=True)
    assert [z for _, z in ds.index] == [1, 2, 3]
    assert ds.n_brain_free_excluded == 2
    assert ds.stats()["n_slices"] == 3


# --- inference consistency and gating --------------------------------------------------------


def test_predict_case_25d_matches_direct_model_call_including_edges():
    case = coded_case(shape=(16, 12, 5))
    case.image[:] = np.random.default_rng(0).normal(size=case.image.shape)
    m = build_model(small_25d_cfg()).eval()
    vol, binary, _ = predict_case(
        m, case, 1, torch.device("cpu"), size_multiple=4, batch_size=2, precision="fp32"
    )
    assert vol.data.shape == case.label.shape
    with torch.no_grad():
        for z in (0, 2, case.n_slices - 1):  # both volume edges and an interior slice
            x = torch.from_numpy(stack_context(case.image, z, 1)[None])
            expected = torch.sigmoid(m(x)).numpy()[0, 0]
            np.testing.assert_allclose(vol.data[:, :, z], expected, atol=1e-6)
    np.testing.assert_array_equal(binary.data, (vol.data >= 0.5).astype(binary.data.dtype))


def test_predict_case_25d_is_independent_of_inference_batch_size():
    case = coded_case(shape=(16, 12, 5))
    case.image[:] = np.random.default_rng(1).normal(size=case.image.shape)
    m = build_model(small_25d_cfg()).eval()
    dev = torch.device("cpu")
    a, _, _ = predict_case(m, case, 1, dev, size_multiple=4, batch_size=1, precision="fp32")
    b, _, _ = predict_case(m, case, 1, dev, size_multiple=4, batch_size=64, precision="fp32")
    np.testing.assert_allclose(a.data, b.data, atol=1e-6)


def test_gating_zeroes_everything_outside_the_centre_slice_brain():
    case = coded_case(shape=(16, 12, 5))
    case.image[:] = np.random.default_rng(2).normal(size=case.image.shape)
    case.brain[:] = False
    case.brain[4:12, 3:9, 1:4] = True
    m = build_model(small_25d_cfg()).eval()
    with torch.no_grad():
        output_conv(m).bias.fill_(50.0)  # model says "lesion" everywhere
    vol, binary, _ = predict_case(m, case, 1, torch.device("cpu"), size_multiple=4)
    assert (vol.data[~case.brain] == 0).all() and (binary.data[~case.brain] == 0).all()
    assert (binary.data[case.brain] == 1).all()
    # slices 0 and 4 have no brain of their own: fully zero even though neighbours have brain
    assert not binary.data[:, :, 0].any() and not binary.data[:, :, 4].any()


# --- augmentation consistency ----------------------------------------------------------------


def test_augmentation_moves_all_six_channels_and_label_identically():
    cfg = load_config(BASE_25D)["augmentation"]
    cfg.update(affine_prob=1.0, intensity_prob=0.0, noise_prob=0.0, rotate_deg=15, translate_px=6)
    t = build_train_transform(cfg, seed=0)
    label = torch.zeros(1, 64, 64)
    label[0, 20:36, 24:44] = 1
    for _ in range(5):
        out = t({"image": label.repeat(6, 1, 1).clone(), "label": label.clone()})
        img, lab = out["image"], out["label"]
        for c in range(1, 6):
            torch.testing.assert_close(img[c], img[0])
        warped = (img[0] > 0.5).float()
        assert float((warped * lab[0]).sum() / ((warped + lab[0]) > 0).sum()) > 0.9
        assert set(torch.unique(lab).tolist()) <= {0.0, 1.0}


def test_augmentation_is_deterministic_for_a_fixed_seed():
    cfg = load_config(BASE_25D)["augmentation"]
    x = {"image": torch.randn(6, 32, 32), "label": (torch.rand(1, 32, 32) > 0.8).float()}
    outs = []
    for _ in range(2):
        t = build_train_transform(cfg, seed=42)
        outs.append([t({k: v.clone() for k, v in x.items()}) for _ in range(3)])
    for a, b in zip(*outs, strict=True):
        torch.testing.assert_close(a["image"], b["image"])
        torch.testing.assert_close(a["label"], b["label"])


# --- paired comparison statistics ------------------------------------------------------------


def _rows(dice, prefix="c"):
    return [
        {"case_id": f"{prefix}{i}", "has_lesion": True, "dice": d, "precision": d, "sensitivity": d}
        for i, d in enumerate(dice)
    ]


def test_paired_table_requires_identical_case_sets():
    with pytest.raises(ValueError, match="case sets differ"):
        paired_table(_rows([0.5, 0.6]), _rows([0.5]), ["dice"])


def test_paired_summary_counts_and_delta():
    base = _rows([0.5, 0.6, 0.7, 0.8])
    cand = _rows([0.6, 0.6, 0.65, 0.9])
    t = paired_table(base, cand, ["dice"])
    s = paired_summary(t, "dice", tie_tolerance=1e-3, n_boot=2000, seed=0)
    assert s["n"] == 4
    assert s["mean_delta"] == pytest.approx((0.1 + 0.0 - 0.05 + 0.1) / 4)
    assert (s["n_improved"], s["n_worse"], s["n_tied"]) == (2, 1, 1)
    assert s["ci95"][0] <= s["mean_delta"] <= s["ci95"][1]


def test_bootstrap_ci_is_seeded_and_brackets_a_constant_shift():
    d = np.full(30, 0.02)
    assert bootstrap_mean_ci(d, n_boot=500, seed=1) == pytest.approx((0.02, 0.02))
    rng = np.random.default_rng(0)
    x = rng.normal(0.03, 0.05, 36)
    assert bootstrap_mean_ci(x, 1000, 7) == bootstrap_mean_ci(x, 1000, 7)
    lo, hi = bootstrap_mean_ci(x, 1000, 7)
    assert lo < x.mean() < hi
