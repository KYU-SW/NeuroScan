"""Phase 3 training pipeline tests on synthetic data (no ISLES data needed)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from monai.losses import DiceLoss
from monai.networks.nets import UNet

from neuroscan_ai.data.split_guard import SealedSplitAccessError, SplitGuard
from neuroscan_ai.datasets.base import CaseRecord
from neuroscan_ai.preprocessing.pipeline import PreprocessConfig, preprocess_case
from neuroscan_ai.reconstruction.volume import PredictionKind, PredictionVolume, VolumeGeometry
from neuroscan_ai.training.augmentation import build_train_transform
from neuroscan_ai.training.config import in_channels, load_config
from neuroscan_ai.training.dataset import (
    CaseVolumes,
    SliceDataset,
    load_case,
    load_split_cases,
    norm_specs,
    stack_context,
)
from neuroscan_ai.training.inference import (
    load_original_gt,
    predict_case,
    restore_and_evaluate,
)
from neuroscan_ai.training.model import build_loss, build_model
from neuroscan_ai.training.normalization import (
    NormalizationSpec,
    background_value,
    brain_mask,
    normalize,
)
from neuroscan_ai.training.precision import autocast, check_precision
from neuroscan_ai.training.sampler import PositiveRatioSampler
from neuroscan_ai.training.trainer import (
    Trainer,
    TrainingAborted,
    load_model_from_checkpoint,
    save_checkpoint,
    validate,
)
from tests.conftest import REPO_ROOT, las_affine, save_nifti

BASE_CFG = REPO_ROOT / "configs" / "2d_baseline.yaml"


def output_conv(m: torch.nn.Module) -> torch.nn.Module:
    convs = (torch.nn.Conv2d, torch.nn.ConvTranspose2d)
    return [mod for mod in m.modules() if isinstance(mod, convs)][-1]


CLIP = NormalizationSpec("clip_zscore")


def brain_volume(shape=(24, 24, 6), bg=0.0, seed=0):
    rng = np.random.default_rng(seed)
    vol = np.full(shape, bg, np.float32)
    vol[6:18, 5:19, :] = rng.normal(200, 40, (12, 14, shape[2]))
    return vol


def small_cfg(**data_overrides):
    cfg = load_config(BASE_CFG)
    cfg["model"]["channels"] = [8, 16, 32]
    cfg["model"]["strides"] = [2, 2]
    cfg["data"].update(data_overrides)
    return cfg


# --- 1-3 normalisation -----------------------------------------------------------------------


def test_normalization_is_finite_and_zero_outside_brain():
    vol = brain_volume()
    mask = brain_mask(vol)
    for method in ("zscore", "clip_zscore", "robust", "log_clip_zscore"):
        out = normalize(vol, mask, NormalizationSpec(method))
        assert out.dtype == np.float32 and np.isfinite(out).all()
        assert (out[~mask] == 0).all()


def test_adc_unit_scale_cancels_exactly():
    adc = brain_volume(seed=1) * 4.5  # ~900, i.e. 1e-6 mm^2/s storage
    mask = brain_mask(adc)
    ref = normalize(adc, mask, CLIP)
    for scale in (1e-3, 1e-6):  # same tissue stored as 1e-3 mm^2/s or mm^2/s
        np.testing.assert_allclose(normalize(adc * scale, mask, CLIP), ref, atol=1e-4)


def test_dwi_normalization_statistics_and_nonzero_background():
    vol = brain_volume(bg=-0.000992)  # sub-strokecase0046-like scaled background
    assert background_value(vol) == pytest.approx(-0.000992)
    mask = brain_mask(vol)
    assert mask.sum() == 12 * 14 * 6
    out = normalize(vol, mask, CLIP)
    assert abs(out[mask].mean()) < 0.05 and abs(out[mask].std() - 1) < 0.05


def test_brain_mask_fraction_guard():
    with pytest.raises(ValueError, match="brain mask covers"):
        brain_mask(np.random.default_rng(0).random((10, 10, 10)).astype(np.float32) + 1)


# --- dataset helpers -------------------------------------------------------------------------


def coded_case(case_id="c1", shape=(16, 12, 5), lesion_z=(2,)) -> CaseVolumes:
    image = np.zeros((2, *shape), np.float32)
    for m in range(2):
        for z in range(shape[2]):
            image[m, :, :, z] = 100 * m + z  # value encodes modality and slice
    label = np.zeros(shape, np.uint8)
    for z in lesion_z:
        label[4:8, 3:6, z] = 1
    meta = {
        "case_id": case_id,
        "original": {
            "spacing": [2.0, 2.0, 4.8],
            "shape": list(shape),
            "affine": np.eye(4).tolist(),
        },
        "processed": {"shape": list(shape), "affine": np.eye(4).tolist()},
        "lesion": {"has_lesion": bool(lesion_z)},
    }
    return CaseVolumes(case_id, image, label, meta, np.ones(shape, bool))


def test_two_channel_sample_shape_and_metadata():
    ds = SliceDataset([coded_case()], context=0, spatial_size=(32, 32))
    s = ds[2]
    assert s["image"].shape == (2, 32, 32) and s["label"].shape == (1, 32, 32)
    assert s["case_id"] == "c1" and s["slice_index"] == 2
    assert s["z_spacing"] == 4.8 and s["has_lesion"] is True and s["slice_has_lesion"] is True
    assert ds[0]["slice_has_lesion"] is False


def test_dwi_adc_gt_slice_alignment():
    ds = SliceDataset([coded_case(lesion_z=(3,))], context=0)
    for i in range(len(ds)):
        s = ds[i]
        z = s["slice_index"]
        assert torch.all(s["image"][0] == z) and torch.all(s["image"][1] == 100 + z)
        assert bool(s["label"].any()) == (z == 3)


def test_context_stacking_is_modality_major_and_edge_replicated():
    img = coded_case().image
    np.testing.assert_array_equal(stack_context(img, 0, 1)[:, 0, 0], [0, 0, 1, 100, 100, 101])
    np.testing.assert_array_equal(stack_context(img, 4, 1)[:, 0, 0], [3, 4, 4, 103, 104, 104])
    assert in_channels(load_config(BASE_CFG)) == 2


# --- 6-7 split guards ------------------------------------------------------------------------


@pytest.fixture
def split_file(tmp_path):
    f = tmp_path / "split.json"
    f.write_text(
        json.dumps(
            {"version": "s", "splits": {"train": ["a", "b"], "val": ["c"], "test": ["t1", "t2"]}}
        )
    )
    return f


def test_test_split_access_is_sealed(split_file):
    g = SplitGuard(split_file)
    with pytest.raises(SealedSplitAccessError):
        g.cases("test")
    with pytest.raises(SealedSplitAccessError):
        g.assert_no_test(["a", "t2"], "training")
    g.cases("train")
    g.cases("val")
    assert g.access_report()["test_cases_accessed"] == []
    assert SplitGuard(split_file, allow_test=True).cases("test") == ["t1", "t2"]


def test_patient_split_guard_rejects_foreign_cases(split_file, tmp_path):
    g = SplitGuard(split_file)
    with pytest.raises(ValueError, match="not in split"):
        load_split_cases(g, "train", tmp_path, ["dwi", "adc"], {}, case_ids=["c"])
    with pytest.raises(SealedSplitAccessError):
        load_split_cases(g, "test", tmp_path, ["dwi", "adc"], {})


# --- 8 sampler -------------------------------------------------------------------------------


def test_positive_ratio_sampler():
    positive = np.array([True] * 10 + [False] * 990)
    s = PositiveRatioSampler(positive, 0.5, 4000, seed=42)
    idx = np.array(list(s))
    frac = positive[idx].mean()
    assert abs(frac - 0.5) < 0.03 and s.last_positive_fraction == pytest.approx(frac)
    assert (~positive[idx]).sum() > 0 and len(set(idx[~positive[idx]])) > 500  # background kept
    assert list(s) == list(idx)  # same epoch -> same draw
    s.set_epoch(1)
    assert list(s) != list(idx)
    with pytest.raises(ValueError):
        PositiveRatioSampler(positive, 1.0, 10, 0)


# --- 9 augmentation --------------------------------------------------------------------------


def test_augmentation_applies_same_spatial_transform_to_image_and_label():
    cfg = load_config(BASE_CFG)["augmentation"]
    cfg.update(affine_prob=1.0, intensity_prob=0.0, noise_prob=0.0, rotate_deg=15, translate_px=6)
    t = build_train_transform(cfg, seed=0)
    label = torch.zeros(1, 64, 64)
    label[0, 20:36, 24:44] = 1
    for _ in range(5):
        out = t({"image": torch.cat([label, label]).clone(), "label": label.clone()})
        img, lab = out["image"], out["label"]
        torch.testing.assert_close(img[0], img[1])  # DWI and ADC moved identically
        warped = (img[0] > 0.5).float()
        inter = (warped * lab[0]).sum()
        assert float(inter / ((warped + lab[0]) > 0).sum()) > 0.9
        assert set(torch.unique(lab).tolist()) <= {0.0, 1.0}  # nearest for label
    assert build_train_transform({"enabled": False}, 0) is None


# --- 10-11 model / logit policy --------------------------------------------------------------


def test_model_config_is_passed_to_monai_unet():
    cfg = load_config(BASE_CFG)
    m = build_model(cfg)
    assert isinstance(m, UNet)
    assert m.num_res_units == cfg["model"]["num_res_units"] == 0
    assert tuple(m.channels) == tuple(cfg["model"]["channels"])
    assert tuple(m.strides) == tuple(cfg["model"]["strides"])
    assert m.in_channels == 2 and m.out_channels == 1 and m.dimensions == 2
    out = m(torch.randn(2, 2, 64, 64))
    assert out.shape == (2, 1, 64, 64)
    cfg["data"]["context_slices"] = 1  # Phase 4 contract: only the input width changes
    assert build_model(cfg)(torch.randn(1, 6, 64, 64)).shape == (1, 1, 64, 64)


def test_output_logits_are_not_normalised():
    """Regression for the failed tiny overfit: shifting the last bias must shift every logit."""
    m = build_model(small_cfg()).eval()
    last = output_conv(m)
    assert last.out_channels == 1
    after_last = [mod for mod in m.modules()][list(m.modules()).index(last) + 1 :]
    assert not any(
        isinstance(mod, torch.nn.modules.batchnorm._NormBase | torch.nn.PReLU) for mod in after_last
    ), "normalisation/activation after the output conv"
    x = torch.randn(2, 2, 32, 32)
    with torch.no_grad():
        before = m(x)
        last.bias += 5.0
        after = m(x)
    torch.testing.assert_close(after - before, torch.full_like(before, 5.0), atol=1e-4, rtol=0)


def test_loss_takes_logits_and_applies_sigmoid_once():
    loss_fn = build_loss(load_config(BASE_CFG))
    logits = torch.randn(4, 1, 16, 16) * 3
    y = (torch.rand(4, 1, 16, 16) > 0.9).float()
    expected = DiceLoss(sigmoid=True, batch=True)(logits, y) + torch.nn.BCEWithLogitsLoss()(
        logits, y
    )
    torch.testing.assert_close(loss_fn(logits, y), expected)


def test_predict_case_probability_is_sigmoid_of_logits_without_overflow():
    case = coded_case()
    case.image[:] = np.random.default_rng(0).normal(size=case.image.shape)
    m = build_model(small_cfg()).eval()
    with torch.no_grad():
        output_conv(m).bias.fill_(-200.0)  # extreme negative logits: np.exp would overflow
    with np.errstate(over="raise"):
        vol, _, _ = predict_case(m, case, 0, torch.device("cpu"), size_multiple=4, precision="fp32")
    assert vol.kind is PredictionKind.PROBABILITY and vol.data.shape == case.label.shape
    assert vol.data.min() >= 0 and vol.data.max() <= 1
    # semantics unchanged: probability == sigmoid(raw logit) on the processed grid
    x = torch.from_numpy(stack_context(case.image, 2, 0)[None])
    with torch.no_grad():
        expected = torch.sigmoid(m(x)).numpy()[0, 0]  # 16x12 needs no padding at multiple 4
    np.testing.assert_allclose(vol.data[:, :, 2], expected, atol=1e-6)


# --- 12-15 loss / backward / AMP / checkpoint ------------------------------------------------


def test_loss_finite_including_empty_targets_and_backward():
    cfg = small_cfg()
    m, loss_fn = build_model(cfg), build_loss(cfg)
    for y in (torch.zeros(2, 1, 32, 32), (torch.rand(2, 1, 32, 32) > 0.95).float()):
        loss = loss_fn(m(torch.randn(2, 2, 32, 32)), y)
        assert torch.isfinite(loss)
        m.zero_grad()
        loss.backward()
        grads = [p.grad for p in m.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_precision_policy_fp32_official_fp16_forbidden():
    cfg = load_config(BASE_CFG)
    assert cfg["training"]["precision"] == "fp32"
    assert "amp" not in cfg["training"]
    assert check_precision("fp32") == "fp32" and check_precision("bf16") == "bf16"
    with pytest.raises(ValueError, match="forbidden"):
        check_precision("fp16")
    with pytest.raises(ValueError, match="unknown"):
        check_precision("fp8")
    x = torch.randn(2, 4)
    with autocast(torch.device("cpu"), "fp32"):
        assert (x @ x.T).dtype == torch.float32  # fp32: no autocast at all


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp32_training_step_on_gpu_without_gradscaler():
    cfg = small_cfg()
    m, loss_fn = build_model(cfg).cuda(), build_loss(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x, y = torch.randn(4, 2, 64, 64, device="cuda"), torch.zeros(4, 1, 64, 64, device="cuda")
    y[:, :, 20:30, 20:30] = 1
    with autocast(torch.device("cuda"), "fp32"):
        logits = m(x)
    assert logits.dtype == torch.float32
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def _guard_trainer(tmp_path, **guards):
    root, split, ids = synthetic_processed(tmp_path, n_cases=1)
    cfg = small_cfg(processed_root=str(root), train_spatial_size=[48, 48])
    cfg["augmentation"]["enabled"] = False
    cfg["sampler"]["samples_per_epoch"] = 16
    cfg["training"].update(batch_size=8, max_epochs=3)
    cfg["guards"].update(guards)
    g = SplitGuard(split)
    cases = load_split_cases(
        g, "train", root, ["dwi", "adc"], norm_specs(cfg["data"]["normalization"])
    )
    (tmp_path / "run").mkdir()
    return Trainer(cfg, g, cases, cases, tmp_path / "run", tmp_path / "m", "g", REPO_ROOT)


def test_gradient_norm_guard_aborts(tmp_path):
    tr = _guard_trainer(tmp_path, max_grad_norm=1e-12)
    with pytest.raises(TrainingAborted, match="gradient norm"):
        tr.fit(3, None)


def test_non_finite_gradient_guard_aborts(tmp_path):
    tr = _guard_trainer(tmp_path)
    with torch.no_grad():
        next(tr.model.parameters()).fill_(float("nan"))
    with pytest.raises(TrainingAborted, match="non-finite"):
        tr.fit(1, None)


def test_grad_norm_is_logged_per_epoch(tmp_path):
    tr = _guard_trainer(tmp_path)
    tr.fit(2, None)
    log = pd.read_csv(tmp_path / "run" / "training_log.csv")
    assert {"grad_norm_mean", "grad_norm_max"} <= set(log.columns)
    assert (log.grad_norm_max >= log.grad_norm_mean).all() and (log.grad_norm_mean > 0).all()


@pytest.mark.parametrize("bias,kind", [(-50.0, "all-zero"), (50.0, "all-one")])
def test_prediction_collapse_guard(tmp_path, bias, kind):
    tr = _guard_trainer(tmp_path, collapse_patience_epochs=2)
    with torch.no_grad():
        output_conv(tr.model).bias.fill_(bias)
    streak = {"zero": 0, "one": 0}
    val = validate(tr.model, tr.val_cases, tr.val_gts, tr.cfg, tr.device, None)
    tr.check_collapse(val, streak, 1)
    with pytest.raises(TrainingAborted, match=kind):
        tr.check_collapse(val, streak, 2)


def test_checkpoint_save_load_roundtrip(tmp_path):
    cfg = small_cfg()
    m = build_model(cfg)
    opt = torch.optim.AdamW(m.parameters())
    save_checkpoint(tmp_path / "best.pt", m, opt, None, {"run_id": "r", "epoch": 3})
    m2, info = load_model_from_checkpoint(tmp_path / "best.pt", cfg, torch.device("cpu"))
    x = torch.randn(1, 2, 32, 32)
    m.eval()
    m2.eval()
    torch.testing.assert_close(m(x), m2(x))
    assert info == {"run_id": "r", "epoch": 3}


def test_dataset_and_sampler_ordering_deterministic():
    cases = [coded_case("a"), coded_case("b", lesion_z=(1, 4))]
    a, b = SliceDataset(cases), SliceDataset(cases)
    assert a.index == b.index == [(c, z) for c in range(2) for z in range(5)]
    s1 = PositiveRatioSampler(a.slice_positive, 0.5, 50, seed=42)
    s2 = PositiveRatioSampler(b.slice_positive, 0.5, 50, seed=42)
    assert list(s1) == list(s2)


# --- 17-20 synthetic end-to-end ---------------------------------------------------------------


def synthetic_processed(tmp_path: Path, n_cases=2):
    """Raw synthetic patients where the lesion is bright on DWI / dark on ADC -> processed/v1."""
    rng = np.random.default_rng(0)
    out_root = tmp_path / "processed"
    ids = []
    for i in range(n_cases):
        cid = f"syn{i}"
        shape = (20, 20, 6)
        aff = las_affine((2.0, 2.0, 4.0))
        brain = np.zeros(shape, bool)
        brain[3:17, 3:17, :] = True
        lesion = np.zeros(shape)
        lesion[6 + i : 11 + i, 7:12, 2:4] = 1.0
        dwi = np.where(brain, rng.normal(100, 5, shape) + 80 * lesion, 0)
        adc = np.where(brain, rng.normal(800, 40, shape) - 400 * lesion, 0)
        d = tmp_path / "raw" / cid
        rec = CaseRecord(
            "synthetic",
            cid,
            {
                "dwi": save_nifti(d / "dwi.nii.gz", dwi, aff),
                "adc": save_nifti(d / "adc.nii.gz", adc, aff),
            },
            label=save_nifti(d / "msk.nii.gz", lesion, aff),
        )
        preprocess_case(rec, PreprocessConfig("vtest", (1.0, 1.0, None)), out_root)
        ids.append(cid)
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"version": "syn", "splits": {"train": ids, "val": ids, "test": ["x"]}})
    )
    return out_root, split, ids


def test_tiny_overfit_pipeline_and_original_space_evaluation(tmp_path):
    root, split, ids = synthetic_processed(tmp_path)
    cfg = small_cfg(processed_root=str(root), train_spatial_size=[48, 48])
    cfg["augmentation"]["enabled"] = False
    cfg["sampler"]["samples_per_epoch"] = 128
    cfg["scheduler"]["name"] = "none"
    cfg["training"].update(batch_size=16, max_epochs=40)
    cfg["inference"]["size_multiple"] = 16
    cfg["checkpoint"]["dir"] = str(tmp_path / "models")
    guard = SplitGuard(split)
    specs = norm_specs(cfg["data"]["normalization"])
    cases = load_split_cases(guard, "train", root, ["dwi", "adc"], specs)
    torch.manual_seed(0)
    tr = Trainer(cfg, guard, cases, cases, tmp_path / "run", tmp_path / "models", "t", REPO_ROOT)
    (tmp_path / "run").mkdir()
    fit = tr.fit(cfg["training"]["max_epochs"], None)
    assert fit["best_val_mean_dice"] > 0.8

    model, info = load_model_from_checkpoint(tmp_path / "models" / "best.pt", cfg, tr.device)
    assert info["epoch"] == fit["best_epoch"] and info["in_channels"] == 2
    res = validate(model, cases, tr.val_gts, cfg, tr.device, tr.loss_fn, keep_predictions=True)
    assert res.mean_dice == pytest.approx(fit["best_val_mean_dice"], abs=1e-6)
    for c in cases:  # predictions are on the ORIGINAL grid (20x20x6), not processed (40x40x6)
        gt, geom = load_original_gt(c.meta, REPO_ROOT)
        assert res.predictions[c.case_id].shape == gt.shape == geom.shape == (20, 20, 6)
    assert guard.access_report()["test_cases_accessed"] == []


def test_reconstruction_and_restoration_integration(tmp_path):
    root, _, ids = synthetic_processed(tmp_path, n_cases=1)
    case = load_case(
        root, ids[0], ["dwi", "adc"], norm_specs(load_config(BASE_CFG)["data"]["normalization"])
    )
    gt, geom = load_original_gt(case.meta, REPO_ROOT)
    perfect = PredictionVolume(
        case.label.copy(), VolumeGeometry.from_meta(case.meta, "processed"), PredictionKind.BINARY
    )
    row, binary = restore_and_evaluate(perfect, gt, geom, case.case_id)
    assert binary.shape == gt.shape and row["dice"] == 1.0  # 2 mm -> 1 mm -> 2 mm is exact


def test_negative_validation_case_handling(tmp_path):
    case = coded_case("neg", lesion_z=())
    case.image[:] = 0.0
    gt = np.zeros(case.label.shape, np.uint8)
    geom = VolumeGeometry.from_meta(case.meta, "processed")
    cfg = small_cfg()
    cfg["inference"]["size_multiple"] = 4
    m = build_model(cfg).eval()
    last = output_conv(m)
    for bias, outcome in ((-50.0, "correct_negative"), (50.0, "false_positive_case")):
        with torch.no_grad():
            last.bias.fill_(bias)
        res = validate(m, [case], {"neg": (gt, geom)}, cfg, torch.device("cpu"), None)
        r = res.rows[0]
        assert r["negative_outcome"] == outcome and r["dice"] is None
        assert res.negative["n_cases"] == 1 and res.positive["n_cases"] == 0
        assert (r["false_positive_volume_ml"] > 0) == (outcome == "false_positive_case")
        assert math.isclose(res.mean_dice, 0.0)


# --- brain-free slice exclusion (train sampling only) -----------------------------------------


def brain_free_case() -> CaseVolumes:
    """z=0 and z=4 brain-free; z=1 tiny lesion (1 voxel); z=2 lesion; z=3 brain, no lesion."""
    case = coded_case("bf", shape=(16, 12, 5), lesion_z=(2,))
    case.brain = np.zeros(case.label.shape, bool)
    case.brain[2:14, 2:10, 1:4] = True
    case.label[8, 6, 1] = 1  # tiny lesion inside the brain
    return case


def test_brain_free_slices_excluded_only_from_training_dataset():
    case = brain_free_case()
    ds = SliceDataset([case], exclude_brain_free_slices=True)
    kept = [z for _, z in ds.index]
    assert kept == [1, 2, 3]  # only brain-free z=0 and z=4 dropped
    assert ds.n_slices_total == 5 and ds.n_brain_free_excluded == 2
    stats = ds.stats()
    assert stats["brain_free_slices_excluded"] == 2
    assert stats["brain_free_excluded_fraction"] == pytest.approx(0.4)
    assert stats["positive_slices"] == 2 and stats["brain_positive_negative_slices"] == 1
    assert ds[0]["brain_voxels"] > 0
    assert [z for _, z in SliceDataset([case]).index] == [0, 1, 2, 3, 4]  # default keeps all


def test_brain_positive_negative_and_tiny_lesion_slices_are_kept():
    ds = SliceDataset([brain_free_case()], exclude_brain_free_slices=True)
    by_z = {z: i for i, (_, z) in enumerate(ds.index)}
    assert 3 in by_z and not ds.slice_positive[by_z[3]]  # lesion-negative brain slice kept
    assert 1 in by_z and ds.slice_positive[by_z[1]]  # 1-voxel lesion slice kept


def test_exclusion_requires_brain_mask():
    case = coded_case()
    case.brain = None
    with pytest.raises(ValueError, match="brain mask"):
        SliceDataset([case], exclude_brain_free_slices=True)


def test_validation_inference_still_covers_every_slice():
    case = brain_free_case()
    m = build_model(small_cfg()).eval()
    prob, binary, _ = predict_case(m, case, 0, torch.device("cpu"), size_multiple=4)
    assert prob.data.shape == case.label.shape  # all 5 slices reconstructed
    assert (prob.data[:, :, 0] == 0).all() and (prob.data[:, :, 4] == 0).all()  # gated
