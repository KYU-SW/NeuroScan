"""Phase 5A: preprocessing equivalence, model loading, prediction, restoration, quantification."""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
import torch
import yaml

from neuroscan_ai.datasets.base import CaseRecord
from neuroscan_ai.inference import predict as predict_mod
from neuroscan_ai.inference.errors import Code, InferenceError
from neuroscan_ai.inference.model_loader import load_inference_model, resolve_device
from neuroscan_ai.inference.predict import Prediction, predict
from neuroscan_ai.inference.preprocess import load_preprocess_config, prepare_case
from neuroscan_ai.inference.quantify import hemisphere_estimate, quantify
from neuroscan_ai.inference.validation import validate_inputs
from neuroscan_ai.preprocessing.pipeline import preprocess_case
from neuroscan_ai.reconstruction.volume import PredictionKind, PredictionVolume, VolumeGeometry
from neuroscan_ai.training.config import load_config
from neuroscan_ai.training.dataset import load_case, norm_specs
from tests.conftest import REPO_ROOT, las_affine
from tests.inference_helpers import (
    BASE_25D,
    INFERENCE_CFG,
    tiny_model_setup,
    write_case,
    write_nifti,
)

PRE_CFG = REPO_ROOT / "configs" / "preprocess_v1.yaml"
INF = yaml.safe_load(INFERENCE_CFG.read_text(encoding="utf-8"))
CPU = torch.device("cpu")


# --- D2: in-memory preprocessing == Phase 1 preprocess_case + Phase 3 load_case ----------------


@pytest.mark.parametrize("spacing", [(2.0, 2.0, 2.0), (1.8, 1.8, 4.8), (1.1458, 1.1458, 4.8)])
def test_in_memory_preprocessing_equals_disk_pipeline(tmp_path, spacing):
    dwi_p, adc_p, label, aff = write_case(tmp_path / "in", spacing=spacing)
    label_p = write_nifti(tmp_path / "in" / "msk.nii.gz", label.astype(np.float32), aff)
    pre = load_preprocess_config(PRE_CFG)
    record = CaseRecord("synthetic", "c1", {"dwi": dwi_p, "adc": adc_p}, label_p)
    meta = preprocess_case(record, pre, tmp_path / "proc")
    cfg = load_config(BASE_25D)
    disk = load_case(
        tmp_path / "proc",
        "c1",
        ["dwi", "adc"],
        norm_specs(cfg["data"]["normalization"]),
        cfg["data"]["brain_mask_policy"],
    )
    report = validate_inputs(dwi_p, adc_p, INF)
    mem = prepare_case("c1", report.dwi.data, report.adc.data, report.dwi.affine, pre, cfg)
    np.testing.assert_array_equal(mem.case.image, disk.image)  # bit-identical model input
    np.testing.assert_array_equal(mem.case.brain, disk.brain)
    assert mem.processed.shape == tuple(meta["processed"]["shape"])
    np.testing.assert_array_equal(mem.processed.affine, np.asarray(meta["processed"]["affine"]))
    assert mem.original.shape == tuple(meta["original"]["shape"])
    np.testing.assert_allclose(
        mem.original.affine, np.asarray(meta["original"]["affine"]), atol=1e-6
    )
    assert VolumeGeometry.from_meta(mem.case.meta, "processed").shape == mem.processed.shape


def test_placeholder_label_is_zero_and_meta_has_no_ground_truth(tmp_path):
    dwi_p, adc_p, _, _ = write_case(tmp_path)
    report = validate_inputs(dwi_p, adc_p, INF)
    mem = prepare_case(
        "c1",
        report.dwi.data,
        report.adc.data,
        report.dwi.affine,
        load_preprocess_config(PRE_CFG),
        load_config(BASE_25D),
    )
    assert not mem.case.label.any()
    assert "lesion" not in mem.case.meta and "source" not in mem.case.meta
    assert set(mem.case.meta) == {"case_id", "original", "processed"}


# --- checkpoint / config verification --------------------------------------------------------


def _expected(inf_path):
    return yaml.safe_load(inf_path.read_text(encoding="utf-8"))["model"]


def test_loader_accepts_matching_checkpoint_with_weights_only(tmp_path):
    inf, ckpt, cfg_path, _ = tiny_model_setup(tmp_path)
    m = load_inference_model(ckpt, cfg_path, CPU, _expected(inf))
    assert not m.model.training and len(m.checkpoint_sha256) == 64
    assert m.info["in_channels"] == 6


@pytest.mark.parametrize(
    "override",
    [{"config_hash": "000000000000"}, {"in_channels": 2}, {"preprocessing_version": "v2"}],
)
def test_loader_rejects_mismatched_checkpoint_info(tmp_path, override):
    inf, ckpt, cfg_path, _ = tiny_model_setup(tmp_path, **override)
    with pytest.raises(InferenceError) as e:
        load_inference_model(ckpt, cfg_path, CPU, _expected(inf))
    assert e.value.issue.code == Code.CONFIG_MISMATCH


def test_loader_rejects_2d_config_and_unexpected_hash(tmp_path):
    inf, ckpt, cfg_path, cfg = tiny_model_setup(tmp_path)
    exp = _expected(inf)
    with pytest.raises(InferenceError) as e:
        load_inference_model(ckpt, cfg_path, CPU, {**exp, "expected_config_hash": "34d18df104da"})
    assert e.value.issue.code == Code.CONFIG_MISMATCH
    cfg["data"]["context_slices"] = 0
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(InferenceError) as e:
        load_inference_model(ckpt, cfg_path, CPU, exp)
    assert e.value.issue.code == Code.CONFIG_MISMATCH


def test_loader_refuses_pickled_objects_and_corrupt_files(tmp_path):
    inf, ckpt, cfg_path, _ = tiny_model_setup(tmp_path)
    exp = _expected(inf)
    obj = torch.load(ckpt, weights_only=True)
    obj["info"]["evil"] = pathlib.PurePosixPath("/x")  # arbitrary class: not allowed
    torch.save(obj, ckpt)
    with pytest.raises(InferenceError) as e:
        load_inference_model(ckpt, cfg_path, CPU, exp)
    assert e.value.issue.code == Code.CHECKPOINT
    ckpt.write_bytes(b"garbage")
    with pytest.raises(InferenceError) as e:
        load_inference_model(ckpt, cfg_path, CPU, exp)
    assert e.value.issue.code == Code.CHECKPOINT


def test_device_policy(monkeypatch):
    assert resolve_device("cpu") == (CPU, [])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dev, issues = resolve_device("auto")
    assert dev == CPU and [i.code for i in issues] == [Code.W_CPU_FALLBACK]
    with pytest.raises(InferenceError) as e:
        resolve_device("cuda")
    assert e.value.issue.code == Code.DEVICE


# --- prediction, OOM retry, restoration ------------------------------------------------------


def _prepared_and_model(tmp_path, bias=None, affine=None, spacing=(1.8, 1.8, 4.8)):
    d, a, _, _ = write_case(tmp_path / "in", spacing=spacing, affine=affine)
    report = validate_inputs(d, a, INF)
    inf, ckpt, cfg_path, cfg = tiny_model_setup(tmp_path, bias=bias)
    loaded = load_inference_model(ckpt, cfg_path, CPU, _expected(inf))
    prepared = prepare_case(
        "c1",
        report.dwi.data,
        report.adc.data,
        report.dwi.affine,
        load_preprocess_config(PRE_CFG),
        cfg,
    )
    return prepared, loaded


def _oblique(deg):
    th = math.radians(deg)
    rot = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    aff = las_affine((1.8, 1.8, 4.8))
    aff[:3, :3] = rot @ aff[:3, :3]
    return aff


@pytest.mark.parametrize(
    "affine", [None, np.diag([1.8, 1.8, 4.8, 1.0]), _oblique(5)], ids=["LAS", "RAS", "oblique"]
)
def test_prediction_is_restored_onto_the_exact_input_grid(tmp_path, affine):
    prepared, loaded = _prepared_and_model(tmp_path, bias=50.0, affine=affine)
    pred = predict(loaded, prepared, allow_cpu_fallback=False)
    out = pred.binary_original
    assert out.data.shape == prepared.original.shape and out.data.dtype == np.uint8
    np.testing.assert_array_equal(out.geometry.affine, prepared.original.affine)
    assert set(np.unique(out.data).tolist()) == {0, 1}
    assert pred.probability_processed.kind is PredictionKind.PROBABILITY
    assert (pred.probability_processed.data[~prepared.case.brain] == 0).all()  # gated


def test_prediction_is_deterministic_on_cpu(tmp_path):
    prepared, loaded = _prepared_and_model(tmp_path)
    a = predict(loaded, prepared, False)
    b = predict(loaded, prepared, False)
    np.testing.assert_array_equal(a.binary_original.data, b.binary_original.data)
    np.testing.assert_array_equal(a.probability_processed.data, b.probability_processed.data)


def test_oom_retries_with_smaller_batches_then_fails_without_fallback(tmp_path, monkeypatch):
    prepared, loaded = _prepared_and_model(tmp_path)
    real, calls = predict_mod._run, []

    def flaky(lm, pc, dev, bs):
        calls.append(bs)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(lm, pc, dev, bs)

    monkeypatch.setattr(predict_mod, "_run", flaky)
    pred = predict(loaded, prepared, False)
    assert calls == [64, 16] and Code.W_OOM_RETRY in [w.code for w in pred.warnings]

    def always(lm, pc, dev, bs):
        raise torch.cuda.OutOfMemoryError("simulated")

    monkeypatch.setattr(predict_mod, "_run", always)
    with pytest.raises(InferenceError) as e:
        predict(loaded, prepared, False)
    assert e.value.issue.code == Code.INFERENCE_OOM


# --- quantification --------------------------------------------------------------------------


def _fake_prediction(prepared, mask_original):
    shape = prepared.processed.shape
    return Prediction(
        PredictionVolume(
            np.zeros(shape, np.float32), prepared.processed, PredictionKind.PROBABILITY
        ),
        PredictionVolume(np.zeros(shape, np.uint8), prepared.processed, PredictionKind.BINARY),
        PredictionVolume(
            np.asarray(mask_original).astype(np.uint8), prepared.original, PredictionKind.BINARY
        ),
        "cpu",
        64,
    )


def test_volume_components_bbox_and_centre(tmp_path):
    prepared, _ = _prepared_and_model(tmp_path, spacing=(2.0, 2.0, 4.0))
    mask = np.zeros(prepared.original.shape, np.uint8)
    mask[10:13, 20:23, 4:6] = 1  # 18 voxels x 16 mm^3 = 0.288 mL
    mask[30:32, 20:22, 4:5] = 1  # 4 voxels = 0.064 mL
    mask[20, 30, 3] = mask[21, 31, 4] = 1  # corner-touching: one 26-connected component
    q, _ = quantify(_fake_prediction(prepared, mask), prepared, INF, 0.5)
    assert q["lesion_detected"] and q["voxel_count"] == 24
    assert q["total_lesion_volume_ml"] == pytest.approx(24 * 16 / 1000)
    assert q["component_count"] == 3
    assert [c["voxel_count"] for c in q["components"]] == [18, 4, 2]  # largest first
    assert q["largest_component_volume_ml"] == pytest.approx(0.288)
    assert q["bounding_box_voxel"] == {"min": [10, 20, 3], "max": [31, 31, 5]}
    aff = prepared.original.affine
    corners = [aff @ np.array([i, j, k, 1.0]) for i in (10, 31) for j in (20, 31) for k in (3, 5)]
    np.testing.assert_allclose(q["bounding_box_world_mm"]["min"], np.min(corners, 0)[:3], atol=1e-3)
    com = np.argwhere(mask).mean(0)
    np.testing.assert_allclose(q["center_of_mass_voxel"], com, atol=1e-3)
    np.testing.assert_allclose(q["center_of_mass_world_mm"], (aff @ [*com, 1])[:3], atol=1e-3)
    assert q["affected_slice_count"] == 3 and q["vascular_territory"] is None


def test_empty_mask_is_a_valid_result(tmp_path):
    prepared, _ = _prepared_and_model(tmp_path)
    empty = np.zeros(prepared.original.shape)
    q, issues = quantify(_fake_prediction(prepared, empty), prepared, INF, 0.5)
    assert not q["lesion_detected"] and q["total_lesion_volume_ml"] == 0
    assert q["component_count"] == 0 and q["bounding_box_voxel"] is None
    assert q["research_score_summary"]["predicted_region"] is None and issues == []


@pytest.mark.parametrize(("flip", "near_low_index"), [(False, "right"), (True, "left")])
def test_hemisphere_follows_world_coordinates_not_array_index(tmp_path, flip, near_low_index):
    aff = las_affine((1.8, 1.8, 4.8))  # LAS: array axis 0 increases toward patient Left
    if flip:
        aff = aff @ np.diag([-1.0, 1, 1, 1])  # RAS: axis 0 increases toward patient Right
    prepared, _ = _prepared_and_model(tmp_path, affine=aff)
    mask = np.zeros(prepared.original.shape, np.uint8)
    mask[12:15, 20:24, 4:6] = 1  # low array index on axis 0, away from the midline
    q, _ = quantify(_fake_prediction(prepared, mask), prepared, INF, 0.5)
    h = q["hemisphere"]
    assert h["status"] == "estimated" and h["estimated"] is True
    other = "left" if near_low_index == "right" else "right"
    assert h[f"{near_low_index}_volume_ml"] > 0 and h[f"{other}_volume_ml"] == 0
    assert q["components"][0]["hemisphere_estimate"] == near_low_index
    assert h["bilateral"] is False


def test_hemisphere_not_available_when_too_oblique(tmp_path):
    prepared, _ = _prepared_and_model(tmp_path, affine=_oblique(15))
    mask = np.zeros(prepared.original.shape, np.uint8)
    mask[12:15, 20:24, 4:6] = 1
    q, _ = quantify(_fake_prediction(prepared, mask), prepared, INF, 0.5)
    assert q["hemisphere"]["status"] == "not_available"
    assert q["hemisphere"]["left_volume_ml"] is None
    assert q["components"][0]["hemisphere_estimate"] == "not_available"


def test_hemisphere_not_available_when_midline_unverified(tmp_path):
    prepared, _ = _prepared_and_model(tmp_path)
    brain = np.zeros(prepared.processed.shape, bool)
    brain[:20, 30:60, :] = True
    brain[:5, 30:90, :] = True  # strongly lopsided "brain"
    h, _ = hemisphere_estimate(
        np.zeros(prepared.original.shape, np.uint8),
        prepared.original.affine,
        (1.8, 1.8, 4.8),
        brain,
        prepared.processed.affine,
        [],
        INF["quantification"]["hemisphere"],
    )
    assert h["status"] == "not_available" and "midline" in h["reason"]


def test_technical_warnings_do_not_change_the_mask(tmp_path):
    prepared, _ = _prepared_and_model(tmp_path, spacing=(2.0, 2.0, 4.0))
    mask = np.zeros(prepared.original.shape, np.uint8)
    mask[8:40:3, 8:40:3, 2:8:3] = 1  # many isolated voxels (> 50 components)
    before = mask.copy()
    q, issues = quantify(_fake_prediction(prepared, mask), prepared, INF, 0.5)
    assert Code.W_MANY_COMPONENTS in [i.code for i in issues] and q["component_count"] > 50
    np.testing.assert_array_equal(mask, before)
    big = np.zeros(prepared.original.shape, np.uint8)
    big[4:44, 4:44, :] = 1
    q, issues = quantify(_fake_prediction(prepared, big), prepared, INF, 0.5)
    assert Code.W_LARGE_LESION in [i.code for i in issues]
    assert q["total_lesion_volume_ml"] == pytest.approx(big.sum() * 16 / 1000)
