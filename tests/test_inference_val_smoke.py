"""Phase 5A end-to-end smoke test on two *validation* patients (raw ISLES files -> CLI service).

The independent pipeline must reproduce the Phase 4 official validation predictions of the same
checkpoint (D12): CPU runs agree exactly with each other, CUDA differs from the saved Phase 4
prediction in <= 0.1 % of voxels, and Dice against the GT matches validation_metrics.csv within
0.001. GT is read here only to check reproduction; the service never sees it. Test split: never.
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from neuroscan_ai.data.split_guard import SplitGuard
from neuroscan_ai.evaluation.segmentation import evaluate_case
from neuroscan_ai.inference import load_inference_model, run_independent_inference
from neuroscan_ai.preprocessing.mask import binarize
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.data

INF_CFG = yaml.safe_load((REPO_ROOT / "configs" / "inference_v1.yaml").read_text(encoding="utf-8"))
PHASE4_RUN = REPO_ROOT / "results" / "phase4" / "25d_baseline_20261024-192617_34d18df104da"
CKPT = REPO_ROOT / INF_CFG["model"]["checkpoint"]
SPLIT = REPO_ROOT / INF_CFG["split_file"]
CASES = ("sub-strokecase0086", "sub-strokecase0199")  # 2 mm isotropic / XY 1.146 mm resampled

if not (CKPT.is_file() and PHASE4_RUN.is_dir() and (REPO_ROOT / "data" / "processed").is_dir()):
    pytest.skip("Phase 4 checkpoint, results or ISLES data not available", allow_module_level=True)


@pytest.fixture(scope="module")
def guard():
    g = SplitGuard(SPLIT)
    val = set(g.cases("val"))
    assert set(CASES) <= val  # validation patients only
    yield g
    assert g.access_report()["test_cases_accessed"] == []


def _paths(case_id):
    meta = json.loads((REPO_ROOT / "data/processed/v1" / case_id / "meta.json").read_text())
    return {k: REPO_ROOT / v for k, v in meta["source"].items()}


def _model(device):
    m = INF_CFG["model"]
    return load_inference_model(CKPT, REPO_ROOT / m["training_config"], torch.device(device), m)


@pytest.fixture(scope="module")
def cpu_model():
    return _model("cpu")


@pytest.fixture(scope="module")
def phase4_dice():
    df = pd.read_csv(PHASE4_RUN / "validation_metrics.csv").set_index("case_id")
    return df["dice"].to_dict()


def _infer(case_id, out, model, device):
    p = _paths(case_id)
    r = run_independent_inference(p["dwi"], p["adc"], out, device=device, loaded_model=model)
    assert r.status == "succeeded" and r.exit_code == 0, r.error
    assert r.metadata["test_split_guard"]["sealed_test_case_detected"] is False
    run_dir = out / r.case_id / r.run_id
    return r, np.asanyarray(nib.load(run_dir / "prediction_mask.nii.gz").dataobj), run_dir


def _dice(mask, case_id):
    gt_img = nib.load(_paths(case_id)["label"])
    spacing = tuple(float(s) for s in gt_img.header.get_zooms()[:3])
    return evaluate_case(mask, binarize(gt_img.get_fdata()), spacing, case_id)["dice"]


@pytest.mark.parametrize("case_id", CASES)
def test_cpu_runs_are_identical_and_reproduce_phase4(
    case_id, tmp_path, guard, cpu_model, phase4_dice
):
    r1, m1, run_dir = _infer(case_id, tmp_path / "a", cpu_model, "cpu")
    _, m2, _ = _infer(case_id, tmp_path / "b", cpu_model, "cpu")
    np.testing.assert_array_equal(m1, m2)  # D12: same CPU conditions -> identical mask
    src = nib.load(_paths(case_id)["dwi"])
    out = nib.load(run_dir / "prediction_mask.nii.gz")
    assert m1.dtype == np.uint8 and set(np.unique(m1).tolist()) <= {0, 1}
    assert m1.shape == src.shape and np.allclose(out.affine, src.affine, atol=1e-5)
    ref = np.asanyarray(nib.load(PHASE4_RUN / "predictions" / f"{case_id}_pred.nii.gz").dataobj)
    mismatch = float((m1 != ref).mean())
    assert mismatch <= 1e-3, f"CPU vs Phase 4 (CUDA) mismatch {mismatch:.5%}"
    assert abs(_dice(m1, case_id) - phase4_dice[case_id]) <= 1e-3
    q = r1.quantification
    assert q["lesion_detected"] and q["total_lesion_volume_ml"] > 0
    assert q["hemisphere"]["estimated"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("case_id", CASES)
def test_cuda_run_reproduces_phase4_prediction(case_id, tmp_path, guard, phase4_dice):
    r, mask, _ = _infer(case_id, tmp_path, _model("cuda"), "cuda")
    assert r.metadata["device"]["used"] == "cuda"
    ref = np.asanyarray(nib.load(PHASE4_RUN / "predictions" / f"{case_id}_pred.nii.gz").dataobj)
    assert ref.shape == mask.shape
    mismatch = float((mask != ref).mean())
    assert mismatch <= 1e-3, f"CUDA vs Phase 4 mismatch {mismatch:.5%}"  # D12
    assert abs(_dice(mask, case_id) - phase4_dice[case_id]) <= 1e-3
