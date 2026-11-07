"""v2 2D training runner through the run-control lifecycle, on SYNTHETIC volumes only (CPU, tiny
model, throwaway git repo). No MRI, no ISLES / SOOP access, no v1 code."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from neuroscan_ai.reproducibility import CASE_KEY_POLICY, PathRoots
from neuroscan_ai.reproducibility.environment import (
    capture_determinism,
    capture_environment,
    capture_source_control,
)
from neuroscan_ai.reproducibility.hashing import alias_list_sha256, sha256_file, sha256_text_lf
from neuroscan_ai.run_control import (
    COMPLETED,
    FAILED_RUNNER,
    GateError,
    RunPlanError,
    build_plan,
    load_post_run,
    run_controlled,
    verify_run,
)
from neuroscan_ai.run_control.outputs import case_path, run_output_path
from neuroscan_ai.v2 import MODEL_ID, CaseFiles, V2TwoDRunner
from neuroscan_ai.v2.dataset import CaseDataError, brain_mask, load_case

REPO = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", "user.name=synthetic", "-c", "user.email=synthetic@example.invalid"]
TRAIN = ["syn-train-01", "syn-train-02"]
VAL = ["syn-val-01", "syn-val-02"]
SHAPE = (24, 20, 5)
POLICIES = {
    "preprocessing": "preprocess_v2_isles_smoke_v1",
    "inference": "v2_2d_slicewise_sigmoid_v1",
    "threshold": "fixed_0.5_smoke_not_validated",
    "brain_mask": "skull_stripped_background_v2_smoke",
    "combination": "single_model_2d",
}


def _cfg(seed=42):
    return {
        "device": "cpu",
        "model": {"base_channels": 4},
        "training": {
            "epochs": 1,
            "seed": seed,
            "batch_size": 4,
            "optimizer": "adam",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "dataloader_workers": 0,
        },
        "inference": {"threshold": 0.5, "pad_multiple": 16, "batch_size": 4},
    }


def _case(folder: Path, idx: int, affine=None, bg=0.0) -> CaseFiles:
    rng = np.random.default_rng(100 + idx)
    g = np.indices(SHAPE).astype(float)
    brain = ((g[0] - 11.5) / 9) ** 2 + ((g[1] - 9.5) / 8) ** 2 + ((g[2] - 2) / 3) ** 2 <= 1
    lesion = brain & (((g[0] - 9) / 3) ** 2 + ((g[1] - 8) / 3) ** 2 <= 1)
    dwi = np.full(SHAPE, bg, np.float32)
    dwi[brain] = 300 + rng.normal(0, 20, int(brain.sum()))
    dwi[lesion] += 400
    adc = np.where(brain, 800 + rng.normal(0, 50, SHAPE), 0).astype(np.float32)
    adc[lesion] -= 400
    aff = np.diag([1.0, 1.0, 3.0, 1.0]) if affine is None else affine
    files = CaseFiles(
        folder / f"c{idx}_dwi.nii.gz", folder / f"c{idx}_adc.nii.gz", folder / f"c{idx}_msk.nii.gz"
    )
    for p, arr in ((files.dwi, dwi), (files.adc, adc), (files.label, lesion.astype(np.uint8))):
        nib.save(nib.Nifti1Image(arr, aff), str(p))
    return files


@pytest.fixture
def w(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([*GIT, "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / ".gitignore").write_text("results/\ndata/\n", encoding="utf-8")
    (repo / "cfg.yaml").write_text("synthetic: true\n", encoding="utf-8")
    (repo / "requirements-lock.txt").write_text("torch==synthetic\n", encoding="utf-8")
    subprocess.run([*GIT, "-C", str(repo), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    data = repo / "data"
    data.mkdir()
    splits = {
        "splits": {"train": TRAIN, "val": VAL, "test": ["syn-test-01"], "pilot": ["soop-pilot-01"]}
    }
    (data / "split.json").write_text(json.dumps(splits), encoding="utf-8")
    files = {a: _case(data, i) for i, a in enumerate(TRAIN + VAL)}
    return SimpleNamespace(
        repo=repo, data=data, files=files, roots=PathRoots(repo, repo / "models", data)
    )


def _env(w):
    return capture_environment(torch_module=torch, lock_file=w.repo / "requirements-lock.txt")


def _plan(w, run_id="syn-run-a", **kw):
    args = {
        "run_id": run_id,
        "purpose": "synthetic",
        "stage": "train",
        "dataset_id": "synthetic",
        "split_name": "train",
        "aliases": TRAIN,
        "allowed_case_count": len(TRAIN),
        "validation_aliases": VAL,
        "model_id": MODEL_ID,
        "checkpoint": None,
        "config": w.repo / "cfg.yaml",
        "split_file": w.data / "split.json",
        "policies": POLICIES,
        "output_root": w.repo / "results/v2_runs",
        "expected": {
            "config_sha256_lf": sha256_text_lf(w.repo / "cfg.yaml"),
            "split_sha256_lf": sha256_text_lf(w.data / "split.json"),
            "package_lock_sha256_lf": _env(w)["package_lock_sha256_lf"],
            "alias_list_sha256": alias_list_sha256(TRAIN + VAL, CASE_KEY_POLICY),
        },
    }
    args.update(kw)
    return build_plan(**args)


def _run(w, plan, seed=42, files=None):
    runner = V2TwoDRunner(_cfg(seed), files or w.files)
    det = capture_determinism(
        torch, seed=None, precision="fp32", amp=False, workers=0, batch_size=4
    )
    return run_controlled(
        plan,
        runner,
        repo_root=w.repo,
        roots=w.roots,
        source_control=capture_source_control(w.repo),
        environment=_env(w),
        determinism=det,
    )


def _hashes(run_dir):
    return {k: v for k, v in load_post_run(run_dir)["outputs"].items() if v}


# ---------------------------------------------------------------- isolation and access
def test_no_v1_imports():
    prefixes = (
        "neuroscan_ai.training",
        "neuroscan_ai.inference",
        "neuroscan_ai.preprocessing",
        "neuroscan_ai.analysis",
        "neuroscan_ai.final_test",
        "monai",
    )
    code = (
        "import sys; sys.path.insert(0, 'ai'); import neuroscan_ai.v2\n"
        f"print([m for m in sys.modules if m.startswith({prefixes!r})])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
    src = "\n".join(
        p.read_text(encoding="utf-8") for p in (REPO / "ai/neuroscan_ai/v2").glob("*.py")
    )
    for forbidden in (
        "neuroscan_ai.training",
        "neuroscan_ai.inference",
        "neuroscan_ai.preprocessing",
        "soop",
        "pretrained=True",
        "final_test",
    ):
        assert forbidden not in src


def test_runner_refuses_without_gate(w):
    plan = _plan(w)
    with pytest.raises(GateError):
        V2TwoDRunner(_cfg(), w.files).run(None, plan)
    assert not plan.run_dir.exists()


@pytest.mark.parametrize("split", ["test", "final_test"])
def test_test_split_refused(w, split):
    with pytest.raises(RunPlanError):
        _plan(w, split_name=split, aliases=["syn-test-01"], allowed_case_count=1)


def test_validation_cases_must_be_val(w):
    with pytest.raises(RunPlanError):
        _plan(w, validation_aliases=["syn-test-01"])


def test_soop_pilot_refused(w):
    with pytest.raises(RunPlanError):
        _plan(w, validation_aliases=["soop-pilot-01"])


# ---------------------------------------------------------------- data policy
def test_brain_mask_and_grid_checks(w, tmp_path):
    case = load_case(w.files["syn-val-01"])
    assert case.image.shape == (2, *SHAPE) and case.brain.any() and not case.brain.all()
    with pytest.raises(CaseDataError):
        brain_mask(np.ones(SHAPE, np.float32))  # no background -> fraction guard
    bad = _case(tmp_path, 9)
    nib.save(
        nib.Nifti1Image(np.zeros(SHAPE, np.float32), np.diag([2.0, 1.0, 3.0, 1.0])), str(bad.adc)
    )
    with pytest.raises(CaseDataError):
        load_case(bad)


# ---------------------------------------------------------------- full synthetic run
def test_full_run_outputs_and_relations(w):
    plan = _plan(w)
    res = _run(w, plan)
    assert res["status"] == COMPLETED, res
    for label in ("checkpoint", "training_history", "validation_metrics", "threshold_record"):
        assert run_output_path(plan.run_dir, label).is_file()
    keys = sorted(load_post_run(plan.run_dir)["outputs"]["probability_maps"])
    assert keys == ["case_0003", "case_0004"]  # validation cases sort after the train aliases
    for key in keys:
        for kind in ("probability_maps", "ungated_predictions", "brain_masks", "gated_predictions"):
            assert case_path(plan.run_dir, kind, key).is_file()
        prob = np.asanyarray(
            nib.load(str(case_path(plan.run_dir, "probability_maps", key))).dataobj
        )
        ung = np.asanyarray(
            nib.load(str(case_path(plan.run_dir, "ungated_predictions", key))).dataobj
        )
        mask = np.asanyarray(nib.load(str(case_path(plan.run_dir, "brain_masks", key))).dataobj)
        gat = np.asanyarray(
            nib.load(str(case_path(plan.run_dir, "gated_predictions", key))).dataobj
        )
        assert prob.shape == SHAPE and np.isfinite(prob).all()
        assert np.array_equal(ung == 1, prob >= 0.5)
        assert np.array_equal(gat == 1, (ung == 1) & (mask == 1))
    thr = json.loads(run_output_path(plan.run_dir, "threshold_record").read_text(encoding="utf-8"))
    assert thr == {"optimized": False, "source": "smoke_test_fixed_not_validated", "threshold": 0.5}
    hist = json.loads(run_output_path(plan.run_dir, "training_history").read_text(encoding="utf-8"))
    assert hist["model"]["model_id"] == MODEL_ID and hist["model"]["pretrained_weights"] is None
    assert hist["model"]["parameters"] > 0 and len(hist["history"]) == 1
    assert verify_run(plan.run_dir)["ok"]
    pre = json.loads((plan.run_dir / "pre_run_manifest.json").read_text(encoding="utf-8"))
    assert pre["access"]["train_cases"] == 2 and pre["access"]["val_cases"] == 2
    assert pre["access"].get("test_cases", 0) == 0 and pre["determinism"]["seed"] == 42
    assert (
        pre["determinism"]["tf32_convolution"] is False
        and pre["determinism"]["cudnn_benchmark"] is False
    )


def test_same_seed_identical_and_different_seed_differs(w):
    a, b, c = _plan(w, "syn-run-a"), _plan(w, "syn-run-b"), _plan(w, "syn-run-c")
    _run(w, a, seed=42)
    _run(w, b, seed=42)
    _run(w, c, seed=7)
    ha, hb, hc = _hashes(a.run_dir), _hashes(b.run_dir), _hashes(c.run_dir)
    assert ha == hb  # checkpoint, history, probabilities, binaries, metrics, aggregate
    assert ha["aggregate_results"]["checkpoint"] != hc["aggregate_results"]["checkpoint"]
    assert ha["probability_maps"] != hc["probability_maps"]


def test_output_tampering_detected(w):
    plan = _plan(w)
    _run(w, plan)
    ck = run_output_path(plan.run_dir, "checkpoint")
    ck.write_bytes(ck.read_bytes() + b"x")
    res = verify_run(plan.run_dir)
    assert not res["ok"] and any("checkpoint" in p for p in res["problems"])


def test_no_alias_or_id_in_outputs(w):
    plan = _plan(w)
    _run(w, plan)
    files = [p for p in plan.run_dir.rglob("*") if p.is_file()]
    texts = [p.read_text(encoding="utf-8") for p in files if p.suffix == ".json"]
    joined = "\n".join(texts + [p.name for p in files])
    assert not any(a in joined for a in TRAIN + VAL)
    assert "sub-" not in joined and "strokecase" not in joined.lower()
    assert str(w.repo) not in joined and w.repo.as_posix() not in joined
    ckpt = torch.load(
        run_output_path(plan.run_dir, "checkpoint"), map_location="cpu", weights_only=False
    )
    assert not any(a in json.dumps(ckpt["meta"]) for a in TRAIN + VAL)


def test_failure_post_run_recorded(w):
    plan = _plan(w)
    broken = copy.copy(w.files)
    broken["syn-train-02"] = CaseFiles(
        w.data / "missing.nii.gz", w.data / "missing.nii.gz", w.data / "missing.nii.gz"
    )
    res = _run(w, plan, files=broken)
    post = load_post_run(plan.run_dir)
    assert res["status"] == FAILED_RUNNER and post["run"]["exit_code"] == 1
    assert post["reproducibility"]["failed_stage"] == "runner" and verify_run(plan.run_dir)["ok"]


def test_run_id_reuse_refused(w):
    plan = _plan(w)
    _run(w, plan)
    before = {p: sha256_file(p) for p in plan.run_dir.rglob("*") if p.is_file()}
    res = _run(w, plan)
    assert res["pre_run_status"] == "rejected" and "output_dir_exists" in res["reasons"]
    assert {p: sha256_file(p) for p in plan.run_dir.rglob("*") if p.is_file()} == before


def test_train_plan_requires_validation_cases(w):
    with pytest.raises(RunPlanError):
        _plan(w, validation_aliases=[])


def test_committed_smoke_config_matches_policy():
    import yaml

    from neuroscan_ai.v2.model_2d import build_model, describe

    cfg = yaml.safe_load((REPO / "configs/v2/2d_smoke.yaml").read_text(encoding="utf-8"))
    info = describe(
        build_model(42, cfg["model"]["base_channels"]), 42, cfg["model"]["base_channels"]
    )
    assert info["model_id"] == cfg["model"]["id"] == MODEL_ID and info["parameters"] == 6497
    assert cfg["model"]["pretrained_weights"] is None and cfg["training"]["epochs"] == 1
    assert cfg["training"]["seed"] == 42 and cfg["training"]["dataloader_workers"] == 0
    assert cfg["inference"]["threshold"] == 0.5
    assert cfg["inference"]["threshold_source"] == "smoke_test_fixed_not_validated"
    d = cfg["determinism"]
    assert d["precision"] == "fp32" and not d["automatic_mixed_precision"]
    assert not d["tf32_matmul"] and not d["tf32_convolution"] and not d["cudnn_benchmark"]
    assert d["torch_deterministic_algorithms"] and d["cudnn_deterministic"]
    assert (cfg["data"]["n_train"], cfg["data"]["n_val"]) == (4, 2)
    assert cfg["policies"]["brain_mask"] == "skull_stripped_background_v2_smoke"
    assert cfg["policies"]["preprocessing"] == "preprocess_v2_isles_smoke_v1"
