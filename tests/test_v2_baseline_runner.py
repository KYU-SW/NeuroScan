"""v2 2D baseline training runner on SYNTHETIC volumes only (CPU, throwaway git repo, the real
``configs/v2/2d_baseline.yaml`` copied in). No MRI, no ISLES / SOOP access, no v1 code.
Synthetic outputs are not performance results."""

from __future__ import annotations

import ast
import copy
import gzip
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from neuroscan_ai.reproducibility import CASE_KEY_POLICY, PathRoots, case_keys
from neuroscan_ai.reproducibility.environment import (
    capture_determinism,
    capture_environment,
    capture_source_control,
)
from neuroscan_ai.reproducibility.hashing import alias_list_sha256, sha256_file, sha256_text_lf
from neuroscan_ai.reproducibility.validation import EXACT
from neuroscan_ai.run_control import (
    COMPLETED,
    FAILED_RUNNER,
    RunPlanError,
    build_plan,
    compare_run_bundles,
    load_post_run,
    verify_run,
)
from neuroscan_ai.run_control.outputs import case_path, run_output_path, training_output_labels
from neuroscan_ai.v2 import baseline_cache as bc
from neuroscan_ai.v2 import baseline_dataset as bd
from neuroscan_ai.v2 import baseline_inference as bi
from neuroscan_ai.v2 import baseline_runner as br
from neuroscan_ai.v2 import baseline_training as bt
from neuroscan_ai.v2.baseline_preprocess import PolicyConfigError
from neuroscan_ai.v2.baseline_preprocess import validate_policy_config as _validate_policy_config
from neuroscan_ai.v2.baseline_selection import select_checkpoint
from neuroscan_ai.v2.dataset import CaseFiles
from neuroscan_ai.v2.unet_2d import EXPECTED_PARAMETERS, build_model, count_parameters

REPO = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", "user.name=synthetic", "-c", "user.email=synthetic@example.invalid"]
TRAIN = ["syn-train-01", "syn-train-02", "syn-train-03"]  # 03: lesion-free (negative case)
VAL = ["syn-val-01", "syn-val-02"]  # 01 lesion-positive, 02 lesion-free
FORBIDDEN = ("sub-strokecase9999",)
SHAPES = {
    "syn-train-01": (16, 16, 4),
    "syn-train-02": (15, 14, 4),
    "syn-val-01": (16, 15, 4),
    "syn-val-02": (14, 16, 4),
    "syn-train-03": (16, 14, 4),
}
NEGATIVE = ("syn-val-02", "syn-train-03")
NEW_MODULES = (
    "baseline_dataset",
    "baseline_training",
    "baseline_inference",
    "baseline_runner",
    "baseline_cache",
    "baseline_progress",
    "baseline_resume",
    "resource_monitor",
)


def _write_case(folder: Path, alias: str, idx: int) -> CaseFiles:
    shape = SHAPES[alias]
    rng = np.random.default_rng(10 + idx)
    g = np.indices(shape).astype(float)
    c = (np.array(shape) - 1) / 2
    brain = ((g[0] - c[0]) / (shape[0] * 0.4)) ** 2 + ((g[1] - c[1]) / (shape[1] * 0.4)) ** 2 <= 1
    lesion = np.zeros(shape, bool)
    if alias not in NEGATIVE:
        lesion = brain & (((g[0] - c[0] - 1) / 2.2) ** 2 + ((g[1] - c[1]) / 2.2) ** 2 <= 1)
        lesion[:, :, [0, 3]] = False  # slices 0 and 3: brain but no lesion (negatives)
    dwi = np.where(brain, 300 + rng.normal(0, 20, shape), 0).astype(np.float32)
    dwi[lesion] += 400
    adc = np.where(brain, 800 + rng.normal(0, 40, shape), 0).astype(np.float32)
    adc[lesion] -= 400
    aff = np.diag([2.0, 2.0, 4.0, 1.0])
    aff[:3, 3] = (-20.0, 10.0, -5.0)
    files = CaseFiles(
        folder / f"{idx}_d.nii.gz", folder / f"{idx}_a.nii.gz", folder / f"{idx}_m.nii.gz"
    )
    nib.save(nib.Nifti1Image(dwi, aff), str(files.dwi))
    nib.save(nib.Nifti1Image(adc, aff), str(files.adc))
    gt = nib.Nifti1Image(lesion.astype(np.float32), aff)
    if alias == "syn-train-01":
        gt.set_data_dtype(np.int16)  # scaled storage -> decoded 0.9999999997671694
    nib.save(gt, str(files.label))
    return files


@pytest.fixture(scope="module")
def w(tmp_path_factory):
    root = tmp_path_factory.mktemp("baseline")
    repo = root / "repo"
    repo.mkdir()
    subprocess.run([*GIT, "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / ".gitignore").write_text("results/\ndata/\n", encoding="utf-8")
    (repo / "configs/v2").mkdir(parents=True)
    shutil.copy(REPO / "configs/v2/2d_baseline.yaml", repo / "configs/v2/2d_baseline.yaml")
    (repo / "requirements-lock.txt").write_text("torch==synthetic\n", encoding="utf-8")
    subprocess.run([*GIT, "-C", str(repo), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    data = repo / "data"
    data.mkdir()
    splits = {
        "splits": {"train": TRAIN, "val": VAL, "test": ["syn-test-01"], "pilot": ["soop-pilot-01"]}
    }
    (data / "split.json").write_text(json.dumps(splits), encoding="utf-8")
    files = {a: _write_case(data, a, i) for i, a in enumerate(TRAIN + VAL)}
    ns = SimpleNamespace(
        repo=repo, data=data, files=files, cfg=repo / "configs/v2/2d_baseline.yaml"
    )
    ns.roots = PathRoots(repo, repo / "models", data)
    ns.caches = {}
    ns.main = _run(ns, "syn-baseline-a")
    return ns


def _env(w):
    return capture_environment(torch_module=torch, lock_file=w.repo / "requirements-lock.txt")


def _plan_kwargs(w, run_id, **kw):
    args = {
        "run_id": run_id,
        "purpose": "synthetic baseline runner test",
        "stage": "train",
        "dataset_id": "synthetic",
        "split_name": "train",
        "aliases": TRAIN,
        "allowed_case_count": len(TRAIN),
        "validation_aliases": VAL,
        "checkpoint": None,
        "split_file": w.data / "split.json",
        "output_root": w.repo / "results/v2_runs",
        "input_files": {a: {"dwi": f.dwi, "adc": f.adc, "gt": f.label} for a, f in w.files.items()},
        "expected": {
            "config_sha256_lf": sha256_text_lf(w.cfg),
            "split_sha256_lf": sha256_text_lf(w.data / "split.json"),
            "package_lock_sha256_lf": _env(w)["package_lock_sha256_lf"],
            "alias_list_sha256": alias_list_sha256(TRAIN + VAL, CASE_KEY_POLICY),
        },
    }
    args.update(kw)
    return args


def _cache(w, cfg_path, train, val):
    """Synthetic disk cache for (config, train, val); built once (results/v2_cache/ of the
    throwaway repo). An invalid config gets no cache (the run is refused before data)."""
    key = (str(cfg_path), tuple(train), tuple(val))
    if key not in w.caches:
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        try:  # same checks as br.prepare_config, without the spies some tests install
            _validate_policy_config(cfg)
            bt.validate_architecture(cfg)
            bt.validate_hyperparameters(cfg)
            br.validate_operational_config(cfg)
        except Exception:  # noqa: BLE001 - refused config: the runner never reaches the cache
            return w.repo / "results/v2_cache/not-built"
        out = w.repo / "results/v2_cache" / f"syn_{len(w.caches):02d}"
        bc.build_cache(
            out,
            config=cfg,
            split_file=w.data / "split.json",
            files={"train": {a: w.files[a] for a in train}, "val": {a: w.files[a] for a in val}},
            git_commit=capture_source_control(w.repo)["git_commit"],
            package_lock_sha256_lf=_env(w)["package_lock_sha256_lf"],
            forbidden_ids=FORBIDDEN,
        )
        w.caches[key] = out
    return w.caches[key]


def _run(w, run_id, cfg=None, epoch_limit=2, sc=None, device="cpu", memory_probe=None, **kw):
    det = capture_determinism(
        torch, seed=None, precision="fp32", amp=False, workers=0, batch_size=32
    )
    plan_kw = _plan_kwargs(w, run_id, **kw)
    return br.run_baseline(
        cfg or w.cfg,
        plan_kw,
        _cache(w, cfg or w.cfg, plan_kw["aliases"], plan_kw["validation_aliases"]),
        repo_root=w.repo,
        roots=w.roots,
        source_control=sc or capture_source_control(w.repo),
        environment=_env(w),
        determinism=det,
        forbidden_ids=FORBIDDEN,
        device=device,
        epoch_limit=epoch_limit,
        memory_probe=memory_probe,
    )


def _rd(w, run_id="syn-baseline-a"):
    return w.repo / "results/v2_runs" / run_id


def _json(w, label, run_id="syn-baseline-a"):
    return json.loads(run_output_path(_rd(w, run_id), label).read_text(encoding="utf-8"))


def _cases(w, aliases):
    keys = case_keys(TRAIN + VAL)
    return bd.load_cases({a: w.files[a] for a in aliases}, keys)


def _bad_cfg(w, name, mutate):
    cfg = yaml.safe_load(w.cfg.read_text(encoding="utf-8"))
    mutate(cfg)
    p = w.repo.parent / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 1-4 order, gate, architecture
def test_01_config_validated_before_dataset(w, monkeypatch):
    calls = []
    monkeypatch.setattr(br, "validate_policy_config", lambda c: calls.append("policy"))
    monkeypatch.setattr(bd, "load_cases", lambda *a, **k: calls.append("dataset"))
    monkeypatch.setattr(bc, "verify_cache", lambda *a, **k: calls.append("dataset"))
    bad = _bad_cfg(w, "bad_batch", lambda c: c["training"].update(batch_size=16))
    with pytest.raises(bt.ConfigValidationError):
        _run(w, "syn-refused-config", cfg=bad)
    assert calls == ["policy"] and not _rd(w, "syn-refused-config").exists()
    bad = _bad_cfg(w, "bad_policy", lambda c: c["preprocessing"]["gt_decode"].update(one_atol=1e-3))
    monkeypatch.undo()
    with pytest.raises(PolicyConfigError):
        _run(w, "syn-refused-policy", cfg=bad)
    assert not _rd(w, "syn-refused-policy").exists()


def test_02_03_no_dataset_or_model_before_gate_approval(w, monkeypatch):
    calls = []
    monkeypatch.setattr(bd, "load_cases", lambda *a, **k: calls.append("dataset"))
    monkeypatch.setattr(bc, "verify_cache", lambda *a, **k: calls.append("dataset"))
    monkeypatch.setattr(br, "build_model", lambda *a, **k: calls.append("model"))
    dirty = {**capture_source_control(w.repo), "working_tree_clean": False}
    # a dirty tree is refused before the gate (cache commit binding): nothing is created
    with pytest.raises(bc.CacheCommitError, match="working_tree_not_clean"):
        _run(w, "syn-rejected-gate", sc=dirty)
    assert calls == [] and not _rd(w, "syn-rejected-gate").exists()


def test_04_parameter_count_fixed():
    assert bt.expected_parameter_count() == EXPECTED_PARAMETERS == 2_439_577
    assert count_parameters(build_model(0)) == EXPECTED_PARAMETERS
    cfg = yaml.safe_load((REPO / "configs/v2/2d_baseline.yaml").read_text(encoding="utf-8"))
    bt.validate_architecture(cfg)
    cfg["model"]["channels"] = [32, 64, 128, 256]
    with pytest.raises(bt.ConfigValidationError):
        bt.validate_architecture(cfg)


@pytest.mark.parametrize(
    "path,value",
    [
        (("batch_size",), 16),
        (("max_epochs",), 50),
        (("num_workers",), 2),
        (("amp",), True),
        (("tf32_matmul",), True),
        (("optimizer", "learning_rate"), 1e-4),
        (("optimizer", "name"), "adam"),
        (("scheduler", "minimum_learning_rate"), 0.0),
        (("loss", "dice_weight"), 2.0),
        (("loss", "dice_smooth"), 1e-5),
        (("gradient_clipping", "max_norm"), 5.0),
        (("sampling", "positive_fraction"), 0.3),
        (("threshold", "value"), 0.4),
        (("early_stopping_patience",), 10),
        (("seed",), 7),
        (("automatic_adjustment",), "allowed"),
    ],
)
def test_hyperparameters_locked(path, value):
    cfg = yaml.safe_load((REPO / "configs/v2/2d_baseline.yaml").read_text(encoding="utf-8"))
    bt.validate_hyperparameters(cfg)
    node = cfg["training"]
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(bt.ConfigValidationError):
        bt.validate_hyperparameters(cfg)


# ---------------------------------------------------------------- 5-11 dataset and sampling
def test_05_dataset_item_shapes_and_no_ids(w):
    cases = _cases(w, TRAIN)
    item = bd.slice_item(cases[1], cases[1].brain_slices[0])
    assert item["image"].shape[0] == 2 and item["target"].shape[0] == 1
    assert item["brain_mask"].shape == item["valid_region_mask"].shape == item["target"].shape
    assert all(s % 8 == 0 for s in item["image"].shape[1:])
    assert item["case_key"].startswith("case_") and isinstance(item["slice_index"], int)
    assert not any(a in json.dumps(item["geometry"]) + item["case_key"] for a in TRAIN + VAL)
    b = bd.collate([bd.slice_item(cases[0], 1), bd.slice_item(cases[1], 1)])
    assert b["image"].shape[0] == 2 and b["image"].dtype == torch.float32


def test_06_gt_decode_applied(w):
    c = _cases(w, ["syn-train-01"])[0]
    raw = np.unique(np.asanyarray(nib.load(str(w.files["syn-train-01"].label)).dataobj))
    assert raw.tolist() == [0.0, 0.9999999997671694]
    rec = c.record["gt_decode"]
    assert rec["policy"] == "binary_mask_decode_v1_atol_1e_6" and rec["snapped_voxels"] > 0
    assert c.original_gt.dtype == bool and c.original_gt.sum() == rec["snapped_voxels"]


def test_07_one_mm_target_grid(w):
    c = _cases(w, ["syn-train-02"])[0]
    assert np.allclose(c.prepared.model_grid.spacing, (1.0, 1.0, 4.0))
    assert c.prepared.model_grid.shape == (30, 28, 4)  # 15 x 2 mm, 14 x 2 mm


def test_08_padding_and_valid_region_mask(w):
    c = _cases(w, ["syn-train-02"])[0]
    item = bd.slice_item(c, 1)
    v = item["valid_region_mask"][0].astype(bool)
    assert item["image"].shape[1:] == (32, 32) and v.sum() == 30 * 28
    assert v[:30, :28].all() and not v[30:].any() and not v[:, 28:].any()
    assert (item["image"][:, ~v] == 0).all() and (item["target"][0][~v] == 0).all()


def test_09_10_11_sampling_deterministic_seeded_and_recorded(w):
    cases = _cases(w, TRAIN)
    o1, r1 = bd.epoch_sampling(cases, 42, 3)
    o2, r2 = bd.epoch_sampling(cases, 42, 3)
    assert o1 == o2 and r1["slice_list_sha256"] == r2["slice_list_sha256"]
    assert bd.epoch_sampling(cases, 7, 3)[1]["slice_list_sha256"] != r1["slice_list_sha256"]
    assert bd.epoch_sampling(cases, 42, 4)[1]["slice_list_sha256"] != r1["slice_list_sha256"]
    assert r1["brain_slices"] == r1["positive_slices"] + r1["negative_slices"] == r1["sampled"]
    assert r1["negative_slices"] > 0 and r1["sampled_positive_ratio"] == 0.5
    assert r1["sampling_seed"] == [42, 3] and r1["epoch"] == 3
    seen = {(ci, z) for e in range(1, 6) for ci, z in bd.epoch_sampling(cases, 42, e)[0]}
    assert seen == {(ci, z) for ci, c in enumerate(cases) for z in c.brain_slices}


# ---------------------------------------------------------------- 12-18 loss and optimiser
def _loss_inputs():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = (torch.rand(2, 1, 16, 16) > 0.7).float()
    valid = torch.zeros(2, 1, 16, 16)
    valid[:, :, :12, :10] = 1
    return logits, target, valid


def test_12_13_14_masked_dice_bce_combined_one_to_one():
    from neuroscan_ai.v2.baseline_slices import masked_bce_loss, masked_soft_dice_loss

    logits, target, valid = _loss_inputs()
    total, dice, bce = bt.combined_loss(logits, target, valid)
    assert torch.isclose(dice, masked_soft_dice_loss(logits, target, valid, smooth=1.0))
    assert torch.isclose(bce, masked_bce_loss(logits, target, valid))
    assert torch.isclose(total, dice + bce)
    crop = (slice(None), slice(None), slice(0, 12), slice(0, 10))
    ones = torch.ones_like(valid[crop])
    t2, _, _ = bt.combined_loss(logits[crop], target[crop], ones)
    assert torch.isclose(total, t2, atol=1e-6)  # padding region does not count


def test_15_padding_gradient_zero():
    logits, target, valid = _loss_inputs()
    with torch.no_grad():
        logits[valid == 0] = 1e4
    total, _, _ = bt.combined_loss(logits, target, valid)
    total.backward()
    assert torch.isfinite(total) and (logits.grad[valid == 0] == 0).all()


def test_16_17_adamw_and_cosine_schedule():
    model = torch.nn.Conv2d(2, 1, 1)
    opt = bt.build_optimizer(model)
    assert isinstance(opt, torch.optim.AdamW)
    g = opt.param_groups[0]
    assert g["lr"] == 1e-3 and g["weight_decay"] == 1e-4
    sched = bt.build_scheduler(opt)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 100 and sched.eta_min == 1e-6
    opt.step()
    sched.step()
    expected = 1e-6 + (1e-3 - 1e-6) * (1 + np.cos(np.pi / 100)) / 2
    assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-9)


def test_18_gradient_clipping(w, monkeypatch):
    seen = []
    orig = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, norm_type=2.0, **kw):
        params = list(params)
        n = orig(params, max_norm, norm_type=norm_type, **kw)
        post = torch.norm(torch.stack([p.grad.norm(2) for p in params if p.grad is not None]))
        seen.append((max_norm, norm_type, float(post)))
        return n

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    cases = _cases(w, TRAIN)
    model = build_model(0)
    order, _ = bd.epoch_sampling(cases, 42, 1)
    tm = bt.train_one_epoch(model, bd.batches(cases, order, 32), bt.build_optimizer(model), "cpu")
    assert seen and all(m == 1.0 and nt == 2 and post <= 1.0 + 1e-5 for m, nt, post in seen)
    assert tm["gradient_norm"]["clip_max_norm"] == 1.0
    assert np.isfinite(tm["train_loss_total"])  # test 19 (one normal epoch) on its own


# ---------------------------------------------------------------- 19-35 full synthetic run
def test_19_run_completed(w):
    assert w.main["status"] == COMPLETED, w.main
    hist = _json(w, "training_history")
    assert hist["epochs_run"] == 2 and hist["synthetic_epoch_limit"] == 2
    assert hist["max_epochs"] == 100 and hist["hyperparameters"]["batch_size"] == 32
    for e in hist["epochs"]:
        for k in (
            "train_loss_total",
            "train_loss_dice",
            "train_loss_bce",
            "validation_loss_total",
            "validation_mean_positive_case_dice",
            "validation_total_false_positive_voxel_volume_ml",
            "validation_lesion_wise_recall",
            "validation_missed_gt_component_count",
            "learning_rate",
            "gradient_norm",
            "sampled_positive_ratio",
            "collapse",
        ):
            assert e[k] is not None, k
    assert (_rd(w) / br.TIMING_NAME).is_file()


def test_20_best_checkpoint_is_selected_epoch(w):
    hist = _json(w, "training_history")
    best = torch.load(run_output_path(_rd(w), "best_checkpoint"), weights_only=False)
    assert best["epoch"] == hist["selection"]["selected_epoch"]
    assert best["kind"] == "best_checkpoint"


def test_21_tie_break_order():
    base = {"threshold": 0.5, "output": "ungated"}
    recs = [
        {
            **base,
            "epoch": 1,
            "mean_dice_positive": 0.80,
            "val_loss": 0.5,
            "false_positive_voxel_volume_ml": 1.0,
        },
        {
            **base,
            "epoch": 2,
            "mean_dice_positive": 0.79995,
            "val_loss": 0.4,
            "false_positive_voxel_volume_ml": 3.0,
        },
        {
            **base,
            "epoch": 3,
            "mean_dice_positive": 0.80,
            "val_loss": 0.4,
            "false_positive_voxel_volume_ml": 2.0,
        },
        {
            **base,
            "epoch": 4,
            "mean_dice_positive": 0.80,
            "val_loss": 0.4,
            "false_positive_voxel_volume_ml": 2.0,
        },
    ]
    assert select_checkpoint(recs)["selected_epoch"] == 3  # dice tie -> loss -> FP -> earlier
    recs[0]["mean_dice_positive"] = 0.9
    assert select_checkpoint(recs)["selected_epoch"] == 1


def test_22_last_checkpoint_contents(w):
    ck = torch.load(run_output_path(_rd(w), "last_checkpoint"), weights_only=False)
    for k in (
        "schema",
        "model_id",
        "architecture",
        "channels",
        "parameter_count",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "epoch",
        "best_metric",
        "seed",
        "config_canonical_sha256",
        "split_canonical_sha256",
        "preprocessing_policy",
        "brain_mask_policy",
        "threshold_policy",
        "package_lock_sha256_lf",
        "git_commit",
        "environment",
        "sampler",
    ):
        assert ck.get(k) is not None, k
    assert ck["parameter_count"] == EXPECTED_PARAMETERS and ck["epoch"] == 2
    assert ck["threshold_policy"] == "baseline_checkpoint_fixed_0_5"


def test_23_24_history_and_validation_metrics(w):
    vm = _json(w, "validation_metrics")
    assert vm["official_output"] == "ungated" and vm["threshold"] == 0.5
    assert len(vm["cases"]) == 2 and vm["ungated"]["output"] == "ungated"
    assert vm["gated_comparison_only"]["output"] == "gated"


def test_25_threshold_record(w):
    assert _json(w, "threshold_record") == {
        "threshold": 0.5,
        "source": "baseline_checkpoint_fixed_0_5",
        "optimized": False,
        "threshold_search_run": False,
    }
    assert not (_rd(w) / "training/threshold_search.json").exists()


def test_26_sampler_record(w):
    s = _json(w, "sampler_record")
    assert [e["epoch"] for e in s["epochs"]] == [1, 2] and s["seed"] == 42
    cases = _cases(w, TRAIN)
    assert (
        s["epochs"][0]["slice_list_sha256"]
        == bd.epoch_sampling(cases, 42, 1)[1]["slice_list_sha256"]
    )


def test_27_preprocessing_record(w):
    p = _json(w, "preprocessing_record")
    assert p["preprocess_policy"] == "preprocess_v2_isles_1mm_inplane_v1"
    assert len(p["train_cases"]) == 3 and len(p["validation_cases"]) == 2
    assert all("gt_decode" in c and "target_size" in c for c in p["train_cases"])
    assert p["inputs_modified"] is False


def test_28_29_30_original_grid_outputs(w):
    keys = case_keys(TRAIN + VAL)
    for alias in VAL:
        k = keys[alias]
        src = nib.load(str(w.files[alias].dwi))
        prob = nib.load(str(case_path(_rd(w), "probability_maps", k)))
        assert prob.shape == src.shape and np.allclose(prob.affine, src.affine)
        p = np.asanyarray(prob.dataobj)
        u = np.asanyarray(nib.load(str(case_path(_rd(w), "ungated_predictions", k))).dataobj)
        m = np.asanyarray(nib.load(str(case_path(_rd(w), "brain_masks", k))).dataobj)
        g = np.asanyarray(nib.load(str(case_path(_rd(w), "gated_predictions", k))).dataobj)
        assert np.array_equal(u.astype(bool), p >= 0.5)
        assert np.array_equal(g.astype(bool), u.astype(bool) & m.astype(bool))
        metric = json.loads(case_path(_rd(w), "patient_metrics", k).read_text(encoding="utf-8"))
        assert metric["official_output"] == "ungated" and "gated_comparison_only" in metric


def test_31_boundary_and_audits(w):
    k = case_keys(TRAIN + VAL)["syn-val-01"]
    m = json.loads(case_path(_rd(w), "patient_metrics", k).read_text(encoding="utf-8"))
    rec = m["output_record"]
    for f in (
        "outside_brain_mask_voxels",
        "outside_brain_mask_ratio",
        "touches_image_edge",
        "min_distance_to_brain_boundary_mm",
        "boundary_warnings",
        "roundtrip_mask_xor_voxels",
        "roundtrip_mask_dice",
    ):
        assert f in rec, f
    assert m["gt_decode_audit"]["policy"] == "binary_mask_decode_v1_atol_1e_6"
    assert len(m["target_grid_audit"]) == 2


def test_32_33_34_35_safety_metrics(w):
    u = _json(w, "validation_metrics")["ungated"]
    for f in (
        "false_positive_voxel_volume_ml",
        "no_gt_overlap_component_volume_ml",
        "no_gt_overlap_component_count",
        "oversegmentation_volume_ml",
        "lesion_recall_exploratory",
        "no_lesion_fp_volume_ml",
    ):
        assert u[f] is not None, f
    assert set(u["small_lesion_recall"]) == {"lt_0.1_ml", "lt_0.5_ml", "lt_1.0_ml"}
    assert u["positive_cases"] == 1 and u["no_lesion_cases"] == 1


# ---------------------------------------------------------------- 36-40 stops and failures
def _failed(w, run_id):
    rd = _rd(w, run_id)
    post = load_post_run(rd)
    fr = json.loads(run_output_path(rd, "failure_record").read_text(encoding="utf-8"))
    assert post["run"]["status"] == FAILED_RUNNER
    assert (rd / "post_run_manifest.sha256").is_file() and verify_run(rd)["ok"]
    assert "failure_record" in post["outputs"]["aggregate_results"]
    assert fr["retried"] is False and fr["hyperparameters_adjusted"] is False
    assert fr["accessed_splits"] == ["train", "val"]
    return fr


def test_36_nan_loss_stops(w, monkeypatch):
    def nan_loss(logits, target, valid):
        z = logits.sum() * float("nan")
        return z, z, z

    monkeypatch.setattr(bt, "combined_loss", nan_loss)
    res = _run(w, "syn-fail-nan", epoch_limit=1)
    assert res["status"] == FAILED_RUNNER
    fr = _failed(w, "syn-fail-nan")
    assert fr["error_type"] == "NonFiniteLossError" and fr["failed_stage"] == "training"
    assert "training/preprocessing_record.json" in fr["outputs_before_failure_sha256"]


def test_37_non_finite_gradient_stops(w, monkeypatch):
    def bad_model(seed):
        m = build_model(seed)
        next(m.parameters()).register_hook(lambda g: g * float("inf"))
        return m

    monkeypatch.setattr(br, "build_model", bad_model)
    _run(w, "syn-fail-grad", epoch_limit=1)
    assert _failed(w, "syn-fail-grad")["error_type"] == "NonFiniteGradientError"


def test_38_all_one_prediction_stops(w, monkeypatch):
    monkeypatch.setattr(
        bi, "predict_model_grid", lambda m, c, d, b: np.ones(c.prepared.model_grid.shape)
    )
    _run(w, "syn-fail-ones", epoch_limit=1)
    fr = _failed(w, "syn-fail-ones")
    assert fr["error_type"] == "PredictionCollapseError" and fr["failed_stage"] == "validation"


def test_probability_range_violation_stops(w, monkeypatch):
    monkeypatch.setattr(bi, "infer_volume", lambda m, img, **k: np.full(img.shape[1:], 1.5))
    _run(w, "syn-fail-range", epoch_limit=1)
    assert _failed(w, "syn-fail-range")["error_type"] == "ProbabilityRangeError"


def test_39_40_oom_not_retried_and_failure_recorded(w, monkeypatch):
    calls = []

    def oom(model, batches, opt, device):
        calls.append(sum(1 for _ in batches))
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")

    monkeypatch.setattr(bt, "train_one_epoch", oom)
    res = _run(w, "syn-fail-oom", epoch_limit=3)
    assert res["status"] == FAILED_RUNNER and len(calls) == 1  # no retry, no smaller batch
    fr = _failed(w, "syn-fail-oom")
    assert fr["error_type"] == "OutOfMemoryError" and fr["failed_stage"] == "training_loop"
    assert fr["validation_cases_succeeded"] == 0 and fr["validation_cases_failed"] == 2
    # a failed run ID is never reused
    assert _run(w, "syn-fail-oom")["pre_run_status"] == "rejected"


def test_lesion_free_validation_refused(w):
    aliases = [*TRAIN, "syn-val-02"]
    kw = _plan_kwargs(w, "syn-fail-noval")
    res = _run(
        w,
        "syn-fail-noval",
        validation_aliases=["syn-val-02"],
        input_files={a: v for a, v in kw["input_files"].items() if a in aliases},
        expected={
            **kw["expected"],
            "alias_list_sha256": alias_list_sha256(aliases, CASE_KEY_POLICY),
        },
    )
    assert res["status"] == FAILED_RUNNER
    fr = json.loads(
        run_output_path(_rd(w, "syn-fail-noval"), "failure_record").read_text(encoding="utf-8")
    )
    assert fr["error_type"] == "MissingValidationResultError"
    assert fr["failed_stage"] == "validation"
    with pytest.raises(bi.MissingValidationResultError):
        bi.evaluate(build_model(0), _cases(w, ["syn-val-02"]), torch.device("cpu"), 32)


def test_epoch_limit_refused_for_real_dataset(w):
    runner = br.BaselineRunner(
        yaml.safe_load(w.cfg.read_text(encoding="utf-8")), w.repo / "no-cache", epoch_limit=1
    )
    plan = SimpleNamespace(dataset_id="isles2022")
    with pytest.raises(bt.ConfigValidationError):
        runner._run(SimpleNamespace(run_dir=w.repo, pre_run={}), plan)


# ---------------------------------------------------------------- 41-46 integrity and privacy
def test_41_checkpoint_hashes_recorded(w):
    post = load_post_run(_rd(w))
    agg = post["outputs"]["aggregate_results"]
    assert set(training_output_labels("baseline_v1")) <= set(agg)
    for label in ("best_checkpoint", "last_checkpoint"):
        assert agg[label] == sha256_file(run_output_path(_rd(w), label))


def test_42_output_tampering_detected(w, tmp_path):
    copy_dir = tmp_path / "copy"
    shutil.copytree(_rd(w), copy_dir)
    assert verify_run(copy_dir)["ok"]
    p = run_output_path(copy_dir, "sampler_record")
    p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")
    res = verify_run(copy_dir)
    assert not res["ok"] and any("sampler_record" in x for x in res["problems"])


def test_43_same_seed_cpu_exact_match(w):
    res = _run(w, "syn-baseline-b")
    assert res["status"] == COMPLETED
    cmp = compare_run_bundles(_rd(w), _rd(w, "syn-baseline-b"))
    assert cmp["level"] == EXACT, cmp


def test_44_test_final_test_soop_refused(w):
    for kw in (
        {"split_name": "test", "aliases": ["syn-test-01"], "allowed_case_count": 1},
        {"validation_aliases": ["soop-pilot-01"]},
    ):
        with pytest.raises(RunPlanError):
            build_plan(
                **{
                    **_plan_kwargs(w, "syn-refused-split", **kw),
                    "model_id": "m",
                    "config": w.cfg,
                    "policies": {
                        k: "p"
                        for k in (
                            "preprocessing",
                            "inference",
                            "threshold",
                            "brain_mask",
                            "combination",
                        )
                    },
                }
            )
    src = "\n".join(
        (REPO / "ai/neuroscan_ai/v2" / f"{m}.py").read_text(encoding="utf-8").lower()
        for m in NEW_MODULES
    )
    for forbidden in ("final_test", "soop", "data/raw", "derivatives", "split_v1"):
        assert forbidden not in src


def test_45_no_v1_imports():
    v1 = (
        "neuroscan_ai.training",
        "neuroscan_ai.inference",
        "neuroscan_ai.preprocessing",
        "neuroscan_ai.analysis",
        "neuroscan_ai.final_test",
    )
    for m in NEW_MODULES:
        tree = ast.parse((REPO / "ai/neuroscan_ai/v2" / f"{m}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n == p or n.startswith(p + ".") for n in names for p in v1), (m, names)


def test_46_no_alias_id_or_absolute_path_in_outputs(w):
    needles = [a.encode() for a in TRAIN + VAL + list(FORBIDDEN)]
    needles += [str(w.repo).encode(), w.repo.as_posix().encode(), str(Path.home()).encode()]
    for p in _rd(w).rglob("*"):
        if not p.is_file():
            continue
        data = p.read_bytes()
        if p.suffix == ".gz":
            data = gzip.decompress(data)
        for n in needles:
            assert n not in data, (p.name, n[:20])
    for label in ("best_checkpoint", "last_checkpoint"):
        ck = torch.load(run_output_path(_rd(w), label), weights_only=False)
        meta = json.dumps({k: v for k, v in ck.items() if not k.endswith("_state")}, default=str)
        assert not any(n.decode() in meta for n in needles)
    assert copy.deepcopy(w.main)["status"] == COMPLETED


# ---------------------------------------------------------------- device and seed
def test_device_enum_and_no_cpu_fallback(w, monkeypatch):
    with pytest.raises(bt.ConfigValidationError):
        bt.resolve_device("mps")
    with pytest.raises(bt.ConfigValidationError):
        _run(w, "syn-refused-device", device="gpu0")
    assert not _rd(w, "syn-refused-device").exists()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bt.resolve_device("cpu").type == "cpu"
    assert bt.resolve_device("cuda_if_available").type == "cpu"  # convenience mode only
    with pytest.raises(bt.DeviceUnavailableError):
        bt.resolve_device("cuda")
    res = _run(w, "syn-fail-nocuda", device="cuda", epoch_limit=1)
    assert res["status"] == FAILED_RUNNER
    fr = _failed(w, "syn-fail-nocuda")
    assert fr["error_type"] == "DeviceUnavailableError" and fr["failed_stage"] == "device"
    assert not (_rd(w, "syn-fail-nocuda") / "training/preprocessing_record.json").exists()


def test_different_seed_changes_outputs(w):
    a, b = build_model(42).state_dict(), build_model(43).state_dict()
    assert any(not torch.equal(a[k], b[k]) for k in a)
    cases = _cases(w, TRAIN)
    assert bd.epoch_sampling(cases, 42, 1)[0] != bd.epoch_sampling(cases, 43, 1)[0]


# ---------------------------------------------------------------- formal device policy (A-2..A-4)
def test_a01_config_default_device_is_cuda():
    cfg = yaml.safe_load((REPO / "configs/v2/2d_baseline.yaml").read_text(encoding="utf-8"))
    assert cfg["training"]["device"] == "cuda"
    cfg["training"]["device"] = "cuda_if_available"
    with pytest.raises(bt.ConfigValidationError):
        bt.validate_hyperparameters(cfg)


@pytest.mark.parametrize(
    "profile,override,ok",
    [
        ("formal_isles", None, True),  # A2: real dataset + cuda
        ("formal_isles", "cuda", True),
        ("formal_isles", "cpu", False),  # A3
        ("formal_isles", "cuda_if_available", False),  # A4
        ("synthetic", "cpu", True),  # A5
        ("synthetic", "cuda_if_available", True),  # A6
        ("synthetic", "cuda", True),
        ("declared_by_caller", "cpu", False),
    ],
)
def test_a02_06_device_policy_matrix(profile, override, ok):
    if ok:
        assert bt.check_device_policy(profile, "cuda", override, None) == (override or "cuda")
    else:
        with pytest.raises(bt.ConfigValidationError):
            bt.check_device_policy(profile, "cuda", override, None)


def test_a03_04_09_12_formal_override_refused_before_gate(w, monkeypatch):
    calls = []
    monkeypatch.setattr(bd, "load_cases", lambda *a, **k: calls.append("dataset"))
    monkeypatch.setattr(bc, "verify_cache", lambda *a, **k: calls.append("dataset"))
    monkeypatch.setattr(br, "build_model", lambda *a, **k: calls.append("model"))
    for dev in ("cpu", "cuda_if_available"):
        with pytest.raises(bt.ConfigValidationError):
            _run(w, "syn-formal-refused", dataset_id="isles2022", device=dev, epoch_limit=None)
    with pytest.raises(bt.ConfigValidationError):  # epoch_limit is synthetic-only
        _run(w, "syn-formal-refused", dataset_id="isles2022", device="cuda", epoch_limit=1)
    assert calls == [] and not _rd(w, "syn-formal-refused").exists()
    monkeypatch.undo()
    # the refused run ID was not consumed: a valid run can still use it
    assert _run(w, "syn-formal-refused")["status"] == COMPLETED


def test_profile_cannot_be_declared_synthetic(w, monkeypatch):
    raw = w.data / "raw"
    raw.mkdir(exist_ok=True)
    files = {}
    for a, f in w.files.items():
        files[a] = CaseFiles(*(Path(shutil.copy(p, raw / p.name)) for p in (f.dwi, f.adc, f.label)))
    inputs = {a: {"dwi": f.dwi, "adc": f.adc, "gt": f.label} for a, f in files.items()}
    with pytest.raises(bt.ConfigValidationError):  # dataset id says synthetic, inputs are raw data
        _run(w, "syn-declared-synthetic", input_files=inputs)
    with pytest.raises(bt.ConfigValidationError):  # unknown dataset id: no profile
        _run(w, "syn-unknown-profile", dataset_id="other-dataset")
    assert not _rd(w, "syn-declared-synthetic").exists()
    assert not _rd(w, "syn-unknown-profile").exists()


def test_a13_14_empty_predictions_are_recorded_not_stopped(w, monkeypatch):
    monkeypatch.setattr(
        bi, "predict_model_grid", lambda m, c, d, b: np.zeros(c.prepared.model_grid.shape)
    )
    res = _run(w, "syn-empty-epochs", epoch_limit=3)
    assert res["status"] == COMPLETED
    hist = _json(w, "training_history", "syn-empty-epochs")
    assert hist["epochs_run"] == 3 and hist["stopped_early"] is False
    assert all(e["collapse"]["all_predictions_empty"] for e in hist["epochs"])
    assert all(e["validation_mean_positive_case_dice"] == 0.0 for e in hist["epochs"])


def test_a15_early_stopping_patience_20(w):
    assert not bt.early_stop(20, 1, 20) and bt.early_stop(21, 1, 20)
    hist = _json(w, "training_history")
    assert hist["early_stopping_patience"] == 20
    cp = hist["collapse_policy"]["consecutive_empty_policy"]
    assert cp == {
        "action": "rely_on_early_stopping",
        "early_stopping_patience": 20,
        "separate_empty_epoch_limit": None,
    }
    assert hist["dataset_profile"] == "synthetic"
    assert hist["device_policy"] == "device_policy_v2_baseline_cuda_required_v1"


def test_a19_collapse_and_device_policy_config_code_doc():
    cfg = yaml.safe_load((REPO / "configs/v2/2d_baseline.yaml").read_text(encoding="utf-8"))
    assert cfg["collapse_policy"] == bt.FIXED_COLLAPSE_POLICY
    assert (
        cfg["collapse_policy"]["consecutive_empty_policy"]["early_stopping_patience"]
        == (cfg["training"]["early_stopping_patience"])
    )
    doc = (REPO / "docs/v2_2d_baseline_run_protocol.md").read_text(encoding="utf-8")
    for code in (
        *bt.FIXED_COLLAPSE_POLICY["stop_immediately"],
        *bt.FIXED_COLLAPSE_POLICY["record_only"],
        "rely_on_early_stopping",
        bt.DEVICE_POLICY,
    ):
        assert code in doc, code
    bad = copy.deepcopy(cfg)
    bad["collapse_policy"]["record_only"].append("all_one_validation_prediction")
    with pytest.raises(bt.ConfigValidationError):
        bt.validate_hyperparameters(bad)


# ---------------------------------------------------------------- gt_case_status_v1
from neuroscan_ai.v2.safety_metrics import (  # noqa: E402
    GT_CASE_STATUS,
    GT_LOST,
    NEGATIVE_CASE,
    POSITIVE_CASE,
    case_metrics,
    gt_case_status,
    summarize,
)


def test_g01_02_03_empty_original_gt_is_negative_case_not_blocked(w):
    assert gt_case_status(0, 0, 0) == NEGATIVE_CASE
    assert gt_case_status(10, 40, 0) == POSITIVE_CASE
    c = _cases(w, ["syn-train-03"])[0]  # loads without error (not blocked)
    assert c.ground_truth_status == NEGATIVE_CASE and c.negative_only_case
    assert c.record["gt_empty_expected"] is True and c.record["negative_only_case"] is True


def test_g04_05_train_negative_case_in_dataset_all_slices_negative(w):
    cases = _cases(w, TRAIN)
    neg = [i for i, c in enumerate(cases) if c.negative_only_case]
    assert len(cases) == 3 and len(neg) == 1
    pos_pool, neg_pool = bd.slice_index(cases)
    zs = cases[neg[0]].brain_slices
    assert zs and cases[neg[0]].positive_slices == []
    assert {(neg[0], z) for z in zs} <= set(neg_pool)
    assert not any(ci == neg[0] for ci, _ in pos_pool)
    _, rec = bd.epoch_sampling(cases, 42, 1)
    assert rec["sampled_positive_ratio"] == 0.5  # 50 % kept over the whole train pool
    p = _json(w, "preprocessing_record")
    negs = [c for c in p["train_cases"] if c["negative_only_case"]]
    assert len(negs) == 1 and negs[0]["ground_truth_status"] == NEGATIVE_CASE


def test_g06_07_08_validation_negative_case_evaluated(w):
    vm = _json(w, "validation_metrics")
    neg = [c for c in vm["cases"] if c["ungated"]["ground_truth_status"] == NEGATIVE_CASE]
    assert len(neg) == 1
    u = neg[0]["ungated"]
    for k in ("dice", "lesion_recall_exploratory", "missed_gt_count", "small_lesion_recall"):
        assert u[k] == "not_applicable", k
    for k in (
        "false_positive_voxel_volume_ml",
        "no_gt_overlap_component_volume_ml",
        "no_gt_overlap_component_count",
        "positive_voxel_ratio",
        "pred_voxels",
    ):
        assert isinstance(u[k], (int, float)) and not isinstance(u[k], bool), k
    assert isinstance(u["prediction_empty"], bool)
    assert neg[0]["gated_comparison_only"]["dice"] == "not_applicable"
    assert "pred_voxels" in neg[0]["gated_comparison_only"]
    assert "outside_brain_mask_voxels" in neg[0]["output_record"]
    assert vm["ungated"]["no_lesion_cases"] == 1


def _tiny_lesion_case(folder: Path, name: str, extra_big: bool, pos):
    shape = (40, 40, 4)
    g = np.indices(shape).astype(float)
    brain = ((g[0] - 19.5) / 15) ** 2 + ((g[1] - 19.5) / 15) ** 2 <= 1
    gt = np.zeros(shape, np.uint8)
    gt[pos[0], pos[1], 2] = 1
    if extra_big:
        gt[14:22, 14:22, 1:3] = 1
    dwi = np.where(brain, 300.0, 0).astype(np.float32)
    adc = np.where(brain, 800.0, 0).astype(np.float32)
    aff = np.diag([0.5, 0.5, 4.0, 1.0])
    files = CaseFiles(*(folder / f"{name}_{r}.nii.gz" for r in ("d", "a", "m")))
    for f, a in zip((files.dwi, files.adc, files.label), (dwi, adc, gt), strict=True):
        nib.save(nib.Nifti1Image(a, aff), str(f))
    return files


def test_g09_10_positive_gt_lost_during_preprocessing_blocked(tmp_path):
    from neuroscan_ai.v2.baseline_preprocess import PreprocessError

    assert gt_case_status(5, 0, 0) == GT_LOST and gt_case_status(50, 40, 1) == GT_LOST
    # 0.5 mm -> 1 mm nearest skips every other source voxel: a 1-voxel GT there vanishes
    outcomes = {}
    for pos in ((25, 25), (26, 26)):
        f = _tiny_lesion_case(tmp_path, f"single{pos[0]}", False, pos)
        try:
            bd.load_case("case_0001", f)
            outcomes[pos] = "kept"
        except PreprocessError as e:
            assert "lost" in str(e)
            outcomes[pos] = "blocked"
    assert set(outcomes.values()) == {"kept", "blocked"}
    lost_pos = next(k for k, v in outcomes.items() if v == "blocked")
    f = _tiny_lesion_case(tmp_path, "twocomp", True, lost_pos)  # big component kept, tiny lost
    with pytest.raises(PreprocessError):
        bd.load_case("case_0002", f)


@pytest.mark.parametrize("bad", [0.6, -1.0, float("nan"), float("inf")])
def test_g11_invalid_gt_blocked(tmp_path, bad):
    from neuroscan_ai.v2.baseline_preprocess import PreprocessError

    f = _tiny_lesion_case(tmp_path, "bad", True, (20, 20))
    img = nib.load(str(f.label))
    arr = np.asanyarray(img.dataobj).astype(np.float32)
    arr[3, 3, 0] = bad
    nib.save(nib.Nifti1Image(arr, img.affine), str(f.label))
    with pytest.raises(PreprocessError):
        bd.load_case("case_0003", f)


def test_g12_13_primary_dice_positive_only():
    aff = np.eye(4)
    gt = np.zeros((8, 8, 3), np.uint8)
    gt[2:5, 2:5, 1] = 1
    pred = gt.copy()
    pred[2, 2, 1] = 0
    pos = case_metrics(pred, gt, aff)
    empty = np.zeros_like(gt)
    fp = empty.copy()
    fp[6, 6, 2] = 1
    neg = case_metrics(fp, empty, aff)
    s = summarize([pos, neg])
    assert s["mean_dice_positive"] == round(pos["dice"], 6)  # negative case not averaged in
    assert neg["dice"] == "not_applicable" and neg["false_positive_voxel_volume_ml"] > 0
    assert s["false_positive_voxel_volume_ml"] == round(
        pos["false_positive_voxel_volume_ml"] + neg["false_positive_voxel_volume_ml"], 6
    )  # negative-case FP enters the checkpoint tie-break total
    assert s["no_lesion_fp_volume_ml"] == neg["pred_volume_ml"]


def test_g14_15_no_case_excluded_or_replaced(w):
    p = _json(w, "preprocessing_record")
    keys = case_keys(TRAIN + VAL)
    assert sorted(c["case_key"] for c in p["train_cases"]) == sorted(keys[a] for a in TRAIN)
    assert sorted(c["case_key"] for c in p["validation_cases"]) == sorted(keys[a] for a in VAL)
    post = load_post_run(_rd(w))
    assert post["run"]["cases_succeeded"] == len(VAL) and post["run"]["cases_failed"] == 0


def test_gt_case_status_config_code_doc():
    cfg = yaml.safe_load((REPO / "configs/v2/2d_baseline.yaml").read_text(encoding="utf-8"))
    assert cfg["gt_case_status"] == GT_CASE_STATUS
    bad = copy.deepcopy(cfg)
    bad["gt_case_status"]["empty_original_gt_is_valid_negative"] = False
    with pytest.raises(PolicyConfigError):
        from neuroscan_ai.v2.baseline_preprocess import validate_policy_config

        validate_policy_config(bad)
    doc = (REPO / "docs/v2_2d_baseline_run_protocol.md").read_text(encoding="utf-8")
    from neuroscan_ai.v2.safety_metrics import GT_CASE_STATES

    assert GT_CASE_STATES == (
        "positive_case",
        "negative_case",
        "gt_lost_during_preprocessing",
        "invalid_gt",
    )
    assert all(f"`{state}`" in doc for state in GT_CASE_STATES)
    for k, v in GT_CASE_STATUS.items():
        assert k in doc and str(v).lower() in doc.lower(), k
