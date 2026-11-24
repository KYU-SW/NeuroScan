"""v2 2.5D baseline runner on SYNTHETIC volumes (CPU, throwaway git repo, real 2.5D config copied
in). No MRI, no ISLES / SOOP access. Synthetic outputs are not performance results."""

from __future__ import annotations

import json
import shutil
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from neuroscan_ai.reproducibility import CASE_KEY_POLICY, PathRoots
from neuroscan_ai.reproducibility.environment import (
    capture_determinism,
    capture_environment,
    capture_source_control,
)
from neuroscan_ai.reproducibility.hashing import alias_list_sha256, sha256_file, sha256_text_lf
from neuroscan_ai.reproducibility.validation import EXACT
from neuroscan_ai.run_control import (
    COMPLETED,
    RunPlanError,
    compare_run_bundles,
    load_post_run,
    verify_run,
)
from neuroscan_ai.run_control.lifecycle import INTERRUPTED, ResumeRefusedError
from neuroscan_ai.run_control.outputs import run_output_path
from neuroscan_ai.v2 import baseline_25d_dataset as bd
from neuroscan_ai.v2 import baseline_25d_runner as br
from neuroscan_ai.v2 import baseline_cache as bc
from neuroscan_ai.v2 import baseline_progress as bp
from neuroscan_ai.v2 import baseline_resume as brs
from neuroscan_ai.v2 import baseline_training as bt
from neuroscan_ai.v2.cuda_smoke_25d import preserved_torch_state
from neuroscan_ai.v2.dataset import CaseFiles

REPO = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", "user.name=synthetic", "-c", "user.email=synthetic@example.invalid"]
TRAIN = ["syn-train-01", "syn-train-02", "syn-train-03"]  # 03: lesion-free
VAL = ["syn-val-01", "syn-val-02"]  # 02: lesion-free
NEGATIVE = ("syn-train-03", "syn-val-02")
FORBIDDEN = ("sub-strokecase9999",)
OFFICIAL = "v2-25d-baseline-seed42-multiseed-v1"
SHAPES = {
    "syn-train-01": (16, 16, 4),
    "syn-train-02": (15, 14, 4),
    "syn-train-03": (16, 14, 4),
    "syn-val-01": (16, 15, 4),
    "syn-val-02": (14, 16, 4),
}


def write_case(folder: Path, alias: str, idx: int) -> CaseFiles:
    shape = SHAPES[alias]
    rng = np.random.default_rng(100 + idx)
    g = np.indices(shape).astype(float)
    c = (np.array(shape) - 1) / 2
    brain = ((g[0] - c[0]) / (shape[0] * 0.4)) ** 2 + ((g[1] - c[1]) / (shape[1] * 0.4)) ** 2 <= 1
    lesion = np.zeros(shape, bool)
    if alias not in NEGATIVE:
        lesion = brain & (((g[0] - c[0] - 1) / 2.2) ** 2 + ((g[1] - c[1]) / 2.2) ** 2 <= 1)
        lesion[:, :, [0, 3]] = False  # slices 0 and 3: neighbour-only positive
    dwi = np.where(brain, 300 + rng.normal(0, 20, shape), 0).astype(np.float32)
    dwi[lesion] += 400
    adc = np.where(brain, 800 + rng.normal(0, 40, shape), 0).astype(np.float32)
    adc[lesion] -= 400
    aff = np.diag([2.0, 2.0, 4.0, 1.0])
    files = CaseFiles(
        folder / f"{idx}_d.nii.gz", folder / f"{idx}_a.nii.gz", folder / f"{idx}_m.nii.gz"
    )
    for arr, p in ((dwi, files.dwi), (adc, files.adc), (lesion.astype(np.float32), files.label)):
        nib.save(nib.Nifti1Image(arr, aff), str(p))
    return files


def make_world(root: Path) -> SimpleNamespace:
    """Throwaway repository, synthetic cases and one cache (nothing is trained)."""
    repo = root / "repo"
    repo.mkdir()
    subprocess.run([*GIT, "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / ".gitignore").write_text("results/\ndata/\n", encoding="utf-8")
    (repo / "configs/v2").mkdir(parents=True)
    cfg = repo / "configs/v2/25d_baseline.yaml"
    shutil.copy(REPO / "configs/v2/25d_baseline.yaml", cfg)
    (repo / "requirements-lock.txt").write_text("torch==synthetic\n", encoding="utf-8")
    subprocess.run([*GIT, "-C", str(repo), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    data = repo / "data"
    data.mkdir()
    splits = {
        "splits": {"train": TRAIN, "val": VAL, "test": ["syn-test-01"], "pilot": ["soop-pilot-01"]}
    }
    (data / "split.json").write_text(json.dumps(splits), encoding="utf-8")
    files = {a: write_case(data, a, i) for i, a in enumerate(TRAIN + VAL)}
    w = SimpleNamespace(repo=repo, data=data, files=files, cfg=cfg)
    w.roots = PathRoots(repo, repo / "models", data)
    w.cache = repo / "results/v2_cache/syn"
    bc.build_cache(
        w.cache,
        config=br.prepare_config(cfg),
        split_file=data / "split.json",
        files={"train": {a: files[a] for a in TRAIN}, "val": {a: files[a] for a in VAL}},
        git_commit=capture_source_control(repo)["git_commit"],
        package_lock_sha256_lf=_env(w)["package_lock_sha256_lf"],
        forbidden_ids=FORBIDDEN,
    )
    return w


def open_world(w):
    """(train, val) cached cases of the synthetic cache."""
    index = json.loads((w.cache / bc.INDEX_NAME).read_text("utf-8"))
    man = {
        k: json.loads((w.cache / k / bc.MANIFEST_NAME).read_text("utf-8")) for k in index["cases"]
    }
    ver = bc.VerifiedCache(w.cache, index, "synthetic", man)
    return bc.open_cases(ver, bc.CacheStore(w.cache))


def _env(w):
    return capture_environment(torch_module=torch, lock_file=w.repo / "requirements-lock.txt")


def _plan(w, run_id, **kw):
    args = {
        "run_id": run_id,
        "purpose": "synthetic 2.5D runner test",
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


def _common(w, epoch_limit, seed, smoke, sc=None, device="cpu"):
    return {
        "repo_root": w.repo,
        "source_control": sc or capture_source_control(w.repo),
        "environment": _env(w),
        "determinism": capture_determinism(
            torch, seed=None, precision="fp32", amp=False, workers=0, batch_size=32
        ),
        "forbidden_ids": FORBIDDEN,
        "device": device,
        "epoch_limit": epoch_limit,
        "seed": seed,
        "smoke": smoke,
        "memory_probe": lambda: 16 * 1024**3,
    }


def _run(w, run_id, epoch_limit=2, seed=None, smoke=False, sc=None, device="cpu", **kw):
    return br.run_baseline(
        w.cfg,
        _plan(w, run_id, seed=seed, **kw),
        w.cache,
        roots=w.roots,
        **_common(w, epoch_limit, seed, smoke, sc, device),
    )


def _resume(w, run_id, epoch_limit=2, seed=None):
    return br.resume_baseline(
        w.cfg, _plan(w, run_id, seed=seed), w.cache, **_common(w, epoch_limit, seed, False)
    )


def _rd(w, run_id):
    return w.repo / "results/v2_runs" / run_id


def _json(w, run_id, label):
    return json.loads(run_output_path(_rd(w, run_id), label).read_text("utf-8"))


def _interrupt_first_epoch(mp):
    real = bt.train_one_epoch
    n = {"k": 0}

    def wrapped(model, batches, opt, device):
        n["k"] += 1
        if n["k"] == 1:
            signal.raise_signal(signal.SIGTERM)
        return real(model, batches, opt, device)

    mp.setattr(bt, "train_one_epoch", wrapped)


@pytest.fixture(autouse=True)
def _restore():
    with preserved_torch_state():
        yield


@pytest.fixture(scope="module")
def w(tmp_path_factory):
    with preserved_torch_state():
        ns = make_world(tmp_path_factory.mktemp("b25d"))
        ns.a = _run(ns, "syn-25d-a")
        ns.b = _run(ns, "syn-25d-b")
        ns.s43 = _run(ns, "syn-25d-s43", seed=43)
        with pytest.MonkeyPatch.context() as mp:
            _interrupt_first_epoch(mp)
            ns.r1 = _run(ns, "syn-25d-r")
        ns.r2 = _resume(ns, "syn-25d-r")
        with pytest.MonkeyPatch.context() as mp:
            _interrupt_first_epoch(mp)
            ns.p1 = _run(ns, "syn-25d-pending")
    return ns


# ---------------------------------------------------------------- run, outputs, records
def test_run_completes_and_verifies(w):
    assert w.a["status"] == COMPLETED and verify_run(_rd(w, "syn-25d-a"))["ok"]
    post = load_post_run(_rd(w, "syn-25d-a"))
    assert post["reproducibility"]["runner_kind"] == "v2_25d_baseline"
    assert len(post["outputs"]["probability_maps"]) == len(VAL)
    pre = json.loads((_rd(w, "syn-25d-a") / "pre_run_manifest.json").read_text("utf-8"))
    assert pre["model"]["architecture_id"] == "neuroscan_v2_25d_baseline_v1"
    assert pre["configuration"]["policies"]["inference_policy_id"] == br.bi.INFERENCE_POLICY
    assert pre["access"]["test_cases"] == 0 and pre["access"]["final_test_accessed"] is False


def test_repeat_is_exact_match_and_seed_changes_result(w):
    a, b = _rd(w, "syn-25d-a"), _rd(w, "syn-25d-b")
    assert compare_run_bundles(a, b)["level"] == EXACT
    other = run_output_path(_rd(w, "syn-25d-s43"), "best_checkpoint")
    assert sha256_file(run_output_path(a, "best_checkpoint")) != sha256_file(other)
    sa, sb = _json(w, "syn-25d-a", "sampler_record"), _json(w, "syn-25d-s43", "sampler_record")
    assert sa["epochs"][0]["slice_list_sha256"] != sb["epochs"][0]["slice_list_sha256"]
    assert (sa["run_seed"], sb["run_seed"], sb["config_reference_seed"]) == (42, 43, 42)


def test_policy_record_in_every_required_place(w):
    rd = _rd(w, "syn-25d-a")
    recs = {
        k: _json(w, "syn-25d-a", k)
        for k in ("preprocessing_record", "training_history", "validation_metrics")
    }
    recs["aggregate"] = json.loads((rd / "aggregate_results/aggregate.json").read_text("utf-8"))
    for k in ("best_checkpoint", "last_checkpoint"):
        recs[k] = torch.load(run_output_path(rd, k), weights_only=False)
        assert recs[k]["input_channel_order"] == list(br.INPUT_CHANNEL_NAMES)
    recs["resume"] = brs.load_resume_checkpoint(rd / brs.RESUME_NAME)
    recs["config"] = br.prepare_config(w.cfg)["policy_record"]
    for name, r in recs.items():
        assert {k: r.get(k) for k in br.POLICY_RECORD} == br.POLICY_RECORD, name
    # the manifests bind the records: config hash and output hashes
    pre = json.loads((rd / "pre_run_manifest.json").read_text("utf-8"))
    assert pre["configuration"]["config"]["sha256_lf"] == sha256_text_lf(w.cfg)
    bound = load_post_run(rd)["outputs"]["aggregate_results"]
    for k in ("preprocessing_record", "best_checkpoint", "last_checkpoint", "training_history"):
        assert bound[k] == sha256_file(run_output_path(rd, k))


def test_policy_difference_is_detected(w):
    runner = br.Baseline25DRunner(br.prepare_config(w.cfg), w.cache)
    rd = _rd(w, "syn-25d-a")
    runner._check_policy_records(rd)
    path = run_output_path(rd, "validation_metrics")
    good = path.read_bytes()
    try:
        bad = {**json.loads(good), "boundary_policy_id": "zero_slice"}
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(bt.TrainingStop, match="boundary_policy_id_mismatch"):
            runner._check_policy_records(rd)
    finally:
        path.write_bytes(good)
    assert verify_run(rd)["ok"]


def test_policy_difference_fails_the_run(w, monkeypatch):
    real = br.Baseline25DRunner._checkpoint_common

    def other(self, *a):
        return {**real(self, *a), "input_channel_policy_id": "other"}

    monkeypatch.setattr(br.Baseline25DRunner, "_checkpoint_common", other)
    res = _run(w, "syn-25d-badpolicy", epoch_limit=1)
    assert res["status"] != COMPLETED
    rec = _json(w, "syn-25d-badpolicy", "failure_record")
    assert rec["runner_stage"] == "policy_records" and rec["retried"] is False


def test_sampler_record_metadata_and_no_identifiers(w):
    rec = _json(w, "syn-25d-a", "sampler_record")
    assert rec["metadata_fields"] == list(bd.METADATA_FIELDS)
    assert all(set(r) == set(bd.METADATA_FIELDS) for r in rec["slice_table"])
    assert any(r["neighbor_only_positive"] for r in rec["slice_table"])
    for p in _rd(w, "syn-25d-a").rglob("*.json"):
        text = p.read_text("utf-8")
        for n in (*TRAIN, *VAL, *FORBIDDEN, str(w.repo), w.repo.as_posix(), str(Path.home())):
            assert n not in text, p.name


def test_threshold_and_official_output(w):
    v = _json(w, "syn-25d-a", "validation_metrics")
    assert v["threshold"] == 0.5 and v["official_output"] == "ungated"
    t = _json(w, "syn-25d-a", "threshold_record")
    assert t["threshold_search_run"] is False and t["optimized"] is False
    assert "search_threshold" not in dir(br) and "search_threshold" not in dir(br.bi)
    h = _json(w, "syn-25d-a", "training_history")
    assert h["selection"]["threshold"] == 0.5 and h["epoch_limit"] == 2
    assert h["performance_interpretation_allowed"] is False


# ---------------------------------------------------------------- resume
def test_resume_equals_uninterrupted_run(w):
    assert w.r1["status"] == INTERRUPTED and w.r2["status"] == COMPLETED
    assert compare_run_bundles(_rd(w, "syn-25d-a"), _rd(w, "syn-25d-r"))["level"] == EXACT
    assert verify_run(_rd(w, "syn-25d-r"), segment="segment_01")["ok"]


def test_resume_refuses_other_seed_and_policy_difference(w, monkeypatch):
    rd = _rd(w, "syn-25d-pending")
    before = {p: sha256_file(p) for p in rd.rglob("*") if p.is_file()}
    with pytest.raises(ResumeRefusedError, match="seed_mismatch"):
        _resume(w, "syn-25d-pending", seed=43)
    real = brs.load_resume_checkpoint
    monkeypatch.setattr(
        brs, "load_resume_checkpoint", lambda p: {**real(p), "neighbor_policy_id": "other"}
    )
    with pytest.raises(ResumeRefusedError, match="neighbor_policy_id_mismatch"):
        _resume(w, "syn-25d-pending")
    monkeypatch.undo()
    path = run_output_path(rd, "preprocessing_record")
    good = path.read_bytes()
    try:
        path.write_text(json.dumps({**json.loads(good), "input_channels": 2}), encoding="utf-8")
        with pytest.raises(ResumeRefusedError, match="preprocessing_record_input_channels"):
            _resume(w, "syn-25d-pending")
    finally:
        path.write_bytes(good)
    assert {p: sha256_file(p) for p in rd.rglob("*") if p.is_file()} == before


def test_progress_is_operational(w):
    p = json.loads((_rd(w, "syn-25d-a") / bp.PROGRESS_NAME).read_text("utf-8"))
    assert p["status"] == "training_completed" and tuple(p) == bp.PROGRESS_FIELDS


# ---------------------------------------------------------------- gate, profiles
def _check(profile, **kw):
    args = {
        "run_id": OFFICIAL,
        "seed": 42,
        "epoch_limit": None,
        "batch_status": "approved",
        "counts": {"train": 176, "val": 37},
    }
    br.check_run_profile(profile, **{**args, **kw})


def test_smoke_profile_exact_conditions():
    ok = {"run_id": "v2-25d-smoke-seed42-a", "epoch_limit": 2, "batch_status": "candidate"}
    _check(br.PROFILE_SMOKE, **ok)
    _check(br.PROFILE_SMOKE, **{**ok, "run_id": "v2-25d-smoke-seed42-b"})
    for bad in (
        {"run_id": "v2-25d-smoke-seed42-c"},
        {"run_id": OFFICIAL},
        {"epoch_limit": 3},
        {"epoch_limit": None},
        {"seed": 43},
        {"seed": None},
        {"counts": {"train": 3, "val": 2}},
    ):
        with pytest.raises(bt.ConfigValidationError):
            _check(br.PROFILE_SMOKE, **{**ok, **bad})


def test_formal_profile_refuses_epoch_limit_candidate_and_smoke_ids():
    _check(bt.PROFILE_FORMAL)
    for bad in (
        {"epoch_limit": 2},
        {"batch_status": "candidate"},
        {"run_id": "v2-25d-smoke-seed42-a"},
    ):
        with pytest.raises(bt.ConfigValidationError):
            _check(bt.PROFILE_FORMAL, **bad)
    with pytest.raises(bt.ConfigValidationError):
        _check("other")


def test_device_and_seed_rules_hold_for_smoke():
    for dev in ("cpu", "cuda_if_available"):
        with pytest.raises(bt.ConfigValidationError):
            br.check_device(br.PROFILE_SMOKE, "cuda", dev, 2)
    assert br.check_device(br.PROFILE_SMOKE, "cuda", None, 2) == "cuda"
    for seed in (None, 7, -1):
        with pytest.raises(bt.ConfigValidationError):
            br.check_seed(br.PROFILE_SMOKE, seed)


def test_refused_before_anything_is_created(w):
    kw = {"dataset_id": br.prepare_config(w.cfg)["dataset_id"], "seed": 42, "device": "cuda"}
    with pytest.raises(bt.ConfigValidationError, match="batch_status"):
        _run(w, OFFICIAL, epoch_limit=None, **kw)
    with pytest.raises(bt.ConfigValidationError, match="epoch_limit"):
        _run(w, OFFICIAL, epoch_limit=2, **kw)
    with pytest.raises(bt.ConfigValidationError, match="smoke profile"):
        _run(w, "syn-smoke", smoke=True)  # a synthetic plan cannot use the smoke profile
    with pytest.raises(bt.ConfigValidationError, match="176"):
        _run(w, "v2-25d-smoke-seed42-a", smoke=True, epoch_limit=2, **kw)  # wrong counts
    for r in (OFFICIAL, "syn-smoke", "v2-25d-smoke-seed42-a"):
        assert not _rd(w, r).exists()


def test_dirty_tree_refused_before_gate(w):
    dirty = {**capture_source_control(w.repo), "working_tree_clean": False}
    with pytest.raises(bc.CacheCommitError, match="working_tree_not_clean"):
        _run(w, "syn-25d-dirty", sc=dirty)
    assert not _rd(w, "syn-25d-dirty").exists()


@pytest.mark.parametrize(
    "kw",
    [
        {"split_name": "test", "aliases": ["syn-test-01"], "allowed_case_count": 1},
        {"split_name": "final_test"},
        {"aliases": [*TRAIN, "soop-pilot-01"], "allowed_case_count": 4},
        {"validation_aliases": ["syn-test-01"]},
    ],
)
def test_test_final_test_and_soop_refused(w, kw):
    with pytest.raises(RunPlanError):
        _run(w, "syn-25d-forbidden", **kw)
    assert not _rd(w, "syn-25d-forbidden").exists()


def test_other_config_cannot_use_the_cache(w):
    cfg = yaml.safe_load(w.cfg.read_text("utf-8"))
    cfg["batch_policy"]["batch_status"] = "approved"
    with pytest.raises(bc.CacheError, match="config_canonical_sha256_mismatch"):
        bc.verify_cache(
            w.cache,
            config=cfg,
            split_file=w.data / "split.json",
            train_aliases=TRAIN,
            val_aliases=VAL,
            input_sha256={},
            package_lock_sha256_lf=_env(w)["package_lock_sha256_lf"],
            git_commit=capture_source_control(w.repo)["git_commit"],
        )
