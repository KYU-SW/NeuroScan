"""Run-bundle reproducibility comparison. Synthetic runs only (tiny CPU model / mock runner in a
throwaway git repo); test 25 reads the real smoke A/B bundles read-only when they exist locally."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neuroscan_ai.reproducibility import CASE_KEY_POLICY, PathRoots
from neuroscan_ai.reproducibility.environment import (
    capture_determinism,
    capture_environment,
    capture_source_control,
)
from neuroscan_ai.reproducibility.hashing import alias_list_sha256, sha256_text_lf
from neuroscan_ai.run_control import (
    NOT_APPLICABLE,
    MockRunner,
    build_plan,
    compare_bundle_contents,
    compare_run_bundles,
    run_controlled,
)
from neuroscan_ai.run_control.outputs import case_path, run_output_path
from neuroscan_ai.v2 import MODEL_ID, V2TwoDRunner
from tests.test_v2_real_runner import POLICIES, TRAIN, VAL, _case, _cfg

REPO = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", "user.name=synthetic", "-c", "user.email=synthetic@example.invalid"]


@pytest.fixture
def w(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([*GIT, "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / ".gitignore").write_text("results/\ndata/\nmodels/\n", encoding="utf-8")
    (repo / "cfg.yaml").write_text("synthetic: true\n", encoding="utf-8")
    (repo / "requirements-lock.txt").write_text("torch==synthetic\n", encoding="utf-8")
    subprocess.run([*GIT, "-C", str(repo), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    data, models = repo / "data", repo / "models"
    data.mkdir()
    models.mkdir()
    (models / "best.pt").write_bytes(b"synthetic input checkpoint")
    splits = {"splits": {"train": TRAIN, "val": VAL, "test": ["syn-test-01"]}}
    (data / "split.json").write_text(json.dumps(splits), encoding="utf-8")
    files = {a: _case(data, i) for i, a in enumerate(TRAIN + VAL)}
    return SimpleNamespace(repo=repo, data=data, files=files, roots=PathRoots(repo, models, data))


def _expected(w, aliases):
    env = capture_environment(torch_module=torch, lock_file=w.repo / "requirements-lock.txt")
    return {
        "config_sha256_lf": sha256_text_lf(w.repo / "cfg.yaml"),
        "split_sha256_lf": sha256_text_lf(w.data / "split.json"),
        "checkpoint_sha256": hashlib.sha256((w.repo / "models/best.pt").read_bytes()).hexdigest(),
        "package_lock_sha256_lf": env["package_lock_sha256_lf"],
        "alias_list_sha256": alias_list_sha256(aliases, CASE_KEY_POLICY),
    }


def _inputs(w, aliases):
    return {
        a: {"dwi": w.files[a].dwi, "adc": w.files[a].adc, "gt": w.files[a].label} for a in aliases
    }


def _train_run(w, run_id, seed=42):
    plan = build_plan(
        run_id=run_id,
        purpose="synthetic bundle",
        stage="train",
        dataset_id="synthetic",
        split_name="train",
        aliases=TRAIN,
        allowed_case_count=len(TRAIN),
        validation_aliases=VAL,
        model_id=MODEL_ID,
        checkpoint=None,
        config=w.repo / "cfg.yaml",
        split_file=w.data / "split.json",
        policies=POLICIES,
        output_root=w.repo / "results/v2_runs",
        expected=_expected(w, TRAIN + VAL),
        input_files=_inputs(w, TRAIN + VAL),
    )
    _go(w, plan, V2TwoDRunner(_cfg(seed), w.files))
    return plan.run_dir


def _val_run(w, run_id, faults=()):
    plan = build_plan(
        run_id=run_id,
        purpose="synthetic bundle",
        stage="validation",
        dataset_id="synthetic",
        split_name="val",
        aliases=VAL,
        allowed_case_count=len(VAL),
        model_id="mock_model",
        checkpoint=w.repo / "models/best.pt",
        config=w.repo / "cfg.yaml",
        split_file=w.data / "split.json",
        policies=POLICIES,
        output_root=w.repo / "results/v2_runs",
        expected=_expected(w, VAL),
        input_files=_inputs(w, VAL),
    )
    _go(w, plan, MockRunner(faults=set(faults)))
    return plan.run_dir


def _go(w, plan, runner):
    env = capture_environment(torch_module=torch, lock_file=w.repo / "requirements-lock.txt")
    det = capture_determinism(
        torch, seed=None, precision="fp32", amp=False, workers=0, batch_size=4
    )
    run_controlled(
        plan,
        runner,
        repo_root=w.repo,
        roots=w.roots,
        source_control=capture_source_control(w.repo),
        environment=env,
        determinism=det,
    )


def _load(run_dir):
    return (
        json.loads((run_dir / "pre_run_manifest.json").read_text(encoding="utf-8")),
        json.loads((run_dir / "post_run_manifest.json").read_text(encoding="utf-8")),
    )


@pytest.fixture
def pair(w):
    """Two identical synthetic train runs (different run_ids, same seed)."""
    return w, _train_run(w, "bundle-run-a"), _train_run(w, "bundle-run-b")


def _contents(a, b):
    return compare_bundle_contents(*a, *b)


def _edited(run_dir, side, dotted, value):
    pre, post = copy.deepcopy(_load(run_dir))
    node = pre if side == "pre" else post
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return pre, post


# ---------------------------------------------------------------- 1-3 checkpoint applicability
def test_01_train_input_checkpoint_not_applicable_both(pair):
    _, a, b = pair
    res = _contents(_load(a), _load(b))
    assert (
        res["not_applicable"] == ["pre.model.checkpoint"]
        and "pre.model.checkpoint" not in res["missing"]
    )
    assert res["level"] == "exact_match"


def test_02_not_applicable_on_one_side_only(pair):
    _, a, b = pair
    edited = _edited(
        b, "pre", "model.checkpoint", {"path": "<CHECKPOINT>/x.pt", "sha256": "f" * 64}
    )
    res = _contents(_load(a), edited)
    assert (
        res["level"] == "mismatch"
        and {"field": "pre.model.checkpoint", "class": "result_defining"} in res["differences"]
    )


def test_03_inference_checkpoint_missing_not_exact(w):
    a = _val_run(w, "val-run-a")
    pre, post = _load(a)
    pre2 = copy.deepcopy(pre)
    pre2["model"]["checkpoint"] = None  # validation stage: input checkpoint required
    res = compare_bundle_contents(pre2, post, pre2, post)
    assert res["level"] == "unavailable" and "pre.model.checkpoint" in res["missing"]
    res2 = compare_bundle_contents(pre, post, pre2, post)
    assert res2["level"] != "exact_match"


# ---------------------------------------------------------------- 4-7 integrity
def test_04_linked_bundles_ok(pair):
    _, a, b = pair
    res = compare_run_bundles(a, b)
    assert (
        res["integrity"]["a"]["ok"]
        and res["integrity"]["b"]["ok"]
        and res["level"] == "exact_match"
    )


@pytest.mark.parametrize("target", ["pre_hash", "seal", "output"])
def test_05_07_integrity_failures_are_mismatch(pair, target):
    w, a, b = pair
    if target == "pre_hash":
        p = b / "post_run_manifest.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        d["pre_run_manifest_sha256"] = "0" * 64
        p.write_text(json.dumps(d), encoding="utf-8")
    elif target == "seal":
        (b / "post_run_manifest.sha256").write_text("0" * 64, encoding="utf-8")
    else:
        p = case_path(b, "probability_maps", "case_0003")
        p.write_bytes(p.read_bytes() + b"x")
    res = compare_run_bundles(a, b)
    assert (
        res["level"] == "mismatch"
        and res["reason"] == "run_integrity_failed"
        and not res["integrity"]["b"]["ok"]
    )


# ---------------------------------------------------------------- 8-10 audit fields excluded
def test_08_run_id_only_differs_exact(pair):
    _, a, b = pair
    pa, pb = _load(a), _load(b)
    assert pa[0]["identity"]["run_id"] != pb[0]["identity"]["run_id"]
    assert _contents(pa, pb)["level"] == "exact_match"


def test_09_timestamps_only_differ_exact(pair):
    _, a, _ = pair
    pre, post = _edited(a, "post", "run.started_utc", "2000-01-01T00:00:00+00:00")
    pre["identity"]["created_at_utc"] = "2000-01-01T00:00:00+00:00"
    post["run"]["ended_utc"] = "2000-01-01T00:00:01+00:00"
    assert _contents(_load(a), (pre, post))["level"] == "exact_match"


def test_10_output_folder_only_differs_exact(pair):
    _, a, _ = pair
    copy_dir = a.parent / "renamed-folder"
    shutil.copytree(a, copy_dir)
    assert compare_run_bundles(a, copy_dir)["level"] == "exact_match"


# ---------------------------------------------------------------- 11-21 classified differences
@pytest.mark.parametrize(
    "side,dotted,value,level",
    [
        ("pre", "source_control.git_commit", "0" * 40, "compatible_but_not_exact"),  # 11
        ("pre", "environment.gpu_driver_version", "999.99", "compatible_but_not_exact"),  # 12
        ("pre", "configuration.config.sha256_lf", "0" * 64, "mismatch"),  # 13
        ("pre", "configuration.split.sha256_lf", "0" * 64, "mismatch"),  # 14
        ("pre", "inputs.file_sha256.case_0001.dwi_sha256", "0" * 64, "mismatch"),  # 15
        ("post", "outputs.aggregate_results.checkpoint", "0" * 64, "mismatch"),  # 16
        ("post", "outputs.probability_maps.case_0003", "0" * 64, "mismatch"),  # 17
        ("post", "outputs.brain_masks.case_0003", "0" * 64, "mismatch"),  # 18
        ("post", "outputs.aggregate_results.aggregate", "0" * 64, "mismatch"),  # 19
        ("post", "reproducibility.accessed_splits", ["train", "test"], "mismatch"),  # 20
        ("pre", "determinism.tf32_convolution", True, "compatible_but_not_exact"),
        ("pre", "configuration.policies.threshold_policy_id", "other", "mismatch"),
        ("pre", "source_control.working_tree_clean", False, "compatible_but_not_exact"),
    ],
)
def test_11_20_classified_differences(pair, side, dotted, value, level):
    _, a, b = pair
    assert _contents(_load(a), _edited(b, side, dotted, value))["level"] == level


def test_21_failed_vs_completed_mismatch(w):
    good = _val_run(w, "val-run-good")
    bad = _val_run(w, "val-run-bad", faults=("raise",))
    res = compare_run_bundles(good, bad)
    assert res["level"] == "mismatch" and res["runs_not_completed"] == ["b"]


def test_22_identical_bundles_exact(pair):
    _, a, b = pair
    res = compare_run_bundles(a, b)
    assert res["level"] == "exact_match" and not res["differences"] and not res["missing"]
    assert (
        run_output_path(a, "checkpoint").read_bytes()
        == run_output_path(b, "checkpoint").read_bytes()
    )


def test_validation_bundles_exact(w):
    assert (
        compare_run_bundles(_val_run(w, "val-run-a"), _val_run(w, "val-run-b"))["level"]
        == "exact_match"
    )


def test_not_applicable_constant():
    assert NOT_APPLICABLE == "not_applicable"


# ---------------------------------------------------------------- 23-24 privacy
def test_23_24_result_has_no_alias_or_path(pair):
    w, a, b = pair
    text = json.dumps(compare_run_bundles(a, b))
    assert not any(x in text for x in TRAIN + VAL)
    assert (
        str(w.repo) not in text and w.repo.as_posix() not in text and str(Path.home()) not in text
    )


# ---------------------------------------------------------------- 25 real smoke bundles (read-only)
SMOKE = REPO / "results/v2_runs"


@pytest.mark.skipif(
    not ((SMOKE / "v2-2d-smoke-seed42-a").is_dir() and (SMOKE / "v2-2d-smoke-seed42-b").is_dir()),
    reason="local smoke runs not present",
)
def test_25_existing_smoke_bundles_exact_read_only():
    dirs = [SMOKE / "v2-2d-smoke-seed42-a", SMOKE / "v2-2d-smoke-seed42-b"]

    def snapshot():
        return {
            p: hashlib.sha256(p.read_bytes()).hexdigest()
            for d in dirs
            for p in d.rglob("*")
            if p.is_file()
        }

    before = snapshot()
    res = compare_run_bundles(*dirs)
    assert snapshot() == before  # nothing written or modified
    # these runs predate input-hash recording: everything else must match exactly
    assert res["integrity"]["a"]["ok"] and res["integrity"]["b"]["ok"]
    assert res["differences"] == [] and res["not_applicable"] == ["pre.model.checkpoint"]
    assert res["missing"] == [
        "pre.inputs.file_sha256 (not recorded in run a)",
        "pre.inputs.file_sha256 (not recorded in run b)",
    ]
    assert res["level"] == "unavailable"


def test_input_hashes_recorded_and_not_recorded(pair, w):
    _, a, _ = pair
    pre, _ = _load(a)
    assert set(pre["inputs"]["file_sha256"]) == {"case_0001", "case_0002", "case_0003", "case_0004"}
    assert set(pre["inputs"]["file_sha256"]["case_0001"]) == {
        "dwi_sha256",
        "adc_sha256",
        "gt_sha256",
    }
    empty = _edited(a, "pre", "inputs.file_sha256", {})
    res = _contents(empty, empty)
    assert res["level"] == "unavailable" and len(res["missing"]) == 2
