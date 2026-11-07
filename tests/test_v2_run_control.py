"""v2 controlled run flow with the synthetic MockRunner: throwaway git repo, synthetic files only.
No MRI, no model, no checkpoint loading, no real dataset."""

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

from neuroscan_ai.reproducibility import CASE_KEY_POLICY, PathRoots
from neuroscan_ai.reproducibility.environment import (
    capture_determinism,
    capture_environment,
    capture_source_control,
)
from neuroscan_ai.reproducibility.hashing import (
    alias_list_sha256,
    canonical_json_sha256,
    sha256_file,
    sha256_text_lf,
)
from neuroscan_ai.run_control import (
    COMPLETED,
    FAILED_INCOMPLETE,
    FAILED_RUNNER,
    FAILED_VALIDATION,
    GateError,
    MockRunner,
    RunPlanError,
    build_plan,
    load_post_run,
    pre_run_gate,
    run_controlled,
    verify_run,
)
from neuroscan_ai.run_control.outputs import case_path

REPO = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", "user.name=synthetic", "-c", "user.email=synthetic@example.invalid"]
VERSIONS = {
    "torch": "9.9",
    "monai": "1.0",
    "numpy": "2.0",
    "nibabel": "5.0",
    "scipy": "1.0",
}.__getitem__
VAL = ["val-a", "val-b", "val-c"]
POLICIES = {
    "preprocessing": "synthetic_pre_v1",
    "inference": "synthetic_inf_v1",
    "threshold": "fixed_0.5",
    "brain_mask": "synthetic_mask_v1",
    "combination": "single_model",
}


@pytest.fixture
def w(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([*GIT, "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / ".gitignore").write_text("results/\ndata/\nmodels/\n", encoding="utf-8")
    (repo / "cfg.yaml").write_text("threshold: 0.5\n", encoding="utf-8")
    (repo / "requirements-lock.txt").write_text("numpy==2.0\n", encoding="utf-8")
    subprocess.run([*GIT, "-C", str(repo), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    (repo / "data").mkdir()
    (repo / "models").mkdir()
    splits = {
        "splits": {
            "train": ["tr-a", "tr-b"],
            "val": VAL,
            "test": ["te-a"],
            "external": ["ext-a", "soop-pilot-01"],
        }
    }
    (repo / "data/split.json").write_text(json.dumps(splits), encoding="utf-8")
    (repo / "models/best.pt").write_bytes(b"synthetic checkpoint (never loaded)")
    return SimpleNamespace(repo=repo, roots=PathRoots(repo, repo / "models", repo / "data"))


def _env(w):
    return capture_environment(versions=VERSIONS, lock_file=w.repo / "requirements-lock.txt")


def _det():
    return capture_determinism(
        None, seed=None, precision="fp32", amp=False, workers=0, batch_size=2
    )


def _expected(w, aliases=VAL):
    return {
        "config_sha256_lf": sha256_text_lf(w.repo / "cfg.yaml"),
        "split_sha256_lf": sha256_text_lf(w.repo / "data/split.json"),
        "checkpoint_sha256": sha256_file(w.repo / "models/best.pt"),
        "package_lock_sha256_lf": _env(w)["package_lock_sha256_lf"],
        "alias_list_sha256": alias_list_sha256(aliases, CASE_KEY_POLICY),
    }


def _plan(w, **kw):
    args = {
        "run_id": "synthetic-run-01",
        "purpose": "unit test",
        "stage": "validation",
        "dataset_id": "synthetic",
        "split_name": "val",
        "aliases": VAL,
        "allowed_case_count": 3,
        "model_id": "mock_model",
        "checkpoint": w.repo / "models/best.pt",
        "config": w.repo / "cfg.yaml",
        "split_file": w.repo / "data/split.json",
        "policies": POLICIES,
        "output_root": w.repo / "results/v2_runs",
        "expected": _expected(w),
    }
    args.update(kw)
    return build_plan(**args)


def _run(w, plan, runner=None, sc=None):
    return run_controlled(
        plan,
        runner or MockRunner(),
        repo_root=w.repo,
        roots=w.roots,
        source_control=sc or capture_source_control(w.repo),
        environment=_env(w),
        determinism=_det(),
    )


def _gate(w, plan, sc=None):
    return pre_run_gate(
        plan,
        repo_root=w.repo,
        roots=w.roots,
        source_control=sc or capture_source_control(w.repo),
        environment=_env(w),
        determinism=_det(),
    )


# ---------------------------------------------------------------- 1-11 plan and gate
def test_01_pre_run_approved(w):
    g = _gate(w, _plan(w))
    assert g.approved and (g.run_dir / "pre_run_manifest.json").is_file()


def test_02_dirty_git_rejected(w):
    (w.repo / "cfg.yaml").write_text("threshold: 0.4\n", encoding="utf-8")
    plan = _plan(w, expected={**_expected(w)})
    g = _gate(w, plan)
    assert not g.approved and "working_tree_not_clean" in g.reasons


@pytest.mark.parametrize(
    "key", ["config_sha256_lf", "split_sha256_lf", "checkpoint_sha256", "package_lock_sha256_lf"]
)
def test_03_06_hash_mismatch_rejected(w, key):
    g = _gate(w, _plan(w, expected={**_expected(w), key: "0" * 64}))
    assert not g.approved and f"{key}_mismatch" in g.reasons


def test_gate_requires_declared_hashes(w):
    g = _gate(w, _plan(w, expected={}))
    assert not g.approved and any(r.startswith("expected_hashes_not_declared") for r in g.reasons)


@pytest.mark.parametrize(
    "kw",
    [
        {
            "stage": "validation",
            "split_name": "train",
            "aliases": ["tr-a", "tr-b"],
            "allowed_case_count": 2,
        },
        {"stage": "train", "split_name": "val"},
        {
            "stage": "inference",
            "split_name": "external",
            "aliases": ["ext-a"],
            "allowed_case_count": 1,
        },
    ],
)
def test_07_split_not_allowed(w, kw):
    with pytest.raises(RunPlanError):
        _plan(w, **kw)


def test_07b_explicitly_approved_inference_split(w):
    plan = _plan(
        w,
        stage="inference",
        split_name="external",
        aliases=["ext-a"],
        allowed_case_count=1,
        approved_splits=["external"],
    )
    assert plan.split_name == "external"


@pytest.mark.parametrize("split", ["test", "final_test"])
def test_08_test_and_final_test_refused(w, split):
    with pytest.raises(RunPlanError):
        _plan(
            w,
            stage="inference",
            split_name=split,
            aliases=["te-a"],
            allowed_case_count=1,
            approved_splits=[split],
        )


def test_09_soop_pilot_refused(w):
    with pytest.raises(RunPlanError):
        _plan(
            w,
            stage="external_validation",
            split_name="external",
            aliases=["ext-a", "soop-pilot-01"],
            allowed_case_count=2,
            approved_splits=["external"],
        )


def test_plan_cases_must_belong_to_split(w):
    with pytest.raises(RunPlanError):
        _plan(w, aliases=["val-a", "tr-a", "val-c"])
    with pytest.raises(RunPlanError):
        _plan(w, allowed_case_count=2)


def test_10_existing_output_dir_rejected(w):
    plan = _plan(w)
    plan.run_dir.mkdir(parents=True)
    g = _gate(w, plan)
    assert not g.approved and "output_dir_exists" in g.reasons


def test_11_runner_without_pre_run_refused(w):
    plan = _plan(w)
    with pytest.raises(GateError):
        MockRunner().run(None, plan)
    rejected = _gate(w, _plan(w, run_id="synthetic-run-02", expected={}))
    with pytest.raises(GateError):
        MockRunner().run(rejected, plan)
    assert not (plan.run_dir / "probability_maps").exists()


def test_11b_tampered_pre_run_refused(w):
    plan = _plan(w)
    g = _gate(w, plan)
    p = g.run_dir / "pre_run_manifest.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["identity"]["purpose"] = "changed"
    p.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(GateError):
        MockRunner().run(g, plan)


# ---------------------------------------------------------------- 12-18 mock run
def test_12_18_mock_run_completed(w):
    plan = _plan(w)
    res = _run(w, plan)
    assert res["status"] == COMPLETED and res["exit_code"] == 0
    for kind in (
        "probability_maps",
        "ungated_predictions",
        "brain_masks",
        "gated_predictions",
    ):  # 13-16
        assert case_path(plan.run_dir, kind, "case_0001").is_file()
    prob = np.asanyarray(
        nib.load(str(case_path(plan.run_dir, "probability_maps", "case_0002"))).dataobj
    )
    ung = np.asanyarray(
        nib.load(str(case_path(plan.run_dir, "ungated_predictions", "case_0002"))).dataobj
    )
    mask = np.asanyarray(nib.load(str(case_path(plan.run_dir, "brain_masks", "case_0002"))).dataobj)
    gat = np.asanyarray(
        nib.load(str(case_path(plan.run_dir, "gated_predictions", "case_0002"))).dataobj
    )
    assert np.array_equal(ung == 1, prob >= 0.5)  # 17
    assert np.array_equal(gat == 1, (ung == 1) & (mask == 1)) and gat.sum() < ung.sum()  # 18


# ---------------------------------------------------------------- 19-24 missing outputs
@pytest.mark.parametrize(
    "kind",
    [
        "probability_maps",
        "ungated_predictions",
        "brain_masks",
        "gated_predictions",
        "patient_metrics",
        "aggregate_results",
    ],
)
def test_19_24_missing_output_fails(w, kind):
    res = _run(w, _plan(w), MockRunner(faults={f"skip_{kind}"}))
    assert res["status"] == FAILED_INCOMPLETE and res["exit_code"] == 1
    assert any(kind in m for m in res["missing_outputs"])


# ---------------------------------------------------------------- 25-28 relation checks
@pytest.mark.parametrize(
    "fault,code",
    [
        ("nan_probability", "probability_nonfinite"),
        ("non_binary_mask", "non_binary:brain_mask"),
        ("shape_mismatch", "shape_mismatch:gated_predictions"),
        ("affine_mismatch", "affine_mismatch:gated_predictions"),
    ],
)
def test_25_28_invalid_outputs_fail(w, fault, code):
    plan = _plan(w)
    res = _run(w, plan, MockRunner(faults={fault}))
    assert res["status"] == FAILED_VALIDATION and f"case_0001:{code}" in res["output_check_errors"]
    assert res["output_check_errors"] == [f"case_0001:{code}"]  # nothing silently repaired


# ---------------------------------------------------------------- 29-32 manifests and reuse
def test_29_failed_run_post_manifest(w):
    plan = _plan(w)
    res = _run(w, plan, MockRunner(faults={"raise"}))
    post = load_post_run(plan.run_dir)
    r = post["reproducibility"]
    assert res["status"] == FAILED_RUNNER and post["run"]["status"] == FAILED_RUNNER
    assert (
        post["run"]["exit_code"] == 1
        and r["failed_stage"] == "runner"
        and r["error_type"] == "RuntimeError"
    )
    assert r["accessed_splits"] == ["val"] and post["run"]["cases_failed"] == 3
    assert len(r["missing_outputs"]) == 3 * 5 + 1


def test_30_31_pre_post_linked_and_output_hashes(w):
    plan = _plan(w)
    res = _run(w, plan)
    pre = json.loads((plan.run_dir / "pre_run_manifest.json").read_text(encoding="utf-8"))
    post = load_post_run(plan.run_dir)
    assert post["pre_run_manifest_sha256"] == canonical_json_sha256(pre) == res["pre_run_sha256"]
    assert post["outputs"]["probability_maps"]["case_0003"] == sha256_file(
        case_path(plan.run_dir, "probability_maps", "case_0003")
    )
    assert set(post["outputs"]["aggregate_results"]) == {"aggregate"}


def test_32_run_id_reuse_refused(w):
    plan = _plan(w)
    _run(w, plan, MockRunner(faults={"raise"}))
    before = {p: p.read_bytes() for p in plan.run_dir.rglob("*") if p.is_file()}
    res = _run(w, plan)
    assert res["pre_run_status"] == "rejected" and "output_dir_exists" in res["reasons"]
    assert {p: p.read_bytes() for p in plan.run_dir.rglob("*") if p.is_file()} == before


def test_rejection_recorded(w):
    plan = _plan(w, expected={**_expected(w), "checkpoint_sha256": "0" * 64})
    res = _run(w, plan)
    rec = json.loads((plan.run_dir / "gate_rejection.json").read_text(encoding="utf-8"))
    assert res["pre_run_status"] == "rejected" and rec["reasons"] == ["checkpoint_sha256_mismatch"]


# ---------------------------------------------------------------- 33-35 privacy
def test_33_35_no_alias_id_or_path_in_manifests_or_files(w):
    plan = _plan(w)
    _run(w, plan)
    texts = [p.read_text(encoding="utf-8") for p in plan.run_dir.rglob("*.json")]
    names = [p.name for p in plan.run_dir.rglob("*")]
    joined = "\n".join(texts + names)
    assert not any(a in joined for a in VAL)  # 33
    assert "sub-" not in joined and "strokecase" not in joined.lower()  # 34
    assert str(w.repo) not in joined and w.repo.as_posix() not in joined  # 35
    assert str(Path.home()) not in joined


def test_runner_error_message_redacted(w):
    class Leaky(MockRunner):
        def run(self, gate, plan):
            raise RuntimeError(f"failed at {w.repo} for val-a")

    plan = _plan(w)
    _run(w, plan, Leaky())
    assert load_post_run(plan.run_dir)["reproducibility"]["error_message"] == "[redacted]"


# ---------------------------------------------------------------- 36-40 isolation
def test_36_39_no_data_model_torch_or_v1_code():
    prefixes = (
        "torch",
        "monai",
        "neuroscan_ai.training",
        "neuroscan_ai.inference",
        "neuroscan_ai.preprocessing",
        "neuroscan_ai.final_test",
        "neuroscan_ai.analysis",
    )
    code = (
        "import sys; sys.path.insert(0, 'ai'); import neuroscan_ai.run_control\n"
        f"print([m for m in sys.modules if m.startswith({prefixes!r})])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
    src = "\n".join(
        p.read_text(encoding="utf-8") for p in (REPO / "ai/neuroscan_ai/run_control").glob("*.py")
    )
    for forbidden in (
        "torch.load",
        "load_state_dict",
        "neuroscan_ai.training",
        "neuroscan_ai.inference",
        "data/raw",
        "ISLES",
        "soop_format",
    ):
        assert forbidden not in src


def test_40_output_root_git_ignored():
    assert (
        subprocess.run(
            ["git", "-C", str(REPO), "check-ignore", "-q", "results/v2_runs/r/x.json"]
        ).returncode
        == 0
    )


def test_cli_refuses_without_mock(tmp_path):
    from scripts.run_v2_controlled import main

    assert main(["--plan", str(tmp_path / "plan.json")]) == 2


def test_post_manifest_is_copy_safe(w):
    plan = _plan(w)
    _run(w, plan)
    post = load_post_run(plan.run_dir)
    assert copy.deepcopy(post)["schema"] == "neuroscan-run-manifest-v1"


# ---------------------------------------------------------------- determinism and tamper checks
def _output_hashes(run_dir):
    post = load_post_run(run_dir)
    return {k: v for k, v in post["outputs"].items() if v}


def test_det_same_seed_identical_outputs(w):
    a, b = _plan(w, run_id="det-run-a"), _plan(w, run_id="det-run-b")
    assert _run(w, a, MockRunner(seed=7))["status"] == COMPLETED
    assert _run(w, b, MockRunner(seed=7))["status"] == COMPLETED
    ha, hb = _output_hashes(a.run_dir), _output_hashes(b.run_dir)
    assert ha == hb and len(ha["probability_maps"]) == 3


def test_det_different_seed_different_outputs(w):
    a, b = _plan(w, run_id="det-run-a"), _plan(w, run_id="det-run-b")
    _run(w, a, MockRunner(seed=7))
    _run(w, b, MockRunner(seed=8))
    ha, hb = _output_hashes(a.run_dir), _output_hashes(b.run_dir)
    assert ha["probability_maps"] != hb["probability_maps"]
    assert ha["aggregate_results"] != hb["aggregate_results"]


def test_det_seed_in_pre_and_post_manifest(w):
    plan = _plan(w)
    _run(w, plan, MockRunner(seed=11))
    pre = json.loads((plan.run_dir / "pre_run_manifest.json").read_text(encoding="utf-8"))
    assert pre["determinism"]["seed"] == 11 == load_post_run(plan.run_dir)["determinism"]["seed"]


def test_det_conflicting_seed_refused(w):
    plan = _plan(w)
    with pytest.raises(ValueError):
        run_controlled(
            plan,
            MockRunner(seed=3),
            repo_root=w.repo,
            roots=w.roots,
            source_control=capture_source_control(w.repo),
            environment=_env(w),
            determinism={**_det(), "seed": 4},
        )
    assert not plan.run_dir.exists()


def test_single_aggregate_per_run(w):
    plan = _plan(w)
    _run(w, plan)
    assert [p.name for p in (plan.run_dir / "aggregate_results").iterdir()] == ["aggregate.json"]
    assert list(load_post_run(plan.run_dir)["outputs"]["aggregate_results"]) == ["aggregate"]


def test_verify_untouched_run_ok(w):
    plan = _plan(w)
    _run(w, plan)
    assert verify_run(plan.run_dir) == {"ok": True, "problems": []}


def test_verify_failed_run_keeps_audit(w):
    plan = _plan(w)
    _run(w, plan, MockRunner(faults={"raise"}))
    res = verify_run(plan.run_dir)
    assert res["ok"] and load_post_run(plan.run_dir)["run"]["status"] == FAILED_RUNNER


@pytest.mark.parametrize("target", ["probability", "aggregate", "post", "pre"])
def test_verify_detects_tampering(w, target):
    plan = _plan(w)
    _run(w, plan)
    if target == "probability":
        p = case_path(plan.run_dir, "probability_maps", "case_0002")
        p.write_bytes(p.read_bytes() + b"x")
    elif target == "aggregate":
        p = plan.run_dir / "aggregate_results/aggregate.json"
        p.write_text(
            p.read_text(encoding="utf-8").replace('"cases": 3', '"cases": 4'), encoding="utf-8"
        )
    else:  # edit a manifest after the run (post: seal no longer matches; pre: link broken)
        p = plan.run_dir / f"{target}_run_manifest.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        d["identity"]["purpose"] = "edited after the run"
        p.write_text(json.dumps(d), encoding="utf-8")
    res = verify_run(plan.run_dir)
    assert not res["ok"] and res["problems"]
