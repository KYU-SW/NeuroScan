"""Final Test protection and procedure (Phase v1 freeze).

Everything runs on a synthetic repository in a temp folder: synthetic patients, a synthetic split
(validation + test), tiny random-weight models and a manifest whose hashes are computed here.
No real test case ID, test file or project checkpoint is used.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import neuroscan_ai.training.dataset as dataset_mod
from neuroscan_ai.data.split_guard import SealedSplitAccessError, SplitGuard
from neuroscan_ai.datasets.base import CaseRecord
from neuroscan_ai.final_test import analysis
from neuroscan_ai.final_test import manifest as mf
from neuroscan_ai.final_test import pipeline as pl
from neuroscan_ai.final_test import run_state as rs
from neuroscan_ai.preprocessing.pipeline import PreprocessConfig, preprocess_case
from neuroscan_ai.research.sources import Isles2022ResearchAdapter, ResearchSourceError
from neuroscan_ai.training.config import config_hash, load_config
from neuroscan_ai.training.model import build_model
from neuroscan_ai.training.trainer import save_checkpoint
from tests.conftest import REPO_ROOT, las_affine, save_nifti

VAL = ["synval1", "synval2", "synval3"]  # synval3: empty GT (negative)
TEST = ["syntest1"]
GIT_ID = ["-c", "user.name=final-test", "-c", "user.email=final-test@example.invalid"]


def git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", *GIT_ID, *args], cwd=root, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def _case(raw: Path, out: Path, cid: str, shift: int, lesion: bool) -> None:
    rng = np.random.default_rng(shift)
    shape, aff = (20, 20, 6), las_affine((2.0, 2.0, 4.0))
    brain = np.zeros(shape, bool)
    brain[3:17, 3:17, :] = True
    les = np.zeros(shape)
    if lesion:
        les[6 + shift : 11 + shift, 7:12, 2:4] = 1.0
    dwi = np.where(brain, rng.normal(100, 5, shape) + 80 * les, 0)
    adc = np.where(brain, rng.normal(800, 40, shape) - 400 * les, 0)
    d = raw / cid
    rec = CaseRecord(
        "synthetic",
        cid,
        {
            "dwi": save_nifti(d / "dwi.nii.gz", dwi, aff),
            "adc": save_nifti(d / "adc.nii.gz", adc, aff),
        },
        label=save_nifti(d / "msk.nii.gz", les, aff),
    )
    preprocess_case(rec, PreprocessConfig("v1", (1.0, 1.0, None)), out)


def _cfg(context: int) -> dict:
    cfg = load_config(REPO_ROOT / "configs" / "2d_baseline.yaml")
    cfg["model"]["channels"], cfg["model"]["strides"] = [8, 16, 32], [2, 2]
    cfg["data"].update(
        processed_root="data/processed/v1",
        split_file="data/splits/split.json",
        case_metadata="results/none.csv",
    )
    if context:
        cfg.update(run_name="25d_syn")
        cfg["data"]["context_slices"] = 1
        cfg["checkpoint"]["dir"], cfg["output"]["results_dir"] = "models/p4", "results/p4"
    return cfg


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("final") / "repo"
    raw, proc = root / "data" / "raw", root / "data" / "processed" / "v1"
    for i, cid in enumerate(VAL + TEST):
        _case(raw, proc, cid, i, lesion=cid != "synval3")
    (root / "data" / "splits").mkdir(parents=True)
    (root / "data/splits/split.json").write_text(
        json.dumps({"version": "syn", "splits": {"train": [], "val": VAL, "test": TEST}}),
        encoding="utf-8",
    )
    man = yaml.safe_load((REPO_ROOT / mf.MANIFEST_PATH).read_text(encoding="utf-8"))
    for key, ctx in (("2d", 0), ("25d", 1)):
        cfg = _cfg(ctx)
        cfg_path = root / "configs" / f"{key}.yaml"
        cfg_path.parent.mkdir(exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        cfg = load_config(cfg_path)
        torch.manual_seed(ctx)
        model = build_model(cfg)
        ck = root / "models" / key / "best.pt"
        opt = torch.optim.AdamW(model.parameters())
        save_checkpoint(
            ck,
            model,
            opt,
            None,
            {"config_hash": config_hash(cfg), "epoch": 1, "in_channels": 2 + 4 * ctx},
        )
        man["models"][key].update(
            checkpoint=f"models/{key}/best.pt",
            checkpoint_sha256=mf.sha256_file(ck),
            training_config=f"configs/{key}.yaml",
            training_config_sha256=mf.sha256_text(cfg_path),
            config_hash=config_hash(cfg),
            best_epoch=1,
            in_channels=2 + 4 * ctx,
            validation_reference=f"results/ref_{key}",
        )
    for name in ("preprocess", "inference"):
        p = root / "configs" / f"{name}.yaml"
        p.write_text(f"name: {name}\n", encoding="utf-8")
        man["configs"][name] = {"path": f"configs/{name}.yaml", "sha256": mf.sha256_text(p)}
    man["split"].update(
        path="data/splits/split.json", sha256=mf.sha256_text(root / "data/splits/split.json")
    )
    man["subgroups"]["metadata"] = "results/none.csv"
    (root / mf.MANIFEST_PATH).write_text(yaml.safe_dump(man, sort_keys=False), encoding="utf-8")
    (root / ".gitignore").write_text("results/\nmodels/\ndata/raw/\ndata/processed/\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "synthetic repo")
    return root


@pytest.fixture
def repo(template, tmp_path) -> Path:
    dst = tmp_path / "repo"
    shutil.copytree(template, dst)
    return dst


@pytest.fixture
def opened(monkeypatch):
    """Record every case loaded and every NIfTI opened."""
    seen: dict[str, list[str]] = {"cases": [], "files": []}
    real_case, real_load = dataset_mod.load_case, nib.load

    def load_case(root, cid, *a, **k):
        seen["cases"].append(cid)
        return real_case(root, cid, *a, **k)

    def load(path, *a, **k):
        seen["files"].append(Path(path).as_posix())
        return real_load(path, *a, **k)

    monkeypatch.setattr(dataset_mod, "load_case", load_case)
    monkeypatch.setattr(nib, "load", load)
    return seen


def freeze(root: Path) -> str:
    """Frozen commit -> manifest-only commit -> lock tag. Returns the approval hash."""
    frozen = git(root, "rev-parse", "HEAD")
    p = root / mf.MANIFEST_PATH
    man = yaml.safe_load(p.read_text(encoding="utf-8"))
    man["git"]["frozen_commit"] = frozen
    p.write_text(yaml.safe_dump(man, sort_keys=False), encoding="utf-8")
    git(root, "commit", "-q", "-am", "freeze manifest")
    git(root, "tag", man["git"]["lock_tag"])
    return mf.sha256_text(p)


def final_opts(root: Path, sha: str | None, **kw) -> pl.RunOptions:
    return pl.RunOptions(
        mode="final-test",
        repo_root=root,
        approve_test=sha,
        confirm_phrase=kw.pop("phrase", mf.CONFIRM_PHRASE),
        device="cpu",
        qa_images=False,
        **kw,
    )


def refused(opts: pl.RunOptions, opened: dict, *names: str) -> None:
    with pytest.raises(pl.PreflightError) as e:
        pl.run(opts)
    failing = {c.name for c in e.value.checks if not c.ok}
    for n in names:
        assert n in failing, (n, failing)
    assert opened["cases"] == [] and not any("syntest" in f for f in opened["files"])
    assert not (opts.repo_root / "results").exists()


# --- dry-run ------------------------------------------------------------------------------------


def test_dry_run_rehearses_everything_on_validation_only(repo, opened):
    res = pl.run(pl.RunOptions(repo_root=repo, device="cpu"))
    out = repo / "results/final_test_dry_run/v1"
    assert res["out"] == out.resolve()
    assert sorted(set(opened["cases"])) == sorted(VAL)
    assert not any("syntest" in f for f in opened["files"])
    for name in (
        "COMPLETED",
        "LOCK",
        "patient_metrics_2d.csv",
        "patient_metrics_25d.csv",
        "paired_metrics.csv",
        "aggregate_metrics.json",
        "subgroup_metrics.json",
        "statistical_comparison.json",
        "negative_case_summary.json",
        "qa_manifest.json",
        "execution_audit.json",
        "environment.json",
        "reference_match.json",
        "predictions_2d",
        "predictions_25d",
        "audit_internal/case_access.json",
    ):
        assert (out / name).exists(), name
    audit = json.loads((out / "execution_audit.json").read_text(encoding="utf-8"))
    assert audit["test_cases_accessed_count"] == 0 and audit["cases_accessed"] == 3
    assert audit["guard_access"]["splits_accessed"] == {"val": 3}
    assert not any(c in json.dumps(audit) for c in VAL + TEST)  # counts only, no case IDs
    pending = {c["name"] for c in audit["preflight_checks"] if not c["ok"]}
    assert pending == {"git_frozen_commit", "git_lock_tag"}  # reported, not blocking
    neg = json.loads((out / "negative_case_summary.json").read_text(encoding="utf-8"))
    assert neg["n_negative"] == 1
    pos = pd.read_csv(out / "paired_metrics.csv")
    assert sorted(pos.case_id) == ["synval1", "synval2"]  # negative never in the positive table
    assert not (repo / "results/final_test").exists()
    assert len(list((out / "qa_images").glob("*.png"))) >= 1


def test_completed_run_is_never_repeated(repo, opened):
    pl.run(pl.RunOptions(repo_root=repo, device="cpu", qa_images=False))
    with pytest.raises(rs.RunStateError, match="completed"):
        pl.run(pl.RunOptions(repo_root=repo, device="cpu", qa_images=False))


@pytest.mark.parametrize(
    "artefact", ["LOCK", "patient_metrics.csv", "aggregate_metrics.json", "predictions_2d"]
)
def test_existing_lock_or_results_refuse(repo, opened, artefact):
    out = repo / "results/final_test_dry_run/v1"
    out.mkdir(parents=True)
    (out / artefact).mkdir() if artefact.startswith("pred") else (out / artefact).write_text("x")
    with pytest.raises(rs.RunStateError, match="existing run artefacts"):
        pl.run(pl.RunOptions(repo_root=repo, device="cpu", qa_images=False))
    assert opened["cases"] == []


def test_dry_run_refuses_test_output_dir_and_approval_args(repo, opened):
    refused(
        pl.RunOptions(repo_root=repo, output_dir=repo / "results/final_test/v1/x"),
        opened,
        "dry_run_output_outside_test_dir",
    )
    refused(
        pl.RunOptions(repo_root=repo, approve_test="abc"), opened, "no_test_approval_in_dry_run"
    )


def test_validation_mode_never_unseals_the_test_split(repo):
    guard = SplitGuard(repo / "data/splits/split.json")
    with pytest.raises(SealedSplitAccessError):
        guard.cases("test")
    with pytest.raises(SealedSplitAccessError):
        guard.assert_no_test(TEST, "validation dry-run")
    assert pl.SPLIT_FOR_MODE == {"validation-dry-run": "val", "final-test": "test"}


# --- final-test approval ------------------------------------------------------------------------


def test_final_test_runs_only_with_full_approval(repo, opened):
    sha = freeze(repo)
    res = pl.run(final_opts(repo, sha))
    out = repo / "results/final_test/v1"
    assert res["out"] == out.resolve() and (out / "COMPLETED").exists() and (out / "LOCK").exists()
    assert sorted(set(opened["cases"])) == TEST
    audit = json.loads((out / "execution_audit.json").read_text(encoding="utf-8"))
    assert audit["mode"] == "final-test" and audit["test_cases_accessed_count"] == 1
    assert all(c["ok"] for c in audit["preflight_checks"])
    assert not (out / "reference_match.json").exists()
    with pytest.raises(rs.RunStateError, match="completed"):
        pl.run(final_opts(repo, sha))


def test_missing_or_wrong_approval_is_refused(repo, opened):
    sha = freeze(repo)
    refused(final_opts(repo, None), opened, "approve_test_matches_manifest_sha256")
    refused(final_opts(repo, "0" * 64), opened, "approve_test_matches_manifest_sha256")
    refused(final_opts(repo, sha, phrase=None), opened, "confirmation_phrase")
    refused(final_opts(repo, sha, phrase="yes"), opened, "confirmation_phrase")
    refused(
        final_opts(repo, sha, output_dir=repo / "results/elsewhere"),
        opened,
        "output_dir_is_manifest_output_dir",
    )


def test_unfrozen_manifest_is_refused(repo, opened):
    sha = mf.sha256_text(repo / mf.MANIFEST_PATH)
    refused(final_opts(repo, sha), opened, "git_frozen_commit", "git_lock_tag")


def test_dirty_tree_is_refused(repo, opened):
    sha = freeze(repo)
    (repo / "notes.txt").write_text("untracked")
    refused(final_opts(repo, sha), opened, "git_clean_tree")


def test_head_and_tag_mismatch_are_refused(repo, opened):
    sha = freeze(repo)
    (repo / "configs/extra.yaml").write_text("x: 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "code change after freeze")
    refused(final_opts(repo, sha), opened, "git_frozen_commit", "git_lock_tag")


def test_tag_on_another_commit_is_refused(repo, opened):
    sha = freeze(repo)
    git(repo, "tag", "-f", "neuroscan-v1.0.0-test-locked", "HEAD~1")
    refused(final_opts(repo, sha), opened, "git_lock_tag")


@pytest.mark.parametrize(
    ("target", "check"),
    [
        ("models/2d/best.pt", "checkpoint_sha256_2d"),
        ("configs/25d.yaml", "config_sha256_25d"),
        ("configs/inference.yaml", "config_sha256_inference"),
        ("data/splits/split.json", "split_sha256"),
    ],
)
def test_changed_artefact_is_refused(repo, opened, target, check):
    sha = freeze(repo)
    with open(repo / target, "ab") as f:
        f.write(b"\n" if target.endswith((".yaml", ".json")) else b"\0")
    if target.endswith((".yaml", ".json")):  # committed change: the hash, not the tree, refuses
        git(repo, "commit", "-q", "-am", "edit")
    refused(final_opts(repo, sha), opened, check)


def test_threshold_change_is_refused(repo, opened):
    sha = freeze(repo)
    p = repo / "configs/2d.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["inference"]["threshold"] = 0.4
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    refused(final_opts(repo, sha), opened, "training_config_policy_2d", "config_sha256_2d")


def test_postprocessing_change_is_refused(repo, opened):
    sha = freeze(repo)
    p = repo / mf.MANIFEST_PATH
    man = yaml.safe_load(p.read_text(encoding="utf-8"))
    man["policy"]["postprocessing"] = "remove_components_below_10_voxels"
    p.write_text(yaml.safe_dump(man, sort_keys=False), encoding="utf-8")
    refused(
        final_opts(repo, sha),
        opened,
        "manifest_policy_matches_code",
        "approve_test_matches_manifest_sha256",
    )
    p2 = repo / "configs/25d.yaml"
    cfg = yaml.safe_load(p2.read_text(encoding="utf-8"))
    cfg["inference"]["postprocessing"] = "largest_component"
    p2.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    refused(final_opts(repo, sha), opened, "training_config_policy_25d")


# --- run state ----------------------------------------------------------------------------------


def test_lock_is_atomic(tmp_path):
    rs.acquire_lock(tmp_path, {"a": 1})
    with pytest.raises(rs.RunStateError, match="exists"):
        rs.acquire_lock(tmp_path, {"a": 2})
    assert json.loads((tmp_path / "LOCK").read_text(encoding="utf-8"))["a"] == 1


def test_failure_is_recorded_and_resume_needs_report_and_reason(repo, opened, tmp_path):
    def boom(*a, **k):
        raise OSError("disk full (simulated)")

    out = repo / "results/final_test_dry_run/v1"
    with pytest.raises(OSError):
        pl.run(pl.RunOptions(repo_root=repo, device="cpu", qa_images=False), evaluate=boom)
    failed = json.loads((out / "FAILED").read_text(encoding="utf-8"))
    assert failed["attempt"] == 1 and "OSError" in failed["reason"]
    assert (out / "LOCK").exists() and (
        out / "failed_attempts/attempt-1/execution_audit.json"
    ).exists()
    assert not list(out.glob(".attempt-*"))

    with pytest.raises(rs.RunStateError, match="resume-after-failure"):
        pl.run(pl.RunOptions(repo_root=repo, device="cpu", qa_images=False))
    report = tmp_path / "failure.txt"
    with pytest.raises(rs.RunStateError, match="failure-report"):
        pl.run(
            pl.RunOptions(
                repo_root=repo,
                device="cpu",
                qa_images=False,
                resume_after_failure=True,
                failure_report=report,
                reason="disk",
            )
        )
    report.write_text("Disk filled up during prediction export; space freed, no code change.")
    with pytest.raises(rs.RunStateError, match="reason"):
        pl.run(
            pl.RunOptions(
                repo_root=repo,
                device="cpu",
                qa_images=False,
                resume_after_failure=True,
                failure_report=report,
                reason=" ",
            )
        )
    res = pl.run(
        pl.RunOptions(
            repo_root=repo,
            device="cpu",
            qa_images=False,
            resume_after_failure=True,
            failure_report=report,
            reason="technical failure: disk full",
        )
    )
    assert res["attempt"] == 2 and (out / "COMPLETED").exists()
    assert (out / "failed_attempts/attempt-1/LOCK").exists()
    assert (out / "failed_attempts/attempt-1/failure_report_for_resume.txt").exists()
    assert res["audit"]["resume"]["reason"] == "technical failure: disk full"


def test_resume_without_failure_is_refused(repo, opened, tmp_path):
    report = tmp_path / "r.txt"
    report.write_text("x")
    with pytest.raises(rs.RunStateError, match="no failed attempt"):
        pl.run(
            pl.RunOptions(
                repo_root=repo, resume_after_failure=True, failure_report=report, reason="r"
            )
        )


# --- separation and analysis --------------------------------------------------------------------


def test_only_the_final_test_package_can_unseal_the_test_split():
    allowed = {"ai/neuroscan_ai/data/split_guard.py"}
    hits = []
    for base in ("ai", "backend", "scripts"):
        for p in (REPO_ROOT / base).rglob("*.py"):
            rel = p.relative_to(REPO_ROOT).as_posix()
            if rel in allowed or rel.startswith("ai/neuroscan_ai/final_test/"):
                continue
            if re.search(r"allow_test\s*=", p.read_text(encoding="utf-8")):
                hits.append(rel)
    assert hits == []


def test_test_prediction_cannot_become_a_research_case(tmp_path):
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {"splits": {"train": [], "val": ["sub-strokecase9001"], "test": ["sub-strokecase9002"]}}
        )
    )
    adapter = Isles2022ResearchAdapter(
        tmp_path / "raw", split, reference_predictions=tmp_path / "results/final_test/v1"
    )
    with pytest.raises(ResearchSourceError) as e:
        adapter.resolve("sub-strokecase9002", "val-001")
    assert e.value.code == "SEALED_TEST_CASE"


def test_final_test_outputs_are_git_ignored():
    for rel in (
        "results/final_test/v1/patient_metrics_2d.csv",
        "results/final_test/v1/predictions_25d/x_pred.nii.gz",
        "results/final_test_dry_run/v1/aggregate_metrics.json",
    ):
        p = subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO_ROOT)
        assert p.returncode == 0, rel


def test_project_manifest_matches_policy_and_contains_no_result():
    man = mf.load_manifest(REPO_ROOT / mf.MANIFEST_PATH)
    assert {k: man["policy"][k] for k in mf.FROZEN_POLICY} == mf.FROZEN_POLICY
    assert man["statistics"]["bootstrap"]["n_boot"] == 10000
    assert man["statistics"]["bootstrap"]["seed"] == 42
    assert man["failure_case"]["below"] == 0.5
    assert man["rules"]["output_dir"] == "results/final_test/v1"
    text = (REPO_ROOT / mf.MANIFEST_PATH).read_text(encoding="utf-8")
    assert "sub-strokecase" not in text
    assert not re.search(r"(?i)\b(mean_dice|dice_mean)\s*:", text)


def _row(cid, dice, has=True, **kw):
    return {
        "case_id": cid,
        "has_lesion": has,
        "dice": dice,
        "iou": dice,
        "sensitivity": dice,
        "precision": kw.get("precision", dice),
        "volume_error_ml": 0.0,
        "absolute_volume_error_ml": 0.0,
        "relative_volume_error_percent": 0.0,
        "component_count_prediction": 1,
        "component_count_gt": 1,
        "negative_outcome": kw.get("neg"),
        "prediction_voxels": kw.get("vox", 0),
        "false_positive_volume_ml": kw.get("fp"),
        "false_positive_component_count": None,
    }


def test_analysis_keeps_negatives_and_undefined_values_apart():
    r2 = [
        _row("a", 0.4, precision=None),
        _row("b", 0.8),
        _row("c", 0.6),
        _row("n", None, False, neg="false_positive_case", vox=5, fp=0.1),
    ]
    r25 = [
        _row("a", 0.6),
        _row("b", 0.7),
        _row("c", 0.6),
        _row("n", None, False, neg="correct_negative"),
    ]
    agg = analysis.aggregate({"2d": r2, "25d": r25})
    assert agg["2d"]["n_positive"] == 3 and agg["2d"]["n_negative"] == 1
    assert agg["2d"]["positive"]["precision"] == {
        **agg["2d"]["positive"]["precision"],
        "n": 2,
        "n_undefined": 1,
    }
    table = analysis.paired_positive_table(r2, r25)
    assert sorted(table.case_id) == ["a", "b", "c"]
    stats = analysis.statistical_comparison(table, 1000, 42)
    assert stats["primary"]["mean_delta"] == pytest.approx((0.2 - 0.1 + 0.0) / 3)
    assert stats["secondary_exploratory"]["precision"]["n_undefined"] == 1
    neg = analysis.negative_summary(r2, r25)
    assert neg["2d"]["correct_negative"] == 0 and neg["25d"]["correct_negative"] == 1
    qa = {q["case_id"]: q["reasons"] for q in analysis.qa_selection(table, neg, 1, 0.5)}
    assert "largest_improvement" in qa["a"] and "dice_below_0.50" in qa["a"]
    assert "largest_deterioration" in qa["b"] and "negative_false_positive_2d" in qa["n"]
    table["grp"] = ["x", "x", "y"]
    sub = analysis.subgroup_summary(table, ["grp"], 10, 1000, 42)["subgroups"]["grp"]
    assert all("descriptive_only" in g for g in sub.values())
