"""Epoch progress, resume checkpoint, safe interruption (signals, host memory guard) and strict
resume of the v2 2D baseline runner on SYNTHETIC volumes only (CPU, throwaway git repo, no MRI,
no ISLES / SOOP access). Synthetic results are not performance results.

Exact-match design: A = 4 epochs without interruption; B = interrupted after epoch 2 (SIGTERM),
explicitly resumed for epochs 3-4. Every result output (best / last checkpoint, history,
validation metrics, sampler record, per-case probability / ungated / brain mask / gated,
aggregate) must be byte-identical; timing, progress and resource records are excluded."""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import shutil
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neuroscan_ai.reproducibility.environment import (
    capture_determinism,
    capture_source_control,
)
from neuroscan_ai.reproducibility.hashing import sha256_file
from neuroscan_ai.reproducibility.validation import EXACT
from neuroscan_ai.run_control import COMPLETED, compare_run_bundles, load_post_run, verify_run
from neuroscan_ai.run_control.lifecycle import (
    INTERRUPTED,
    INTERRUPTED_EXIT_CODE,
    INTERRUPTED_RESOURCE_GUARD,
    POST_RUN_NAME,
    ResumeRefusedError,
)
from neuroscan_ai.run_control.outputs import case_path, run_output_path
from neuroscan_ai.v2 import baseline_cache as bc
from neuroscan_ai.v2 import baseline_progress as bp
from neuroscan_ai.v2 import baseline_resume as brs
from neuroscan_ai.v2 import baseline_runner as br
from neuroscan_ai.v2 import baseline_training as bt
from neuroscan_ai.v2 import resource_monitor as rm
from tests.test_v2_baseline_runner import (
    FORBIDDEN,
    GIT,
    REPO,
    TRAIN,
    VAL,
    _cache,
    _env,
    _plan_kwargs,
    _write_case,
)

GIB = 1024**3
EPOCHS = 4
RESULT_LABELS = (
    "best_checkpoint",
    "last_checkpoint",
    "training_history",
    "validation_metrics",
    "sampler_record",
    "threshold_record",
    "preprocessing_record",
    "aggregate",
)
PER_CASE = (
    "probability_maps",
    "ungated_predictions",
    "brain_masks",
    "gated_predictions",
    "patient_metrics",
)


def _det():
    return capture_determinism(
        torch, seed=None, precision="fp32", amp=False, workers=0, batch_size=32
    )


def _run(w, run_id, memory_probe=None, cache=None, sc=None, **kw):
    plan_kw = _plan_kwargs(w, run_id, **kw)
    return br.run_baseline(
        w.cfg,
        plan_kw,
        cache or _cache(w, w.cfg, plan_kw["aliases"], plan_kw["validation_aliases"]),
        repo_root=w.repo,
        roots=w.roots,
        source_control=sc or capture_source_control(w.repo),
        environment=_env(w),
        determinism=_det(),
        forbidden_ids=FORBIDDEN,
        device="cpu",
        epoch_limit=EPOCHS,
        memory_probe=memory_probe,
    )


def _resume(
    w, run_id, sc=None, env=None, cfg=None, cache=None, epoch_limit=EPOCHS, memory_probe=None, **kw
):
    plan_kw = _plan_kwargs(w, run_id, **kw)
    if cfg is not None:
        plan_kw["expected"] = {}
    return br.resume_baseline(
        cfg or w.cfg,
        plan_kw,
        cache or _cache(w, w.cfg, plan_kw["aliases"], plan_kw["validation_aliases"]),
        repo_root=w.repo,
        source_control=sc or capture_source_control(w.repo),
        environment=env or _env(w),
        determinism=_det(),
        forbidden_ids=FORBIDDEN,
        device="cpu",
        epoch_limit=epoch_limit,
        memory_probe=memory_probe,
    )


def _signal_in_epoch(mp, epoch, sig=None, hard=False):
    """Patch train_one_epoch: raise a signal (or a hard KeyboardInterrupt) during ``epoch``."""
    real = bt.train_one_epoch
    calls = {"n": 0}

    def wrapped(model, batches, opt, device):
        calls["n"] += 1
        if calls["n"] == epoch:
            if hard:
                raise KeyboardInterrupt("forced stop inside the epoch (simulated kill)")
            signal.raise_signal(sig)
        return real(model, batches, opt, device)

    mp.setattr(bt, "train_one_epoch", wrapped)
    return calls


def _rd(w, run_id):
    return w.repo / "results/v2_runs" / run_id


def _tree(path: Path) -> dict[str, str]:
    return {
        p.relative_to(path).as_posix(): sha256_file(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(scope="module")
def w(tmp_path_factory):
    root = tmp_path_factory.mktemp("resume")
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
    splits = {"splits": {"train": TRAIN, "val": VAL, "test": ["syn-test-01"]}}
    (data / "split.json").write_text(json.dumps(splits), encoding="utf-8")
    files = {a: _write_case(data, a, i) for i, a in enumerate(TRAIN + VAL)}
    from neuroscan_ai.reproducibility import PathRoots

    ns = SimpleNamespace(
        repo=repo, data=data, files=files, cfg=repo / "configs/v2/2d_baseline.yaml"
    )
    ns.roots = PathRoots(repo, repo / "models", data)
    ns.caches = {}
    ns.handlers_before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    # A: continuous
    ns.a = _run(ns, "syn-cont")
    with pytest.MonkeyPatch.context() as mp:
        ns.sigterm_calls = _signal_in_epoch(mp, 2, signal.SIGTERM)
        ns.b1 = _run(ns, "syn-sigterm")  # interrupted after epoch 2
    ns.b1_tree = _tree(_rd(ns, "syn-sigterm"))
    ns.b1_progress = json.loads((_rd(ns, "syn-sigterm") / bp.PROGRESS_NAME).read_text("utf-8"))
    ns.b1_payload = brs.load_resume_checkpoint(_rd(ns, "syn-sigterm") / brs.RESUME_NAME)
    ns.b2 = _resume(ns, "syn-sigterm")
    with pytest.MonkeyPatch.context() as mp:
        _signal_in_epoch(mp, 3, signal.SIGINT)
        ns.k1 = _run(ns, "syn-keyboard")  # KeyboardInterrupt (SIGINT) after epoch 3
    ns.k2 = _resume(ns, "syn-keyboard")
    probes = iter([3 * GIB] + [16 * GIB] * 99)
    ns.g1 = _run(ns, "syn-guard", memory_probe=lambda: next(probes))  # guard after epoch 1
    ns.g1_tree = _tree(_rd(ns, "syn-guard"))
    ns.g2 = _resume(ns, "syn-guard", memory_probe=lambda: 16 * GIB)
    with pytest.MonkeyPatch.context() as mp:
        _signal_in_epoch(mp, 3, hard=True)
        with pytest.raises(KeyboardInterrupt):
            _run(ns, "syn-killed")  # forced stop inside epoch 3: nothing written then
    ns.killed_tree = _tree(_rd(ns, "syn-killed"))
    ns.kill2 = _resume(ns, "syn-killed")
    # pending: interrupted after epoch 1 and never resumed (target of the refusal tests)
    with pytest.MonkeyPatch.context() as mp:
        _signal_in_epoch(mp, 1, signal.SIGTERM)
        ns.pending = _run(ns, "syn-pending")
    # like the real Run A: killed in epoch 1 before any checkpoint
    with pytest.MonkeyPatch.context() as mp:
        _signal_in_epoch(mp, 1, hard=True)
        with pytest.raises(KeyboardInterrupt):
            _run(ns, "syn-a-like")
    # another interrupted run (target of the foreign-checkpoint test)
    with pytest.MonkeyPatch.context() as mp:
        _signal_in_epoch(mp, 1, signal.SIGTERM)
        ns.foreign = _run(ns, "syn-foreign")
    # interrupted twice: segment_01 after epoch 1, segment_02 after epoch 2, then completed
    runs = ns.repo / "results/v2_runs"
    with pytest.MonkeyPatch.context() as mp:
        _signal_in_epoch(mp, 1, signal.SIGTERM)
        ns.t1 = _run(ns, "syn-twice")
    ns.twice_seg1 = _tree(_rd(ns, "syn-twice") / "segments/segment_01")
    ns.dirs_before_resume = sorted(p.name for p in runs.iterdir())
    with pytest.MonkeyPatch.context() as mp:
        _signal_in_epoch(mp, 1, signal.SIGTERM)  # first epoch of the resumed segment (epoch 2)
        ns.t2 = _resume(ns, "syn-twice")
    ns.twice_seg1_after_2 = _tree(_rd(ns, "syn-twice") / "segments/segment_01")
    ns.twice_seg2 = _tree(_rd(ns, "syn-twice") / "segments/segment_02")
    ns.t3 = _resume(ns, "syn-twice")
    ns.dirs_after_resume = sorted(p.name for p in runs.iterdir())
    return ns


def _assert_exact(w, run_a, run_b):
    a, b = _rd(w, run_a), _rd(w, run_b)
    assert compare_run_bundles(a, b)["level"] == EXACT
    for label in RESULT_LABELS:
        assert sha256_file(run_output_path(a, label)) == sha256_file(run_output_path(b, label)), (
            label
        )
    keys = sorted(p.stem for p in (a / "patient_metrics").glob("*.json"))
    for kind in PER_CASE:
        for k in keys:
            assert sha256_file(case_path(a, kind, k)) == sha256_file(case_path(b, kind, k))
    ma = torch.load(run_output_path(a, "last_checkpoint"), weights_only=False)["model_state"]
    mb = torch.load(run_output_path(b, "last_checkpoint"), weights_only=False)["model_state"]
    assert all(torch.equal(ma[k], mb[k]) for k in ma)  # final model state


# ---------------------------------------------------------------- 13-15 progress
def test_13_14_15_progress_atomic_per_epoch_and_private(w, tmp_path, monkeypatch):
    # atomic write: tmp + flush + fsync + os.replace (never a partial file)
    calls = []
    import os

    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(
        os, "replace", lambda a, b: (calls.append("replace"), real_replace(a, b))[1]
    )
    target = tmp_path / "training/progress.json"
    bp.atomic_write_bytes(target, b"{}")
    assert calls == ["fsync", "replace"] and not (tmp_path / "training/progress.json.tmp").exists()
    monkeypatch.undo()
    # a failure before os.replace keeps the previous complete file
    target.write_text('{"old": true}', encoding="utf-8")
    monkeypatch.setattr(os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("killed")))
    with pytest.raises(OSError):
        bp.atomic_write_bytes(target, b'{"new": true}')
    monkeypatch.undo()
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    # per epoch: the interrupted run's progress describes its last completed epoch
    p = w.b1_progress
    assert tuple(p) == bp.PROGRESS_FIELDS
    assert p["completed_epoch"] == 2 and p["status"] == INTERRUPTED and p["max_epochs"] == EPOCHS
    assert p["early_stopping_counter"] == 2 - p["best_epoch"]
    final = json.loads((_rd(w, "syn-cont") / bp.PROGRESS_NAME).read_text(encoding="utf-8"))
    assert final["status"] == "training_completed" and final["completed_epoch"] == EPOCHS
    hist = json.loads(run_output_path(_rd(w, "syn-cont"), "training_history").read_text("utf-8"))
    assert final["latest_validation_loss"] == hist["epochs"][-1]["validation_loss_total"]
    # privacy: no alias, subject ID, absolute path; refused before writing
    text = json.dumps(final)
    for n in (*TRAIN, *VAL, *FORBIDDEN, str(w.repo), w.repo.as_posix(), str(Path.home())):
        assert n not in text
    rec = dict(final, run_id=TRAIN[0])
    with pytest.raises(bp.ProgressRecordError):
        bp.write_progress(tmp_path / "p.json", rec, TRAIN)
    with pytest.raises(bp.ProgressRecordError):
        bp.write_progress(tmp_path / "p.json", dict(final, run_id="C:\\x"))
    # progress is never part of a result hash
    post = load_post_run(_rd(w, "syn-cont"))
    assert "progress" not in json.dumps(post["outputs"])


def test_14_progress_updated_every_epoch(w, monkeypatch):
    seen = []
    real = bp.write_progress

    def spy(path, record, needles=()):
        seen.append((record["completed_epoch"], record["status"]))
        return real(path, record, needles)

    monkeypatch.setattr(bp, "write_progress", spy)
    res = _run(w, "syn-progress-spy")
    assert res["status"] == COMPLETED
    assert seen == [
        (1, "running"),
        (2, "running"),
        (3, "running"),
        (4, "running"),
        (4, "training_completed"),
    ]


# ---------------------------------------------------------------- 16-21 checkpoint, RNG, sampler
def test_16_resume_checkpoint_atomic_and_self_verifying(w, tmp_path):
    rd = _rd(w, "syn-sigterm")
    assert not (rd / "training" / brs.RESUME_TMP_NAME).exists()
    payload = w.b1_payload
    assert payload["completed_epoch"] == 2 and payload["next_epoch"] == 3
    for k in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "best_state",
        "best_selection",
        "best_epoch",
        "early_stopping_counter",
        "history",
        "sampling",
        "rng_state",
        *brs.IDENTITY_FIELDS,
    ):
        assert k in payload, k
    meta = json.dumps(
        {k: v for k, v in payload.items() if not k.endswith(("_state", "rng_state"))}, default=str
    )
    for n in (*TRAIN, *VAL, *FORBIDDEN, str(w.repo), w.repo.as_posix()):
        assert n not in meta
    # snapshots are device-independent (every tensor on the CPU)
    for part in ("best_state", "optimizer_state"):
        tensors = []
        stack = [payload[part]]
        while stack:
            x = stack.pop()
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, dict):
                stack += list(x.values())
            elif isinstance(x, (list, tuple)):
                stack += list(x)
        assert tensors and all(t.device.type == "cpu" for t in tensors), part
    # corrupted payload -> refused
    bad = tmp_path / "ck.pt"
    shutil.copy(rd / brs.RESUME_NAME, bad)
    env = torch.load(bad, weights_only=True)
    env["payload"] = env["payload"][:-1] + bytes([env["payload"][-1] ^ 1])
    torch.save(env, bad)
    with pytest.raises(brs.ResumeRefused, match="hash_invalid"):
        brs.load_resume_checkpoint(bad)


def test_17_18_rng_state_saved_and_restored():
    import random

    import numpy as np

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    s = brs.capture_rng()
    a = (torch.rand(3), np.random.rand(3), random.random())
    torch.rand(5), np.random.rand(5), random.random()  # advance
    brs.restore_rng(s)
    b = (torch.rand(3), np.random.rand(3), random.random())
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    brs.restore_rng(s)
    assert brs.rng_fingerprint(s) == brs.rng_fingerprint(brs.capture_rng())


def test_17_rng_in_checkpoint_matches_continuous_run(w):
    """RNG state stored after epoch 2 of the interrupted run equals the continuous run's state
    after epoch 2 (nothing in training consumes global RNG differently)."""
    pa = brs.load_resume_checkpoint(_rd(w, "syn-cont") / brs.RESUME_NAME)
    assert brs.rng_fingerprint(pa["rng_state"]) == brs.rng_fingerprint(w.b1_payload["rng_state"])


def test_19_20_21_sampler_state_saved_restored_and_hash_equal(w):
    p = w.b1_payload
    assert [r["epoch"] for r in p["sampling"]] == [1, 2]
    assert p["last_sampling_list_sha256"] == p["sampling"][-1]["slice_list_sha256"]
    sa = json.loads(run_output_path(_rd(w, "syn-cont"), "sampler_record").read_text("utf-8"))
    sb = json.loads(run_output_path(_rd(w, "syn-sigterm"), "sampler_record").read_text("utf-8"))
    assert sa == sb and [e["epoch"] for e in sb["epochs"]] == [1, 2, 3, 4]
    assert [e["slice_list_sha256"] for e in sa["epochs"]] == [
        e["slice_list_sha256"] for e in sb["epochs"]
    ]


# ---------------------------------------------------------------- 22 exact match
def test_22_continuous_and_resumed_runs_exact_match(w):
    assert w.a["status"] == COMPLETED
    assert w.b1["status"] == INTERRUPTED and w.b1["exit_code"] == INTERRUPTED_EXIT_CODE
    assert w.b2["status"] == COMPLETED and w.b2["resumed"] is True
    assert verify_run(_rd(w, "syn-sigterm"))["ok"]
    assert verify_run(_rd(w, "syn-sigterm"), segment="segment_01")["ok"]
    _assert_exact(w, "syn-cont", "syn-sigterm")
    # the resumed post-run records the resume (audit field, not compared)
    post = load_post_run(_rd(w, "syn-sigterm"))
    assert post["reproducibility"]["resumed"] is True
    assert post["reproducibility"]["interrupted_segments"] == 1


def test_22b_other_interruptions_also_exact_match(w):
    for rid, first in (("syn-keyboard", w.k1), ("syn-guard", w.g1), ("syn-killed", None)):
        if first is not None:
            assert first["exit_code"] == INTERRUPTED_EXIT_CODE
        assert load_post_run(_rd(w, rid))["run"]["status"] == COMPLETED
        _assert_exact(w, "syn-cont", rid)


# ---------------------------------------------------------------- 23-30 resume refusals
def _refused(w, fn) -> tuple[str, ...]:
    before = _tree(_rd(w, "syn-pending"))
    with pytest.raises(ResumeRefusedError) as e:
        fn()
    assert _tree(_rd(w, "syn-pending")) == before  # a refusal writes nothing
    return e.value.reasons


def test_23_git_commit_mismatch_refused(w):
    sc = {**capture_source_control(w.repo), "git_commit": "0" * 40}
    assert "git_commit_mismatch" in _refused(w, lambda: _resume(w, "syn-pending", sc=sc))
    dirty = {**capture_source_control(w.repo), "working_tree_clean": False}
    assert "working_tree_not_clean" in _refused(w, lambda: _resume(w, "syn-pending", sc=dirty))


def test_24_config_hash_mismatch_refused(w, tmp_path):
    cfg = tmp_path / "2d_baseline.yaml"
    cfg.write_text(w.cfg.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    r = _refused(w, lambda: _resume(w, "syn-pending", cfg=cfg))
    assert "config_hash_mismatch" in r


def test_25_split_hash_mismatch_refused(w, tmp_path):
    s = tmp_path / "split.json"
    s.write_text(
        json.dumps(json.loads((w.data / "split.json").read_text("utf-8")), indent=1),
        encoding="utf-8",
    )
    assert "split_hash_mismatch" in _refused(w, lambda: _resume(w, "syn-pending", split_file=s))


def test_26_input_hash_mismatch_refused(w):
    kw = _plan_kwargs(w, "syn-pending")
    inputs = dict(kw["input_files"])
    a = TRAIN[0]
    inputs[a] = {**inputs[a], "dwi": inputs[a]["adc"]}  # another file's content
    assert "input_hash_mismatch" in _refused(
        w, lambda: _resume(w, "syn-pending", input_files=inputs)
    )


def test_27_cache_hash_mismatch_refused(w):
    other = w.repo / "results/v2_cache/syn_other_version"
    if not other.exists():
        bc.build_cache(
            other,
            config=br.prepare_config(w.cfg),
            split_file=w.data / "split.json",
            files={"train": {a: w.files[a] for a in TRAIN}, "val": {a: w.files[a] for a in VAL}},
            git_commit=capture_source_control(w.repo)["git_commit"],
            package_lock_sha256_lf=_env(w)["package_lock_sha256_lf"],
        )
    r = _refused(w, lambda: _resume(w, "syn-pending", cache=other))
    assert r == ("cache_index_sha256_mismatch",)  # valid cache, but not the run's cache
    bad = w.repo / "results/v2_cache/syn_tampered"
    if not bad.exists():
        shutil.copytree(other, bad)
        p = next(bad.glob("case_*/dwi.npy"))
        raw = bytearray(p.read_bytes())
        raw[-1] ^= 0xFF
        p.write_bytes(bytes(raw))
    r = _refused(w, lambda: _resume(w, "syn-pending", cache=bad))
    assert r[0] == "cache_integrity_failed"


def test_28_environment_mismatch_refused(w):
    env = _env(w)
    for field, value in (
        ("packages", {**env["packages"], "torch": "0.0"}),
        ("cudnn", "other"),
        ("gpu_driver_version", "other"),
        ("gpu_name", "other"),
        ("cuda_runtime", "other"),
    ):
        r = _refused(w, lambda f=field, v=value: _resume(w, "syn-pending", env={**env, f: v}))
        assert "environment_mismatch" in r
    r = _refused(
        w, lambda: _resume(w, "syn-pending", env={**env, "package_lock_sha256_lf": "0" * 64})
    )
    assert "package_lock_hash_mismatch" in r


def test_29_completed_run_resume_refused(w):
    with pytest.raises(ResumeRefusedError) as e:
        _resume(w, "syn-cont")
    assert e.value.reasons == ("run_already_finalised",)


def test_30_run_without_resume_checkpoint_refused(w):
    rd = _rd(w, "syn-a-like")
    names = set(_tree(rd))
    assert names == {"pre_run_manifest.json", "training/preprocessing_record.json"}
    with pytest.raises(ResumeRefusedError) as e:
        _resume(w, "syn-a-like")
    assert {"resume_checkpoint_missing", "progress_missing"} <= set(e.value.reasons)
    assert set(_tree(rd)) == names
    # the real externally terminated run ID is retired: never resumed, never reused
    with pytest.raises(ResumeRefusedError, match="run_id_retired"):
        _resume(w, "v2-2d-baseline-seed42-a")
    with pytest.raises(bt.ConfigValidationError, match="retired"):
        _run(w, "v2-2d-baseline-seed42-a")
    assert not _rd(w, "v2-2d-baseline-seed42-a").exists()


def test_30b_real_run_a_untouched_and_not_resumable():
    rd = REPO / "results/v2_runs/v2-2d-baseline-seed42-a"
    if not rd.is_dir():
        pytest.skip("real Run A folder not present on this machine")
    files = sorted(p.relative_to(rd).as_posix() for p in rd.rglob("*") if p.is_file())
    assert files == ["pre_run_manifest.json", "training/preprocessing_record.json"]
    assert not (rd / brs.RESUME_NAME).exists() and not (rd / POST_RUN_NAME).exists()
    assert "v2-2d-baseline-seed42-a" in brs.RETIRED_RUN_IDS


def test_resume_checks_are_strict_on_epoch_limit_and_failed_runs(w):
    r = _refused(w, lambda: _resume(w, "syn-pending", epoch_limit=3))
    assert "epoch_limit_mismatch" in r


# ---------------------------------------------------------------- 31-33 signals
def test_31_sigterm_safe_stop(w):
    rd = _rd(w, "syn-sigterm")
    rec = json.loads((rd / "segments/segment_01/interruption_record.json").read_text("utf-8"))
    assert rec["status"] == INTERRUPTED and rec["reason"] == "sigterm"
    assert rec["completed_epoch"] == 2 and rec["next_epoch"] == 3
    assert rec["resume_allowed"] is True and rec["automatic_resume"] is False
    assert rec["retried"] is False
    seg_post = json.loads((rd / "segments/segment_01" / POST_RUN_NAME).read_text("utf-8"))
    assert seg_post["run"]["status"] == INTERRUPTED and seg_post["run"]["exit_code"] == 3
    assert seg_post["reproducibility"]["interruption"]["interruption_record_sha256"] == (
        hashlib.sha256(
            (rd / "segments/segment_01/interruption_record.json").read_bytes()
        ).hexdigest()
    )
    assert w.sigterm_calls["n"] == 2  # the signal arrived in epoch 2; epoch 2 still completed
    assert "training/progress.json" in w.b1_tree and "training/resume_checkpoint.pt" in w.b1_tree
    assert not any(k.startswith(("training/best", "training/last")) for k in w.b1_tree)
    # handlers are restored after every run
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == w.handlers_before


def test_32_keyboard_interrupt_safe_stop(w):
    rd = _rd(w, "syn-keyboard")
    rec = json.loads((rd / "segments/segment_01/interruption_record.json").read_text("utf-8"))
    assert rec["reason"] == "keyboard_interrupt" and rec["completed_epoch"] == 3
    assert w.k1["status"] == INTERRUPTED and w.k2["status"] == COMPLETED
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_33_forced_stop_keeps_last_completed_epoch_only(w):
    # killed inside epoch 3: nothing new written; checkpoint / progress describe epoch 2
    t = w.killed_tree
    assert not any(k.startswith("segments/") for k in t)
    assert "training/failure_record.json" not in t and POST_RUN_NAME not in t
    doc = (REPO / "docs/v2_2d_baseline_run_protocol.md").read_text(encoding="utf-8")
    assert "SIGKILL" in doc and "마지막 완료 Epoch" in doc
    assert w.kill2["status"] == COMPLETED


# ---------------------------------------------------------------- 34-42 resources and guard
def test_34_35_36_resource_fields(w, monkeypatch):
    p = json.loads((_rd(w, "syn-cont") / bp.PROGRESS_NAME).read_text(encoding="utf-8"))
    import sys

    if sys.platform in ("win32", "linux"):
        assert p["process_rss_bytes"] > 0 and p["process_peak_rss_bytes"] >= p["process_rss_bytes"]
        assert p["system_available_memory_bytes"] > 0
    assert p["cuda_memory_allocated_bytes"] is None  # CPU synthetic run
    # CUDA path: values come from torch.cuda counters; peak reset at run start
    calls = []
    for name, val in (
        ("memory_allocated", 1),
        ("memory_reserved", 2),
        ("max_memory_allocated", 3),
        ("max_memory_reserved", 4),
    ):
        monkeypatch.setattr(torch.cuda, name, lambda d, v=val: v)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: calls.append(d))
    dev = torch.device("cuda")
    assert rm.reset_cuda_peak(dev) and calls == [dev]
    assert rm.cuda_memory(dev) == {
        "cuda_memory_allocated_bytes": 1,
        "cuda_memory_reserved_bytes": 2,
        "cuda_max_memory_allocated_bytes": 3,
        "cuda_max_memory_reserved_bytes": 4,
    }
    src = (REPO / "ai/neuroscan_ai/v2/baseline_runner.py").read_text(encoding="utf-8")
    assert "rm.reset_cuda_peak(device)" in src


def test_37_38_guard_safe_stop_then_resume(w):
    assert w.g1["status"] == INTERRUPTED_RESOURCE_GUARD
    rd = _rd(w, "syn-guard")
    rec = json.loads((rd / "segments/segment_01/interruption_record.json").read_text("utf-8"))
    g = rec["host_memory_guard"]
    assert rec["status"] == INTERRUPTED_RESOURCE_GUARD and rec["reason"] == "host_memory_guard"
    assert g["policy_id"] == "host_memory_guard_v1" and g["triggered"] is True
    assert g["minimum_available_bytes"] == 4 * GIB and g["system_available_memory_bytes"] == 3 * GIB
    assert rec["completed_epoch"] == 1 and rec["resume_allowed"] is True
    assert w.g2["status"] == COMPLETED
    assert not rm.check_host_memory(4 * GIB)["triggered"]
    assert rm.check_host_memory(4 * GIB - 1)["triggered"]
    assert not rm.check_host_memory(None)["triggered"]  # no reading never stops a run


def test_39_40_41_42_nothing_automatic(w):
    # no automatic resume: an interrupted run stays interrupted until resume_baseline is called
    assert not (_rd(w, "syn-pending") / POST_RUN_NAME).exists()
    assert (
        json.loads((_rd(w, "syn-pending") / bp.PROGRESS_NAME).read_text("utf-8"))["status"]
        == INTERRUPTED
    )
    for rid in ("syn-sigterm", "syn-guard", "syn-keyboard"):
        rec = json.loads(
            (_rd(w, rid) / "segments/segment_01/interruption_record.json").read_text("utf-8")
        )
        assert rec["automatic_resume"] is False and rec["retried"] is False
        assert rec["batch_size_changed"] is False and rec["cache_limit_changed"] is False
        hist = json.loads(run_output_path(_rd(w, rid), "training_history").read_text("utf-8"))
        assert hist["hyperparameters"]["batch_size"] == 32
    pre = json.loads((_rd(w, "syn-guard") / "training/preprocessing_record.json").read_text())
    assert pre["cache"]["max_active_cases"] == 2
    # the only resume path is the explicit entry point
    src = (REPO / "ai/neuroscan_ai/v2/baseline_runner.py").read_text(encoding="utf-8")
    assert src.count("resume_controlled(") == 1
    cfg = br.prepare_config(w.cfg)
    assert cfg["resume"]["automatic_resume"] is False
    assert cfg["host_memory_guard"]["automatic_resume"] is False
    assert cfg["cache"]["max_active_cases"] == 2


def test_42_oom_not_retried_and_run_not_resumable(w, monkeypatch):
    def oom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (synthetic)")

    monkeypatch.setattr(bt, "train_one_epoch", oom)
    res = _run(w, "syn-oom")
    assert res["status"] == "failed_runner_error"
    monkeypatch.undo()
    with pytest.raises(ResumeRefusedError, match="run_already_finalised"):
        _resume(w, "syn-oom")


def test_config_code_doc_agree():
    cfg = br.prepare_config(REPO / "configs/v2/2d_baseline.yaml")
    for block, code in br.OPERATIONAL_BLOCKS.items():
        assert cfg[block] == code, block
    bad = dict(cfg, host_memory_guard={**cfg["host_memory_guard"], "minimum_available_gib": 2.0})
    with pytest.raises(bt.ConfigValidationError):
        br.validate_operational_config(bad)
    doc = (REPO / "docs/v2_2d_baseline_run_protocol.md").read_text(encoding="utf-8")
    for token in (
        "disk_backed_preprocessed_cache_v1",
        "host_memory_guard_v1",
        "progress.json",
        "resume_checkpoint.pt",
        "interrupted_resource_guard",
        "v2-2d-baseline-seed42-a-r2",
    ):
        assert token in doc, token


# ---------------------------------------------------------------- 43-45 access and privacy
def test_43_no_test_final_test_or_pilot_access(w):
    for rid in ("syn-cont", "syn-sigterm"):
        post = load_post_run(_rd(w, rid))
        assert post["reproducibility"]["accessed_splits"] == ["train"]
        assert set(post["access"]["allowed_splits"]) == {"train", "val"}
    pre = json.loads((_rd(w, "syn-cont") / "pre_run_manifest.json").read_text("utf-8"))
    assert pre["access"]["disallowed_split_access"] in ([], False, None, 0)


def test_44_no_v1_imports_in_new_modules():
    v1 = (
        "neuroscan_ai.training",
        "neuroscan_ai.inference",
        "neuroscan_ai.preprocessing",
        "neuroscan_ai.analysis",
        "neuroscan_ai.final_test",
    )
    for m in (
        "baseline_cache",
        "baseline_progress",
        "baseline_resume",
        "resource_monitor",
        "baseline_runner",
    ):
        src = (REPO / "ai/neuroscan_ai/v2" / f"{m}.py").read_text(encoding="utf-8")
        assert "soop" not in src.lower(), m
        for node in ast.walk(ast.parse(src)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n == p or n.startswith(p + ".") for n in names for p in v1), m


def test_45_no_alias_id_path_or_credential_in_run_records(w):
    needles = [a.encode() for a in (*TRAIN, *VAL, *FORBIDDEN)]
    needles += [str(w.repo).encode(), w.repo.as_posix().encode(), str(Path.home()).encode()]
    needles += [b"password", b"api_key", b"token="]
    for rid in ("syn-sigterm", "syn-guard", "syn-killed"):
        for p in _rd(w, rid).rglob("*"):
            if not p.is_file():
                continue
            data = p.read_bytes()
            if p.suffix == ".gz":
                data = gzip.decompress(data)
            if p.suffix == ".pt":
                continue  # binary tensors; metadata checked in test_16
            for n in needles:
                assert n not in data, (p.name, n[:20])


# ---------------------------------------------------------------- cache commit binding (runs)
def _spy_nothing_created(monkeypatch):
    calls = []
    monkeypatch.setattr(bc, "verify_cache", lambda *a, **k: calls.append("dataset"))
    monkeypatch.setattr(br, "build_model", lambda *a, **k: calls.append("model"))
    return calls


def test_cache_commit_same_commit_allowed(w):
    head = capture_source_control(w.repo)["git_commit"]
    cache = _cache(w, w.cfg, TRAIN, VAL)
    idx = json.loads((cache / bc.INDEX_NAME).read_text(encoding="utf-8"))
    assert idx["git_commit"] == head and w.a["status"] == COMPLETED


def test_cache_commit_other_or_missing_refused_before_gate(w, monkeypatch):
    other = w.repo / "results/v2_cache/syn_other_commit"
    if not other.exists():
        bc.build_cache(
            other,
            config=br.prepare_config(w.cfg),
            split_file=w.data / "split.json",
            files={"train": {a: w.files[a] for a in TRAIN}, "val": {a: w.files[a] for a in VAL}},
            git_commit="f" * 40,  # synthetic cache from another commit: same policy applies
            package_lock_sha256_lf=_env(w)["package_lock_sha256_lf"],
        )
    missing = w.repo / "results/v2_cache/syn_missing_commit"
    if not missing.exists():
        shutil.copytree(_cache(w, w.cfg, TRAIN, VAL), missing)
        idx = json.loads((missing / bc.INDEX_NAME).read_text(encoding="utf-8"))
        idx.pop("git_commit")
        (missing / bc.INDEX_NAME).write_text(json.dumps(idx), encoding="utf-8")
        from neuroscan_ai.reproducibility.hashing import canonical_json_sha256

        (missing / bc.INDEX_SEAL).write_text(canonical_json_sha256(idx), encoding="utf-8")
    calls = _spy_nothing_created(monkeypatch)
    dirty = {**capture_source_control(w.repo), "working_tree_clean": False}
    for rid, kw, reason in (
        ("syn-commit-other", {"cache": other}, "cache_git_commit_mismatch"),
        ("syn-commit-missing", {"cache": missing}, "cache_git_commit_missing"),
        ("syn-commit-dirty", {"sc": dirty}, "working_tree_not_clean"),
    ):
        with pytest.raises(bc.CacheCommitError, match=reason):
            _run(w, rid, **kw)
        assert not _rd(w, rid).exists()  # no folder, run ID unused
    assert calls == []  # no dataset, no model
    monkeypatch.undo()
    assert _run(w, "syn-commit-other")["status"] == COMPLETED  # the ID was not consumed


def test_cache_commit_changed_after_cache_build_refused(tmp_path, monkeypatch):
    """Fresh repo: cache built at commit X, then a new code commit -> refused before the gate."""
    root = tmp_path
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
    (data / "split.json").write_text(
        json.dumps({"splits": {"train": TRAIN, "val": VAL}}), encoding="utf-8"
    )
    from neuroscan_ai.reproducibility import PathRoots

    ns = SimpleNamespace(
        repo=repo,
        data=data,
        files={a: _write_case(data, a, i) for i, a in enumerate(TRAIN + VAL)},
        cfg=repo / "configs/v2/2d_baseline.yaml",
        roots=PathRoots(repo, repo / "models", data),
        caches={},
    )
    cache = _cache(ns, ns.cfg, TRAIN, VAL)  # built at commit X
    (repo / "code_change.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run([*GIT, "-C", str(repo), "add", "."], check=True)
    subprocess.run([*GIT, "-C", str(repo), "commit", "-q", "-m", "code"], check=True)
    calls = _spy_nothing_created(monkeypatch)
    with pytest.raises(bc.CacheCommitError, match="cache_git_commit_mismatch"):
        _run(ns, "syn-after-commit", cache=cache)
    assert calls == [] and not (repo / "results/v2_runs/syn-after-commit").exists()


# ---------------------------------------------------------------- run ID and segments
def test_foreign_checkpoint_refused(w):
    rd = _rd(w, "syn-foreign")
    for name in (brs.RESUME_NAME, bp.PROGRESS_NAME):  # another run's state copied in
        shutil.copy(_rd(w, "syn-pending") / name, rd / name)
    before = _tree(rd)
    with pytest.raises(ResumeRefusedError) as e:
        _resume(w, "syn-foreign")
    assert "run_id_mismatch" in e.value.reasons
    assert "progress_checkpoint_epoch_mismatch" in e.value.reasons
    assert _tree(rd) == before


def test_same_id_resume_segments_monotonic_and_unchanged(w):
    assert w.t1["status"] == INTERRUPTED and w.t2["status"] == INTERRUPTED
    assert w.t3["status"] == COMPLETED and w.t3["resumed"] is True
    rd = _rd(w, "syn-twice")
    assert sorted(p.name for p in (rd / "segments").iterdir()) == ["segment_01", "segment_02"]
    assert w.twice_seg1 == w.twice_seg1_after_2 == _tree(rd / "segments/segment_01")
    assert w.twice_seg2 == _tree(rd / "segments/segment_02")
    recs = [
        json.loads((rd / f"segments/segment_0{i}/interruption_record.json").read_text("utf-8"))
        for i in (1, 2)
    ]
    assert [r["completed_epoch"] for r in recs] == [1, 2]
    for seg in ("segment_01", "segment_02"):
        assert verify_run(rd, segment=seg)["ok"]
    assert load_post_run(rd)["reproducibility"]["interrupted_segments"] == 2
    assert verify_run(rd)["ok"]
    # resume adds no run folder (same run ID, same folder)
    assert w.dirs_before_resume == w.dirs_after_resume
    _assert_exact(w, "syn-cont", "syn-twice")
