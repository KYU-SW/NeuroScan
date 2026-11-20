"""Explicit run seed for the v2 2D baseline (``seed_policy_v2_baseline_explicit_run_seed_v1``) on
SYNTHETIC volumes only (CPU, throwaway git repo, no MRI, no ISLES / SOOP access, no model run on
real data). Formal-profile acceptance is checked up to the run-control hand-off (``run_controlled``
is replaced by a recorder), so no formal training ever starts. Synthetic outputs are not results."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import torch

from neuroscan_ai.reproducibility.environment import capture_determinism, capture_source_control
from neuroscan_ai.reproducibility.hashing import canonical_json_sha256
from neuroscan_ai.reproducibility.validation import EXACT
from neuroscan_ai.run_control import (
    COMPLETED,
    RunPlanError,
    build_plan,
    compare_run_bundles,
    run_controlled,
)
from neuroscan_ai.run_control.gate import APPROVED, GateResult
from neuroscan_ai.run_control.lifecycle import INTERRUPTED_RESOURCE_GUARD, ResumeRefusedError
from neuroscan_ai.run_control.outputs import run_output_path
from neuroscan_ai.v2 import baseline_cache as bc
from neuroscan_ai.v2 import baseline_dataset as bd
from neuroscan_ai.v2 import baseline_resume as brs
from neuroscan_ai.v2 import baseline_runner as br
from neuroscan_ai.v2 import baseline_training as bt
from neuroscan_ai.v2.unet_2d import MODEL_ID, build_model
from tests import test_v2_baseline_runner as _rt
from tests.test_v2_baseline_runner import (
    FORBIDDEN,
    REPO,
    TRAIN,
    _cache,
    _env,
    _plan_kwargs,
)

w = _rt.w  # the shared synthetic workspace fixture (throwaway repo, synthetic volumes)

GIB = 1024**3
FORMAL = {"dataset_id": "isles2022"}  # a formal marker: the profile becomes formal_isles


def _det():
    return capture_determinism(
        torch, seed=None, precision="fp32", amp=False, workers=0, batch_size=32
    )


def _plan(w, run_id, plan_seed, **kw):
    plan_kw = _plan_kwargs(w, run_id, **kw)
    plan_kw["seed"] = plan_seed
    return plan_kw


def _go(w, run_id, seed=None, plan_seed="same", device="cpu", epoch_limit=2, probe=None, **kw):
    plan_kw = _plan(w, run_id, seed if plan_seed == "same" else plan_seed, **kw)
    known = set(plan_kw["aliases"]) | set(plan_kw["validation_aliases"]) <= set(w.files)
    cache = (
        _cache(w, w.cfg, plan_kw["aliases"], plan_kw["validation_aliases"])
        if known
        else w.repo / "results/v2_cache/not-built"  # refused plans never reach the cache
    )
    return br.run_baseline(
        w.cfg,
        plan_kw,
        cache,
        repo_root=w.repo,
        roots=w.roots,
        source_control=capture_source_control(w.repo),
        environment=_env(w),
        determinism=_det(),
        forbidden_ids=FORBIDDEN,
        device=device,
        epoch_limit=epoch_limit,
        memory_probe=probe,
        seed=seed,
    )


def _resume(w, run_id, seed=None, plan_seed="same", probe=None):
    plan_kw = _plan(w, run_id, seed if plan_seed == "same" else plan_seed)
    return br.resume_baseline(
        w.cfg,
        plan_kw,
        _cache(w, w.cfg, plan_kw["aliases"], plan_kw["validation_aliases"]),
        repo_root=w.repo,
        source_control=capture_source_control(w.repo),
        environment=_env(w),
        determinism=_det(),
        forbidden_ids=FORBIDDEN,
        device="cpu",
        epoch_limit=2,
        memory_probe=probe,
        seed=seed,
    )


def _rd(w, run_id):
    return w.repo / "results/v2_runs" / run_id


def _js(w, run_id, label):
    return json.loads(run_output_path(_rd(w, run_id), label).read_text(encoding="utf-8"))


def _manifest(w, run_id, name="pre_run_manifest.json"):
    return json.loads((_rd(w, run_id) / name).read_text(encoding="utf-8"))


def _ckpt(w, run_id, label):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(
            run_output_path(_rd(w, run_id), label), map_location="cpu", weights_only=True
        )


def _tree(p: Path) -> dict[str, bytes]:
    return {q.relative_to(p).as_posix(): q.read_bytes() for q in p.rglob("*") if q.is_file()}


@pytest.fixture
def spies(monkeypatch):
    calls = []
    monkeypatch.setattr(bd, "load_cases", lambda *a, **k: calls.append("dataset"))
    monkeypatch.setattr(bc, "verify_cache", lambda *a, **k: calls.append("dataset"))
    monkeypatch.setattr(br, "build_model", lambda *a, **k: calls.append("model"))
    return calls


@pytest.fixture(scope="module")
def runs(w):
    """Synthetic runs shared by the recording / difference tests (2 epochs each, CPU)."""
    out = {
        "s43a": _go(w, "syn-seed43-a", seed=43),
        "s43b": _go(w, "syn-seed43-b", seed=43),
        "s44": _go(w, "syn-seed44", seed=44),
        # interrupted after epoch 1 by the host memory guard (3 GiB < 4 GiB)
        "s43i": _go(w, "syn-seed43-int", seed=43, probe=lambda: 3 * GIB),
    }
    return out


# ---------------------------------------------------------------- 1-8, 16: CLI / plan policy
def test_01_16_formal_run_without_seed_refused_before_gate(w, spies):
    with pytest.raises(bt.ConfigValidationError, match="explicit --seed"):
        _go(w, "syn-formal-noseed", seed=None, device="cuda", epoch_limit=None, **FORMAL)
    assert spies == [] and not _rd(w, "syn-formal-noseed").exists()


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_02_03_04_formal_seeds_accepted_and_handed_to_run_control(w, monkeypatch, seed):
    seen = {}

    def recorder(plan, runner, **kw):  # stands in for the gate: nothing is opened or created
        seen.update(plan=plan, runner=runner, determinism=kw["determinism"])
        return {"pre_run_status": "recorded"}

    monkeypatch.setattr(br, "run_controlled", recorder)
    rid = f"syn-formal-seed{seed}"
    res = _go(w, rid, seed=seed, device="cuda", epoch_limit=None, **FORMAL)
    assert res == {"pre_run_status": "recorded"}
    assert seen["plan"].seed == seed and seen["runner"].seed == seed
    assert seen["runner"].run_seed == seed and seen["runner"].dataset_profile == bt.PROFILE_FORMAL
    assert not _rd(w, rid).exists()
    assert bt.check_run_seed(bt.PROFILE_FORMAL, seed) == seed


@pytest.mark.parametrize("seed", [0, 1, 41, 45, 1000])
def test_05_16_formal_seed_outside_allow_list_refused(w, spies, seed):
    assert seed not in bt.FORMAL_SEEDS
    with pytest.raises(bt.ConfigValidationError, match="approved formal seed list"):
        _go(w, "syn-formal-unlisted", seed=seed, device="cuda", epoch_limit=None, **FORMAL)
    assert spies == [] and not _rd(w, "syn-formal-unlisted").exists()


def test_06_negative_seed_refused(w, spies):
    for profile in (bt.PROFILE_FORMAL, bt.PROFILE_SYNTHETIC):
        with pytest.raises(bt.ConfigValidationError, match="non-negative"):
            bt.check_run_seed(profile, -1)
    with pytest.raises(bt.ConfigValidationError):
        bt.parse_seed_text("-43")
    with pytest.raises(RunPlanError, match="non-negative"):  # RunPlan refuses it on its own
        build_plan(
            **{
                **_plan(w, "syn-neg", -1),
                "model_id": MODEL_ID,
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
    with pytest.raises((bt.ConfigValidationError, RunPlanError)):
        _go(w, "syn-neg", seed=-1)
    assert spies == [] and not _rd(w, "syn-neg").exists()


@pytest.mark.parametrize("bad", ["abc", "4.3", "43.0", " 43", "+43", "", "0x2b", "4 3", "４３"])
def test_07_non_integer_seed_text_refused(bad):
    with pytest.raises(bt.ConfigValidationError):
        bt.parse_seed_text(bad)


def test_07_non_integer_seed_values_refused(w, spies):
    for bad in ("43", 43.0, True, None):
        if bad is None:
            continue
        with pytest.raises(bt.ConfigValidationError):
            bt.check_run_seed(bt.PROFILE_FORMAL, bad)
    assert bt.parse_seed_text("43") == 43 and bt.parse_seed_text(None) is None
    with pytest.raises(bt.ConfigValidationError):
        _go(w, "syn-strseed", seed="43", plan_seed=None)
    assert spies == [] and not _rd(w, "syn-strseed").exists()


def _cli():
    spec = importlib.util.spec_from_file_location(
        "run_v2_2d_baseline_cli", REPO / "scripts/run_v2_2d_baseline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("seed_text", [None, "abc", "-1", "45", "43.5", ""])
def test_01_05_06_07_cli_refuses_bad_seed_before_anything(seed_text, monkeypatch, capsys):
    cli = _cli()
    touched = []
    monkeypatch.setattr(cli, "capture_source_control", lambda *a: touched.append("sc"))
    monkeypatch.setattr(cli, "run_baseline", lambda *a, **k: touched.append("run"))
    monkeypatch.setattr(cli, "resume_baseline", lambda *a, **k: touched.append("resume"))
    rid = "syn-cli-seed-refused"
    for resume in (False, True):
        code = cli.run(
            rid,
            Path("unused-alias-map.json"),
            Path("unused-split.json"),
            "0" * 64,
            Path("unused-cache"),
            None,
            resume=resume,
            seed_text=seed_text,
        )
        assert code == 2
    assert touched == [] and not (REPO / "results/v2_runs" / rid).exists()
    assert "REFUSED" in capsys.readouterr().err


def test_cli_seed_is_required_option_in_help():
    src = (REPO / "scripts/run_v2_2d_baseline.py").read_text(encoding="utf-8")
    assert 'r.add_argument("--seed", default=None)' in src
    assert "check_run_seed(PROFILE_FORMAL, parse_seed_text(seed_text))" in src
    assert '"seed": seed,' in src  # RunPlan and run_baseline / resume_baseline get the same seed


def test_08_16_cli_and_runplan_seed_mismatch_refused(w, spies):
    for seed, plan_seed in ((43, 44), (43, None), (None, 43)):
        with pytest.raises(bt.ConfigValidationError, match="RunPlan"):
            _go(w, "syn-seed-mismatch", seed=seed, plan_seed=plan_seed)
    with pytest.raises(bt.ConfigValidationError):  # formal: same refusal
        _go(
            w, "syn-seed-mismatch", seed=43, plan_seed=44, device="cuda", epoch_limit=None, **FORMAL
        )
    assert spies == [] and not _rd(w, "syn-seed-mismatch").exists()


def test_08_16_run_control_refuses_plan_runner_seed_mismatch(w, spies):
    plan_kw = _plan(w, "syn-seed-mismatch-rc", 44)
    plan = build_plan(
        **{
            **plan_kw,
            "model_id": MODEL_ID,
            "config": w.cfg,
            "policies": dict(br.prepare_config(w.cfg)["policies"]),
        }
    )
    runner = br.BaselineRunner(
        br.prepare_config(w.cfg), Path("unused"), run_seed=43, dataset_profile=bt.PROFILE_SYNTHETIC
    )
    with pytest.raises(ValueError, match="run plan seed differs"):
        run_controlled(
            plan,
            runner,
            repo_root=w.repo,
            roots=w.roots,
            source_control=capture_source_control(w.repo),
            environment=_env(w),
            determinism=_det(),
            forbidden_ids=FORBIDDEN,
        )
    assert spies == [] and not _rd(w, "syn-seed-mismatch-rc").exists()


# ---------------------------------------------------------------- 9-10: recorded everywhere
def test_09_manifests_history_sampler_aggregate_record_the_seed(w, runs):
    assert runs["s43a"]["status"] == COMPLETED and runs["s44"]["status"] == COMPLETED
    for rid, seed in (("syn-seed43-a", 43), ("syn-seed44", 44)):
        assert _manifest(w, rid)["determinism"]["seed"] == seed
        assert _manifest(w, rid, "post_run_manifest.json")["determinism"]["seed"] == seed
        h = _js(w, rid, "training_history")
        assert h["seed"] == seed and h["hyperparameters"]["seed"] == seed
        assert h["seed_policy"] == bt.SEED_POLICY and h["config_reference_seed"] == 42
        assert h["model"]["init_seed"] == seed
        s = _js(w, rid, "sampler_record")
        assert s["seed"] == seed and all(e["sampling_seed"][0] == seed for e in s["epochs"])
        assert _js(w, rid, "aggregate")["seed"] == seed
        assert _js(w, rid, "validation_metrics")["seed"] == seed
        expected = {"seed_policy": bt.SEED_POLICY, "run_seed": seed, "config_reference_seed": 42}
        for label in ("training_history", "sampler_record", "aggregate", "validation_metrics"):
            rec = _js(w, rid, label)
            assert {k: rec[k] for k in expected} == expected, label


def test_10_checkpoints_record_the_seed(w, runs):
    for rid, seed in (("syn-seed43-a", 43), ("syn-seed44", 44)):
        for label in ("best_checkpoint", "last_checkpoint"):
            c = _ckpt(w, rid, label)
            assert c["seed"] == seed and c["sampler"]["seed"] == seed
            assert c["seed_policy"] == bt.SEED_POLICY
            assert c["run_seed"] == seed and c["config_reference_seed"] == 42
        payload = brs.load_resume_checkpoint(_rd(w, rid) / brs.RESUME_NAME)
        assert payload["seed"] == seed and payload["run_seed"] == seed
        assert payload["seed_policy"] == bt.SEED_POLICY and payload["config_reference_seed"] == 42
        assert all(r["sampling_seed"][0] == seed for r in payload["sampling"])


# ---------------------------------------------------------------- 11: resume needs the same seed
def test_11_runner_preflight_checks_checkpoint_and_sampler_seed(w, runs):
    """Runner-side checks alone (lifecycle bypassed): a seed-44 runner on the interrupted seed-43
    run is refused on the checkpoint identity seed, the checkpoint run_seed and the sampler seed;
    nothing is written and no segment is added."""
    assert runs["s43i"]["status"] == INTERRUPTED_RESOURCE_GUARD
    rd = _rd(w, "syn-seed43-int")
    before = _tree(rd)
    plan_kw = _plan(w, "syn-seed43-int", 44)
    cache = _cache(w, w.cfg, plan_kw["aliases"], plan_kw["validation_aliases"])
    plan, runner = br._build(w.cfg, plan_kw, cache, w.repo, FORBIDDEN, "cpu", 2, None, 44)
    pre = _manifest(w, "syn-seed43-int")
    gate = GateResult(APPROVED, (), rd, canonical_json_sha256(pre), pre)
    reasons = runner.resume_preflight(gate, plan)
    assert {"seed_mismatch", "checkpoint_run_seed_mismatch", "sampler_seed_mismatch"} <= set(
        reasons
    )
    assert _tree(rd) == before and sorted(p.name for p in (rd / "segments").iterdir()) == [
        "segment_01"
    ]
    # the same runner with the run's own seed passes the runner-side checks
    plan_kw = _plan(w, "syn-seed43-int", 43)
    plan, runner = br._build(w.cfg, plan_kw, cache, w.repo, FORBIDDEN, "cpu", 2, None, 43)
    assert runner.resume_preflight(gate, plan) == []
    assert _tree(rd) == before


def test_11_resume_with_other_seed_refused_same_seed_completes(w, runs):
    assert runs["s43i"]["status"] == INTERRUPTED_RESOURCE_GUARD
    rd = _rd(w, "syn-seed43-int")
    before = _tree(rd)
    ok = lambda: 64 * GIB  # noqa: E731 - plenty of host memory for the resume
    with pytest.raises(ResumeRefusedError) as e:
        _resume(w, "syn-seed43-int", seed=44, probe=ok)
    assert "seed_mismatch" in e.value.reasons and "determinism_mismatch" in e.value.reasons
    with pytest.raises(ResumeRefusedError) as e:  # no seed: config reference 42, not 43
        _resume(w, "syn-seed43-int", seed=None, probe=ok)
    assert "seed_mismatch" in e.value.reasons
    with pytest.raises(bt.ConfigValidationError, match="RunPlan"):
        _resume(w, "syn-seed43-int", seed=43, plan_seed=44, probe=ok)
    assert _tree(rd) == before  # every refusal wrote nothing (no file, no new segment)
    assert sorted(p.name for p in (rd / "segments").iterdir()) == ["segment_01"]
    assert _resume(w, "syn-seed43-int", seed=43, probe=ok)["status"] == COMPLETED


def test_11_checkpoint_seed_mismatch_refused_by_identity_check(w, runs):
    payload = brs.load_resume_checkpoint(_rd(w, "syn-seed43-a") / brs.RESUME_NAME)
    assert brs.check_resume_identity(payload, {**payload, "seed": 44}) == ["seed_mismatch"]
    assert "seed" in brs.IDENTITY_FIELDS


# ---------------------------------------------------------------- 12-15: same / other seed
def test_12_same_seed_exact_match(w, runs):
    res = compare_run_bundles(_rd(w, "syn-seed43-a"), _rd(w, "syn-seed43-b"))
    assert res["level"] == EXACT and res["differences"] == []
    # the interrupted-and-resumed seed-43 run equals the continuous seed-43 run as well
    res = compare_run_bundles(_rd(w, "syn-seed43-a"), _rd(w, "syn-seed43-int"))
    assert res["level"] == EXACT


def test_13_other_seed_other_initial_weights(w, runs):
    models = {s: build_model(s).state_dict() for s in bt.FORMAL_SEEDS}
    for a, b in ((42, 43), (42, 44), (43, 44)):
        assert any(not torch.equal(models[a][k], models[b][k]) for k in models[a])
    for s in bt.FORMAL_SEEDS:  # the same seed always gives the same initial weights
        again = build_model(s).state_dict()
        assert all(torch.equal(models[s][k], again[k]) for k in again)
    a = _ckpt(w, "syn-seed43-a", "last_checkpoint")["model_state"]
    b = _ckpt(w, "syn-seed44", "last_checkpoint")["model_state"]
    assert any(not torch.equal(a[k], b[k]) for k in a)


def test_14_other_seed_other_sampling_hash(w, runs):
    a = _js(w, "syn-seed43-a", "sampler_record")["epochs"]
    b = _js(w, "syn-seed44", "sampler_record")["epochs"]
    assert [e["slice_list_sha256"] for e in a] != [e["slice_list_sha256"] for e in b]
    assert a[0]["slice_list_sha256"] != b[0]["slice_list_sha256"]


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k != "file_sha256":
            out.update(_flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def test_15_only_seed_and_run_id_differ(w, runs):
    a, b = _flat(_manifest(w, "syn-seed43-a")), _flat(_manifest(w, "syn-seed44"))
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff == {"identity.run_id", "identity.created_at_utc", "determinism.seed"}
    ca, cb = _ckpt(w, "syn-seed43-a", "best_checkpoint"), _ckpt(w, "syn-seed44", "best_checkpoint")
    same = (
        "config_canonical_sha256",
        "split_canonical_sha256",
        "git_commit",
        "model_id",
        "parameter_count",
        "package_lock_sha256_lf",
        "preprocessing_policy",
        "threshold_policy",
    )
    assert all(ca[k] == cb[k] for k in same)
    ha, hb = _js(w, "syn-seed43-a", "training_history"), _js(w, "syn-seed44", "training_history")
    assert {**ha["hyperparameters"], "seed": 0} == {**hb["hyperparameters"], "seed": 0}
    assert _js(w, "syn-seed43-a", "preprocessing_record") == _js(
        w, "syn-seed44", "preprocessing_record"
    )  # the cache / preprocessing do not depend on the seed
    assert _js(w, "syn-seed43-a", "threshold_record") == _js(w, "syn-seed44", "threshold_record")


# ---------------------------------------------------------------- 17-18: access and v1
def test_17_no_test_final_test_or_soop_access(w, runs):
    for rid in ("syn-seed43-a", "syn-seed44"):
        acc = _manifest(w, rid)["access"]
        assert acc["test_cases"] == 0 and acc["final_test_accessed"] is False
        assert acc["disallowed_split_access"] is False
    with pytest.raises(RunPlanError):
        _go(
            w,
            "syn-seed-test",
            seed=43,
            aliases=["syn-test-01"],
            allowed_case_count=1,
            split_name="test",
        )
    with pytest.raises(RunPlanError):
        _go(
            w,
            "syn-seed-soop",
            seed=43,
            aliases=[*TRAIN, "soop-pilot-01"],
            allowed_case_count=len(TRAIN) + 1,
        )
    assert not _rd(w, "syn-seed-test").exists() and not _rd(w, "syn-seed-soop").exists()


V1 = ("training", "inference", "preprocessing", "analysis", "final_test")


def test_18_no_v1_imports_and_v1_unchanged():
    files = [
        "ai/neuroscan_ai/v2/baseline_training.py",
        "ai/neuroscan_ai/v2/baseline_runner.py",
        "ai/neuroscan_ai/run_control/plan.py",
        "ai/neuroscan_ai/run_control/lifecycle.py",
        "scripts/run_v2_2d_baseline.py",
    ]
    for f in files:
        tree = ast.parse((REPO / f).read_text(encoding="utf-8"))
        mods = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        mods += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not [m for m in mods if any(m.startswith(f"neuroscan_ai.{v}") for v in V1)], f
    paths = [f"ai/neuroscan_ai/{v}" for v in V1] + ["data/splits", "models"]
    paths += [p.relative_to(REPO).as_posix() for p in (REPO / "configs").glob("*.yaml")]
    diff = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--name-only", "HEAD", "--", *paths],
        capture_output=True,
        text=True,
    ).stdout.split()
    assert diff == []


def test_config_reference_seed_kept_and_policy_documented():
    import yaml

    cfg = yaml.safe_load((REPO / "configs/v2/2d_baseline.yaml").read_text(encoding="utf-8"))
    assert cfg["training"]["seed"] == 42 == bt.FIXED_TRAINING["seed"]
    assert bt.FORMAL_SEEDS == (42, 43, 44)
    doc = (REPO / "docs/v2_2d_baseline_run_protocol.md").read_text(encoding="utf-8")
    assert bt.SEED_POLICY in doc and "--seed" in doc
