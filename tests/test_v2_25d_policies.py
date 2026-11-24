"""v2 2.5D config / code / document consistency (no data, no model)."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from neuroscan_ai.v2 import baseline_25d_dataset as bd
from neuroscan_ai.v2 import baseline_25d_runner as br
from neuroscan_ai.v2 import baseline_training as bt
from neuroscan_ai.v2 import cuda_smoke_25d as cs
from neuroscan_ai.v2 import unet_25d as m25
from neuroscan_ai.v2.baseline_preprocess import PolicyConfigError

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/v2/25d_baseline.yaml"
DOC = REPO / "docs/v2_25d_baseline_protocol.md"
EXPECTED = {
    "model_id": "neuroscan_v2_25d_baseline_v1",
    "runner_kind": "v2_25d_baseline",
    "input_channels": 6,
    "input_channel_policy_id": "input_25d_slice_major_dwi_adc_pairs_v1",
    "neighbor_policy_id": "neighbor_fixed_adjacent_index_pm1_v1",
    "boundary_policy_id": "edge_slice_repeat_v1",
    "normalization_policy_id": "normalization_case_volume_zscore_brainmask_no_slice_renorm_v1",
    "inference_policy_id": "v2_25d_slicewise_stack_sigmoid_original_grid_v1",
    "threshold_policy_id": "threshold_v2_25d_initial_fixed_0_50_not_frozen_v1",
    "combination_policy_id": "single_model_25d",
}


@pytest.fixture()
def cfg():
    return yaml.safe_load(CONFIG.read_text("utf-8"))


def _prepare(tmp_path, cfg):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return br.prepare_config(p)


def test_policy_record_code_config_document(cfg):
    assert br.POLICY_RECORD == EXPECTED == cfg["policy_record"]
    doc = DOC.read_text("utf-8")
    for v in EXPECTED.values():
        assert str(v) in doc
    assert br.prepare_config(CONFIG)["policy_record"] == EXPECTED


def test_model_block(cfg):
    m = cfg["model"]
    assert m["trainable_parameters"] == m25.EXPECTED_PARAMETERS == 2_440_873
    assert m["input_channels"] == list(m25.INPUT_CHANNEL_NAMES) and m["in_channels"] == 6
    doc = DOC.read_text("utf-8")
    for n in ("2,440,873", "2,439,591", "1,929,281", "1,477,785"):
        assert n in doc


@pytest.mark.parametrize("key", list(EXPECTED))
def test_any_policy_difference_refused(cfg, tmp_path, key):
    cfg["policy_record"][key] = "other"
    with pytest.raises(bt.ConfigValidationError, match="policy_record"):
        _prepare(tmp_path, cfg)


@pytest.mark.parametrize(
    "block,key,value",
    [
        ("model", "in_channels", 2),
        ("model", "id", "neuroscan_v2_2d_baseline_v1"),
        ("model", "channels", [32, 64, 128, 256]),
        ("model", "trainable_parameters", 2439577),
        ("model", "input_channels", ["dwi_z-1", "dwi_z", "dwi_z+1", "adc_z-1", "adc_z", "adc_z+1"]),
        ("training", "batch_size", 16),
        ("training", "seed", 43),
        ("training", "amp", True),
        ("policies", "threshold", "threshold_v2_2d_baseline_fixed_0_50_v1"),
        ("policies", "inference", "v2_2d_slicewise_sigmoid_original_grid_v1"),
        ("batch_policy", "batch_status", "confirmed"),
        ("batch_policy", "batch_size", 16),
        ("batch_policy", "automatic_reduction", True),
        ("threshold_search", "enabled", True),
        ("cache", "max_active_cases", 4),
    ],
)
def test_changed_config_refused(cfg, tmp_path, block, key, value):
    _prepare(tmp_path, copy.deepcopy(cfg))
    cfg[block][key] = value
    with pytest.raises(bt.ConfigValidationError):
        _prepare(tmp_path, cfg)


def test_preprocessing_policy_still_checked(cfg, tmp_path):
    cfg["preprocessing"]["gt_decode"]["one_atol"] = 1e-3
    with pytest.raises(PolicyConfigError):
        _prepare(tmp_path, cfg)


def test_only_the_approved_blocks_differ_from_the_2d_config(cfg):
    two = yaml.safe_load((REPO / "configs/v2/2d_baseline.yaml").read_text("utf-8"))
    same = [
        "dataset_id",
        "data",
        "preprocessing",
        "outputs",
        "selection",
        "safety_metrics",
        "training",
        "gt_case_status",
        "collapse_policy",
        "determinism",
        "run_control",
        "cache",
        "progress",
        "resume",
        "resource_monitor",
        "host_memory_guard",
    ]
    for k in same:
        assert cfg[k] == two[k], k
    assert "development_freeze" not in cfg  # the 2D freeze is not inherited


def test_batch_is_a_candidate_and_smoke_settings_match_code(cfg):
    b = cfg["batch_policy"]
    assert b["batch_status"] == "candidate" and b["batch_size"] == 32
    assert tuple(b["smoke_input_shape"]) == cs.BATCH_SHAPE
    assert b["smoke_max_memory_reserved_gib_lt"] == cs.MAX_RESERVED_GIB
    assert b["smoke_min_available_ram_gib"] == cs.MIN_AVAILABLE_RAM_GIB
    s = cfg["smoke"]
    assert tuple(s["run_ids"]) == br.SMOKE_RUN_IDS and s["seed"] == br.SMOKE_SEED
    assert s["epoch_limit"] == br.SMOKE_EPOCH_LIMIT == 2
    assert {"train": s["train_cases"], "val": s["validation_cases"]} == br.SMOKE_COUNTS


def test_success_criteria_recorded_but_not_read_by_code(cfg):
    p, g = cfg["success_criteria"]["primary"], cfg["success_criteria"]["guardrails"]
    p1 = p["P1_thick_lt_0_1_ml"]
    assert (p1["detected_total_gte"], p1["components_total"]) == (24, 51)
    assert p1["seeds_not_below_2d_gte"] == 2 and p1["judged_on"] == "raw_counts"
    assert p["P2_common_missed_lt"] == 28 and p["P3_recall_lt_0_1_ml_mean_gte"] == 0.644
    assert p["P4_oversegmentation_ml_mean"] == {"lte": 119.74, "improved_if_lt": 112.48}
    assert g == {
        "dice_mean_gte": 0.7607,
        "lesion_recall_mean_gte": 0.7437,
        "false_positive_volume_ml_mean_lte": 128.70,
        "dice_sample_sd_lte": 0.0179,
        "negative_case_fp_ml_each_seed_lte": 0.5,
        "outside_brain_mask_voxels_each_seed_lte": 20,
        "gt_components_lost": 0,
        "forbidden_split_access": 0,
    }
    ref = cfg["success_criteria"]["reference_2d"]["thick_lt_0_1_ml_detected_by_seed"]
    assert sum(ref.values()) == 21
    doc = DOC.read_text("utf-8")
    for s in ("24/51", "119.74", "0.7607", "0.7437", "128.70", "0.0179", "SUCCESS", "PARTIAL"):
        assert s in doc
    for f in (REPO / "ai").rglob("*.py"):
        assert "success_criteria" not in f.read_text("utf-8"), f.name


def test_no_threshold_search_in_production_code():
    for f in (*(REPO / "ai/neuroscan_ai/v2").glob("*25d*.py"), *REPO.glob("scripts/*25d*.py")):
        names = {n.id for n in ast.walk(ast.parse(f.read_text("utf-8"))) if isinstance(n, ast.Name)}
        assert "search_threshold" not in names and "THRESHOLD_CANDIDATES" not in names


def test_document_states_limits_and_boundaries():
    doc = DOC.read_text("utf-8")
    for s in ("4.0 mm", "9.6 mm", "10.0 mm", "짝지은 실험이 아니다", "candidate", "AST 검사"):
        assert s in doc
    for s in bd.METADATA_FIELDS:
        assert s in doc


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_cli_refuses_before_any_data_access(monkeypatch, capsys, tmp_path):
    m = _script("run_v2_25d_baseline")
    for name in ("isles_case_files", "run_baseline", "resume_baseline", "capture_source_control"):
        monkeypatch.setattr(m, name, lambda *a, **k: pytest.fail("must not be reached"))
    a = (tmp_path / "a.json", tmp_path / "s.json", "0" * 64, tmp_path)
    off = "v2-25d-baseline-seed42-multiseed-v1"
    assert m.run(off, *a, seed_text="42") == 2
    assert "batch_status" in capsys.readouterr().err
    for rid, seed, smoke, msg in (
        (off, None, False, "seed"),
        (off, "45", False, "seed"),
        ("v2-25d-smoke-seed42-a", "42", False, "smoke run IDs"),
        ("v2-25d-smoke-seed42-a", "43", True, "seed"),
        (off, "42", True, "smoke run ID"),
        ("v2-2d-baseline-seed42-a", "42", False, "retired"),
        ("v2-25d-smoke-seed42-a", "42", True, "package lock"),  # profile ok; stops at the lock
    ):
        assert m.run(rid, *a, seed_text=seed, smoke=smoke) == 2
        assert msg in capsys.readouterr().err
    assert m.run(off, *a, device="cpu", seed_text="42") == 2


def test_cli_validate_opens_no_data():
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/run_v2_25d_baseline.py"), "validate"],
        capture_output=True,
        encoding="utf-8",
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["batch_policy"]["batch_status"] == "candidate"


def test_cache_script_uses_the_25d_config_and_refuses(monkeypatch, capsys, tmp_path):
    m = _script("prepare_v2_25d_cache")
    assert m.CONFIG == CONFIG
    monkeypatch.setattr(m, "build_cache", lambda *a, **k: pytest.fail("must not build"))
    monkeypatch.setattr(m, "isles_case_files", lambda *a, **k: pytest.fail("must not open"))
    argv = ["x", "--alias-map", "a", "--split", "s", "--package-lock-sha256", "0" * 64]
    monkeypatch.setattr(sys, "argv", [*argv, "--cache-dir", str(tmp_path)])
    assert m.main() == 2 and "package lock" in capsys.readouterr().err
