"""v2 2D baseline: ungated evaluation, lesion-wise safety metrics, checkpoint selection, threshold
search, access isolation and the run-control training-output profile. Synthetic data only;
no MRI, no ISLES / SOOP access, no formal training, no v1 code."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from neuroscan_ai.run_control import build_plan
from neuroscan_ai.run_control.outputs import (
    BASELINE_TRAINING_FILES,
    TRAINING_FILES,
    run_output_path,
    training_output_labels,
)
from neuroscan_ai.v2.baseline_outputs import build_outputs
from neuroscan_ai.v2.baseline_preprocess import prepare_arrays
from neuroscan_ai.v2.baseline_selection import (
    THRESHOLD_CANDIDATES,
    SelectionError,
    search_threshold,
    select_checkpoint,
    write_once,
)
from neuroscan_ai.v2.safety_metrics import (
    MATCH_RULE,
    case_metrics,
    summarize,
    ungated_vs_gated,
)
from tests.test_run_bundle_comparison import _expected, _go, _inputs, w  # noqa: F401
from tests.test_v2_baseline_geometry import _affine, _volume
from tests.test_v2_real_runner import POLICIES, TRAIN, VAL, _cfg

REPO = Path(__file__).resolve().parents[1]
AFF = np.diag([1.0, 1.0, 2.0, 1.0])  # 2 uL voxels -> 0.002 mL
NEW_MODULES = (
    "neuroscan_ai.v2.unet_2d",
    "neuroscan_ai.v2.geometry",
    "neuroscan_ai.v2.baseline_preprocess",
    "neuroscan_ai.v2.baseline_outputs",
    "neuroscan_ai.v2.safety_metrics",
    "neuroscan_ai.v2.baseline_selection",
    "neuroscan_ai.v2.baseline_slices",
)


def _blank(shape=(20, 20, 6)):
    return np.zeros(shape, np.uint8)


# ---------------------------------------------------------------- 16-17 ungated / gated
def test_16_evaluation_is_ungated_based():
    dwi, adc, gt = _volume((32, 30, 5), seed=1)
    aff = _affine((0.9, 0.9, 5.0))
    case = prepare_arrays(dwi, adc, gt, {"dwi": aff, "adc": aff, "gt": aff})
    res = build_outputs(np.full(case.model_grid.shape, 0.9), case, 0.5)
    assert (
        res.record["official_output"] == "ungated" and res.record["gated_role"] == "comparison_only"
    )
    m = case_metrics(res.ungated, gt, aff)
    assert m["output"] == "ungated"
    with pytest.raises(ValueError):  # ungated and gated never mixed in one summary
        summarize([m, case_metrics(res.gated, gt, aff, output="gated")])
    rec = {
        "epoch": 1,
        "mean_dice_positive": 0.5,
        "val_loss": 0.3,
        "false_positive_voxel_volume_ml": 1.0,
    }
    with pytest.raises(SelectionError):
        select_checkpoint([{**rec, "threshold": 0.5, "output": "gated"}])
    with pytest.raises(SelectionError):
        select_checkpoint([{**rec, "threshold": 0.4, "output": "ungated"}])


def test_17_gated_is_comparison_only_and_outside_mask_kept():
    dwi, adc, gt = _volume((32, 30, 5), seed=2)
    aff = _affine((1.0, 1.0, 5.0))
    case = prepare_arrays(dwi, adc, gt, {"dwi": aff, "adc": aff, "gt": aff})
    prob = np.full(case.model_grid.shape, 0.9)  # predicts everywhere, also outside the brain
    res = build_outputs(prob, case, 0.5)
    outside = res.ungated.astype(bool) & ~res.brain_mask.astype(bool)
    rec = res.record
    assert outside.any() and rec["outside_brain_mask_voxels"] == int(outside.sum())
    assert rec["outside_brain_mask_ratio"] == round(int(outside.sum()) / int(res.ungated.sum()), 6)
    assert rec["touches_image_edge"] is True
    assert set(rec["boundary_warnings"]) == {"BOUNDARY_CONTACT", "EXTENDS_OUTSIDE_BRAINMASK"}
    assert "ungated_outside_brain_voxels" not in rec
    assert res.ungated.all()  # nothing removed from the official output
    assert np.array_equal(res.gated, res.ungated & res.brain_mask)
    assert (res.gated <= res.ungated).all()
    u = summarize([case_metrics(res.ungated, gt, aff)])
    g = summarize([case_metrics(res.gated, gt, aff, output="gated")])
    diff = ungated_vs_gated(u, g)
    assert diff["official"] == "ungated"
    assert diff["difference_gated_minus_ungated"]["false_positive_voxel_volume_ml"] < 0
    with pytest.raises(ValueError):
        ungated_vs_gated(g, u)


# ---------------------------------------------------------------- 18-21 safety metrics
def test_18_overlap_of_one_voxel_recorded_as_exploratory_candidate():
    gt, pred = _blank(), _blank()
    gt[2:6, 2:6, 1] = 1  # 16 voxels
    pred[5:9, 5:9, 1] = 1  # 16 voxels, 1 shared voxel (5, 5, 1)
    m = case_metrics(pred, gt, AFF)
    pairs = m["components"]["pairs"]
    assert m["components"]["match_rule"] == MATCH_RULE
    assert pairs == [
        {
            "pred_component": 1,
            "gt_component": 1,
            "overlap_voxels": 1,
            "pred_overlap_ratio": 0.0625,
            "gt_overlap_ratio": 0.0625,
            "dice": 0.0625,
            "iou": round(1 / 31, 6),
        }
    ]
    assert m["components"]["gt_components"][0]["detected_exploratory"] is True
    assert m["missed_gt_count"] == 0
    assert m["false_positive_voxel_volume_ml"] == pytest.approx(15 * 0.002)


def test_19_missed_gt_components_aggregated():
    gt, pred = _blank(), _blank()
    gt[1:4, 1:4, 1] = 1  # 9 vox, detected
    gt[10, 10, 3] = 1  # 1 vox, missed
    gt[14:16, 14:17, 4] = 1  # 6 vox, missed
    pred[2, 2, 1] = 1
    c1 = case_metrics(pred, gt, AFF)
    gt2 = _blank()
    gt2[5:7, 5:7, 2] = 1  # 4 vox, missed (empty prediction)
    c2 = case_metrics(_blank(), gt2, AFF)
    assert c1["missed_gt_count"] == 2 and sorted(c1["missed_gt_volumes_ml"]) == [0.002, 0.012]
    s = summarize([c1, c2])
    assert s["gt_components"] == 4 and s["missed_gt_count"] == 3
    assert s["lesion_recall_exploratory"] == 0.25
    assert s["missed_gt_volume_ml"] == {"n": 3, "min": 0.002, "median": 0.008, "max": 0.012}
    assert s["case_sensitivity_exploratory"] == 0.5
    assert c2["dice"] == 0.0


def test_20_small_lesion_recall_bins():
    aff = np.diag([1.0, 1.0, 1.0, 1.0])  # 1 uL voxels
    gt, pred = np.zeros((40, 40, 40), np.uint8), np.zeros((40, 40, 40), np.uint8)
    gt[1:4, 1:4, 1:4] = 1  # 27 uL = 0.027 mL  (< 0.1)       detected
    gt[10:15, 10:15, 10:14] = 1  # 100 uL = 0.1 mL (not < 0.1) missed
    gt[20:28, 20:28, 1:9] = 1  # 512 uL = 0.512 mL           detected
    gt[1:13, 25:37, 20:30] = 1  # 1440 uL = 1.44 mL           missed
    pred[2, 2, 2] = 1
    pred[24, 24, 4] = 1
    s = summarize([case_metrics(pred, gt, aff)])
    b = s["small_lesion_recall"]
    assert b["lt_0.1_ml"] == {"gt_components": 1, "detected": 1, "recall": 1.0}
    assert b["lt_0.5_ml"] == {"gt_components": 2, "detected": 1, "recall": 0.5}
    assert b["lt_1.0_ml"] == {"gt_components": 3, "detected": 2, "recall": round(2 / 3, 6)}


def test_21_no_lesion_fp_volume_and_slice_groups():
    pred = _blank()
    pred[3:5, 3:5, 2] = 1  # 4 voxels on a no-lesion case
    neg = case_metrics(pred, _blank(), AFF)
    gt = _blank()
    gt[8:10, 8:10, 1] = 1
    thick = case_metrics(gt, gt, np.diag([1.0, 1.0, 6.0, 1.0]))
    s = summarize([neg, thick])
    assert s["no_lesion_cases"] == 1 and s["no_lesion_fp_volume_ml"] == 0.008
    assert s["no_lesion_cases_with_prediction"] == 1
    assert neg["slice_group"] == "thin" and thick["slice_group"] == "thick"
    assert s["slice_groups"]["thin"]["cases"] == 1 and s["slice_groups"]["thick"]["cases"] == 1
    assert s["slice_groups"]["thick"]["mean_dice_positive"] == 1.0
    assert s["false_positive_voxel_volume_ml"] == 0.008


# ---------------------------------------------------------------- selection and threshold
def test_selection_rule_and_tie_breaks():
    base = {"threshold": 0.5, "output": "ungated"}
    recs = [
        {
            **base,
            "epoch": 3,
            "mean_dice_positive": 0.60000,
            "val_loss": 0.30,
            "false_positive_voxel_volume_ml": 2.0,
        },
        {
            **base,
            "epoch": 5,
            "mean_dice_positive": 0.60008,
            "val_loss": 0.25,
            "false_positive_voxel_volume_ml": 3.0,
        },
        {
            **base,
            "epoch": 7,
            "mean_dice_positive": 0.60009,
            "val_loss": 0.25,
            "false_positive_voxel_volume_ml": 1.0,
        },
        {
            **base,
            "epoch": 9,
            "mean_dice_positive": 0.55000,
            "val_loss": 0.10,
            "false_positive_voxel_volume_ml": 0.1,
        },
    ]
    r = select_checkpoint(recs)
    assert r["tied_epochs"] == [3, 5, 7] and r["selected_epoch"] == 7  # loss tie -> lower FP
    recs[2]["false_positive_voxel_volume_ml"] = 3.0
    assert select_checkpoint(recs)["selected_epoch"] == 5  # full tie -> earlier epoch
    recs[0]["mean_dice_positive"] = 0.59990  # now outside 1e-4 of the best
    assert 3 not in select_checkpoint(recs)["tied_epochs"]


def test_22_threshold_candidates_exact():
    expected = [float(Decimal(f"0.{d:02d}")) for d in range(10, 91, 5)]
    assert list(THRESHOLD_CANDIDATES) == expected
    assert len(THRESHOLD_CANDIDATES) == 17
    assert THRESHOLD_CANDIDATES[0] == 0.1 and THRESHOLD_CANDIDATES[-1] == 0.9
    assert 0.5 in THRESHOLD_CANDIDATES and 0.15 in THRESHOLD_CANDIDATES


def test_threshold_search_single_val_with_tie_break(tmp_path):
    calls = []

    def evaluate(t):
        calls.append(t)
        d = 0.7 if t in (0.35, 0.55, 0.6) else 0.5
        return {
            "output": "ungated",
            "mean_dice_positive": d,
            "lesion_recall_exploratory": 0.8,
            "missed_gt_count": 1,
            "small_lesion_recall": {
                "lt_0.1_ml": {"gt_components": 2, "detected": 1, "recall": 0.5}
            },
            "false_positive_voxel_volume_ml": 1.0,
            "no_lesion_fp_volume_ml": 0.2,
        }

    rec = search_threshold(evaluate, "a" * 64)
    assert calls == list(THRESHOLD_CANDIDATES)  # every candidate evaluated exactly once
    assert rec["selected_threshold"] == 0.55  # tie 0.35 / 0.55 / 0.6 -> closest to 0.5
    assert rec["split"] == "val" and rec["held_out_splits_used"] is False
    assert rec["checkpoint_fixed_before_search"] is True and rec["searches_performed"] == 1
    assert "optimistic" in rec["optimistic_bias"] and rec["fp_threshold_optimised"] is False
    assert rec["rows"][0]["missed_small_lesions"] == {"lt_0.1_ml": 1}
    with pytest.raises(SelectionError):
        search_threshold(evaluate, "a" * 64, split="test")
    with pytest.raises(SelectionError):
        search_threshold(evaluate, None)
    p = write_once(tmp_path / "training/threshold_search.json", rec)
    with pytest.raises(FileExistsError):
        write_once(p, rec)


# ---------------------------------------------------------------- 23-25 access and v1 isolation
def test_23_no_test_final_test_or_soop_access(monkeypatch, tmp_path):
    opened = []
    real_open = builtins.open

    def spy(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    import nibabel as nib

    dwi, adc, gt = _volume((28, 26, 4), seed=3)
    aff = _affine((0.8, 0.8, 5.0))
    paths = []
    for n, arr in (("dwi", dwi), ("adc", adc), ("gt", gt)):
        p = tmp_path / f"syn_{n}.nii.gz"
        nib.save(nib.Nifti1Image(arr, aff), str(p))
        paths.append(p)
    opened.clear()
    from neuroscan_ai.v2.baseline_preprocess import load_and_prepare

    case = load_and_prepare(*paths)
    res = build_outputs(np.full(case.model_grid.shape, 0.6), case, 0.5)
    summarize([case_metrics(res.ungated, gt, aff)])
    bad = ("test", "final", "soop", "isles", "data/raw", "derivatives")
    root = tmp_path.as_posix().lower()
    rel = [o.replace("\\", "/").lower().replace(root, "") for o in opened]
    assert rel  # the spy saw the synthetic reads
    touched = [o for o in rel if any(b in o for b in bad)]
    assert touched == []
    src = "\n".join(
        (REPO / "ai" / (m.replace(".", "/") + ".py")).read_text(encoding="utf-8").lower()
        for m in NEW_MODULES
    )
    for forbidden in ("final_test", "soop", "data/raw", "derivatives", "split_v1"):
        assert forbidden not in src


def test_24_no_v1_imports():
    prefixes = (
        "neuroscan_ai.training",
        "neuroscan_ai.inference",
        "neuroscan_ai.preprocessing",
        "neuroscan_ai.analysis",
        "neuroscan_ai.final_test",
        "monai",
    )
    code = (
        "import sys; sys.path.insert(0, 'ai')\n"
        + "".join(f"import {m}\n" for m in NEW_MODULES)
        + f"print([m for m in sys.modules if m.startswith({prefixes!r})])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_25_v1_frozen_paths_unchanged():
    frozen = [
        "ai/neuroscan_ai/training",
        "ai/neuroscan_ai/inference",
        "ai/neuroscan_ai/preprocessing",
        "ai/neuroscan_ai/analysis",
        "ai/neuroscan_ai/final_test",
        "configs/2d_baseline.yaml",
        "configs/25d_baseline.yaml",
        "data/splits",
    ]
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *frozen],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == ""


# ---------------------------------------------------------------- config and run control
def test_committed_baseline_config_matches_code():
    import yaml

    from neuroscan_ai.v2.baseline_preprocess import BRAIN_MASK_POLICY, PREPROCESS_POLICY
    from neuroscan_ai.v2.unet_2d import CHANNELS, EXPECTED_PARAMETERS, MODEL_ID

    cfg = yaml.safe_load((REPO / "configs/v2/2d_baseline.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["id"] == MODEL_ID
    assert cfg["model"]["channels"] == list(CHANNELS)
    assert cfg["model"]["trainable_parameters"] == EXPECTED_PARAMETERS
    assert cfg["preprocessing"]["policy"] == PREPROCESS_POLICY
    assert cfg["policies"]["brain_mask"] == BRAIN_MASK_POLICY
    tc = cfg["threshold_search"]["candidates"]
    assert (tc["start"], tc["stop"], tc["step"]) == (0.10, 0.90, 0.05)
    assert cfg["outputs"]["official_output"] == "ungated"
    assert cfg["selection"]["dice_tie_tolerance"] == 1e-4
    assert all(v is None for v in cfg["training"].values())  # pending approval
    assert set(cfg["data"]["forbidden_splits"]) == {"test", "final_test"}


def test_training_output_profiles():
    assert training_output_labels("smoke_v1") == tuple(TRAINING_FILES)
    base = training_output_labels("baseline_v1")
    assert set(TRAINING_FILES) < set(base)
    assert {
        "last_checkpoint",
        "selection_record",
        "threshold_search",
        "preprocessing_audit",
    } <= set(base)
    for label in BASELINE_TRAINING_FILES:
        assert run_output_path(Path("r"), label).parts[1] == "training"
    with pytest.raises(ValueError):
        training_output_labels("unknown")


def _train_plan(w, run_id):  # noqa: F811
    return build_plan(
        run_id=run_id,
        purpose="synthetic baseline profile",
        stage="train",
        dataset_id="synthetic",
        split_name="train",
        aliases=TRAIN,
        allowed_case_count=len(TRAIN),
        validation_aliases=VAL,
        model_id="synthetic",
        checkpoint=None,
        config=w.repo / "cfg.yaml",
        split_file=w.data / "split.json",
        policies=POLICIES,
        output_root=w.repo / "results/v2_runs",
        expected=_expected(w, TRAIN + VAL),
        input_files=_inputs(w, TRAIN + VAL),
    )


def _baseline_runner(w, write_extra):  # noqa: F811
    from neuroscan_ai.v2 import V2TwoDRunner

    class _Runner(V2TwoDRunner):
        training_outputs = "baseline_v1"

        def run(self, gate, plan):
            super().run(gate, plan)
            if write_extra:
                for label in set(BASELINE_TRAINING_FILES) - set(TRAINING_FILES):
                    run_output_path(gate.run_dir, label).write_text("{}", encoding="utf-8")

    return _Runner(_cfg(42), w.files)


def test_baseline_profile_requires_extra_training_outputs(w):  # noqa: F811
    p1 = _train_plan(w, "baseline-profile-missing")
    _go(w, p1, _baseline_runner(w, write_extra=False))
    post = json.loads((p1.run_dir / "post_run_manifest.json").read_text(encoding="utf-8"))
    assert post["run"]["status"] == "failed_incomplete_outputs"
    rec = post["reproducibility"]
    assert "run:last_checkpoint" in rec["missing_outputs"]
    assert "threshold_search" in rec["required_outputs"]
    p2 = _train_plan(w, "baseline-profile-complete")
    _go(w, p2, _baseline_runner(w, write_extra=True))
    post2 = json.loads((p2.run_dir / "post_run_manifest.json").read_text(encoding="utf-8"))
    assert post2["run"]["status"] == "completed"
    assert "threshold_search" in post2["outputs"]["aggregate_results"]


def test_unknown_profile_refused_before_gate(w):  # noqa: F811
    from neuroscan_ai.v2 import V2TwoDRunner

    class _Bad(V2TwoDRunner):
        training_outputs = "nope"

    plan = _train_plan(w, "baseline-profile-unknown")
    with pytest.raises(ValueError):
        _go(w, plan, _Bad(_cfg(42), w.files))
    assert not plan.run_dir.exists()


def test_17b_boundary_record_interior_and_empty():
    from neuroscan_ai.v2.baseline_outputs import boundary_record
    from neuroscan_ai.v2.geometry import Grid

    grid = Grid((20, 20, 12), _affine((1.0, 1.0, 1.0)))
    brain = np.zeros(grid.shape, bool)
    brain[2:18, 2:18, 1:11] = True
    pred = np.zeros(grid.shape, np.uint8)
    pred[9:11, 9:11, 5:7] = 1  # deep inside the brain mask
    rec = boundary_record(pred, brain, grid)
    assert rec["outside_brain_mask_voxels"] == 0 and rec["outside_brain_mask_ratio"] == 0.0
    assert rec["touches_image_edge"] is False and rec["boundary_warnings"] == []
    assert rec["min_distance_to_brain_boundary_mm"] > 3.0
    pred[0, 9, 5] = 1  # outside the brain and on the image edge: kept and flagged
    rec = boundary_record(pred, brain, grid)
    assert rec["outside_brain_mask_voxels"] == 1 and rec["touches_image_edge"] is True
    assert "EXTENDS_OUTSIDE_BRAINMASK" in rec["boundary_warnings"]
    empty = boundary_record(np.zeros(grid.shape, np.uint8), brain, grid)
    assert empty["outside_brain_mask_voxels"] == 0 and empty["outside_brain_mask_ratio"] is None
    assert empty["boundary_warnings"] == [] and empty["touches_image_edge"] is False
