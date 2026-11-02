"""Metrics-only failure analysis: selection, aliases, flags, privacy and access guards.

Synthetic Final Test tables only (synthetic case keys sub-strokecase91NN); no real case ID,
image, prediction or project result is used.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import pandas as pd
import pytest

from neuroscan_ai.failure_analysis import audit as au
from neuroscan_ai.failure_analysis import schemas as sc
from neuroscan_ai.failure_analysis import selection as sel
from neuroscan_ai.failure_analysis import taxonomy as tx
from neuroscan_ai.failure_analysis.run import FailureAnalysisError, run
from tests.conftest import REPO_ROOT

K = "sub-strokecase91"  # synthetic key prefix


def row(cid, dice, spacing="thin", **kw):
    has = kw.pop("has_lesion", True)
    tp, fp, fn = kw.pop("tp", 100), kw.pop("fp", 10), kw.pop("fn", 10)
    r = {
        "case_id": f"{K}{cid}",
        "has_lesion": has,
        "dice": dice,
        "iou": dice,
        "sensitivity": kw.pop("sens", dice),
        "precision": kw.pop("prec", dice),
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "gt_voxels": tp + fn,
        "prediction_voxels": kw.pop("pred", tp + fp),
        "gt_volume_ml": 1.0,
        "prediction_volume_ml": 1.0,
        "volume_error_ml": 0.0,
        "absolute_volume_error_ml": 0.0,
        "relative_volume_error_percent": 0.0,
        "component_count_gt": 2,
        "component_count_prediction": 3,
        "negative_outcome": kw.pop("neg", None),
        "false_positive_volume_ml": kw.pop("fpv", None),
        "false_positive_component_count": kw.pop("fpc", None),
        "z_spacing": 4.8 if spacing == "thick" else 2.0,
        "volume_group": "[1,10) mL",
        "spacing_group": spacing,
        "adc_unit_group": "mm2/s",
        "obliquity_group": "[1,10) deg",
        "center": "unavailable",
    }
    assert not kw
    return r


def dataset():
    """(2D rows, 2.5D rows). Expected groups are documented in the tests below."""
    d2 = [
        row("01", 0.30, "thick"),
        row("02", 0.40, sens=0.3, prec=0.8, tp=30, fp=8, fn=70),
        row("03", 0.70, "thick"),
        row("04", 0.60),
        row("05", 0.70),
        row("06", 0.80),
        row("07", 0.90),
        row("08", 0.88),
        row("09", 0.90, "thick"),
        row("10", 0.86, "thick"),
        row("11", 0.75, "thick"),
        row("12", 0.85),
        row("13", 0.95, "thick"),
        row(
            "90",
            None,
            has_lesion=False,
            tp=0,
            fp=1,
            fn=0,
            neg="false_positive_case",
            fpv=0.008,
            fpc=1,
        ),
    ]
    d25 = [
        row("01", 0.0, "thick", sens=0.0, prec=0.0, tp=0, fp=50, fn=80, pred=50),
        row("02", 0.60),
        row("03", 0.45, "thick", sens=0.8, prec=0.3, tp=80, fp=190, fn=20),
        row("04", 0.90),
        row("05", 0.95),
        row("06", 0.81),
        row("07", 0.92),
        row("08", 0.86),
        row("09", 0.88, "thick"),
        row("10", 0.87, "thick"),
        row("11", 0.65, "thick"),
        row("12", 0.85),
        row("13", 0.95, "thick"),
        row(
            "90",
            None,
            has_lesion=False,
            tp=0,
            fp=2,
            fn=0,
            neg="false_positive_case",
            fpv=0.016,
            fpc=2,
        ),
    ]
    return d2, d25


def write_final(folder: Path, d2=None, d25=None) -> Path:
    if d2 is None:
        d2, d25 = dataset()
    folder.mkdir(parents=True)
    pd.DataFrame(d2).to_csv(folder / "patient_metrics_2d.csv", index=False)
    pd.DataFrame(d25).to_csv(folder / "patient_metrics_25d.csv", index=False)
    pd.DataFrame([{"case_id": r["case_id"]} for r in d2]).to_csv(folder / "paired_metrics.csv")
    fails = [{"case_id": f"{K}{c}", "dice_2d": 0, "dice_25d": 0} for c in ("01", "02", "03")]
    for name, obj in (
        ("aggregate_metrics.json", {}),
        ("subgroup_metrics.json", {}),
        ("negative_case_summary.json", {}),
        ("execution_audit.json", {}),
        ("qa_manifest.json", {"failure_cases": fails, "selected": []}),
    ):
        (folder / name).write_text(json.dumps(obj), encoding="utf-8")
    (folder / "COMPLETED").write_text("{}")
    return folder


@pytest.fixture
def env(tmp_path):
    return write_final(tmp_path / "final"), tmp_path / "fa" / "v1"


def aliases(out: Path) -> dict[str, str]:
    """original ID -> alias (reads the internal map, as only the tests may)."""
    m = json.loads((out / "_internal/aliases.json").read_text(encoding="utf-8"))
    return {v: k for k, v in m.items()}


def groups_by_id(d2, d25):
    recs = sel.build_records(d2, d25)
    return {g: [c.replace(K, "") for c in ids] for g, ids in sel.select(recs).items()}


# --- 1-7 selection -------------------------------------------------------------------------------


def test_groups_follow_the_fixed_rules():
    g = groups_by_id(*dataset())
    assert g["F"] == ["01", "02", "03"]  # ordered by the lower Dice of the two models
    assert g["Z"] == ["01"]
    assert g["I"] == ["04", "05", "02"]  # +0.30, +0.25, +0.20
    assert g["W"] == ["01", "03", "11"]  # -0.30, -0.25, -0.10
    assert g["N"] == ["90"]
    assert g["G"] == ["08", "09"]  # thin median 0.87 -> 08; thick median 0.89 -> 09


# --- 8-10 determinism and aliases ----------------------------------------------------------------


def test_ties_and_input_order_are_deterministic():
    d2 = [row(c, 0.6) for c in ("05", "03", "04", "01", "02")]
    d25 = [row(c, 0.7) for c in ("05", "03", "04", "01", "02")]  # all deltas tie at +0.1
    first = groups_by_id(d2, d25)
    assert first["I"] == ["01", "02", "03"] and first["W"] == ["01", "02", "03"]
    for seed in range(5):
        random.Random(seed).shuffle(d2)
        random.Random(seed + 10).shuffle(d25)
        assert groups_by_id(d2, d25) == first


def test_aliases_follow_group_priority_and_are_reused(env):
    final, out = env
    run(final, out, REPO_ROOT)
    a = aliases(out)
    expect = {
        "01": "fa-01",
        "02": "fa-02",
        "03": "fa-03",
        "04": "fa-04",
        "05": "fa-05",
        "11": "fa-06",
        "90": "fa-07",
        "08": "fa-08",
        "09": "fa-09",
    }
    assert {k.replace(K, ""): v for k, v in a.items()} == expect
    s = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    one = next(x for x in s["aliases"] if x["alias"] == "fa-01")
    assert one["groups"] == ["F", "Z", "W"] and len(one["reasons"]) == 3
    assert s["groups"]["I"] == ["fa-04", "fa-05", "fa-02"]
    assert len(list((out / "cases").iterdir())) == 9


# --- 11-12 privacy -------------------------------------------------------------------------------


def test_original_ids_only_in_internal_files(env):
    final, out = env
    run(final, out, REPO_ROOT)
    for p in out.rglob("*"):
        if p.is_file() and "_internal" not in p.parts:
            assert K not in p.read_text(encoding="utf-8"), p
    assert K in (out / "_internal/aliases.json").read_text(encoding="utf-8")


def test_console_has_no_original_ids(env, capsys):
    final, out = env
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import analyze_failure_cases as cli

    assert cli.main(["--final-test-dir", str(final), "--output-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "metrics-only: 9 aliases" in printed and K not in printed


# --- 13-18 taxonomy ------------------------------------------------------------------------------


def summaries(env):
    final, out = env
    run(final, out, REPO_ROOT)
    a = aliases(out)
    return {
        k.replace(K, ""): json.loads((out / "cases" / v / "summary.json").read_text("utf-8"))
        for k, v in a.items()
    }


def test_t1_t2_t3(env):
    s = summaries(env)
    assert s["02"]["taxonomy_flags"]["2d"]["T1"] == sc.FLAGGED
    assert s["02"]["taxonomy_flags"]["2d"]["T2"] == sc.NOT_FLAGGED
    assert s["03"]["taxonomy_flags"]["25d"]["T2"] == sc.FLAGGED
    assert s["01"]["taxonomy_flags"]["25d"]["T3"] == sc.FLAGGED
    assert s["01"]["taxonomy_flags"]["2d"]["T3"] == sc.NOT_FLAGGED
    assert s["90"]["taxonomy_flags"]["2d"] == {c: sc.NOT_APPLICABLE for c in ("T1", "T2", "T3")}


def test_pending_codes_and_t8_candidate(env):
    s = summaries(env)
    codes = [p["code"] for p in s["02"]["pending_image_review"]]
    assert codes == ["T4", "T5", "T6", "T7", "T9"]
    assert [p["code"] for p in s["90"]["pending_image_review"]] == ["T5", "T7", "T9"]
    assert s["03"]["taxonomy_flags"]["T8"] == sc.REVIEW_CANDIDATE  # thick, 2.5D worse
    assert s["09"]["taxonomy_flags"]["T8"] == sc.REVIEW_CANDIDATE
    assert s["02"]["taxonomy_flags"]["T8"] == sc.NOT_FLAGGED  # thin
    assert s["90"]["taxonomy_flags"]["T8"] == sc.NOT_APPLICABLE
    assert s["02"]["observations"]["component_count_difference_prediction_minus_gt"] == {
        "2d": 1.0,
        "25d": 1.0,
    }


def test_observations_and_hypotheses_are_separate(env):
    for s in summaries(env).values():
        assert set(s) >= {
            "observations",
            "taxonomy_flags",
            "pending_image_review",
            "hypotheses",
            "unknowns",
            "suggested_experiments",
        }
        assert s["hypotheses"] == [] and s["suggested_experiments"] == []
        assert "metrics" in s["observations"] and "metadata" in s["observations"]


# --- 19-22 access guards -------------------------------------------------------------------------


def test_no_nifti_model_or_map_and_final_test_untouched(env):
    final, out = env
    before = au.snapshot(final)
    res = run(final, out, REPO_ROOT)
    a = res["audit"]
    assert a["access"] == {
        "test_image_access": 0,
        "dwi_access": 0,
        "adc_access": 0,
        "gt_nifti_access": 0,
        "prediction_nifti_access": 0,
        "nifti_open_attempts_refused": 0,
    }
    assert a["model_inference_runs"] == 0 and a["probability_maps_generated"] == 0
    assert a["image_or_map_files_written"] == []
    assert not [p for p in out.rglob("*") if p.suffix.lower() in (".nii", ".gz", ".png")]
    assert au.snapshot(final) == before and a["final_test_result"]["unchanged_after_run"]
    assert a["public_output_original_id_scan"]["passed"]


def test_nifti_open_is_refused_inside_the_guard(tmp_path):
    counter = {"nifti_open_attempts": 0}
    with au.no_image_access(counter), pytest.raises(au.MetricsOnlyViolation):
        nib.load(tmp_path / "x.nii.gz")
    assert counter["nifti_open_attempts"] == 1
    assert nib.load is not None and nib.load.__name__ == "load"  # restored


def test_metrics_only_run_does_not_import_torch(env):
    final, out = env
    code = (
        "import sys; sys.path.insert(0, 'ai');"
        "from pathlib import Path; from neuroscan_ai.failure_analysis.run import run;"
        f"run(Path(r'{final}'), Path(r'{out}'), Path('.'));"
        "print('TORCH', 'torch' in sys.modules)"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "TORCH False" in p.stdout


# --- 23-25 Git exclusion and refusals ------------------------------------------------------------


def test_output_is_git_ignored():
    p = subprocess.run(
        ["git", "check-ignore", "-q", "results/failure_analysis/v1/x.json"], cwd=REPO_ROOT
    )
    assert p.returncode == 0


@pytest.mark.parametrize("damage", ["no_completed", "failed_marker", "missing_csv"])
def test_incomplete_final_test_is_refused(env, damage):
    final, out = env
    if damage == "no_completed":
        (final / "COMPLETED").unlink()
    elif damage == "failed_marker":
        (final / "FAILED").write_text("{}")
    else:
        (final / "patient_metrics_25d.csv").unlink()
    with pytest.raises(FailureAnalysisError, match="incomplete"):
        run(final, out, REPO_ROOT)
    assert not out.exists()


def test_existing_output_is_never_overwritten(env):
    final, out = env
    run(final, out, REPO_ROOT)
    marker = (out / "audit.json").read_text(encoding="utf-8")
    with pytest.raises(FailureAnalysisError, match="never overwritten"):
        run(final, out, REPO_ROOT)
    assert (out / "audit.json").read_text(encoding="utf-8") == marker


def test_aggregate_counts(env):
    final, out = env
    agg = run(final, out, REPO_ROOT)["aggregate"]
    assert agg["unique_aliases"] == 9
    assert agg["group_counts"] == {"F": 3, "Z": 1, "I": 3, "W": 3, "N": 1, "G": 2}
    assert agg["consistent_with_final_test_failure_cases"] is True
    assert agg["taxonomy_counts"]["25d"]["T3"][sc.FLAGGED] == 1
    assert agg["pending_image_review_counts"]["T4"] == 8  # positives only
    rows = tx.overview_rows(
        [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((out / "cases").glob("*/summary.json"))
        ]
    )
    assert [r["alias"] for r in rows] == [f"fa-{i:02d}" for i in range(1, 10)]
