"""Image review of selected failure cases — synthetic patients only.

A synthetic repository (raw DWI / ADC / GT, processed meta.json, stored binary predictions,
patient metrics computed from those arrays) goes through the metrics-only stage and then the
image review. No real case ID, image, prediction or project result is used.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neuroscan_ai.evaluation.segmentation import evaluate_case
from neuroscan_ai.failure_analysis import image_review as ir
from neuroscan_ai.failure_analysis import image_run as rr
from neuroscan_ai.failure_analysis import schemas as sc
from neuroscan_ai.failure_analysis.audit import snapshot
from neuroscan_ai.failure_analysis.run import run as run_metrics
from neuroscan_ai.research.compare import FN, FP, TP
from tests.conftest import REPO_ROOT

K = "sub-strokecase92"
SHAPE, SPACING = (24, 24, 8), (1.0, 1.0, 3.0)
AFF = np.diag([*SPACING, 1.0])
PHRASE = rr.APPROVAL_PHRASE


def box(*sl) -> np.ndarray:
    m = np.zeros(SHAPE, np.uint8)
    m[sl] = 1
    return m


G1 = box(slice(8, 14), slice(8, 14), slice(3, 6))  # 108 voxels
G2 = box(slice(17, 19), slice(17, 19), slice(4, 5))  # 4 voxels = 0.012 mL (small)
FAR = box(slice(18, 20), slice(5, 7), slice(0, 1))  # > 10 mm from G1 / G2
BRAIN = box(slice(4, 21), slice(4, 21), slice(0, 8)).astype(bool)


def shifted(m, dx):
    return np.roll(m, dx, axis=0)


def cases():
    split = (G1 | G2).copy()
    split[10:12, :, :] = 0  # cuts G1 into two parts -> split group
    return {
        "01": ("thin", G1 | G2, G1, G1 | G2 | FAR),  # 2D misses small G2; 2.5D adds far FP
        "02": ("thin", G1 | G2, shifted(G1, 1) | G2, split),  # boundary shift / split
        "03": (
            "thick",
            G1,
            box(slice(8, 9), slice(8, 9), slice(3, 4)),
            box(slice(15, 17), slice(15, 17), slice(3, 4)),
        ),  # under-seg / disjoint
        "04": ("thin", G1 | G2, G1 | G2, G1 | G2),  # both perfect
        "90": (
            "thin",
            np.zeros(SHAPE, np.uint8),
            box(slice(6, 7), slice(6, 7), slice(1, 2)),
            box(slice(6, 8), slice(6, 7), slice(1, 2)),
        ),  # negative
    }


def save(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(arr), AFF), str(path))


def build(tmp: Path) -> tuple[Path, Path, Path]:
    root = tmp / "repo"
    final = root / "results/final_test/v1"
    rows = {"2d": [], "25d": []}
    for c, (spacing_group, gt, p2, p25) in cases().items():
        cid = f"{K}{c}"
        lesion_signal = gt | (FAR if c == "01" else 0)
        dwi = np.where(BRAIN, 100.0 + 200.0 * lesion_signal, 0.0)
        adc = np.where(BRAIN, 800.0 - 500.0 * lesion_signal, 0.0)
        raw = root / "data/raw" / cid
        save(raw / "dwi.nii.gz", dwi.astype(np.float32))
        save(raw / "adc.nii.gz", adc.astype(np.float32))
        save(raw / "msk.nii.gz", gt.astype(np.float32))
        meta = {
            "source": {
                "dwi": f"data/raw/{cid}/dwi.nii.gz",
                "adc": f"data/raw/{cid}/adc.nii.gz",
                "label": f"data/raw/{cid}/msk.nii.gz",
            }
        }
        (root / "data/processed/v1" / cid).mkdir(parents=True)
        (root / "data/processed/v1" / cid / "meta.json").write_text(json.dumps(meta))
        for m, p in (("2d", p2), ("25d", p25)):
            save(final / f"predictions_{m}" / f"{cid}_pred.nii.gz", p.astype(np.uint8))
            r = evaluate_case(p, gt, SPACING, cid)
            r.update(
                spacing_group=spacing_group,
                volume_group="[1,10) mL",
                center="unavailable",
                adc_unit_group="mm2/s",
                obliquity_group="[1,10) deg",
            )
            rows[m].append(r)
    for m in sc.MODELS:
        pd.DataFrame(rows[m]).to_csv(final / f"patient_metrics_{m}.csv", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(final / "paired_metrics.csv")
    for name in (
        "aggregate_metrics.json",
        "subgroup_metrics.json",
        "negative_case_summary.json",
        "execution_audit.json",
    ):
        (final / name).write_text("{}")
    (final / "qa_manifest.json").write_text(json.dumps({"failure_cases": [], "selected": []}))
    (final / "COMPLETED").write_text("{}")
    fa = root / "results/failure_analysis/v1"
    run_metrics(final, fa, root)
    return root, final, fa


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    return build(tmp_path_factory.mktemp("img"))


@pytest.fixture
def env(template, tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(template[0], root)
    return root, root / "results/final_test/v1", root / "results/failure_analysis/v1"


@pytest.fixture(scope="module")
def reviewed(template, tmp_path_factory):
    root = tmp_path_factory.mktemp("rev") / "repo"
    shutil.copytree(template[0], root)
    final, fa = root / "results/final_test/v1", root / "results/failure_analysis/v1"
    before = snapshot(final)
    res = rr.run(fa, final, root, PHRASE)
    return root, final, fa, res, before


def alias_of(fa: Path) -> dict[str, str]:
    m = json.loads((fa / "_internal/aliases.json").read_text(encoding="utf-8"))
    return {v.replace(K, ""): k for k, v in m.items()}


def summary(fa, res_or_none, c):
    a = alias_of(fa)[c]
    return json.loads((fa / "image_review/cases" / a / "summary.json").read_text("utf-8"))


# --- approval, selection limits ------------------------------------------------------------------


@pytest.mark.parametrize("phrase", [None, "yes", PHRASE.lower()])
def test_no_approval_no_image_access(env, monkeypatch, phrase):
    root, final, fa = env
    opened = []
    monkeypatch.setattr(nib, "load", lambda *a, **k: opened.append(a) or None)
    with pytest.raises(rr.ImageReviewRefused, match="approval"):
        rr.run(fa, final, root, phrase)
    assert opened == [] and not (fa / "image_review").exists()


def test_files_outside_the_selection_are_refused(tmp_path):
    a, b = tmp_path / "a.nii.gz", tmp_path / "b.nii.gz"
    save(a, np.zeros((2, 2, 2)))
    save(b, np.zeros((2, 2, 2)))
    counter = {}
    with rr.allow_list({a: "dwi"}, counter):
        nib.load(a)
        with pytest.raises(rr.NotAllowedFile):
            nib.load(b)
    assert counter == {"dwi": 1, "refused": 1}


def test_more_than_ten_aliases_is_refused(env):
    root, final, fa = env
    p = fa / "_internal/aliases.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    m.update({f"fa-{i:02d}": f"x{i}" for i in range(20, 31)})
    p.write_text(json.dumps(m))
    s = json.loads((fa / "selection.json").read_text(encoding="utf-8"))
    s["aliases"] += [{"alias": f"fa-{i:02d}", "groups": [], "reasons": []} for i in range(20, 31)]
    (fa / "selection.json").write_text(json.dumps(s))
    with pytest.raises(rr.ImageReviewRefused, match="maximum 10"):
        rr.run(fa, final, root, PHRASE)


@pytest.mark.parametrize("damage", ["completed", "failed", "existing_output"])
def test_preflight_refusals(env, damage):
    root, final, fa = env
    if damage == "completed":
        (final / "COMPLETED").unlink()
    elif damage == "failed":
        (final / "FAILED").write_text("{}")
    else:
        (fa / "image_review").mkdir()
    with pytest.raises(rr.ImageReviewRefused):
        rr.run(fa, final, root, PHRASE)


# --- end-to-end on synthetic cases ---------------------------------------------------------------


def test_run_opens_only_selected_files_and_changes_nothing(reviewed):
    root, final, fa, res, before = reviewed
    a = res["audit"]
    n = a["aliases_accessed"]
    assert n == a["aliases_analyzed"] == len(alias_of(fa)) <= 10
    assert a["files_opened"] == {c: n for c in rr.CATEGORIES}
    assert a["refused_file_attempts"] == 0
    assert a["model_inference_runs"] == 0 and a["probability_maps_generated"] == 0
    assert a["torch_imported_by_this_run"] is False
    assert a["final_test_unchanged"] and snapshot(final) == before
    assert a["metrics_only_files_unchanged"]
    assert a["public_output_original_id_scan"]["passed"]
    assert a["difference_maps_written"] == 3 * n and a["figures_written"] > 0


def test_outputs_contain_aliases_only(reviewed):
    root, final, fa, res, _ = reviewed
    out = fa / "image_review"
    for p in out.rglob("*"):
        if "_internal" in p.parts:
            continue
        assert K not in p.name
        if p.suffix in (".json", ".csv"):
            text = p.read_text(encoding="utf-8")
            assert K not in text and str(root) not in text
    pngs = list(out.rglob("*.png"))
    assert pngs and all(p.name.startswith("fa-") for p in pngs)
    assert not list(out.rglob("*prob*"))


def test_difference_maps_match_stored_counts(reviewed):
    root, final, fa, res, _ = reviewed
    for c, (_, gt, p2, p25) in cases().items():
        a = alias_of(fa)[c]
        for m, p in (("2d", p2), ("25d", p25)):
            img = nib.load(fa / "image_review/cases" / a / "maps" / f"difference_{m}.nii.gz")
            d = np.asanyarray(img.dataobj)
            assert d.shape == gt.shape and np.allclose(img.affine, AFF)
            r = evaluate_case(p, gt, SPACING)
            assert ((d == TP).sum(), (d == FP).sum(), (d == FN).sum()) == (
                r["tp_voxels"],
                r["fp_voxels"],
                r["fn_voxels"],
            )


def test_count_mismatch_stops_only_that_case(env):
    root, final, fa = env
    a = alias_of(fa)["04"]
    p = fa / "cases" / a / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s["observations"]["metrics"]["2d"]["tp_voxels"] += 1
    p.write_text(json.dumps(s))
    res = rr.run(fa, final, root, PHRASE)
    stopped = res["audit"]["aliases_stopped"]
    assert [x["alias"] for x in stopped] == [a] and "differ" in stopped[0]["reason"]
    assert res["audit"]["aliases_analyzed"] == res["audit"]["aliases_accessed"] - 1
    assert not (fa / "image_review/cases" / a / "maps").exists()


def test_flags_t1_to_t9(reviewed):
    root, final, fa, res, _ = reviewed
    s1, s2, s3, s4 = (summary(fa, res, c) for c in ("01", "02", "03", "04"))
    assert s1["taxonomy_flags"]["2d"]["T4"] == sc.FLAGGED  # small GT component missed
    assert s1["taxonomy_flags"]["25d"]["T4"] == sc.NOT_FLAGGED
    assert s1["taxonomy_flags"]["25d"]["T5"] == sc.FLAGGED  # far FP component
    assert s1["taxonomy_flags"]["25d"]["T9"] == sc.REVIEW_CANDIDATE  # lesion-like FP signal
    assert s2["taxonomy_flags"]["2d"]["T6"] == sc.REVIEW_CANDIDATE  # boundary shift
    assert s2["taxonomy_flags"]["25d"]["T7"] == sc.FLAGGED  # split
    assert s3["taxonomy_flags"]["2d"]["T1"] == sc.FLAGGED
    assert s3["taxonomy_flags"]["25d"]["T3"] == sc.FLAGGED  # TP = 0
    assert s3["taxonomy_flags"]["T8"] == sc.REVIEW_CANDIDATE  # thick + 2.5D worse
    assert all(
        s4["taxonomy_flags"][m][c] == sc.NOT_FLAGGED
        for m in sc.MODELS
        for c in ("T1", "T2", "T3", "T4", "T5", "T7")
    )
    neg = summary(fa, res, "90")
    assert neg["taxonomy_flags"]["2d"]["T1"] == sc.NOT_APPLICABLE


def test_observations_and_hypothesis_templates(reviewed):
    root, final, fa, res, _ = reviewed
    for c in ("01", "02", "03"):
        s = summary(fa, res, c)
        assert all(o["status"] == "observed" for o in s["observations"])
        obs_ids = {o["id"] for o in s["observations"]}
        for h in s["hypotheses"]:
            assert h["status"] == "template" and h["confidence"] in ("possible", "undetermined")
            assert h["alternative_explanations"] and h["verification"]
            assert set(h["evidence"]) <= obs_ids
        assert s["alternative_explanations"] and "suggested_experiments" in s


def test_components_and_matching_files(reviewed):
    root, final, fa, res, _ = reviewed
    d = fa / "image_review/cases" / alias_of(fa)["01"]
    gt = pd.read_csv(d / "components_gt.csv")
    assert len(gt) == 2 and set(gt.columns) >= {
        "nearest_2d_mm",
        "nearest_25d_mm",
        "brain_boundary_mm",
        "representative_voxel",
    }
    p25 = pd.read_csv(d / "components_25d.csv")
    assert (p25.nearest_other_mm > 10).sum() == 1
    m = json.loads((d / "matching_25d.json").read_text(encoding="utf-8"))
    assert {g["group_type"] for g in m["groups"]} >= {"prediction_only"}
    i = json.loads((d / "intensity.json").read_text(encoding="utf-8"))
    assert i["2d"]["dwi"]["fp"]["median"] is None  # empty region stays undefined
    assert i["2d"]["dwi"]["fp"]["voxel_count"] == 0


# --- unit pieces ----------------------------------------------------------------------------------


def test_26_connectivity_and_representative_inside():
    m = np.zeros((5, 5, 5), np.uint8)
    m[1, 1, 1] = m[2, 2, 2] = 1  # corner neighbours -> one component
    m[4, 4, 0] = 1
    rows = ir.component_table(m, (1, 1, 1), np.eye(4), None, np.ones_like(m, bool))
    assert [r["voxel_count"] for r in rows] == [2, 1]
    for r in rows:
        assert m[tuple(r["representative_voxel"])] == 1


def test_distances_in_mm():
    a = np.zeros((10, 10, 6), np.uint8)
    b = np.zeros_like(a)
    a[2, 2, 1] = 1
    b[2, 2, 4] = 1  # 3 slices apart at 3 mm
    rows = ir.component_table(a, (1.0, 1.0, 3.0), np.eye(4), b, np.ones_like(a, bool))
    assert rows[0]["nearest_other_mm"] == pytest.approx(9.0)
    brain = np.zeros_like(a, bool)
    brain[1:9, 1:9, :] = True
    rows = ir.component_table(a, (1.0, 1.0, 3.0), np.eye(4), None, brain)
    assert rows[0]["brain_boundary_mm"] == pytest.approx(2.0)


def test_slice_candidates_dedup_max5_and_edges():
    gt = np.zeros((4, 4, 10), np.uint8)
    gt[1, 1, 0] = 1
    d = np.zeros_like(gt)
    d[1, 1, 0], d[2, 2, 3], d[1, 2, 5] = FN, FP, FP
    md = np.zeros_like(gt)
    md[0, 0, 7] = ir.MD_2D_ONLY
    s = ir.select_slices(gt, d, d, md, [0, 9, 8, 6])
    assert [x["z"] for x in s] == [0, 3, 7, 9, 8]  # gt/fn merged at 0, max 5
    assert "fn_area_max" in s[0]["reasons"] and len(s) == 5
    assert ir.neighbours(0, 10) == [None, 0, 1] and ir.neighbours(9, 10) == [8, 9, None]


def test_output_is_git_ignored():
    p = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "results/failure_analysis/v1/image_review/cases/fa-01/summary.json",
        ],
        cwd=REPO_ROOT,
    )
    assert p.returncode == 0
