"""Component safety analysis (stage 1): synthetic data only, no real MRI, no model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neuroscan_ai.component_analysis import ComponentAnalysisError, analyze_case, extract_components
from neuroscan_ai.component_analysis.boundary import BrainBoundary
from neuroscan_ai.component_analysis.extraction import reconstructs
from neuroscan_ai.component_analysis.records import is_git_ignored, write_record

REPO = Path(__file__).resolve().parents[1]
SHAPE = (20, 20, 10)
AFF = np.diag([1.0, 1.0, 2.0, 1.0])


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.array(a)
    a.flags.writeable = False  # any in-place change by the analysis would raise
    return a


def _case(pred, **kw):
    pred = np.asarray(pred, dtype=np.uint8)
    brain = np.zeros(pred.shape, np.uint8)
    brain[2:-2, 2:-2, 1:-1] = 1
    args = dict(
        alias="synth-01",
        prediction=_ro(pred),
        probability=_ro(pred * 0.9),
        dwi=_ro(np.full(pred.shape, 100.0)),
        adc=_ro(np.full(pred.shape, 700.0)),
        affine=AFF,
        brain_mask=_ro(brain),
        brain_mask_policy_id="synthetic",
        adc_storage_status="P",
        adc_content_status="confirmed_adc",
        adc_unit_status="confirmed",
    )
    args.update(kw)
    return analyze_case(**args)


def _blobs():
    m = np.zeros(SHAPE, np.uint8)
    m[5:8, 5:8, 3:5] = 1  # 18 voxels
    m[12:14, 12:14, 6] = 1  # 4 voxels
    m[10, 4, 2] = 1  # 1 voxel
    return m


# ---------------------------------------------------------------- 1-9 extraction
def test_01_no_component():
    r = _case(np.zeros(SHAPE, np.uint8))
    assert r["case"]["component_count"] == 0 and r["components"] == []


def test_02_one_component():
    m = np.zeros(SHAPE, np.uint8)
    m[6:9, 6:9, 4] = 1
    r = _case(m)
    assert r["case"]["component_count"] == 1 and r["components"][0]["voxel_count"] == 9


def test_03_separate_components():
    r = _case(_blobs())
    assert [c["voxel_count"] for c in r["components"]] == [18, 4, 1]


def test_04_diagonal_26_connectivity():
    m = np.zeros(SHAPE, np.uint8)
    m[5, 5, 5] = m[6, 6, 6] = m[7, 7, 7] = 1  # corner-touching only
    labels, n = extract_components(m)
    assert n == 1 and (labels > 0).sum() == 3


def test_05_ids_by_volume():
    labels, n = extract_components(_blobs())
    assert (
        n == 3
        and (labels == 1).sum() == 18
        and (labels == 2).sum() == 4
        and (labels == 3).sum() == 1
    )


def test_06_equal_volume_ordered_by_k_then_j_then_i():
    m = np.zeros(SHAPE, np.uint8)
    m[2, 2, 8] = 1  # k=8
    m[15, 15, 1] = 1  # k=1, j=15
    m[15, 5, 1] = 1  # k=1, j=5  -> first
    labels, _ = extract_components(m)
    assert labels[15, 5, 1] == 1 and labels[15, 15, 1] == 2 and labels[2, 2, 8] == 3
    m2 = np.zeros(SHAPE, np.uint8)
    m2[12, 5, 1] = m2[4, 5, 1] = 1  # same k, j -> smaller i first
    lab2, _ = extract_components(m2)
    assert lab2[4, 5, 1] == 1 and lab2[12, 5, 1] == 2
    assert np.array_equal(extract_components(m)[0], labels)  # deterministic


def test_07_union_equals_prediction():
    m = _blobs()
    labels, _ = extract_components(m)
    assert reconstructs(labels, m)


def test_08_small_components_kept():
    r = _case(_blobs())
    assert sum(c["voxel_count"] for c in r["components"]) == int(_blobs().sum())
    assert r["components"][-1]["voxel_count"] == 1


def test_09_volume_ml():
    r = _case(_blobs())
    c = r["components"][0]
    assert c["voxel_volume_mm3"] == 2.0 and c["volume_ml"] == pytest.approx(18 * 2.0 / 1000)


def test_10_world_centroid_oblique():
    th = np.deg2rad(20)
    aff = np.eye(4)
    aff[:3, :3] = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    ) @ np.diag([1.0, 1.5, 3.0])
    aff[:3, 3] = [10, -5, 2]
    m = np.zeros(SHAPE, np.uint8)
    m[4:6, 6, 3] = 1  # centroid (4.5, 6, 3)
    c = _case(m, affine=aff)["components"][0]
    assert c["centroid_ijk"] == [4.5, 6.0, 3.0]
    assert np.allclose(c["centroid_world_mm"], (aff @ [4.5, 6, 3, 1])[:3], atol=1e-3)
    assert c["voxel_volume_mm3"] == pytest.approx(4.5)


# ---------------------------------------------------------------- 11-14 intensities
def test_11_dwi_stats():
    m = np.zeros(SHAPE, np.uint8)
    m[6:8, 6, 4] = 1
    dwi = np.zeros(SHAPE)
    dwi[6, 6, 4], dwi[7, 6, 4] = 10, 30
    d = _case(m, dwi=dwi)["components"][0]["dwi"]
    assert (d["mean"], d["median"], d["min"], d["max"], d["n"]) == (20, 20, 10, 30, 2)


def test_12_adc_stats_with_background_code():
    m = np.zeros(SHAPE, np.uint8)
    m[6:9, 6, 4] = 1
    adc = np.zeros(SHAPE)
    adc[6, 6, 4], adc[7, 6, 4], adc[8, 6, 4] = 500, 900, -1
    a = _case(m, adc=adc, adc_background_code=-1)["components"][0]["adc"]
    assert (
        a["valid_voxel_count"] == 2 and a["background_code_voxel_count"] == 1 and a["mean"] == 700
    )


def test_13_adc_negative_values_kept():
    m = np.zeros(SHAPE, np.uint8)
    m[6:8, 6, 4] = 1
    adc = np.zeros(SHAPE)
    adc[6, 6, 4], adc[7, 6, 4] = -200, 600
    adc_ro = _ro(adc)
    a = _case(m, adc=adc_ro)["components"][0]["adc"]
    assert a["negative_voxel_count"] == 1 and a["min"] == -200 and a["mean"] == 200
    assert adc_ro[6, 6, 4] == -200


@pytest.mark.parametrize("unit", ["unknown", "probable_adc", None])
def test_14_adc_unknown_warning(unit):
    m = np.zeros(SHAPE, np.uint8)
    m[6, 6, 4] = 1
    r = _case(m, adc_unit_status=unit)
    assert "ADC_STORAGE_UNKNOWN" in r["components"][0]["warning_codes"]
    assert "ADC_STORAGE_UNKNOWN" in r["case"]["warnings_summary"]["case_level_codes"]
    ok = _case(m)
    assert "ADC_STORAGE_UNKNOWN" not in ok["components"][0]["warning_codes"]


# ---------------------------------------------------------------- 15-20 brain mask boundary
def test_15_component_inside_brain():
    m = np.zeros(SHAPE, np.uint8)
    m[9:11, 9:11, 4:6] = 1
    c = _case(m)["components"][0]
    b = c["brain_mask_boundary"]
    assert b["voxels_outside_brain_mask"] == 0 and not b["touches_boundary_3mm"]
    assert not {"BOUNDARY_CONTACT", "EXTENDS_OUTSIDE_BRAINMASK"} & set(c["warning_codes"])


def test_16_boundary_contact():
    m = np.zeros(SHAPE, np.uint8)
    m[2, 9, 4] = 1  # first brain voxel next to the edge of the brain mask
    c = _case(m)["components"][0]
    b = c["brain_mask_boundary"]
    assert b["min_distance_to_brain_boundary_mm"] == 1.0 and b["touches_boundary_1_inplane_voxel"]
    w = [x for x in c["warnings"] if x["code"] == "BOUNDARY_CONTACT"][0]
    assert w["provisional_threshold"] is True


def test_17_component_outside_brain():
    m = np.zeros(SHAPE, np.uint8)
    m[0:2, 9, 4] = 1  # entirely outside the brain mask
    c = _case(m)["components"][0]
    assert c["brain_mask_boundary"]["voxels_outside_brain_mask"] == 2
    assert c["brain_mask_boundary"]["outside_brain_mask_ratio"] == 1.0
    assert "EXTENDS_OUTSIDE_BRAINMASK" in c["warning_codes"]


def test_18_outside_voxels_preserved():
    m = np.zeros(SHAPE, np.uint8)
    m[0:5, 9, 4] = 1  # crosses the brain-mask boundary
    pred = _ro(m)
    r = _case(pred)
    assert r["components"][0]["voxel_count"] == 5 and r["case"]["prediction_voxels"] == 5
    assert r["components"][0]["brain_mask_boundary"]["voxels_outside_brain_mask"] == 2
    assert int(pred.sum()) == 5


def test_19_image_edge_contact():
    m = np.zeros(SHAPE, np.uint8)
    m[0, 9, 4] = 1
    assert _case(m)["components"][0]["brain_mask_boundary"]["touches_image_edge"] is True
    m2 = np.zeros(SHAPE, np.uint8)
    m2[9, 9, 4] = 1
    assert _case(m2)["components"][0]["brain_mask_boundary"]["touches_image_edge"] is False


def test_20_distance_uses_spacing():
    brain = np.zeros(SHAPE, bool)
    brain[:, :, 0:5] = True
    comp = np.zeros(SHAPE, bool)
    comp[9, 9, 2] = True  # 3 slices from the first outside slice (k=5)
    assert (
        BrainBoundary(brain, (1.0, 1.0, 2.0)).metrics(comp)["min_distance_to_brain_boundary_mm"]
        == 6.0
    )
    assert (
        BrainBoundary(brain, (1.0, 1.0, 0.5)).metrics(comp)["min_distance_to_brain_boundary_mm"]
        == 1.5
    )


# ---------------------------------------------------------------- 21-25 2D / 2.5D agreement
def _models():
    m = np.zeros(SHAPE, np.uint8)
    m[5:8, 5:8, 4] = 1  # component 1 (9 voxels): both
    m[12, 12, 6] = 1  # component 2: 2D only
    m[15, 4, 2] = 1  # component 3: 2.5D only
    p2 = np.zeros(SHAPE, np.uint8)
    p2[5:8, 5:8, 4] = 1
    p2[12, 12, 6] = 1
    p25 = np.zeros(SHAPE, np.uint8)
    p25[5:8, 5:7, 4] = 1
    p25[15, 4, 2] = 1
    prob2 = p2 * 0.8
    prob2[5, 5, 4] = 0.6
    return m, p2, p25, prob2, p25 * 0.7


def test_21_23_agreement_status():
    m, p2, p25, pr2, pr25 = _models()
    r = _case(m, pred_2d=p2, pred_25d=p25, prob_2d=pr2, prob_25d=pr25)
    st = [c["model_agreement"]["status"] for c in r["components"]]
    assert st == ["both", "25d_only", "2d_only"]  # equal volume: smaller k first
    reasons = [
        [w["detail"].get("reason") for w in c["warnings"] if w["code"] == "MODEL_DISAGREEMENT"]
        for c in r["components"]
    ]
    assert reasons == [[], ["25d_only"], ["2d_only"]]


def test_24_dice_iou():
    m, p2, p25, pr2, pr25 = _models()
    a = _case(m, pred_2d=p2, pred_25d=p25)["components"][0]["model_agreement"]
    assert a["dice_2d_25d"] == pytest.approx(2 * 6 / (9 + 6), abs=1e-6)
    assert a["iou_2d_25d"] == pytest.approx(6 / 9, abs=1e-6)
    assert a["overlap_2d_ratio"] == 1.0 and a["overlap_25d_ratio"] == pytest.approx(6 / 9, abs=1e-6)


def test_25_probability_mean_max():
    m, p2, p25, pr2, pr25 = _models()
    a = _case(m, pred_2d=p2, pred_25d=p25, prob_2d=pr2, prob_25d=pr25)["components"][0][
        "model_agreement"
    ]
    assert a["conf_2d_max"] == pytest.approx(0.8) and a["conf_2d_mean"] == pytest.approx(
        (8 * 0.8 + 0.6) / 9
    )
    assert a["conf_25d_max"] == pytest.approx(0.7) and a["conf_25d_mean"] == pytest.approx(
        6 * 0.7 / 9
    )


def test_agreement_unavailable_without_both_models():
    m, p2, *_ = _models()
    a = _case(m, pred_2d=p2)["components"][0]["model_agreement"]
    assert a["status"] == "neither_or_unavailable" and a["available"] is False


# ---------------------------------------------------------------- 26-27 provisional warnings
def test_26_small_lesion_provisional():
    c = _case(_blobs())["components"][0]
    w = [x for x in c["warnings"] if x["code"] == "SMALL_LESION"][0]
    assert w["provisional_threshold"] and w["threshold_source"] == "design_candidate_not_validated"


def test_27_thick_slice_provisional():
    m = np.zeros(SHAPE, np.uint8)
    m[9, 9, 4] = 1
    thin = _case(m)  # 2 mm
    assert "THICK_SLICE" not in thin["components"][0]["warning_codes"]
    thick = _case(m, affine=np.diag([0.9, 0.9, 6.0, 1.0]))
    assert "THICK_SLICE" in thick["components"][0]["warning_codes"]
    assert thick["case"]["effective_z_spacing_mm"] == 6.0
    by_thickness = _case(m, slice_thickness_mm=5.0, spacing_between_slices_mm=2.0)
    assert "THICK_SLICE" in by_thickness["case"]["warnings_summary"]["case_level_codes"]


# ---------------------------------------------------------------- 28-36 GT, privacy, invariance
def test_28_runs_without_gt():
    r = _case(_blobs())
    assert r["case"]["gt_used"] is False and "development_only_gt" not in r["components"][0]


def test_29_gt_does_not_change_prediction():
    pred = _ro(_blobs())
    gt = np.zeros(SHAPE, np.uint8)
    gt[5, 5, 3] = 1
    a, b = _case(pred), _case(pred, gt=gt)

    def strip(r):
        return [{k: v for k, v in c.items() if k != "development_only_gt"} for c in r["components"]]

    assert strip(a) == strip(b) and b["components"][0]["development_only_gt"]["matches_gt"] is True


def test_30_alias_only_output():
    r = _case(_blobs())
    text = json.dumps(r)
    assert (
        r["case"]["alias"] == "synth-01" and "sub-" not in text and "strokecase" not in text.lower()
    )


@pytest.mark.parametrize("alias", ["sub-strokecase9991", "sub-zz123", "Synth 01", "../x"])
def test_31_original_like_alias_refused(alias):
    with pytest.raises(ComponentAnalysisError):
        _case(_blobs(), alias=alias)


def test_31b_forbidden_id_in_metadata_refused():
    with pytest.raises(ComponentAnalysisError):
        _case(_blobs(), brain_mask_policy_id="from-sub-zz123")


def test_32_no_overwrite(tmp_path):
    r = _case(_blobs())
    write_record(tmp_path, r)
    with pytest.raises(ComponentAnalysisError):
        write_record(tmp_path, r)


def test_33_output_root_git_ignored():
    assert is_git_ignored(REPO / "results/component_analysis/synth-01/component_record.json", REPO)


def test_33b_cli_refuses_output_outside_root(tmp_path):
    from scripts.analyze_prediction_components import main

    argv = [
        "--alias",
        "synth-01",
        "--prediction",
        "p",
        "--probability",
        "p",
        "--dwi",
        "p",
        "--adc",
        "p",
        "--brain-mask",
        "p",
        "--brain-mask-policy",
        "x",
        "--adc-storage-status",
        "P",
        "--out",
        str(tmp_path),
    ]
    assert main(argv) == 2 and not any(tmp_path.iterdir())


def test_34_35_no_v1_code_and_no_model_imported():
    prefixes = (
        "neuroscan_ai.inference",
        "neuroscan_ai.training",
        "neuroscan_ai.preprocessing",
        "neuroscan_ai.analysis",
        "neuroscan_ai.final_test",
        "torch",
    )
    code = (
        "import sys; sys.path.insert(0, 'ai'); import neuroscan_ai.component_analysis\n"
        f"print([m for m in sys.modules if m.startswith({prefixes!r})])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
    src = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (REPO / "ai/neuroscan_ai/component_analysis").glob("*.py")
    )
    for forbidden in (
        "neuroscan_ai.inference",
        "neuroscan_ai.training",
        "neuroscan_ai.preprocessing",
        "import torch",
        "load_state_dict",
    ):
        assert forbidden not in src


def test_36_prediction_voxels_unchanged():
    raw = _blobs()
    pred, prob = _ro(raw), _ro(raw * 0.9)
    before = (pred.copy(), prob.copy())
    r = _case(pred, probability=prob)
    assert np.array_equal(pred, before[0]) and np.array_equal(prob, before[1])
    assert r["case"]["prediction_voxels"] == int(raw.sum())


def test_rejects_non_binary_and_bad_probability():
    m = _blobs().astype(float)
    m[0, 0, 0] = 0.5
    with pytest.raises(ComponentAnalysisError):
        _case(_blobs(), prediction=m)
    with pytest.raises(ComponentAnalysisError):
        _case(_blobs(), probability=np.full(SHAPE, 1.5))
