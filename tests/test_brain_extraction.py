"""Brain-extraction framework — synthetic NIfTI only; no external tool, no MRI, no weights."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroscan_ai.brain_extraction import figures, qa, selection
from neuroscan_ai.brain_extraction.audit import find_identifiers, guard_public
from neuroscan_ai.brain_extraction.interfaces import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_REVIEW,
    BrainExtractionError,
    RunRequest,
    validate_alias,
)
from neuroscan_ai.brain_extraction.runners import (
    MASK_NAME,
    ExternalExecutionDisabled,
    HDBetRunner,
    HDBetSpec,
    MockHDBetRunner,
    MockSynthStripRunner,
    SynthStripRunner,
    SynthStripSpec,
    V1BackgroundRunner,
    disabled_executor,
    from_orientation,
    to_orientation,
)

REPO = Path(__file__).resolve().parents[1]
SHAPE = (40, 44, 20)
OBLIQUE = np.array(
    [[-0.93, 0.08, 0.0, 120.0], [0.07, 0.92, 1.0, -110.0], [0.0, -0.17, 5.9, -60.0], [0, 0, 0, 1]]
)


def _ellipsoid(shape=SHAPE, r=(0.7, 0.7, 0.75)):
    g = np.indices(shape).astype(float)
    c = [(s - 1) / 2 for s in shape]
    d = sum(((g[i] - c[i]) / (r[i] * s / 2)) ** 2 for i, s in enumerate(shape))
    return d <= 1.0


BRAIN = _ellipsoid()
HEAD = _ellipsoid(r=(0.9, 0.9, 0.95))
GT = np.zeros(SHAPE, bool)
GT[18:21, 20:23, 9:11] = True  # component 1
GT[25, 26, 12] = True  # component 2 (single voxel)


def _trace(path: Path, affine=OBLIQUE, codes=(2, 2)) -> Path:
    data = np.where(BRAIN, 600, np.where(HEAD, 200, 0)).astype(np.int16)  # brain + scalp shell
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, code=codes[0])
    img.set_sform(affine, code=codes[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def _mask_img(data, affine=OBLIQUE, dtype=np.uint8, codes=(2, 2)):
    img = nib.Nifti1Image(np.asarray(data).astype(dtype), affine)
    img.set_qform(affine, code=codes[0])
    img.set_sform(affine, code=codes[1])
    return img


@pytest.fixture
def case(tmp_path):
    t = _trace(tmp_path / "in/trace.nii.gz")
    g = tmp_path / "in/gt.nii.gz"
    nib.save(_mask_img(GT), str(g))
    return t, g, tmp_path / "out"


def _req(case, tool="synthstrip", alias="soop-pilot-01"):
    t, g, out = case
    return RunRequest(alias, t, g, tool, "test", {}, out)


def _trace_img():
    return nib.Nifti1Image(np.zeros(SHAPE, np.int16), OBLIQUE)


# 1-5 values ------------------------------------------------------------------------------------
def test_normal_binary_mask_passes(case):
    res = MockSynthStripRunner(lambda img: BRAIN).run(_req(case))
    assert res.status == STATUS_PASS and res.qa["values"]["binary"]
    out = nib.load(str(res.mask_path))
    assert out.get_data_dtype() == np.uint8 and set(np.unique(out.dataobj)) == {0, 1}


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (np.zeros(SHAPE), "empty_mask"),
        (np.ones(SHAPE), "all_one_mask"),
        (np.where(BRAIN, np.nan, 0.0), "non_finite"),
        (np.where(BRAIN, np.inf, 0.0), "non_finite"),
        (np.where(BRAIN, 2.0, 0.0), "not_binary"),
        (np.where(BRAIN, 0.5, 0.0), "not_binary"),
    ],
)
def test_invalid_mask_values_fail(data, reason):
    v = qa.check_values(data)
    assert reason in v["failures"]
    q = qa.run_qa(_trace_img(), _mask_img(data, dtype=np.float32), None)
    assert q["classification"]["status"] == STATUS_FAIL


# 6-10 geometry ---------------------------------------------------------------------------------
def test_shape_mismatch_fails():
    g = qa.check_geometry(_trace_img(), _mask_img(np.zeros((40, 44, 19))))
    assert "shape_equal" in g["failures"]


def test_affine_mismatch_fails():
    a = OBLIQUE.copy()
    a[0, 3] += 0.01
    assert "affine_equal" in qa.check_geometry(_trace_img(), _mask_img(BRAIN, a))["failures"]


def test_orientation_mismatch_fails():
    flipped = OBLIQUE.copy()
    flipped[:, 0] *= -1
    g = qa.check_geometry(_trace_img(), _mask_img(BRAIN, flipped))
    assert "orientation_equal" in g["failures"]


def test_geometry_mismatch_is_never_resampled(case):
    a = OBLIQUE.copy()
    a[1, 3] += 2.0
    res = MockSynthStripRunner(lambda img: _mask_img(BRAIN, a)).run(_req(case))
    assert res.status == STATUS_FAIL and res.mask_path is None
    assert not (case[2] / "soop-pilot-01/synthstrip" / MASK_NAME).exists()
    assert "mask_resampling" in res.audit["not_performed"]


def test_oblique_affine_preserved(case):
    res = MockSynthStripRunner(lambda img: BRAIN).run(_req(case))
    out = nib.load(str(res.mask_path))
    assert np.array_equal(out.affine, nib.load(str(case[0])).affine)
    assert res.qa["geometry"]["max_affine_absdiff"] == 0.0


def test_qform_sform_codes_checked(tmp_path):
    g = qa.check_geometry(_trace_img(), _mask_img(BRAIN, codes=(1, 1)))
    ref = nib.Nifti1Image(np.zeros(SHAPE, np.int16), OBLIQUE)
    ref.set_qform(OBLIQUE, code=2)
    ref.set_sform(OBLIQUE, code=2)
    g = qa.check_geometry(ref, _mask_img(BRAIN, codes=(1, 1)))
    assert g["qform_code"] == [2, 1] and "codes_equal" in g["review"] and not g["failures"]
    q = qa.run_qa(ref, _mask_img(BRAIN, codes=(1, 1)), None)
    assert q["classification"]["status"] == STATUS_REVIEW
    assert q["classification"]["review_reasons"] == ["header_codes_differ"]
    g2 = qa.check_geometry(ref, _mask_img(BRAIN, codes=(2, 2)))
    assert not g2["failures"] and not g2["review"]


# 11-13 GT preservation -------------------------------------------------------------------------
def test_gt_fully_inside_passes():
    r = qa.gt_preservation(BRAIN, GT)
    assert r["gt_outside_voxels"] == 0 and r["gt_preservation_failure"] is False


def test_one_gt_voxel_outside_fails(case):
    def cut(img):
        m = BRAIN.copy()
        m[25, 26, 12] = False  # removes exactly the single-voxel GT component
        return m

    res = MockSynthStripRunner(cut).run(_req(case))
    lesion = res.qa["lesion"]
    assert lesion["gt_outside_voxels"] == 1 and lesion["gt_preservation_failure"] is True
    assert res.status == STATUS_FAIL
    assert "gt_preservation_failure" in res.qa["classification"]["fail_reasons"]
    # the stored mask is the tool mask unchanged (no automatic dilation / repair)
    assert int(np.asarray(nib.load(str(res.mask_path)).dataobj)[25, 26, 12]) == 0


def test_gt_component_counts():
    m = BRAIN.copy()
    m[18, 20, 9] = False
    m[25, 26, 12] = False
    r = qa.gt_preservation(m, GT)
    assert r["gt_components"] == 2 and r["gt_components_with_outside"] == 2
    assert [c["outside"] for c in r["per_component"]] == [1, 1]
    assert r["gt_outside_fraction"] == pytest.approx(2 / GT.sum(), abs=1e-6)


# 14-17 structure -------------------------------------------------------------------------------
def test_26_connectivity_component_count():
    m = np.zeros((6, 6, 6), bool)
    m[1, 1, 1] = m[2, 2, 2] = True  # corner-touching: one component with 26-connectivity
    m[4, 4, 4] = True
    assert qa.components(m)[1] == 2


def test_largest_component_fraction():
    m = np.zeros((10, 10, 10), bool)
    m[1:4, 1:4, 1:4] = True  # 27
    m[7:9, 7:8, 7:8] = True  # 2  (not connected)
    s = qa.structure(m, np.eye(4))
    assert s["n_components_26"] == 2 and s["largest_component_fraction"] == pytest.approx(27 / 29)


def test_boundary_contact_fraction():
    m = np.zeros((5, 5, 5), bool)
    m[0, 2, 2] = m[2, 2, 2] = True
    assert qa.boundary_contact(m) == 0.5


def test_bounding_box_and_slice_ranges():
    m = np.zeros((10, 12, 8), bool)
    m[2:5, 3:9, 1:4] = True
    b = qa.bounding_box(m)
    assert b["bbox_min"] == [2, 3, 1] and b["bbox_max"] == [4, 8, 3]
    assert b["slices_with_mask"] == [3, 6, 3]


def test_baseline_comparison_and_review_thresholds(case):
    base = HEAD
    res = MockSynthStripRunner(lambda img: BRAIN).run(
        _req(case), baseline=base, thresholds=qa.ReviewThresholds(min_volume_ratio_vs_baseline=0.9)
    )
    b = res.qa["baseline"]
    assert b["baseline_available"] and 0 < b["volume_ratio_vs_baseline"] < 0.9
    assert res.status == STATUS_REVIEW
    assert res.qa["classification"]["review_reasons"] == ["volume_low_vs_baseline"]


def test_review_rules_are_not_evaluated_without_thresholds(case):
    res = MockSynthStripRunner(lambda img: BRAIN).run(_req(case))
    c = res.qa["classification"]
    assert c["status"] == STATUS_PASS and "fragmented" in c["review_rules_not_evaluated"]


# 18-19 aliases / IDs ---------------------------------------------------------------------------
def test_alias_only_in_public_outputs(case):
    res = MockSynthStripRunner(lambda img: BRAIN).run(_req(case))
    folder = case[2] / "soop-pilot-01" / "synthstrip"
    assert sorted(p.name for p in folder.iterdir()) == [
        "audit.json", MASK_NAME, "figure_info.json", "qa.json", "run.log", "tool_input.nii.gz",
        "tool_mask.nii.gz",
    ]  # fmt: skip
    for name in ("audit.json", "qa.json", "figure_info.json", "run.log"):
        text = (folder / name).read_text(encoding="utf-8")
        assert not find_identifiers(text) and str(case[0].parent) not in text
    assert res.alias == "soop-pilot-01"


@pytest.mark.parametrize(
    "bad", ["sub-zz123", "sub-strokecase9991", "case_01", "Soop-1", "", "x sub-zz9"]
)
def test_original_ids_rejected_as_alias(bad):
    with pytest.raises(BrainExtractionError):
        validate_alias(bad)


def test_original_id_in_output_is_refused(case):
    with pytest.raises(BrainExtractionError, match="identifier"):
        guard_public({"note": "sub-strokecase9992"})
    with pytest.raises(BrainExtractionError, match="identifier"):
        guard_public({"x": "case-xyz"}, forbidden=("case-xyz",))
    res_dir = case[2] / "soop-pilot-01" / "synthstrip"
    with pytest.raises(BrainExtractionError, match="identifier"):
        MockSynthStripRunner(lambda img: BRAIN).run(_req(case), forbidden_ids=("soop-pilot-01",))
    assert not res_dir.exists() and not res_dir.with_name("synthstrip.part").exists()


# 20-23 runners ---------------------------------------------------------------------------------
def test_mock_synthstrip_runner_records_spec(case):
    res = MockSynthStripRunner(lambda img: BRAIN).run(_req(case))
    a = res.audit
    assert a["mock_runner"] is True and a["external_tool_executed"] is False
    assert a["command"][:2] == ["mri_synthstrip", "-i"] and "-b" in a["command"]
    assert "--no-csf" not in a["command"] and "-g" not in a["command"]
    assert a["spec"]["version"] == "1.8" and a["spec"]["container_digest"] is None
    assert {"container_digest", "model_sha256", "device"} <= set(a["spec"])


def test_mock_hdbet_runner_reorients_losslessly(case):
    seen = {}

    def fn(img):
        seen["orientation"] = "".join(nib.aff2axcodes(img.affine))
        seen["shape"] = img.shape
        tool_in = nib.Nifti1Image(BRAIN.astype(np.uint8), OBLIQUE)
        tool_in.set_qform(OBLIQUE, code=2)
        return to_orientation(tool_in, "LAS")[0]

    res = MockHDBetRunner(fn).run(_req(case, tool="hd_bet"))
    assert seen["orientation"] == "LAS"
    assert res.status == STATUS_PASS
    assert np.array_equal(np.asarray(nib.load(str(res.mask_path)).dataobj) == 1, BRAIN)
    a = res.audit
    assert [t["to"] for t in a["orientation_transforms"]] == [
        "LAS",
        "".join(nib.aff2axcodes(OBLIQUE)),
    ]
    assert "--save_bet_mask" in a["command"] and "--disable_tta" not in a["command"]
    assert {"weights_sha256", "tta", "device"} <= set(a["spec"])


def test_orientation_roundtrip_is_exact():
    img = nib.Nifti1Image(np.arange(np.prod(SHAPE)).reshape(SHAPE).astype(np.int32), OBLIQUE)
    std, _ = to_orientation(img, "LAS")
    back, _ = from_orientation(std, img)
    assert np.array_equal(np.asarray(back.dataobj), np.asarray(img.dataobj))
    assert np.allclose(back.affine, img.affine, atol=1e-9)


def test_v1_background_runner_interface(case):
    res = V1BackgroundRunner().run(_req(case, tool="v1_background"))
    m = np.asarray(nib.load(str(res.mask_path)).dataobj) == 1
    assert np.array_equal(m, HEAD)  # background rule keeps the synthetic scalp shell
    assert res.audit["spec"]["policy"] == "skull_stripped_background"
    assert res.audit["command"] is None and res.audit["external_tool_executed"] is False


@pytest.mark.parametrize("runner_cls", [SynthStripRunner, HDBetRunner])
def test_real_runners_refuse_execution(case, runner_cls):
    spec = SynthStripSpec() if runner_cls is SynthStripRunner else HDBetSpec()
    tool = "synthstrip" if runner_cls is SynthStripRunner else "hd_bet"
    res = runner_cls(spec).run(_req(case, tool=tool))
    assert res.status == STATUS_FAIL and res.audit["execution_error"] == "ExternalExecutionDisabled"
    assert res.mask_path is None
    with pytest.raises(ExternalExecutionDisabled):
        disabled_executor(["x"], Path("."))


def test_execution_failure_is_recorded(case):
    def boom(img):
        raise RuntimeError("tool crashed")

    res = MockSynthStripRunner(boom).run(_req(case))
    assert res.status == STATUS_FAIL
    assert res.qa["classification"]["fail_reasons"] == ["execution_failure"]
    assert "tool step failed: RuntimeError" in res.log


# 24-25 outputs ---------------------------------------------------------------------------------
def test_existing_output_is_not_overwritten(case):
    MockSynthStripRunner(lambda img: BRAIN).run(_req(case))
    with pytest.raises(BrainExtractionError, match="overwrite"):
        MockSynthStripRunner(lambda img: BRAIN).run(_req(case))


@pytest.mark.parametrize(
    "path",
    [
        "results/brain_extraction_pilot/soop-pilot-01/synthstrip/qa.json",
        "data/interim/soop_brain_masks/soop-pilot-01/brain_mask.nii.gz",
        "data/interim/isles_brain_masks/isles-bet-01/brain_mask.nii.gz",
    ],
)
def test_output_paths_are_git_ignored(path):
    try:
        r = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO)
    except OSError:
        pytest.skip("git not available")
    assert r.returncode == 0


# 26-28 no external calls / MRI / weights -------------------------------------------------------
PKG = REPO / "ai/neuroscan_ai/brain_extraction"
FORBIDDEN_MODULES = ("subprocess", "urllib", "requests", "http", "socket", "docker", "torch",
                     "HD_BET", "surfa", "multiprocessing")  # fmt: skip


@pytest.mark.parametrize(
    "module", sorted(p.name for p in PKG.glob("*.py") if p.name != "executors.py")
)
def test_package_imports_no_process_network_or_model(module):
    tree = ast.parse((PKG / module).read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m.split(".")[0] in FORBIDDEN_MODULES]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in
             ("system", "popen", "Popen", "urlopen", "startfile")]  # fmt: skip
    assert not calls


def test_no_external_call_network_or_foreign_file_access(case, monkeypatch):
    import os
    import urllib.request

    def trap(*a, **k):
        raise AssertionError("forbidden call")

    for mod, name in ((subprocess, "run"), (subprocess, "Popen"), (os, "system"),
                      (urllib.request, "urlopen")):  # fmt: skip
        monkeypatch.setattr(mod, name, trap)
    opened = []
    real_load = nib.load

    def spy(path, *a, **k):
        opened.append(Path(path).resolve())
        return real_load(path, *a, **k)

    monkeypatch.setattr(nib, "load", spy)
    MockSynthStripRunner(lambda img: BRAIN).run(_req(case))
    MockHDBetRunner(lambda img: np.asarray(img.dataobj) > 0).run(_req(case, tool="hd_bet"))
    root = case[0].parent.parent.resolve()
    assert opened and all(root in p.parents for p in opened)  # synthetic tmp files only
    assert not list(Path.home().glob("hd-bet_params"))  # no weight folder created by this run


# figures ---------------------------------------------------------------------------------------
def test_case_figure_layout(tmp_path):
    trace = np.where(BRAIN, 600, np.where(HEAD, 200, 0)).astype(float)
    masks = {"v1_background": HEAD, "synthstrip": BRAIN, "hd_bet": BRAIN}
    cats = ["small_lesion", "thick_slice"]
    fig, info = figures.case_figure(trace, GT, masks, "isles-bet-01", cats, primary="synthstrip")
    problems = figures.save_case_figure(fig, tmp_path / "f.png")
    assert problems == [] and (tmp_path / "f.png").is_file()
    assert info["gt_max_axial"] == int(np.argmax(GT.sum(axis=(0, 1))))
    with pytest.raises(BrainExtractionError):
        figures.save_case_figure(figures.case_figure(trace, GT, masks, "isles-bet-01")[0],
                                 tmp_path / "f.png")  # fmt: skip


def test_figure_colours_and_alias_rules():
    assert figures.STYLE["v1_background"]["linestyle"] == "--"
    assert figures.STYLE["synthstrip"]["color"] == "#0072B2"
    assert figures.STYLE["hd_bet"]["color"] == "#E69F00"
    with pytest.raises(BrainExtractionError):
        figures.case_figure(np.zeros(SHAPE), None, {"synthstrip": BRAIN}, "sub-zz001")


def test_layout_check_detects_title_overlap():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 4))
    a1 = fig.add_axes((0.1, 0.1, 0.8, 0.4))
    a2 = fig.add_axes((0.1, 0.52, 0.8, 0.4))
    a1.set_title("a long title that sits on the axes above", fontsize=14)
    a2.set_title("top")
    assert any("overlaps axes" in p for p in figures.check_layout(fig))
    plt.close(fig)


# selection -------------------------------------------------------------------------------------
def _rows(n=30, split="train"):
    rng = np.random.default_rng(3)
    return [
        {"case_id": f"case{i:03d}", "split": split,
         "lesion_volume_ml": float(rng.choice([0.4, 5.0, 70.0])),
         "slice_spacing_mm": float(rng.choice([2.0, 5.0])),
         "obliquity_deg": float(rng.choice([0.5, 12.0]))}
        for i in range(n)
    ]  # fmt: skip


def test_isles_selection_is_deterministic_and_bounded():
    a = selection.select_isles_cases(_rows())
    b = selection.select_isles_cases(list(reversed(_rows())))
    assert a == b
    cases = a["public_summary"]["cases"]
    assert 1 <= len(cases) <= 6 and [c["alias"] for c in cases][0] == "isles-bet-01"
    assert all(v >= 2 for v in a["public_summary"]["category_coverage"].values())
    assert "case" not in json.dumps(a["public_summary"]["cases"]).replace("cases", "")


def test_isles_selection_refuses_non_train():
    rows = _rows(5) + [{**_rows(1)[0], "case_id": "v1", "split": "validation"}]
    with pytest.raises(BrainExtractionError, match="Train"):
        selection.select_isles_cases(rows)
    with pytest.raises(BrainExtractionError):
        selection.select_isles_cases(_rows(3, split="test"))


def test_isles_category_rules():
    c = selection.IslesCriteria()
    row = {"lesion_volume_ml": 0.5, "slice_spacing_mm": 5.0, "obliquity_deg": 10.0}
    assert selection.categories(row, c) == {"small_lesion", "thick_slice", "high_obliquity"}
    assert selection.categories({**row, "lesion_volume_ml": 0.0}, c) == {
        "thick_slice",
        "high_obliquity",
    }
    assert selection.categories({"lesion_volume_ml": 50.0, "slice_spacing_mm": 3.0,
                                 "obliquity_deg": 9.9}, c) == {"large_lesion"}  # fmt: skip


def test_soop_pilot_plan_has_aliases_only():
    plan = selection.soop_pilot_plan()
    assert [p["alias"] for p in plan] == [f"soop-pilot-{i:02d}" for i in range(1, 6)]
    assert all(set(p) == {"alias", "dataset", "use"} for p in plan)


def test_tool_selection_gates():
    recs = [
        {"tool": "synthstrip", "alias": "a", "status": "pass", "visual_qa": "approved"},
        {"tool": "synthstrip", "alias": "b", "status": "review_required", "visual_qa": "approved"},
        {"tool": "hd_bet", "alias": "a", "status": "fail", "gt_preservation_failure": True,
         "visual_qa": "approved"},
        {"tool": "hd_bet", "alias": "b", "status": "pass", "visual_qa": "approved"},
    ]  # fmt: skip
    s = selection.summarize_tools(recs)
    assert s["eligible"] == ["synthstrip"] and s["per_tool"]["hd_bet"]["gt_failures"] == 1
    none = selection.summarize_tools([{**recs[2]}])
    assert none["eligible"] == [] and "not fixed" in none["decision"]
    missing_visual = selection.summarize_tools([{**recs[0], "visual_qa": None}])
    assert missing_visual["eligible"] == []
