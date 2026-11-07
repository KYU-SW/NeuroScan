"""SOOP format adapter (K3) and sidecar aggregation (K4) — synthetic NIfTI / JSON only."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroscan_ai.datasets import soop
from neuroscan_ai.datasets.soop import (
    SoopCase,
    SoopFormatError,
    adc_content_status,
    adc_storage_group,
    build_case,
    discover_soop,
    sha256_file,
    singleton_plan,
    subject_paths,
    write_format_copy,
)
from neuroscan_ai.datasets.soop_metadata import aggregate, eligible_subjects, parse_bval

REPO = Path(__file__).resolve().parents[1]
SID = "sub-9876"  # synthetic subject id; must never reach an output file
OBLIQUE = np.array(
    [[-0.93, 0.05, 0.0, 120.0], [0.04, 0.92, 0.9, -110.0], [0.0, -0.17, 5.95, -60.0], [0, 0, 0, 1]]
)
_ADC_TYPE = ["ORIGINAL", "PRIMARY", "ADC", "UNSPECIFIED", "ADC", "UNSPECIFIED"]
SIDECAR = {
    "P": {
        "SeriesDescription": "ddADC",
        "ProtocolName": "ddADC",
        "ImageType": _ADC_TYPE,
        "PhilipsRescaleSlope": 0.000966995,
        "PhilipsRescaleIntercept": 0,
        "PhilipsScaleSlope": 0.12,
        "PhilipsRWVSlope": 9.66995e-07,
        "PhilipsRWVIntercept": 0,
        "SoftwareVersions": "5.6.1\\5.6.1.0",
        "UsePhilipsFloatNotDisplayScaling": 1,
    },
    "S": {
        "SeriesDescription": "dADC",
        "ProtocolName": "dADC",
        "ImageType": ["ORIGINAL", "PRIMARY", "ADC", "ADC", "UNSPECIFIED"],
        "PhilipsRescaleSlope": 0.966995,
        "PhilipsRescaleIntercept": 0,
        "PhilipsScaleSlope": 0.0043,
        "PhilipsRWVSlope": 0.966995,
        "PhilipsRWVIntercept": 0,
        "SoftwareVersions": "3.2.3",
        "UsePhilipsFloatNotDisplayScaling": 1,
    },
    "N": {
        "SeriesDescription": "dADC",
        "ProtocolName": "dADC",
        "ImageType": _ADC_TYPE,
        "PhilipsRescaleSlope": 1.27773e-06,
        "PhilipsRescaleIntercept": -0.00193634,
        "PhilipsScaleSlope": 0.13,
        "PhilipsRWVSlope": 1.27773e-06,
        "PhilipsRWVIntercept": -0.00193634,
        "SoftwareVersions": "5.6.1\\5.6.1.0",
        "UsePhilipsFloatNotDisplayScaling": 1,
    },
}


def _save(path: Path, data: np.ndarray, affine=OBLIQUE, scaling=None, codes=(2, 2)) -> Path:
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(data.dtype)
    img = nib.Nifti1Image(data, affine, hdr)
    img.set_qform(affine, code=codes[0])
    img.set_sform(affine, code=codes[1])
    img.header.set_xyzt_units("mm", "sec")
    if scaling is not None:
        img.header.set_slope_inter(*scaling)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def _volume(shape=(12, 10, 6), seed=0, dtype=np.int16):
    rng = np.random.default_rng(seed)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=shape, dtype=dtype)


def _mask(shape=(12, 10, 6)):
    m = np.zeros(shape, np.uint8)
    m[3:6, 3:6, 2:4] = 1
    return m


def make_subject(
    root: Path,
    sid=SID,
    *,
    group="P",
    four_d=True,
    with_acute=True,
    extra_masks=("lesion",),
    adc_affine=OBLIQUE,
    label=None,
) -> Path:
    p = subject_paths(sid)
    shape = (12, 10, 6)
    vshape = (*shape, 1) if four_d else shape
    _save(root / p["trace_nii"], np.abs(_volume(vshape, 1)) % 4096)
    _save(
        root / p["adc_nii"],
        _volume(vshape, 2),
        affine=adc_affine,
        scaling=(0.5208160877227783, 17066.1015625),
    )
    for k, js in (("trace_json", {"SeriesDescription": "sB1000"}), ("adc_json", SIDECAR[group])):
        (root / p[k]).write_text(json.dumps(js), "utf-8")
    m = _mask(shape) if label is None else label
    if with_acute:
        _save(root / p["mask_acute"], m)
    for k in extra_masks:
        _save(root / p[f"mask_{k}"], m)
    return root


def _case(root: Path) -> SoopCase:
    cases, _ = discover_soop(root)
    return cases[0]


# 1-3 discovery ---------------------------------------------------------------------------------
def test_discovers_required_bids_files(tmp_path):
    make_subject(tmp_path)
    cases, excluded = discover_soop(tmp_path)
    assert len(cases) == 1 and not excluded
    assert set(soop.REQUIRED) <= set(cases[0].files)


def test_acute_mask_is_the_primary_label(tmp_path):
    make_subject(tmp_path, extra_masks=("lesion", "chronic"))
    out = build_case(_case(tmp_path), "soop-t-01", tmp_path / "out")
    src = sha256_file(tmp_path / subject_paths(SID)["mask_acute"])
    assert out["audit"]["derived_sha256"]["label"] == src  # 3D mask: byte copy of desc-lesionAcute
    assert out["metadata"]["primary_ground_truth"] == "desc-lesionAcute"
    assert out["metadata"]["optional_masks_present"] == ["chronic", "lesion"]


def test_missing_acute_mask_is_excluded_not_negative(tmp_path):
    make_subject(tmp_path, with_acute=False)
    cases, excluded = discover_soop(tmp_path)
    assert cases == [] and excluded == {SID: ["mask_acute"]}


# 4-7 singleton axis ----------------------------------------------------------------------------
def test_3d_input_passes_through_as_byte_copy(tmp_path):
    src = _save(tmp_path / "a.nii.gz", _volume())
    res = write_format_copy(src, tmp_path / "b.nii.gz")
    assert res["plan"] == "keep" and res["method"] == "byte-copy"
    assert sha256_file(src) == sha256_file(tmp_path / "b.nii.gz")


def test_4d_trailing_singleton_is_removed(tmp_path):
    src = _save(tmp_path / "a.nii.gz", _volume((12, 10, 6, 1)))
    res = write_format_copy(src, tmp_path / "b.nii.gz")
    out = nib.load(str(tmp_path / "b.nii.gz"))
    assert res["plan"] == "squeeze_last" and out.shape == (12, 10, 6)
    assert int(out.header["dim"][0]) == 3


def test_last_axis_larger_than_one_is_refused(tmp_path):
    src = _save(tmp_path / "a.nii.gz", _volume((12, 10, 6, 2)))
    with pytest.raises(SoopFormatError, match="last axis"):
        write_format_copy(src, tmp_path / "b.nii.gz")
    assert not (tmp_path / "b.nii.gz").exists()


@pytest.mark.parametrize("shape", [(12, 1, 6, 1), (12, 10, 1, 1), (12, 10, 1)])
def test_spatial_singleton_axis_is_refused(tmp_path, shape):
    src = _save(tmp_path / "a.nii.gz", _volume(shape))
    with pytest.raises(SoopFormatError, match="spatial axis"):
        write_format_copy(src, tmp_path / "b.nii.gz")


def test_5d_is_refused(tmp_path):
    src = _save(tmp_path / "a.nii.gz", _volume((12, 10, 6, 1, 1)))
    with pytest.raises(SoopFormatError):
        singleton_plan(nib.load(str(src)))


# 8-12 geometry ---------------------------------------------------------------------------------
def test_affine_mismatch_is_refused(tmp_path):
    shifted = OBLIQUE.copy()
    shifted[0, 3] += 0.5
    make_subject(tmp_path, adc_affine=shifted)
    with pytest.raises(SoopFormatError, match="affine"):
        build_case(_case(tmp_path), "soop-t-01", tmp_path / "out")
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_shape_mismatch_is_refused(tmp_path):
    make_subject(tmp_path, label=np.zeros((12, 10, 5), np.uint8))
    with pytest.raises(SoopFormatError, match="shape"):
        build_case(_case(tmp_path), "soop-t-01", tmp_path / "out")


def test_spacing_mismatch_is_refused():
    a = nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), np.diag([1.0, 1.0, 6.0, 1.0]))
    b = nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), np.diag([1.0, 1.0, 5.0, 1.0]))
    with pytest.raises(SoopFormatError, match="affine|spacing"):
        soop.check_same_grid(a, b, "x")


def test_orientation_is_recorded(tmp_path):
    make_subject(tmp_path)
    g = build_case(_case(tmp_path), "soop-t-01", tmp_path / "out")["metadata"]["geometry"]
    assert g["orientation"] == "".join(nib.aff2axcodes(OBLIQUE))
    assert g["obliquity_deg"] > 1.0


def test_oblique_affine_is_preserved_exactly(tmp_path):
    src = _save(tmp_path / "a.nii.gz", _volume((12, 10, 6, 1)))
    write_format_copy(src, tmp_path / "b.nii.gz")
    s, o = nib.load(str(src)), nib.load(str(tmp_path / "b.nii.gz"))
    assert np.array_equal(s.affine, o.affine)
    assert np.array_equal(s.get_qform(), o.get_qform()) and np.array_equal(
        s.get_sform(), o.get_sform()
    )


# 13-14 binary GT -------------------------------------------------------------------------------
def test_binary_mask_is_accepted(tmp_path):
    make_subject(tmp_path)
    m = build_case(_case(tmp_path), "soop-t-01", tmp_path / "out")["metadata"]["masks"]["acute"]
    assert m["values"] == [0.0, 1.0] and m["voxels"] == 18


def test_non_binary_mask_is_refused_not_binarised(tmp_path):
    lab = _mask()
    lab[0, 0, 0] = 2
    make_subject(tmp_path, label=lab)
    with pytest.raises(SoopFormatError, match="not binary"):
        build_case(_case(tmp_path), "soop-t-01", tmp_path / "out")
    assert not (tmp_path / "out/soop-t-01").exists()


# 15-18 value / header preservation -------------------------------------------------------------
def test_dtype_raw_and_scaled_values_are_preserved(tmp_path):
    src = _save(tmp_path / "a.nii.gz", _volume((12, 10, 6, 1)), scaling=(0.52, 17066.1))
    res = write_format_copy(src, tmp_path / "b.nii.gz")
    s, o = nib.load(str(src)), nib.load(str(tmp_path / "b.nii.gz"))
    assert o.get_data_dtype() == np.int16
    assert np.array_equal(
        np.asarray(s.dataobj, np.float64)[..., 0], np.asarray(o.dataobj, np.float64)
    )
    assert np.array_equal(s.get_fdata(dtype=np.float32)[..., 0], o.get_fdata(dtype=np.float32))
    assert res["raw_equal"] and res["scaled_f64_equal"] and res["scaled_f32_equal"]


def test_scl_slope_and_inter_are_kept(tmp_path):
    src = _save(tmp_path / "a.nii.gz", _volume((12, 10, 6, 1)), scaling=(0.0604, 1979.95))
    write_format_copy(src, tmp_path / "b.nii.gz")
    s, o = nib.load(str(src)), nib.load(str(tmp_path / "b.nii.gz"))
    assert (s.dataobj.slope, s.dataobj.inter) == (o.dataobj.slope, o.dataobj.inter)


def test_naive_float_resave_would_be_detected(tmp_path):
    """A float32 re-save changes float64 analysis values: the verifier must catch it."""
    src = _save(tmp_path / "a.nii.gz", _volume((12, 10, 6, 1)), scaling=(0.52, 17066.1))
    s = nib.load(str(src))
    hdr = s.header.copy()
    hdr.set_data_shape((12, 10, 6))
    hdr.set_data_dtype(np.float32)
    bad = nib.Nifti1Image(s.get_fdata(dtype=np.float32)[..., 0], None, hdr)
    bad.header.set_slope_inter(1.0, 0.0)
    nib.save(bad, str(tmp_path / "bad.nii.gz"))
    with pytest.raises(SoopFormatError, match="not preserved"):
        soop.verify_preserved(s, nib.load(str(tmp_path / "bad.nii.gz")), "squeeze_last")


def test_qform_sform_codes_and_frame_fields_are_kept(tmp_path):
    src = _save(tmp_path / "a.nii.gz", _volume((12, 10, 6, 1)), codes=(1, 2))
    res = write_format_copy(src, tmp_path / "b.nii.gz")
    o = nib.load(str(tmp_path / "b.nii.gz"))
    assert (int(o.header["qform_code"]), int(o.header["sform_code"])) == (1, 2)
    assert res["frame_equal"] and o.header.get_xyzt_units() == ("mm", "sec")


def test_source_files_are_unchanged(tmp_path):
    make_subject(tmp_path, extra_masks=("lesion", "chronic"))
    case = _case(tmp_path)
    before = {k: sha256_file(p) for k, p in case.files.items()}
    out = build_case(case, "soop-t-01", tmp_path / "out")
    assert {k: sha256_file(p) for k, p in case.files.items()} == before
    assert out["audit"]["source_unchanged"] is True


def test_existing_output_is_never_overwritten(tmp_path):
    make_subject(tmp_path)
    build_case(_case(tmp_path), "soop-t-01", tmp_path / "out")
    with pytest.raises(SoopFormatError, match="exists"):
        build_case(_case(tmp_path), "soop-t-01", tmp_path / "out")


# 19 no subject id in outputs -------------------------------------------------------------------
def test_subject_id_never_written_to_outputs(tmp_path):
    make_subject(tmp_path)
    build_case(_case(tmp_path), "soop-t-01", tmp_path / "out")
    folder = tmp_path / "out/soop-t-01"
    assert sorted(p.name for p in folder.iterdir()) == [
        "adc.nii.gz",
        "audit.json",
        "dwi.nii.gz",
        "label.nii.gz",
        "metadata.json",
    ]
    for name in ("metadata.json", "audit.json"):
        text = (folder / name).read_text(encoding="utf-8")
        assert SID not in text and "9876" not in text and str(tmp_path) not in text


# 20-21 ADC storage group -----------------------------------------------------------------------
@pytest.mark.parametrize("group", ["P", "S", "N"])
def test_adc_storage_groups(group):
    assert adc_storage_group(SIDECAR[group])[0] == group


@pytest.mark.parametrize(
    "sidecar",
    [
        {},
        {"SeriesDescription": "ADC"},
        {**SIDECAR["N"], "PhilipsRescaleIntercept": 0},
        {**SIDECAR["P"], "PhilipsRescaleSlope": "0.000966995"},
        # the 5 full-cohort cases that matched the old P rule: same RS, RWVSlope x1000, SW 5.5.2
        {**SIDECAR["P"], "PhilipsRWVSlope": 0.000966995, "SoftwareVersions": "5.5.2\\5.5.2.0"},
        {**SIDECAR["P"], "PhilipsRWVSlope": 0.000966995},
        {**SIDECAR["P"], "SoftwareVersions": "5.5.2\\5.5.2.0"},
        {**SIDECAR["P"], "ProtocolName": "other"},
        {**SIDECAR["P"], "UsePhilipsFloatNotDisplayScaling": 0},
        {**SIDECAR["P"], "PhilipsScaleSlope": None},
        {**SIDECAR["P"], "PhilipsScaleSlope": -0.1},
        {k: v for k, v in SIDECAR["S"].items() if k != "PhilipsRWVSlope"},
        {**SIDECAR["N"], "PhilipsRWVIntercept": 0},
        {**SIDECAR["S"], "UsePhilipsFloatNotDisplayScaling": True},
    ],
)
def test_unmatched_sidecar_is_unknown(sidecar):
    assert adc_storage_group(sidecar)[0] == "unknown"


def test_adc_values_are_not_changed(tmp_path):
    make_subject(tmp_path, group="N")
    case = _case(tmp_path)
    out = build_case(case, "soop-t-01", tmp_path / "out")
    s = nib.load(str(case.files["adc_nii"]))
    o = nib.load(str(tmp_path / "out/soop-t-01/adc.nii.gz"))
    changed = int(
        (np.asarray(s.dataobj, np.float64)[..., 0] != np.asarray(o.dataobj, np.float64)).sum()
    )
    assert changed == 0
    md = out["metadata"]
    assert md["adc_storage_group"] == "N" and "NOT a physical" in md["adc_storage_note"]
    assert md["raw_dtype"] == "int16" and md["nifti_scl_slope"] == pytest.approx(0.5208160877227783)


# 22-24 forbidden operations / frozen code ------------------------------------------------------
FORBIDDEN_IMPORTS = (
    "neuroscan_ai.preprocessing",
    "neuroscan_ai.training",
    "neuroscan_ai.inference",
    "neuroscan_ai.final_test",
    "torch",
    "monai",
    "scipy",
)


@pytest.mark.parametrize("module", ["soop.py", "soop_metadata.py"])
def test_adapter_imports_no_resampling_brain_or_model_code(module):
    tree = ast.parse((REPO / "ai/neuroscan_ai/datasets" / module).read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m.startswith(FORBIDDEN_IMPORTS)]


def test_no_resampling_or_brain_extraction_is_called(tmp_path, monkeypatch):
    import neuroscan_ai.preprocessing.pipeline as pp
    import neuroscan_ai.training.brain_gating as bg

    calls = []

    def trap(*a, **k):
        calls.append(1)
        raise AssertionError("forbidden call")

    monkeypatch.setattr(pp, "resample_to_spacing", trap)
    monkeypatch.setattr(pp, "preprocess_case", trap)
    monkeypatch.setattr(bg, "compute_brain_mask", trap)
    make_subject(tmp_path)
    out = build_case(_case(tmp_path), "soop-t-01", tmp_path / "out")
    assert calls == []
    assert {"resampling", "brain_extraction", "adc_value_conversion"} <= set(
        out["audit"]["not_performed"]
    )


FROZEN = (
    "ai/neuroscan_ai/final_test",
    "ai/neuroscan_ai/preprocessing",
    "ai/neuroscan_ai/inference",
    "ai/neuroscan_ai/training",
    "configs",
    "data/splits",
)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")


def test_frozen_v1_code_and_configs_are_unmodified():
    changed = _git("diff", "--name-only", "HEAD", "--", *FROZEN).stdout.split()
    assert changed == []


# 25 output paths are git-ignored ---------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "data/interim/soop_format_v1/soop-x-01/adc.nii.gz",
        "data/interim/soop_format_v1/soop-x-01/metadata.json",
        "data/raw/soop_pilot/_internal/aliases.json",
        "data/raw/soop_meta_v1_1_2/_internal/plan.json",
        "results/soop_compatibility/v1/sidecar_summary.json",
    ],
)
def test_output_paths_are_git_ignored(path):
    _git("rev-parse", "--git-dir")
    assert subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO).returncode == 0


# K4 aggregation --------------------------------------------------------------------------------
def test_eligible_subjects_from_paths():
    paths = set(subject_paths("sub-synthA").values()) | {
        v for k, v in subject_paths("sub-synthB").items() if k != "mask_acute"
    }
    eligible, missing = eligible_subjects(paths)
    assert eligible == ["sub-synthA"] and missing == {"mask_acute": 1}


def test_aggregate_contains_counts_only():
    pairs = [
        (
            {"SliceThickness": 5, "SpacingBetweenSlices": 6, "Manufacturer": "Philips"},
            SIDECAR[g],
        )
        for g in ("P", "P", "N", "S")
    ] + [({}, {"SeriesDescription": "sB0", "ProtocolName": "sB0", "ImageType": ["M", "SE"]})]
    s = aggregate(pairs, metadata_incomplete=2)
    assert s["n_subjects"] == 5 and s["metadata_incomplete"] == 2
    assert s["adc_storage_groups"] == {"P": 2, "S": 1, "N": 1, "unknown": 1}
    assert s["adc_content_status"] == {
        "confirmed_adc": 4,
        "probable_adc": 0,
        "suspicious_non_adc": 1,
        "unknown": 0,
    }
    assert s["eligible_by_content"] == 4
    assert s["pool_policy_by_content"] == {
        "eligible_candidate": 4,
        "review_required": 0,
        "auto_excluded": 1,
    }
    assert s["trace_fields"]["SliceThickness"]["missing"] == 1
    assert s["trace_fields"]["SliceThickness"]["numeric"]["median"] == 5.0
    assert "sub-" not in json.dumps(s)


def test_parse_bval():
    r = parse_bval("0 1000 1000\n")
    assert r == {"rows": 1, "n_values": 3, "unique_b_values": [0.0, 1000.0]}


# ADC content status --------------------------------------------------------------------------
_T = ["ORIGINAL", "PRIMARY"]


@pytest.mark.parametrize(
    ("series", "protocol", "image_type", "expected"),
    [
        ("ddADC", "ddADC", [*_T, "ADC", "UNSPECIFIED"], "confirmed_adc"),
        ("dAdc", "dAdc", [*_T, "ADC", "SE"], "confirmed_adc"),
        (
            "Apparent Diffusion Coefficient (mm2/s)",
            "Apparent Diffusion Coefficient (mm2/s)",
            ["DERIVED", "PRIMARY", "ADC", "ADC"],
            "confirmed_adc",
        ),
        ("dAX DWI b1000 CLEAR", "dAX DWI b1000 CLEAR", [*_T, "ADC", "SE"], "probable_adc"),
        ("sB0", "sB0", [*_T, "M", "SE", "M", "SE"], "suspicious_non_adc"),
        (
            "DWI Scans B0 B500 B1000",
            "DWI Scans B0 B500 B1000",
            [*_T, "M", "ND"],
            "suspicious_non_adc",
        ),
        ("B1000", "B1000", [*_T, "OTHER"], "suspicious_non_adc"),
        ("AX DWI", "AX DWI SENSE", [*_T, "M", "SE"], "suspicious_non_adc"),
        # contradictions / insufficient evidence -> unknown
        ("dADC", "dADC", ["DERIVED", "PRIMARY", "DIFFUSION", "MIXED", "REAL"], "unknown"),
        ("sB0", "sB0", [*_T, "ADC"], "unknown"),
        ("ddADC", "AX DWI", [*_T, "ADC"], "unknown"),
        ("ddADC", None, [*_T, "ADC"], "unknown"),
        ("ddADC", "ddADC", None, "unknown"),
        ("T2", "T2", [*_T, "M"], "unknown"),
    ],
)
def test_adc_content_status(series, protocol, image_type, expected):
    sc = {"SeriesDescription": series, "ProtocolName": protocol, "ImageType": image_type}
    assert adc_content_status(sc)[0] == expected


def test_content_status_is_recorded_and_data_untouched(tmp_path):
    make_subject(tmp_path)
    md = build_case(_case(tmp_path), "soop-t-01", tmp_path / "out")["metadata"]
    assert md["adc_content_status"] == "confirmed_adc"
    assert md["v2_pool_status"] == "eligible_candidate" and md["is_pilot_case"] is False
    assert "NOT a physical" in md["adc_storage_note"]


def test_suspicious_content_is_not_eligible_for_pool():
    assert soop.ADC_CONTENT_ELIGIBLE == ("confirmed_adc",)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("confirmed_adc", "eligible_candidate"),
        ("probable_adc", "review_required"),
        ("suspicious_non_adc", "auto_excluded"),
        ("unknown", "auto_excluded"),
        ("anything-else", "auto_excluded"),
    ],
)
def test_v2_pool_policy(content, expected):
    assert soop.v2_pool_status(content, pilot=False) == expected


@pytest.mark.parametrize("content", list(soop.ADC_CONTENT_STATUSES))
def test_pilot_cases_are_always_excluded(content):
    assert soop.v2_pool_status(content, pilot=True) == "pilot_permanently_excluded"


def test_pilot_flag_is_recorded(tmp_path):
    make_subject(tmp_path)
    md = build_case(_case(tmp_path), "soop-t-01", tmp_path / "out", pilot=True)["metadata"]
    assert md["is_pilot_case"] is True and md["v2_pool_status"] == "pilot_permanently_excluded"
