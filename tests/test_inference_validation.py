"""Phase 5A input validation and sealed test-split guard (synthetic data only)."""

from __future__ import annotations

import gzip
import math
import os

import nibabel as nib
import numpy as np
import pytest
import yaml

from neuroscan_ai.inference import sealed_guard
from neuroscan_ai.inference.errors import Code, InferenceError
from neuroscan_ai.inference.validation import obliquity_deg, validate_inputs
from tests.conftest import REPO_ROOT, las_affine
from tests.inference_helpers import INFERENCE_CFG, synthetic_volumes, write_case, write_nifti

INF = yaml.safe_load(INFERENCE_CFG.read_text(encoding="utf-8"))
SPLIT = REPO_ROOT / "data" / "splits" / "split_v1_seed42.json"
# A sealed test-split ID used only as a path *string*; no file of this case is ever opened.
SEALED_ID = "sub-strokecase0001"
VAL_ID = "sub-strokecase0086"


def codes(issues):
    return [i.code for i in issues]


def test_valid_pair_has_no_errors_and_path_free_stats(tmp_path):
    d, a, _, _ = write_case(tmp_path)
    r = validate_inputs(d, a, INF)
    assert r.errors == []
    assert Code.W_POSSIBLE_SWAP not in codes(r.warnings)
    assert Code.W_ADC_UNIT_UNUSUAL not in codes(r.warnings)
    assert r.dwi.data.dtype == np.float32 and r.dwi.shape == (48, 48, 10)
    assert len(r.dwi.sha256) == 64
    assert str(tmp_path) not in repr(r.stats)


@pytest.mark.parametrize("missing", ["dwi", "adc"])
def test_missing_file(tmp_path, missing):
    d, a, _, _ = write_case(tmp_path)
    (d if missing == "dwi" else a).unlink()
    r = validate_inputs(d, a, INF)
    assert codes(r.errors) == [Code.INPUT_NOT_FOUND]
    assert missing.upper() in r.errors[0].message


def test_wrong_extension_and_not_nifti(tmp_path):
    d, a, _, _ = write_case(tmp_path)
    txt = tmp_path / "dwi.txt"
    txt.write_text("hello")
    assert codes(validate_inputs(txt, a, INF).errors) == [Code.INPUT_FORMAT]
    fake = tmp_path / "fake.nii.gz"
    fake.write_bytes(gzip.compress(b"not a nifti header at all" * 20))
    assert codes(validate_inputs(fake, a, INF).errors) == [Code.INPUT_FORMAT]


def test_truncated_file_is_corrupt_and_message_has_no_path(tmp_path):
    d, a, _, _ = write_case(tmp_path)
    raw = d.read_bytes()
    d.write_bytes(raw[: len(raw) // 2])
    r = validate_inputs(d, a, INF)
    assert codes(r.errors) == [Code.INPUT_CORRUPT]
    assert str(tmp_path) not in (r.errors[0].message + (r.errors[0].detail or ""))


def test_4d_inputs(tmp_path):
    dwi, adc, _, _ = synthetic_volumes()
    aff = las_affine((1.8, 1.8, 4.8))
    ok_d = write_nifti(tmp_path / "d1.nii.gz", dwi[..., None], aff)
    ok_a = write_nifti(tmp_path / "a1.nii.gz", adc[..., None], aff)
    assert validate_inputs(ok_d, ok_a, INF).errors == []  # t = 1 is squeezed
    bad = write_nifti(tmp_path / "d2.nii.gz", np.stack([dwi, dwi], -1), aff)
    assert codes(validate_inputs(bad, ok_a, INF).errors) == [Code.INPUT_NOT_3D]


def test_grid_mismatches(tmp_path):
    dwi, adc, _, _ = synthetic_volumes()
    aff = las_affine((1.8, 1.8, 4.8))
    d = write_nifti(tmp_path / "d.nii.gz", dwi, aff)
    a_shape = write_nifti(tmp_path / "a_shape.nii.gz", adc[:, :, :9], aff)
    shifted = aff.copy()
    shifted[0, 3] += 1.0
    a_shift = write_nifti(tmp_path / "a_shift.nii.gz", adc, shifted)
    flipped = aff.copy()
    flipped[0, 0] *= -1
    a_flip = write_nifti(tmp_path / "a_flip.nii.gz", adc, flipped)
    for a, detail in ((a_shape, "shape"), (a_shift, "affine"), (a_flip, "orientation")):
        r = validate_inputs(d, a, INF)
        assert codes(r.errors) == [Code.GRID_MISMATCH] and detail in r.errors[0].detail


def test_nonfinite_and_empty(tmp_path):
    dwi, adc, _, _ = synthetic_volumes()
    aff = las_affine((1.8, 1.8, 4.8))
    a = write_nifti(tmp_path / "a.nii.gz", adc, aff)
    nan = dwi.copy()
    nan[20, 20, 4] = np.nan
    assert codes(validate_inputs(write_nifti(tmp_path / "n.nii.gz", nan, aff), a, INF).errors) == [
        Code.NONFINITE
    ]
    inf = dwi.copy()
    inf[20, 20, 4] = np.inf
    assert codes(validate_inputs(write_nifti(tmp_path / "i.nii.gz", inf, aff), a, INF).errors) == [
        Code.NONFINITE
    ]
    const = np.zeros_like(dwi)
    assert codes(
        validate_inputs(write_nifti(tmp_path / "c.nii.gz", const, aff), a, INF).errors
    ) == [Code.EMPTY_IMAGE]


def test_too_small_and_invalid_spacing(tmp_path):
    d, a, _, _ = write_case(tmp_path / "s", shape=(20, 20, 10))
    assert Code.SHAPE_TOO_SMALL in codes(validate_inputs(d, a, INF).errors)
    d, a, _, _ = write_case(tmp_path / "p", spacing=(0.3, 0.3, 4.8))
    assert codes(validate_inputs(d, a, INF).errors) == [Code.SPACING_INVALID]
    d, a, _, _ = write_case(tmp_path / "z", spacing=(1.8, 1.8, 12.0))
    assert codes(validate_inputs(d, a, INF).errors) == [Code.SPACING_INVALID]


def test_spacing_outside_training_range_is_a_warning(tmp_path):
    d, a, _, _ = write_case(tmp_path, spacing=(1.0, 1.0, 1.0))
    r = validate_inputs(d, a, INF)
    assert r.errors == [] and Code.W_SPACING_OUT_OF_TRAIN_RANGE in codes(r.warnings)


def test_not_skull_stripped_is_an_error(tmp_path):
    dwi, adc, _, _ = synthetic_volumes()
    rng = np.random.default_rng(1)
    dwi = dwi + rng.normal(50, 5, dwi.shape).astype(np.float32)  # no constant background
    aff = las_affine((1.8, 1.8, 4.8))
    r = validate_inputs(
        write_nifti(tmp_path / "d.nii.gz", dwi, aff),
        write_nifti(tmp_path / "a.nii.gz", adc, aff),
        INF,
    )
    assert codes(r.errors) == [Code.BRAIN_MASK]


@pytest.mark.parametrize("scale", [1.0, 1e3, 1e6])
def test_three_training_adc_units_pass_without_warning(tmp_path, scale):
    d, a, _, _ = write_case(tmp_path, adc_scale=scale)
    r = validate_inputs(d, a, INF)
    assert r.errors == []
    assert not {Code.W_ADC_UNIT_UNUSUAL, Code.W_POSSIBLE_SWAP} & set(codes(r.warnings))


def test_unusual_adc_unit_warns(tmp_path):
    d, a, _, _ = write_case(tmp_path, adc_scale=1e9)
    r = validate_inputs(d, a, INF)
    assert r.errors == [] and Code.W_ADC_UNIT_UNUSUAL in codes(r.warnings)


def test_swapped_inputs_warn_but_are_not_changed(tmp_path):
    d, a, _, _ = write_case(tmp_path, swap=True)
    r = validate_inputs(d, a, INF)
    assert r.errors == [] and Code.W_POSSIBLE_SWAP in codes(r.warnings)
    assert float(np.median(r.dwi.data[r.dwi.data != 0])) < 1.0  # still the file given as DWI


def test_header_text_is_reported_by_field_name_only(tmp_path):
    d, a, _, _ = write_case(tmp_path, header_text={"descrip": "PATIENT^NAME 19700101"})
    r = validate_inputs(d, a, INF)
    w = [i for i in r.warnings if i.code == Code.W_HEADER_TEXT_PRESENT]
    assert len(w) == 2 and "descrip" in w[0].detail
    assert all("PATIENT" not in (i.message + (i.detail or "")) for i in r.warnings)


def test_orientation_and_obliquity_warnings(tmp_path):
    ras = np.diag([1.8, 1.8, 4.8, 1.0])
    d, a, _, _ = write_case(tmp_path / "ras", affine=ras)
    r = validate_inputs(d, a, INF)
    assert r.errors == [] and Code.W_ORIENTATION_UNUSUAL in codes(r.warnings)
    th = math.radians(5)
    rot = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    obl = las_affine((1.8, 1.8, 4.8))
    obl[:3, :3] = rot @ obl[:3, :3]
    assert obliquity_deg(obl) == pytest.approx(5.0, abs=1e-6)
    assert obliquity_deg(las_affine()) == pytest.approx(0.0, abs=1e-6)
    d, a, _, _ = write_case(tmp_path / "obl", affine=obl)
    assert Code.W_OBLIQUE in codes(validate_inputs(d, a, INF).warnings)


# --- sealed test guard -----------------------------------------------------------------------


def test_guard_refuses_sealed_case_paths_without_opening_anything(tmp_path, monkeypatch):
    opened, resolved = [], []
    real_open, real_realpath = open, os.path.realpath
    monkeypatch.setattr(
        "builtins.open", lambda f, *a, **k: opened.append(str(f)) or real_open(f, *a, **k)
    )
    monkeypatch.setattr(
        sealed_guard.os.path, "realpath", lambda p: resolved.append(p) or real_realpath(p)
    )
    monkeypatch.setattr(nib, "load", lambda *a, **k: pytest.fail("nibabel.load called"))
    paths = [
        tmp_path / "raw" / SEALED_ID / "ses-0001" / "dwi" / f"{SEALED_ID}_ses-0001_dwi.nii.gz",
        tmp_path / "copy" / f"{SEALED_ID}_adc.nii.gz",
    ]
    for p in paths:
        with pytest.raises(InferenceError) as e:
            sealed_guard.check_not_sealed([p, tmp_path / "other.nii.gz"], SPLIT)
        assert e.value.issue.code == Code.SEALED_TEST_CASE
        assert SEALED_ID not in e.value.issue.message + (e.value.issue.detail or "")
    assert not [f for f in opened if SEALED_ID in f]
    assert resolved == []  # refused on the path string, before any symlink resolution


def test_guard_allows_validation_and_unrelated_paths(tmp_path):
    rec = sealed_guard.check_not_sealed(
        [tmp_path / f"{VAL_ID}_dwi.nii.gz", tmp_path / "anon_adc.nii.gz"], SPLIT
    )
    assert rec["checked"] and rec["sealed_test_case_detected"] is False
    assert rec["isles_case_ids_in_input_paths"] == 1
    assert VAL_ID not in repr(rec)
