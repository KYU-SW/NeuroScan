"""Integration tests against the real ISLES 2022 data (skipped when the data is absent).

Run only these with:  pytest -m data
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

from neuroscan_ai.data.split import SPLITS
from neuroscan_ai.datasets import discover
from neuroscan_ai.io import nifti
from neuroscan_ai.preprocessing.geometry import reorient
from neuroscan_ai.preprocessing.mask import binarize, lesion_volume_ml
from neuroscan_ai.qa.dataset_qa import sha256_manifest
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.data

EMPTY_GT = {"sub-strokecase0150", "sub-strokecase0151", "sub-strokecase0170"}
QA_DIR = REPO_ROOT / "results" / "qa"
SPLIT_FILE = REPO_ROOT / "data" / "splits" / "split_v1_seed42.json"
PREPROCESS_CONFIG = REPO_ROOT / "configs" / "preprocess_v1.yaml"


@pytest.fixture(scope="module")
def records():
    root = REPO_ROOT / "data" / "raw" / "ISLES2022"
    if not (root / "dataset_description.json").exists():
        pytest.skip("ISLES 2022 raw data not available")
    return discover("isles2022", root)


@pytest.fixture(scope="module")
def headers(records):
    return {r.case_id: {k: nib.load(p) for k, p in r.all_paths().items()} for r in records}


def test_pairing_all_250_cases_have_dwi_adc_gt(records):
    assert len(records) == 250
    assert [r.case_id for r in records] == [f"sub-strokecase{i:04d}" for i in range(1, 251)]
    for r in records:
        for key, path in r.all_paths().items():
            assert path.exists(), f"{r.case_id} missing {key}"
            assert r.case_id in path.name


def test_affine_consistency_dwi_adc_gt(headers):
    for case, imgs in headers.items():
        assert nifti.same_grid(imgs["adc"], imgs["dwi"]), case
        assert nifti.same_grid(imgs["label"], imgs["dwi"]), case


def test_spacing_header_matches_affine_and_slice_axis_is_last(headers):
    for case, imgs in headers.items():
        for key in ("dwi", "adc", "label"):
            sp = nifti.spacing_from_affine(imgs[key].affine)
            np.testing.assert_allclose(
                imgs[key].header.get_zooms()[:3], sp, atol=1e-3, err_msg=case
            )
            assert all(s > 0 for s in sp)
            assert sp[2] >= max(sp[:2]) - 1e-3, case


def test_orientation_all_las(headers):
    for case, imgs in headers.items():
        for key in ("dwi", "adc", "label"):
            assert nifti.orientation(imgs[key].affine) == "LAS", f"{case} {key}"


def test_binary_mask_conversion_on_real_masks(headers):
    empty = set()
    for case, imgs in headers.items():
        values = set(np.unique(binarize(imgs["label"].get_fdata())).tolist())
        assert values in ({0, 1}, {0}), case
        if values == {0}:
            empty.add(case)
    assert empty == EMPTY_GT


def test_lesion_volume_matches_qa_summary(headers):
    qa_csv = QA_DIR / "case_summary.csv"
    if not qa_csv.exists():
        pytest.skip("run scripts/run_qa.py first")
    qa = pd.read_csv(qa_csv).set_index("case_id")
    for case, imgs in headers.items():
        m = binarize(imgs["label"].get_fdata())
        vol = lesion_volume_ml(m, nifti.spacing_from_affine(imgs["label"].affine))
        assert vol == pytest.approx(qa.loc[case, "lesion_volume_ml"], abs=1e-5), case
        assert bool(qa.loc[case, "has_lesion"]) == (case not in EMPTY_GT)


def test_raw_files_unchanged_since_qa(isles_root):
    manifest_file = QA_DIR / "raw_manifest.json"
    if not manifest_file.exists():
        pytest.skip("run scripts/run_qa.py first")
    recorded = json.loads(manifest_file.read_text(encoding="utf-8"))["files"]
    assert sha256_manifest(isles_root) == recorded


# --- split -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def split():
    if not SPLIT_FILE.exists():
        pytest.skip("run scripts/make_split.py first")
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))


def test_split_overlap_zero(split):
    sets = [set(split["splits"][s]) for s in SPLITS]
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1 :])


def test_split_coverage_100_percent(split, records):
    covered = set().union(*(split["splits"][s] for s in SPLITS))
    assert covered == {r.case_id for r in records}
    assert sum(len(split["splits"][s]) for s in SPLITS) == 250


def test_split_is_reproducible_from_qa(split):
    from neuroscan_ai.data.split import make_patient_split
    from scripts.make_split import load_cases

    again = make_patient_split(load_cases(QA_DIR / "case_summary.csv"), seed=split["seed"])
    assert again["splits"] == split["splits"]
    assert [len(split["stats"][s]["negatives"]) for s in SPLITS] == [1, 1, 1]


# --- processed/v1 ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def processed():
    if not PREPROCESS_CONFIG.exists():
        pytest.skip("preprocess config not created yet")
    cfg = yaml.safe_load(PREPROCESS_CONFIG.read_text(encoding="utf-8"))
    out = REPO_ROOT / cfg["out_root"]
    if not (out / "cases.csv").exists():
        pytest.skip("processed dataset not generated yet")
    return cfg, out


def test_processed_reload_matches_meta(processed, records, headers):
    cfg, out = processed
    target = cfg["target_spacing"]
    for r in records:
        case_dir = out / r.case_id
        meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["case_id"] == r.case_id
        assert meta["preprocessing_version"] == cfg["version"]
        orig = headers[r.case_id]["dwi"]
        assert meta["original"]["shape"] == list(orig.shape)
        np.testing.assert_allclose(meta["original"]["affine"], orig.affine, atol=1e-5)

        for key in ("dwi", "adc", "mask"):
            img = nib.load(case_dir / f"{key}.nii.gz")
            arr = np.asanyarray(img.dataobj)
            assert arr.dtype == (np.uint8 if key == "mask" else np.float32), f"{r.case_id} {key}"
            assert list(arr.shape) == meta["processed"]["shape"]
            np.testing.assert_allclose(img.affine, meta["processed"]["affine"], atol=1e-5)
            assert nifti.orientation(img.affine) == "RAS"
            assert np.isfinite(arr).all()
            if key == "mask":
                assert set(np.unique(arr).tolist()) <= {0, 1}
                assert bool(arr.any()) == meta["lesion"]["has_lesion"]

        sp = meta["processed"]["spacing"]
        orig_sp = meta["original"]["spacing"]
        for ax in range(3):
            expected = orig_sp[ax] if target[ax] is None else target[ax]
            if abs(expected - orig_sp[ax]) <= 1e-3:
                expected = orig_sp[ax]
            assert sp[ax] == pytest.approx(expected, abs=1e-4), r.case_id
        if target[2] is None:  # z kept: processed slice k is exactly original slice k
            assert meta["processed"]["shape"][2] == meta["original"]["shape"][2]
            _, orig_ras = reorient(np.zeros(orig.shape[:3]), orig.affine, "RAS")
            to_orig = np.linalg.inv(orig_ras) @ np.array(meta["processed"]["affine"])
            for ijk in [(0, 0, 0), (5, 7, 3), (10, 2, meta["processed"]["shape"][2] - 1)]:
                k_orig = (to_orig @ np.array([*ijk, 1.0]))[2]
                assert k_orig == pytest.approx(ijk[2], abs=1e-4), r.case_id
        assert meta["lesion"]["has_lesion"] == (r.case_id not in EMPTY_GT)
        assert not meta["lesion"]["lost_in_processing"]


def test_processed_does_not_write_into_raw(processed, isles_root):
    _, out = processed
    assert isles_root.resolve() not in Path(out).resolve().parents
