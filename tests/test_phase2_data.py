"""Phase 2 integration tests on real ISLES 2022 GT (skipped when data/processed is absent).

Expected values are not arbitrary tolerances: the restored mask must equal the Phase 1
round-trip method exactly, and the per-case floor is the Phase 1 B' minimum (0.9682) recorded in
results/qa/spacing_policy_comparison.json.
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from neuroscan_ai.analysis.components import components
from neuroscan_ai.evaluation.segmentation import evaluate_case
from neuroscan_ai.io import nifti
from neuroscan_ai.preprocessing.geometry import reorient, resample_to_grid, resample_to_spacing
from neuroscan_ai.preprocessing.mask import binarize
from neuroscan_ai.reconstruction.restore import restore_to_original
from neuroscan_ai.reconstruction.volume import (
    PredictionKind,
    VolumeGeometry,
    decompose_volume,
    reconstruct_volume,
)
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.data

PROCESSED = REPO_ROOT / "data" / "processed" / "v1"
SPLIT_FILE = REPO_ROOT / "data" / "splits" / "split_v1_seed42.json"
NEGATIVES = ("sub-strokecase0150", "sub-strokecase0151", "sub-strokecase0170")
PHASE1_MIN_DICE = 0.9682  # B' all-case minimum, Phase 1 policy comparison
# Highly oblique (>= 30 deg) and thick-slice cases checked explicitly in addition to the test split
EXTRA_CASES = (
    "sub-strokecase0031",
    "sub-strokecase0158",
    "sub-strokecase0089",
    "sub-strokecase0248",
)


@pytest.fixture(scope="module")
def split():
    if not (PROCESSED / "cases.csv").exists() or not SPLIT_FILE.exists():
        pytest.skip("processed/v1 or split not available")
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["splits"]


def load(case_id):
    meta = json.loads((PROCESSED / case_id / "meta.json").read_text(encoding="utf-8"))
    proc = np.asanyarray(nib.load(PROCESSED / case_id / "mask.nii.gz").dataobj)
    orig_img = nib.load(REPO_ROOT / meta["source"]["label"])
    return meta, proc, orig_img, binarize(orig_img.get_fdata())


def test_test_split_37_gt_reconstruction_is_identical(split):
    assert len(split["test"]) == 37
    rng = np.random.default_rng(0)
    for cid in split["test"]:
        meta, proc, _, _ = load(cid)
        geom = VolumeGeometry.from_meta(meta, "processed")
        pairs = decompose_volume(proc)
        out = reconstruct_volume([pairs[i] for i in rng.permutation(len(pairs))], geom, "binary")
        np.testing.assert_array_equal(out.data, proc, err_msg=cid)
        np.testing.assert_allclose(out.geometry.affine, meta["processed"]["affine"])
        r = evaluate_case(out.data, proc, geom.spacing, cid)
        if meta["lesion"]["has_lesion"]:
            assert (r["dice"], r["iou"], r["sensitivity"], r["precision"]) == (1, 1, 1, 1)
            assert r["volume_error_ml"] == 0
        else:
            assert r["negative_outcome"] == "correct_negative"


def test_original_space_round_trip_matches_phase1(split):
    cases = sorted(set(split["test"]) | set(NEGATIVES) | set(EXTRA_CASES))
    for cid in cases:
        meta, proc, orig_img, gt = load(cid)
        orig = VolumeGeometry(tuple(orig_img.shape), orig_img.affine)
        restored = restore_to_original(
            reconstruct_volume(
                decompose_volume(proc), VolumeGeometry.from_meta(meta, "processed"), "binary"
            ),
            orig,
        )
        assert restored.data.shape == gt.shape
        np.testing.assert_allclose(restored.geometry.affine, orig_img.affine, atol=1e-5)
        assert restored.geometry.orientation == meta["original"]["orientation"] == "LAS"

        # independent Phase 1 round trip in RAS space must give the same mask (flipped)
        ras_gt, ras_aff = reorient(gt, orig_img.affine, "RAS")
        p, p_aff, _ = resample_to_spacing(ras_gt, ras_aff, (1.0, 1.0, None), order=1)
        back = resample_to_grid((p >= 0.5).astype(np.uint8), p_aff, ras_aff, ras_gt.shape, 1)
        ras_restored, _ = reorient(restored.data, orig_img.affine, "RAS")
        np.testing.assert_array_equal(ras_restored, (back >= 0.5).astype(np.uint8), err_msg=cid)

        sp = nifti.spacing_from_affine(orig_img.affine)
        r = evaluate_case(restored.data, gt, sp, cid)
        if cid in NEGATIVES:
            assert r["negative_outcome"] == "correct_negative"
            continue
        assert r["dice"] >= PHASE1_MIN_DICE, cid
        if np.allclose(sp[:2], 2.0):  # integer XY ratio 2:1 -> exact inverse
            assert r["dice"] == 1.0, cid


def test_oblique_affine_lesion_position_preserved(split):
    for cid in ("sub-strokecase0031", "sub-strokecase0158", "sub-strokecase0089"):
        meta, proc, orig_img, gt = load(cid)
        assert np.degrees(nib.affines.obliquity(orig_img.affine)).max() > 30
        restored = restore_to_original(
            reconstruct_volume(
                decompose_volume(proc), VolumeGeometry.from_meta(meta, "processed"), "binary"
            ),
            VolumeGeometry(tuple(orig_img.shape), orig_img.affine),
        )
        sp = nifti.spacing_from_affine(orig_img.affine)
        c_gt = components(gt, sp, orig_img.affine)[0]["centroid_world_mm"]
        c_rs = components(restored.data, sp, orig_img.affine)[0]["centroid_world_mm"]
        assert np.linalg.norm(np.subtract(c_gt, c_rs)) < min(sp), cid


def test_probability_restoration_on_real_case(split):
    meta, proc, orig_img, _ = load(split["test"][0])
    geom = VolumeGeometry.from_meta(meta, "processed")
    prob = reconstruct_volume(
        [(z, s.astype(np.float32) * 0.8) for z, s in decompose_volume(proc)], geom, "probability"
    )
    assert prob.kind is PredictionKind.PROBABILITY
    restored = restore_to_original(prob, VolumeGeometry(tuple(orig_img.shape), orig_img.affine))
    assert restored.data.dtype == np.float32
    assert 0.0 <= restored.data.min() and restored.data.max() <= 0.8 + 1e-6
