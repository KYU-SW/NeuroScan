"""Slice -> volume reconstruction and processed -> original restoration (synthetic data)."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from neuroscan_ai.analysis.components import components
from neuroscan_ai.datasets.base import CaseRecord
from neuroscan_ai.io import nifti
from neuroscan_ai.preprocessing.mask import binarize
from neuroscan_ai.preprocessing.pipeline import PreprocessConfig, preprocess_case
from neuroscan_ai.reconstruction.restore import restore_to_original
from neuroscan_ai.reconstruction.volume import (
    PredictionKind,
    PredictionVolume,
    ReconstructionError,
    VolumeGeometry,
    decompose_volume,
    reconstruct_volume,
)
from tests.conftest import make_synthetic_case, save_nifti


def geometry(shape=(6, 5, 4), spacing=(1.0, 1.0, 4.8)) -> VolumeGeometry:
    """A processed-like RAS grid."""
    return VolumeGeometry(shape, np.diag([*spacing, 1.0]))


def oblique_las_affine(spacing=(2.0, 2.0, 4.8), degrees=25.0) -> np.ndarray:
    t = np.radians(degrees)
    rot = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    aff = np.eye(4)
    aff[:3, :3] = rot @ np.diag([-spacing[0], spacing[1], spacing[2]])
    aff[:3, 3] = (90.0, -100.0, -40.0)
    return aff


# --- reconstruction --------------------------------------------------------------------------


def test_slice_order_is_by_index_not_by_input_order():
    g = geometry()
    vol = np.random.default_rng(0).integers(0, 2, g.shape).astype(np.uint8)
    pairs = decompose_volume(vol)
    shuffled = [pairs[i] for i in np.random.default_rng(1).permutation(len(pairs))]
    out = reconstruct_volume(shuffled, g, "binary")
    np.testing.assert_array_equal(out.data, vol)


def test_missing_slice_detected():
    g = geometry()
    pairs = decompose_volume(np.zeros(g.shape, np.uint8))
    with pytest.raises(ReconstructionError, match="missing slice indices: \\[2\\]"):
        reconstruct_volume([p for p in pairs if p[0] != 2], g, "binary")


def test_duplicate_slice_detected():
    g = geometry()
    pairs = decompose_volume(np.zeros(g.shape, np.uint8))
    with pytest.raises(ReconstructionError, match="duplicate slice index 1"):
        reconstruct_volume([*pairs, pairs[1]], g, "binary")


def test_invalid_slice_shape_detected():
    g = geometry()
    pairs = decompose_volume(np.zeros(g.shape, np.uint8))
    pairs[0] = (0, np.zeros((5, 6), np.uint8))  # transposed
    with pytest.raises(ReconstructionError, match="shape"):
        reconstruct_volume(pairs, g, "binary")


@pytest.mark.parametrize("bad_index", [-1, 4, 1.0, True])
def test_slice_index_range_and_type_detected(bad_index):
    g = geometry()
    pairs = decompose_volume(np.zeros(g.shape, np.uint8))[1:]
    with pytest.raises(ReconstructionError):
        reconstruct_volume([(bad_index, np.zeros((6, 5), np.uint8)), *pairs], g, "binary")


def test_binary_values_validated():
    g = geometry()
    pairs = decompose_volume(np.zeros(g.shape, np.uint8))
    pairs[0] = (0, np.full((6, 5), 0.7))
    with pytest.raises(ReconstructionError, match="binary"):
        reconstruct_volume(pairs, g, "binary")


@pytest.mark.parametrize("bad", [np.nan, 1.5, -0.1])
def test_probability_values_validated(bad):
    g = geometry()
    pairs = decompose_volume(np.zeros(g.shape, np.float32))
    arr = np.zeros((6, 5), np.float32)
    arr[0, 0] = bad
    pairs[3] = (3, arr)
    with pytest.raises(ReconstructionError, match="probability"):
        reconstruct_volume(pairs, g, "probability")


def test_binary_mask_reconstruction_is_identical():
    g = geometry()
    vol = np.zeros(g.shape, np.uint8)
    vol[1:3, 2:4, 1:3] = 1
    out = reconstruct_volume(decompose_volume(vol), g, PredictionKind.BINARY)
    assert out.data.dtype == np.uint8
    np.testing.assert_array_equal(out.data, vol)


def test_probability_map_reconstruction_is_not_thresholded():
    g = geometry()
    prob = np.random.default_rng(2).random(g.shape).astype(np.float32)
    out = reconstruct_volume(decompose_volume(prob), g, "probability")
    assert out.data.dtype == np.float32
    np.testing.assert_array_equal(out.data, prob)  # e.g. 0.3 stays 0.3
    assert 0 < np.mean((out.data > 0) & (out.data < 1))


def test_geometry_affine_spacing_orientation_preserved(tmp_path):
    g = geometry(spacing=(1.0, 1.0, 4.8))
    out = reconstruct_volume(decompose_volume(np.zeros(g.shape, np.uint8)), g, "binary")
    assert out.geometry is g
    np.testing.assert_array_equal(out.geometry.affine, g.affine)
    np.testing.assert_allclose(out.geometry.spacing, (1.0, 1.0, 4.8))
    assert out.geometry.orientation == "RAS"
    out.save(tmp_path / "pred.nii.gz")
    img = nib.load(tmp_path / "pred.nii.gz")
    np.testing.assert_allclose(img.affine, g.affine, atol=1e-6)
    assert img.shape == g.shape


# --- restoration -----------------------------------------------------------------------------


def _processed_case(tmp_path, record):
    cfg = PreprocessConfig(version="vtest", target_spacing=(1.0, 1.0, None))
    meta = preprocess_case(record, cfg, tmp_path / "out")
    case_dir = tmp_path / "out" / record.case_id
    mask = np.asanyarray(nib.load(case_dir / "mask.nii.gz").dataobj)
    return meta, mask


def test_original_space_restoration_geometry_and_labels(tmp_path):
    rec = make_synthetic_case(tmp_path / "raw", spacing=(2.0, 2.0, 4.8), shape=(20, 18, 6))
    meta, proc_mask = _processed_case(tmp_path, rec)
    proc = VolumeGeometry.from_meta(meta, "processed")
    orig = VolumeGeometry.from_meta(meta, "original")
    restored = restore_to_original(
        reconstruct_volume(decompose_volume(proc_mask), proc, "binary"), orig
    )
    gt_img = nib.load(rec.label)
    assert restored.data.shape == gt_img.shape
    np.testing.assert_allclose(restored.geometry.affine, gt_img.affine, atol=1e-5)
    assert restored.geometry.orientation == "LAS"
    assert set(np.unique(restored.data).tolist()) == {0, 1}
    np.testing.assert_array_equal(restored.data, binarize(gt_img.get_fdata()))


def test_probability_restoration_stays_continuous_and_matches_binary_path(tmp_path):
    rec = make_synthetic_case(tmp_path / "raw", spacing=(1.5, 1.5, 4.0))
    meta, proc_mask = _processed_case(tmp_path, rec)
    proc = VolumeGeometry.from_meta(meta, "processed")
    orig = VolumeGeometry.from_meta(meta, "original")
    prob = PredictionVolume(proc_mask.astype(np.float32), proc, PredictionKind.PROBABILITY)
    restored_prob = restore_to_original(prob, orig)
    assert restored_prob.data.dtype == np.float32
    assert restored_prob.data.min() >= 0.0 and restored_prob.data.max() <= 1.0
    assert ((restored_prob.data > 0) & (restored_prob.data < 1)).any()  # not thresholded
    restored_bin = restore_to_original(
        PredictionVolume(proc_mask, proc, PredictionKind.BINARY), orig
    )
    np.testing.assert_array_equal(restored_prob.data >= 0.5, restored_bin.data.astype(bool))


def test_oblique_affine_round_trip(tmp_path):
    shape = (24, 22, 8)
    aff = oblique_las_affine((1.8, 1.8, 4.8), degrees=30.0)
    rng = np.random.default_rng(0)
    gt = np.zeros(shape)
    gt[6:14, 8:15, 2:6] = 1.0
    d = tmp_path / "raw" / "obl"
    rec = CaseRecord(
        dataset="synthetic",
        case_id="obl",
        images={
            "dwi": save_nifti(d / "dwi.nii.gz", rng.random(shape), aff),
            "adc": save_nifti(d / "adc.nii.gz", rng.random(shape), aff),
        },
        label=save_nifti(d / "msk.nii.gz", gt, aff),
    )
    assert nifti.orientation(aff) == "LAS"
    meta, proc_mask = _processed_case(tmp_path, rec)
    assert json.loads((tmp_path / "out" / "obl" / "meta.json").read_text())["case_id"] == "obl"
    restored = restore_to_original(
        reconstruct_volume(
            decompose_volume(proc_mask), VolumeGeometry.from_meta(meta, "processed"), "binary"
        ),
        VolumeGeometry.from_meta(meta, "original"),
    )
    gt_bin = binarize(gt)
    dice = 2 * (restored.data & gt_bin).sum() / (restored.data.sum() + gt_bin.sum())
    assert dice >= 0.97
    # lesion stays at the same physical position
    c_gt = components(gt_bin, (1.8, 1.8, 4.8), aff)[0]["centroid_world_mm"]
    c_rs = components(restored.data, (1.8, 1.8, 4.8), aff)[0]["centroid_world_mm"]
    assert np.linalg.norm(np.subtract(c_gt, c_rs)) < 0.5
