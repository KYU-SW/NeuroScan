"""Unit tests on synthetic volumes (no ISLES data needed)."""

from __future__ import annotations

import hashlib
import json

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from neuroscan_ai.datasets.isles2022 import discover_isles2022
from neuroscan_ai.io import nifti
from neuroscan_ai.preprocessing.geometry import reorient, resample_to_grid, resample_to_spacing
from neuroscan_ai.preprocessing.mask import binarize, lesion_volume_ml
from neuroscan_ai.preprocessing.pipeline import PreprocessConfig, preprocess_case
from tests.conftest import las_affine, make_synthetic_case, save_nifti


def world(affine: np.ndarray, ijk) -> np.ndarray:
    return (affine @ np.append(np.asarray(ijk, float), 1.0))[:3]


# --- binary mask / volume ------------------------------------------------------------------


def test_binarize_handles_scaling_artefact_and_is_exactly_binary():
    raw = np.array([0.0, 0.9999999997671694, 1.0, 0.5, 0.51, 1e-9])
    out = binarize(raw)
    assert out.dtype == np.uint8
    assert out.tolist() == [0, 1, 1, 0, 1, 0]
    assert set(np.unique(out).tolist()) == {0, 1}


def test_binarize_empty_mask_stays_empty():
    assert set(np.unique(binarize(np.zeros((4, 4, 4)))).tolist()) == {0}


def test_lesion_volume_ml():
    m = np.zeros((10, 10, 10), np.uint8)
    m[:5, :2, :1] = 1  # 10 voxels
    assert lesion_volume_ml(m, (2.0, 2.0, 2.0)) == pytest.approx(10 * 8 / 1000)
    assert lesion_volume_ml(m, (1.0, 1.0, 4.8)) == pytest.approx(10 * 4.8 / 1000)


# --- orientation -----------------------------------------------------------------------------


def test_reorient_las_to_ras_is_lossless_and_preserves_world_positions():
    data = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    aff = las_affine((2.0, 2.0, 5.0))
    out, out_aff = reorient(data, aff, "RAS")
    assert nifti.orientation(aff) == "LAS" and nifti.orientation(out_aff) == "RAS"
    assert out.shape == data.shape
    for ijk in np.ndindex(*data.shape):
        # find the same world point in the output and compare values
        oi = np.round(np.linalg.solve(out_aff, np.append(world(aff, ijk), 1))[:3]).astype(int)
        assert out[tuple(oi)] == data[ijk]
    np.testing.assert_array_equal(out[:, :, 1], data[::-1, :, 1])  # only x flipped


# --- spacing ---------------------------------------------------------------------------------


def test_resample_identity_when_already_at_target():
    data = np.random.default_rng(1).random((8, 8, 5)).astype(np.float32)
    aff = las_affine((2.0, 2.0, 2.0))
    out, out_aff, axes = resample_to_spacing(data, aff, (2.0, 2.0, None), order=1)
    assert axes == [False, False, False]
    np.testing.assert_array_equal(out, data)
    np.testing.assert_array_equal(out_aff, aff)


def test_resample_xy_keeps_z_slices_voxel_for_voxel():
    rng = np.random.default_rng(2)
    data = rng.random((24, 24, 6)).astype(np.float32)
    data[:, :, 3] = 7.0  # a constant slice must stay exactly constant
    aff = las_affine((1.0, 1.0, 4.8))
    out, out_aff, axes = resample_to_spacing(data, aff, (2.0, 2.0, None), order=1)
    assert axes == [True, True, False]
    assert out.shape == (12, 12, 6)
    np.testing.assert_allclose(nifti.spacing_from_affine(out_aff), (2.0, 2.0, 4.8))
    np.testing.assert_allclose(out[:, :, 3], 7.0)
    for k in range(6):  # each output slice k sits at the same world z as input slice k
        assert world(out_aff, (0, 0, k))[2] == pytest.approx(world(aff, (0, 0, k))[2])


def test_resample_keeps_field_of_view_corner_aligned():
    aff = las_affine((1.0, 1.0, 2.0))
    out, out_aff, _ = resample_to_spacing(np.zeros((20, 20, 4)), aff, (2.0, 2.0, 2.0), order=1)
    corner_in = world(aff, (-0.5, -0.5, -0.5))
    corner_out = world(out_aff, (-0.5, -0.5, -0.5))
    np.testing.assert_allclose(corner_out, corner_in, atol=1e-9)


def test_resampled_affine_agrees_with_simpleitk(tmp_path):
    aff = las_affine((1.1458, 1.1458, 4.8))
    out, out_aff, _ = resample_to_spacing(np.zeros((30, 30, 5)), aff, (2.0, 2.0, None), order=1)
    path = save_nifti(tmp_path / "r.nii.gz", out.astype(np.float32), out_aff)
    img = sitk.ReadImage(str(path))
    np.testing.assert_allclose(img.GetSpacing(), nifti.spacing_from_affine(out_aff), atol=1e-4)
    ras = world(out_aff, (3, 4, 2))
    lps = img.TransformIndexToPhysicalPoint((3, 4, 2))
    np.testing.assert_allclose(lps, (-ras[0], -ras[1], ras[2]), atol=1e-3)


def test_resample_to_grid_identity_and_roundtrip():
    rng = np.random.default_rng(3)
    data = rng.random((10, 10, 4)).astype(np.float32)
    aff = las_affine((2.0, 2.0, 4.8))
    np.testing.assert_array_equal(resample_to_grid(data, aff, aff, data.shape, order=1), data)

    smooth = np.fromfunction(lambda i, j, k: i + 2 * j + 0 * k, (40, 40, 4)).astype(np.float32)
    aff1 = las_affine((1.0, 1.0, 4.8))
    up, up_aff, _ = resample_to_spacing(smooth, aff1, (0.5, 0.5, None), order=1)
    back = resample_to_grid(up, up_aff, aff1, smooth.shape, order=1)
    np.testing.assert_allclose(back[2:-2, 2:-2], smooth[2:-2, 2:-2], atol=1e-4)


# --- ISLES adapter ---------------------------------------------------------------------------


def test_isles_adapter_pairs_files_by_bids_name(tmp_path):
    for case in ("sub-strokecase0002", "sub-strokecase0001"):
        (tmp_path / case / "ses-0001" / "dwi").mkdir(parents=True)
    recs = discover_isles2022(tmp_path)
    assert [r.case_id for r in recs] == ["sub-strokecase0001", "sub-strokecase0002"]
    r = recs[0]
    assert r.images["dwi"].name == "sub-strokecase0001_ses-0001_dwi.nii.gz"
    assert r.images["adc"].name == "sub-strokecase0001_ses-0001_adc.nii.gz"
    assert r.label.parts[-4:] == (
        "derivatives",
        "sub-strokecase0001",
        "ses-0001",
        "sub-strokecase0001_ses-0001_msk.nii.gz",
    )
    assert r.extra["flair"].parent.name == "anat"
    assert list(r.images) == ["dwi", "adc"]  # FLAIR is not a model input


# --- full case pipeline ----------------------------------------------------------------------


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("lesion", [True, False])
def test_preprocess_case_end_to_end(tmp_path, lesion):
    rec = make_synthetic_case(tmp_path / "raw", lesion=lesion)
    before = {k: _sha(p) for k, p in rec.all_paths().items()}
    cfg = PreprocessConfig(version="vtest", target_spacing=(1.0, 1.0, None))
    meta = preprocess_case(rec, cfg, tmp_path / "out")

    assert {k: _sha(p) for k, p in rec.all_paths().items()} == before  # raw untouched
    case_dir = tmp_path / "out" / rec.case_id
    on_disk = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    assert on_disk == meta
    for key in (
        "case_id",
        "original",
        "lesion",
        "processed",
        "preprocessing_version",
        "transforms",
    ):
        assert key in meta
    for key in ("shape", "spacing", "affine", "orientation", "dtype"):
        assert key in meta["original"] and key in meta["processed"]
    assert meta["original"]["orientation"] == "LAS" and meta["processed"]["orientation"] == "RAS"
    assert meta["original"]["dtype"]["adc"] == "int16"

    imgs = {k: nib.load(case_dir / f"{k}.nii.gz") for k in ("dwi", "adc", "mask")}
    for k, img in imgs.items():
        assert img.get_data_dtype() == (np.uint8 if k == "mask" else np.float32)
        assert list(img.shape) == meta["processed"]["shape"]
        np.testing.assert_allclose(img.affine, meta["processed"]["affine"], atol=1e-5)
    assert meta["processed"]["shape"] == [30, 27, 6]  # 20x18 @1.5mm -> 30x27 @1mm, z kept
    np.testing.assert_allclose(meta["processed"]["spacing"], (1.0, 1.0, 4.0))

    mask = np.asanyarray(imgs["mask"].dataobj)
    expected_values = {0, 1} if lesion else {0}
    assert set(np.unique(mask).tolist()) == expected_values
    assert meta["lesion"]["has_lesion"] is lesion
    assert meta["lesion"]["original_voxel_count"] == (7 * 5 * 2 if lesion else 0)
    assert meta["lesion"]["original_volume_ml"] == pytest.approx(
        (70 * 1.5 * 1.5 * 4.0 / 1000) if lesion else 0.0
    )
    if lesion:
        assert meta["lesion"]["processed_volume_ml"] == pytest.approx(
            meta["lesion"]["original_volume_ml"], rel=0.15
        )
