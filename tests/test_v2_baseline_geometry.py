"""v2 2D baseline: architecture and 1 mm in-plane geometry (forward + inverse). Synthetic arrays
only, CPU only. No MRI, no ISLES / SOOP access, no training, no v1 code."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuroscan_ai.v2.baseline_outputs import (
    OutputGridError,
    build_outputs,
    restore_mask,
    restore_probability,
)
from neuroscan_ai.v2.baseline_preprocess import (
    BRAIN_MASK_POLICY,
    PREPROCESS_POLICY,
    PreprocessError,
    load_and_prepare,
    prepare_arrays,
)
from neuroscan_ai.v2.geometry import (
    Grid,
    check_on_grid,
    crop_to,
    inplane_target_grid,
    pad_to_multiple,
    resample,
    resample_mask,
    validate_grid,
)
from neuroscan_ai.v2.unet_2d import (
    CHANNELS,
    EXPECTED_PARAMETERS,
    INPUT_MULTIPLE,
    MODEL_ID,
    build_model,
    count_parameters,
    describe,
)


def _affine(spacing=(1.0, 1.0, 5.0), rot_deg=0.0, origin=(-40.0, 30.0, -10.0)):
    """Voxel->world affine; optional rotation about the world z axis (oblique in-plane)."""
    t = np.deg2rad(rot_deg)
    r = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
    aff = np.eye(4)
    aff[:3, :3] = r @ np.diag(spacing)
    aff[:3, 3] = origin
    return aff


def _volume(shape=(40, 36, 6), seed=0, lesion=True):
    """Synthetic skull-stripped DWI / ADC (background exactly 0) and a binary GT."""
    rng = np.random.default_rng(seed)
    g = np.indices(shape).astype(float)
    c = (np.array(shape) - 1) / 2
    brain = (
        ((g[0] - c[0]) / (shape[0] * 0.38)) ** 2
        + ((g[1] - c[1]) / (shape[1] * 0.38)) ** 2
        + ((g[2] - c[2]) / (shape[2] * 0.7)) ** 2
    ) <= 1
    gt = np.zeros(shape, np.uint8)
    if lesion:
        gt[(((g[0] - c[0] + 4) / 4) ** 2 + ((g[1] - c[1]) / 3) ** 2 <= 1) & brain] = 1
    dwi = np.where(brain, 300 + rng.normal(0, 20, shape), 0).astype(np.float32)
    dwi[gt == 1] += 400
    adc = np.where(brain, 800 + rng.normal(0, 40, shape), 0).astype(np.float32)
    adc[gt == 1] -= 400
    return dwi, adc, gt


def _prep(spacing=(1.0, 1.0, 5.0), rot=0.0, shape=(40, 36, 6), seed=0):
    dwi, adc, gt = _volume(shape, seed)
    aff = _affine(spacing, rot)
    return prepare_arrays(dwi, adc, gt, {"dwi": aff, "adc": aff, "gt": aff}), (dwi, adc, gt, aff)


# ---------------------------------------------------------------- 1-3 architecture
def test_01_parameter_count_is_approved_value():
    m = build_model(seed=0)
    assert CHANNELS == (36, 72, 144, 288)
    assert count_parameters(m) == EXPECTED_PARAMETERS == 2_439_577
    d = describe(m, 0)
    assert d["model_id"] == MODEL_ID and d["trainable_parameters"] == 2_439_577
    assert d["in_channels"] == 2 and d["out_channels"] == 1 and d["pretrained_weights"] is None


def test_02_input_must_be_multiple_of_8():
    m = build_model(seed=0).eval()
    assert INPUT_MULTIPLE == 8
    with torch.no_grad():
        assert m(torch.zeros(1, 2, 48, 40)).shape == (1, 1, 48, 40)
        with pytest.raises(ValueError):
            m(torch.zeros(1, 2, 44, 40))


def test_03_pad_crop_round_trip():
    x = np.random.default_rng(1).normal(size=(2, 37, 29)).astype(np.float32)
    padded, size = pad_to_multiple(x, INPUT_MULTIPLE)
    assert padded.shape == (2, 40, 32) and size == (37, 29)
    assert np.array_equal(crop_to(padded, size), x)
    assert np.array_equal(pad_to_multiple(padded, 8)[0], padded)  # already a multiple


# ---------------------------------------------------------------- 4-8 target grid
def test_04_oblique_affine_direction_kept_and_world_consistent():
    src = validate_grid((30, 26, 4), _affine((0.8, 0.8, 4.0), rot_deg=23.0))
    dst = inplane_target_grid(src)
    for ax in range(3):
        a = src.affine[:3, ax] / np.linalg.norm(src.affine[:3, ax])
        b = dst.affine[:3, ax] / np.linalg.norm(dst.affine[:3, ax])
        assert np.allclose(a, b, atol=1e-12)  # direction cosines unchanged
    assert np.allclose(dst.spacing, (1.0, 1.0, 4.0))
    # voxel-0 outer corner fixed
    corner = np.array([-0.5, -0.5, -0.5, 1.0])
    assert np.allclose(src.affine @ corner, dst.affine @ corner, atol=1e-9)
    # a linear function of world position is reproduced exactly (away from the far edge)
    coef = np.array([0.7, -1.3, 0.4])

    def world_field(grid):
        idx = np.indices(grid.shape).reshape(3, -1)
        xyz = grid.affine[:3, :3] @ idx + grid.affine[:3, 3:4]
        return (coef @ xyz).reshape(grid.shape)

    out = resample(world_field(src), src, dst, order=1)
    inner = (slice(1, dst.shape[0] - 2), slice(1, dst.shape[1] - 2), slice(None))
    assert np.allclose(out[inner], world_field(dst)[inner], atol=1e-6)


def test_05_one_mm_input_is_copied_exactly():
    (case, (dwi, _adc, gt, aff)) = _prep(spacing=(1.0, 1.0, 5.0))
    assert case.model_grid.shape == dwi.shape
    assert np.array_equal(case.model_grid.affine, aff)
    assert case.record["resampled"] is False
    assert np.array_equal(case.label, gt == 1)
    g = validate_grid(dwi.shape, aff)
    assert np.array_equal(resample(dwi, g, g, order=1), dwi.astype(np.float64))


def test_06_two_mm_input_upsampled_to_one_mm():
    src = validate_grid((20, 18, 5), _affine((2.0, 2.0, 5.0)))
    dst = inplane_target_grid(src)
    assert dst.shape == (40, 36, 5) and np.allclose(dst.spacing, (1.0, 1.0, 5.0))


def test_07_non_square_inplane_spacing():
    src = validate_grid((25, 16, 3), _affine((0.8, 1.25, 6.0)))
    dst = inplane_target_grid(src)
    assert dst.shape == (20, 20, 3)  # ceil(25*0.8)=20, ceil(16*1.25)=20
    assert np.allclose(dst.spacing, (1.0, 1.0, 6.0))
    src2 = validate_grid((23, 15, 3), _affine((0.9, 1.1, 6.0)))
    assert inplane_target_grid(src2).shape == (21, 17, 3)  # ceil(20.7)=21, ceil(16.5)=17


def test_08_slice_axis_kept():
    src = validate_grid((30, 30, 7), _affine((0.7, 0.7, 5.5), rot_deg=10))
    dst = inplane_target_grid(src)
    assert dst.shape[2] == 7
    assert np.allclose(dst.affine[:3, 2], src.affine[:3, 2], atol=1e-12)
    data = np.random.default_rng(2).random(src.shape)
    out = resample(data, src, dst, order=1)
    assert out.shape[2] == src.shape[2]


# ---------------------------------------------------------------- 9-10 interpolation policy
def test_09_dwi_adc_gt_brain_share_one_grid():
    case, _ = _prep(spacing=(0.8, 0.9, 5.0), rot=15)
    assert case.image.shape == (2, *case.model_grid.shape)
    assert case.label.shape == case.brain.shape == case.model_grid.shape
    r = case.record
    assert (
        r["preprocess_policy"] == PREPROCESS_POLICY and r["brain_mask_policy"] == BRAIN_MASK_POLICY
    )
    assert r["interpolation"] == {
        "dwi": "linear",
        "adc": "linear",
        "gt": "nearest",
        "brain_mask": "nearest",
    }
    assert r["brain_mask_grid"] == "original" and r["slice_axis_resampled"] is False
    assert case.original_brain.shape == case.original_grid.shape


def test_10_gt_and_brain_nearest_binary_and_bad_labels_refused():
    case, (dwi, adc, gt, aff) = _prep(spacing=(0.75, 0.75, 5.0))
    for m in (case.label, case.brain):
        assert m.dtype == bool
    assert case.label.any()
    bad = gt.copy()
    bad[0, 0, 0] = 2
    with pytest.raises(PreprocessError):
        prepare_arrays(dwi, adc, bad, {"dwi": aff, "adc": aff, "gt": aff})
    shifted = aff.copy()
    shifted[0, 3] += 0.5
    with pytest.raises(PreprocessError):
        prepare_arrays(dwi, adc, gt, {"dwi": aff, "adc": shifted, "gt": aff})
    with pytest.raises(PreprocessError):
        prepare_arrays(dwi, adc[:-1], gt, {"dwi": aff, "adc": aff, "gt": aff})


# ---------------------------------------------------------------- 11-13 inverse to original grid
def test_11_probability_restored_to_original_grid():
    case, (dwi, *_rest) = _prep(spacing=(0.8, 1.2, 5.0), rot=20)
    prob = np.random.default_rng(3).random(case.model_grid.shape)
    out = restore_probability(prob, case.model_grid, case.original_grid)
    assert out.shape == dwi.shape and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    res = build_outputs(prob, case, 0.5)
    assert np.array_equal(res.ungated, (res.probability >= np.float32(0.5)).astype(np.uint8))
    assert res.record["threshold_applied_on"] == "original_grid"
    assert case.original_grid.shape != case.model_grid.shape
    with pytest.raises(OutputGridError):  # probability must be on the model grid
        build_outputs(np.zeros(case.original_grid.shape), case, 0.5)
    with pytest.raises(OutputGridError):  # out-of-range probability is refused, not clipped
        build_outputs(np.full(case.model_grid.shape, 1.5), case, 0.5)


def test_12_shape_affine_orientation_spacing_restored_and_checked():
    case, (_dwi, _adc, _gt, aff) = _prep(spacing=(0.9, 0.9, 5.0), rot=30)
    res = build_outputs(np.full(case.model_grid.shape, 0.2), case, 0.5)
    for arr in (res.probability, res.ungated, res.brain_mask, res.gated):
        assert check_on_grid(arr, res.grid, case.original_grid) == []
        assert arr.shape == case.original_grid.shape
    assert np.allclose(res.grid.affine, aff) and res.grid.axcodes == case.original_grid.axcodes
    flipped = aff.copy()
    flipped[:3, 0] *= -1
    bad = Grid(case.original_grid.shape, flipped)
    assert "orientation_mismatch" in check_on_grid(res.ungated, bad, case.original_grid)
    wrong_sp = aff.copy()
    wrong_sp[:3, 1] *= 1.5
    assert "spacing_mismatch" in check_on_grid(
        res.ungated, Grid(case.original_grid.shape, wrong_sp), case.original_grid
    )
    for m in (res.ungated, res.brain_mask, res.gated):
        assert set(np.unique(m).tolist()) <= {0, 1}


@pytest.mark.parametrize(
    "spacing",
    [(1.0, 1.0, 5.0), (2.0, 2.0, 5.0), (1.5, 1.25, 4.0), (1.3, 1.7, 4.0), (1.1, 1.1, 4.0)],
)
def test_13_small_components_survive_round_trip(spacing):
    from neuroscan_ai.component_analysis.extraction import extract_components

    src = validate_grid((16, 14, 3), _affine(spacing, rot_deg=12))
    dst = inplane_target_grid(src)
    gt = np.zeros(src.shape, np.uint8)
    gt[4, 5, 1] = 1  # single voxel
    gt[10:12, 9:11, 2] = 1  # 2x2
    model_gt = resample_mask(gt, src, dst)
    # nearest round trip of a mask is exact when the original spacing is >= 1 mm
    assert np.array_equal(restore_mask(model_gt, dst, src), gt == 1)
    # pipeline path: a perfect model-grid prediction of the resampled GT, restored linearly and
    # thresholded on the original grid, keeps every small component
    prob_back = restore_probability(model_gt.astype(np.float64), dst, src)
    assert np.array_equal(prob_back >= 0.5, gt == 1)
    assert extract_components((prob_back >= 0.5).astype(np.uint8))[1] == 2


# ---------------------------------------------------------------- 14-15 determinism, originals
def test_14_determinism():
    a, _ = _prep(spacing=(0.8, 0.9, 5.0), rot=17)
    b, _ = _prep(spacing=(0.8, 0.9, 5.0), rot=17)
    assert a.image.tobytes() == b.image.tobytes()
    assert np.array_equal(a.label, b.label) and np.array_equal(a.brain, b.brain)
    assert a.record == b.record
    m1, m2 = build_model(7).eval(), build_model(7).eval()
    x = torch.from_numpy(a.image[:, :, :, 0][None, :, :40, :32].copy())
    with torch.no_grad():
        assert torch.equal(m1(x), m2(x))
    p = np.random.default_rng(4).random(a.model_grid.shape)
    r1, r2 = build_outputs(p, a, 0.5), build_outputs(p, b, 0.5)
    assert r1.probability.tobytes() == r2.probability.tobytes()


def test_15_original_arrays_and_files_unchanged(tmp_path):
    import hashlib

    import nibabel as nib

    dwi, adc, gt = _volume((30, 28, 4), seed=5)
    aff = _affine((0.8, 0.8, 5.0), rot_deg=8)
    copies = [x.copy() for x in (dwi, adc, gt)]
    aff_copy = aff.copy()
    case = prepare_arrays(dwi, adc, gt, {"dwi": aff, "adc": aff, "gt": aff})
    build_outputs(np.full(case.model_grid.shape, 0.7), case, 0.5)
    for x, c in zip((dwi, adc, gt), copies, strict=True):
        assert np.array_equal(x, c)
    assert np.array_equal(aff, aff_copy)
    paths = []
    for name, arr in (("dwi", dwi), ("adc", adc), ("gt", gt)):
        p = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(arr, aff), str(p))
        paths.append(p)
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    loaded = load_and_prepare(*paths)
    assert loaded.model_grid.shape == case.model_grid.shape
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == before
    assert sorted(x.name for x in tmp_path.iterdir()) == ["adc.nii.gz", "dwi.nii.gz", "gt.nii.gz"]
