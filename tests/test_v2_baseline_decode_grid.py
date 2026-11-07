"""v2 baseline: GT decode policy ``binary_mask_decode_v1_atol_1e_6``, target-size integer snap and
the ISLES thick-slice subgroup. Synthetic arrays only (CPU); no MRI, no ISLES / SOOP access."""

from __future__ import annotations

import hashlib
import math

import nibabel as nib
import numpy as np
import pytest

from neuroscan_ai.v2.baseline_preprocess import PreprocessError, load_and_prepare, prepare_arrays
from neuroscan_ai.v2.geometry import (
    BINARY_DECODE_POLICY,
    BinaryDecodeError,
    decode_binary_mask,
    inplane_target_grid,
    inplane_target_sizes,
    validate_grid,
)
from neuroscan_ai.v2.safety_metrics import SLICE_SUBGROUP_POLICY, case_metrics, slice_subgroup
from tests.test_v2_baseline_geometry import _affine, _volume

SCALED_ONE = 0.9999999997671694  # int16 + scl_slope/inter decoding seen in ISLES GT


def _gt_with(value, shape=(6, 6, 3)):
    a = np.zeros(shape)
    a[2:4, 2:4, 1] = 1.0
    a[2, 2, 1] = value
    return a


# ---------------------------------------------------------------- GT decode 1-14
def test_gt01_exact_binary_accepted():
    m, r = decode_binary_mask(_gt_with(1.0).astype(np.uint8))
    assert m.dtype == bool and m.sum() == 4
    assert r["policy"] == BINARY_DECODE_POLICY == "binary_mask_decode_v1_atol_1e_6"
    assert r["snapped_voxels"] == 0 and r["max_abs_correction"] == 0.0
    assert r["rejected_value_count"] == 0 and r["tolerance_abs"] == 1e-6


def test_gt02_scaled_one_snapped_to_one():
    m, r = decode_binary_mask(_gt_with(SCALED_ONE))
    assert m[2, 2, 1] and m.sum() == 4
    assert r["snapped_voxels"] == 1 and r["snapped_value_kinds"] == [SCALED_ONE]


@pytest.mark.parametrize(
    "value,expected",
    [(1 + 5e-7, True), (1 - 5e-7, True), (0 + 5e-7, False), (-5e-7, False)],
    ids=["gt03_one_plus", "gt04_one_minus", "gt05_zero_plus", "gt06_zero_minus"],
)
def test_gt03_06_within_tolerance_snapped(value, expected):
    m, r = decode_binary_mask(_gt_with(value))
    assert bool(m[2, 2, 1]) is expected
    assert r["snapped_voxels"] == 1 and r["max_abs_correction"] == pytest.approx(5e-7, rel=1e-6)


@pytest.mark.parametrize(
    "value",
    [1 + 2e-6, 0.999, 0.6, -1.0, 2.0],
    ids=["gt07_one_plus_2e-6", "gt08_0.999", "gt09_0.6", "gt10_negative_label", "gt10_label_2"],
)
def test_gt07_10_out_of_tolerance_refused(value):
    with pytest.raises(BinaryDecodeError) as exc:
        decode_binary_mask(_gt_with(value))
    assert exc.value.record["rejected_value_count"] == 1
    with pytest.raises(PreprocessError):
        dwi, adc, _ = _volume((6, 6, 3))
        aff = _affine((1.0, 1.0, 5.0))
        prepare_arrays(dwi, adc, _gt_with(value), {"dwi": aff, "adc": aff, "gt": aff})


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_gt11_non_finite_refused(value):
    with pytest.raises(BinaryDecodeError) as exc:
        decode_binary_mask(_gt_with(value))
    assert exc.value.record["non_finite_count"] == 1


def test_gt12_nifti_scaled_values_and_gt13_inputs_unchanged(tmp_path):
    dwi, adc, gt = _volume((30, 28, 4), seed=3)
    aff = _affine((2.0, 2.0, 4.8), rot_deg=12)
    paths = {}
    for name, arr in (("dwi", dwi), ("adc", adc)):
        paths[name] = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(arr, aff), str(paths[name]))
    img = nib.Nifti1Image(gt.astype(np.float32), aff)
    img.set_data_dtype(np.int16)  # nibabel picks slope/inter -> decoded 0.9999999997671694
    paths["gt"] = tmp_path / "gt.nii.gz"
    nib.save(img, str(paths["gt"]))
    decoded = np.asanyarray(nib.load(str(paths["gt"])).dataobj)
    assert np.unique(decoded).tolist() == [0.0, SCALED_ONE]  # same decoding as the ISLES GT
    gt_copy = gt.copy()
    before = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    case = load_and_prepare(paths["dwi"], paths["adc"], paths["gt"])
    rec = case.record["gt_decode"]
    assert rec["rejected_value_count"] == 0 and rec["snapped_voxels"] > 0
    assert int(case.record["gt_voxels_original"]) == int(gt.sum())
    assert {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()} == before
    assert np.array_equal(gt, gt_copy)
    arr = decoded.copy()
    decode_binary_mask(arr)
    assert np.array_equal(arr, decoded)  # decode never modifies its input


def test_gt14_correction_count_and_max_recorded():
    a = _gt_with(1.0)
    a[2, 3, 1] = 1 - 3e-7
    a[0, 0, 0] = 8e-7
    a[5, 5, 2] = SCALED_ONE
    m, r = decode_binary_mask(a)
    assert r["snapped_voxels"] == 3
    assert r["max_abs_correction"] == pytest.approx(8e-7, rel=1e-6)
    assert m.sum() == 5 and not m[0, 0, 0]
    assert len(r["raw_decoded_unique_values"]) == r["raw_decoded_unique_count"] == 5


def test_gt_case_metrics_accept_decoded_gt():
    g = _gt_with(SCALED_ONE)
    m = case_metrics(g >= 0.5, g, np.eye(4))
    assert m["dice"] == 1.0


# ---------------------------------------------------------------- target size 1-12
def _size(n, sp):
    return validate_grid((n, n, 5), _affine((sp, sp, 5.0)))


def test_grid01_two_mm_112_to_224():
    assert inplane_target_grid(_size(112, 2.0)).shape[:2] == (224, 224)


def test_grid02_header_noise_snapped():
    g = validate_grid((112, 112, 73), _affine((2.00000001, 2.00000006, 2.0)))
    rec = inplane_target_sizes(g)
    assert inplane_target_grid(g).shape == (224, 224, 73)
    assert all(r["integer_snap_applied"] for r in rec)
    assert rec[1]["raw_target_size"] > 224 and rec[1]["snap_changed_size_vs_ceil"]
    assert abs(rec[1]["coverage_difference_mm"]) < 1e-4


def test_grid03_meaningfully_non_integer_uses_ceil():
    for n, sp in ((23, 0.9), (15, 1.1), (112, 2.0005), (100, 1.7969)):
        r = inplane_target_sizes(_size(n, sp))[0]
        assert not r["integer_snap_applied"]
        assert r["final_target_size"] == math.ceil(n * sp)
        assert r["coverage_difference_mm"] > 0  # field of view never cut


def test_grid04_sub_millimetre_0875():
    g = validate_grid((256, 256, 73), _affine((0.87499998, 0.87500003, 2.0)))
    assert inplane_target_grid(g).shape == (224, 224, 73)


def test_grid05_non_square_spacing():
    g = validate_grid((25, 16, 3), _affine((0.8, 1.25, 6.0)))
    assert inplane_target_grid(g).shape == (20, 20, 3)
    g = validate_grid((23, 15, 3), _affine((0.9, 1.1, 6.0)))
    assert inplane_target_grid(g).shape == (21, 17, 3)


def test_grid06_07_oblique_cosines_and_z_kept():
    src = validate_grid((112, 112, 30), _affine((2.00000006, 1.99999998, 4.8), rot_deg=27))
    dst = inplane_target_grid(src)
    d_src = src.affine[:3, :3] / np.linalg.norm(src.affine[:3, :3], axis=0)
    d_dst = dst.affine[:3, :3] / np.linalg.norm(dst.affine[:3, :3], axis=0)
    assert np.allclose(d_src, d_dst, atol=1e-12)
    assert dst.shape[2] == 30 and np.array_equal(dst.affine[:3, 2], src.affine[:3, 2])
    assert np.allclose(dst.spacing[:2], (1.0, 1.0), atol=1e-12)


def test_grid08_all_arrays_on_one_target_and_09_coverage_recorded():
    dwi, adc, gt = _volume((56, 56, 4), seed=1)
    aff = _affine((2.00000006, 2.0, 4.8), rot_deg=20)
    case = prepare_arrays(dwi, adc, gt, {"dwi": aff, "adc": aff, "gt": aff})
    assert case.model_grid.shape == (112, 112, 4)
    assert case.image.shape[1:] == case.label.shape == case.brain.shape == case.model_grid.shape
    for r in case.record["target_size"]:
        assert set(r) >= {
            "raw_target_size",
            "nearest_integer_size",
            "final_target_size",
            "integer_snap_applied",
            "original_coverage_mm",
            "target_coverage_mm",
            "coverage_difference_mm",
        }


def test_grid10_one_mm_input_not_resampled():
    dwi, adc, gt = _volume((40, 36, 4), seed=2)
    aff = _affine((1.0, 1.0, 5.0), rot_deg=9)
    case = prepare_arrays(dwi, adc, gt, {"dwi": aff, "adc": aff, "gt": aff})
    assert case.record["resampled"] is False and case.model_grid.shape == (40, 36, 4)
    assert all(r["integer_snap_applied"] for r in case.record["target_size"])


def test_grid11_deterministic():
    g = validate_grid((97, 61, 7), _affine((1.7969, 2.00000006, 4.8), rot_deg=7))
    a, b = inplane_target_grid(g), inplane_target_grid(g)
    assert a.shape == b.shape and np.array_equal(a.affine, b.affine)
    assert inplane_target_sizes(g) == inplane_target_sizes(g)


# ---------------------------------------------------------------- thick subgroup
def test_thick_subgroup_separate_from_soop_warning():
    s48 = slice_subgroup(4.8)
    assert s48["policy"] == SLICE_SUBGROUP_POLICY == "isles_v1_qa_thick_subgroup_3mm"
    warn = "component_safety_provisional_thick_warning"
    assert s48["slice_group"] == "thick" and s48[warn] is False
    assert slice_subgroup(3.0)["slice_group"] == "thin"
    assert slice_subgroup(3.0001)["slice_group"] == "thick"
    assert slice_subgroup(5.5)[warn] is True
    assert slice_subgroup(2.0, slice_thickness_mm=5.0)[warn] is True
    m = case_metrics(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)), np.diag([1.0, 1.0, 4.8, 1.0]))
    assert m["slice_group"] == "thick" and m["slice_subgroup_policy"] == SLICE_SUBGROUP_POLICY


# ---------------------------------------------------------------- config <-> code policy check
def _cfg():
    import yaml

    from tests.test_v2_baseline_safety import REPO

    return yaml.safe_load((REPO / "configs/v2/2d_baseline.yaml").read_text(encoding="utf-8"))


def test_config_policies_match_code():
    from neuroscan_ai.v2.baseline_preprocess import policy_config_errors, validate_policy_config

    cfg = _cfg()
    assert policy_config_errors(cfg) == []
    validate_policy_config(cfg)
    thick = cfg["safety_metrics"]["thick_slice_policies"]
    assert thick["isles_subgroup"]["policy_id"] == "isles_v1_qa_thick_subgroup_3mm"
    assert thick["soop_provisional_warning"]["policy_id"] == "soop_provisional_thick_warning_5mm"
    gd = cfg["preprocessing"]["gt_decode"]
    assert gd["use_get_unscaled"] is False and gd["use_threshold_binarization"] is False


@pytest.mark.parametrize(
    "path,value",
    [
        (("preprocessing", "gt_decode", "one_atol"), 1e-5),
        (("preprocessing", "gt_decode", "zero_atol"), 1e-7),
        (("preprocessing", "gt_decode", "use_get_unscaled"), True),
        (("preprocessing", "gt_decode", "use_threshold_binarization"), True),
        (("preprocessing", "gt_decode", "decoded_values"), False),
        (("preprocessing", "gt_decode", "policy_id"), "threshold_gt_0p5"),
        (("preprocessing", "target_grid", "size_policy"), ["ceil"]),
        (("preprocessing", "target_grid", "integer_snap_rtol"), 1e-3),
        (("preprocessing", "target_grid", "integer_snap_atol"), 1e-5),
        (("preprocessing", "target_grid", "inplane_spacing_mm"), [1.0, 2.0]),
        (("preprocessing", "target_grid", "preserve_z_spacing"), False),
        (
            (
                "safety_metrics",
                "thick_slice_policies",
                "isles_subgroup",
                "effective_z_spacing_mm_gt",
            ),
            5.0,
        ),
        (
            (
                "safety_metrics",
                "thick_slice_policies",
                "soop_provisional_warning",
                "effective_z_spacing_mm_gt",
            ),
            3.0,
        ),
        (
            (
                "safety_metrics",
                "thick_slice_policies",
                "soop_provisional_warning",
                "slice_thickness_mm_gte",
            ),
            3.0,
        ),
        (
            ("safety_metrics", "thick_slice_policies", "soop_provisional_warning", "provisional"),
            False,
        ),
        (
            ("safety_metrics", "thick_slice_policies", "soop_provisional_warning", "policy_id"),
            "isles_v1_qa_thick_subgroup_3mm",
        ),
    ],
)
def test_config_code_mismatch_refused(path, value):
    from neuroscan_ai.v2.baseline_preprocess import PolicyConfigError, validate_policy_config

    cfg = _cfg()
    node = cfg
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(PolicyConfigError):
        validate_policy_config(cfg)


def test_config_thick_rules_not_merged():
    from neuroscan_ai.v2.baseline_preprocess import PolicyConfigError, validate_policy_config

    cfg = _cfg()
    thick = cfg["safety_metrics"]["thick_slice_policies"]
    thick["isles_subgroup"]["slice_thickness_mm_gte"] = 5.0  # merging warning fields
    with pytest.raises(PolicyConfigError):
        validate_policy_config(cfg)
    cfg = _cfg()
    del cfg["safety_metrics"]["thick_slice_policies"]["soop_provisional_warning"]
    with pytest.raises(PolicyConfigError):
        validate_policy_config(cfg)
    cfg = _cfg()
    del cfg["preprocessing"]["gt_decode"]
    with pytest.raises(PolicyConfigError):
        validate_policy_config(cfg)
