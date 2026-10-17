"""Metric engine on synthetic predictions (scenarios A-H) and report aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from neuroscan_ai.evaluation.report import results_dir, summarize
from neuroscan_ai.evaluation.segmentation import binarize_prediction, evaluate_case

SP = (1.0, 1.0, 2.0)  # 2 mm^3 voxels


def cube(shape=(20, 20, 10), lo=(5, 5, 3), hi=(10, 10, 6)) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    m[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = 1
    return m


GT = cube()  # 5*5*3 = 75 voxels = 0.15 mL


def test_A_perfect_prediction():
    r = evaluate_case(GT, GT, SP, "A")
    assert r["dice"] == r["iou"] == r["sensitivity"] == r["precision"] == 1.0
    assert r["volume_error_ml"] == 0 and r["absolute_volume_error_ml"] == 0
    assert r["relative_volume_error_percent"] == 0
    assert r["negative_outcome"] is None


def test_B_empty_prediction_against_positive_gt():
    r = evaluate_case(np.zeros_like(GT), GT, SP)
    assert r["dice"] == 0 and r["iou"] == 0 and r["sensitivity"] == 0
    assert r["precision"] is None  # 0/0: undefined, not silently 0 or 1
    assert r["volume_error_ml"] == pytest.approx(-0.15)
    assert r["relative_volume_error_percent"] == pytest.approx(-100.0)


def test_C_prediction_larger_than_gt():
    big = cube(lo=(4, 4, 2), hi=(11, 11, 7))  # 7*7*5 = 245 voxels, contains GT
    r = evaluate_case(big, GT, SP)
    assert r["sensitivity"] == 1.0
    assert r["precision"] == pytest.approx(75 / 245)
    assert r["volume_error_ml"] == pytest.approx((245 - 75) * 2 / 1000)
    assert r["volume_error_ml"] > 0


def test_D_prediction_smaller_than_gt():
    small = cube(lo=(6, 6, 4), hi=(9, 9, 5))  # 3*3*1 = 9 voxels inside GT
    r = evaluate_case(small, GT, SP)
    assert r["precision"] == 1.0
    assert r["sensitivity"] == pytest.approx(9 / 75)
    assert r["volume_error_ml"] == pytest.approx(-(75 - 9) * 2 / 1000)
    assert r["absolute_volume_error_ml"] == pytest.approx((75 - 9) * 2 / 1000)


def test_E_partial_overlap():
    shifted = cube(lo=(7, 5, 3), hi=(12, 10, 6))  # shift x by 2: overlap 3*5*3 = 45
    r = evaluate_case(shifted, GT, SP)
    tp, fp, fn = 45, 30, 30
    assert (r["tp_voxels"], r["fp_voxels"], r["fn_voxels"]) == (tp, fp, fn)
    assert r["dice"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert r["iou"] == pytest.approx(tp / (tp + fp + fn))
    assert r["dice"] == pytest.approx(2 * r["iou"] / (1 + r["iou"]))


def test_F_negative_gt_empty_prediction_is_correct_negative():
    r = evaluate_case(np.zeros_like(GT), np.zeros_like(GT), SP, "neg")
    assert r["has_lesion"] is False
    assert r["negative_outcome"] == "correct_negative"
    assert r["any_false_positive"] is False
    assert r["predicted_lesion_voxels"] == 0 and r["false_positive_volume_ml"] == 0
    for m in ("dice", "iou", "sensitivity", "precision", "relative_volume_error_percent"):
        assert r[m] is None


def test_G_negative_gt_positive_prediction_is_false_positive():
    fp = cube(lo=(0, 0, 0), hi=(2, 3, 1))  # 6 voxels
    r = evaluate_case(fp, np.zeros_like(GT), SP)
    assert r["negative_outcome"] == "false_positive_case"
    assert r["any_false_positive"] is True
    assert r["predicted_lesion_voxels"] == 6
    assert r["false_positive_volume_ml"] == pytest.approx(6 * 2 / 1000)
    assert r["false_positive_component_count"] == 1
    assert r["dice"] is None


def test_H_tiny_lesion_below_1ml():
    gt = np.zeros((10, 10, 10), np.uint8)
    gt[4, 4, 4] = 1  # single voxel, 0.002 mL
    pred = gt.copy()
    pred[4, 5, 4] = 1
    r = evaluate_case(pred, gt, SP)
    assert r["gt_volume_ml"] == pytest.approx(0.002)
    assert r["dice"] == pytest.approx(2 / 3)
    for k, v in r.items():
        assert not (isinstance(v, float) and np.isnan(v)), k


def test_probability_threshold_is_configurable():
    prob = np.array([0.2, 0.5, 0.7])
    assert binarize_prediction(prob).tolist() == [0, 1, 1]
    assert binarize_prediction(prob, threshold=0.6).tolist() == [0, 0, 1]


def test_inputs_validated():
    with pytest.raises(ValueError, match="binary"):
        evaluate_case(np.full(GT.shape, 0.4), GT, SP)
    with pytest.raises(ValueError, match="shape"):
        evaluate_case(GT[:-1], GT, SP)
    with pytest.raises(ValueError, match="spacing"):
        evaluate_case(GT, GT, (1.0, 0.0, 2.0))


def test_report_separates_positive_and_negative_patients():
    rows = [
        {**evaluate_case(GT, GT, SP, "p1"), "spacing_group": "thin"},
        {**evaluate_case(np.zeros_like(GT), GT, SP, "p2"), "spacing_group": "thick"},
        {**evaluate_case(np.zeros_like(GT), np.zeros_like(GT), SP, "n1"), "spacing_group": "thin"},
        {**evaluate_case(GT, np.zeros_like(GT), SP, "n2"), "spacing_group": "thin"},
    ]
    s = summarize(rows, subgroup_keys=["spacing_group"])
    assert s["positive"]["n_cases"] == 2
    assert s["positive"]["dice"]["mean"] == pytest.approx(0.5)  # negatives not mixed in
    assert s["positive"]["precision"]["n"] == 1 and s["positive"]["precision"]["n_undefined"] == 1
    assert s["negative"]["n_cases"] == 2
    assert s["negative"]["correct_negative"] == 1 and s["negative"]["false_positive_cases"] == 1
    assert s["subgroups"]["spacing_group"]["thin"]["positive"]["dice"]["mean"] == 1.0
    assert s["subgroups"]["spacing_group"]["thick"]["negative"]["n_cases"] == 0


def test_results_dir_scopes(tmp_path):
    assert results_dir(tmp_path, "internal_test", "run1") == tmp_path / "internal_test" / "run1"
    assert results_dir(tmp_path, "external_validation", "ds", "r") == (
        tmp_path / "external_validation" / "ds" / "r"
    )
    with pytest.raises(ValueError):
        results_dir(tmp_path, "test", "run1")
