"""Lesion quantification and connected-component analysis."""

from __future__ import annotations

import numpy as np
import pytest

from neuroscan_ai.analysis.components import components, count_components
from neuroscan_ai.analysis.lesion import lesion_summary
from neuroscan_ai.data.case_metadata import adc_unit_group, obliquity_group, volume_group


def two_lesions() -> np.ndarray:
    m = np.zeros((20, 20, 8), np.uint8)
    m[2:6, 3:5, 1:3] = 1  # 4*2*2 = 16 voxels
    m[10, 12, 5] = 1  # single voxel
    return m


def test_lesion_volume_uses_grid_spacing():
    s = lesion_summary(two_lesions(), (2.0, 2.0, 4.8))
    assert s["voxel_count"] == 17
    assert s["volume_ml"] == pytest.approx(17 * 2 * 2 * 4.8 / 1000)


def test_lesion_positive_slices_first_last():
    s = lesion_summary(two_lesions(), (1, 1, 1))
    assert s["positive_slice_count"] == 3  # z = 1, 2, 5
    assert (s["first_slice"], s["last_slice"]) == (1, 5)


def test_empty_mask_summary():
    s = lesion_summary(np.zeros((4, 4, 4)), (1, 1, 1))
    assert s["voxel_count"] == 0 and s["volume_ml"] == 0
    assert s["positive_slice_count"] == 0 and s["first_slice"] is None
    assert s["component_count"] == 0 and s["components"] == []


def test_tiny_lesion_quantified_not_removed():
    m = np.zeros((5, 5, 5), np.uint8)
    m[2, 2, 2] = 1
    s = lesion_summary(m, (1.0, 1.0, 2.0))
    assert s["component_count"] == 1
    assert s["components"][0]["volume_ml"] == pytest.approx(0.002)
    assert m.sum() == 1  # analysis does not modify the mask


def test_component_count_uses_26_connectivity():
    m = np.zeros((4, 4, 4), np.uint8)
    m[0, 0, 0] = m[1, 1, 1] = 1  # corner-touching only
    assert count_components(m) == 1
    m[3, 3, 3] = 0
    m[0, 0, 3] = 1  # separated
    assert count_components(m) == 2


def test_bounding_box_inclusive_and_sorted_by_size():
    comps = components(two_lesions(), (1, 1, 1))
    assert [c["voxel_count"] for c in comps] == [16, 1]
    assert comps[0]["bbox_min"] == [2, 3, 1] and comps[0]["bbox_max"] == [5, 4, 2]
    assert comps[1]["bbox_min"] == comps[1]["bbox_max"] == [10, 12, 5]


def test_centroid_voxel_and_world():
    aff = np.diag([2.0, 2.0, 4.8, 1.0])
    aff[:3, 3] = (-10.0, 5.0, 1.0)
    big = components(two_lesions(), (2.0, 2.0, 4.8), aff)[0]
    assert big["centroid_voxel"] == pytest.approx([3.5, 3.5, 1.5])
    assert big["centroid_world_mm"] == pytest.approx([-10 + 7.0, 5 + 7.0, 1 + 7.2])


def test_subgroup_classifiers():
    assert volume_group(0) == "0 (empty)" and volume_group(0.4) == "(0,1) mL"
    assert volume_group(5) == "[1,10) mL" and volume_group(60) == "[50,inf) mL"
    assert adc_unit_group(0.0009) == "mm2/s"
    assert adc_unit_group(0.9) == "1e-3 mm2/s"
    assert adc_unit_group(920.0) == "1e-6 mm2/s"
    assert obliquity_group(0.2) == "<1 deg" and obliquity_group(25) == ">=20 deg"
