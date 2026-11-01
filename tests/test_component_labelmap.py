"""Phase 7-A: component display labelmap — one labelling yields the JSON components and the
uint8 labelmap; nothing is removed, merged or changed in the prediction mask."""

from __future__ import annotations

import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from neuroscan_ai.analysis.components import (
    OTHER_DISPLAY_LABEL,
    check_labelmap,
    component_analysis,
    components,
)
from neuroscan_ai.inference import run_independent_inference
from tests.inference_helpers import tiny_model_setup, write_case

SPACING = (1.0, 1.0, 2.0)


def _ring() -> np.ndarray:
    """Hollow square ring (one 26-connected component whose centroid is NOT inside it), a block
    and two single voxels."""
    m = np.zeros((30, 30, 12), dtype=np.uint8)
    m[5:15, 5:15, 4:6] = 1
    m[7:13, 7:13, 4:6] = 0  # hole: centroid (9.5, 9.5, 4.5) is background
    m[20:26, 20:24, 2:9] = 1
    m[2, 27, 1] = 1
    m[27, 2, 10] = 1
    return m


def test_labelmap_matches_json_and_mask():
    mask = _ring()
    before = mask.copy()
    labelmap, comps = component_analysis(mask, SPACING, None, slice_axis=2)
    assert np.array_equal(mask, before)  # input untouched
    assert labelmap.dtype == np.uint8 and labelmap.shape == mask.shape
    assert np.array_equal(labelmap > 0, mask > 0)
    assert int((labelmap > 0).sum()) == int(mask.sum())
    assert [c["size_rank"] for c in comps] == [1, 2, 3, 4]
    assert [c["display_label"] for c in comps] == [1, 2, 3, 4]
    for c in comps:
        assert int((labelmap == c["display_label"]).sum()) == c["voxel_count"]
    assert check_labelmap(labelmap, mask, comps) == []
    # the original component fields are unchanged by the extension
    plain = components(mask, SPACING)
    for a, b in zip(plain, comps, strict=True):
        assert {k: b[k] for k in a} == a


def test_representative_voxel_is_inside_even_when_the_centroid_is_not():
    mask = _ring()
    labelmap, comps = component_analysis(mask, SPACING, None, slice_axis=2)
    ring = next(c for c in comps if c["voxel_count"] == 10 * 10 * 2 - 6 * 6 * 2)
    centroid = np.round(ring["centroid_voxel"]).astype(int)
    assert mask[tuple(centroid)] == 0  # the centroid lies in the hole
    for c in comps:
        assert mask[tuple(c["representative_voxel"])] == 1
        assert labelmap[tuple(c["representative_voxel"])] == c["display_label"]


def test_slice_range_and_single_voxel_components():
    mask = _ring()
    _, comps = component_analysis(mask, SPACING, None, slice_axis=2)
    by_size = {c["voxel_count"]: c for c in comps}
    assert by_size[6 * 4 * 7]["slice_range"] == {"axis": 2, "first": 2, "last": 8}
    singles = [c for c in comps if c["voxel_count"] == 1]
    assert len(singles) == 2
    for c in singles:
        assert c["slice_range"]["first"] == c["slice_range"]["last"]
        assert c["representative_voxel"] == [int(v) for v in c["bbox_min"]]
    _, comps0 = component_analysis(mask, SPACING, None, slice_axis=0)
    assert next(c for c in comps0 if c["voxel_count"] == 168)["slice_range"] == {
        "axis": 0,
        "first": 20,
        "last": 25,
    }


def _isolated_voxels(n: int) -> np.ndarray:
    """n single-voxel components (26-connectivity needs a gap of one voxel) of growing size:
    component i gets i % 3 + 1 voxels along z, so sizes are not all equal."""
    m = np.zeros((40, 40, 8), dtype=np.uint8)
    for i in range(n):
        x, y = 2 * (i % 20), 2 * (i // 20)
        m[x, y, 0 : (i % 3) + 1] = 1
    return m


@pytest.mark.parametrize("n", [254, 300])
def test_up_to_254_labels_then_reserved_other(n):
    mask = _isolated_voxels(n)
    labelmap, comps = component_analysis(mask, SPACING, None, slice_axis=2)
    assert len(comps) == n  # every component kept in the JSON, nothing merged away
    assert int((labelmap > 0).sum()) == int(mask.sum())
    assert check_labelmap(labelmap, mask, comps) == []
    shown = sorted(set(np.unique(labelmap).tolist()) - {0})
    if n == 254:
        assert shown == list(range(1, 255)) and OTHER_DISPLAY_LABEL not in shown
    else:
        assert shown == [*range(1, 255), OTHER_DISPLAY_LABEL]
        other = [c for c in comps if c["display_label"] == OTHER_DISPLAY_LABEL]
        assert len(other) == n - 254 and all(c["size_rank"] > 254 for c in other)
        assert int((labelmap == 255).sum()) == sum(c["voxel_count"] for c in other)
    # ranks follow size (largest first)
    sizes = [c["voxel_count"] for c in comps]
    assert sizes == sorted(sizes, reverse=True)


def test_oblique_affine_world_positions():
    a, z = np.radians(31.0), np.radians(15.0)
    rot = np.array(
        [
            [np.cos(z), -np.sin(z) * np.cos(a), np.sin(z) * np.sin(a)],
            [np.sin(z), np.cos(z) * np.cos(a), -np.cos(z) * np.sin(a)],
            [0, np.sin(a), np.cos(a)],
        ]
    )
    aff = np.eye(4)
    aff[:3, :3] = rot @ np.diag([-1.8, 1.8, 2.5])
    aff[:3, 3] = [101.2, -88.4, -40.7]
    mask = _ring()
    _, comps = component_analysis(mask, (1.8, 1.8, 2.5), aff, slice_axis=2)
    inv = np.linalg.inv(aff)
    for c in comps:
        rep = np.array(c["representative_voxel"])
        mm = np.array(c["representative_world_mm"])
        np.testing.assert_allclose(mm, (aff @ [*rep, 1])[:3], atol=1e-9)
        np.testing.assert_allclose((inv @ [*mm, 1])[:3], rep, atol=1e-9)


def test_check_labelmap_detects_inconsistencies():
    mask = _ring()
    labelmap, comps = component_analysis(mask, SPACING, None, slice_axis=2)
    moved = labelmap.copy()
    moved[0, 0, 0] = 1
    assert "labelmap foreground differs from mask" in check_labelmap(moved, mask, comps)
    swapped = labelmap.copy()
    swapped[labelmap == 1] = 2
    assert check_labelmap(swapped, mask, comps)
    assert check_labelmap(labelmap.astype(np.int16), mask, comps) == ["labelmap shape/dtype"]


def test_engine_writes_a_consistent_labelmap_and_keeps_the_mask(tmp_path):
    d, a, _, aff = write_case(tmp_path / "in")
    inf, *_ = tiny_model_setup(tmp_path, bias=50.0)
    r = run_independent_inference(d, a, tmp_path / "out", inference_config_path=inf, device="cpu")
    assert r.status == "succeeded"
    run_dir = tmp_path / "out" / r.case_id / r.run_id
    mask_img = nib.load(run_dir / "prediction_mask.nii.gz")
    comp_img = nib.load(run_dir / "prediction_components.nii.gz")
    mask = np.asanyarray(mask_img.dataobj)
    labelmap = np.asanyarray(comp_img.dataobj)
    assert labelmap.dtype == np.uint8 and labelmap.shape == mask.shape
    np.testing.assert_allclose(comp_img.affine, mask_img.affine, atol=1e-6)
    assert nib.aff2axcodes(comp_img.affine) == nib.aff2axcodes(mask_img.affine)
    quant = json.loads((run_dir / "quantification.json").read_text(encoding="utf-8"))
    assert check_labelmap(labelmap, mask, quant["components"]) == []
    assert int((labelmap > 0).sum()) == int(mask.sum()) == quant["voxel_count"]
    meta = json.loads((run_dir / "inference_metadata.json").read_text(encoding="utf-8"))
    files = meta["outputs"]["files"]
    for rel in ("prediction_mask.nii.gz", "prediction_components.nii.gz"):
        assert files[rel] == hashlib.sha256((run_dir / rel).read_bytes()).hexdigest()
    # recomputing from the saved mask gives the same labelmap (deterministic, mask unchanged)
    again, _ = component_analysis(mask, nib.affines.voxel_sizes(mask_img.affine), None, 2)
    assert np.array_equal(again, labelmap)
