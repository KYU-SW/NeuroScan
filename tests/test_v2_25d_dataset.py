"""v2 2.5D stack, boundary, metadata and sampling (synthetic, CPU)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from neuroscan_ai.v2 import baseline_25d_dataset as bd
from neuroscan_ai.v2 import baseline_dataset as ref2d  # comparison only
from tests.test_v2_25d_runner import FORBIDDEN, TRAIN, VAL, make_world, open_world


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    w = make_world(tmp_path_factory.mktemp("ds25d"))
    train, val = open_world(w)
    return train + val


def _volume(nz, seed=0):
    return np.random.default_rng(seed).standard_normal((2, 8, 8, nz)).astype(np.float32)


def _fake(nz, positive=(), spacing=2.0):
    return SimpleNamespace(
        case_key="case_0001",
        positive_slices=list(positive),
        brain_slices=list(range(nz)),
        model_grid_shape=(8, 8, nz),
        model_grid_spacing=(1.0, 1.0, spacing),
    )


@pytest.mark.parametrize(
    "z,nz,expected",
    [
        (0, 1, [0, 0, 0]),
        (0, 2, [0, 0, 1]),
        (1, 2, [0, 1, 1]),
        (0, 5, [0, 0, 1]),
        (2, 5, [1, 2, 3]),
        (4, 5, [3, 4, 4]),
    ],
)
def test_source_indices(z, nz, expected):
    assert bd.source_slice_indices(z, nz) == expected


def test_source_indices_refuse_out_of_range_no_wraparound():
    for z, nz in ((-1, 3), (3, 3), (0, 0), (True, 3), (1.0, 3)):
        with pytest.raises(bd.StackError):
            bd.source_slice_indices(z, nz)


@pytest.mark.parametrize("nz", [1, 2, 3, 6])
def test_channel_order_and_bitwise_source(nz):
    v = _volume(nz)
    for z in range(nz):
        s = bd.stack_slices(v, z)
        assert s.shape == (6, 8, 8) and s.dtype == np.float32
        for i, src in enumerate(bd.source_slice_indices(z, nz)):
            assert np.array_equal(s[2 * i], v[0, :, :, src])  # DWI
            assert np.array_equal(s[2 * i + 1], v[1, :, :, src])  # ADC, same source index
        assert np.array_equal(s[2:4], v[:, :, :, z])
        assert all(s[c].any() for c in range(6))  # never a zero slice


def test_stack_is_a_copy_and_never_renormalised():
    v = _volume(3) * np.float32(7) + np.float32(5)  # far from z-scored: must come through
    before = v.copy()
    s = bd.stack_slices(v, 1)
    assert np.array_equal(s[2], v[0, :, :, 1])
    s += 1
    assert np.array_equal(v, before)
    with pytest.raises(bd.StackError):
        bd.stack_slices(v[:1], 1)
    with pytest.raises(bd.StackError):
        bd.stack_slices(v.astype(np.float64), 1)


def test_cached_stack_equals_cache_volume_and_2d_input(cases):
    for c in cases:
        dwi, adc = c.store.array(c.case_key, "dwi"), c.store.array(c.case_key, "adc")
        n = c.model_grid_shape[2]
        for z in range(n):
            s = bd.stack_slices(bd.image_view(c), z)
            for i, src in enumerate(bd.source_slice_indices(z, n)):
                assert np.array_equal(s[2 * i], dwi[src]) and np.array_equal(s[2 * i + 1], adc[src])
            a, b = bd.slice_item(c, z), ref2d.slice_item(c, z)
            assert a["image"][2:4].tobytes() == b["image"].tobytes()
            for k in ("target", "brain_mask", "valid_region_mask"):
                assert a[k].tobytes() == b[k].tobytes()


def test_training_and_inference_use_the_same_stack(cases, monkeypatch):
    c = cases[0]
    view = bd.NeighborStackView(bd.image_view(c))
    assert view.shape == (6, *c.model_grid_shape)
    h, w = c.model_grid_shape[:2]
    for z in range(c.model_grid_shape[2]):
        assert np.array_equal(view[:, :, :, z], bd.slice_item(c, z)["image"][:, :h, :w])
    with pytest.raises(TypeError):
        view[:, :, :, 0:2]
    calls = []
    real = bd.stack_slices
    monkeypatch.setattr(bd, "stack_slices", lambda v, z: (calls.append(z), real(v, z))[1])
    bd.slice_item(c, 1)
    _ = view[:, :, :, 2]
    assert calls == [1, 2]


def test_at_most_two_active_cases_and_read_only(cases):
    store = cases[0].store
    for c in cases:
        bd.slice_item(c, 1)
    assert store.peak_active_cases <= 2
    assert not store.array(cases[0].case_key, "dwi").flags.writeable


def test_metadata_classes():
    c = _fake(6, positive=[2, 3])
    m = [bd.sample_metadata(c, z) for z in range(6)]
    assert [x["center_positive"] for x in m] == [False, False, True, True, False, False]
    assert [x["neighbor_only_positive"] for x in m] == [False, True, False, False, True, False]
    assert [x["positive_neighbor_count"] for x in m] == [0, 1, 1, 1, 1, 0]
    assert [x["edge_repeat_applied"] for x in m] == [True, False, False, False, False, True]
    assert tuple(m[0]) == bd.METADATA_FIELDS


def test_repeated_edge_is_not_a_positive_neighbour():
    m = bd.sample_metadata(_fake(1, positive=[0]), 0)
    assert m["center_positive"] and m["positive_neighbor_count"] == 0
    m = bd.sample_metadata(_fake(3, positive=[0]), 0)
    assert m["positive_neighbor_count"] == 0 and m["source_slice_indices"] == [0, 0, 1]
    both = bd.sample_metadata(_fake(3, positive=[0, 2]), 1)
    assert both["neighbor_only_positive"] and both["positive_neighbor_count"] == 2


def test_thick_subgroup_from_spacing():
    for mm, g in ((2.0, "thin"), (3.0, "thin"), (4.8, "thick")):
        m = bd.sample_metadata(_fake(3, spacing=mm), 1)
        assert m["thick_slice_subgroup"] == g and m["z_spacing_mm"] == mm


def test_neighbor_only_stays_negative_and_positive_matches_gt(cases):
    pos, neg = bd.slice_index(cases)
    for ci, z in pos + neg:
        gt = bool(cases[ci].slice_arrays(z)[1].any())
        meta = bd.sample_metadata(cases[ci], z)
        assert meta["center_positive"] == gt == ((ci, z) in pos)
    only = [(ci, z) for ci, z in neg if bd.sample_metadata(cases[ci], z)["neighbor_only_positive"]]
    assert only and not set(only) & set(pos)
    for ci, z in only:
        assert not bd.slice_item(cases[ci], z)["target"].any()


def test_sampling_deterministic_seed_dependent_and_equal_to_2d_order(cases):
    o1, r1 = bd.epoch_sampling(cases, 42, 3)
    o2, r2 = bd.epoch_sampling(cases, 42, 3)
    assert o1 == o2 and r1 == r2
    for seed, epoch in ((43, 3), (42, 4)):
        o, r = bd.epoch_sampling(cases, seed, epoch)
        assert o != o1 and r["slice_list_sha256"] != r1["slice_list_sha256"]
        assert r["source_list_sha256"] != r1["source_list_sha256"]
    o2d, r2d = ref2d.epoch_sampling(cases, 42, 3)
    assert o2d == o1 and {k: r1[k] for k in r2d} == r2d
    a = next(bd.batches(cases, o1, 4))
    b = next(bd.batches(cases, o2, 4))
    assert a["image"].numpy().tobytes() == b["image"].numpy().tobytes()
    assert a["metadata"] == b["metadata"]
    assert r1["sampled_neighbor_only_positive"] == sum(
        bd.sample_metadata(cases[ci], z)["neighbor_only_positive"] for ci, z in o1
    )


def test_positive_fraction_counts_centre_only(cases):
    order, rec = bd.epoch_sampling(cases, 1, 1)
    n = sum(bd.sample_metadata(cases[ci], z)["center_positive"] for ci, z in order)
    assert n == rec["sampled_positive"] == round(0.5 * len(order))
    with pytest.raises(bd.SamplingError):
        bd.epoch_sampling([_fake(3)], 1, 1)


def test_padding_and_valid_region(cases):
    b = bd.collate([bd.slice_item(c, 1) for c in cases])
    h, w = b["image"].shape[-2:]
    assert h % 8 == w % 8 == 0 and b["image"].shape[1] == 6
    for i, c in enumerate(cases):
        hh, ww = c.model_grid_shape[:2]
        v = b["valid_region_mask"][i, 0]
        assert v[:hh, :ww].all() and not v[hh:].any() and not v[:, ww:].any()
        assert not b["image"][i, :, hh:].any() and not b["image"][i, :, :, ww:].any()


def test_batch_metadata_has_no_identifiers(cases):
    b = bd.collate([bd.slice_item(c, 1) for c in cases])
    text = json.dumps([b["metadata"], b["case_key"], bd.slice_table(cases)])
    assert not any(n in text for n in (*TRAIN, *VAL, *FORBIDDEN, ":\\\\", "/home/"))
