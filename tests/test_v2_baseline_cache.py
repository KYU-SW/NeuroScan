"""Disk-backed cache ``disk_backed_preprocessed_cache_v1`` on SYNTHETIC volumes only (no MRI, no
ISLES / SOOP access). Cache build, integrity gate, memory-mapped access and RAM bounds."""

from __future__ import annotations

import json
import shutil
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from neuroscan_ai.reproducibility import CASE_KEY_POLICY, case_keys
from neuroscan_ai.reproducibility.hashing import (
    alias_list_sha256,
    canonical_json_sha256,
    sha256_file,
)
from neuroscan_ai.v2 import baseline_cache as bc
from neuroscan_ai.v2 import baseline_dataset as bd
from neuroscan_ai.v2 import baseline_inference as bi
from neuroscan_ai.v2.dataset import CaseFiles
from neuroscan_ai.v2.unet_2d import build_model
from tests.test_v2_baseline_runner import FORBIDDEN, REPO, TRAIN, VAL, _write_case

CFG = yaml.safe_load((REPO / "configs/v2/2d_baseline.yaml").read_text(encoding="utf-8"))
LOCK = "a" * 64
COMMIT = "0" * 40


def _inputs(files, keys):
    return {
        keys[a]: {
            "adc_sha256": sha256_file(f.adc),
            "dwi_sha256": sha256_file(f.dwi),
            "gt_sha256": sha256_file(f.label),
        }
        for a, f in files.items()
    }


@pytest.fixture(scope="module")
def c(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache")
    data = root / "data"
    data.mkdir()
    split = data / "split.json"
    split.write_text(json.dumps({"splits": {"train": TRAIN, "val": VAL}}), encoding="utf-8")
    files = {a: _write_case(data, a, i) for i, a in enumerate(TRAIN + VAL)}
    keys = case_keys(TRAIN + VAL)
    out = root / "results/v2_cache/syn_v1"
    summary = bc.build_cache(
        out,
        config=CFG,
        split_file=split,
        files={"train": {a: files[a] for a in TRAIN}, "val": {a: files[a] for a in VAL}},
        git_commit=COMMIT,
        package_lock_sha256_lf=LOCK,
        forbidden_ids=FORBIDDEN,
    )
    ns = SimpleNamespace(root=root, data=data, split=split, files=files, keys=keys, out=out)
    ns.summary = summary
    ns.inputs = _inputs(files, keys)
    return ns


def _verify(c, cache=None, **kw):
    args = dict(
        config=CFG,
        split_file=c.split,
        train_aliases=TRAIN,
        val_aliases=VAL,
        input_sha256=c.inputs,
        package_lock_sha256_lf=LOCK,
        git_commit=COMMIT,
    )
    args.update(kw)
    return bc.verify_cache(cache or c.out, **args)


def _copy(c, tmp_path):
    dst = tmp_path / "cache_copy"
    shutil.copytree(c.out, dst)
    return dst


def _forge(cache: Path, key: str, mutate=None, array=None, data=None):
    """Consistent forgery: change a manifest field and / or an array, then recompute every hash
    and seal (so only the content checks can catch it)."""
    cdir = cache / key
    m = json.loads((cdir / bc.MANIFEST_NAME).read_text(encoding="utf-8"))
    if array is not None:
        p = cdir / f"{array}.npy"
        np.save(p, data, allow_pickle=False)
        m["arrays"][array]["sha256"] = sha256_file(p)
    if mutate is not None:
        mutate(m)
    (cdir / bc.MANIFEST_NAME).write_text(json.dumps(m), encoding="utf-8")
    seal = canonical_json_sha256(m)
    (cdir / bc.MANIFEST_SEAL).write_text(seal, encoding="utf-8")
    idx = json.loads((cache / bc.INDEX_NAME).read_text(encoding="utf-8"))
    idx["cases"][key]["manifest_sha256"] = seal
    (cache / bc.INDEX_NAME).write_text(json.dumps(idx), encoding="utf-8")
    (cache / bc.INDEX_SEAL).write_text(canonical_json_sha256(idx), encoding="utf-8")


def _reasons(fn):
    with pytest.raises(bc.CacheError) as e:
        fn()
    return e.value.reasons


# ---------------------------------------------------------------- structure
def test_cache_structure_dtypes_and_layout(c):
    v = _verify(c)
    assert v.index["counts"] == {"train": len(TRAIN), "val": len(VAL)}
    assert sorted(v.manifests) == sorted(c.keys.values())
    for key, m in v.manifests.items():
        names = {p.name for p in (c.out / key).iterdir()}
        assert names == {f"{n}.npy" for n in bc.ARRAYS} | {bc.MANIFEST_NAME, bc.MANIFEST_SEAL}
        for n, (dtype, grid, _) in bc.ARRAYS.items():
            a = np.load(c.out / key / f"{n}.npy", mmap_mode="r")
            s = m[f"{grid}_grid"]["shape"]
            assert str(a.dtype) == dtype and a.shape == (s[2], s[0], s[1])  # slice-major
        mg, og = m["model_grid"]["spacing_mm_display"], m["original_grid"]["spacing_mm_display"]
        assert mg[:2] == [1.0, 1.0] and mg[2] == og[2]  # 1 mm in-plane, z kept
        assert m["gt_decode_policy"] == "binary_mask_decode_v1_atol_1e_6"
    top = {p.name for p in c.out.iterdir() if not p.name.startswith("case_")}
    assert top == {bc.INDEX_NAME, bc.INDEX_SEAL, bc.SUMMARY_NAME, "_internal"}
    assert c.summary["source_files_modified"] is False


def test_cache_arrays_equal_in_memory_preprocessing(c):
    """The cache stores exactly the in-memory preprocessing result (padding applied later)."""
    v = _verify(c)
    store = bc.CacheStore(c.out)
    train, val = bc.open_cases(v, store)
    mem = bd.load_cases({a: c.files[a] for a in TRAIN}, c.keys)
    for cc, mc in zip(train, mem, strict=True):
        assert cc.case_key == mc.case_key
        assert cc.brain_slices == mc.brain_slices and cc.positive_slices == mc.positive_slices
        assert cc.record == mc.record
        for z in mc.brain_slices:
            a, b = bd.slice_item(cc, z), bd.slice_item(mc, z)
            for k in ("image", "target", "brain_mask", "valid_region_mask"):
                assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k
            assert a["geometry"] == b["geometry"]
    # validation on the original grid: identical metrics from cache and from memory
    mval = bd.load_cases({a: c.files[a] for a in VAL}, c.keys)
    model = build_model(42)
    ea = bi.evaluate(model, val, torch.device("cpu"), 32)
    eb = bi.evaluate(model, mval, torch.device("cpu"), 32)
    assert ea["ungated_summary"] == eb["ungated_summary"]
    assert ea["per_case_gated"] == eb["per_case_gated"]
    for k in ea["outputs"]:
        assert np.array_equal(ea["outputs"][k].probability, eb["outputs"][k].probability)
        assert np.array_equal(ea["outputs"][k].gated, eb["outputs"][k].gated)


# ---------------------------------------------------------------- 1-4 RAM policy
def _big_cases(folder: Path, n: int):
    files = {}
    for i in range(n):
        shape = (64, 64, 10)
        g = np.indices(shape).astype(float)
        brain = ((g[0] - 31.5) / 26) ** 2 + ((g[1] - 31.5) / 26) ** 2 <= 1
        lesion = brain & (((g[0] - 30) / 5) ** 2 + ((g[1] - 33) / 5) ** 2 <= 1)
        lesion[:, :, :2] = False
        rng = np.random.default_rng(i)
        dwi = np.where(brain, 300 + rng.normal(0, 20, shape), 0).astype(np.float32)
        adc = np.where(brain, 800 + rng.normal(0, 40, shape), 0).astype(np.float32)
        aff = np.diag([2.0, 2.0, 3.0, 1.0])
        f = CaseFiles(*(folder / f"big{i}_{r}.nii.gz" for r in ("d", "a", "m")))
        for p, arr in zip(
            (f.dwi, f.adc, f.label), (dwi, adc, lesion.astype(np.uint8)), strict=True
        ):
            nib.save(nib.Nifti1Image(arr, aff), str(p))
        files[f"syn-big-{i:02d}"] = f
    return files


def test_01_to_04_no_full_volumes_in_ram_mmap_and_two_active_cases(tmp_path, monkeypatch):
    files = _big_cases(tmp_path, 13)
    aliases = sorted(files)
    train, val = aliases[:12], aliases[12:]
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"splits": {"train": train, "val": val}}), encoding="utf-8")
    out = tmp_path / "cache"
    bc.build_cache(
        out,
        config=CFG,
        split_file=split,
        files={"train": {a: files[a] for a in train}, "val": {a: files[a] for a in val}},
        git_commit=COMMIT,
        package_lock_sha256_lf=LOCK,
    )
    keys = case_keys(aliases)
    v = bc.verify_cache(
        out,
        config=CFG,
        split_file=split,
        train_aliases=train,
        val_aliases=val,
        input_sha256=_inputs(files, keys),
        package_lock_sha256_lf=LOCK,
        git_commit=COMMIT,
    )
    # in-memory baseline: every prepared volume held at once
    tracemalloc.start()
    mem = bd.load_cases({a: files[a] for a in train}, keys)
    mem_bytes = tracemalloc.get_traced_memory()[0]
    del mem
    tracemalloc.stop()
    loads = []
    real_load = np.load

    def spy(*a, **k):
        loads.append(k.get("mmap_mode"))
        return real_load(*a, **k)

    monkeypatch.setattr(bc.np, "load", spy)
    tracemalloc.start()
    store = bc.CacheStore(out)
    cases, _ = bc.open_cases(v, store)
    meta_bytes = tracemalloc.get_traced_memory()[0]
    order, _ = bd.epoch_sampling(cases, 42, 1)
    tracemalloc.reset_peak()
    n = batch_bytes = 0
    for b in bd.batches(cases, order, 4):
        n += int(b["image"].shape[0])
        batch_bytes = max(
            batch_bytes,
            sum(int(b[k].nbytes) for k in ("image", "target", "brain_mask", "valid_region_mask")),
        )
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert n == len(order)
    # 1: only metadata stays; no case holds an ndarray; the peak is a few batches, not the
    # dataset (it does not grow with the number of cases)
    assert meta_bytes < mem_bytes / 20, (meta_bytes, mem_bytes)
    assert peak < 4 * batch_bytes and peak < mem_bytes / 4, (peak, batch_bytes, mem_bytes)
    for case in cases:
        assert not any(isinstance(x, np.ndarray) for x in vars(case).values())
    # 2: every array read is a read-only memory map
    assert loads and set(loads) == {"r"}
    arr = store.array(cases[0].case_key, "dwi")
    assert isinstance(arr, np.memmap) and arr.mode == "r" and not arr.flags.writeable
    # 3-4: at most 2 cases open, least-recently-used evicted, limit fixed
    assert store.peak_active_cases == 2 and len(store.active_cases) <= 2
    store.close()
    for k in [cs.case_key for cs in cases[:3]]:
        store.array(k, "gt")
    assert store.active_cases == (cases[1].case_key, cases[2].case_key)
    store.array(cases[1].case_key, "gt")  # touch -> most recent
    store.array(cases[3].case_key, "gt")
    assert store.active_cases == (cases[1].case_key, cases[3].case_key)
    with pytest.raises(bc.CacheError):
        bc.CacheStore(out, max_active_cases=3)
    with pytest.raises(bc.CacheError):
        bc.CacheStore(out, mmap_mode="r+")


# ---------------------------------------------------------------- 5-12 integrity gate
def test_05_input_hash_checked(c):
    k = c.keys[TRAIN[0]]
    bad = {**c.inputs, k: {**c.inputs[k], "dwi_sha256": "0" * 64}}
    assert f"{k}:input_hash_mismatch" in _reasons(lambda: _verify(c, input_sha256=bad))


def test_06_config_hash_checked(c):
    cfg = yaml.safe_load(yaml.safe_dump(CFG))
    cfg["run_name"] = "other"
    assert "config_canonical_sha256_mismatch" in _reasons(lambda: _verify(c, config=cfg))


def test_07_split_hash_checked(c, tmp_path):
    s = tmp_path / "split.json"
    s.write_text(json.dumps({"splits": {"train": TRAIN, "val": VAL}, "note": "x"}), "utf-8")
    assert "split_canonical_sha256_mismatch" in _reasons(lambda: _verify(c, split_file=s))
    # different case set (alias list) and counts
    r = _reasons(lambda: _verify(c, train_aliases=TRAIN[:2]))
    assert "alias_list_sha256_mismatch" in r and "case_count_mismatch" in r


def test_08_policy_checked(c, tmp_path):
    cache = _copy(c, tmp_path)
    k = c.keys[VAL[0]]
    _forge(cache, k, mutate=lambda m: m.update(preprocessing_policy="other_policy"))
    assert f"{k}:preprocessing_policy_mismatch" in _reasons(lambda: _verify(c, cache))
    cfg = yaml.safe_load(yaml.safe_dump(CFG))
    cfg["cache"]["max_active_cases"] = 4
    assert "cache_policy_mismatch" in _reasons(lambda: _verify(c, config=cfg))


def test_09_12_array_hash_and_tampering_detected(c, tmp_path):
    cache = _copy(c, tmp_path)
    k = c.keys[TRAIN[1]]
    p = cache / k / "dwi.npy"
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 0xFF
    p.write_bytes(bytes(raw))
    assert f"{k}:cache_array_hash_mismatch:dwi" in _reasons(lambda: _verify(c, cache))
    cache2 = tmp_path / "c2"
    shutil.copytree(c.out, cache2)
    (cache2 / k / bc.MANIFEST_NAME).write_text(
        (cache2 / k / bc.MANIFEST_NAME)
        .read_text(encoding="utf-8")
        .replace('"split": "train"', '"split": "val"'),
        encoding="utf-8",
    )
    assert any("seal" in r for r in _reasons(lambda: _verify(c, cache2)))
    cache3 = tmp_path / "c3"
    shutil.copytree(c.out, cache3)
    (cache3 / bc.INDEX_SEAL).write_text("0" * 64, encoding="utf-8")
    assert "cache_index_seal_mismatch" in _reasons(lambda: _verify(c, cache3))


@pytest.mark.parametrize(
    "array,make,reason",
    [
        ("dwi", lambda a: a.astype(np.float64), "cache_array_dtype:dwi"),
        ("dwi", lambda a: a[:-1], "cache_array_shape:dwi"),
        (
            "adc",
            lambda a: np.where(a == a.flat[0], np.nan, a).astype(np.float32),
            "cache_array_non_finite:adc",
        ),
        ("gt", lambda a: np.where(a == 1, 2, a).astype(np.uint8), "cache_array_not_binary:gt"),
        ("brain_mask", lambda a: (a * 3).astype(np.uint8), "cache_array_not_binary:brain_mask"),
        (
            "valid_region_mask",
            lambda a: np.zeros_like(a),
            "cache_array_not_all_one:valid_region_mask",
        ),
    ],
)
def test_shape_dtype_binary_and_nan_checked(c, tmp_path, array, make, reason):
    cache = _copy(c, tmp_path)
    k = c.keys[TRAIN[0]]
    data = make(np.load(cache / k / f"{array}.npy"))
    _forge(cache, k, array=array, data=data)
    assert f"{k}:{reason}" in _reasons(lambda: _verify(c, cache))


def test_gt_status_checked(c, tmp_path):
    cache = _copy(c, tmp_path)
    k = c.keys[TRAIN[0]]  # positive case declared negative
    _forge(cache, k, mutate=lambda m: m.update(ground_truth_status="negative_case"))
    assert f"{k}:gt_status_mismatch" in _reasons(lambda: _verify(c, cache))
    cache2 = tmp_path / "c2"
    shutil.copytree(c.out, cache2)
    _forge(cache2, k, mutate=lambda m: m.update(positive_slices=[]))
    assert f"{k}:slice_index_mismatch" in _reasons(lambda: _verify(c, cache2))


def test_10_overwrite_refused(c):
    before = {p.relative_to(c.out).as_posix(): sha256_file(p) for p in c.out.rglob("*.*")}
    with pytest.raises(bc.CacheError, match="overwrite"):
        bc.build_cache(
            c.out,
            config=CFG,
            split_file=c.split,
            files={"train": {a: c.files[a] for a in TRAIN}, "val": {a: c.files[a] for a in VAL}},
            git_commit=COMMIT,
            package_lock_sha256_lf=LOCK,
        )
    after = {p.relative_to(c.out).as_posix(): sha256_file(p) for p in c.out.rglob("*.*")}
    assert before == after


def test_11_missing_cache_or_case_refused(c, tmp_path):
    assert _reasons(lambda: _verify(c, tmp_path / "absent")) == ("cache_index_missing",)
    cache = _copy(c, tmp_path)
    shutil.rmtree(cache / c.keys[VAL[1]])  # a copy only; the built cache is untouched
    r = _reasons(lambda: _verify(c, cache))
    assert "cache_case_folders_differ_from_index" in r


def test_cache_records_hold_no_alias_id_or_path(c):
    needles = [*TRAIN, *VAL, *FORBIDDEN, str(c.root), c.root.as_posix(), str(Path.home())]
    needles += [Path(f).name for fs in c.files.values() for f in (fs.dwi, fs.adc, fs.label)]
    for p in c.out.rglob("*.json"):
        if "_internal" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        assert not any(n in text for n in needles), p.name
    m = json.loads((c.out / c.keys[TRAIN[0]] / bc.MANIFEST_NAME).read_text(encoding="utf-8"))
    for k in (
        "source_sha256",
        "config_canonical_sha256",
        "split_canonical_sha256",
        "git_commit",
        "package_lock_sha256_lf",
        "environment",
        "gt_snapped_voxels",
        "original_grid",
    ):
        assert k in m
    assert m["alias_list_sha256"] == alias_list_sha256(TRAIN + VAL, CASE_KEY_POLICY)
    assert m["gt_snapped_voxels"] > 0  # syn-train-01 has scaled GT storage


def test_build_refuses_blocked_case_and_publishes_nothing(c, tmp_path):
    from tests.test_v2_baseline_runner import NEGATIVE  # noqa: F401 (same synthetic family)

    bad = dict(c.files)
    f = bad[TRAIN[0]]
    img = nib.load(str(f.label))
    arr = np.asanyarray(img.dataobj).astype(np.float32)
    arr[0, 0, 0] = 0.6  # invalid GT label
    lab = tmp_path / "bad_m.nii.gz"
    nib.save(nib.Nifti1Image(arr, img.affine), str(lab))
    bad[TRAIN[0]] = CaseFiles(f.dwi, f.adc, lab)
    out = tmp_path / "v_bad"
    with pytest.raises(Exception):  # noqa: B017 - PreprocessError from the decode policy
        bc.build_cache(
            out,
            config=CFG,
            split_file=c.split,
            files={"train": {a: bad[a] for a in TRAIN}, "val": {a: bad[a] for a in VAL}},
            git_commit=COMMIT,
            package_lock_sha256_lf=LOCK,
        )
    assert not out.exists()


# ---------------------------------------------------------------- commit binding
def test_commit_binding_in_gate_and_build(c, tmp_path):
    assert _verify(c).index["git_commit"] == COMMIT  # same commit: allowed
    assert "cache_git_commit_mismatch" in _reasons(lambda: _verify(c, git_commit="1" * 40))
    assert "current_git_commit_unknown" in _reasons(lambda: _verify(c, git_commit=None))
    with pytest.raises(bc.CacheCommitError, match="cache_git_commit_missing"):
        bc.build_cache(
            tmp_path / "no_commit",
            config=CFG,
            split_file=c.split,
            files={"train": {a: c.files[a] for a in TRAIN}, "val": {a: c.files[a] for a in VAL}},
            git_commit=None,
            package_lock_sha256_lf=LOCK,
        )
    assert not (tmp_path / "no_commit").exists()
    clean = {"working_tree_clean": True, "git_commit": COMMIT}
    assert bc.check_cache_commit(c.out, clean) == []
    assert bc.check_cache_commit(c.out, {**clean, "git_commit": "1" * 40}) == [
        "cache_git_commit_mismatch"
    ]
    assert "working_tree_not_clean" in bc.check_cache_commit(
        c.out, {**clean, "working_tree_clean": False}
    )
    cache = _copy(c, tmp_path)  # commit removed from the (resealed) index
    idx = json.loads((cache / bc.INDEX_NAME).read_text(encoding="utf-8"))
    idx["git_commit"] = None
    (cache / bc.INDEX_NAME).write_text(json.dumps(idx), encoding="utf-8")
    (cache / bc.INDEX_SEAL).write_text(canonical_json_sha256(idx), encoding="utf-8")
    assert bc.check_cache_commit(cache, clean) == ["cache_git_commit_missing"]
    # a case manifest from another commit (consistently resealed) is refused too
    cache2 = tmp_path / "c2"
    shutil.copytree(c.out, cache2)
    k = c.keys[VAL[0]]
    _forge(cache2, k, mutate=lambda m: m.update(git_commit="2" * 40))
    assert f"{k}:git_commit_mismatch" in _reasons(lambda: _verify(c, cache2))


# ---------------------------------------------------------------- axis order
def test_axis_order_recorded_and_storage_logical_mapping(c):
    v = _verify(c)
    store = bc.CacheStore(c.out)
    train, val = bc.open_cases(v, store)
    mem = {m.case_key: m for m in bd.load_cases(c.files, c.keys)}
    rng = np.random.default_rng(0)
    for case in train + val:
        m = v.manifests[case.case_key]
        assert m["storage_axis_order"] == ["z", "x", "y"]
        assert m["logical_axis_order"] == ["x", "y", "z"] and m["slice_axis"] == 2
        assert m["transpose_policy_id"] == "cache_slice_first_zxy_to_logical_xyz_v1"
        mc = mem[case.case_key]
        p = mc.prepared
        logical = {
            "dwi": p.image[0],
            "adc": p.image[1],
            "gt": p.label,
            "brain_mask": p.brain,
            "valid_region_mask": np.ones(p.model_grid.shape, np.uint8),
            "original_gt": mc.original_gt,
            "original_brain_mask": p.original_brain,
        }
        for name, ref in logical.items():
            assert m["arrays"][name]["storage_axis_order"] == ["z", "x", "y"]
            stored = store.array(case.case_key, name)
            view = store.volume(case.case_key, name)
            x, y, z = ref.shape
            assert stored.shape == (z, x, y) and view.shape == (x, y, z)
            assert np.array_equal(view, np.asarray(ref).astype(view.dtype))  # exact
            for zz in range(z):  # slice index mapping
                assert np.array_equal(stored[zz], view[:, :, zz])
            for _ in range(20):  # coordinate mapping
                i, j, kk = (int(rng.integers(0, s)) for s in (x, y, z))
                assert stored[kk, i, j] == view[i, j, kk] == np.asarray(ref)[i, j, kk]
        # DWI / ADC / GT / mask alignment
        gt_v = store.volume(case.case_key, "gt").astype(bool)
        brain_v = store.volume(case.case_key, "brain_mask").astype(bool)
        assert np.array_equal(gt_v, p.label) and np.array_equal(brain_v, p.brain)
        if gt_v.any():
            dwi_v = store.volume(case.case_key, "dwi")
            adc_v = store.volume(case.case_key, "adc")
            other = brain_v & ~gt_v
            assert dwi_v[gt_v].mean() > dwi_v[other].mean()  # synthetic lesion: bright DWI
            assert adc_v[gt_v].mean() < adc_v[other].mean()  # and dark ADC at the same voxels
        # affines act on logical (x, y, z) indices
        ijk = np.array([1.0, 2.0, 1.0, 1.0])
        assert np.array_equal(case.model_grid.affine @ ijk, p.model_grid.affine @ ijk)
        assert np.array_equal(case.original_grid.affine, p.original_grid.affine)
    store.close()


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda m: m.update(storage_axis_order=["x", "y", "z"]), "storage_axis_order_mismatch"),
        (lambda m: m.update(slice_axis=0), "slice_axis_mismatch"),
        (lambda m: m.update(transpose_policy_id="other"), "transpose_policy_id_mismatch"),
        (
            lambda m: m["arrays"]["gt"].update(storage_axis_order=["x", "y", "z"]),
            "cache_array_axis_order:gt",
        ),
    ],
)
def test_axis_order_tampering_refused(c, tmp_path, mutate, reason):
    cache = _copy(c, tmp_path)
    k = c.keys[TRAIN[0]]
    _forge(cache, k, mutate=mutate)
    assert f"{k}:{reason}" in _reasons(lambda: _verify(c, cache))


# ---------------------------------------------------------------- affine precision
def _oblique_case(folder: Path) -> CaseFiles:
    shape = (20, 18, 5)
    g = np.indices(shape).astype(float)
    brain = ((g[0] - 9.5) / 8) ** 2 + ((g[1] - 8.5) / 7) ** 2 <= 1
    lesion = brain & (((g[0] - 10) / 2.5) ** 2 + ((g[1] - 8) / 2.5) ** 2 <= 1)
    lesion[:, :, [0, 4]] = False
    rng = np.random.default_rng(7)
    dwi = np.where(brain, 300 + rng.normal(0, 20, shape), 0).astype(np.float32)
    dwi[lesion] += 400
    adc = np.where(brain, 800 + rng.normal(0, 40, shape), 0).astype(np.float32)
    a, b = 0.3, 0.11  # rotation about z and tilt about x: irrational direction cosines
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    aff = np.eye(4)
    aff[:3, :3] = rz @ rx @ np.diag([1.7, 1.9, 4.3])
    aff[:3, 3] = (-31.123456789, 12.987654321, -7.000000001)
    f = CaseFiles(*(folder / f"obl_{r}.nii.gz" for r in ("d", "a", "m")))
    for p, arr in zip((f.dwi, f.adc, f.label), (dwi, adc, lesion.astype(np.uint8)), strict=True):
        nib.save(nib.Nifti1Image(arr, aff), str(p))
    return f


def test_oblique_affine_full_precision_and_restore(c, tmp_path):
    from neuroscan_ai.v2.baseline_outputs import build_outputs
    from neuroscan_ai.v2.geometry import check_on_grid

    f = _oblique_case(tmp_path)
    train, val = [TRAIN[0]], ["syn-obl-01"]
    files = {TRAIN[0]: c.files[TRAIN[0]], "syn-obl-01": f}
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"splits": {"train": train, "val": val}}), encoding="utf-8")
    out = tmp_path / "obl_cache"
    bc.build_cache(
        out,
        config=CFG,
        split_file=split,
        files={"train": {train[0]: files[train[0]]}, "val": {"syn-obl-01": f}},
        git_commit=COMMIT,
        package_lock_sha256_lf=LOCK,
    )
    keys = case_keys(train + val)
    kw = dict(
        config=CFG,
        split_file=split,
        train_aliases=train,
        val_aliases=val,
        input_sha256=_inputs(files, keys),
        package_lock_sha256_lf=LOCK,
        git_commit=COMMIT,
    )
    v = bc.verify_cache(out, **kw)
    store = bc.CacheStore(out)
    _, (cached,) = bc.open_cases(v, store)
    mem = bd.load_case(keys["syn-obl-01"], f)
    m = v.manifests[cached.case_key]
    exact = np.asarray(m["original_grid"]["affine_exact"], np.float64)
    nib_aff = nib.load(str(f.dwi)).affine.astype(np.float64)
    assert np.array_equal(exact, mem.prepared.original_grid.affine)
    assert np.array_equal(exact, nib_aff)  # nothing rounded on the way
    assert np.array_equal(cached.original_grid.affine, mem.prepared.original_grid.affine)
    assert np.array_equal(cached.model_grid.affine, mem.prepared.model_grid.affine)
    assert cached.original_grid.affine.dtype == np.float64
    disp = np.asarray(m["original_grid"]["affine_display"])
    assert not np.array_equal(disp, exact) and np.allclose(disp, exact, atol=1e-8)
    # display values never reach a computation
    rec = dict(m["model_grid"], affine_display=[[0.0] * 4] * 4, spacing_mm_display=[9, 9, 9])
    assert np.array_equal(bc._grid(rec).affine, cached.model_grid.affine)
    # restore to the original grid: identical to the in-memory path (difference 0)
    prob = np.random.default_rng(1).random(cached.model_grid.shape).astype(np.float32)
    ra = build_outputs(prob, cached.prepared, 0.5)
    rb = build_outputs(prob, mem.prepared, 0.5)
    for k in ("probability", "ungated", "brain_mask", "gated"):
        assert np.array_equal(getattr(ra, k), getattr(rb, k)), k
    assert check_on_grid(ra.probability, ra.grid, mem.prepared.original_grid) == []
    store.close()
    del ra, rb, cached, v
    # a display value that no longer matches the exact affine is refused
    key = keys["syn-obl-01"]
    _forge(out, key, mutate=lambda mm: mm["original_grid"].update(affine_display=[[1.0] * 4] * 4))
    r = _reasons(lambda: bc.verify_cache(out, **kw))
    assert f"{key}:original_grid_affine_display_mismatch" in r
