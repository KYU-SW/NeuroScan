"""Phase 7-B Research Validation Mode: difference labelmap, matching, validation-only adapter,
verified bundle, atomic registration, schema v2 upgrade and the research API (off = 404).

Synthetic data and a synthetic split file only (case keys sub-strokecase9xxx do not exist in
ISLES); no validation or test image is read here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neuroscan_ai.evaluation.segmentation import evaluate_case
from neuroscan_ai.research.bundle import (
    ResearchBuildError,
    build_research_bundle,
    compare_reference,
)
from neuroscan_ai.research.compare import FN, FP, TP, check_difference, difference_labelmap
from neuroscan_ai.research.matching import check_matching, match_components
from neuroscan_ai.research.sources import Isles2022ResearchAdapter, ResearchSourceError
from tests.api_helpers import API, client_for, make_settings
from tests.inference_helpers import tiny_model_setup, write_case, write_nifti

pytest.importorskip("fastapi")

from neuroscan_api.db.models import ResearchCase, ResearchFile  # noqa: E402
from neuroscan_api.db.session import Database  # noqa: E402
from neuroscan_api.services import research as registry  # noqa: E402
from neuroscan_api.storage import Storage  # noqa: E402

AFF = np.diag([-2.0, 2.0, 2.0, 1.0])
VAL, TEST, TRAIN = "sub-strokecase9001", "sub-strokecase9002", "sub-strokecase9003"


# --- difference labelmap ------------------------------------------------------------------------


def test_difference_labelmap_matches_evaluate_case():
    pred = np.zeros((8, 8, 4), np.uint8)
    gt = np.zeros_like(pred)
    pred[1:4, 1:4, 1] = 1
    gt[2:5, 2:5, 1] = 1
    gt[6, 6, 3] = 1
    diff = difference_labelmap(pred, gt)
    assert diff.dtype == np.uint8 and diff.shape == pred.shape
    m = evaluate_case(pred, gt, (2, 2, 2))
    counts = np.bincount(diff.ravel(), minlength=4)
    assert (counts[TP], counts[FP], counts[FN]) == (m["tp_voxels"], m["fp_voxels"], m["fn_voxels"])
    assert counts[TP] + counts[FP] == pred.sum() and counts[TP] + counts[FN] == gt.sum()
    assert check_difference(diff, pred, gt, m) == []
    bad = diff.copy()
    bad[0, 0, 0] = FP
    assert check_difference(bad, pred, gt, m)
    with pytest.raises(ValueError):
        difference_labelmap(pred * 2, gt)
    with pytest.raises(ValueError):
        difference_labelmap(pred, gt[:, :, :3])


# --- component matching -----------------------------------------------------------------------


def _blob(m, x, y, z=2, w=2):
    m[x : x + w, y : y + w, z] = 1


def test_matching_group_types_and_totals():
    pred = np.zeros((40, 40, 6), np.uint8)
    gt = np.zeros_like(pred)
    _blob(pred, 1, 1, w=4)  # one_to_one with GT A
    _blob(gt, 2, 2, w=4)
    _blob(pred, 10, 1)  # split: two prediction parts over GT B
    _blob(pred, 10, 6)
    gt[10:12, 1:8, 2] = 1
    pred[20:22, 1:9, 2] = 1  # merge: one prediction over GT C and GT D
    _blob(gt, 20, 1)
    _blob(gt, 20, 6)
    pred[30:32, 1:5, 2] = 1  # complex: P1-G1, P1-G2, P2-G2 (gaps keep the parts separate)
    pred[30:32, 8:10, 2] = 1
    gt[30:32, 0:2, 2] = 1
    gt[30:32, 4:9, 2] = 1
    pred[1, 30, 4] = 1  # prediction_only, 1 voxel
    gt[20, 30, 4] = 1  # ground_truth_only, 1 voxel
    m = match_components(pred, gt, (1, 1, 1), np.eye(4), 2)
    types = sorted(g["group_type"] for g in m["groups"])
    assert types == sorted(
        ["one_to_one", "split", "merge", "complex", "prediction_only", "ground_truth_only"]
    )
    metrics = evaluate_case(pred, gt, (1, 1, 1))
    assert check_matching(m, metrics) == []
    for g in m["groups"]:
        assert g["overlap_voxel_count"] <= g["union_voxel_count"]
        if g["group_type"] in ("prediction_only", "ground_truth_only"):
            assert g["overlap_voxel_count"] == 0 and g["iou"] == 0.0
            assert g["centroid_distance_mm"] is None
        else:
            assert 0 < g["iou"] <= 1 and g["centroid_distance_mm"] is not None
    only = next(p for p in m["prediction_components"] if p["group_type"] == "prediction_only")
    assert only["tp_voxels"] == 0 and only["fp_voxels"] == 1 and not only["gt_overlap"]
    missed = next(g for g in m["ground_truth_components"] if g["group_type"] == "ground_truth_only")
    assert missed["detected_voxels"] == 0 and missed["missed_voxels"] == 1
    split = next(g for g in m["groups"] if g["group_type"] == "split")
    assert (
        len(split["prediction_component_ids"]) == 2
        and len(split["ground_truth_component_ids"]) == 1
    )


def test_matching_empty_ground_truth():
    pred = np.zeros((10, 10, 3), np.uint8)
    pred[1, 1, 1] = 1
    pred[5:7, 5:7, 1] = 1
    m = match_components(pred, np.zeros_like(pred), (1, 1, 1), np.eye(4), 2)
    assert [g["group_type"] for g in m["groups"]] == ["prediction_only", "prediction_only"]
    assert m["ground_truth_components"] == []
    assert check_matching(m, evaluate_case(pred, np.zeros_like(pred), (1, 1, 1))) == []


# --- adapter: validation only, decided before any file is opened ----------------------------


@pytest.fixture
def split_file(tmp_path):
    p = tmp_path / "split.json"
    p.write_text(json.dumps({"splits": {"train": [TRAIN], "val": [VAL], "test": [TEST]}}))
    return p


def test_adapter_refuses_test_and_train_before_opening_files(split_file, tmp_path, monkeypatch):
    def no_io(*_a, **_k):
        raise AssertionError("a file was opened")

    monkeypatch.setattr(nib, "load", no_io)
    adapter = Isles2022ResearchAdapter(tmp_path / "missing", split_file)
    for key, code in ((TEST, "SEALED_TEST_CASE"), (TRAIN, "NOT_VALIDATION")):
        with pytest.raises(ResearchSourceError) as e:
            adapter.resolve(key, "val-001")
        assert e.value.code == code
    for alias in ("val-1", "VAL-001", "x/val-001", "val-0001"):
        with pytest.raises(ResearchSourceError):
            adapter.resolve(VAL, alias)
    with pytest.raises(ResearchSourceError):
        adapter.resolve("../" + VAL, "val-001")
    src = adapter.resolve(VAL, "val-001")
    assert src.split == "val" and src.alias == "val-001"
    assert src.ground_truth.name == f"{VAL}_ses-0001_msk.nii.gz"


# --- verified bundle ----------------------------------------------------------------------------


def _bids_case(root: Path, key: str, gt=None, gt_affine=None, lesion=True):
    """Synthetic ISLES-like layout for one case; returns the GT array written."""
    d, a, label, aff = write_case(root / "tmpcase", lesion=lesion)
    sub = root / key / "ses-0001" / "dwi"
    sub.mkdir(parents=True, exist_ok=True)
    (d).replace(sub / f"{key}_ses-0001_dwi.nii.gz")
    (a).replace(sub / f"{key}_ses-0001_adc.nii.gz")
    gt_arr = label.astype(np.float64) if gt is None else gt
    write_nifti(
        root / "derivatives" / key / "ses-0001" / f"{key}_ses-0001_msk.nii.gz",
        gt_arr,
        aff if gt_affine is None else gt_affine,
    )
    return gt_arr, aff


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("model")
    inf, *_ = tiny_model_setup(tmp, bias=None)
    return inf


def _build(
    tmp_path,
    split_file,
    tiny_model,
    *,
    gt=None,
    gt_affine=None,
    reference=None,
    require_reference=False,
    alias="val-001",
    lesion=True,
):
    data = tmp_path / "data"
    _bids_case(data, VAL, gt, gt_affine, lesion)
    adapter = Isles2022ResearchAdapter(data, split_file, reference)
    source = adapter.resolve(VAL, alias)
    staging = tmp_path / "staging" / alias
    manifest = build_research_bundle(
        source,
        staging,
        device="cpu",
        require_reference=require_reference,
        inference_kwargs={"inference_config_path": tiny_model},
    )
    return source, staging, manifest


def test_bundle_is_consistent_and_private(tmp_path, split_file, tiny_model):
    source, staging, manifest = _build(tmp_path, split_file, tiny_model)
    files = sorted(p.name for p in staging.iterdir())
    assert files == sorted(
        [
            "adc.nii.gz",
            "components.json",
            "difference.nii.gz",
            "dwi.nii.gz",
            "matching.json",
            "metrics.json",
            "prediction_components.nii.gz",
            "prediction_mask.nii.gz",
            "research_manifest.json",
        ]
    )
    pred = np.asanyarray(nib.load(staging / "prediction_mask.nii.gz").dataobj)
    diff_img = nib.load(staging / "difference.nii.gz")
    diff = np.asanyarray(diff_img.dataobj)
    gt = np.asanyarray(nib.load(source.ground_truth).dataobj) > 0.5
    m = json.loads((staging / "metrics.json").read_text())["metrics"]
    assert diff.dtype == np.uint8 and diff.shape == pred.shape
    np.testing.assert_allclose(diff_img.affine, nib.load(staging / "dwi.nii.gz").affine, atol=1e-5)
    assert check_difference(diff, pred, gt, m) == []
    assert m == {
        k: v
        for k, v in evaluate_case(pred, gt, nib.affines.voxel_sizes(diff_img.affine)).items()
        if k != "case_id"
    }
    matching = json.loads((staging / "matching.json").read_text())
    assert check_matching(matching, m) == []
    text = "".join(
        (staging / n).read_text()
        for n in ("metrics.json", "components.json", "matching.json", "research_manifest.json")
    )
    assert VAL not in text and str(tmp_path) not in text and "\\\\" not in text
    for vid, v in manifest["volumes"].items():
        assert hashlib.sha256((staging / v["file"]).read_bytes()).hexdigest() == v["sha256"], vid
    assert manifest["split"] == "val" and manifest["alias"] == "val-001"
    assert manifest["model"]["threshold"] == 0.5


def test_bundle_reproducibility_check(tmp_path, split_file, tiny_model):
    _, staging, _ = _build(tmp_path / "a", split_file, tiny_model)
    pred_img = nib.load(staging / "prediction_mask.nii.gz")
    pred = np.asanyarray(pred_img.dataobj)
    same = tmp_path / "ref_same" / f"{VAL}_pred.nii.gz"
    write_nifti(same, pred, pred_img.affine)
    _, staging2, manifest = _build(
        tmp_path / "b", split_file, tiny_model, reference=same.parent, require_reference=True
    )
    assert manifest["reproducibility"]["identical"] is True
    assert manifest["reproducibility"]["mismatched_voxels"] == 0
    changed = pred.copy()
    changed[0, 0, 0] = 1 - changed[0, 0, 0]
    other = tmp_path / "ref_diff" / f"{VAL}_pred.nii.gz"
    write_nifti(other, changed, pred_img.affine)
    with pytest.raises(ResearchBuildError) as e:
        _build(
            tmp_path / "c", split_file, tiny_model, reference=other.parent, require_reference=True
        )
    assert e.value.code == "PREDICTION_MISMATCH" and e.value.details["mismatched_voxels"] == 1
    with pytest.raises(ResearchBuildError) as e:
        _build(
            tmp_path / "d",
            split_file,
            tiny_model,
            reference=tmp_path / "none",
            require_reference=True,
        )
    assert e.value.code == "REFERENCE_MISSING"
    assert compare_reference(pred, pred_img.affine, same)["identical"] is True


@pytest.mark.parametrize(
    "kind,code",
    [("grid", "GRID_MISMATCH"), ("values", "GT_NOT_BINARY"), ("affine", "INVALID_AFFINE")],
)
def test_bundle_refuses_bad_ground_truth(tmp_path, split_file, tiny_model, kind, code):
    shape = (48, 48, 10)
    gt = np.zeros(shape)
    gt_affine = None
    if kind == "grid":
        gt = np.zeros((48, 48, 9))
    elif kind == "values":
        gt[1, 1, 1] = 2
    with pytest.raises(ResearchBuildError) as e:
        if kind == "affine":
            _build_with_singular_gt_affine(tmp_path, split_file, tiny_model)
        else:
            _build(tmp_path, split_file, tiny_model, gt=gt, gt_affine=gt_affine)
    assert e.value.code == code


def _build_with_singular_gt_affine(tmp_path, split_file, tiny_model):
    """GT file whose stored sform is singular (nibabel cannot be asked to write one directly)."""
    data = tmp_path / "data"
    _bids_case(data, VAL)
    path = data / "derivatives" / VAL / "ses-0001" / f"{VAL}_ses-0001_msk.nii.gz"
    hdr = nib.load(path).header.copy()
    for row in ("srow_x", "srow_y", "srow_z"):
        hdr[row] = [0.0, 0.0, 0.0, 0.0]
    hdr["sform_code"] = 1
    hdr["qform_code"] = 0
    img = nib.Nifti1Image(np.zeros((48, 48, 10), np.float32), None, header=hdr)
    nib.save(img, path)
    assert abs(np.linalg.det(nib.load(path).affine[:3, :3])) < 1e-9
    source = Isles2022ResearchAdapter(data, split_file).resolve(VAL, "val-001")
    build_research_bundle(
        source,
        tmp_path / "staging",
        device="cpu",
        require_reference=False,
        inference_kwargs={"inference_config_path": tiny_model},
    )


def test_bundle_negative_ground_truth_keeps_undefined_overlap_metrics(
    tmp_path, split_file, tiny_model
):
    _, staging, _ = _build(tmp_path, split_file, tiny_model, gt=np.zeros((48, 48, 10)))
    m = json.loads((staging / "metrics.json").read_text())["metrics"]
    assert m["has_lesion"] is False
    assert m["dice"] is None and m["iou"] is None and m["sensitivity"] is None
    assert m["negative_outcome"] in ("correct_negative", "false_positive_case")
    assert m["false_positive_component_count"] == m["component_count_prediction"]
    assert m["fp_voxels"] == m["prediction_voxels"] and m["tp_voxels"] == m["fn_voxels"] == 0


# --- registration (atomic) and the research API ------------------------------------------------


@pytest.fixture
def registered(tmp_path, split_file, tiny_model):
    settings = make_settings(tmp_path, research_enabled=True)
    storage = Storage(settings.storage_root)
    db = Database.open(settings.database_path)
    db.init_schema()
    source, _, _ = _build(tmp_path / "src", split_file, tiny_model)
    out = []
    for alias in ("val-001", "val-002"):
        staging = registry.staging_dir(storage)
        src = source.__class__(**{**source.__dict__, "alias": alias})
        build_research_bundle(
            src,
            staging,
            device="cpu",
            require_reference=False,
            inference_kwargs={"inference_config_path": tiny_model},
        )
        registry.register_bundle(
            db, storage, staging, alias, {"dataset_id": "isles2022", "case_key": VAL}
        )
        out.append(alias)
    db.dispose()
    return settings, storage, out


def test_registration_is_atomic_and_refuses_duplicates(
    tmp_path, split_file, tiny_model, monkeypatch
):
    settings = make_settings(tmp_path)
    storage = Storage(settings.storage_root)
    db = Database.open(settings.database_path)
    db.init_schema()
    source, _, _ = _build(tmp_path / "src", split_file, tiny_model)
    staging = registry.staging_dir(storage)
    build_research_bundle(
        source,
        staging,
        device="cpu",
        require_reference=False,
        inference_kwargs={"inference_config_path": tiny_model},
    )

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(registry, "_write_alias_map", boom)
    with pytest.raises(OSError):
        registry.register_bundle(db, storage, staging, "val-001", {"case_key": VAL})
    assert staging.is_dir() and not (storage.area("research") / "val-001").exists()
    with db.read() as s:
        assert s.query(ResearchCase).count() == 0 and s.query(ResearchFile).count() == 0
    monkeypatch.undo()
    registry.register_bundle(db, storage, staging, "val-001", {"case_key": VAL})
    alias_map = json.loads((storage.area("research") / "_internal" / "aliases.json").read_text())
    assert alias_map == {"val-001": {"case_key": VAL}}
    with db.read() as s:
        row = s.query(ResearchCase).one()
        assert row.alias == "val-001" and row.split == "val" and row.threshold == 0.5
        assert VAL not in json.dumps({k: str(v) for k, v in row.__dict__.items()})
        assert s.query(ResearchFile).count() == 5
    staging2 = registry.staging_dir(storage)
    build_research_bundle(
        source,
        staging2,
        device="cpu",
        require_reference=False,
        inference_kwargs={"inference_config_path": tiny_model},
    )
    with pytest.raises(registry.ResearchRegistrationError):
        registry.register_bundle(db, storage, staging2, "val-001", {"case_key": VAL})
    (staging2 / "metrics.json").write_text("{}")  # tampered bundle
    with pytest.raises(registry.ResearchRegistrationError):
        registry.register_bundle(db, storage, staging2, "val-002", {"case_key": VAL})
    db.dispose()


def test_research_api_is_absent_when_disabled(registered, tmp_path):
    settings, _, _ = registered
    on = client_for(settings)
    file_id = on.get(f"{API}/research/cases/val-001").json()["volumes"][0]["file_id"]
    off = client_for(settings.model_copy(update={"research_enabled": False}))
    for path in (
        f"{API}/research/cases",
        f"{API}/research/cases/val-001",
        f"{API}/files/{file_id}",
    ):
        assert off.get(path).status_code == 404
    assert on.get(f"{API}/files/{file_id}").status_code == 200


def test_research_api_alias_only_and_isolated(registered):
    settings, storage, aliases = registered
    c = client_for(settings)
    listing = c.get(f"{API}/research/cases").json()
    assert [x["alias"] for x in listing["cases"]] == aliases
    assert all(x["split"] == "val" and x["has_ground_truth"] for x in listing["cases"])
    d = c.get(f"{API}/research/cases/val-001")
    assert d.status_code == 200
    body = d.json()
    text = d.text
    assert VAL not in text and str(storage.root) not in text and "aliases.json" not in text
    assert {v["id"] for v in body["volumes"]} == {
        "dwi",
        "adc",
        "prediction_mask",
        "prediction_components",
        "difference",
    }
    assert [x["key"] for x in body["legend"]] == ["tp", "fp", "fn"]
    assert body["gt_only_color"] == "#CC79A7" and "임상 진단 결과가 아닙니다" in body["notice"]
    m = body["metrics"]
    diff = next(v for v in body["volumes"] if v["id"] == "difference")
    r = c.get(diff["url"])
    assert r.status_code == 200 and hashlib.sha256(r.content).hexdigest() == diff["sha256"]
    lab = np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(r.content)).dataobj)
    counts = np.bincount(lab.ravel(), minlength=4)
    assert (counts[1], counts[2], counts[3]) == (m["tp_voxels"], m["fp_voxels"], m["fn_voxels"])
    for bad in ("val-1", "val-999", "sub-strokecase9001", "..%2fval-001"):
        assert c.get(f"{API}/research/cases/{bad}").status_code == 404
    # general API never sees research data
    assert c.get(f"{API}/cases/{body['alias']}").status_code == 404
    assert c.get(f"{API}/runs/{diff['file_id']}").status_code == 404
    assert c.get(f"{API}/runs/{diff['file_id']}/result").status_code == 404


# --- schema v2 upgrade --------------------------------------------------------------------------


def test_schema_v1_database_is_upgraded_with_backup_and_no_data_loss(tmp_path):
    db_path = tmp_path / "storage" / "neuroscan.db"
    db = Database.open(db_path)
    db.init_schema()
    with db.write() as s:
        from neuroscan_api.db.models import Case

        s.add(Case(id="case-0123456789ab", dwi_sha256="a" * 64, adc_sha256="b" * 64))
    db.dispose()
    con = sqlite3.connect(db_path)  # turn it back into a v1 database
    con.execute("DROP TABLE research_files")
    con.execute("DROP TABLE research_cases")
    con.execute("UPDATE schema_version SET version = 1")
    con.commit()
    con.close()
    db = Database.open(db_path)
    db.init_schema()
    backups = list(db_path.parent.glob("neuroscan.db.v1-*.bak"))
    assert len(backups) == 1
    con = sqlite3.connect(db_path)
    assert con.execute("SELECT version FROM schema_version").fetchone()[0] == 2
    assert con.execute("SELECT id FROM cases").fetchall() == [("case-0123456789ab",)]
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"research_cases", "research_files", "cases", "inference_runs"} <= tables
    con.close()
    bak = sqlite3.connect(backups[0])
    assert bak.execute("SELECT version FROM schema_version").fetchone()[0] == 1
    assert bak.execute("SELECT id FROM cases").fetchall() == [("case-0123456789ab",)]
    bak.close()
    db.init_schema()  # idempotent at v2: no second backup
    assert len(list(db_path.parent.glob("neuroscan.db.v1-*.bak"))) == 1
    db.dispose()
    con = sqlite3.connect(db_path)
    con.execute("UPDATE schema_version SET version = 99")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError):
        Database.open(db_path).init_schema()
