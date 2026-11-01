"""Phase 5A service-level contract: output files, no overwrite, failure policy, privacy, CLI."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from neuroscan_ai.inference import run_independent_inference, schemas
from neuroscan_ai.inference import service as service_mod
from neuroscan_ai.inference.errors import Code, InferenceError
from neuroscan_ai.inference.outputs import RunDirectory
from tests.conftest import REPO_ROOT
from tests.inference_helpers import tiny_model_setup, write_case

PHI_TEXT = "PATIENT^NAME 19700101"
NAMES = ("Hong_Gildong_DWI.nii.gz", "Hong_Gildong_ADC.nii.gz")
EXPECTED_FILES = {
    "prediction_mask.nii.gz",
    "prediction_components.nii.gz",
    "quantification.json",
    "inference_metadata.json",
    "manifest.json",
    "inputs/dwi.nii.gz",
    "inputs/adc.nii.gz",
    "logs/inference.log",
}


def _run(tmp_path, bias=50.0, **kwargs):
    d, a, _, aff = write_case(tmp_path / "in", header_text={"descrip": PHI_TEXT}, names=NAMES)
    inf, *_ = tiny_model_setup(tmp_path, bias=bias)
    r = run_independent_inference(
        d, a, tmp_path / "out", inference_config_path=inf, device="cpu", **kwargs
    )
    return r, d, a, aff


def _files(run_dir):
    return {p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()}


def _all_text(run_dir) -> str:
    """Every output file as text; NIfTI files are decompressed (headers included)."""
    chunks = []
    for p in run_dir.rglob("*"):
        if p.is_file():
            raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
            chunks.append(raw.decode("latin-1"))
    return "\n".join(chunks)


def test_successful_run_writes_the_full_verified_contract(tmp_path):
    r, d, _, aff = _run(tmp_path)
    assert r.status == "succeeded" and r.exit_code == 0 and r.error is None
    run_dir = tmp_path / "out" / r.case_id / r.run_id
    assert _files(run_dir) == EXPECTED_FILES
    assert not list((tmp_path / "out" / r.case_id).glob(".*"))  # no staging / failed left
    mask = nib.load(run_dir / "prediction_mask.nii.gz")
    data = np.asanyarray(mask.dataobj)
    src = nib.load(d)
    assert data.dtype == np.uint8 and set(np.unique(data).tolist()) <= {0, 1}
    assert data.shape == src.shape and np.allclose(mask.affine, src.affine, atol=1e-5)
    assert int(mask.header["sform_code"]) == 1 and int(data.sum()) > 0
    for rel in ("quantification.json", "inference_metadata.json", "manifest.json"):
        obj = json.loads((run_dir / rel).read_text(encoding="utf-8"))
        schema = obj["schema_version"]
        assert schemas.problems(obj, schema, NAMES) == [], rel


def test_inputs_are_deidentified_copies_not_file_copies(tmp_path):
    r, d, a, _ = _run(tmp_path)
    run_dir = tmp_path / "out" / r.case_id / r.run_id
    for role, src in (("dwi", d), ("adc", a)):
        out = nib.load(run_dir / "inputs" / f"{role}.nii.gz")
        orig = nib.load(src)
        np.testing.assert_array_equal(np.asanyarray(out.dataobj), orig.get_fdata(dtype=np.float32))
        np.testing.assert_allclose(out.affine, orig.affine, atol=1e-5)
        assert bytes(out.header["descrip"].tobytes()).strip(b"\x00 ") == b""
        assert not len(out.header.extensions)
        assert (run_dir / "inputs" / f"{role}.nii.gz").read_bytes() != src.read_bytes()


def test_no_phi_paths_or_input_names_anywhere_in_outputs(tmp_path):
    r, *_ = _run(tmp_path)
    text = _all_text(tmp_path / "out" / r.case_id / r.run_id)
    assert PHI_TEXT not in text and "PATIENT" not in text
    assert "Hong_Gildong" not in text
    assert str(tmp_path) not in text and str(tmp_path).replace("\\", "/") not in text
    assert str(REPO_ROOT) not in text and "C:\\" not in text
    meta = r.metadata
    assert "file_name" not in meta["inputs"]["dwi"] and len(meta["inputs"]["dwi"]["sha256"]) == 64
    assert meta["ground_truth_used"] is False
    assert meta["model"]["checkpoint_path"].startswith("external:")


def test_record_input_names_is_opt_in(tmp_path):
    r, *_ = _run(tmp_path, record_input_names=True)
    assert r.metadata["inputs"]["dwi"]["file_name"] == NAMES[0]


def test_no_ground_truth_or_disease_probability_fields(tmp_path):
    r, *_ = _run(tmp_path)
    run_dir = tmp_path / "out" / r.case_id / r.run_id
    blob = "\n".join(
        (run_dir / f).read_text(encoding="utf-8")
        for f in ("quantification.json", "inference_metadata.json", "manifest.json")
    ).lower()
    for forbidden in ('"dice"', '"ground_truth"', "stroke probability", "뇌졸중 확률", '"gt"'):
        assert forbidden not in blob
    q = r.quantification
    assert q["research_score_summary"]["calibrated"] is False
    assert "not probabilities of disease" in q["research_score_summary"]["note"]
    assert q["hemisphere"]["estimated"] is True and q["vascular_territory"] is None
    assert q["disclaimer"] == schemas.DISCLAIMER


def test_manifest_is_relative_and_inputs_not_shared_by_default(tmp_path):
    r, *_ = _run(tmp_path)
    run_dir = tmp_path / "out" / r.case_id / r.run_id
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for v in m["volumes"]:
        assert not schemas.is_absolute_path_text(v["uri"]) and (run_dir / v["uri"]).is_file()
        assert v["mime_type"] == "application/gzip" and len(v["affine"]) == 4
    assert {v["id"] for v in m["volumes"]} == {
        "dwi",
        "adc",
        "prediction_mask",
        "prediction_components",
    }
    comp = next(v for v in m["volumes"] if v["id"] == "prediction_components")
    assert comp["kind"] == "derived_review_labelmap" and comp["derived_from"] == "prediction_mask"
    assert "prediction_components.nii.gz" in m["sharing"]["shareable_by_default"]
    assert "inputs/" in m["sharing"]["not_shareable_by_default"]
    assert all(not s.startswith("inputs") for s in m["sharing"]["shareable_by_default"])
    assert m["views"]["axial"]["array_axis"] == 2  # LAS input: axis 2 points superior
    assert m["render_3d"]["mask_available"] is True


def test_no_lesion_is_a_normal_success(tmp_path):
    r, *_ = _run(tmp_path, bias=-50.0)
    assert r.status == "succeeded" and r.exit_code == 0
    q = r.quantification
    assert q["lesion_detected"] is False and q["total_lesion_volume_ml"] == 0
    run_dir = tmp_path / "out" / r.case_id / r.run_id
    assert int(np.asanyarray(nib.load(run_dir / "prediction_mask.nii.gz").dataobj).sum()) == 0
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert m["render_3d"]["empty_mask"] is True


def test_export_inputs_can_be_disabled(tmp_path):
    r, *_ = _run(tmp_path, export_inputs=False)
    run_dir = tmp_path / "out" / r.case_id / r.run_id
    assert not (run_dir / "inputs").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [v["id"] for v in manifest["volumes"]] == ["prediction_mask", "prediction_components"]
    assert manifest["inputs_exported"] is False


def test_existing_run_folders_are_never_overwritten(tmp_path, monkeypatch):
    existing = tmp_path / "out" / "case-x" / "20260101T000000Z-deadbeef"
    existing.mkdir(parents=True)
    sentinel = existing / "keep.txt"
    sentinel.write_text("original")
    with pytest.raises(InferenceError) as e:
        RunDirectory(tmp_path / "out", "case-x", "20260101T000000Z-deadbeef")
    assert e.value.issue.code == Code.OUTPUT_EXISTS
    monkeypatch.setattr(service_mod, "new_run_id", lambda: "20260101T000000Z-deadbeef")
    d, a, _, _ = write_case(tmp_path / "in")
    inf, *_ = tiny_model_setup(tmp_path)
    r = run_independent_inference(
        d, a, tmp_path / "out", inference_config_path=inf, device="cpu", case_id="case-x"
    )
    assert r.status == "failed" and r.exit_code == 8 and r.error["code"] == "E_OUTPUT_EXISTS"
    assert sentinel.read_text() == "original" and _files(existing) == {"keep.txt"}


def test_commit_refuses_if_final_folder_appeared(tmp_path):
    run = RunDirectory(tmp_path, "case-y", "20260101T000000Z-00000001")
    run.final.mkdir()
    with pytest.raises(InferenceError) as e:
        run.commit()
    assert e.value.issue.code == Code.OUTPUT_EXISTS


def test_failure_keeps_only_metadata_and_logs(tmp_path, monkeypatch):
    real, calls = schemas.problems, []

    def fail_on_metadata(obj, schema, names=()):
        calls.append(schema)
        return ["forced"] if schema == schemas.METADATA_SCHEMA else real(obj, schema, names)

    monkeypatch.setattr(service_mod.schemas, "problems", fail_on_metadata)
    r, *_ = _run(tmp_path)  # fails after mask, inputs and JSON were written to staging
    assert r.status == "failed" and r.exit_code == 7 and r.error["code"] == "E_OUTPUT_VERIFY"
    case_dir = tmp_path / "out" / r.case_id
    assert not (case_dir / r.run_id).exists()
    failed = case_dir / f".{r.run_id}.failed"
    assert _files(failed) == {"inference_metadata.json", "logs/inference.log"}
    meta = json.loads((failed / "inference_metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed" and meta["error"]["code"] == "E_OUTPUT_VERIFY"
    assert meta["outputs"]["files"] == {}


def test_input_errors_stop_before_any_model_or_image_work(tmp_path, monkeypatch):
    d, a, _, _ = write_case(tmp_path / "in")
    a.unlink()
    monkeypatch.setattr(
        service_mod, "load_inference_model", lambda *x, **k: pytest.fail("model loaded")
    )
    r = run_independent_inference(d, a, tmp_path / "out", device="cpu")
    assert r.exit_code == 3 and r.error["code"] == "E_INPUT_NOT_FOUND"
    failed = tmp_path / "out" / r.case_id / f".{r.run_id}.failed"
    assert _files(failed) == {"inference_metadata.json", "logs/inference.log"}


def test_sealed_test_case_is_refused_with_exit_9(tmp_path, monkeypatch):
    monkeypatch.setattr(service_mod, "validate_inputs", lambda *a: pytest.fail("validated"))
    sealed = tmp_path / "sub-strokecase0001_ses-0001_dwi.nii.gz"  # path string only
    r = run_independent_inference(sealed, tmp_path / "adc.nii.gz", tmp_path / "out")
    assert r.exit_code == 9 and r.error["code"] == "E_SEALED_TEST_CASE"


def test_strict_mode_promotes_warnings(tmp_path):
    r, *_ = _run(tmp_path, strict=True)  # header text -> W_HEADER_TEXT_PRESENT -> error
    assert r.status == "failed" and r.exit_code == 3 and r.error["code"] == "E_STRICT_WARNING"


@pytest.mark.parametrize("bad", ["../escape", "a/b", "x.nii.gz", "", "ü"])
def test_invalid_case_id_creates_nothing(tmp_path, bad):
    d, a, _, _ = write_case(tmp_path / "in")
    r = run_independent_inference(d, a, tmp_path / "out", case_id=bad)
    assert r.exit_code == 2 and r.error["code"] == "E_INVALID_ARGUMENT"
    assert not (tmp_path / "out").exists()


def test_same_input_gives_identical_results_across_runs(tmp_path):
    d, a, _, _ = write_case(tmp_path / "in")
    inf, *_ = tiny_model_setup(tmp_path)
    runs = [
        run_independent_inference(d, a, tmp_path / "out", inference_config_path=inf, device="cpu")
        for _ in range(2)
    ]
    assert runs[0].run_id != runs[1].run_id
    masks = [
        np.asanyarray(
            nib.load(tmp_path / "out" / r.case_id / r.run_id / "prediction_mask.nii.gz").dataobj
        )
        for r in runs
    ]
    np.testing.assert_array_equal(*masks)
    strip = ("case_id", "run_id")
    q0, q1 = ({k: v for k, v in r.quantification.items() if k not in strip} for r in runs)
    assert q0 == q1


def test_cli_json_output_and_exit_codes(tmp_path):
    d, a, _, _ = write_case(tmp_path / "in")
    inf, *_ = tiny_model_setup(tmp_path)
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "infer_case.py"), "--dwi", str(d)]
    ok = subprocess.run(
        [
            *cmd,
            "--adc",
            str(a),
            "--output",
            str(tmp_path / "out"),
            "--inference-config",
            str(inf),
            "--device",
            "cpu",
            "--json",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert ok.returncode == 0, ok.stderr
    res = json.loads(ok.stdout)
    assert res["status"] == "succeeded" and res["quantification"]["schema_version"]
    missing = subprocess.run(
        [*cmd, "--adc", str(tmp_path / "nope.nii.gz"), "--output", str(tmp_path / "out")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert missing.returncode == 3 and "E_INPUT_NOT_FOUND" in missing.stderr
    assert str(tmp_path) not in missing.stderr


def test_engine_does_not_import_web_or_database_packages():
    code = (
        "import sys, neuroscan_ai.inference; "
        "bad={'fastapi','starlette','pydantic','sqlalchemy','flask','django','requests'}; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "[]"


def test_contract_checker_catches_violations():
    base = {"schema_version": schemas.MANIFEST_SCHEMA}
    assert "missing key volumes" in schemas.problems(base, schemas.MANIFEST_SCHEMA)
    assert any("forbidden key" in p for p in schemas.problems({"dice": 1}, schemas.MANIFEST_SCHEMA))
    assert "absolute path in output" in schemas.problems(
        {"x": "C:\\Users\\a\\b.nii"}, schemas.MANIFEST_SCHEMA
    )
    assert "absolute path in output" in schemas.problems({"x": "/home/a"}, schemas.MANIFEST_SCHEMA)
    assert any(
        "forbidden phrase" in p
        for p in schemas.problems({"x": "Stroke probability 87%"}, schemas.MANIFEST_SCHEMA)
    )
    assert "input file name in output" in schemas.problems(
        {"x": "from Kim_DWI.nii.gz"}, schemas.MANIFEST_SCHEMA, ["Kim_DWI.nii.gz"]
    )
