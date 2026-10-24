"""Phase 5B end-to-end with the real Phase 5A engine (tiny checkpoint, CPU): result allow-list
mapping, file_id downloads, storage confinement, privacy and CLI equivalence."""

from __future__ import annotations

import gzip
import io
import json
import os

import nibabel as nib
import numpy as np
import pytest
from sqlalchemy import select, update

from neuroscan_api.db.models import FileResource
from neuroscan_api.safety import violations
from neuroscan_api.worker.loop import default_engine
from tests.api_helpers import (
    API,
    PHI_NAMES,
    PHI_TEXT,
    case_files,
    client_for,
    make_settings,
    make_worker,
    upload,
)
from tests.inference_helpers import tiny_model_setup

pytestmark = [pytest.mark.api, pytest.mark.filterwarnings("ignore::DeprecationWarning")]


@pytest.fixture
def env(tmp_path):
    inf, *_ = tiny_model_setup(tmp_path, bias=50.0)
    settings = make_settings(tmp_path, inference_config=str(inf))
    with client_for(settings) as c:
        d, a = case_files(tmp_path)
        case_id = upload(c, d, a).json()["case_id"]
        run_id = c.post(f"{API}/cases/{case_id}/runs").json()["run_id"]
        w = make_worker(c, default_engine())
        try:
            assert w.run_once()
        finally:
            w.stop()
        yield {
            "client": c,
            "case_id": case_id,
            "run_id": run_id,
            "dwi": d,
            "adc": a,
            "inf": inf,
            "tmp": tmp_path,
        }


def _rows(client, run_id):
    with client.app.state.ctx.db.read() as s:
        return {
            f.role: f for f in s.scalars(select(FileResource).where(FileResource.run_id == run_id))
        }


def test_result_is_an_explicit_safe_mapping(env):
    c, rid = env["client"], env["run_id"]
    st = c.get(f"{API}/runs/{rid}").json()
    assert st["status"] == "succeeded", st
    r = c.get(f"{API}/runs/{rid}/result")
    assert r.status_code == 200
    res = r.json()
    assert res["lesion"]["detected"] is True and res["lesion"]["total_volume_ml"] > 0
    assert res["lesion"]["prediction_threshold"] == 0.5 and res["lesion"]["display_text"] is None
    assert res["location"]["hemisphere"]["estimated"] is True
    assert res["research_score_summary"]["calibrated"] is False
    assert res["vascular_territory"] is None and res["disclaimer"]["en"]
    assert {v["id"] for v in res["viewer"]["volumes"]} == {"dwi", "adc", "prediction_mask"}
    for v in res["viewer"]["volumes"]:
        assert v["url"] == f"{API}/files/{v['file_id']}" and "uri" not in v
    assert violations(res, extra_strings=[*PHI_NAMES, "Hong_Gildong", PHI_TEXT]) == []
    for bad in ("checkpoint_path", "rel_path", "run_dir", "Traceback", str(env["tmp"]), "dice"):
        assert bad not in r.text


def test_deidentified_inputs_and_mask_are_downloadable(env):
    c, rid = env["client"], env["run_id"]
    vols = {v["id"]: v for v in c.get(f"{API}/runs/{rid}/result").json()["viewer"]["volumes"]}
    for vid, src in (("dwi", env["dwi"]), ("adc", env["adc"])):
        r = c.get(vols[vid]["url"])
        assert r.status_code == 200 and r.headers["content-type"] == "application/gzip"
        assert f'filename="{vid}.nii.gz"' in r.headers["content-disposition"]
        assert r.headers["cache-control"] == "no-store"
        img = nib.Nifti1Image.from_bytes(gzip.decompress(r.content))
        np.testing.assert_array_equal(
            np.asanyarray(img.dataobj), nib.load(src).get_fdata(dtype=np.float32)
        )
        assert bytes(img.header["descrip"].tobytes()).strip(b"\x00 ") == b""  # header text gone
        assert PHI_TEXT.encode() not in gzip.decompress(r.content)
    m = c.get(vols["prediction_mask"]["url"])
    mask = nib.Nifti1Image.from_bytes(gzip.decompress(m.content))
    data = np.asanyarray(mask.dataobj)
    assert data.dtype == np.uint8 and set(np.unique(data).tolist()) <= {0, 1}
    assert data.shape == nib.load(env["dwi"]).shape
    np.testing.assert_allclose(mask.affine, nib.load(env["dwi"]).affine, atol=1e-5)


def test_raw_uploads_logs_and_unknown_ids_are_not_served(env):
    c = env["client"]
    with c.app.state.ctx.db.read() as s:
        raw = s.scalars(select(FileResource).where(FileResource.role.like("upload_%"))).all()
    assert len(raw) == 2
    for f in raw:
        r = c.get(f"{API}/files/{f.id}")
        assert r.status_code == 404 and r.json()["error"]["code"] == "FILE_NOT_FOUND"
    assert "inference_log" not in _rows(c, env["run_id"])
    for fid in ("00000000-0000-4000-8000-000000000000", "../../etc/passwd", "x"):
        assert c.get(f"{API}/files/{fid}").status_code == 404


@pytest.mark.parametrize(
    "rel", ["../../../outside.txt", "C:/Windows/win.ini", "/etc/passwd", "..\\..\\x", ""]
)
def test_paths_outside_storage_are_never_served(env, rel):
    c, rid = env["client"], env["run_id"]
    mask = _rows(c, rid)["prediction_mask"]
    with c.app.state.ctx.db.write() as s:
        s.execute(update(FileResource).where(FileResource.id == mask.id).values(rel_path=rel))
    r = c.get(f"{API}/files/{mask.id}")
    assert r.status_code == 404 and str(env["tmp"]) not in r.text


def test_symlinks_are_refused(env):
    c, rid = env["client"], env["run_id"]
    mask = _rows(c, rid)["prediction_mask"]
    storage = c.app.state.ctx.storage
    link = storage.runs / env["case_id"] / "link.nii.gz"
    try:
        os.symlink(storage.resolve("runs", mask.rel_path), link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this system")
    with c.app.state.ctx.db.write() as s:
        s.execute(
            update(FileResource)
            .where(FileResource.id == mask.id)
            .values(rel_path=f"{env['case_id']}/link.nii.gz")
        )
    assert c.get(f"{API}/files/{mask.id}").status_code == 404


def test_tampered_files_fail_integrity(env):
    c, rid = env["client"], env["run_id"]
    rows = _rows(c, rid)
    storage = c.app.state.ctx.storage
    p = storage.resolve("runs", rows["prediction_mask"].rel_path)
    p.write_bytes(p.read_bytes() + b"x")
    r = c.get(f"{API}/files/{rows['prediction_mask'].id}")
    assert r.status_code == 500 and r.json()["error"]["code"] == "FILE_INTEGRITY_ERROR"
    q = storage.resolve("runs", rows["quantification_json"].rel_path)
    q.write_text(q.read_text(encoding="utf-8").replace("0.5", "0.6"), encoding="utf-8")
    r = c.get(f"{API}/runs/{rid}/result")
    assert r.status_code == 500 and r.json()["error"]["code"] == "RESULT_INTEGRITY_ERROR"


def test_missing_output_is_404(env):
    c, rid = env["client"], env["run_id"]
    rows = _rows(c, rid)
    c.app.state.ctx.storage.resolve("runs", rows["manifest_json"].rel_path).unlink()
    r = c.get(f"{API}/runs/{rid}/result")
    assert r.status_code == 404 and r.json()["error"]["code"] == "OUTPUT_NOT_FOUND"


def test_api_matches_cli_service_output(env):
    from neuroscan_ai.inference import run_independent_inference

    c, rid = env["client"], env["run_id"]
    vols = {v["id"]: v for v in c.get(f"{API}/runs/{rid}/result").json()["viewer"]["volumes"]}
    api_mask = np.asanyarray(
        nib.Nifti1Image.from_bytes(
            gzip.decompress(c.get(vols["prediction_mask"]["url"]).content)
        ).dataobj
    )
    cli = run_independent_inference(
        env["dwi"],
        env["adc"],
        env["tmp"] / "cli_out",
        inference_config_path=env["inf"],
        device="cpu",
    )
    cli_mask = np.asanyarray(nib.load(os.path.join(cli.run_dir, "prediction_mask.nii.gz")).dataobj)
    np.testing.assert_array_equal(api_mask, cli_mask)
    res = c.get(f"{API}/runs/{rid}/result").json()
    assert res["lesion"]["total_volume_ml"] == cli.quantification["total_lesion_volume_ml"]


def test_no_lesion_display_text(tmp_path):
    inf, *_ = tiny_model_setup(tmp_path, bias=-50.0)
    with client_for(make_settings(tmp_path, inference_config=str(inf))) as c:
        d, a = case_files(tmp_path)
        cid = upload(c, d, a).json()["case_id"]
        rid = c.post(f"{API}/cases/{cid}/runs").json()["run_id"]
        w = make_worker(c, default_engine())
        try:
            w.run_once()
        finally:
            w.stop()
        res = c.get(f"{API}/runs/{rid}/result").json()
    assert res["lesion"]["detected"] is False
    assert "does not mean that no lesion is present" in res["lesion"]["display_text"]
    assert res["viewer"]["render_3d"]["empty_mask"] is True


def test_metadata_download_has_no_absolute_paths(env):
    c, rid = env["client"], env["run_id"]
    meta = _rows(c, rid)["metadata_json"]
    r = c.get(f"{API}/files/{meta.id}")
    assert r.status_code == 200
    body = json.load(io.BytesIO(r.content))
    assert body["ground_truth_used"] is False and str(env["tmp"]) not in r.text
    assert "Hong_Gildong" not in r.text


def test_junction_pointing_outside_storage_is_refused(env):
    import subprocess
    import sys

    if sys.platform != "win32":
        pytest.skip("directory junctions are Windows-specific")
    c, rid = env["client"], env["run_id"]
    outside = env["tmp"] / "outside_dir"
    outside.mkdir()
    (outside / "secret.nii.gz").write_bytes(b"secret")
    storage = c.app.state.ctx.storage
    junction = storage.runs / env["case_id"] / "jn"
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)], capture_output=True
    )
    if made.returncode != 0:
        pytest.skip("could not create a junction")
    mask = _rows(c, rid)["prediction_mask"]
    with c.app.state.ctx.db.write() as s:
        s.execute(
            update(FileResource)
            .where(FileResource.id == mask.id)
            .values(rel_path=f"{env['case_id']}/jn/secret.nii.gz", size_bytes=6)
        )
    r = c.get(f"{API}/files/{mask.id}")
    assert r.status_code == 404 and b"secret" not in r.content


def test_storage_resolve_refuses_symlinks(tmp_path, monkeypatch):
    from neuroscan_api.storage import Storage, StorageError

    st = Storage(tmp_path / "s")
    (st.runs / "c").mkdir()
    (st.runs / "c" / "f.json").write_text("{}")
    assert st.resolve("runs", "c/f.json").is_file()
    monkeypatch.setattr("neuroscan_api.storage.os.path.islink", lambda p: str(p).endswith("f.json"))
    with pytest.raises(StorageError):
        st.resolve("runs", "c/f.json")
