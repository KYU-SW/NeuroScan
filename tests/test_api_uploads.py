"""Phase 5B uploads: streaming limits, minimal NIfTI checks, sealed filename guard, storage
confinement, case lookup and deletion."""

from __future__ import annotations

import gzip
import io

import pytest
from sqlalchemy import select

from neuroscan_api import uploads as uploads_mod
from neuroscan_api.db.models import Case, FileResource
from tests.api_helpers import (
    API,
    PHI_NAMES,
    PHI_TEXT,
    case_files,
    client_for,
    make_settings,
    upload,
)

pytestmark = [pytest.mark.api, pytest.mark.filterwarnings("ignore::DeprecationWarning")]


@pytest.fixture
def client(tmp_path):
    with client_for(make_settings(tmp_path)) as c:
        yield c


def _storage(client):
    return client.app.state.ctx.storage


def _no_temp_left(client):
    return not any(_storage(client).temp.iterdir())


def test_upload_creates_case_with_server_names_and_no_client_names(client, tmp_path):
    d, a = case_files(tmp_path)
    r = upload(client, d, a)
    assert r.status_code == 201, r.text
    body = r.json()
    cid = body["case_id"]
    assert cid.startswith("case-") and body["status"] == "ready" and body["runs"] == []
    assert [i["role"] for i in body["inputs"]] == ["dwi", "adc"]
    assert all(i["downloadable"] is False and len(i["sha256"]) == 64 for i in body["inputs"])
    assert r.headers["location"] if "location" in r.headers else True
    case_dir = _storage(client).uploads / cid
    assert sorted(p.name for p in case_dir.iterdir()) == ["adc.nii.gz", "dwi.nii.gz"]
    assert (case_dir / "dwi.nii.gz").read_bytes() == d.read_bytes()
    assert "Hong" not in r.text and PHI_TEXT not in r.text and str(tmp_path) not in r.text
    with client.app.state.ctx.db.read() as s:
        rows = s.scalars(select(FileResource).where(FileResource.case_id == cid)).all()
    assert {f.role for f in rows} == {"upload_dwi_raw", "upload_adc_raw"}
    assert all(not f.servable and not f.shareable for f in rows)
    assert all("Hong" not in f.rel_path for f in rows)
    assert _no_temp_left(client)
    assert client.get(f"{API}/cases/{cid}").json()["case_id"] == cid


def test_plain_nii_is_accepted(client, tmp_path):
    d, a = case_files(tmp_path, names=("x_dwi.nii", "x_adc.nii"))
    r = upload(client, d, a)
    assert r.status_code == 201
    assert (_storage(client).uploads / r.json()["case_id"] / "dwi.nii").is_file()


@pytest.mark.parametrize(
    ("files", "code"),
    [
        ({"dwi": ("d.nii.gz", b"x")}, "MISSING_FILE"),
        (
            {"dwi": ("d.nii.gz", b"x"), "adc": ("a.nii.gz", b"x"), "flair": ("f.nii.gz", b"x")},
            "UNEXPECTED_FIELD",
        ),
        ({"dwi": ("d.txt", b"x"), "adc": ("a.nii.gz", b"x")}, "UNSUPPORTED_EXTENSION"),
        ({"dwi": ("d.nii.gz", b""), "adc": ("a.nii.gz", b"")}, "MISSING_FILE"),
    ],
)
def test_bad_forms(client, files, code):
    r = client.post(f"{API}/cases", files=files)
    assert r.status_code == 400 and r.json()["error"]["code"] == code
    assert _no_temp_left(client) and not any(_storage(client).uploads.iterdir())


def test_non_multipart_and_missing_length(client):
    r = client.post(f"{API}/cases", json={"dwi": "x"})
    assert r.status_code == 415
    r = client.post(
        f"{API}/cases",
        content=iter([b"--x\r\n"]),  # streamed body: no Content-Length
        headers={"content-type": "multipart/form-data; boundary=x"},
    )
    assert r.status_code == 411 and r.json()["error"]["code"] == "LENGTH_REQUIRED"


def test_request_and_file_size_limits(tmp_path):
    d, a = case_files(tmp_path)
    with client_for(make_settings(tmp_path / "r", limits={"max_request_bytes": 1000})) as c:
        r = upload(c, d, a)
        assert r.status_code == 413 and r.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    with client_for(make_settings(tmp_path / "f", limits={"max_file_bytes": 2000})) as c:
        r = upload(c, d, a)
        assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"
        assert _no_temp_left(c) and not any(_storage(c).uploads.iterdir())


def test_gzip_bomb_corrupt_gzip_and_bad_header(tmp_path, client):
    d, a = case_files(tmp_path)
    with client_for(make_settings(tmp_path / "b", limits={"max_uncompressed_bytes": 50_000})) as c:
        r = upload(c, d, a)
        assert r.status_code == 422 and r.json()["error"]["code"] == "ARCHIVE_TOO_LARGE"
        assert _no_temp_left(c)
    for payload, check in (
        (b"\x1f\x8bnot really gzip" * 10, "gzip"),
        (gzip.compress(b"\x00" * 400), "sizeof_hdr"),
    ):
        r = client.post(
            f"{API}/cases",
            files={"dwi": ("d.nii.gz", payload), "adc": ("a.nii.gz", a.read_bytes())},
        )
        assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_NIFTI"
        assert r.json()["error"]["details"]["check"] == check
    assert _no_temp_left(client)


def test_header_dimension_limits(tmp_path):
    d, a = case_files(tmp_path)  # 48 x 48 x 10
    with client_for(make_settings(tmp_path / "ax", limits={"max_axis": 40})) as c:
        r = upload(c, d, a)
        assert r.status_code == 422 and r.json()["error"]["details"]["check"] == "axis size"


def test_sealed_test_filename_is_refused_before_reading(client, tmp_path, monkeypatch):
    d, a = case_files(tmp_path)

    async def never(*args, **kwargs):
        pytest.fail("upload content was read")

    monkeypatch.setattr(uploads_mod, "save_stream", never)
    monkeypatch.setattr("neuroscan_api.services.cases.save_stream", never)
    r = upload(client, d, a, dwi_name="sub-strokecase0001_ses-0001_dwi.nii.gz")  # test split ID
    assert r.status_code == 403 and r.json()["error"]["code"] == "SEALED_TEST_CASE"
    assert "strokecase" not in r.text
    assert _no_temp_left(client) and not any(_storage(client).uploads.iterdir())


def test_validation_case_filename_is_allowed(client, tmp_path):
    d, a = case_files(tmp_path)
    r = upload(client, d, a, dwi_name="sub-strokecase0086_ses-0001_dwi.nii.gz")
    assert r.status_code == 201


def test_traversal_file_names_cannot_escape(client, tmp_path):
    d, a = case_files(tmp_path)
    r = upload(client, d, a, dwi_name="../../../evil.nii.gz", adc_name="C:\\Windows\\a.nii.gz")
    assert r.status_code == 201
    root = _storage(client).root
    assert not (root.parent / "evil.nii.gz").exists() and not (root / "evil.nii.gz").exists()
    names = sorted(p.name for p in (_storage(client).uploads / r.json()["case_id"]).iterdir())
    assert names == ["adc.nii.gz", "dwi.nii.gz"]


def test_insufficient_storage(client, tmp_path, monkeypatch):
    d, a = case_files(tmp_path)
    monkeypatch.setattr(
        "neuroscan_api.storage.shutil.disk_usage", lambda p: type("U", (), {"free": 0})()
    )
    client.app.state.ctx.settings.limits.min_free_bytes = 1
    r = upload(client, d, a)
    assert r.status_code == 507 and r.json()["error"]["code"] == "INSUFFICIENT_STORAGE"
    assert not any(_storage(client).uploads.iterdir())


def test_db_failure_rolls_back_uploaded_files(client, tmp_path, monkeypatch):
    d, a = case_files(tmp_path)
    from neuroscan_api.services import cases as cases_mod

    def broken_audit(*args, **kwargs):
        raise RuntimeError("db write failed")

    monkeypatch.setattr(cases_mod, "audit", broken_audit)
    from fastapi.testclient import TestClient

    r = TestClient(client.app, raise_server_exceptions=False).post(
        f"{API}/cases",
        files={"dwi": (d.name, d.read_bytes()), "adc": (a.name, a.read_bytes())},
    )
    assert r.status_code == 500
    assert not any(_storage(client).uploads.iterdir()) and _no_temp_left(client)
    with client.app.state.ctx.db.read() as s:
        assert s.scalars(select(Case)).all() == []


def test_unknown_and_malformed_case_ids_are_404(client):
    for cid in ("case-000000000000", "../etc", "case-XYZ", "x" * 300):
        r = client.get(f"{API}/cases/{cid}")
        assert r.status_code == 404


def test_case_delete_requires_confirm_and_removes_files(client, tmp_path):
    d, a = case_files(tmp_path)
    cid = upload(client, d, a).json()["case_id"]
    assert client.delete(f"{API}/cases/{cid}").json()["error"]["code"] == "CONFIRM_REQUIRED"
    r = client.delete(f"{API}/cases/{cid}?confirm=true")
    assert r.status_code == 200 and r.json()["files_deleted"] == 2
    assert not (_storage(client).uploads / cid).exists()
    assert client.get(f"{API}/cases/{cid}").status_code == 404
    with client.app.state.ctx.db.read() as s:
        assert s.scalars(select(FileResource).where(FileResource.case_id == cid)).all() == []
    assert client.delete(f"{API}/cases/{cid}?confirm=true").status_code == 404


def test_upload_does_not_buffer_whole_file(monkeypatch, tmp_path, client):
    """save_stream reads at most 1 MiB per call."""
    d, a = case_files(tmp_path)
    sizes = []
    from starlette.datastructures import UploadFile

    real = UploadFile.read

    async def spy(self, size=-1):
        sizes.append(size)
        return await real(self, size)

    monkeypatch.setattr(UploadFile, "read", spy)
    assert upload(client, d, a).status_code == 201
    assert sizes and all(0 < s <= uploads_mod.CHUNK for s in sizes)
    assert io.DEFAULT_BUFFER_SIZE <= uploads_mod.CHUNK
    assert PHI_NAMES[0] != "dwi.nii.gz"
