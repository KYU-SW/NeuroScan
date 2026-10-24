"""Phase 5B: health, model info, localhost / Host / CORS policy, docs exposure, error format,
torch-free API process."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.api_helpers import API, FakeEngine, client_for, make_settings, make_worker

pytestmark = [
    pytest.mark.api,
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]
from tests.conftest import REPO_ROOT  # noqa: E402


@pytest.fixture
def client(tmp_path):
    with client_for(make_settings(tmp_path)) as c:
        yield c


def test_health_without_worker_is_degraded(client):
    r = client.get(f"{API}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded" and body["worker"] == "offline" and body["database"] == "ok"
    assert body["queue_depth"] == 0 and "diagnosis" in body["disclaimer"]["en"]


def test_model_info_requires_a_ready_worker_and_hides_paths(client):
    r = client.get(f"{API}/models/current")
    assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_READY"
    w = make_worker(client, FakeEngine().api())
    try:
        r = client.get(f"{API}/models/current")
        assert r.status_code == 200
        m = r.json()
        assert m["checkpoint_sha256"] == "a" * 64 and m["input_channels"] == 6
        assert m["context_slices"] == 1 and m["threshold"] == 0.5
        assert "checkpoint_path" not in m and "path" not in str(m).lower()
        assert client.get(f"{API}/health").json()["status"] == "ok"
    finally:
        w.stop()
    assert client.get(f"{API}/health").json()["worker"] == "offline"


def test_disallowed_host_is_rejected(client):
    r = client.get(f"{API}/health", headers={"host": "evil.example.com"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "HOST_NOT_ALLOWED"


def test_cors_blocked_by_default_and_explicit_origin_allowed(tmp_path, client):
    r = client.get(f"{API}/health", headers={"origin": "http://localhost:5173"})
    assert "access-control-allow-origin" not in r.headers
    s = make_settings(tmp_path / "cors", cors_origins=["http://localhost:5173"])
    with client_for(s) as c:
        r = c.get(f"{API}/health", headers={"origin": "http://localhost:5173"})
        assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
        r = c.get(f"{API}/health", headers={"origin": "http://evil.example.com"})
        assert "access-control-allow-origin" not in r.headers


def test_docs_only_in_debug_mode(tmp_path, client):
    assert client.get(f"{API}/docs").status_code == 200
    spec = client.get(f"{API}/openapi.json").json()
    assert {"/api/v1/cases", "/api/v1/files/{file_id}", "/api/v1/runs/{run_id}/result"} <= set(
        spec["paths"]
    )
    with client_for(make_settings(tmp_path / "prod", debug=False)) as c:
        assert c.get(f"{API}/docs").status_code == 404
        assert c.get(f"{API}/openapi.json").status_code == 404


def test_safe_default_headers_and_json_errors(client):
    r = client.get(f"{API}/health")
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff" and len(r.headers["x-request-id"]) == 32
    r = client.get(f"{API}/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"


def test_validation_errors_do_not_echo_input(client):
    r = client.post(
        f"{API}/cases/case-000000000000/runs", json={"allow_rerun": "Hong_Gildong", "x": 1}
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "Hong_Gildong" not in r.text


def test_internal_errors_hide_details(tmp_path, monkeypatch):
    from neuroscan_api import routes

    def boom(*a, **k):
        raise RuntimeError("C:\\Users\\someone\\secret.nii.gz")

    monkeypatch.setattr(routes.cases, "get_case", boom)
    with client_for(make_settings(tmp_path)) as c:
        c.app.router  # noqa: B018
        from fastapi.testclient import TestClient

        tc = TestClient(c.app, raise_server_exceptions=False)
        r = tc.get(f"{API}/cases/case-000000000000")
    assert r.status_code == 500 and r.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "secret" not in r.text and "Users" not in r.text and "Traceback" not in r.text


def test_launcher_refuses_non_loopback(monkeypatch, tmp_path):
    import neuroscan_api.__main__ as launcher
    from neuroscan_api.settings import is_loopback

    assert is_loopback("127.0.0.1") and is_loopback("localhost") and is_loopback("::1")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.0.10")
    monkeypatch.setattr(launcher, "load_settings", lambda: make_settings(tmp_path, host="0.0.0.0"))
    monkeypatch.setattr(launcher.uvicorn, "run", lambda *a, **k: pytest.fail("server started"))
    assert launcher.main() == 2


def test_api_process_never_imports_torch(tmp_path):
    code = (
        "import sys; sys.path.insert(0, 'backend');"
        "from neuroscan_api.main import create_app;"
        "from neuroscan_api.settings import load_settings;"
        f"create_app(load_settings(storage_root=r'{tmp_path / 's'}'));"
        "print(sorted(m for m in ('torch', 'monai', 'neuroscan_ai.inference') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "backend")},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
