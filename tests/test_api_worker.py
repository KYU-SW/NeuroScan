"""Phase 5B runs and the single GPU worker (fake Phase 5A engine): state transitions, queue
limits, cancellation, lease exclusivity, serial execution, recovery, OOM reload policy."""

from __future__ import annotations

import datetime as dt
import threading

import pytest
from sqlalchemy import select, update

from neuroscan_api.db.models import FileResource, InferenceRun, WorkerStatus, utcnow
from neuroscan_api.worker.loop import LeaseHeldError
from tests.api_helpers import (
    API,
    FakeEngine,
    case_files,
    client_for,
    make_settings,
    make_worker,
    upload,
)

pytestmark = [pytest.mark.api, pytest.mark.filterwarnings("ignore::DeprecationWarning")]


@pytest.fixture
def client(tmp_path):
    with client_for(make_settings(tmp_path)) as c:
        yield c


@pytest.fixture
def case_id(client, tmp_path):
    d, a = case_files(tmp_path)
    return upload(client, d, a).json()["case_id"]


def _new_case(client, tmp_path, i):
    d, a = case_files(tmp_path / f"c{i}")
    return upload(client, d, a).json()["case_id"]


def _run(client, case_id, **body):
    return client.post(f"{API}/cases/{case_id}/runs", json=body or None)


def _db(client):
    return client.app.state.ctx.db


def test_run_lifecycle_with_worker(client, case_id):
    r = _run(client, case_id)
    assert r.status_code == 202
    run = r.json()
    assert (
        run["status"] == "queued" and run["queue_position"] == 1 and run["worker_online"] is False
    )
    st = client.get(f"{API}/runs/{run['run_id']}").json()
    assert st["progress"] == 0.0 and st["current_stage"] is None
    assert client.get(f"{API}/runs/{run['run_id']}/result").json()["error"]["code"] == (
        "RUN_NOT_FINISHED"
    )
    engine = FakeEngine()
    w = make_worker(client, engine.api())
    try:
        assert w.run_once() is True
    finally:
        w.stop()
    st = client.get(f"{API}/runs/{run['run_id']}").json()
    assert st["status"] == "succeeded" and st["progress"] == 1.0 and st["current_stage"] == "done"
    assert st["model"]["checkpoint_sha256"] == "a" * 64 and st["error"] is None
    with _db(client).read() as s:
        roles = {
            f.role: f
            for f in s.scalars(select(FileResource).where(FileResource.run_id == run["run_id"]))
        }
    assert set(roles) == {
        "prediction_mask",
        "quantification_json",
        "manifest_json",
        "metadata_json",
        "input_dwi_deid",
        "input_adc_deid",
    }
    assert roles["prediction_mask"].shareable and not roles["input_dwi_deid"].shareable
    assert all(f.servable for f in roles.values())


def test_duplicate_active_run_and_result_exists_policy(client, case_id):
    first = _run(client, case_id).json()["run_id"]
    r = _run(client, case_id)
    assert r.status_code == 409 and r.json()["error"]["code"] == "RUN_ALREADY_ACTIVE"
    assert r.json()["error"]["details"]["run_id"] == first
    w = make_worker(client, FakeEngine().api())
    try:
        w.run_once()
        r = _run(client, case_id)
        assert r.status_code == 409 and r.json()["error"]["code"] == "RESULT_EXISTS"
        r = _run(client, case_id, allow_rerun=True)
        assert r.status_code == 202
        w.run_once()
    finally:
        w.stop()
    with _db(client).read() as s:
        runs = s.scalars(select(InferenceRun).where(InferenceRun.case_id == case_id)).all()
    assert [r.status for r in runs] == ["succeeded", "succeeded"]
    assert runs[0].run_dir_rel != runs[1].run_dir_rel  # a new folder, nothing overwritten


def test_queue_limit(tmp_path):
    with client_for(make_settings(tmp_path, limits={"max_queued_runs": 2})) as c:
        ids = [_new_case(c, tmp_path, i) for i in range(3)]
        assert _run(c, ids[0]).status_code == 202 and _run(c, ids[1]).status_code == 202
        r = _run(c, ids[2])
        assert r.status_code == 429 and r.json()["error"]["code"] == "QUEUE_FULL"
        assert r.headers["retry-after"] == "30"


def test_cancel_queued_only(client, case_id):
    rid = _run(client, case_id).json()["run_id"]
    r = client.post(f"{API}/runs/{rid}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert client.post(f"{API}/runs/{rid}/cancel").status_code == 200  # idempotent
    assert client.get(f"{API}/runs/{rid}/result").json()["error"]["code"] == "RUN_CANCELLED"
    rid2 = _run(client, case_id).json()["run_id"]
    w = make_worker(client, FakeEngine().api())
    try:
        assert w.claim() == (rid2, case_id)  # now running
        r = client.post(f"{API}/runs/{rid2}/cancel")
        assert r.status_code == 409 and r.json()["error"]["code"] == "RUN_NOT_CANCELLABLE"
        w.process(rid2, case_id)
    finally:
        w.stop()
    assert client.post(f"{API}/runs/{rid2}/cancel").status_code == 409  # finished


def test_delete_refuses_running_and_cancels_queued(client, case_id):
    rid = _run(client, case_id).json()["run_id"]
    w = make_worker(client, FakeEngine().api())
    try:
        w.claim()
        assert client.delete(f"{API}/cases/{case_id}?confirm=true").json()["error"]["code"] == (
            "RUN_ACTIVE"
        )
        w.process(rid, case_id)
    finally:
        w.stop()
    r = client.delete(f"{API}/cases/{case_id}?confirm=true")
    assert r.status_code == 200 and r.json()["files_deleted"] >= 8
    storage = client.app.state.ctx.storage
    assert not (storage.uploads / case_id).exists() and not (storage.runs / case_id).exists()


def test_failed_engine_result_is_reported(client, case_id):
    rid = _run(client, case_id).json()["run_id"]
    w = make_worker(client, FakeEngine(status="failed").api())
    try:
        w.run_once()
    finally:
        w.stop()
    st = client.get(f"{API}/runs/{rid}").json()
    assert st["status"] == "failed" and st["error"]["code"] == "E_BRAIN_MASK"
    assert st["error"]["retryable"] is False
    r = client.get(f"{API}/runs/{rid}/result")
    assert r.status_code == 409 and r.json()["error"]["details"]["engine_code"] == "E_BRAIN_MASK"


def test_engine_exception_fails_run_but_worker_continues(client, tmp_path):
    ids = [_new_case(client, tmp_path, i) for i in range(2)]
    rids = [_run(client, c).json()["run_id"] for c in ids]
    engine = FakeEngine()
    real = engine.run_inference

    def flaky(*a, **k):
        if engine.runs == 0:
            engine.runs += 1
            raise RuntimeError("C:\\Users\\x\\boom")
        return real(*a, **k)

    engine.run_inference = flaky
    w = make_worker(client, engine.api())
    try:
        assert w.run_once() and w.run_once()
    finally:
        w.stop()
    a, b = (client.get(f"{API}/runs/{r}").json() for r in rids)
    assert a["status"] == "failed" and a["error"]["code"] == "E_INTERNAL"
    assert "Users" not in str(a) and b["status"] == "succeeded"


def test_progress_stages_are_recorded(client, case_id):
    rid = _run(client, case_id).json()["run_id"]
    seen = []

    def watch(loaded, n):
        with _db(client).read() as s:
            r = s.get(InferenceRun, rid)
            seen.append((r.current_stage, r.progress))

    w = make_worker(client, FakeEngine(on_run=watch).api())
    try:
        w.run_once()
    finally:
        w.stop()
    assert seen == [("quantifying", 0.70)]


# --- single GPU worker -----------------------------------------------------------------------


def test_second_worker_is_refused_while_lease_is_fresh(client):
    w1 = make_worker(client, FakeEngine().api())
    try:
        w2 = make_worker(client, FakeEngine().api(), start=False)
        with pytest.raises(LeaseHeldError):
            w2.start()
    finally:
        w1.stop()
    w3 = make_worker(client, FakeEngine().api())  # stopped lease can be taken
    w3.stop()


def test_stale_lease_can_be_taken_over(client):
    w1 = make_worker(client, FakeEngine().api())
    w1._stop.set()
    w1._hb_thread.join()
    with _db(client).write() as s:
        s.execute(update(WorkerStatus).values(heartbeat_at=utcnow() - dt.timedelta(seconds=300)))
    w2 = make_worker(client, FakeEngine().api())
    w2.stop()
    with _db(client).read() as s:
        assert s.get(WorkerStatus, 1).worker_id == w2.worker_id


def test_serial_execution_and_single_model_load(client, tmp_path):
    ids = [_new_case(client, tmp_path, i) for i in range(4)]
    for c in ids:
        _run(client, c)
    engine = FakeEngine(delay=0.05)
    w = make_worker(client, engine.api())
    t = threading.Thread(target=w.run_forever)
    t.start()
    try:
        other = make_worker(client, FakeEngine().api(), start=False)
        with pytest.raises(LeaseHeldError):
            other.start()  # a second GPU worker cannot join while the first is alive
        deadline = utcnow() + dt.timedelta(seconds=20)
        while utcnow() < deadline:
            with _db(client).read() as s:
                left = s.scalars(
                    select(InferenceRun).where(InferenceRun.status.in_(("queued", "running")))
                ).all()
            if not left:
                break
    finally:
        w.stop()
        t.join(timeout=10)
    assert engine.runs == 4 and engine.max_active == 1
    assert engine.loads == 1 and w.load_count == 1
    with _db(client).read() as s:
        assert {r.status for r in s.scalars(select(InferenceRun))} == {"succeeded"}
        assert s.get(WorkerStatus, 1).load_count == 1


def test_stale_and_timed_out_runs_are_failed_by_the_api(client, tmp_path):
    a, b = (_new_case(client, tmp_path, i) for i in range(2))
    ra, rb = _run(client, a).json()["run_id"], _run(client, b).json()["run_id"]
    old = utcnow() - dt.timedelta(seconds=1000)
    with _db(client).write() as s:
        s.execute(
            update(InferenceRun)
            .where(InferenceRun.id == ra)
            .values(status="running", started_at=utcnow(), heartbeat_at=old, worker_id="x")
        )
        s.execute(
            update(InferenceRun)
            .where(InferenceRun.id == rb)
            .values(status="running", started_at=old, heartbeat_at=utcnow(), worker_id="x")
        )
    assert client.get(f"{API}/runs/{ra}").json()["error"]["code"] == "E_WORKER_LOST"
    assert client.get(f"{API}/runs/{rb}").json()["error"]["code"] == "E_TIMEOUT"


def test_late_worker_result_never_overwrites_api_decision(client, case_id):
    rid = _run(client, case_id).json()["run_id"]

    def api_marks_lost(loaded, n):
        with _db(client).write() as s:
            s.execute(
                update(InferenceRun)
                .where(InferenceRun.id == rid)
                .values(status="failed", error_code="E_WORKER_LOST")
            )

    w = make_worker(client, FakeEngine(on_run=api_marks_lost).api())
    try:
        w.run_once()
    finally:
        w.stop()
    st = client.get(f"{API}/runs/{rid}").json()
    assert st["status"] == "failed" and st["error"]["code"] == "E_WORKER_LOST"
    with _db(client).read() as s:
        assert s.scalars(select(FileResource).where(FileResource.run_id == rid)).all() == []


def test_restart_recovers_running_runs_and_cleans_partials(client, case_id):
    rid = _run(client, case_id).json()["run_id"]
    with _db(client).write() as s:
        s.execute(
            update(InferenceRun)
            .where(InferenceRun.id == rid)
            .values(status="running", worker_id="dead-worker", heartbeat_at=utcnow())
        )
    storage = client.app.state.ctx.storage
    partial = storage.runs / case_id / ".20260101T000000Z-deadbeef.partial"
    (partial / "inputs").mkdir(parents=True)
    (partial / "logs").mkdir()
    (partial / "prediction_mask.nii.gz").write_bytes(b"m")
    (partial / "inputs" / "dwi.nii.gz").write_bytes(b"d")
    (partial / "logs" / "inference.log").write_text("log")
    w = make_worker(client, FakeEngine().api())
    w.stop()
    st = client.get(f"{API}/runs/{rid}").json()
    assert st["status"] == "failed" and st["error"]["code"] == "E_WORKER_RESTARTED"
    abandoned = storage.runs / case_id / ".20260101T000000Z-deadbeef.abandoned"
    assert not partial.exists() and abandoned.is_dir()
    assert sorted(
        p.relative_to(abandoned).as_posix() for p in abandoned.rglob("*") if p.is_file()
    ) == ["logs/inference.log"]


# --- CUDA OOM reload policy ------------------------------------------------------------------


def _move_to_cpu(loaded, n):
    loaded.model.param.device.type = "cpu"  # what Phase 5A does on an OOM CPU fallback
    return [{"code": "W_CPU_FALLBACK", "message": "GPU memory was insufficient; ran on CPU."}]


def test_cpu_fallback_triggers_a_verified_reload(client, case_id):
    rid = _run(client, case_id).json()["run_id"]
    engine = FakeEngine(on_run=_move_to_cpu)
    w = make_worker(client, engine.api())
    first = w.loaded
    try:
        w.run_once()
        assert engine.loads == 2 and w.loaded is not first
        assert next(w.loaded.model.parameters()).device.type == "cuda"
        assert not w.degraded
    finally:
        w.stop()
    st = client.get(f"{API}/runs/{rid}").json()
    assert st["status"] == "succeeded" and [x["code"] for x in st["warnings"]] == ["W_CPU_FALLBACK"]


def test_device_mismatch_without_warning_also_reloads(client, case_id):
    _run(client, case_id)

    def silent_move(loaded, n):
        loaded.model.param.device.type = "cpu"

    engine = FakeEngine(on_run=silent_move)
    w = make_worker(client, engine.api())
    try:
        w.run_once()
    finally:
        w.stop()
    assert engine.loads == 2


def test_failed_reload_degrades_worker_and_stops_claiming(client, tmp_path):
    a, b = (_new_case(client, tmp_path, i) for i in range(2))
    ra, rb = _run(client, a).json()["run_id"], _run(client, b).json()["run_id"]
    engine = FakeEngine(on_run=_move_to_cpu, fail_load_after=1)
    w = make_worker(client, engine.api())
    try:
        assert w.run_once() is True
        assert w.degraded and w.loaded is None
        assert w.run_once() is False  # no further claims
        assert client.get(f"{API}/models/current").status_code == 503
        assert client.get(f"{API}/health").json()["worker"] == "degraded"
    finally:
        w.stop()
    st = client.get(f"{API}/runs/{ra}").json()
    assert st["status"] == "failed" and st["error"]["code"] == "E_MODEL_RELOAD_FAILED"
    assert client.get(f"{API}/runs/{rb}").json()["status"] == "queued"
    with _db(client).read() as s:
        assert s.scalars(select(FileResource).where(FileResource.run_id == ra)).all() == []


def test_initial_model_load_failure_starts_degraded(client, case_id):
    _run(client, case_id)
    engine = FakeEngine(fail_load_after=0)
    w = make_worker(client, engine.api())
    try:
        assert w.degraded and w.run_once() is False
    finally:
        w.stop()
