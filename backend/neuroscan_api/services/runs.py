"""Run creation, status, queued-only cancellation and stale-run recovery.

All state transitions are compare-and-set updates inside ``BEGIN IMMEDIATE`` transactions, so the
API and the worker can never both move the same run.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import func, select, update

from neuroscan_api import API_PREFIX
from neuroscan_api.context import AppContext, audit, iso, new_uuid, worker_state
from neuroscan_api.db.models import ACTIVE_STATUSES, Case, InferenceRun, WorkerStatus, utcnow
from neuroscan_api.errors import ApiError, engine_retryable
from neuroscan_api.schemas import (
    RunCreateResponse,
    RunError,
    RunModel,
    RunStatusResponse,
    WarningItem,
)


def run_not_found() -> ApiError:
    return ApiError(404, "RUN_NOT_FOUND", "Run not found.")


def _links(run: InferenceRun) -> dict[str, str]:
    return {
        "self": f"{API_PREFIX}/runs/{run.id}",
        "result": f"{API_PREFIX}/runs/{run.id}/result",
        "case": f"{API_PREFIX}/cases/{run.case_id}",
    }


def _queue_position(s, run: InferenceRun) -> int | None:
    if run.status != "queued":
        return None
    ahead = s.scalar(
        select(func.count())
        .select_from(InferenceRun)
        .where(InferenceRun.status == "queued", InferenceRun.created_at < run.created_at)
    )
    return int(ahead) + 1


def create_run(ctx: AppContext, case_id: str, allow_rerun: bool) -> RunCreateResponse:
    limits = ctx.settings.limits
    with ctx.db.write() as s:
        case = s.get(Case, case_id)
        if case is None or case.status != "ready":
            raise ApiError(404, "CASE_NOT_FOUND", "Case not found.")
        runs = s.scalars(select(InferenceRun).where(InferenceRun.case_id == case_id)).all()
        active = [r for r in runs if r.status in ACTIVE_STATUSES]
        if active:
            raise ApiError(
                409,
                "RUN_ALREADY_ACTIVE",
                "A run of this case is already queued or running.",
                details={"run_id": active[0].id},
            )
        worker = s.get(WorkerStatus, 1)
        model_sha = worker.checkpoint_sha256 if worker else None
        done = [
            r
            for r in runs
            if r.status == "succeeded" and (model_sha is None or r.checkpoint_sha256 == model_sha)
        ]
        if done and not allow_rerun:
            raise ApiError(
                409,
                "RESULT_EXISTS",
                "A result for this case and model already exists; set allow_rerun to run again.",
                details={"run_id": done[-1].id},
            )
        queued = s.scalar(
            select(func.count()).select_from(InferenceRun).where(InferenceRun.status == "queued")
        )
        if queued >= limits.max_queued_runs:
            raise ApiError(
                429,
                "QUEUE_FULL",
                "Too many runs are waiting; try again later.",
                retryable=True,
                headers={"Retry-After": "30"},
            )
        run = InferenceRun(
            id=new_uuid(),
            case_id=case_id,
            status="queued",
            progress=0.0,
            allow_rerun=allow_rerun,
            created_at=utcnow(),
        )
        s.add(run)
        s.flush()
        position = _queue_position(s, run)
        online = worker_state(worker, ctx.settings.worker.lease_stale_after_s) == "online"
        audit(s, "run_queued", case_id=case_id, run_id=run.id, rerun=allow_rerun)
    return RunCreateResponse(
        run_id=run.id,
        case_id=case_id,
        status="queued",
        queue_position=position,
        worker_online=online,
        links=_links(run),
    )


def status_response(s, run: InferenceRun) -> RunStatusResponse:
    error = None
    if run.error_code:
        error = RunError(
            code=run.error_code,
            message=run.error_message or "Run failed.",
            retryable=engine_retryable(run.error_code),
        )
    return RunStatusResponse(
        run_id=run.id,
        case_id=run.case_id,
        status=run.status,
        current_stage=run.current_stage,
        progress=max(0.0, min(1.0, run.progress or 0.0)),
        queue_position=_queue_position(s, run),
        created_at=iso(run.created_at),
        started_at=iso(run.started_at),
        completed_at=iso(run.completed_at),
        warning_count=run.warning_count or 0,
        warnings=[WarningItem(code=w["code"], message=w["message"]) for w in run.warnings or []],
        error=error,
        model=RunModel(
            model_name=run.model_name,
            checkpoint_sha256=run.checkpoint_sha256,
            config_hash=run.config_hash,
        ),
        links=_links(run),
    )


def get_run_status(ctx: AppContext, run_id: str) -> RunStatusResponse:
    mark_stale_runs(ctx)
    with ctx.db.read() as s:
        run = s.get(InferenceRun, run_id)
        if run is None:
            raise run_not_found()
        return status_response(s, run)


def cancel_run(ctx: AppContext, run_id: str) -> RunStatusResponse:
    with ctx.db.write() as s:
        run = s.get(InferenceRun, run_id)
        if run is None:
            raise run_not_found()
        if run.status == "queued":
            n = s.execute(
                update(InferenceRun)
                .where(InferenceRun.id == run_id, InferenceRun.status == "queued")
                .values(status="cancelled", completed_at=utcnow(), progress=0.0)
            ).rowcount
            if n == 1:
                audit(s, "run_cancelled", case_id=run.case_id, run_id=run_id)
            s.refresh(run)
        if run.status != "cancelled":
            raise ApiError(
                409,
                "RUN_NOT_CANCELLABLE",
                "Only queued runs can be cancelled.",
                details={"status": run.status},
            )
        return status_response(s, run)


def mark_stale_runs(ctx: AppContext, now: dt.datetime | None = None) -> int:
    """running + no heartbeat for run_stale_after_s -> E_WORKER_LOST;
    running longer than run_timeout_s -> E_TIMEOUT. Compare-and-set; returns rows changed."""
    w = ctx.settings.worker
    now = now or utcnow()
    lost_before = now - dt.timedelta(seconds=w.run_stale_after_s)
    timeout_before = now - dt.timedelta(seconds=w.run_timeout_s)
    changed = 0
    with ctx.db.write() as s:
        for code, message, cond in (
            (
                "E_WORKER_LOST",
                "The inference worker stopped responding.",
                InferenceRun.heartbeat_at < lost_before,
            ),
            (
                "E_TIMEOUT",
                "The run exceeded the time limit.",
                InferenceRun.started_at < timeout_before,
            ),
        ):
            ids = s.scalars(
                select(InferenceRun.id).where(InferenceRun.status == "running", cond)
            ).all()
            for rid in ids:
                n = s.execute(
                    update(InferenceRun)
                    .where(InferenceRun.id == rid, InferenceRun.status == "running")
                    .values(
                        status="failed",
                        error_code=code,
                        error_message=message,
                        completed_at=now,
                    )
                ).rowcount
                if n:
                    changed += n
                    audit(s, "run_failed", run_id=rid, code=code)
    return changed
