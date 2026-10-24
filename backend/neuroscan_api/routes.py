"""HTTP routes (thin): parse/validate identifiers, call services, return explicit models.

Malformed IDs answer 404 exactly like unknown IDs. Blocking work (disk, SQLite, hashing) runs in
the thread pool via plain ``def`` handlers; only the upload handler is async (streaming).
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import FileResponse
from sqlalchemy import func, select

from neuroscan_api import API_PREFIX, __version__
from neuroscan_api.context import CASE_ID, UUID_ID, AppContext, iso, worker_state
from neuroscan_api.db.models import InferenceRun, WorkerStatus
from neuroscan_api.errors import ApiError
from neuroscan_api.schemas import (
    CaseDeleteResponse,
    CaseResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    RunCreateRequest,
    RunCreateResponse,
    RunResultResponse,
    RunStatusResponse,
)
from neuroscan_api.services import cases, files, results, runs

router = APIRouter(prefix=API_PREFIX)
OPTIONAL_BODY = Body(None)  # POST /runs body is optional ({"allow_rerun": false} by default)
ERRORS = {
    code: {"model": ErrorResponse}
    for code in (400, 403, 404, 409, 411, 413, 415, 422, 429, 500, 503, 507)
}
UPLOAD_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["dwi", "adc"],
                    "properties": {
                        "dwi": {"type": "string", "format": "binary"},
                        "adc": {"type": "string", "format": "binary"},
                    },
                }
            }
        },
    }
}


def ctx_of(request: Request) -> AppContext:
    return request.app.state.ctx


def _case_id(case_id: str) -> str:
    if not CASE_ID.fullmatch(case_id):
        raise ApiError(404, "CASE_NOT_FOUND", "Case not found.")
    return case_id


def _uuid(value: str, code: str, what: str) -> str:
    if not UUID_ID.fullmatch(value):
        raise ApiError(404, code, f"{what} not found.")
    return value


@router.get("/health", response_model=HealthResponse, responses=ERRORS)
def health(request: Request) -> HealthResponse:
    ctx = ctx_of(request)
    try:
        runs.mark_stale_runs(ctx)
        with ctx.db.read() as s:
            depth = s.scalar(
                select(func.count())
                .select_from(InferenceRun)
                .where(InferenceRun.status == "queued")
            )
            state = worker_state(s.get(WorkerStatus, 1), ctx.settings.worker.lease_stale_after_s)
    except Exception:
        raise ApiError(
            503, "DATABASE_UNAVAILABLE", "Database is unavailable.", retryable=True
        ) from None
    return HealthResponse(
        status="ok" if state == "online" else "degraded",
        database="ok",
        worker=state,
        queue_depth=int(depth),
        api_version=__version__,
    )


@router.get("/models/current", response_model=ModelInfoResponse, responses=ERRORS)
def current_model(request: Request) -> ModelInfoResponse:
    ctx = ctx_of(request)
    with ctx.db.read() as s:
        row = s.get(WorkerStatus, 1)
    state = worker_state(row, ctx.settings.worker.lease_stale_after_s)
    if state != "online" or row is None or row.checkpoint_sha256 is None:
        raise ApiError(
            503,
            "MODEL_NOT_READY",
            "The inference worker is not ready.",
            details={"worker": state},
            retryable=True,
        )
    return ModelInfoResponse(
        model_name=row.model_name,
        model_type=row.model_type,
        training_run_id=row.training_run_id,
        checkpoint_sha256=row.checkpoint_sha256,
        config_hash=row.config_hash,
        input_channels=row.in_channels,
        context_slices=row.context_slices,
        threshold=row.threshold,
        device=row.device,
        validation_positive_mean_dice=row.val_dice,
        loaded_at=iso(row.loaded_at),
        worker_state=row.state,
    )


@router.post(
    "/cases",
    status_code=201,
    response_model=CaseResponse,
    responses=ERRORS,
    openapi_extra=UPLOAD_OPENAPI,
)
async def create_case(request: Request) -> CaseResponse:
    ctx = ctx_of(request)
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Use multipart/form-data.")
    try:
        form = await request.form(max_files=2, max_fields=2)
    except Exception as exc:
        if "too many" in str(exc).lower():  # starlette's max_files / max_fields guard
            raise ApiError(
                400, "UNEXPECTED_FIELD", "Only the fields 'dwi' and 'adc' are accepted."
            ) from None
        raise ApiError(400, "INVALID_MULTIPART", "Upload could not be parsed.") from None
    try:
        return await cases.create_case(ctx, form)
    finally:
        await form.close()


@router.get("/cases/{case_id}", response_model=CaseResponse, responses=ERRORS)
def get_case(request: Request, case_id: str) -> CaseResponse:
    return cases.get_case(ctx_of(request), _case_id(case_id))


@router.delete("/cases/{case_id}", response_model=CaseDeleteResponse, responses=ERRORS)
def delete_case(request: Request, case_id: str, confirm: bool = Query(False)) -> CaseDeleteResponse:
    return cases.delete_case(ctx_of(request), _case_id(case_id), confirm)


@router.post(
    "/cases/{case_id}/runs", status_code=202, response_model=RunCreateResponse, responses=ERRORS
)
def create_run(
    request: Request, case_id: str, body: RunCreateRequest | None = OPTIONAL_BODY
) -> RunCreateResponse:
    allow = body.allow_rerun if body else False
    return runs.create_run(ctx_of(request), _case_id(case_id), allow)


@router.get("/runs/{run_id}", response_model=RunStatusResponse, responses=ERRORS)
def get_run(request: Request, run_id: str) -> RunStatusResponse:
    return runs.get_run_status(ctx_of(request), _uuid(run_id, "RUN_NOT_FOUND", "Run"))


@router.get("/runs/{run_id}/result", response_model=RunResultResponse, responses=ERRORS)
def get_result(request: Request, run_id: str) -> RunResultResponse:
    return results.get_result(ctx_of(request), _uuid(run_id, "RUN_NOT_FOUND", "Run"))


@router.post("/runs/{run_id}/cancel", response_model=RunStatusResponse, responses=ERRORS)
def cancel_run(request: Request, run_id: str) -> RunStatusResponse:
    return runs.cancel_run(ctx_of(request), _uuid(run_id, "RUN_NOT_FOUND", "Run"))


@router.get(
    "/files/{file_id}",
    response_class=FileResponse,
    responses={
        200: {"content": {"application/gzip": {}, "application/json": {}}},
        **{k: v for k, v in ERRORS.items() if k in (404, 500)},
    },
)
def get_file(request: Request, file_id: str):
    served = files.serve_file(ctx_of(request), _uuid(file_id, "FILE_NOT_FOUND", "File"))
    return FileResponse(
        served.path,
        media_type=served.media_type,
        filename=served.download_name,
        headers={"ETag": f'"{served.sha256}"', "Cache-Control": "no-store"},
    )
