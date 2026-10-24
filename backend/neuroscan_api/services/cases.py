"""Case creation (streamed upload -> atomic commit), lookup and deletion."""

from __future__ import annotations

import os

from fastapi import UploadFile
from sqlalchemy import select

from neuroscan_api import API_PREFIX
from neuroscan_api.context import AppContext, audit, iso, new_case_id, new_uuid
from neuroscan_api.db.models import Case, FileResource, InferenceRun, utcnow
from neuroscan_api.errors import ApiError
from neuroscan_api.schemas import CaseDeleteResponse, CaseInput, CaseResponse, RunSummary
from neuroscan_api.storage import insufficient_storage, is_no_space
from neuroscan_api.uploads import (
    TempUpload,
    check_sealed_filename,
    inspect_nifti,
    normalized_suffix,
    save_stream,
)

ROLES = ("dwi", "adc")
MIME = {".nii.gz": "application/gzip", ".nii": "application/octet-stream"}


def case_not_found() -> ApiError:
    return ApiError(404, "CASE_NOT_FOUND", "Case not found.")


async def create_case(ctx: AppContext, form) -> CaseResponse:
    keys = list(form.keys())
    unexpected = sorted(set(keys) - set(ROLES))
    if unexpected or len(keys) != len(set(keys)):
        raise ApiError(400, "UNEXPECTED_FIELD", "Only the fields 'dwi' and 'adc' are accepted.")
    uploads: dict[str, UploadFile] = {}
    for role in ROLES:
        values = form.getlist(role)
        if len(values) != 1 or not hasattr(values[0], "read"):
            raise ApiError(400, "MISSING_FILE", f"Exactly one {role.upper()} file is required.")
        uploads[role] = values[0]
    # name checks first: nothing of a refused file is read or stored
    for up in uploads.values():
        normalized_suffix(up.filename)
        check_sealed_filename(up.filename, ctx.guard)
    ctx.storage.ensure_free_space(ctx.settings.limits.min_free_bytes)

    temps: list[TempUpload] = []
    case_id = new_case_id()
    case_dir = ctx.storage.uploads / case_id
    moved: list = []
    try:
        for role in ROLES:
            t = await save_stream(uploads[role], role, ctx.storage, ctx.settings.limits)
            temps.append(t)
            inspect_nifti(t, ctx.settings.limits)
        try:
            case_dir.mkdir(exist_ok=False)
            for t in temps:
                dest = case_dir / f"{t.role}{t.suffix}"
                os.replace(t.path, dest)
                moved.append(dest)
        except OSError as exc:
            if is_no_space(exc):
                raise insufficient_storage() from None
            raise
        by_role = {t.role: t for t in temps}
        with ctx.db.write() as s:
            case = Case(
                id=case_id,
                dwi_sha256=by_role["dwi"].sha256,
                adc_sha256=by_role["adc"].sha256,
                raw_retained=True,
            )
            s.add(case)
            for t in temps:
                s.add(
                    FileResource(
                        id=new_uuid(),
                        case_id=case_id,
                        run_id=None,
                        role=f"upload_{t.role}_raw",
                        storage_area="uploads",
                        rel_path=f"{case_id}/{t.role}{t.suffix}",
                        mime_type=MIME[t.suffix],
                        size_bytes=t.size_bytes,
                        sha256=t.sha256,
                        servable=False,  # B5: raw uploads may carry header PHI
                        shareable=False,
                    )
                )
            audit(s, "case_created", case_id=case_id)
    except BaseException:
        for p in moved:
            p.unlink(missing_ok=True)
        if case_dir.exists() and not any(case_dir.iterdir()):
            case_dir.rmdir()
        raise
    finally:
        for t in temps:
            t.path.unlink(missing_ok=True)
    return get_case(ctx, case_id)


def get_case(ctx: AppContext, case_id: str) -> CaseResponse:
    with ctx.db.read() as s:
        case = s.get(Case, case_id)
        if case is None or case.status != "ready":
            raise case_not_found()
        files = s.scalars(
            select(FileResource).where(
                FileResource.case_id == case_id, FileResource.role.like("upload_%")
            )
        ).all()
        runs = s.scalars(
            select(InferenceRun)
            .where(InferenceRun.case_id == case_id)
            .order_by(InferenceRun.created_at)
        ).all()
        inputs = sorted(
            (
                CaseInput(
                    role=f.role.split("_")[1],
                    sha256=f.sha256,
                    size_bytes=f.size_bytes,
                    raw_retained=case.raw_retained,
                )
                for f in files
            ),
            key=lambda c: ROLES.index(c.role),
        )
        return CaseResponse(
            case_id=case.id,
            created_at=iso(case.created_at),
            status=case.status,
            inputs=inputs,
            runs=[
                RunSummary(
                    run_id=r.id,
                    status=r.status,
                    created_at=iso(r.created_at),
                    completed_at=iso(r.completed_at),
                )
                for r in runs
            ],
            links={
                "self": f"{API_PREFIX}/cases/{case.id}",
                "runs": f"{API_PREFIX}/cases/{case.id}/runs",
            },
        )


def delete_case(ctx: AppContext, case_id: str, confirm: bool) -> CaseDeleteResponse:
    if not confirm:
        raise ApiError(400, "CONFIRM_REQUIRED", "Add ?confirm=true to delete a case.")
    with ctx.db.write() as s:
        case = s.get(Case, case_id)
        if case is None or case.status != "ready":
            raise case_not_found()
        runs = s.scalars(select(InferenceRun).where(InferenceRun.case_id == case_id)).all()
        if any(r.status == "running" for r in runs):
            raise ApiError(409, "RUN_ACTIVE", "A run of this case is in progress.")
        now = utcnow()
        for r in runs:
            if r.status == "queued":  # the worker can no longer claim it
                r.status, r.completed_at = "cancelled", now
        case.status = "deleting"
        n_runs = len(runs)
    files = ctx.storage.remove_tree("uploads", case_id) + ctx.storage.remove_tree("runs", case_id)
    with ctx.db.write() as s:
        case = s.get(Case, case_id)
        if case is not None:
            s.delete(case)  # runs and file resources cascade
        audit(s, "case_deleted", case_id=case_id, runs=n_runs, files=files)
    return CaseDeleteResponse(
        case_id=case_id, deleted=True, runs_deleted=n_runs, files_deleted=files
    )
