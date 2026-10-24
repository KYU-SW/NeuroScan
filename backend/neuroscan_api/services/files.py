"""file_id registry: the only way result files leave the server. Unknown ids, non-servable roles
(raw uploads, logs) and any path that is unsafe or outside storage all return the same 404."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from neuroscan_api.context import AppContext, audit
from neuroscan_api.db.models import FileResource, InferenceRun
from neuroscan_api.errors import ApiError
from neuroscan_api.storage import StorageError

DOWNLOAD_NAMES = {
    "input_dwi_deid": "dwi.nii.gz",
    "input_adc_deid": "adc.nii.gz",
    "prediction_mask": "prediction_mask.nii.gz",
    "quantification_json": "quantification.json",
    "metadata_json": "inference_metadata.json",
    "manifest_json": "manifest.json",
}


@dataclass
class ServedFile:
    path: Path
    media_type: str
    download_name: str
    sha256: str


def file_not_found() -> ApiError:
    return ApiError(404, "FILE_NOT_FOUND", "File not found.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def verified_path(ctx: AppContext, row: FileResource) -> Path:
    try:
        path = ctx.storage.resolve(row.storage_area, row.rel_path)
    except StorageError:
        raise file_not_found() from None
    if not path.is_file() or path.stat().st_size != row.size_bytes or _sha256(path) != row.sha256:
        raise ApiError(500, "FILE_INTEGRITY_ERROR", "Stored file failed integrity verification.")
    return path


def serve_file(ctx: AppContext, file_id: str) -> ServedFile:
    with ctx.db.read() as s:
        row = s.get(FileResource, file_id)
        if row is None or not row.servable or row.role not in DOWNLOAD_NAMES:
            raise file_not_found()
        if row.run_id is not None:
            run = s.get(InferenceRun, row.run_id)
            if run is None or run.status != "succeeded":
                raise file_not_found()
    path = verified_path(ctx, row)
    with ctx.db.write() as s:
        audit(
            s, "file_served", case_id=row.case_id, run_id=row.run_id, file_id=row.id, role=row.role
        )
    return ServedFile(path, row.mime_type, DOWNLOAD_NAMES[row.role], row.sha256)


def load_run_json(ctx: AppContext, row: FileResource | None) -> dict:
    if row is None:
        raise ApiError(404, "OUTPUT_NOT_FOUND", "Result output is missing.")
    try:
        path = verified_path(ctx, row)
    except ApiError as exc:
        if exc.status == 404:
            raise ApiError(404, "OUTPUT_NOT_FOUND", "Result output is missing.") from None
        raise ApiError(
            500, "RESULT_INTEGRITY_ERROR", "Result failed integrity verification."
        ) from None
    return json.loads(path.read_text(encoding="utf-8"))
