# NeuroScan-ai backend (Phase 5B, development MVP)

> ⚠️ Research prototype — **not a medical device**. Results support medical image review and do
> not establish a diagnosis; model scores are uncalibrated and are not probabilities of disease.
> This build has **no authentication** and must only run on `127.0.0.1` / `localhost`.

Two processes share `storage/` (git-ignored) and its SQLite database:

| Process | Command | Imports torch? |
|---|---|---|
| API (FastAPI + uvicorn) | `python -m neuroscan_api` | **no** |
| GPU worker (exactly one) | `python -m neuroscan_api.worker` | yes (loads the Phase 4 model once) |

## Setup

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-api-lock.txt
$env:PYTHONPATH = "backend"            # or: pip install -e . --no-deps (registers neuroscan_api)
.venv\Scripts\python.exe -m neuroscan_api            # terminal 1: http://127.0.0.1:8000
.venv\Scripts\python.exe -m neuroscan_api.worker     # terminal 2: refuses to start if another worker is alive
```

Settings: `configs/api_v1.yaml`; environment overrides `NEUROSCAN_API_CONFIG`,
`NEUROSCAN_STORAGE_ROOT`, `NEUROSCAN_DEBUG=1` (development mode: Swagger at `/api/v1/docs`).
The launcher refuses non-loopback hosts. Do not start uvicorn directly with `--host 0.0.0.0`.

## API (`/api/v1`)

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | database, worker (online/offline/stale/degraded/loading), queue depth |
| GET | `/models/current` | from the worker (checkpoint SHA-256, config hash, 6 channels, threshold); 503 until ready |
| POST | `/cases` | multipart `dwi`, `adc` (`.nii`/`.nii.gz`); 201 with `case_id` |
| GET | `/cases/{case_id}` | input hashes/sizes, runs (no file names, raw uploads not downloadable) |
| DELETE | `/cases/{case_id}?confirm=true` | deletes uploads, results and DB rows (409 while running) |
| POST | `/cases/{case_id}/runs` | body optional `{"allow_rerun": false}`; 202 queued |
| GET | `/runs/{run_id}` | status, stage, progress, warnings, error |
| GET | `/runs/{run_id}/result` | allow-listed quantification + viewer volumes (file URLs) |
| POST | `/runs/{run_id}/cancel` | queued runs only (409 otherwise) |
| GET | `/files/{file_id}` | de-identified DWI/ADC, prediction mask, result JSON only |

Errors: `{"error": {"code", "message", "request_id", "retryable", "details"}}` — no paths, no
tracebacks. Engine failures appear as `status: failed` with the Phase 5A code.

## Data handling

- Raw uploads stay under `storage/uploads/<case_id>/` with server names; they may carry NIfTI
  header text and are **never served**. The viewer gets the Phase 5A de-identified copies.
- Upload limits: 64 MiB per file, 140 MiB per request (Content-Length required), 512 MiB after
  gzip, ≤ 64M voxels, ≤ 1024 per axis, ≥ 2 GiB free disk; queue ≤ 20 runs.
- File names whose ISLES case ID belongs to the sealed test split are refused (403) before any
  byte is read. Client file names are never stored or logged.
- Production use would additionally require authentication/authorisation, HTTPS, encrypted
  storage, access auditing, retention policies and regulatory/IRB review.

Details: [docs/phase5b_fastapi_backend.md](../docs/phase5b_fastapi_backend.md).
