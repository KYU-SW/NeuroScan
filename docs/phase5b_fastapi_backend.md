# Phase 5B — FastAPI 백엔드 MVP

작성 2026-10-25 · 기준 커밋 `e639f69` (Phase 5A) · 상태: MVP 구현 완료 (미커밋) ·
**Test Split 미사용 (접근 기록 0명)**

> ⚠️ 연구 및 의료영상 검토 보조용 프로토타입이며 의료기기가 아닙니다. 인증이 없는 개발 버전이므로
> localhost에서만 실행합니다.

## 1. 확정 결정 (2026-10-24 승인)

| ID | 결정 |
|---|---|
| B1 | FastAPI + SQLite WAL + 독립 단일 GPU worker 프로세스(A+). API 프로세스는 torch·모델을 import/로딩하지 않음. worker는 정확히 1개, SQLite queue로 순차 처리. Redis/Celery/RQ/MySQL 미사용 |
| B2 | 백엔드 패키지 `backend/neuroscan_api/` |
| B3 | SQLAlchemy 2.x, WAL, foreign key, busy_timeout, `BEGIN IMMEDIATE` claim, `create_all` + `schema_version`. Alembic·MySQL 제외 |
| B4 | 저장소 `storage/{uploads,runs,temp,neuroscan.db}` (Git 무시). `results/` = 연구 평가, `outputs/` = CLI, `storage/` = API — 섞지 않음 |
| B5 | 원본 업로드는 자동 삭제하지 않음. `servable=false`, `shareable=false`, 다운로드 API 없음, 응답·manifest에 경로 비노출. case 삭제 시 원본과 결과 함께 삭제. 기간 삭제·purge_after_success는 후속 기능 |
| B6 | 진행 중 run → 409 `RUN_ALREADY_ACTIVE`; 같은 모델의 성공 결과 → 409 `RESULT_EXISTS`; `allow_rerun=true` → 덮어쓰지 않는 새 run |
| B7 | queued만 취소. running 취소 미구현 |
| B8 | 파일당 압축 64 MiB, 요청 140 MiB, gzip 해제 512 MiB, voxel ≤ 64M, 축 ≤ 1024, 큐 ≤ 20, timeout 표시 기준 300 s |
| B9 | 설치: fastapi, uvicorn, python-multipart, sqlalchemy, httpx (`requirements-api-lock.txt`) |
| B10 | Phase 5A 비식별 DWI·ADC를 Viewer용 file resource로 제공, 원본 업로드는 제공하지 않음 |
| B11 | Swagger/OpenAPI는 개발 모드(`debug: true`)에서만 |
| B12 | Content-Length 선검사 + UploadFile 1 MiB 단위 저장 (전체 파일을 메모리에 올리지 않음) |

**OOM 복구 정책**: 작업 종료마다 실제 parameter device, `LoadedModel.device`, worker 기대 device를
비교하고, 불일치 또는 CPU fallback 발생 시 기존 `LoadedModel` 폐기 → CUDA cache 정리 →
`load_inference_model()` 재호출 → checkpoint SHA·config hash·in_channels·context_slices 재검증 → 성공 시에만
다음 claim. 실패 시 worker `degraded`, claim 중단, 현재 run failed, API 유지. Phase 5A 코드 무수정.

## 2. 아키텍처

```
client ──127.0.0.1:8000──► API 프로세스 (FastAPI, torch import 없음)
                             GuardMiddleware: request id · Host allow-list · Content-Length 선검사
                             · no-store/nosniff · 경로 없는 access log
                             routes → services(cases, runs, results, files) → SQLite(WAL) + storage/
                                              ▲ claim/progress/heartbeat/결과 (BEGIN IMMEDIATE, CAS)
GPU worker 프로세스 (정확히 1개, worker_status lease) ── load_inference_model 1회
   └─ run_independent_inference(uploads/<case>/dwi|adc, output_root=storage/runs, loaded_model, progress)
```

- 엔진 run ID는 엔진이 만든다 → API `run_id`(UUID4)와 `engine_run_id`를 분리 저장.
- 진행 단계(엔진 callback은 단계 완료 후 호출): starting → validating(0.05) → preprocessing(0.15/0.25)
  → predicting(0.40, 복원 포함) → quantifying(0.70) → writing(0.80) → registering(0.95) → succeeded(1.0).
  엔진이 예측과 복원을 한 단계로 묶어 restoring은 별도 관측 불가.
- `storage/`: `uploads/<case_id>/{dwi,adc}.nii[.gz]`(원본, 비제공), `runs/<case_id>/<engine_run_id>/`(엔진 출력),
  `temp/`(업로드 중 `.part`), `neuroscan.db`. 같은 파일시스템에서 `os.replace`로 원자적 확정.

## 3. DB schema (SCHEMA_VERSION 1)

| 테이블 | 핵심 컬럼 | 제약 |
|---|---|---|
| `cases` | id `case-<12hex>`, created/updated, status, dwi/adc sha256, raw_retained | 하위 cascade |
| `inference_runs` | id UUID4, case_id FK, status, current_stage, progress, created/started/completed/heartbeat, worker_id, attempt, allow_rerun, engine_run_id(unique), run_dir_rel, model_name, checkpoint_sha256, config_hash, device_used, warning_count, warnings JSON(code+message), error_code/message, exit_code, lesion_detected, total_lesion_volume_ml | idx(status, created), idx(case, created), FK CASCADE |
| `file_resources` | id UUID4, case_id FK, run_id FK, role, storage_area, rel_path, mime, size, sha256, servable, shareable | unique(area, rel_path), unique(run_id, role) |
| `audit_events` | ts, event, case_id/run_id/file_id(FK 없음), detail JSON(경로·이름 없음) | idx(ts), idx(case_id) |
| `worker_status` | id=1 lease: worker_id, pid, state, heartbeat, 모델 정보, load_count, current_run_id | 단일 행 |
| `schema_version` | version | 불일치 시 시작 거부 |

SQLite: `isolation_level=None` + 명시적 `BEGIN`/`BEGIN IMMEDIATE`, WAL, foreign_keys=ON, busy_timeout 5 s.
영상 binary·입력 파일명·개인정보는 저장하지 않음.

## 4. API 계약 요약

`backend/README.md` 표 참고. 오류 코드: 400 `MISSING_FILE`/`UNEXPECTED_FIELD`/`UNSUPPORTED_EXTENSION`/
`INVALID_MULTIPART`/`CONFIRM_REQUIRED`/`HOST_NOT_ALLOWED`, 403 `SEALED_TEST_CASE`, 404 `*_NOT_FOUND`/
`OUTPUT_NOT_FOUND`, 409 `RUN_ALREADY_ACTIVE`/`RESULT_EXISTS`/`RUN_NOT_FINISHED`/`RUN_FAILED`/`RUN_CANCELLED`/
`RUN_NOT_CANCELLABLE`/`RUN_ACTIVE`, 411 `LENGTH_REQUIRED`, 413 `FILE_TOO_LARGE`/`REQUEST_TOO_LARGE`,
415 `UNSUPPORTED_MEDIA_TYPE`, 422 `INVALID_NIFTI`/`ARCHIVE_TOO_LARGE`/`VALIDATION_ERROR`, 429 `QUEUE_FULL`
(+Retry-After), 500 `RESULT_INTEGRITY_ERROR`/`FILE_INTEGRITY_ERROR`/`INTERNAL_ERROR`, 503
`MODEL_NOT_READY`/`DATABASE_UNAVAILABLE`, 507 `INSUFFICIENT_STORAGE`. 엔진 실패는 run `failed` +
엔진 코드(`retryable` 표시, `errors.ENGINE_CODES`). 추가 run 코드: `E_WORKER_LOST`, `E_WORKER_RESTARTED`,
`E_TIMEOUT`, `E_MODEL_RELOAD_FAILED`, `E_REGISTRATION`.

결과(`RunResultResponse`)는 quantification·manifest를 **필드별 명시 매핑**: lesion(검출·부피·component·
slice·threshold·미검출 문구), location(bbox·중심·반구 estimated), components(최대 100, truncated 표시),
research_score_summary(`calibrated: false`), vascular_territory `null`, flags, warnings, viewer(volume별
file_id URL, shape·spacing·affine·orientation, views, render_3d), 고지문. 응답 전체를 금지어 검사
(절대 경로, 파일명, header 문자열, traceback, GT·Dice, 질병 확률·진단 확정 표현)한 뒤 반환.

## 5. 검증 결과

- pytest **271 passed, 1 skipped** (기존 209 + API 62; skip = Windows에서 관리자 권한 없이 symlink 생성 불가 —
  symlink 거부는 단위 테스트, storage 밖을 가리키는 directory junction은 실제 생성해 404 확인). ruff check/format 통과.
- API 프로세스: `neuroscan_api.main` import·앱 생성 후 `torch`, `monai`, `neuroscan_ai.inference` 미로딩(subprocess 검사).
- worker: 4개 작업 직렬 처리 시 최대 동시 실행 1, 모델 로딩 1회, 실행 중 두 번째 worker lease 거부,
  stale lease 인수, 재시작 시 running → `E_WORKER_RESTARTED`와 `.partial` 정리(로그만 보존),
  heartbeat 유실 → `E_WORKER_LOST`, 300 s 초과 → `E_TIMEOUT`, API가 실패 처리한 run에 늦게 도착한 결과는 버림.
- OOM 정책: CPU fallback(경고 포함/미포함) 후 checkpoint에서 재로딩·재검증(로딩 2회), 재로딩 실패 시
  `degraded`·claim 중단·현재 run `E_MODEL_RELOAD_FAILED`·파일 미등록·`/models/current` 503.
- 업로드: 1 MiB 단위 읽기 확인, 크기·Content-Length·gzip 폭탄·손상 gzip·header·축 크기·저장 공간·DB 실패
  rollback, Test 파일명 403(내용 읽기 0회), traversal 파일명 무해화, 실패 시 temp·case 폴더 정리.
- 실제 Phase 5A 엔진(작은 checkpoint): 결과 매핑·금지어 0, 비식별 DWI/ADC·mask 다운로드(header 문자열 제거,
  uint8 0/1, 원본 shape·affine), 원본 업로드·로그·storage 밖 경로·조작된 경로 404, 변조 파일 500, API mask = CLI mask.
- **Validation 0086 API smoke** (실제 Phase 4 checkpoint, worker device auto/CUDA): API mask vs Phase 4 예측
  **불일치 0 voxel**, vs CLI service **0 voxel**, Dice 0.94677 = Phase 4 0.94677, 48.976 mL, Test 접근 0.
- 실제 두 프로세스(`python -m neuroscan_api` + `python -m neuroscan_api.worker`): health online, 두 번째 worker
  exit 3, 업로드→run→succeeded→result, Host 위조 400, 운영 모드 docs 404, 로그에 경로·파일명 없음.

## 6. 한계와 후속

- 실행 중 run 취소 미지원(B7). timeout은 표시 기준이며 멈춘 동기 호출을 강제 종료하지 않음(worker 재시작 필요).
- worker를 강제 종료하면 lease가 `lease_stale_after_s`(30 s) 지난 뒤 인수 가능.
- 원본 업로드 자동 삭제·기간 보존 정책(purge_after_success, retention)은 후속 기능(B5).
- Starlette가 `httpx` 기반 TestClient에 deprecation 경고를 출력(`httpx2` 권장) — 승인 목록 밖이라 설치하지 않음.
- Python 3.11에는 `os.path.isjunction`이 없어 junction 자체는 경로 해석 후 storage 경계 검사로 차단(내부를
  가리키는 junction은 storage 안이므로 허용).
- 패키지 등록: `pyproject.toml`에 `backend` 경로를 추가했으나 editable 재설치는 하지 않음 — 실행 시
  `PYTHONPATH=backend` 또는 `pip install -e . --no-deps` 필요.
- 운영 전 필수: 인증·권한, HTTPS, 저장 암호화, 접근 감사, 보존 정책, MySQL·Alembic, 규제·IRB 검토.
