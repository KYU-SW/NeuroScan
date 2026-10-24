"""Single GPU worker: DB lease, queue claim, Phase 5A service call, result registration.

- At most one live worker: a single-row lease in ``worker_status`` (BEGIN IMMEDIATE); a fresh
  lease held by another worker refuses start-up, a stale one may be taken over.
- The model is loaded once at start-up; jobs run strictly one after another.
- After every job the model placement is checked (parameter device, LoadedModel.device, expected
  device, CPU-fallback warning). On any mismatch the LoadedModel is discarded and reloaded from
  the checkpoint with full re-verification; if that fails the worker turns ``degraded``, stops
  claiming and the current run fails with E_MODEL_RELOAD_FAILED. Phase 5A code is not modified.
- Every write is compare-and-set on (run.status = running, run.worker_id = me), so a run that
  the API already marked stale/timed out is never overwritten by a late worker result.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select, update

from neuroscan_api.context import audit, new_uuid
from neuroscan_api.db.models import FileResource, InferenceRun, WorkerStatus, utcnow
from neuroscan_api.db.session import Database
from neuroscan_api.settings import REPO_ROOT, Settings
from neuroscan_api.storage import Storage, StorageError

LOGGER = logging.getLogger("neuroscan_api.worker")

# engine progress callback (called after a stage finishes) -> (next API stage, progress)
STAGE_AFTER = {
    "sealed_guard": ("validating", 0.05),
    "validation": ("preprocessing", 0.15),
    "load_model": ("preprocessing", 0.25),
    "preprocess": ("predicting", 0.40),
    "predict_restore": ("quantifying", 0.70),
    "quantify": ("writing", 0.80),
    "write_outputs": ("registering", 0.95),
    "done": ("registering", 0.95),
}
# engine output (run-relative) -> (role, mime type, servable, shareable)
OUTPUT_ROLES = {
    "prediction_mask.nii.gz": ("prediction_mask", "application/gzip", True, True),
    "quantification.json": ("quantification_json", "application/json", True, True),
    "manifest.json": ("manifest_json", "application/json", True, True),
    "inference_metadata.json": ("metadata_json", "application/json", True, True),
    "inputs/dwi.nii.gz": ("input_dwi_deid", "application/gzip", True, False),
    "inputs/adc.nii.gz": ("input_adc_deid", "application/gzip", True, False),
}


class LeaseHeldError(RuntimeError):
    """Another live worker holds the GPU lease."""


class LeaseLostError(RuntimeError):
    """This worker no longer owns the lease."""


@dataclass
class EngineAPI:
    resolve_device: Callable[[str], tuple[Any, list]]
    load_model: Callable[..., Any]
    run_inference: Callable[..., Any]


def default_engine() -> EngineAPI:
    # the only place torch / the model enter the backend: the worker process
    from neuroscan_ai.inference import (
        load_inference_model,
        resolve_device,
        run_independent_inference,
    )

    return EngineAPI(resolve_device, load_inference_model, run_independent_inference)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _repo(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


class Worker:
    def __init__(
        self, settings: Settings, db: Database, storage: Storage, engine: EngineAPI, worker_id=None
    ) -> None:
        self.settings, self.db, self.storage, self.engine = settings, db, storage, engine
        self.worker_id = worker_id or new_uuid()
        self.inf_cfg = yaml.safe_load(settings.inference_config.read_text(encoding="utf-8"))
        self.loaded = None
        self.expected_device = None
        self.initial_checkpoint_sha: str | None = None
        self.load_count = 0
        self.degraded = False
        self.current_run_id: str | None = None
        self._stop = threading.Event()
        self._hb_thread: threading.Thread | None = None

    # --- lease ------------------------------------------------------------------------------

    def acquire_lease(self) -> None:
        now = utcnow()
        with self.db.write() as s:
            row = s.get(WorkerStatus, 1)
            if row is not None and row.worker_id != self.worker_id and row.state != "stopped":
                age = (now - row.heartbeat_at).total_seconds()
                if age < self.settings.worker.lease_stale_after_s:
                    raise LeaseHeldError("another GPU worker is active")
            if row is None:
                row = WorkerStatus(id=1)
                s.add(row)
            row.worker_id, row.pid, row.state = self.worker_id, os.getpid(), "loading"
            row.started_at = row.heartbeat_at = now
            row.current_run_id = None
            row.load_count = 0
            for f in ("model_name", "checkpoint_sha256", "config_hash", "device", "loaded_at"):
                setattr(row, f, None)
            audit(s, "worker_started", worker=self.worker_id)

    def _owned(self, s) -> WorkerStatus:
        row = s.get(WorkerStatus, 1)
        if row is None or row.worker_id != self.worker_id:
            raise LeaseLostError("worker lease lost")
        return row

    def set_state(self, state: str) -> None:
        with self.db.write() as s:
            row = self._owned(s)
            row.state, row.heartbeat_at = state, utcnow()
            if state in ("idle", "degraded", "stopped"):
                row.current_run_id = None

    def heartbeat(self) -> None:
        now = utcnow()
        with self.db.write() as s:
            row = self._owned(s)
            row.heartbeat_at = now
            if self.current_run_id:
                s.execute(
                    update(InferenceRun)
                    .where(
                        InferenceRun.id == self.current_run_id,
                        InferenceRun.status == "running",
                        InferenceRun.worker_id == self.worker_id,
                    )
                    .values(heartbeat_at=now)
                )

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.settings.worker.heartbeat_interval_s):
            try:
                self.heartbeat()
            except LeaseLostError:
                LOGGER.error("lease lost; stopping")
                self._stop.set()
            except Exception as exc:  # never let the heartbeat thread die silently
                LOGGER.warning("heartbeat failed: %s", type(exc).__name__)

    # --- start-up recovery ------------------------------------------------------------------

    def recover(self) -> int:
        """Previous worker's running runs -> failed; abandoned engine staging folders cleaned."""
        now = utcnow()
        with self.db.write() as s:
            ids = s.scalars(
                select(InferenceRun.id).where(
                    InferenceRun.status == "running",
                    (InferenceRun.worker_id != self.worker_id) | InferenceRun.worker_id.is_(None),
                )
            ).all()
            for rid in ids:
                s.execute(
                    update(InferenceRun)
                    .where(InferenceRun.id == rid, InferenceRun.status == "running")
                    .values(
                        status="failed",
                        error_code="E_WORKER_RESTARTED",
                        error_message="The inference worker restarted during this run.",
                        completed_at=now,
                    )
                )
                audit(s, "run_failed", run_id=rid, code="E_WORKER_RESTARTED")
        self._clean_partials()
        return len(ids)

    def _clean_partials(self) -> None:
        """Engine staging folders left by a crash: keep logs only (same rule as D5)."""
        for case_dir in self.storage.runs.iterdir():
            if case_dir.is_symlink() or not case_dir.is_dir():
                continue
            for d in case_dir.glob(".*.partial"):
                if d.is_symlink() or not d.is_dir():
                    continue
                for f in sorted(d.rglob("*"), reverse=True):
                    if f.is_file() and not f.is_symlink() and f.parent.name != "logs":
                        f.unlink()
                    elif f.is_dir() and f.name != "logs" and not any(f.iterdir()):
                        f.rmdir()
                target = d.with_name(d.name.replace(".partial", ".abandoned"))
                if not target.exists():
                    d.rename(target)

    # --- model ------------------------------------------------------------------------------

    def load_model(self) -> bool:
        m = self.inf_cfg["model"]
        try:
            if self.expected_device is None:
                self.expected_device, _ = self.engine.resolve_device(self.settings.worker.device)
            loaded = self.engine.load_model(
                _repo(m["checkpoint"]), _repo(m["training_config"]), self.expected_device, m
            )
            self._verify(loaded)
        except Exception as exc:
            LOGGER.error("model load failed: %s", type(exc).__name__)
            self.loaded = None
            return False
        self.loaded = loaded
        self.load_count += 1
        if self.initial_checkpoint_sha is None:
            self.initial_checkpoint_sha = loaded.checkpoint_sha256
        cfg = loaded.training_cfg
        with self.db.write() as s:
            row = self._owned(s)
            row.model_name, row.model_type = m["model_name"], m["model_type"]
            row.training_run_id = loaded.info.get("run_id")
            row.checkpoint_sha256, row.config_hash = loaded.checkpoint_sha256, loaded.config_hash
            row.device = loaded.device.type
            row.in_channels = int(loaded.info["in_channels"])
            row.context_slices = int(cfg["data"]["context_slices"])
            row.threshold = float(cfg["inference"]["threshold"])
            row.val_dice = loaded.info.get("val_mean_dice")
            row.loaded_at, row.load_count, row.state = utcnow(), self.load_count, "idle"
            audit(s, "model_loaded", worker=self.worker_id, count=self.load_count)
        return True

    def _verify(self, loaded) -> None:
        m = self.inf_cfg["model"]
        cfg = loaded.training_cfg
        checks = [
            loaded.config_hash == m["expected_config_hash"],
            int(loaded.info["in_channels"]) == m["expected_in_channels"],
            int(cfg["data"]["context_slices"]) == m["expected_context_slices"],
            self.initial_checkpoint_sha is None
            or loaded.checkpoint_sha256 == self.initial_checkpoint_sha,
            next(loaded.model.parameters()).device.type == self.expected_device.type,
        ]
        if not all(checks):
            raise RuntimeError("loaded model failed verification")

    def model_needs_reload(self, result) -> bool:
        if self.loaded is None:
            return True
        try:
            param_device = next(self.loaded.model.parameters()).device.type
        except Exception:
            return True
        expected = self.expected_device.type
        fallback = any(
            w.get("code") == "W_CPU_FALLBACK" for w in (getattr(result, "warnings", None) or [])
        )
        return (
            param_device != expected
            or self.loaded.device.type != expected
            or (fallback and expected != "cpu")
        )

    def reload_model(self) -> bool:
        LOGGER.warning(
            "model placement changed (e.g. CUDA OOM fallback); reloading from checkpoint"
        )
        self.loaded = None
        gc.collect()
        try:
            import torch  # worker process only

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return self.load_model()

    # --- queue ------------------------------------------------------------------------------

    def claim(self) -> tuple[str, str] | None:
        now = utcnow()
        with self.db.write() as s:
            row = self._owned(s)
            if row.state != "idle" or self.degraded:
                return None
            run = s.scalars(
                select(InferenceRun)
                .where(InferenceRun.status == "queued")
                .order_by(InferenceRun.created_at, InferenceRun.id)
                .limit(1)
            ).first()
            if run is None:
                return None
            n = s.execute(
                update(InferenceRun)
                .where(InferenceRun.id == run.id, InferenceRun.status == "queued")
                .values(
                    status="running",
                    worker_id=self.worker_id,
                    started_at=now,
                    heartbeat_at=now,
                    current_stage="starting",
                    progress=0.0,
                    attempt=InferenceRun.attempt + 1,
                )
            ).rowcount
            if n != 1:
                return None
            row.state, row.current_run_id, row.heartbeat_at = "busy", run.id, now
            audit(s, "run_started", case_id=run.case_id, run_id=run.id)
            return run.id, run.case_id

    def _cas(self, s, run_id: str, **values) -> int:
        return s.execute(
            update(InferenceRun)
            .where(
                InferenceRun.id == run_id,
                InferenceRun.status == "running",
                InferenceRun.worker_id == self.worker_id,
            )
            .values(**values)
        ).rowcount

    def _progress_cb(self, run_id: str):
        def cb(stage: str, _fraction: float) -> None:
            mapped = STAGE_AFTER.get(stage)
            if mapped is None:
                return
            try:
                with self.db.write() as s:
                    self._cas(
                        s,
                        run_id,
                        current_stage=mapped[0],
                        progress=mapped[1],
                        heartbeat_at=utcnow(),
                    )
            except Exception as exc:  # progress must never break inference
                LOGGER.warning("progress update failed: %s", type(exc).__name__)

        return cb

    def _fail(self, run_id: str, code: str, message: str, **extra) -> None:
        with self.db.write() as s:
            n = self._cas(
                s,
                run_id,
                status="failed",
                error_code=code,
                error_message=message[:300],
                completed_at=utcnow(),
                **extra,
            )
            if n:
                audit(s, "run_failed", run_id=run_id, code=code)

    def _inputs(self, case_id: str) -> dict[str, Path]:
        with self.db.read() as s:
            rows = s.scalars(
                select(FileResource).where(
                    FileResource.case_id == case_id, FileResource.role.like("upload_%_raw")
                )
            ).all()
        paths = {}
        for r in rows:
            paths[r.role.split("_")[1]] = self.storage.resolve(r.storage_area, r.rel_path)
        if set(paths) != {"dwi", "adc"}:
            raise StorageError("inputs missing")
        return paths

    def process(self, run_id: str, case_id: str) -> None:
        self.current_run_id = run_id
        result = None
        try:
            try:
                inputs = self._inputs(case_id)
            except StorageError:
                self._fail(run_id, "E_INPUT_NOT_FOUND", "Uploaded inputs are missing.")
                return
            try:
                result = self.engine.run_inference(
                    inputs["dwi"],
                    inputs["adc"],
                    self.storage.runs,
                    case_id=case_id,
                    loaded_model=self.loaded,
                    device=self.settings.worker.device,
                    inference_config_path=self.settings.inference_config,
                    progress=self._progress_cb(run_id),
                )
            except KeyboardInterrupt:
                self._fail(run_id, "E_INTERRUPTED", "Inference was interrupted.")
                raise
            except Exception as exc:
                LOGGER.error("engine raised %s", type(exc).__name__)
                self._fail(run_id, "E_INTERNAL", "An internal error occurred.")
                result = None
            reload_failed = False
            if self.model_needs_reload(result):
                reload_failed = not self.reload_model()
            if reload_failed:
                self.degraded = True
                self._fail(
                    run_id,
                    "E_MODEL_RELOAD_FAILED",
                    "The model could not be reloaded after a device change.",
                )
            elif result is not None:
                self._finalize(run_id, result)
        finally:
            self.current_run_id = None
            try:
                self.set_state("degraded" if self.degraded else "idle")
            except LeaseLostError:
                self._stop.set()

    def _finalize(self, run_id: str, result) -> None:
        meta = result.metadata or {}
        summary = {
            "engine_run_id": result.run_id,
            "warnings": [{"code": w["code"], "message": w["message"]} for w in result.warnings],
            "warning_count": len(result.warnings),
            "exit_code": result.exit_code,
            "model_name": (meta.get("model") or {}).get("model_name"),
            "checkpoint_sha256": (meta.get("model") or {}).get("checkpoint_sha256"),
            "config_hash": (meta.get("model") or {}).get("config_hash"),
            "device_used": (meta.get("device") or {}).get("used"),
        }
        run_rel = None
        if result.run_dir:
            try:
                run_rel = self.storage.relative("runs", Path(result.run_dir))
            except StorageError:
                run_rel = None
        if result.status != "succeeded":
            err = result.error or {}
            self._fail(
                run_id,
                err.get("code", "E_INTERNAL"),
                err.get("message", "Run failed."),
                run_dir_rel=run_rel,
                **summary,
            )
            return
        try:
            if run_rel is None:
                raise StorageError("run directory outside storage")
            files = []
            for rel_in_run, sha in (meta.get("outputs") or {}).get("files", {}).items():
                if rel_in_run not in OUTPUT_ROLES:
                    raise StorageError("unexpected engine output")
                path = self.storage.resolve("runs", f"{run_rel}/{rel_in_run}")
                if _sha256(path) != sha:
                    raise StorageError("output hash mismatch")
                files.append((rel_in_run, path.stat().st_size, sha))
            meta_path = self.storage.resolve("runs", f"{run_rel}/inference_metadata.json")
            files.append(("inference_metadata.json", meta_path.stat().st_size, _sha256(meta_path)))
            if {f[0] for f in files} != set(OUTPUT_ROLES):
                raise StorageError("engine outputs incomplete")
        except StorageError as exc:
            LOGGER.error("output registration refused: %s", exc)
            self._fail(
                run_id, "E_OUTPUT_VERIFY", "Result files failed verification.", run_dir_rel=run_rel
            )
            return
        quant = result.quantification or {}
        try:
            with self.db.write() as s:
                n = self._cas(
                    s,
                    run_id,
                    status="succeeded",
                    current_stage="done",
                    progress=1.0,
                    completed_at=utcnow(),
                    run_dir_rel=run_rel,
                    lesion_detected=bool(quant.get("lesion_detected")),
                    total_lesion_volume_ml=float(quant.get("total_lesion_volume_ml", 0.0)),
                    **summary,
                )
                if n != 1:  # API already failed it (stale / timeout): keep that decision
                    audit(s, "result_discarded", run_id=run_id)
                    return
                case_id = s.get(InferenceRun, run_id).case_id
                for rel_in_run, size, sha in files:
                    role, mime, servable, shareable = OUTPUT_ROLES[rel_in_run]
                    s.add(
                        FileResource(
                            id=new_uuid(),
                            case_id=case_id,
                            run_id=run_id,
                            role=role,
                            storage_area="runs",
                            rel_path=f"{run_rel}/{rel_in_run}",
                            mime_type=mime,
                            size_bytes=size,
                            sha256=sha,
                            servable=servable,
                            shareable=shareable,
                        )
                    )
                audit(s, "run_succeeded", case_id=case_id, run_id=run_id)
        except Exception as exc:  # transaction rolled back as a whole
            LOGGER.error("registration failed: %s", type(exc).__name__)
            self._fail(run_id, "E_REGISTRATION", "Result registration failed.")

    # --- loop -------------------------------------------------------------------------------

    def start(self) -> None:
        self.acquire_lease()
        self.recover()
        if not self.load_model():
            self.degraded = True
            self.set_state("degraded")
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()

    def run_once(self) -> bool:
        if self.degraded or self._stop.is_set():
            return False
        job = self.claim()
        if job is None:
            return False
        self.process(*job)
        return True

    def run_forever(self) -> None:
        while not self._stop.is_set():
            if not self.run_once():
                time.sleep(self.settings.worker.poll_interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._hb_thread:
            self._hb_thread.join(timeout=5)
        try:
            self.set_state("stopped")
        except LeaseLostError:
            pass
