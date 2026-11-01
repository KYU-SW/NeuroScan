"""Phase 5B API test helpers: settings on a temporary storage root, uploads, a fake engine."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from neuroscan_api.main import create_app  # noqa: E402
from neuroscan_api.settings import load_settings  # noqa: E402
from neuroscan_api.worker.loop import EngineAPI, Worker  # noqa: E402
from tests.inference_helpers import write_case  # noqa: E402

API = "/api/v1"
PHI_NAMES = ("Hong_Gildong_DWI.nii.gz", "Hong_Gildong_ADC.nii.gz")
PHI_TEXT = "PATIENT^NAME 19700101"


def make_settings(tmp: Path, **overrides):
    limits = overrides.pop("limits", {})
    worker = overrides.pop("worker", {})
    base_limits = {"min_free_bytes": 0}
    base_worker = {"device": "cpu", "heartbeat_interval_s": 0.2, "poll_interval_s": 0.05}
    base = {
        "storage_root": str(tmp / "storage"),
        "allowed_hosts": ["testserver", "127.0.0.1", "localhost"],
        "debug": True,
    }
    return load_settings(
        limits={**base_limits, **limits}, worker={**base_worker, **worker}, **{**base, **overrides}
    )


def client_for(settings) -> TestClient:
    return TestClient(create_app(settings))


def case_files(tmp: Path, names=PHI_NAMES, header_text=PHI_TEXT, **kw):
    d, a, _, _ = write_case(
        tmp / "upload_src", names=names, header_text={"descrip": header_text}, **kw
    )
    return d, a


def upload(client, dwi: Path, adc: Path, dwi_name=None, adc_name=None):
    files = {
        "dwi": (dwi_name or dwi.name, dwi.read_bytes(), "application/gzip"),
        "adc": (adc_name or adc.name, adc.read_bytes(), "application/gzip"),
    }
    return client.post(f"{API}/cases", files=files)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeParam:
    def __init__(self, device="cuda"):
        self.device = SimpleNamespace(type=device)


class FakeModel:
    def __init__(self, device="cuda"):
        self.param = FakeParam(device)

    def parameters(self):
        return iter([self.param])


def fake_loaded(device="cuda", sha="a" * 64, config_hash="34d18df104da"):
    return SimpleNamespace(
        model=FakeModel(device),
        device=SimpleNamespace(type=device),
        info={"run_id": "fake", "in_channels": 6, "val_mean_dice": 0.74},
        training_cfg={"data": {"context_slices": 1}, "inference": {"threshold": 0.5}},
        checkpoint_sha256=sha,
        config_hash=config_hash,
    )


class FakeEngine:
    """Phase 5A stand-in: writes the engine's output layout and returns an InferenceResult-like
    object. Records concurrency and load counts."""

    def __init__(self, delay=0.0, fail_load_after=None, on_run=None, status="succeeded"):
        self.delay, self.fail_load_after, self.on_run, self.status = (
            delay,
            fail_load_after,
            on_run,
            status,
        )
        self.loads = self.runs = self.active = self.max_active = 0
        self.lock = threading.Lock()

    def api(self) -> EngineAPI:
        return EngineAPI(self.resolve_device, self.load_model, self.run_inference)

    def resolve_device(self, requested):
        return SimpleNamespace(type="cuda"), []

    def load_model(self, checkpoint, config, device, expected):
        self.loads += 1
        if self.fail_load_after is not None and self.loads > self.fail_load_after:
            raise RuntimeError("simulated reload failure")
        return fake_loaded()

    def run_inference(self, dwi, adc, output_root, *, case_id, loaded_model, progress, **kw):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.runs += 1
            n = self.runs
        try:
            for stage in ("sealed_guard", "validation", "preprocess", "predict_restore"):
                progress(stage, 0.0)
            time.sleep(self.delay)
            warnings = []
            if self.on_run:
                warnings = self.on_run(loaded_model, n) or []
            if self.status != "succeeded":
                return SimpleNamespace(
                    status="failed",
                    exit_code=3,
                    run_id=f"20260101T000000Z-{n:08x}",
                    run_dir=None,
                    warnings=warnings,
                    metadata={},
                    quantification=None,
                    error={"code": "E_BRAIN_MASK", "message": "Only skull-stripped images."},
                )
            run_id = f"20260101T000000Z-{n:08x}"
            run_dir = Path(output_root) / case_id / run_id
            (run_dir / "inputs").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            contents = {
                "prediction_mask.nii.gz": b"mask",
                "prediction_components.nii.gz": b"components",
                "inputs/dwi.nii.gz": b"dwi",
                "inputs/adc.nii.gz": b"adc",
                "quantification.json": b"{}",
                "manifest.json": b"{}",
            }
            for rel, data in contents.items():
                (run_dir / rel).write_bytes(data)
            (run_dir / "logs" / "inference.log").write_text("log")
            files = {rel: hashlib.sha256(d).hexdigest() for rel, d in contents.items()}
            meta = {
                "outputs": {"files": files},
                "model": {
                    "model_name": "fake",
                    "checkpoint_sha256": "a" * 64,
                    "config_hash": "34d18df104da",
                },
                "device": {"used": "cuda"},
            }
            (run_dir / "inference_metadata.json").write_text(json.dumps(meta))
            progress("done", 1.0)
            return SimpleNamespace(
                status="succeeded",
                exit_code=0,
                run_id=run_id,
                run_dir=str(run_dir),
                warnings=warnings,
                metadata=meta,
                error=None,
                quantification={"lesion_detected": True, "total_lesion_volume_ml": 1.5},
            )
        finally:
            with self.lock:
                self.active -= 1


def make_worker(client, engine_api, start=True) -> Worker:
    ctx = client.app.state.ctx
    w = Worker(ctx.settings, ctx.db, ctx.storage, engine_api)
    if start:
        w.start()
    return w


def all_json_text(*responses) -> str:
    return "\n".join(json.dumps(r.json(), ensure_ascii=False) for r in responses)
