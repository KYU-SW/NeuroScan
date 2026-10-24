"""API settings: configs/api_v1.yaml plus NEUROSCAN_* environment overrides (no torch import)."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "api_v1.yaml"


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_file_bytes: int = 64 * 2**20
    max_request_bytes: int = 140 * 2**20
    max_uncompressed_bytes: int = 512 * 2**20
    max_voxels: int = 64 * 2**20
    max_axis: int = 1024
    min_free_bytes: int = 2 * 2**30
    max_queued_runs: int = 20


class WorkerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: str = "auto"
    poll_interval_s: float = 0.5
    heartbeat_interval_s: float = 5.0
    lease_stale_after_s: float = 30.0
    run_stale_after_s: float = 60.0
    run_timeout_s: float = 300.0


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = "api_v1"
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    allowed_hosts: list[str] = ["127.0.0.1", "localhost"]
    cors_origins: list[str] = []
    storage_root: Path = Path("storage")
    inference_config: Path = Path("configs/inference_v1.yaml")
    limits: Limits = Limits()
    worker: WorkerSettings = WorkerSettings()

    def resolved(self) -> Settings:
        def absolute(p: Path) -> Path:
            return p if p.is_absolute() else REPO_ROOT / p

        return self.model_copy(
            update={
                "storage_root": absolute(self.storage_root),
                "inference_config": absolute(self.inference_config),
            }
        )

    @property
    def database_path(self) -> Path:
        return self.storage_root / "neuroscan.db"

    def split_file(self) -> Path:
        """Split definition named by the inference config (case IDs only, read for the guard)."""
        cfg = yaml.safe_load(self.inference_config.read_text(encoding="utf-8"))
        p = Path(cfg["split_file"])
        return p if p.is_absolute() else REPO_ROOT / p


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def load_settings(path: str | Path | None = None, **overrides) -> Settings:
    path = Path(path or os.environ.get("NEUROSCAN_API_CONFIG") or DEFAULT_CONFIG)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if os.environ.get("NEUROSCAN_STORAGE_ROOT"):
        data["storage_root"] = os.environ["NEUROSCAN_STORAGE_ROOT"]
    if os.environ.get("NEUROSCAN_DEBUG"):
        data["debug"] = os.environ["NEUROSCAN_DEBUG"].lower() in ("1", "true", "yes")
    data.update(overrides)
    return Settings(**data).resolved()
