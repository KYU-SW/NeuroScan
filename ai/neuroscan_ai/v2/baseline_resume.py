"""Resume checkpoint ``training/resume_checkpoint.pt`` (v2 2D baseline).

Written after EVERY completed epoch, atomically: the payload is serialised in memory, its SHA-256
is computed, an envelope ``{schema, payload_sha256, payload}`` is written to
``training/resume_checkpoint.tmp`` (flush + fsync) and ``os.replace``'d over the previous file.
The checkpoint therefore always describes one fully completed epoch; a process killed during an
epoch (or during the write) keeps the previous epoch's checkpoint. Loading uses
``torch.load(weights_only=True)`` and refuses a payload whose SHA-256 differs.

Payload: run ID, completed / next epoch, model / optimiser / scheduler state, best-checkpoint
selection state (selected snapshot, selection records, best epoch, early-stopping counter),
training history, sampler records and the last sampling-list hash, Python / NumPy / Torch CPU /
Torch CUDA RNG states, seed, config / split / alias-list / input-aggregate / cache-index /
package-lock hashes, git commit, environment, determinism, model ID, parameter count and the
preprocessing / brain-mask / threshold policy IDs. No alias, subject ID or absolute path.

Resume is never automatic. ``check_resume_identity`` lists every difference between the payload
and the current run; any difference refuses the resume (``ResumeRefused``).
"""

from __future__ import annotations

import hashlib
import io
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuroscan_ai.v2.baseline_progress import atomic_write_bytes

RESUME_SCHEMA = "neuroscan-v2-baseline-resume-v1"
RESUME_NAME = "training/resume_checkpoint.pt"
RESUME_TMP_NAME = "resume_checkpoint.tmp"
RESUME_POLICY: dict[str, Any] = {
    "enabled": True,
    "update_frequency": "epoch",
    "automatic_resume": False,
    "exact_environment_required": True,
}
# run IDs that may never be resumed nor reused (operational decisions, not results)
RETIRED_RUN_IDS: dict[str, str] = {
    "v2-2d-baseline-seed42-a": (
        "externally_terminated (host memory pressure); no resume checkpoint; never reused"
    ),
}
# payload fields compared with the current run before a resume (all must be equal)
IDENTITY_FIELDS = (
    "schema",
    "run_id",
    "seed",
    "epoch_limit",
    "max_epochs",
    "config_canonical_sha256",
    "split_canonical_sha256",
    "alias_list_sha256",
    "input_aggregate_sha256",
    "cache_index_sha256",
    "package_lock_sha256_lf",
    "git_commit",
    "environment",
    "determinism",
    "model_id",
    "parameter_count",
    "preprocessing_policy",
    "brain_mask_policy",
    "threshold_policy",
    "sampling_policy",
)


class ResumeRefused(RuntimeError):
    stage = "resume"

    def __init__(self, reasons: list[str] | tuple[str, ...]):
        self.reasons = tuple(reasons)
        super().__init__("resume refused: " + ", ".join(self.reasons))


# ---------------------------------------------------------------- RNG state
def capture_rng() -> dict[str, Any]:
    version, state, gauss = random.getstate()
    kind, keys, pos, has_gauss, cached = np.random.get_state(legacy=True)
    cuda = (
        [s.clone() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else []
    )
    return {
        "python": {"version": int(version), "state": [int(v) for v in state], "gauss": gauss},
        "numpy": {
            "kind": str(kind),
            "keys": [int(v) for v in keys],
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": cuda,
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    py = state["python"]
    random.setstate((py["version"], tuple(py["state"]), py["gauss"]))
    n = state["numpy"]
    np.random.set_state(
        (
            n["kind"],
            np.asarray(n["keys"], dtype=np.uint32),
            n["pos"],
            n["has_gauss"],
            n["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    cuda = list(state["torch_cuda"])
    if cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(cuda):
            raise ResumeRefused(["cuda_rng_state_device_count_differs"])
        torch.cuda.set_rng_state_all(cuda)


def rng_fingerprint(state: Mapping[str, Any]) -> str:
    """Stable SHA-256 of an RNG state (tests / audit)."""
    h = hashlib.sha256()
    py, n = state["python"], state["numpy"]
    h.update(repr((py["version"], py["state"], py["gauss"])).encode())
    h.update(repr((n["kind"], n["keys"], n["pos"], n["has_gauss"], n["cached_gaussian"])).encode())
    h.update(state["torch_cpu"].numpy().tobytes())
    for s in state["torch_cuda"]:
        h.update(s.numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- checkpoint file
def save_resume_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically write the checkpoint; returns the SHA-256 of the file written."""
    buf = io.BytesIO()
    torch.save(dict(payload), buf)
    body = buf.getvalue()
    env = io.BytesIO()
    torch.save(
        {
            "schema": RESUME_SCHEMA,
            "payload_sha256": hashlib.sha256(body).hexdigest(),
            "payload": body,
        },
        env,
    )
    data = env.getvalue()
    atomic_write_bytes(Path(path), data, RESUME_TMP_NAME)
    return hashlib.sha256(data).hexdigest()


def load_resume_checkpoint(path: Path) -> dict[str, Any]:
    """Load and integrity-check a resume checkpoint (``ResumeRefused`` on any problem)."""
    path = Path(path)
    if not path.is_file():
        raise ResumeRefused(["resume_checkpoint_missing"])
    try:
        env = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is a refusal
        raise ResumeRefused(["resume_checkpoint_unreadable"]) from exc
    if not isinstance(env, dict) or env.get("schema") != RESUME_SCHEMA:
        raise ResumeRefused(["resume_checkpoint_schema"])
    body = env.get("payload")
    if not isinstance(body, bytes) or hashlib.sha256(body).hexdigest() != env.get("payload_sha256"):
        raise ResumeRefused(["resume_checkpoint_hash_invalid"])
    payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    if payload.get("schema") != RESUME_SCHEMA:
        raise ResumeRefused(["resume_checkpoint_schema"])
    return payload


def check_resume_identity(payload: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Reason codes for every identity field that differs (empty list = identical)."""
    return [f"{k}_mismatch" for k in IDENTITY_FIELDS if payload.get(k) != current.get(k)]


# ---------------------------------------------------------------- signals
class InterruptController:
    """Safe-stop requests from SIGINT (KeyboardInterrupt), SIGTERM and, on Windows, SIGBREAK.

    The first signal only sets ``requested``; the runner finishes the current epoch (batches,
    validation, resume checkpoint, progress), then stops with status ``interrupted``. A second
    signal raises ``KeyboardInterrupt`` at once (no record; the last completed epoch's resume
    checkpoint stays valid). SIGKILL, Windows ``TerminateProcess`` and an operating-system or
    tool process reaper cannot be caught: nothing is written then, and only the resume
    checkpoint of the last completed epoch can be trusted. Handlers are installed only in the
    main thread and restored on exit."""

    SIGNALS = ("SIGINT", "SIGTERM", "SIGBREAK")

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.requested: str | None = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> InterruptController:
        import signal  # noqa: PLC0415
        import threading  # noqa: PLC0415

        if self.enabled and threading.current_thread() is threading.main_thread():
            for name in self.SIGNALS:
                sig = getattr(signal, name, None)
                if sig is None:
                    continue
                try:
                    self._previous[sig] = signal.signal(sig, self._handle)
                except (ValueError, OSError):
                    continue
        return self

    def _handle(self, signum, frame) -> None:  # noqa: ARG002
        import signal  # noqa: PLC0415

        name = signal.Signals(signum).name
        if self.requested is None:
            self.requested = name
            return
        raise KeyboardInterrupt(f"second {name}: immediate stop")

    def __exit__(self, *exc) -> bool:
        import signal  # noqa: PLC0415

        for sig, prev in self._previous.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError, TypeError):
                continue
        self._previous.clear()
        return False

    @property
    def reason(self) -> str | None:
        if self.requested is None:
            return None
        return "keyboard_interrupt" if self.requested == "SIGINT" else self.requested.lower()
