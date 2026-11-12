"""Epoch progress record ``training/progress.json`` and atomic file replacement.

The progress file is an OPERATIONAL monitoring record: rewritten after every epoch (and on an
interruption or at the end), never listed in a post-run manifest output hash and never part of a
reproducibility comparison. It holds case-free numbers only: no alias, subject ID, file name,
host path or user name (checked before every write).

Atomic write (``atomic_write_bytes``): ``<name>.tmp`` is written, flushed and fsync'ed, then
``os.replace`` swaps it in. A reader sees either the previous or the new complete file. A process
killed before ``os.replace`` leaves the previous file intact (plus a stale ``.tmp``).
"""

from __future__ import annotations

import datetime
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROGRESS_SCHEMA = "neuroscan-v2-baseline-progress-v1"
PROGRESS_NAME = "training/progress.json"
PROGRESS_POLICY: dict[str, Any] = {
    "atomic_write": True,
    "update_frequency": "epoch",
    "excluded_from_reproducibility_hash": True,
}
PROGRESS_FIELDS = (
    "schema",
    "run_id",
    "status",
    "completed_epoch",
    "max_epochs",
    "best_epoch",
    "early_stopping_counter",
    "latest_train_loss_total",
    "latest_train_loss_dice",
    "latest_train_loss_bce",
    "latest_validation_loss",
    "latest_validation_mean_positive_case_dice",
    "latest_false_positive_voxel_volume_ml",
    "latest_lesion_wise_recall",
    "learning_rate",
    "gradient_norm",
    "sampled_positive_ratio",
    "epoch_duration_seconds",
    "elapsed_seconds",
    "process_rss_bytes",
    "process_peak_rss_bytes",
    "system_available_memory_bytes",
    "cuda_memory_allocated_bytes",
    "cuda_memory_reserved_bytes",
    "cuda_max_memory_allocated_bytes",
    "cuda_max_memory_reserved_bytes",
    "updated_at_utc",
)
PROGRESS_STATUSES = (
    "running",
    "interrupted",
    "interrupted_resource_guard",
    "training_completed",
)
_ABS_PATH = re.compile(r"([A-Za-z]:[\\/])|(^|[\s'\"(])/[^\s]")


class ProgressRecordError(ValueError):
    pass


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def atomic_write_bytes(path: Path, data: bytes, tmp_name: str | None = None) -> None:
    """Write ``data`` to ``path`` via a fsync'ed temporary file and ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(tmp_name or path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def check_public_text(text: str, needles: Iterable[str] = ()) -> None:
    """Refuse text that carries an absolute path or any of ``needles`` (aliases / IDs)."""
    if _ABS_PATH.search(text):
        raise ProgressRecordError("record contains an absolute path; refused")
    if any(n and n in text for n in needles):
        raise ProgressRecordError("record contains a case alias or subject ID; refused")


def write_progress(path: Path, record: Mapping[str, Any], needles: Iterable[str] = ()) -> None:
    if tuple(record) != PROGRESS_FIELDS:
        raise ProgressRecordError("progress fields differ from PROGRESS_FIELDS")
    if record["schema"] != PROGRESS_SCHEMA or record["status"] not in PROGRESS_STATUSES:
        raise ProgressRecordError("unknown progress schema or status")
    text = json.dumps(record, indent=2, allow_nan=False)
    check_public_text(text, needles)
    atomic_write_bytes(Path(path), text.encode("utf-8"))


def read_progress(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != PROGRESS_SCHEMA:
        raise ProgressRecordError("unknown progress schema")
    return data
