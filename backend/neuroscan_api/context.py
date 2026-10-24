"""Shared application context and small helpers (IDs, timestamps, audit events)."""

from __future__ import annotations

import datetime as dt
import re
import uuid
from dataclasses import dataclass

from neuroscan_api.db.models import AuditEvent, WorkerStatus, utcnow
from neuroscan_api.db.session import Database
from neuroscan_api.settings import Settings
from neuroscan_api.storage import Storage

CASE_ID = re.compile(r"^case-[0-9a-f]{12}$")
UUID_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


@dataclass
class AppContext:
    settings: Settings
    db: Database
    storage: Storage
    guard: object  # neuroscan_ai.data.split_guard.SplitGuard (split file read only)


def new_case_id() -> str:
    return f"case-{uuid.uuid4().hex[:12]}"


def new_uuid() -> str:
    return str(uuid.uuid4())


def iso(t: dt.datetime | None) -> str | None:
    return None if t is None else t.replace(microsecond=0).isoformat() + "Z"


def audit(session, event: str, *, case_id=None, run_id=None, file_id=None, **detail) -> None:
    session.add(
        AuditEvent(
            event=event, case_id=case_id, run_id=run_id, file_id=file_id, detail=detail or None
        )
    )


def worker_state(row: WorkerStatus | None, stale_after_s: float) -> str:
    """online | offline | stale | degraded | loading (as seen by the API)."""
    if row is None or row.state == "stopped":
        return "offline"
    if (utcnow() - row.heartbeat_at).total_seconds() > stale_after_s:
        return "stale"
    if row.state == "degraded":
        return "degraded"
    if row.state == "loading":
        return "loading"
    return "online"
