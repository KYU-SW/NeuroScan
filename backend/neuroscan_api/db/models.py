"""SQLAlchemy 2.x models (SQLite now, MySQL-compatible types). No image data is stored: only IDs,
states, hashes, storage-relative paths and small summaries. No input file names, no PHI."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

SCHEMA_VERSION = 1

RUN_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")
ACTIVE_STATUSES = ("queued", "running")


def utcnow() -> dt.datetime:
    """Naive UTC (SQLite has no time zones); serialised with a trailing 'Z'."""
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class SchemaVersion(Base):
    __tablename__ = "schema_version"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    applied_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)


class Case(Base):
    __tablename__ = "cases"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    status: Mapped[str] = mapped_column(String(16), default="ready")
    dwi_sha256: Mapped[str] = mapped_column(String(64), index=True)
    adc_sha256: Mapped[str] = mapped_column(String(64))
    raw_retained: Mapped[bool] = mapped_column(Boolean, default=True)
    runs: Mapped[list[InferenceRun]] = relationship(
        back_populates="case", cascade="all, delete-orphan", passive_deletes=True
    )
    files: Mapped[list[FileResource]] = relationship(
        back_populates="case", cascade="all, delete-orphan", passive_deletes=True
    )


class InferenceRun(Base):
    __tablename__ = "inference_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(12), default="queued")
    current_stage: Mapped[str | None] = mapped_column(String(16), nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    heartbeat_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    allow_rerun: Mapped[bool] = mapped_column(Boolean, default=False)
    engine_run_id: Mapped[str | None] = mapped_column(String(40), nullable=True, unique=True)
    run_dir_rel: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(16), nullable=True)
    device_used: Mapped[str | None] = mapped_column(String(8), nullable=True)
    warning_count: Mapped[int] = mapped_column(Integer, default=0)
    warnings: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(300), nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lesion_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    total_lesion_volume_ml: Mapped[float | None] = mapped_column(Float, nullable=True)
    case: Mapped[Case] = relationship(back_populates="runs")
    __table_args__ = (
        Index("ix_runs_status_created", "status", "created_at"),
        Index("ix_runs_case_created", "case_id", "created_at"),
    )


class FileResource(Base):
    __tablename__ = "file_resources"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("inference_runs.id", ondelete="CASCADE"), nullable=True
    )
    role: Mapped[str] = mapped_column(String(24))
    storage_area: Mapped[str] = mapped_column(String(8))  # "uploads" | "runs"
    rel_path: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    servable: Mapped[bool] = mapped_column(Boolean, default=False)
    shareable: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    case: Mapped[Case] = relationship(back_populates="files")
    __table_args__ = (
        UniqueConstraint("storage_area", "rel_path", name="uq_file_location"),
        UniqueConstraint("run_id", "role", name="uq_file_run_role"),
    )


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)
    event: Mapped[str] = mapped_column(String(32))
    case_id: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)  # no FK
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    file_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    client: Mapped[str] = mapped_column(String(16), default="local")


class WorkerStatus(Base):
    """Single-row lease (id = 1): at most one live GPU worker."""

    __tablename__ = "worker_status"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    worker_id: Mapped[str] = mapped_column(String(36))
    pid: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(12))  # loading|idle|busy|degraded|stopped
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    heartbeat_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    training_run_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(16), nullable=True)
    device: Mapped[str | None] = mapped_column(String(8), nullable=True)
    in_channels: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_slices: Mapped[int | None] = mapped_column(Integer, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_dice: Mapped[float | None] = mapped_column(Float, nullable=True)
    loaded_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    load_count: Mapped[int] = mapped_column(Integer, default=0)
    current_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
