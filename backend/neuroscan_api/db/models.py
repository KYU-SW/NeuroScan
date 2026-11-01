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

SCHEMA_VERSION = 2
# v1 -> v2 (Phase 7-B): adds research_cases and research_files only; no existing table or
# column changes (see Database.init_schema for the upgrade and the pre-upgrade backup).

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


class ResearchCase(Base):
    """Phase 7-B research validation case (validation split only). Metadata and storage-relative
    paths only - metrics, components and matching live in JSON files of the bundle. No source
    case ID: the alias -> case mapping is kept in a local file outside the database."""

    __tablename__ = "research_cases"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    alias: Mapped[str] = mapped_column(String(16), unique=True)
    dataset_id: Mapped[str] = mapped_column(String(32))
    split: Mapped[str] = mapped_column(String(8))
    source_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(16), nullable=True)
    threshold: Mapped[float] = mapped_column(Float)
    preprocessing_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    bundle_rel: Mapped[str] = mapped_column(String(255))
    metrics_path: Mapped[str] = mapped_column(String(255))
    component_matching_path: Mapped[str] = mapped_column(String(255))
    has_ground_truth: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    files: Mapped[list[ResearchFile]] = relationship(
        back_populates="research_case", cascade="all, delete-orphan", passive_deletes=True
    )


class ResearchFile(Base):
    """Servable research file (storage_area "research"). Kept apart from file_resources, whose
    case_id references general cases; /files/{id} serves these only when research mode is on."""

    __tablename__ = "research_files"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    research_case_id: Mapped[str] = mapped_column(
        ForeignKey("research_cases.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(32))
    storage_area: Mapped[str] = mapped_column(String(8), default="research")
    rel_path: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    research_case: Mapped[ResearchCase] = relationship(back_populates="files")
    __table_args__ = (UniqueConstraint("research_case_id", "role", name="uq_research_file_role"),)


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
