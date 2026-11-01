"""SQLite engine/session: WAL, foreign keys, busy_timeout, explicit BEGIN / BEGIN IMMEDIATE.

pysqlite's own transaction handling is disabled (isolation_level=None) so that SQLAlchemy emits
the BEGIN itself; the ``immediate`` database uses ``BEGIN IMMEDIATE`` for queue claims, lease and
state transitions, which serialises writers and makes compare-and-set updates safe across the
API and worker processes.
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from neuroscan_api.db.models import SCHEMA_VERSION, Base, SchemaVersion, utcnow

BUSY_TIMEOUT_MS = 5000
LOGGER = logging.getLogger("neuroscan_api")
# schema versions this code can upgrade from (additive changes only)
UPGRADABLE_FROM = (1,)


def _engine(path: Path, immediate: bool) -> Engine:
    engine = create_engine(
        f"sqlite:///{path.as_posix()}",
        connect_args={"check_same_thread": False, "timeout": BUSY_TIMEOUT_MS / 1000},
    )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):
        dbapi_conn.isolation_level = None
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    @event.listens_for(engine, "begin")
    def _on_begin(conn):
        conn.exec_driver_sql("BEGIN IMMEDIATE" if immediate else "BEGIN")

    return engine


@dataclass
class Database:
    path: Path
    engine: Engine
    immediate_engine: Engine
    _read: sessionmaker
    _write: sessionmaker

    @classmethod
    def open(cls, path: Path) -> Database:
        path.parent.mkdir(parents=True, exist_ok=True)
        e, ei = _engine(path, False), _engine(path, True)
        return cls(
            path,
            e,
            ei,
            sessionmaker(e, expire_on_commit=False),
            sessionmaker(ei, expire_on_commit=False),
        )

    def stored_version(self) -> int | None:
        if not self.path.exists():
            return None
        con = sqlite3.connect(self.path)
        try:
            if not con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
            ).fetchone():
                return None
            row = con.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
        finally:
            con.close()
        return None if row is None else int(row[0])

    def backup(self, suffix: str) -> Path:
        """Consistent copy (SQLite online backup, WAL included) next to the database."""
        target = self.path.with_name(f"{self.path.name}.{suffix}.bak")
        src, dst = sqlite3.connect(self.path), sqlite3.connect(target)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        return target

    def init_schema(self) -> None:
        """Create a new database, or upgrade an older one explicitly.

        Upgrade policy (no Alembic yet): only additive versions listed in UPGRADABLE_FROM are
        upgraded. Before anything is written, a full backup ``<db>.v<old>-<UTC>.bak`` is made;
        the upgrade then only creates the new tables (``create_all`` never alters or drops
        existing tables or rows) and records the new version. Unknown or newer versions are
        refused without touching the file.
        """
        version = self.stored_version()
        if version is not None and version != SCHEMA_VERSION:
            if version not in UPGRADABLE_FROM:
                raise RuntimeError(
                    f"database schema version {version} != expected {SCHEMA_VERSION}"
                )
            stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
            target = self.backup(f"v{version}-{stamp}")
            LOGGER.warning(
                "database schema upgrade v%s -> v%s (backup %s)",
                version,
                SCHEMA_VERSION,
                target.name,
            )
        Base.metadata.create_all(self.immediate_engine)
        with self.write() as s:
            row = s.get(SchemaVersion, 1)
            if row is None:
                s.add(SchemaVersion(id=1, version=SCHEMA_VERSION))
            elif row.version in UPGRADABLE_FROM:
                row.version = SCHEMA_VERSION
                row.applied_at = utcnow()
            elif row.version != SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema version {row.version} != expected {SCHEMA_VERSION}"
                )

    @contextmanager
    def read(self) -> Iterator[Session]:
        with self._read() as s:
            yield s

    @contextmanager
    def write(self) -> Iterator[Session]:
        """BEGIN IMMEDIATE transaction; commits on success, rolls back on any exception."""
        with self._write() as s, s.begin():
            yield s

    def ping(self) -> bool:
        with self.read() as s:
            s.execute(select(1))
        return True

    def dispose(self) -> None:
        self.engine.dispose()
        self.immediate_engine.dispose()
