"""SQLite engine/session: WAL, foreign keys, busy_timeout, explicit BEGIN / BEGIN IMMEDIATE.

pysqlite's own transaction handling is disabled (isolation_level=None) so that SQLAlchemy emits
the BEGIN itself; the ``immediate`` database uses ``BEGIN IMMEDIATE`` for queue claims, lease and
state transitions, which serialises writers and makes compare-and-set updates safe across the
API and worker processes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from neuroscan_api.db.models import SCHEMA_VERSION, Base, SchemaVersion

BUSY_TIMEOUT_MS = 5000


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

    def init_schema(self) -> None:
        Base.metadata.create_all(self.immediate_engine)
        with self.write() as s:
            row = s.get(SchemaVersion, 1)
            if row is None:
                s.add(SchemaVersion(id=1, version=SCHEMA_VERSION))
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
