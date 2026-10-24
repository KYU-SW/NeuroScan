"""``python -m neuroscan_api.worker``: the single GPU worker process (exits 3 if another live
worker holds the lease)."""

from __future__ import annotations

import logging
import sys

from neuroscan_api.db.session import Database
from neuroscan_api.settings import load_settings
from neuroscan_api.storage import Storage
from neuroscan_api.worker.loop import LeaseHeldError, Worker, default_engine


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = load_settings()
    storage = Storage(settings.storage_root)
    db = Database.open(settings.database_path)
    db.init_schema()
    worker = Worker(settings, db, storage, default_engine())
    try:
        worker.start()
    except LeaseHeldError:
        print("[neuroscan-worker] another GPU worker is already running", file=sys.stderr)
        return 3
    print(f"[neuroscan-worker] ready (model loads: {worker.load_count})", file=sys.stderr)
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        print("[neuroscan-worker] interrupted", file=sys.stderr)
    finally:
        worker.stop()
        db.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
