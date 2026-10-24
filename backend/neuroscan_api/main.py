"""FastAPI application factory. Imports no torch and loads no model (B1)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neuroscan_ai.data.split_guard import SplitGuard  # torch-free
from neuroscan_api import API_PREFIX, __version__
from neuroscan_api.context import AppContext
from neuroscan_api.db.session import Database
from neuroscan_api.errors import install_handlers
from neuroscan_api.routes import router
from neuroscan_api.security import GuardMiddleware
from neuroscan_api.services.runs import mark_stale_runs
from neuroscan_api.settings import Settings, load_settings
from neuroscan_api.storage import Storage

LOGGER = logging.getLogger("neuroscan_api")
STALE_SWEEP_S = 10.0
DESCRIPTION = (
    "NeuroScan-ai research prototype API (not a medical device; no authentication; localhost "
    "only). Results support medical image review and do not establish a diagnosis. Model "
    "scores are uncalibrated and are not probabilities of disease."
)


def build_context(settings: Settings) -> AppContext:
    storage = Storage(settings.storage_root)
    db = Database.open(settings.database_path)
    db.init_schema()
    for stale in storage.temp.glob("*.part"):  # interrupted uploads from a previous process
        stale.unlink(missing_ok=True)
    return AppContext(settings, db, storage, SplitGuard(settings.split_file()))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    ctx = build_context(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def sweep():
            while True:
                await asyncio.sleep(STALE_SWEEP_S)
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(mark_stale_runs, ctx)

        task = asyncio.create_task(sweep())
        try:
            yield
        finally:
            task.cancel()
            ctx.db.dispose()

    debug = settings.debug
    app = FastAPI(
        title="NeuroScan-ai API",
        version=__version__,
        description=DESCRIPTION,
        docs_url=f"{API_PREFIX}/docs" if debug else None,
        redoc_url=None,
        openapi_url=f"{API_PREFIX}/openapi.json" if debug else None,
        lifespan=lifespan,
    )
    app.state.ctx = ctx
    install_handlers(app)
    app.include_router(router)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["content-type"],
        )
    app.add_middleware(
        GuardMiddleware,
        allowed_hosts=settings.allowed_hosts,
        max_request_bytes=settings.limits.max_request_bytes,
    )
    return app
