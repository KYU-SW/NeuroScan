"""Launcher: ``python -m neuroscan_api`` — refuses non-loopback hosts (no authentication)."""

from __future__ import annotations

import logging
import sys

import uvicorn

from neuroscan_api.main import create_app
from neuroscan_api.settings import is_loopback, load_settings


def main() -> int:
    settings = load_settings()
    if not is_loopback(settings.host):
        print(
            "[neuroscan-api] refusing to bind a non-loopback host: this development build has no "
            "authentication and must only listen on 127.0.0.1 / localhost.",
            file=sys.stderr,
        )
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # uvicorn's own access log is disabled: GuardMiddleware logs method, path, status, request id
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
