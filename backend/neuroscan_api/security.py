"""Pure ASGI guard middleware: request id, Host allow-list (DNS-rebinding defence), upload size
pre-check before the body is read, safe default response headers and a path-free access log."""

from __future__ import annotations

import json
import logging
import time
import uuid

from neuroscan_api import API_PREFIX

ACCESS_LOG = logging.getLogger("neuroscan_api.access")
UPLOAD_PATH = f"{API_PREFIX}/cases"


def _error(code: str, message: str, request_id: str) -> bytes:
    return json.dumps(
        {
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "retryable": False,
                "details": {},
            }
        }
    ).encode()


class GuardMiddleware:
    def __init__(self, app, allowed_hosts: list[str], max_request_bytes: int) -> None:
        self.app = app
        self.allowed_hosts = {h.lower() for h in allowed_hosts}
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        started = time.perf_counter()
        status_holder = {"status": 0}

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                extra = [
                    (b"x-request-id", request_id.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                ]
                names = {k.lower() for k, _ in message.get("headers", [])}
                if b"cache-control" not in names:
                    extra.append((b"cache-control", b"no-store"))
                message["headers"] = list(message.get("headers", [])) + extra
            await send(message)

        async def reject(status: int, code: str, message: str):
            body = _error(code, message, request_id)
            await send_with_headers(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})

        try:
            host = headers.get("host", "").rsplit(":", 1)[0].strip("[]").lower()
            if host not in self.allowed_hosts:
                return await reject(400, "HOST_NOT_ALLOWED", "Host is not allowed.")
            if scope["method"] == "POST" and scope["path"].rstrip("/") == UPLOAD_PATH:
                length = headers.get("content-length")
                if length is None or not length.isdigit():
                    return await reject(411, "LENGTH_REQUIRED", "Content-Length is required.")
                if int(length) > self.max_request_bytes:
                    return await reject(413, "REQUEST_TOO_LARGE", "Upload request is too large.")
            return await self.app(scope, receive, send_with_headers)
        finally:
            # route template would need routing info; the raw path holds only server IDs
            ACCESS_LOG.info(
                "%s %s %s %.1fms rid=%s",
                scope["method"],
                scope["path"],
                status_holder["status"],
                (time.perf_counter() - started) * 1000,
                request_id,
            )
