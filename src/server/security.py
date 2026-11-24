"""Access control for the local server.

* **Loopback by default.** Binding to any other address always requires a token, so the
  interpreter (and the microphone audio sent to it) is never open to the network unauthenticated.
* **Token** from ``server.token_env`` (default ``SWI_SERVER_TOKEN``); generated and printed once
  when required but not set. Clients send it as ``Authorization: Bearer <token>`` or ``?token=``
  (browsers cannot set headers on WebSockets). Compared in constant time.
* **Origin check.** A web page on another site could otherwise open a WebSocket to
  ``ws://127.0.0.1:8000`` from the user's browser and listen in. Browser requests must come from
  the server's own origin (or ``server.allowed_origins``); requests without an ``Origin`` header
  (curl, Python clients) are not browser-driven and rely on the token.
* **Rate limit** on the HTTP API per client address (sliding one-minute window).
"""

from __future__ import annotations

import hmac
import ipaddress
import secrets
import time
from collections import defaultdict, deque


def is_loopback(host: str) -> bool:
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class AccessPolicy:
    def __init__(self, token: str | None, *, required: bool, allowed_origins: list[str] | None = None,
                 rate_limit_per_min: int = 120) -> None:
        if required and not token:
            raise ValueError("a token is required but none was given")
        self.token = token
        self.required = required
        self.allowed_origins = {o.rstrip("/").lower() for o in (allowed_origins or [])}
        self.rate_limit = rate_limit_per_min
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    @classmethod
    def for_host(cls, host: str, token: str | None, *, require_token: bool = False,
                 allowed_origins: list[str] | None = None, rate_limit_per_min: int = 120) -> tuple[AccessPolicy, bool]:
        """Returns (policy, token_was_generated)."""
        required = require_token or not is_loopback(host)
        generated = False
        if required and not token:
            token, generated = secrets.token_urlsafe(24), True
        return cls(token, required=required, allowed_origins=allowed_origins,
                   rate_limit_per_min=rate_limit_per_min), generated

    def token_ok(self, presented: str | None) -> bool:
        if not self.required:
            return True
        return bool(presented) and hmac.compare_digest(presented.encode(), (self.token or "").encode())

    def origin_ok(self, origin: str | None, host_header: str | None) -> bool:
        if not origin:
            return True                        # not a browser request
        origin = origin.rstrip("/").lower()
        if origin in self.allowed_origins:
            return True
        if not host_header:
            return False
        host = host_header.lower()
        return origin in {f"http://{host}", f"https://{host}"}

    def rate_ok(self, client: str) -> bool:
        now = time.monotonic()
        q = self._hits[client]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= self.rate_limit:
            return False
        q.append(now)
        return True


def presented_token(headers: dict[str, str] | object, query: dict[str, str] | object) -> str | None:
    auth = headers.get("authorization", "")  # type: ignore[attr-defined]
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return query.get("token")  # type: ignore[attr-defined]
