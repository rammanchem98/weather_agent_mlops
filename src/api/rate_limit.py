# src/api/rate_limit.py
import time
from collections import deque

from fastapi import Request

# The page is public and every /chat call costs at least one LLM call
# (the injection classifier), so cap how fast a single client can send.
CHAT_MAX_REQUESTS = 10
CHAT_WINDOW_SECONDS = 60


class RateLimiter:
    """Sliding-window limit per key (client IP), kept in memory.

    Deliberately simple: counts live in this process only, so they reset on
    restart and each Cloud Run instance counts separately (N instances means up
    to N x the limit). The deploy caps --max-instances to keep that bounded."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        hits = self._hits.setdefault(key, deque())

        # forget requests that have slid out of the window
        while hits and now - hits[0] >= self.window_seconds:
            hits.popleft()

        if len(hits) >= self.max_requests:
            return False
        hits.append(now)

        if len(self._hits) > 10_000:
            self._forget_idle_clients(now)
        return True

    def _forget_idle_clients(self, now: float) -> None:
        # stops the dict growing forever with IPs that never come back
        idle = [k for k, h in self._hits.items() if not h or now - h[-1] >= self.window_seconds]
        for key in idle:
            del self._hits[key]


def client_ip(request: Request) -> str:
    # Cloud Run appends the real client IP to X-Forwarded-For. Earlier entries
    # are whatever the client sent and can be faked, so only the last is trusted.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


chat_rate_limiter = RateLimiter(CHAT_MAX_REQUESTS, CHAT_WINDOW_SECONDS)
