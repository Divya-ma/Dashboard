"""Per-endpoint rate limiting for the QH API.

The API limits requests per token AND per endpoint, in minute / hour / day windows
(default 7/min, 420/h, 10,080/day; /ohlc/ and /tas/ have their own). Rejected (429)
requests still count against the limit, so the goal is to never get rejected: every
request is recorded when it is sent, and `acquire()` blocks until ALL three windows have
room. The server's own X-RateLimit-* headers always win (`observe`), because the account
can have a different limit from the documented one.

Blocking is bounded by `max_wait_seconds`: waiting out an exhausted hourly or daily
window would freeze a Dash callback, so that case raises RateLimitError instead.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping

from adapters.base import RateLimitError

_MINUTE, _HOUR, _DAY = 60.0, 3600.0, 86400.0


@dataclass(frozen=True)
class RateLimit:
    per_minute: int
    per_hour: int
    per_day: int


DEFAULT_LIMIT = RateLimit(7, 420, 10_080)
OHLC_LIMIT = RateLimit(30, 1_800, 43_200)
TAS_LIMIT = RateLimit(10, 600, 14_400)


class EndpointRateLimiter:
    """Sliding-window limiter (minute, hour, day) for one endpoint. Thread-safe."""

    def __init__(
        self,
        limit: RateLimit,
        name: str = "endpoint",
        max_wait_seconds: float = 90.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
    ):
        self._windows = [(_MINUTE, limit.per_minute), (_HOUR, limit.per_hour), (_DAY, limit.per_day)]
        self.name = name
        self._max_wait = max_wait_seconds
        self._clock = clock
        self._sleep = sleep
        self._wall_clock = wall_clock
        self._stamps: deque[float] = deque()
        self._blocked_until = 0.0  # monotonic time before which nothing may be sent
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._stamps and now - self._stamps[0] >= _DAY:
            self._stamps.popleft()

    def _wait_needed(self, now: float) -> float:
        wait = max(0.0, self._blocked_until - now)
        for span, cap in self._windows:
            in_window = [t for t in self._stamps if now - t < span]
            if len(in_window) >= cap:
                # room appears once enough of the oldest stamps in this window have expired
                wait = max(wait, in_window[len(in_window) - cap] + span - now)
        return wait

    def acquire(self) -> None:
        """Block until a request may be sent, then record it. Raises RateLimitError if the
        wait would exceed max_wait_seconds (e.g. the hourly/daily budget is spent)."""
        while True:
            with self._lock:
                now = self._clock()
                self._prune(now)
                wait = self._wait_needed(now)
                if wait <= 0:
                    self._stamps.append(now)
                    return
            if wait > self._max_wait:
                raise RateLimitError(
                    f"{self.name}: rate limit budget exhausted; next request possible in {wait:.0f}s "
                    f"(longer than the {self._max_wait:.0f}s this call will wait)."
                )
            self._sleep(wait + 0.05)

    def block_for(self, seconds: float) -> None:
        """Forbid sending anything for `seconds` (server told us to back off)."""
        with self._lock:
            self._blocked_until = max(self._blocked_until, self._clock() + max(0.0, seconds))

    def observe(self, headers: Mapping[str, str]) -> None:
        """Honour X-RateLimit-Remaining / -Reset: at 0 remaining, wait until the window resets."""
        try:
            remaining = int(headers.get("X-RateLimit-Remaining", ""))
            reset = float(headers.get("X-RateLimit-Reset", ""))
        except (TypeError, ValueError):
            return
        if remaining <= 0:
            self.block_for(reset - self._wall_clock() + 0.5)

    def remaining_this_minute(self) -> int:
        """Requests still available in the minute window (for status displays)."""
        with self._lock:
            now = self._clock()
            used = sum(1 for t in self._stamps if now - t < _MINUTE)
            return max(0, self._windows[0][1] - used)
