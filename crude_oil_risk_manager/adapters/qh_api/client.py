"""HTTP transport for the QH API: auth, per-endpoint rate limiting, retries, pagination.

Follows the API's published scripting rules: Bearer token on every request; never call one
endpoint from parallel threads (a lock per endpoint serialises them); on 429 wait
Retry-After then retry (a few attempts); on 5xx / timeouts retry with 1s, 2s, 4s... waits;
never retry immediately. Public endpoints (limit=None) are not limited and need no token.

Each request costs budget, so paginated fetches are capped by `max_pages`.
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

import requests

from adapters.base import (
    APIConnectionError,
    APIError,
    APIServerError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from adapters.qh_api.rate_limiter import DEFAULT_LIMIT, EndpointRateLimiter, RateLimit

logger = logging.getLogger(__name__)

BASE_URL = "https://qh-api.corp.hertshtengroup.com/apis"
REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 5  # per request, across 429 / 5xx / timeout retries
MAX_429_ATTEMPTS = 3


@dataclass(frozen=True)
class EndpointSpec:
    """One API endpoint: how to call it and how hard it may be called."""

    name: str
    method: str
    path: str
    limit: RateLimit | None = DEFAULT_LIMIT  # None = public, unlimited, no token
    params: frozenset[str] = frozenset()  # allowed query params; empty = not checked
    pagination: str | None = None  # "offset" (limit/offset) | "page" (page/page_size) | None
    notes: str = ""


def serialize_param(value: Any, sep: str = ",") -> str:
    """Query-string form of a parameter value (lists joined, bools lower-cased, dates ISO)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set, frozenset)):
        return sep.join(serialize_param(v) for v in value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


class QHClient:
    """Rate-limited, retrying client for qh-api.corp.hertshtengroup.com."""

    def __init__(
        self,
        token_provider: Callable[[], str],
        base_url: str = BASE_URL,
        session: requests.Session | None = None,
        max_wait_seconds: float = 90.0,
        sleep: Callable[[float], None] = time.sleep,
        limiter_clock: Callable[[], float] = time.monotonic,
    ):
        """`token_provider` is called per request so a token changed in Settings applies at once."""
        self._token_provider = token_provider
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._max_wait = max_wait_seconds
        self._sleep = sleep
        self._limiter_clock = limiter_clock
        self._limiters: dict[str, EndpointRateLimiter] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _limiter_and_lock(self, spec: EndpointSpec) -> tuple[EndpointRateLimiter | None, threading.Lock]:
        # Limits are per endpoint (path), so GET and POST on the same path share one budget.
        with self._registry_lock:
            lock = self._locks.setdefault(spec.path, threading.Lock())
            if spec.limit is None:
                return None, lock
            limiter = self._limiters.get(spec.path)
            if limiter is None:
                limiter = EndpointRateLimiter(
                    spec.limit, name=spec.path, max_wait_seconds=self._max_wait,
                    clock=self._limiter_clock, sleep=self._sleep,
                )
                self._limiters[spec.path] = limiter
            return limiter, lock

    def limiter_for(self, spec: EndpointSpec) -> EndpointRateLimiter | None:
        """The shared limiter for an endpoint (None if public) - e.g. to show remaining budget."""
        return self._limiter_and_lock(spec)[0]

    def _headers(self, spec: EndpointSpec, token: str | None = None) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if spec.limit is not None:  # public endpoints need no token
            token = token or self._token_provider()
            if not token:
                raise AuthenticationError(
                    "API access token not configured. Please enter your Bearer token in the Settings tab."
                )
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _retry_wait(self, attempt: int) -> float:
        return float(2 ** attempt)  # 1s, 2s, 4s, ...

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def request(
        self, spec: EndpointSpec, params: dict | None = None, json_body: dict | None = None,
        path_args: dict | None = None, token: str | None = None, max_attempts: int | None = None,
    ) -> Any:
        """Call one endpoint and return its parsed JSON. Params with a None value are dropped.

        `path_args` fills `{placeholders}` in the URL path; the rate-limit budget stays shared
        across every value, since the limit is per endpoint, not per URL. `token` overrides the
        configured token for this call only (e.g. testing a token typed in Settings);
        `max_attempts` lowers the retry count for callers that poll again soon anyway.

        Raises the adapters.base APIError subtypes: AuthenticationError (401),
        RateLimitError (429 persisted, or the local budget is exhausted for longer than
        max_wait_seconds), APIServerError (5xx after retries), APITimeoutError,
        APIConnectionError, or APIError for any other failure.
        """
        query = {k: serialize_param(v) for k, v in (params or {}).items() if v is not None}
        limiter, lock = self._limiter_and_lock(spec)
        path = spec.path.format(**path_args) if path_args else spec.path
        url = f"{self._base_url}{path}"
        rate_limited = 0
        attempts = max_attempts or MAX_ATTEMPTS

        with lock:  # one in-flight request per endpoint
            for attempt in range(attempts):
                last = attempt == attempts - 1
                headers = self._headers(spec, token)
                if limiter is not None:
                    limiter.acquire()
                try:
                    response = self._session.request(
                        spec.method, url, params=query or None, json=json_body,
                        headers=headers, timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                except requests.exceptions.Timeout as exc:
                    if last:
                        raise APITimeoutError(f"{spec.path}: request timed out.") from exc
                    self._sleep(self._retry_wait(attempt))
                    continue
                except requests.exceptions.ConnectionError as exc:
                    if last:
                        raise APIConnectionError(f"{spec.path}: could not connect to the QH API.") from exc
                    self._sleep(self._retry_wait(attempt))
                    continue

                if limiter is not None:
                    limiter.observe(response.headers)
                status = response.status_code

                if status == 401:
                    raise AuthenticationError("Invalid or expired access token.")
                if status == 429:
                    rate_limited += 1
                    try:
                        retry_after = float(response.headers.get("Retry-After", 60))
                    except ValueError:
                        retry_after = 60.0
                    if rate_limited >= MAX_429_ATTEMPTS or last or retry_after > self._max_wait:
                        if limiter is not None:
                            limiter.block_for(retry_after)  # keep later callers off too
                        raise RateLimitError(f"{spec.path}: rate limited; retry in {retry_after:.0f}s.")
                    logger.warning("%s rate limited (429); waiting %.0fs", spec.path, retry_after)
                    if limiter is not None:
                        limiter.block_for(retry_after)
                    self._sleep(retry_after)
                    continue
                if 500 <= status < 600:
                    if last:
                        raise APIServerError(f"{spec.path}: server error {status}.")
                    self._sleep(self._retry_wait(attempt))
                    continue
                if status >= 400:
                    raise APIError(f"{spec.path}: HTTP {status}: {response.text[:300]}")

                try:
                    return response.json()
                except ValueError as exc:
                    raise APIError(f"{spec.path}: response was not valid JSON.") from exc

        raise APIError(f"{spec.path}: request failed after {attempts} attempts.")  # pragma: no cover

    def request_all(
        self, spec: EndpointSpec, params: dict | None = None, page_size: int = 1000, max_pages: int = 10
    ) -> list:
        """Collect `results` across pages of a paginated endpoint (one request per page).

        Stops at the last page or after `max_pages` requests - each page spends rate-limit
        budget, so the cap is deliberate. Raises ValueError if the endpoint isn't paginated.
        """
        if spec.pagination not in ("offset", "page"):
            raise ValueError(f"{spec.name} is not a paginated endpoint")
        params = dict(params or {})
        rows: list = []
        for page_index in range(max_pages):
            if spec.pagination == "offset":
                params.update(limit=page_size, offset=page_index * page_size)
            else:
                params.update(page=page_index + 1, page_size=page_size)
            payload = self.request(spec, params)
            batch = payload.get("results", []) if isinstance(payload, dict) else payload
            rows.extend(batch)
            last_page = isinstance(payload, dict) and "next" in payload and not payload["next"]
            if last_page or len(batch) < page_size:
                break
        else:
            logger.warning("%s: stopped after max_pages=%d; more pages exist", spec.name, max_pages)
        return rows
