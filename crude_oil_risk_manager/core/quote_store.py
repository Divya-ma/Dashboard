"""Price feeds for the Curve Compare page: TAQ quotes and /fairvalue/ prices for the whole ladder.

QuoteStore - latest best bid / best ask per symbol from /taq/ (trade and quote). /taq/ has no
    "latest quote" call: it returns the stream of quote CHANGES of a product for given dates and a
    time-of-day window (UTC). A quiet back-month contract with a standing bid and ask therefore
    produces no rows for hours, so the look-back is staged per symbol:

      1. no quote yet            -> last BOOTSTRAP_SECONDS (cheap; finds everything that is active)
      2. still nothing           -> yesterday and today in full, at most LONG_CHUNK symbols per
                                    request (quiet symbols have few rows), retried every
                                    LONG_RETRY_SECONDS. Some contracts have no quotes at all for days.
      3. has a quote             -> only the last WINDOW_SECONDS, overlapping the next poll.

FairValueStore - /fairvalue/ prices for every symbol the page needs, refreshed every few minutes.
    /fairvalue/ prices almost the whole 36-month ladder, but a quiet back month's price can be
    hours or days old, so each value keeps its own timestamp and the page shows its age.

Both only poll what the page has asked for (`request`); demand lapses DEMAND_TTL_SECONDS after the
page last asked, so nothing is polled while the page is closed. /taq/ allows 7 requests/minute per
token and the loop keeps to ~5; /fairvalue/ shares its 7/minute with the live price poll, so it is
called only every FAIR_REFRESH_SECONDS.

The /taq/ reply is paged (10,000 rows) and sorted OLDEST FIRST, so page 1 holds the oldest quotes of
the window and the newest are on the LAST page. Reading only page 1 would keep an old quote for any busy
symbol, so a multi-page reply is read from the last page backwards until every symbol asked for has been
seen (at most MAX_EXTRA_PAGES more requests), and the chunk size is halved for the next request.
"""

import logging
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from adapters.base import APIError, AuthenticationError, RateLimitError
from core.curve_compare import Quote

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 120
BOOTSTRAP_SECONDS = 900
LONG_RETRY_SECONDS = 1800
DEMAND_TTL_SECONDS = 300
CHUNK_SIZE = 25
LONG_CHUNK = 5
MIN_CHUNK_SIZE = 3
MAX_EXTRA_PAGES = 4  # further /taq/ pages read when a reply spans several, newest first
POLL_INTERVAL_SECONDS = 12.5  # ~4.8 requests/minute, under the 7/minute limit
PRIMARY_BONUS_SECONDS = 40  # the curve's own symbols are re-polled sooner than its fallbacks

FAIR_REFRESH_SECONDS = 600
FAIR_CHUNK = 100
FAIR_RETRY_SECONDS = 60


def parse_quotes(rows: list[dict]) -> dict[str, Quote]:
    """Newest two-sided quote of each instrument in a /taq/ reply."""
    latest: dict[str, tuple[float, dict]] = {}
    for row in rows:
        bid, ask, symbol = row.get("best_bid"), row.get("best_ask"), row.get("instrument")
        if not symbol or bid is None or ask is None:
            continue
        try:
            ts = float(row["timestamp"]) / 1e6  # microseconds since the epoch
            if not (math.isfinite(float(bid)) and math.isfinite(float(ask)) and math.isfinite(ts)):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        if symbol not in latest or ts >= latest[symbol][0]:
            latest[symbol] = (ts, row)
    return {
        symbol: Quote(
            float(row["best_bid"]), float(row["best_ask"]),
            float(row.get("bid_size") or 0.0), float(row.get("ask_size") or 0.0), ts,
        )
        for symbol, (ts, row) in latest.items()
    }


class QuoteStore:
    def __init__(
        self, api, clock: Callable[[], float] = time.time, autostart: bool = True,
        poll_interval: float = POLL_INTERVAL_SECONDS, chunk_size: int = CHUNK_SIZE,
    ):
        self._api = api
        self._clock = clock
        self._autostart = autostart
        self._poll_interval = poll_interval
        self._chunk = chunk_size
        self._long_chunk = LONG_CHUNK
        self._lock = threading.Lock()
        self._quotes: dict[str, Quote] = {}
        self._demand: dict[tuple, tuple[float, list[str], int]] = {}  # key -> (expires, symbols, n primary)
        self._attempted: dict[str, float] = {}
        self._bootstrapped: dict[str, float] = {}  # symbol -> when its short look-back was tried
        self._long_tried: dict[str, float] = {}
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None
        self.requests_made = 0

    # -- demand ---------------------------------------------------------

    def request(self, key: tuple, symbols: list[str], primary: int = 0) -> None:
        """The page needs these symbols (the first `primary` matter most). Starts the poller on first use."""
        with self._lock:
            self._demand[key] = (self._clock() + DEMAND_TTL_SECONDS, list(symbols), primary)
        if self._autostart:
            self.start()
        self._wake.set()

    def _active(self) -> tuple[list[str], set[str]]:
        """(wanted symbols, the primary ones) from unexpired demand; drops expired demand."""
        now = self._clock()
        with self._lock:
            self._demand = {k: v for k, v in self._demand.items() if v[0] > now}
            wanted: dict[str, None] = {}
            primary: set[str] = set()
            for _, symbols, n_primary in self._demand.values():
                wanted.update(dict.fromkeys(symbols))
                primary.update(symbols[:n_primary])
        return list(wanted), primary

    def quotes(self, symbols: list[str] | None = None) -> dict[str, Quote]:
        with self._lock:
            if symbols is None:
                return dict(self._quotes)
            return {s: self._quotes[s] for s in symbols if s in self._quotes}

    # -- polling --------------------------------------------------------

    def _mode(self, symbol: str, now: float) -> str:
        if symbol in self._quotes:
            return "short"
        if symbol not in self._bootstrapped:
            return "boot"
        if now - self._long_tried.get(symbol, 0.0) >= LONG_RETRY_SECONDS:
            return "long"
        return "short"

    def _pick(self) -> list[tuple[str, str]]:
        """(symbol, mode) for the next request: most overdue first, a few long look-backs at most."""
        wanted, primary = self._active()
        now = self._clock()
        order = {s: i for i, s in enumerate(wanted)}
        wanted.sort(key=lambda s: (self._attempted.get(s, 0.0) - (PRIMARY_BONUS_SECONDS if s in primary else 0), order[s]))
        picked, longs = [], 0
        for symbol in wanted:
            mode = self._mode(symbol, now)
            if mode == "long":
                if longs >= self._long_chunk:
                    continue
                longs += 1
            picked.append((symbol, mode))
            if len(picked) >= self._chunk:
                break
        return picked

    def refresh_once(self) -> int:
        """One /taq/ request for the symbols that are most overdue. Returns how many got a newer quote."""
        chunk = self._pick()
        if not chunk:
            return 0
        now = self._clock()
        today = datetime.fromtimestamp(now, tz=timezone.utc)
        yesterday = (today - timedelta(days=1)).date().isoformat()
        day_start = today.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        products = []
        for symbol, mode in chunk:
            if mode == "long":
                self._long_tried[symbol] = now
                products.append({"id": symbol, "dates": [yesterday, today.date().isoformat()],
                                 "start": "00:00:00", "end": "23:59:59"})
            else:
                if mode == "boot":
                    self._bootstrapped[symbol] = now
                window = BOOTSTRAP_SECONDS if mode == "boot" else WINDOW_SECONDS
                start = max(day_start, now - window)
                products.append({
                    "id": symbol, "dates": [today.date().isoformat()],
                    "start": datetime.fromtimestamp(start, tz=timezone.utc).strftime("%H:%M:%S"),
                    "end": today.strftime("%H:%M:%S"),
                })
            self._attempted[symbol] = now
        try:
            reply = self._api.taq(products)
        except AuthenticationError:
            self.last_error = "API token rejected: check it in Settings"
            return 0
        except RateLimitError:
            self.last_error = "Quote rate limit reached; waiting"
            return 0
        except APIError as exc:
            self.last_error = f"Quote request failed: {exc}"
            return 0
        self.requests_made += 1
        self.last_error = None
        rows = list(reply.get("data", [])) if isinstance(reply, dict) else []
        total_pages = int(reply.get("total_pages") or 1) if isinstance(reply, dict) else 1
        if total_pages > 1:
            self._chunk = max(MIN_CHUNK_SIZE, self._chunk // 2)
            self._long_chunk = max(1, self._long_chunk // 2)
            logger.info("/taq/ reply has %d pages; reading the newest, asking for %d symbols per request from now on", total_pages, self._chunk)
            rows += self._newest_pages(products, total_pages, {symbol for symbol, _ in chunk})
        fresh = parse_quotes(rows)
        updated = 0
        with self._lock:
            for symbol, quote in fresh.items():
                old = self._quotes.get(symbol)
                if old is None or quote.ts >= old.ts:
                    self._quotes[symbol] = quote
                    updated += 1
        return updated

    def _newest_pages(self, products: list[dict], total_pages: int, wanted: set[str]) -> list[dict]:
        """Rows of the last pages of a multi-page reply, newest page first, until every wanted symbol has
        been seen on them. Page 1 (already read) holds the OLDEST rows, so a symbol seen only there is not
        done: its newer quotes, if any, are on a later page."""
        rows: list[dict] = []
        seen: set = set()
        for page in range(total_pages, 1, -1):
            if wanted <= seen or total_pages - page >= MAX_EXTRA_PAGES:
                break
            try:
                reply = self._api.taq(products, page=page)
            except APIError as exc:  # the first page's rows are still used
                self.last_error = f"Quote request failed: {exc}"
                break
            self.requests_made += 1
            batch = reply.get("data", []) if isinstance(reply, dict) else []
            rows += batch
            seen = seen | {r.get("instrument") for r in batch}
        return rows

    # -- thread ---------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, name="quote-store", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while True:
            try:
                wanted, _ = self._active()
                if not wanted:
                    self._wake.clear()
                    self._wake.wait(timeout=60)
                    continue
                self.refresh_once()
            except Exception:  # noqa: BLE001 - the poller must outlive any single failure
                logger.exception("Quote poll crashed")
            time.sleep(self._poll_interval)


class FairValueStore:
    """/fairvalue/ price and its own timestamp for every symbol the page needs."""

    def __init__(
        self, api, clock: Callable[[], float] = time.time, autostart: bool = True,
        refresh_seconds: float = FAIR_REFRESH_SECONDS, chunk_size: int = FAIR_CHUNK,
    ):
        self._api = api
        self._clock = clock
        self._autostart = autostart
        self._refresh = refresh_seconds
        self._chunk = chunk_size
        self._lock = threading.Lock()
        self._prices: dict[str, tuple[float, float]] = {}  # symbol -> (price, unix seconds)
        self._wanted: dict[str, float] = {}  # symbol -> demand expiry
        self._fetched: dict[str, float] = {}  # symbol -> when it was last asked for
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def request(self, symbols: list[str]) -> None:
        expires = self._clock() + DEMAND_TTL_SECONDS
        with self._lock:
            new = any(s not in self._wanted for s in symbols)
            for symbol in symbols:
                self._wanted[symbol] = expires
        if self._autostart:
            self.start()
        if new:
            self._wake.set()

    def prices(self, symbols: list[str] | None = None) -> dict[str, tuple[float, float]]:
        with self._lock:
            if symbols is None:
                return dict(self._prices)
            return {s: self._prices[s] for s in symbols if s in self._prices}

    def _due(self) -> list[str]:
        now = self._clock()
        with self._lock:
            self._wanted = {s: e for s, e in self._wanted.items() if e > now}
            return [s for s in self._wanted if now - self._fetched.get(s, -math.inf) >= self._refresh]

    def refresh_once(self) -> int:
        """Ask /fairvalue/ for every symbol that is due. Returns how many prices came back."""
        due = self._due()
        got = 0
        for i in range(0, len(due), self._chunk):
            chunk = due[i:i + self._chunk]
            try:
                payload = self._api.fairvalue(chunk)
            except (AuthenticationError, APIError) as exc:
                self.last_error = (
                    "API token rejected: check it in Settings" if isinstance(exc, AuthenticationError)
                    else f"Fair value request failed: {exc}"
                )
                with self._lock:  # try again in a minute, not in a tight loop
                    for symbol in due:
                        self._fetched[symbol] = self._clock() - self._refresh + FAIR_RETRY_SECONDS
                return got
            self.last_error = None
            rows = payload.get("data", []) if isinstance(payload, dict) else payload
            now = self._clock()
            with self._lock:
                for symbol in chunk:
                    self._fetched[symbol] = now
                for row in rows or []:
                    try:
                        price, ts = float(row["Price"]), float(row["Timestamp"]) / 1000.0
                    except (KeyError, TypeError, ValueError):
                        continue
                    if row.get("Contract") in self._wanted and math.isfinite(price):
                        self._prices[row["Contract"]] = (price, ts)
                        got += 1
        return got

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, name="fairvalue-store", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while True:
            try:
                if self._due():
                    self.refresh_once()
                else:
                    self._wake.clear()
                    self._wake.wait(timeout=30)
            except Exception:  # noqa: BLE001
                logger.exception("Fair value poll crashed")
                time.sleep(30)
