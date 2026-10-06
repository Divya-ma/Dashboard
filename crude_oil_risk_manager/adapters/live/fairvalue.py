"""Live market data vendor adapter: fetches instantaneous fair-value prices from /fairvalue/.

Replaces the previous /ohlc/ 1-minute-candle live poll (adapters/live/vendor.py — still
the adapter used for HISTORICAL data, untouched) with a faster, lighter feed:
/fairvalue/ returns one current price per contract and is polled every 10 seconds
instead of every 60 (see LIVE_POLL_INTERVAL_MS in ui/app.py). /fairvalue/ carries its
own independent 7 req/min rate limit, separate from /ohlc/'s, so the two endpoints
never contend for the same budget. The request goes through the shared QHApi client
(adapters/qh_api), which enforces the minute/hour/day limits and the retry rules; at one
request per 10s poll (6 req/min) the poll stays under the per-minute limit by itself.

Contract selection (the "products" query param) is NOT simply the caller's `symbols`
argument. Per spec the live feed must cover only:
  1. every currently-traded leg symbol across OPEN (and legacy PARTIALLY_CLOSED)
     structures, plus every leg of a SHELL structure (nothing in it is "traded" yet by
     definition, so all of its legs are included — you want a live price while still
     deciding whether to enter a structure, not only after), and
  2. the correlation watchlist's instrument symbols,
rebuilt on structure/watchlist change rather than on every poll. This adapter
re-derives (1) fresh from the repository on every call (two cheap, already-indexed
SQLite reads) and only logs/rebuilds its cached view when the resulting symbol set
actually differs from the previous poll, so nothing is redundantly recomputed every
10 seconds. (2) is read from the `WATCHLIST_SETTING_KEY` settings-table entry. If the
combined set is empty, a poll is a silent no-op (no HTTP request, nothing to log) —
get_live_prices logs a one-time warning the first time this happens so it is not
mistaken for a working-but-quiet feed.

KNOWN GAP: nothing in this codebase currently writes `WATCHLIST_SETTING_KEY`. The
correlation watchlist today lives only in a client-side session dcc.Store
(ui/layouts/correlation_tab.py, storage_type="session") with no server-side
persistence, and adding that persistence means touching
ui/callbacks/correlation_callbacks.py, which is out of scope for this change (scope is
limited to the live price adapter and its registration in ui/app.py). Until something
writes `WATCHLIST_SETTING_KEY`, watchlist symbols are simply absent from the live feed
(an empty list read from settings) rather than silently faked or substituted.
"""

import logging
from datetime import datetime, timezone
from typing import Callable

from adapters.base import LiveDataAdapter, LivePrice, SymbolTranslator
from adapters.qh_api import QHApi, make_api
from core.models import StructureStatus
from db.repository import Repository

logger = logging.getLogger(__name__)

# The poll repeats every 10s, so a failed request is retried at most once rather than for
# the client's full backoff (which would stall the callback past the next poll).
_POLL_MAX_ATTEMPTS = 2

# Settings-table key the correlation watchlist would need to be persisted under for its
# symbols to be picked up here — see the KNOWN GAP note in the module docstring.
WATCHLIST_SETTING_KEY = "correlation_watchlist_symbols"

# Structure statuses whose legs feed the live poll: SHELL is included (see module docstring)
# so a structure you're still evaluating gets live prices, not just one you've already traded.
_LIVE_FEED_STATUSES = [StructureStatus.SHELL, StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED]


class FairValueLiveAdapter(LiveDataAdapter):
    """Live price adapter backed by the qh-api /fairvalue/ endpoint."""

    def __init__(self, repository: Repository, staleness_threshold_seconds: float, api: QHApi | None = None):
        """`api` should be the app's shared QHApi so /fairvalue/ has one rate-limit budget."""
        self._repository = repository
        self._api = api or make_api(lambda: repository.get_setting("api_access_token", ""))
        self._staleness_threshold_seconds = staleness_threshold_seconds
        # Product set requested on the previous poll; None until the first poll so that
        # poll always logs once (including the "nothing to poll" case) even if the target
        # set turns out to be empty from the very start.
        self._last_symbols: frozenset[str] | None = None
        # Extra API symbols (the Curve Kinks ladder: outrights, spreads, flies) priced in the SAME
        # request as the position prices, so they cost no extra call from the 7/min budget.
        self._curve_symbols: Callable[[], list[str]] | None = None
        self._curve_prices: dict[str, tuple[float, float]] = {}

    def set_curve_symbols_provider(self, provider: Callable[[], list[str]] | None) -> None:
        """Register the function returning the extra API symbols to price on every poll."""
        self._curve_symbols = provider

    def curve_prices(self) -> dict[str, tuple[float, float]]:
        """{API symbol: (price, unix seconds)} from the latest poll, for the extra symbols only.

        Kept apart from get_live_prices' result on purpose: these are curve structures, not
        positions, and must never reach the P&L code. Symbols the API did not return are absent."""
        return dict(self._curve_prices)

    def set_staleness_threshold(self, seconds: float) -> None:
        """Change the staleness threshold used to flag prices as stale."""
        self._staleness_threshold_seconds = seconds

    def get_access_token(self) -> str:
        """Return the current Bearer token, fetched from the settings table."""
        token = self._repository.get_setting("api_access_token", "")
        if not token:
            raise RuntimeError(
                "API access token not configured. Please enter your Bearer "
                "token in the Settings tab."
            )
        return token

    def check_connection(self, token: str, symbol: str = "CLZ26") -> int:
        """Make one real /fairvalue/ call with the given token, without touching the saved token.

        Returns the number of prices returned for `symbol`. Raises the same APIError
        subtypes as get_live_prices (e.g. AuthenticationError for a bad token).
        """
        api_symbol = SymbolTranslator.internal_to_api(symbol)
        return len(self._fetch([api_symbol], {api_symbol: symbol}, token))

    # ------------------------------------------------------------------
    # Contract selection: open structures + watchlist, not the caller's `symbols`
    # ------------------------------------------------------------------

    def _open_structure_leg_symbols(self) -> set[str]:
        """Unique internal leg symbols to keep live for every SHELL/OPEN/PARTIALLY_CLOSED structure.

        OPEN/PARTIALLY_CLOSED structures contribute only their currently-traded legs
        (leg.is_traded) — a partial exit can leave some legs flat, and there's no reason
        to poll those. A SHELL structure has no traded legs yet by definition, so every
        one of its legs is included instead (see module docstring).
        """
        structures = self._repository.get_all_structures(status_filter=_LIVE_FEED_STATUSES)
        symbols: set[str] = set()
        for structure in structures:
            if structure.status == StructureStatus.SHELL:
                symbols.update(leg.contract.symbol for leg in structure.legs)
            else:
                symbols.update(leg.contract.symbol for leg in structure.legs if leg.is_traded)
        return symbols

    def _watchlist_symbols(self) -> set[str]:
        """Instrument symbols from the persisted correlation watchlist, if any (see module docstring)."""
        return set(self._repository.get_setting(WATCHLIST_SETTING_KEY, []) or [])

    def _target_symbols(self) -> set[str]:
        """The full set of internal symbols the live feed should cover right now."""
        return self._open_structure_leg_symbols() | self._watchlist_symbols()

    # ------------------------------------------------------------------
    # Public interface (LiveDataAdapter)
    # ------------------------------------------------------------------

    def get_live_prices(self, symbols: list[str]) -> dict[str, LivePrice]:
        """Return a map of internal_symbol -> LivePrice.

        `symbols` is accepted for interface compatibility with LiveDataAdapter (callers
        are unchanged) but is deliberately NOT what gets requested from the API: per
        spec, the product list is derived internally from open structures + the
        correlation watchlist (see module docstring), not from whatever the caller
        happens to pass in.
        """
        target = self._target_symbols()
        if self._last_symbols is None or target != self._last_symbols:
            if not target:
                logger.warning(
                    "Live feed has nothing to poll: no SHELL/OPEN/PARTIALLY_CLOSED structure legs "
                    "and no correlation watchlist symbols. No live prices will be requested until "
                    "that changes."
                )
            else:
                logger.info(
                    "Live feed contract list changed: %d -> %d symbols",
                    len(self._last_symbols or ()), len(target),
                )
            self._last_symbols = frozenset(target)
        extras = list(self._curve_symbols()) if self._curve_symbols is not None else []
        if not target and not extras:
            return {}

        internal_by_api: dict[str, str] = {}
        for symbol in target:
            try:
                internal_by_api[SymbolTranslator.internal_to_api(symbol)] = symbol
            except ValueError:
                logger.warning("Skipping %r: not a translatable outright symbol for /fairvalue/", symbol)
        extras = [code for code in extras if code not in internal_by_api]
        if not internal_by_api and not extras:
            return {}

        token = self.get_access_token()
        return self._fetch([*internal_by_api, *extras], internal_by_api, token)

    def _fetch(
        self, api_codes: list[str], internal_by_api: dict[str, str], token: str
    ) -> dict[str, LivePrice]:
        payload = self._api.fairvalue(api_codes, token=token, max_attempts=_POLL_MAX_ATTEMPTS)
        rows = payload.get("data", []) if isinstance(payload, dict) else payload
        now = datetime.now(timezone.utc)
        results: dict[str, LivePrice] = {}
        extra_codes = set(api_codes) - set(internal_by_api)
        extra_prices: dict[str, tuple[float, float]] = {}
        for row in rows:
            api_code = row["Contract"]
            internal_symbol = internal_by_api.get(api_code)
            if internal_symbol is None:
                if api_code in extra_codes:
                    extra_prices[api_code] = (row["Price"], row["Timestamp"] / 1000)
                # anything else is a product we did not request; skip it
                continue
            price = row["Price"]
            price_time = datetime.fromtimestamp(row["Timestamp"] / 1000, tz=timezone.utc)
            is_stale = (now - price_time).total_seconds() > self._staleness_threshold_seconds
            results[internal_symbol] = LivePrice(
                symbol=internal_symbol,
                price=price,
                timestamp=price_time,
                is_stale=is_stale,
                # /fairvalue/ returns a single instantaneous price, not a candle: there is
                # no real OHLC/volume breakdown, so those fields collapse to the fair
                # value price (open/high/low) and zero (volume). Nothing downstream
                # currently consumes these beyond the raw store payload (see
                # ui/callbacks/shell_callbacks.py:fetch_live_prices).
                raw_open=price,
                raw_high=price,
                raw_low=price,
                raw_volume=0.0,
            )

        if extra_codes:
            self._curve_prices = extra_prices
        for symbol in internal_by_api.values():
            if symbol not in results:
                logger.warning("No fair value data returned for %s", symbol)
        return results
