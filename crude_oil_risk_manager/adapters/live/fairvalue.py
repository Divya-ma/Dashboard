"""Live market data vendor adapter: fetches instantaneous fair-value prices from /fairvalue/.

Replaces the previous /ohlc/ 1-minute-candle live poll (adapters/live/vendor.py — still
the adapter used for HISTORICAL data, untouched) with a faster, lighter feed:
/fairvalue/ returns one current price per contract and is polled every 10 seconds
instead of every 60 (see LIVE_POLL_INTERVAL_MS in ui/app.py). /fairvalue/ carries its
own independent 7 req/min rate limit, separate from /ohlc/'s, so the two endpoints
never contend for the same budget; at one request per 10s poll (6 req/min) this
adapter stays under that limit on its own, without needing a token-bucket limiter.

Contract selection (the "products" query param) is NOT simply the caller's `symbols`
argument. Per spec the live feed must cover only:
  1. every unique leg symbol across OPEN (and legacy PARTIALLY_CLOSED) structures, and
  2. the correlation watchlist's instrument symbols,
rebuilt on structure/watchlist change rather than on every poll. This adapter
re-derives (1) fresh from the repository on every call (two cheap, already-indexed
SQLite reads) and only logs/rebuilds its cached view when the resulting symbol set
actually differs from the previous poll, so nothing is redundantly recomputed every
10 seconds. (2) is read from the `WATCHLIST_SETTING_KEY` settings-table entry.

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

import requests

from adapters.base import (
    APIConnectionError,
    APIServerError,
    APITimeoutError,
    AuthenticationError,
    LiveDataAdapter,
    LivePrice,
    RateLimitError,
    SymbolTranslator,
)
from core.models import StructureStatus
from db.repository import Repository

logger = logging.getLogger(__name__)

_BASE_URL = "https://qh-api.corp.hertshtengroup.com/apis"
_FAIRVALUE_ENDPOINT = f"{_BASE_URL}/fairvalue/"
_REQUEST_TIMEOUT_SECONDS = 10

# Settings-table key the correlation watchlist would need to be persisted under for its
# symbols to be picked up here — see the KNOWN GAP note in the module docstring.
WATCHLIST_SETTING_KEY = "correlation_watchlist_symbols"

_OPEN_STATUSES = [StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED]


class FairValueLiveAdapter(LiveDataAdapter):
    """Live price adapter backed by the qh-api /fairvalue/ endpoint."""

    def __init__(self, repository: Repository, staleness_threshold_seconds: float):
        self._repository = repository
        self._staleness_threshold_seconds = staleness_threshold_seconds
        self._last_symbols: frozenset[str] = frozenset()  # product set requested on the previous poll

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
        """Unique internal leg symbols across every OPEN/PARTIALLY_CLOSED structure."""
        structures = self._repository.get_all_structures(status_filter=_OPEN_STATUSES)
        return {
            leg.contract.symbol
            for structure in structures
            for leg in structure.legs
            if leg.is_traded
        }

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
        if target != self._last_symbols:
            logger.info(
                "Live feed contract list changed: %d -> %d symbols", len(self._last_symbols), len(target)
            )
            self._last_symbols = frozenset(target)
        if not target:
            return {}

        internal_by_api: dict[str, str] = {}
        for symbol in target:
            try:
                internal_by_api[SymbolTranslator.internal_to_api(symbol)] = symbol
            except ValueError:
                logger.warning("Skipping %r: not a translatable outright symbol for /fairvalue/", symbol)
        if not internal_by_api:
            return {}

        token = self.get_access_token()
        return self._fetch(list(internal_by_api), internal_by_api, token)

    def _fetch(
        self, api_codes: list[str], internal_by_api: dict[str, str], token: str
    ) -> dict[str, LivePrice]:
        params = {"products": ",".join(api_codes)}
        headers = {"Authorization": f"Bearer {token}", "accept": "application/json"}

        try:
            response = requests.get(
                _FAIRVALUE_ENDPOINT, params=params, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS
            )
        except requests.exceptions.Timeout as exc:
            raise APITimeoutError("Request to the fair value API timed out.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise APIConnectionError("Could not connect to the fair value API.") from exc

        if response.status_code == 401:
            raise AuthenticationError("Invalid or expired access token.")
        if response.status_code == 429:
            raise RateLimitError("API rate limit exceeded.")
        if 500 <= response.status_code < 600:
            raise APIServerError(f"Fair value API returned server error {response.status_code}.")
        response.raise_for_status()

        payload = response.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else payload
        now = datetime.now(timezone.utc)
        results: dict[str, LivePrice] = {}
        for row in rows:
            api_code = row["Contract"]
            internal_symbol = internal_by_api.get(api_code)
            if internal_symbol is None:
                # Response contains a product we did not request; skip it.
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

        for symbol in internal_by_api.values():
            if symbol not in results:
                logger.warning("No fair value data returned for %s", symbol)
        return results
