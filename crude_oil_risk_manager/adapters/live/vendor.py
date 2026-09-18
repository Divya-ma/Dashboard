"""Live market data vendor adapter: fetches the latest 1-minute candle per symbol."""

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
from db.repository import Repository

logger = logging.getLogger(__name__)

_BASE_URL = "https://qh-api.corp.hertshtengroup.com/apis"
_OHLC_ENDPOINT = f"{_BASE_URL}/ohlc/"
_REQUEST_TIMEOUT_SECONDS = 10
_MAX_INSTRUMENTS_PER_REQUEST = 50


class VendorLiveAdapter(LiveDataAdapter):
    """Live price adapter backed by the qh-api OHLC endpoint."""

    def __init__(self, repository: Repository, staleness_threshold_seconds: float):
        self._repository = repository
        self._staleness_threshold_seconds = staleness_threshold_seconds

    def get_access_token(self) -> str:
        """Return the current Bearer token, fetched from the settings table."""
        token = self._repository.get_setting("api_access_token", "")
        if not token:
            raise RuntimeError(
                "API access token not configured. Please enter your Bearer "
                "token in the Settings tab."
            )
        return token

    def get_live_prices(self, symbols: list[str]) -> dict[str, LivePrice]:
        """Return a map of internal_symbol -> LivePrice for the given internal symbols."""
        if not symbols:
            return {}

        token = self.get_access_token()
        internal_by_api: dict[str, str] = {
            SymbolTranslator.internal_to_api(symbol): symbol for symbol in symbols
        }

        results: dict[str, LivePrice] = {}
        api_codes = list(internal_by_api.keys())
        for batch_start in range(0, len(api_codes), _MAX_INSTRUMENTS_PER_REQUEST):
            batch = api_codes[batch_start : batch_start + _MAX_INSTRUMENTS_PER_REQUEST]
            results.update(self._fetch_batch(batch, internal_by_api, token))

        for symbol in symbols:
            if symbol not in results:
                logger.warning("No data returned for %s", symbol)

        return results

    def _fetch_batch(
        self, api_codes: list[str], internal_by_api: dict[str, str], token: str
    ) -> dict[str, LivePrice]:
        params = {
            "instruments": ",".join(api_codes),
            "interval": "1M",
            "count": 1,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "accept": "application/json",
        }

        try:
            response = requests.get(
                _OHLC_ENDPOINT, params=params, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS
            )
        except requests.exceptions.Timeout as exc:
            raise APITimeoutError("Request to the live price API timed out.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise APIConnectionError("Could not connect to the live price API.") from exc

        if response.status_code == 401:
            raise AuthenticationError("Invalid or expired access token.")
        if response.status_code == 429:
            raise RateLimitError("API rate limit exceeded.")
        if 500 <= response.status_code < 600:
            raise APIServerError(f"Live price API returned server error {response.status_code}.")
        response.raise_for_status()

        candles = response.json()
        now = datetime.now(timezone.utc)
        results: dict[str, LivePrice] = {}
        for candle in candles:
            api_code = candle["product"]
            internal_symbol = internal_by_api.get(api_code)
            if internal_symbol is None:
                # Response contains a product we did not request; skip it.
                continue
            candle_time = datetime.fromtimestamp(candle["time"] / 1000, tz=timezone.utc)
            is_stale = (now - candle_time).total_seconds() > self._staleness_threshold_seconds
            results[internal_symbol] = LivePrice(
                symbol=internal_symbol,
                price=candle["close"],
                timestamp=candle_time,
                is_stale=is_stale,
                raw_open=candle["open"],
                raw_high=candle["high"],
                raw_low=candle["low"],
                raw_volume=candle["volume"],
            )
        return results
