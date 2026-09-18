"""Mock historical data adapter: synthetic OHLC data for testing. No file I/O, no API calls."""

import hashlib
import math
import random
from datetime import datetime, timedelta, timezone

import pandas as pd

from adapters.base import HistoricalDataAdapter

_OHLC_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
_DEFAULT_COUNT = 252  # roughly one trading year


class MockHistoricalAdapter(HistoricalDataAdapter):
    """Deterministic synthetic OHLC generator for tests and local dev."""

    def __init__(self, start_price: float = 75.0, daily_vol: float = 0.02, seed: int = 42):
        self._start_price = start_price
        self._daily_vol = daily_vol
        self._seed = seed

    def _symbol_seed(self, symbol: str) -> int:
        """Derive a stable per-symbol seed from the base seed and symbol name."""
        digest = hashlib.sha256(f"{self._seed}:{symbol}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    @staticmethod
    def _as_utc(dt: datetime) -> datetime:
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    def _date_range(
        self, start: datetime | None, end: datetime | None, count: int | None
    ) -> list[datetime]:
        end = self._as_utc(end) if end is not None else datetime.now(timezone.utc)
        if start is not None:
            start = self._as_utc(start)
            n_days = (end.date() - start.date()).days + 1
            dates = [start + timedelta(days=i) for i in range(n_days)]
            if count is not None:
                dates = dates[-count:]
            return dates
        n = count or _DEFAULT_COUNT
        return [end - timedelta(days=i) for i in range(n - 1, -1, -1)]

    def get_ohlc(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        count: int | None = None,
    ) -> pd.DataFrame:
        """Return a deterministic synthetic daily OHLC DataFrame for the requested range."""
        dates = self._date_range(start, end, count)
        rnd = random.Random(self._symbol_seed(symbol))
        price = self._start_price

        rows = []
        for dt in dates:
            daily_return = rnd.gauss(0, self._daily_vol)
            open_price = price
            close_price = round(open_price * math.exp(daily_return), 2)
            high_price = round(
                max(open_price, close_price) * (1 + abs(rnd.gauss(0, self._daily_vol / 2))), 2
            )
            low_price = round(
                min(open_price, close_price) * (1 - abs(rnd.gauss(0, self._daily_vol / 2))), 2
            )
            volume = round(abs(rnd.gauss(50_000, 10_000)), 2)
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": self._as_utc(dt),
                    "open": round(open_price, 2),
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )
            price = close_price

        return pd.DataFrame(rows, columns=_OHLC_COLUMNS)

    def get_ohlc_bulk(
        self,
        symbols: list[str],
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        count: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return {symbol: DataFrame} by calling get_ohlc per symbol."""
        return {symbol: self.get_ohlc(symbol, interval, start, end, count) for symbol in symbols}
