"""DataLoader: the single place analysis modules read price series from local Parquet.

Analysis modules (correlation, VaR, regression, ...) must never read Parquet
directly; they go through DataLoader. The historical adapter is used only to
backfill a symbol whose Parquet file does not exist yet — never for normal reads.

Timestamps are normalised to midnight UTC so that series from different
exchanges (whose daily candles may carry different intraday timestamps) can be
aligned on calendar date with a plain inner join.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from adapters.base import (
    API_CODE_TO_PRODUCT,
    APIError,
    HistoricalDataAdapter,
    SymbolTranslator,
)
from core.exceptions import (
    CrudeOilRiskError,
    DataNotAvailableError,
    InsufficientDataError,
    InvalidSymbolError,
)

logger = logging.getLogger(__name__)

_MIN_COMMON_DATES = 20
_DEFAULT_BACKFILL_DAYS = 365 * 3


class DataLoader:
    """Loads daily close series and price differences from local Parquet files."""

    def __init__(self, data_dir: str, historical_adapter: HistoricalDataAdapter):
        self._data_dir = Path(data_dir)
        self._adapter = historical_adapter

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parquet_path(self, symbol: str) -> Path:
        """Path of the Parquet file for an internal symbol: {data_dir}/{API_CODE}/{API_SYMBOL}.parquet."""
        try:
            api_symbol = SymbolTranslator.internal_to_api(symbol)
            api_code, _ = SymbolTranslator._match_product_prefix(api_symbol, API_CODE_TO_PRODUCT)
        except ValueError as exc:
            raise InvalidSymbolError(f"Unrecognized symbol {symbol!r}: {exc}") from exc
        return self._data_dir / api_code / f"{api_symbol}.parquet"

    def _read_series(self, symbol: str) -> pd.Series | None:
        """Read the full close series from disk; None if no Parquet file exists."""
        path = self._parquet_path(symbol)
        if not path.exists():
            return None
        df = pd.read_parquet(path, columns=["timestamp", "close"])
        if df.empty:
            return pd.Series(dtype=float, name=symbol, index=pd.DatetimeIndex([], tz="UTC"))
        index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)).normalize()
        series = pd.Series(df["close"].astype(float).to_numpy(), index=index, name=symbol)
        series = series.sort_index()
        series = series[~series.index.duplicated(keep="last")]
        return series.dropna()

    def _backfill(self, symbol: str, start: date | None, end: date | None) -> None:
        """Trigger an adapter backfill for a symbol with no local data."""
        backfill = getattr(self._adapter, "backfill_symbol", None)
        range_desc = f"start={start}, end={end}"
        if backfill is None:
            raise DataNotAvailableError(
                f"No local price data for {symbol} ({range_desc}) and the configured "
                f"historical adapter does not support backfill."
            )

        default_start = datetime.now(timezone.utc) - timedelta(days=_DEFAULT_BACKFILL_DAYS)
        backfill_start = default_start
        if start is not None:
            requested = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
            backfill_start = min(requested, default_start)
        backfill_end = None
        if end is not None:
            backfill_end = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)

        logger.info("No local data for %s; backfilling from %s", symbol, backfill_start.date())
        try:
            rows = backfill(symbol, backfill_start, backfill_end)
        except (APIError, RuntimeError, ValueError) as exc:
            raise DataNotAvailableError(
                f"Backfill failed for {symbol} ({range_desc}): {exc}"
            ) from exc
        if not rows:
            raise DataNotAvailableError(
                f"Backfill returned 0 rows for {symbol} ({range_desc})."
            )

    @staticmethod
    def _filter_range(series: pd.Series, start: date | None, end: date | None) -> pd.Series:
        if start is not None:
            series = series[series.index >= pd.Timestamp(start).tz_localize("UTC")]
        if end is not None:
            series = series[series.index < pd.Timestamp(end).tz_localize("UTC") + pd.Timedelta(days=1)]
        return series

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_close_series(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        min_rows: int = 20,
    ) -> pd.Series:
        """Load daily closes for an internal symbol as an ascending UTC-indexed Series.

        Backfills via the adapter if no local file exists. Never returns an
        empty Series or one containing NaN.

        Raises DataNotAvailableError if no data can be obtained, and
        InsufficientDataError if fewer than min_rows rows remain in range.
        """
        series = self._read_series(symbol)
        if series is None:
            self._backfill(symbol, start, end)
            series = self._read_series(symbol)
        if series is None or series.empty:
            raise DataNotAvailableError(
                f"No price data available for {symbol} (start={start}, end={end})."
            )

        series = self._filter_range(series, start, end).dropna().sort_index()
        if len(series) < min_rows:
            raise InsufficientDataError(
                f"Insufficient data for {symbol}: {len(series)} rows available "
                f"(start={start}, end={end}), minimum required is {min_rows}."
            )
        series.name = symbol
        return series

    def load_close_series_bulk(
        self,
        symbols: list[str],
        start: date | None = None,
        end: date | None = None,
        min_rows: int = 20,
    ) -> dict[str, pd.Series]:
        """Load close series for many symbols.

        Collects every failure and raises a single DataNotAvailableError
        listing ALL failed symbols, so the caller sees the full picture.
        """
        results: dict[str, pd.Series] = {}
        failures: dict[str, str] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.load_close_series(symbol, start, end, min_rows)
            except CrudeOilRiskError as exc:
                failures[symbol] = str(exc)

        if failures:
            details = "; ".join(f"{sym}: {msg}" for sym, msg in failures.items())
            raise DataNotAvailableError(
                f"Price data unavailable for {len(failures)} of {len(symbols)} symbol(s): {details}"
            )
        return results

    def load_price_differences(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        min_rows: int = 20,
    ) -> pd.Series:
        """Daily price differences (close_t - close_{t-1}); min_rows applies to the diff series."""
        closes = self.load_close_series(symbol, start, end, min_rows=min_rows + 1)
        diffs = closes.diff().dropna()
        diffs.name = symbol
        return diffs

    def align_series(
        self, series_a: pd.Series, series_b: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Inner-join two series on their common dates (no forward-fill).

        Raises InsufficientDataError if fewer than 20 common dates remain.
        """
        common = series_a.index.intersection(series_b.index).sort_values()
        dropped_a = len(series_a) - len(common)
        dropped_b = len(series_b) - len(common)
        logger.info(
            "Aligned %s and %s: %d common dates (dropped %d from %s, %d from %s)",
            series_a.name, series_b.name, len(common),
            dropped_a, series_a.name, dropped_b, series_b.name,
        )
        if len(common) < _MIN_COMMON_DATES:
            raise InsufficientDataError(
                f"Insufficient common dates for {series_a.name} and {series_b.name}: "
                f"{len(common)} available, minimum required is {_MIN_COMMON_DATES}."
            )
        return series_a.loc[common], series_b.loc[common]

    def load_daily_ohlc(self, symbol: str) -> pd.DataFrame | None:
        """Local daily candles [open, high, low, close] indexed by midnight-UTC date.

        None if the symbol has no Parquet file. Never backfills (used by the VaR &
        Scenarios tab, which must not trigger API calls).
        """
        path = self._parquet_path(symbol)
        if not path.exists():
            return None
        df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
        if df.empty:
            return None
        index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)).normalize()
        frame = df[["open", "high", "low", "close"]].astype(float).set_axis(index).sort_index()
        return frame[~frame.index.duplicated(keep="last")].dropna()

    def get_available_date_range(self, symbol: str) -> tuple[date, date] | None:
        """(earliest_date, latest_date) in the local Parquet, or None if no data exists."""
        series = self._read_series(symbol)
        if series is None or series.empty:
            return None
        return series.index[0].date(), series.index[-1].date()
