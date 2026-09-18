"""Historical market data vendor adapter: local Parquet cache + rate-limited API fallback."""

import logging
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from adapters.base import (
    API_CODE_TO_PRODUCT,
    APIConnectionError,
    APIError,
    APIServerError,
    APITimeoutError,
    AuthenticationError,
    HistoricalDataAdapter,
    RateLimitError,
    SymbolTranslator,
)
from core.models.structure import Structure
from db.repository import Repository

logger = logging.getLogger(__name__)

_BASE_URL = "https://qh-api.corp.hertshtengroup.com/apis"
_OHLC_ENDPOINT = f"{_BASE_URL}/ohlc/"
_REQUEST_TIMEOUT_SECONDS = 10
_MAX_INSTRUMENTS_PER_REQUEST = 50
_MAX_ROWS_PER_REQUEST = 10_000

_OHLC_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
_SYNC_LOG_COLUMNS = ["symbol", "last_sync_utc", "row_count", "status", "error_msg"]


class RateLimitedQueue:
    """Token-bucket rate limiter ensuring API calls never exceed calls_per_minute."""

    def __init__(self, calls_per_minute: int, refill_period_seconds: float = 60.0):
        self._calls_per_minute = calls_per_minute
        self._refill_period_seconds = refill_period_seconds
        self._tokens = calls_per_minute
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        if now - self._last_refill >= self._refill_period_seconds:
            self._tokens = self._calls_per_minute
            self._last_refill = now

    def acquire(self) -> None:
        """Block until a token is available, consuming one."""
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens > 0:
                    self._tokens -= 1
                    return
                wait_time = self._refill_period_seconds - (time.monotonic() - self._last_refill)
            wait_time = max(wait_time, 0.0)
            logger.debug("Rate limiter: no tokens available, waiting %.2fs", wait_time)
            if wait_time > 1:
                logger.info("Rate limiter: waiting %.1fs for token refill", wait_time)
            time.sleep(min(max(wait_time, 0.01), self._refill_period_seconds))

    def remaining_tokens(self) -> int:
        """Return the number of tokens currently available."""
        with self._lock:
            self._refill_locked()
            return self._tokens


class VendorHistoricalAdapter(HistoricalDataAdapter):
    """Historical OHLC adapter backed by a local Parquet cache with API fallback."""

    def __init__(self, repository: Repository, data_dir: str, calls_per_minute: int = 6):
        self._repository = repository
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        (self._data_dir / "_metadata").mkdir(parents=True, exist_ok=True)
        self._rate_limiter = RateLimitedQueue(calls_per_minute)

    # ------------------------------------------------------------------
    # Local cache methods
    # ------------------------------------------------------------------

    def _get_parquet_path(self, api_symbol: str) -> Path:
        """Return the Parquet path for an API symbol, e.g. 'COZ26' -> data_dir/CO/COZ26.parquet."""
        api_code, _ = SymbolTranslator._match_product_prefix(api_symbol, API_CODE_TO_PRODUCT)
        directory = self._data_dir / api_code
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{api_symbol}.parquet"

    def _load_local(self, api_symbol: str) -> pd.DataFrame | None:
        """Load the cached Parquet file for an API symbol, translating symbol to internal."""
        path = self._get_parquet_path(api_symbol)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if df.empty:
            return df
        internal_symbol = SymbolTranslator.api_to_internal(api_symbol)
        df = df.copy()
        df["symbol"] = internal_symbol
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        elif df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        return df.sort_values("timestamp").reset_index(drop=True)

    def _save_local(self, api_symbol: str, df: pd.DataFrame) -> None:
        """Deduplicate on timestamp, sort ascending, and persist to the local Parquet cache."""
        path = self._get_parquet_path(api_symbol)
        internal_symbol = SymbolTranslator.api_to_internal(api_symbol)
        if df is None or df.empty:
            out = pd.DataFrame(columns=_OHLC_COLUMNS)
        else:
            out = df.copy()
            out["symbol"] = internal_symbol
            out = out.drop_duplicates(subset="timestamp", keep="last")
            out = out.sort_values("timestamp").reset_index(drop=True)
        out.to_parquet(path, index=False)
        self._update_sync_log(internal_symbol, status="ok", row_count=len(out))

    def _sync_log_path(self) -> Path:
        return self._data_dir / "_metadata" / "sync_log.parquet"

    def _read_sync_log(self) -> pd.DataFrame:
        path = self._sync_log_path()
        if not path.exists():
            return pd.DataFrame(columns=_SYNC_LOG_COLUMNS)
        return pd.read_parquet(path)

    def _update_sync_log(
        self, symbol: str, status: str, error_msg: str = "", row_count: int | None = None
    ) -> None:
        """Upsert a row in sync_log.parquet for the given (internal) symbol."""
        log_df = self._read_sync_log()
        if row_count is None:
            existing = log_df.loc[log_df["symbol"] == symbol, "row_count"]
            row_count = int(existing.iloc[0]) if len(existing) else 0
        log_df = log_df[log_df["symbol"] != symbol]
        new_row = pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "last_sync_utc": datetime.now(timezone.utc).isoformat(),
                    "row_count": row_count,
                    "status": status,
                    "error_msg": error_msg,
                }
            ]
        )
        log_df = pd.concat([log_df, new_row], ignore_index=True)
        log_df.to_parquet(self._sync_log_path(), index=False)

    # ------------------------------------------------------------------
    # API fetch
    # ------------------------------------------------------------------

    def _fetch_from_api(
        self,
        api_symbols: list[str],
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        count: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Call the API and return {api_symbol: DataFrame} (internal symbol column, batched)."""
        token = self._repository.get_setting("api_access_token", "")
        if not token:
            raise RuntimeError(
                "API access token not configured. Please enter your Bearer "
                "token in the Settings tab."
            )

        result: dict[str, pd.DataFrame] = {}
        for i in range(0, len(api_symbols), _MAX_INSTRUMENTS_PER_REQUEST):
            chunk = api_symbols[i : i + _MAX_INSTRUMENTS_PER_REQUEST]
            self._rate_limiter.acquire()
            result.update(self._fetch_chunk(chunk, interval, token, start, end, count))
        return result

    @staticmethod
    def _fetch_chunk(
        api_symbols: list[str],
        interval: str,
        token: str,
        start: datetime | None,
        end: datetime | None,
        count: int | None,
    ) -> dict[str, pd.DataFrame]:
        params: dict[str, str | int] = {
            "instruments": ",".join(api_symbols),
            "interval": interval,
        }
        if start is not None:
            params["start"] = int(start.timestamp())
        if end is not None:
            params["end"] = int(end.timestamp())
        if count is not None:
            params["count"] = count
        headers = {"Authorization": f"Bearer {token}", "accept": "application/json"}

        try:
            response = requests.get(
                _OHLC_ENDPOINT, params=params, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS
            )
        except requests.exceptions.Timeout as exc:
            raise APITimeoutError("Request to the historical price API timed out.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise APIConnectionError("Could not connect to the historical price API.") from exc

        if response.status_code == 401:
            raise AuthenticationError("Invalid or expired access token.")
        if response.status_code == 429:
            raise RateLimitError("API rate limit exceeded.")
        if 500 <= response.status_code < 600:
            raise APIServerError(f"Historical API returned server error {response.status_code}.")
        response.raise_for_status()

        candles = response.json()
        by_api_symbol: dict[str, list[dict]] = {}
        for candle in candles:
            by_api_symbol.setdefault(candle["product"], []).append(candle)

        result: dict[str, pd.DataFrame] = {}
        for api_symbol, rows in by_api_symbol.items():
            internal_symbol = SymbolTranslator.api_to_internal(api_symbol)
            df = pd.DataFrame(
                [
                    {
                        "symbol": internal_symbol,
                        "timestamp": datetime.fromtimestamp(row["time"] / 1000, tz=timezone.utc),
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                    }
                    for row in rows
                ],
                columns=_OHLC_COLUMNS,
            )
            result[api_symbol] = df.sort_values("timestamp").reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    # Cache-range helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _covers_request(
        df: pd.DataFrame | None,
        start: datetime | None,
        end: datetime | None,
        count: int | None,
    ) -> bool:
        if df is None or df.empty:
            return False
        if count is not None and len(df) < count:
            return False
        if start is not None and df["timestamp"].min() > start:
            return False
        if end is not None and df["timestamp"].max() < end:
            return False
        return True

    @staticmethod
    def _filter(
        df: pd.DataFrame, start: datetime | None, end: datetime | None, count: int | None
    ) -> pd.DataFrame:
        filtered = df
        if start is not None:
            filtered = filtered[filtered["timestamp"] >= start]
        if end is not None:
            filtered = filtered[filtered["timestamp"] <= end]
        filtered = filtered.sort_values("timestamp")
        if count is not None:
            filtered = filtered.tail(count)
        return filtered.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public interface (HistoricalDataAdapter)
    # ------------------------------------------------------------------

    def get_ohlc(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        count: int | None = None,
    ) -> pd.DataFrame:
        api_symbol = SymbolTranslator.internal_to_api(symbol)

        if interval == "1D":
            local_df = self._load_local(api_symbol)
            if self._covers_request(local_df, start, end, count):
                result = self._filter(local_df, start, end, count)
            else:
                fetched = self._fetch_from_api([api_symbol], interval, start=start, end=end, count=count)
                new_df = fetched.get(api_symbol, pd.DataFrame(columns=_OHLC_COLUMNS))
                merged = pd.concat([local_df, new_df]) if local_df is not None else new_df
                self._save_local(api_symbol, merged)
                merged = self._load_local(api_symbol)
                result = self._filter(merged, start, end, count)
        else:
            fetched = self._fetch_from_api([api_symbol], interval, start=start, end=end, count=count)
            result = fetched.get(api_symbol, pd.DataFrame(columns=_OHLC_COLUMNS))

        if result.empty:
            logger.warning(
                "No OHLC data for %s (interval=%s, start=%s, end=%s, count=%s)",
                symbol, interval, start, end, count,
            )
        return result.reset_index(drop=True)

    def get_ohlc_bulk(
        self,
        symbols: list[str],
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        count: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}

        if interval == "1D":
            need_fetch: list[str] = []
            for symbol in symbols:
                api_symbol = SymbolTranslator.internal_to_api(symbol)
                local_df = self._load_local(api_symbol)
                if self._covers_request(local_df, start, end, count):
                    results[symbol] = self._filter(local_df, start, end, count)
                else:
                    need_fetch.append(symbol)

            if need_fetch:
                api_symbols = [SymbolTranslator.internal_to_api(s) for s in need_fetch]
                fetched = self._fetch_from_api(api_symbols, interval, start=start, end=end, count=count)
                for symbol in need_fetch:
                    api_symbol = SymbolTranslator.internal_to_api(symbol)
                    new_df = fetched.get(api_symbol, pd.DataFrame(columns=_OHLC_COLUMNS))
                    local_df = self._load_local(api_symbol)
                    merged = pd.concat([local_df, new_df]) if local_df is not None else new_df
                    self._save_local(api_symbol, merged)
                    merged = self._load_local(api_symbol)
                    results[symbol] = self._filter(merged, start, end, count)
        else:
            api_symbols = [SymbolTranslator.internal_to_api(s) for s in symbols]
            fetched = self._fetch_from_api(api_symbols, interval, start=start, end=end, count=count)
            for symbol in symbols:
                api_symbol = SymbolTranslator.internal_to_api(symbol)
                results[symbol] = fetched.get(api_symbol, pd.DataFrame(columns=_OHLC_COLUMNS))

        for symbol in symbols:
            if symbol not in results:
                results[symbol] = pd.DataFrame(columns=_OHLC_COLUMNS)
            if results[symbol].empty:
                logger.warning("No OHLC data for %s (interval=%s)", symbol, interval)

        return results

    # ------------------------------------------------------------------
    # Morning sync
    # ------------------------------------------------------------------

    def get_symbols_needing_sync(self, repository: Repository) -> list[str]:
        """Return internal symbols whose sync_log entry is missing or not from today (UTC)."""
        contracts = repository.get_all_contracts()
        today = datetime.now(timezone.utc).date()
        log_df = self._read_sync_log()

        needing: list[str] = []
        for contract in contracts:
            symbol = contract.symbol
            row = log_df[log_df["symbol"] == symbol]
            if row.empty:
                needing.append(symbol)
                continue
            last_sync = pd.to_datetime(row.iloc[0]["last_sync_utc"])
            if last_sync.date() != today:
                needing.append(symbol)
        return needing

    def run_morning_sync(self, repository: Repository) -> dict[str, str]:
        """Fetch the latest 1D candle for all symbols not yet synced today. Synchronous."""
        logger.info("Morning sync: starting at %s", datetime.now(timezone.utc).isoformat())
        symbols = self.get_symbols_needing_sync(repository)
        if not symbols:
            logger.info("Morning sync: all symbols up to date")
            return {}

        results: dict[str, str] = {}
        api_symbols = [SymbolTranslator.internal_to_api(s) for s in symbols]
        try:
            fetched = self._fetch_from_api(api_symbols, "1D", count=1)
        except (APIError, RuntimeError) as exc:
            logger.error("Morning sync: batch fetch failed: %s", exc)
            for symbol in symbols:
                self._update_sync_log(symbol, status="error", error_msg=str(exc))
                results[symbol] = f"error: {exc}"
            logger.info("Morning sync: completed at %s", datetime.now(timezone.utc).isoformat())
            return results

        for symbol in symbols:
            api_symbol = SymbolTranslator.internal_to_api(symbol)
            new_df = fetched.get(api_symbol)
            if new_df is None or new_df.empty:
                logger.warning("Morning sync: no data returned for %s", symbol)
                self._update_sync_log(symbol, status="error", error_msg="no data returned")
                results[symbol] = "error: no data returned"
                continue
            try:
                local_df = self._load_local(api_symbol)
                merged = pd.concat([local_df, new_df]) if local_df is not None else new_df
                self._save_local(api_symbol, merged)
                results[symbol] = "ok"
            except Exception as exc:  # noqa: BLE001 - record and continue syncing other symbols
                logger.error("Morning sync: failed to save %s: %s", symbol, exc)
                self._update_sync_log(symbol, status="error", error_msg=str(exc))
                results[symbol] = f"error: {exc}"

        logger.info("Morning sync: completed at %s", datetime.now(timezone.utc).isoformat())
        return results

    def run_morning_sync_async(self, repository: Repository) -> threading.Thread:
        """Run run_morning_sync on a background daemon thread so app startup is not blocked."""
        thread = threading.Thread(
            target=self.run_morning_sync, args=(repository,), daemon=True, name="morning-sync"
        )
        thread.start()
        return thread

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    def _backfill_api_symbol(
        self, api_symbol: str, start: datetime, end: datetime, log_label: str
    ) -> int:
        total_days = (end - start).days + 1
        if total_days <= _MAX_ROWS_PER_REQUEST:
            chunks = [(start, end)]
        else:
            n_chunks = math.ceil(total_days / _MAX_ROWS_PER_REQUEST)
            logger.info(
                "Backfill %s: splitting into %d requests (%d days)", log_label, n_chunks, total_days
            )
            chunks = []
            current_start = start
            for _ in range(n_chunks):
                current_end = min(current_start + timedelta(days=_MAX_ROWS_PER_REQUEST - 1), end)
                chunks.append((current_start, current_end))
                current_start = current_end + timedelta(days=1)

        total_saved = 0
        for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            if len(chunks) > 1:
                logger.info(
                    "Backfill %s: request %d/%d (%s to %s)",
                    log_label, i, len(chunks), chunk_start, chunk_end,
                )
            fetched = self._fetch_from_api([api_symbol], "1D", start=chunk_start, end=chunk_end)
            df = fetched.get(api_symbol, pd.DataFrame(columns=_OHLC_COLUMNS))
            local_df = self._load_local(api_symbol)
            merged = pd.concat([local_df, df]) if local_df is not None else df
            self._save_local(api_symbol, merged)
            total_saved += len(df)

        self._update_sync_log(log_label, status="ok")
        return total_saved

    def backfill_symbol(self, symbol: str, start: datetime, end: datetime | None = None) -> int:
        """Fetch full 1D history for an internal symbol from start to end (default: today)."""
        end = end or datetime.now(timezone.utc)
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        api_symbol = SymbolTranslator.internal_to_api(symbol)
        return self._backfill_api_symbol(api_symbol, start, end, log_label=symbol)

    def backfill_structure(self, structure: Structure, start: datetime) -> dict[str, int]:
        """Backfill each unique contract in the structure, plus the structure's own symbol."""
        results: dict[str, int] = {}
        seen_symbols: set[str] = set()
        for leg in structure.legs:
            symbol = leg.contract.symbol
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            results[symbol] = self.backfill_symbol(symbol, start)

        try:
            structure_api_symbol = SymbolTranslator.structure_to_api_symbol(structure.legs)
        except NotImplementedError as exc:
            logger.warning("Backfill structure %s: %s", structure.structure_id, exc)
            return results

        end = datetime.now(timezone.utc)
        results[structure_api_symbol] = self._backfill_api_symbol(
            structure_api_symbol, start, end, log_label=structure_api_symbol
        )
        return results
