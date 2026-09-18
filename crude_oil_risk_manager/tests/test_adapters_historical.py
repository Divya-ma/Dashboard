"""Tests for adapters.mock.mock_historical.MockHistoricalAdapter and
adapters.historical.vendor.VendorHistoricalAdapter / RateLimitedQueue.
"""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from adapters.historical.vendor import RateLimitedQueue, VendorHistoricalAdapter
from adapters.mock.mock_historical import MockHistoricalAdapter
from db.repository import Repository

_OHLC_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


# ---------- MockHistoricalAdapter ----------


def test_mock_historical_get_ohlc_returns_correct_columns():
    adapter = MockHistoricalAdapter()
    df = adapter.get_ohlc("CLZ26", "1D", count=10)
    assert list(df.columns) == _OHLC_COLUMNS


def test_mock_historical_get_ohlc_timestamps_are_utc_aware():
    adapter = MockHistoricalAdapter()
    df = adapter.get_ohlc("CLZ26", "1D", count=5)
    assert all(ts.tzinfo is not None for ts in df["timestamp"])
    assert all(ts.utcoffset() == timedelta(0) for ts in df["timestamp"])


def test_mock_historical_get_ohlc_count_returns_correct_row_count():
    adapter = MockHistoricalAdapter()
    df = adapter.get_ohlc("CLZ26", "1D", count=17)
    assert len(df) == 17


def test_mock_historical_get_ohlc_bulk_returns_dict():
    adapter = MockHistoricalAdapter()
    result = adapter.get_ohlc_bulk(["CLZ26", "COZ26"], "1D", count=5)
    assert set(result.keys()) == {"CLZ26", "COZ26"}
    assert all(len(df) == 5 for df in result.values())


# ---------- VendorHistoricalAdapter: local cache ----------


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture
def vendor_adapter(repo, tmp_path):
    return VendorHistoricalAdapter(repo, data_dir=str(tmp_path / "historical"), calls_per_minute=6)


def make_ohlc_df(symbol: str, dates: list[datetime]) -> pd.DataFrame:
    rows = [
        {
            "symbol": symbol,
            "timestamp": dt,
            "open": 75.0,
            "high": 76.0,
            "low": 74.0,
            "close": 75.5,
            "volume": 1000.0,
        }
        for dt in dates
    ]
    return pd.DataFrame(rows, columns=_OHLC_COLUMNS)


def test_save_local_then_load_local_roundtrips(vendor_adapter):
    dates = [datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc)]
    df = make_ohlc_df("CLZ26", dates)
    vendor_adapter._save_local("CLZ26", df)

    loaded = vendor_adapter._load_local("CLZ26")
    assert loaded is not None
    assert len(loaded) == 2
    assert list(loaded["timestamp"]) == sorted(dates)
    assert (loaded["symbol"] == "CLZ26").all()


def test_save_local_deduplicates_on_timestamp(vendor_adapter):
    dup_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    df1 = make_ohlc_df("CLZ26", [dup_date])
    vendor_adapter._save_local("CLZ26", df1)

    df2 = make_ohlc_df("CLZ26", [dup_date])
    df2.loc[0, "close"] = 99.99
    vendor_adapter._save_local("CLZ26", df2)

    loaded = vendor_adapter._load_local("CLZ26")
    assert len(loaded) == 1
    assert loaded.iloc[0]["close"] == 99.99


def test_load_local_returns_none_when_missing(vendor_adapter):
    assert vendor_adapter._load_local("CLZ26") is None


# ---------- VendorHistoricalAdapter: get_ohlc cache behavior ----------


def test_get_ohlc_returns_cached_data_without_api_call(vendor_adapter, mocker):
    dates = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    df = make_ohlc_df("CLZ26", dates)
    vendor_adapter._save_local("CLZ26", df)

    fetch_mock = mocker.patch.object(vendor_adapter, "_fetch_from_api")
    result = vendor_adapter.get_ohlc("CLZ26", "1D", count=1)

    fetch_mock.assert_not_called()
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "CLZ26"


def test_get_ohlc_calls_api_when_cache_missing(vendor_adapter, mocker):
    api_df = make_ohlc_df("CLZ26", [datetime(2026, 1, 1, tzinfo=timezone.utc)])
    fetch_mock = mocker.patch.object(
        vendor_adapter, "_fetch_from_api", return_value={"CLZ26": api_df}
    )

    result = vendor_adapter.get_ohlc("CLZ26", "1D", count=1)

    fetch_mock.assert_called_once()
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "CLZ26"


# ---------- RateLimitedQueue ----------


def test_rate_limited_queue_blocks_until_refill():
    queue = RateLimitedQueue(calls_per_minute=2, refill_period_seconds=0.3)
    queue.acquire()
    queue.acquire()
    assert queue.remaining_tokens() == 0

    start = time.monotonic()
    queue.acquire()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.25


def test_rate_limited_queue_remaining_tokens_decrements():
    queue = RateLimitedQueue(calls_per_minute=3, refill_period_seconds=60.0)
    assert queue.remaining_tokens() == 3
    queue.acquire()
    assert queue.remaining_tokens() == 2


# ---------- run_morning_sync ----------


def test_run_morning_sync_skips_symbols_already_synced_today(vendor_adapter, repo, mocker):
    contract = repo.get_contract("CLZ26")
    assert contract is None
    from core.models import Contract

    contract = Contract(
        product="CL",
        contract_month=12,
        contract_year=2026,
        symbol="CLZ26",
        multiplier=1000.0,
        tick_size=0.01,
        tick_value=10.0,
    )
    repo.save_contract(contract)

    vendor_adapter._update_sync_log("CLZ26", status="ok", row_count=1)

    fetch_mock = mocker.patch.object(vendor_adapter, "_fetch_from_api")
    result = vendor_adapter.run_morning_sync(repo)

    fetch_mock.assert_not_called()
    assert result == {}


def test_run_morning_sync_fetches_symbols_not_synced_today(vendor_adapter, repo, mocker):
    from core.models import Contract

    contract = Contract(
        product="CL",
        contract_month=12,
        contract_year=2026,
        symbol="CLZ26",
        multiplier=1000.0,
        tick_size=0.01,
        tick_value=10.0,
    )
    repo.save_contract(contract)

    api_df = make_ohlc_df("CLZ26", [datetime.now(timezone.utc)])
    mocker.patch.object(vendor_adapter, "_fetch_from_api", return_value={"CLZ26": api_df})

    result = vendor_adapter.run_morning_sync(repo)
    assert result == {"CLZ26": "ok"}


# ---------- backfill_symbol ----------


def test_backfill_symbol_splits_into_multiple_requests_when_rows_exceed_limit(
    vendor_adapter, mocker
):
    start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=25_000)  # comfortably > 10,000 rows

    def fake_fetch(api_symbols, interval, start=None, end=None, count=None):
        return {api_symbols[0]: make_ohlc_df("CLZ26", [start])}

    fetch_mock = mocker.patch.object(vendor_adapter, "_fetch_from_api", side_effect=fake_fetch)

    vendor_adapter.backfill_symbol("CLZ26", start, end)

    assert fetch_mock.call_count == 3  # 25001 days / 10000 rows per request, rounded up


def test_backfill_symbol_raises_when_start_after_end(vendor_adapter):
    start = datetime(2026, 1, 2, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        vendor_adapter.backfill_symbol("CLZ26", start, end)
