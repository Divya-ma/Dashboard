"""Tests for adapters.mock.mock_historical.MockHistoricalAdapter and
adapters.historical.vendor.VendorHistoricalAdapter.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from adapters.historical.vendor import VendorHistoricalAdapter
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
    return VendorHistoricalAdapter(repo, data_dir=str(tmp_path / "historical"))


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


def test_symbol_with_error_status_today_is_retried(vendor_adapter, repo):
    from core.models import Contract

    repo.save_contract(
        Contract(
            product="CL", contract_month=12, contract_year=2026, symbol="CLZ26",
            multiplier=1000.0, tick_size=0.01, tick_value=10.0,
        )
    )
    vendor_adapter._update_sync_log("CLZ26", status="error", error_msg="API down")
    assert vendor_adapter.get_symbols_needing_sync(repo) == ["CLZ26"]

    vendor_adapter._update_sync_log("CLZ26", status="ok", row_count=1)
    assert vendor_adapter.get_symbols_needing_sync(repo) == []


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


# ---------- get_symbols_needing_sync: extra_symbols ----------


def test_get_symbols_needing_sync_includes_extra_symbols(vendor_adapter, repo):
    assert vendor_adapter.get_symbols_needing_sync(repo, extra_symbols=["CLZ26-F27"]) == ["CLZ26-F27"]
    vendor_adapter._update_sync_log("CLZ26-F27", status="ok", row_count=1)
    assert vendor_adapter.get_symbols_needing_sync(repo, extra_symbols=["CLZ26-F27"]) == []


# ---------- backfill_symbols (batched) ----------


def test_backfill_symbols_batches_into_one_api_call(vendor_adapter, mocker):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 5, tzinfo=timezone.utc)

    def fake_fetch(api_symbols, interval, start=None, end=None, count=None):
        return {sym: make_ohlc_df(sym, [start]) for sym in api_symbols}

    fetch_mock = mocker.patch.object(vendor_adapter, "_fetch_from_api", side_effect=fake_fetch)
    counts = vendor_adapter.backfill_symbols(["CLZ26", "CLF27", "CLG27"], start, end)

    fetch_mock.assert_called_once()
    assert counts == {"CLZ26": 1, "CLF27": 1, "CLG27": 1}
    assert vendor_adapter._load_local("CLZ26") is not None


def test_backfill_symbols_falls_back_per_symbol_when_range_too_long(vendor_adapter, mocker):
    start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=25_000)  # comfortably > 10,000 rows
    backfill_mock = mocker.patch.object(vendor_adapter, "backfill_symbol", return_value=7)

    counts = vendor_adapter.backfill_symbols(["CLZ26", "CLF27"], start, end)

    assert backfill_mock.call_count == 2
    assert counts == {"CLZ26": 7, "CLF27": 7}


def test_backfill_symbols_raises_when_start_after_end(vendor_adapter):
    start = datetime(2026, 1, 2, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        vendor_adapter.backfill_symbols(["CLZ26"], start, end)


def test_backfill_symbols_empty_list_is_noop(vendor_adapter, mocker):
    fetch_mock = mocker.patch.object(vendor_adapter, "_fetch_from_api")
    assert vendor_adapter.backfill_symbols([], datetime.now(timezone.utc)) == {}
    fetch_mock.assert_not_called()


# ---------- run_morning_sync: full backfill vs incremental split ----------


def test_run_morning_sync_full_backfills_symbols_with_no_local_history(vendor_adapter, repo, mocker):
    from core.models import Contract

    repo.save_contract(
        Contract(product="CL", contract_month=12, contract_year=2026, symbol="CLZ26",
                 multiplier=1000.0, tick_size=0.01, tick_value=10.0)
    )

    def fake_fetch(api_symbols, interval, start=None, end=None, count=None):
        assert count is None  # full backfill uses a date range, not "latest 1 candle"
        return {api_symbols[0]: make_ohlc_df(api_symbols[0], [start])}

    fetch_mock = mocker.patch.object(vendor_adapter, "_fetch_from_api", side_effect=fake_fetch)
    result = vendor_adapter.run_morning_sync(repo)

    fetch_mock.assert_called_once()
    assert result == {"CLZ26": "ok"}
    assert vendor_adapter._load_local("CLZ26") is not None


def test_run_morning_sync_mixes_full_backfill_and_incremental(vendor_adapter, repo, mocker):
    from core.models import Contract

    repo.save_contract(
        Contract(product="CL", contract_month=12, contract_year=2026, symbol="CLZ26",
                 multiplier=1000.0, tick_size=0.01, tick_value=10.0)
    )
    # CLF27 already has local history -> only needs the "latest candle" incremental path.
    # Written directly (not via _save_local, which would also mark today's sync log "ok"
    # and make get_symbols_needing_sync skip it outright).
    path = Path(vendor_adapter._data_dir) / "CL" / "CLF27.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    make_ohlc_df("CLF27", [datetime(2025, 1, 1, tzinfo=timezone.utc)]).to_parquet(path, index=False)

    calls = []

    def fake_fetch(api_symbols, interval, start=None, end=None, count=None):
        calls.append((tuple(api_symbols), count))
        return {sym: make_ohlc_df(sym, [datetime.now(timezone.utc)]) for sym in api_symbols}

    mocker.patch.object(vendor_adapter, "_fetch_from_api", side_effect=fake_fetch)
    result = vendor_adapter.run_morning_sync(repo, extra_symbols=["CLF27"])

    assert result == {"CLZ26": "ok", "CLF27": "ok"}
    assert len(calls) == 2  # one batched call for the full-backfill bucket, one for the incremental bucket
    full_backfill_call = next(c for c in calls if c[1] is None)
    incremental_call = next(c for c in calls if c[1] == 1)
    assert full_backfill_call[0] == ("CLZ26",)
    assert incremental_call[0] == ("CLF27",)


def test_run_morning_sync_extra_symbols_are_backfilled(vendor_adapter, repo, mocker):
    mocker.patch.object(
        vendor_adapter, "_fetch_from_api",
        side_effect=lambda api_symbols, interval, start=None, end=None, count=None: {
            s: make_ohlc_df(s, [datetime.now(timezone.utc)]) for s in api_symbols
        },
    )
    result = vendor_adapter.run_morning_sync(repo, extra_symbols=["CLZ26-F27"])
    assert result == {"CLZ26-F27": "ok"}


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


# ---------- get_sync_summary / get_local_data_summary ----------


def test_sync_summary_empty_before_any_sync(vendor_adapter):
    assert vendor_adapter.get_sync_summary() == {
        "last_sync_utc": None, "symbols_tracked": 0, "symbols_ok": 0, "symbols_error": 0,
    }


def test_sync_summary_counts_statuses_and_reports_latest_sync(vendor_adapter):
    vendor_adapter._update_sync_log("CLZ26", status="ok", row_count=5)
    vendor_adapter._update_sync_log("CLF27", status="error", error_msg="no data returned")
    summary = vendor_adapter.get_sync_summary()
    assert summary["symbols_tracked"] == 2
    assert summary["symbols_ok"] == 1
    assert summary["symbols_error"] == 1
    assert summary["last_sync_utc"].tzinfo is not None
    assert (datetime.now(timezone.utc) - summary["last_sync_utc"]).total_seconds() < 60


def test_local_data_summary_counts_symbol_files_but_not_the_sync_log(vendor_adapter):
    assert vendor_adapter.get_local_data_summary() == {"symbol_count": 0, "total_bytes": 0}

    vendor_adapter._save_local("CLZ26", make_ohlc_df("CLZ26", [datetime(2026, 1, 1, tzinfo=timezone.utc)]))
    vendor_adapter._save_local("COZ26", make_ohlc_df("BRNZ26", [datetime(2026, 1, 1, tzinfo=timezone.utc)]))

    summary = vendor_adapter.get_local_data_summary()  # sync_log.parquet exists now but must not count
    assert summary["symbol_count"] == 2
    assert summary["total_bytes"] > 0


# ---------- API transport (goes through the shared QH API client) ----------


def _candle(product="CLZ26", ts_ms=1_700_000_000_000, close=75.0):
    return {"product": product, "time": ts_ms, "open": 74.0, "high": 76.0, "low": 73.0, "close": close, "volume": 10}


def test_fetch_from_api_calls_ohlc_with_translated_params_and_parses_candles(vendor_adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    request = mocker.patch(
        "adapters.qh_api.client.requests.Session.request",
        return_value=mocker.Mock(headers={}, status_code=200, json=lambda: [_candle("COZ26")]),
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = vendor_adapter._fetch_from_api(["COZ26"], "1D", start=start, count=5)

    assert request.call_args.args[1].endswith("/apis/ohlc/")
    assert request.call_args.kwargs["params"] == {
        "instruments": "COZ26", "interval": "1D", "start": str(int(start.timestamp())), "count": "5",
    }
    assert request.call_args.kwargs["headers"]["Authorization"] == "Bearer tok"
    assert list(result["COZ26"].columns) == _OHLC_COLUMNS
    assert result["COZ26"].iloc[0]["symbol"] == "BRNZ26"  # API code translated back to internal


def test_fetch_from_api_batches_at_50_instruments_per_request(vendor_adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    request = mocker.patch(
        "adapters.qh_api.client.requests.Session.request",
        return_value=mocker.Mock(headers={}, status_code=200, json=lambda: []),
    )
    symbols = [f"CLZ{26}"] + [f"CL{m}{y:02d}" for m in "FGHJKMNQUVXZ" for y in (26, 27, 28, 29, 30)]
    assert len(symbols) > 50
    vendor_adapter._fetch_from_api(symbols, "1D", count=1)
    assert request.call_count == 2


def test_fetch_from_api_without_token_raises_runtime_error(vendor_adapter):
    with pytest.raises(RuntimeError):
        vendor_adapter._fetch_from_api(["CLZ26"], "1D", count=1)


def test_adapters_built_with_one_api_share_each_endpoints_limiter(repo, tmp_path):
    from adapters.live.fairvalue import FairValueLiveAdapter
    from adapters.qh_api import ENDPOINTS, make_api

    api = make_api(lambda: "tok")
    live = FairValueLiveAdapter(repo, 10.0, api=api)
    hist = VendorHistoricalAdapter(repo, data_dir=str(tmp_path / "h"), api=api)
    assert live._api is hist._api is api
    assert api.client.limiter_for(ENDPOINTS["ohlc"]) is api.client.limiter_for(ENDPOINTS["ohlc"])
