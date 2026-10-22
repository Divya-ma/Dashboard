"""MFE / MAE from intraday candles as well as daily closes, only when the API has requests to spare."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from adapters.qh_api.rate_limiter import EndpointRateLimiter, RateLimit
from core.journal import excursion
from core.journal_service import JournalService
from core.journal_snapshot import intraday_interval, intraday_price_path
from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType
from db.repository import Repository

T0 = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)


def contract(symbol, month):
    return Contract(product="CL", contract_month=month, contract_year=2026, symbol=symbol, multiplier=1000.0, tick_size=0.01, tick_value=10.0)


def spread(status=StructureStatus.CLOSED):
    legs = [Leg(contract=contract("CLX26", 11), ratio=1, lots=0.0, entry_price=75.0, average_entry_price=75.0, direction="buy"),
            Leg(contract=contract("CLZ26", 12), ratio=-1, lots=0.0, entry_price=74.0, average_entry_price=74.0, direction="buy")]
    return Structure(name="spr", structure_type=StructureType.SPREAD, products=["CL"], legs=legs, status=status)


def trades_for(s, hours=3, lots=2):
    return [Trade(structure_id=s.structure_id, leg_id=s.legs[0].leg_id, event_type=TradeEventType.TRADE, lots=lots, price=1.0, direction="buy", timestamp=T0),
            Trade(structure_id=s.structure_id, leg_id=s.legs[0].leg_id, event_type=TradeEventType.FULL_EXIT, lots=lots, price=1.2, direction="sell", timestamp=T0 + timedelta(hours=hours))]


# ---------- the rate limiter's spare room ----------


def test_spare_is_the_least_room_in_the_minute_hour_and_day_windows():
    now = [0.0]
    limiter = EndpointRateLimiter(RateLimit(5, 100, 1000), clock=lambda: now[0], sleep=lambda s: None)
    assert limiter.spare() == 5
    for _ in range(3):
        limiter.acquire()
    assert limiter.spare() == 2
    now[0] = 61.0  # the minute window has emptied
    assert limiter.spare() == 5
    limiter.block_for(30)
    assert limiter.spare() == 0  # told to back off


def test_the_api_reports_the_spare_requests_of_an_endpoint():
    from adapters.qh_api import make_api

    api = make_api(lambda: "token")
    assert api.spare_requests("ohlc") == 30


# ---------- marking the path ----------


def test_an_intraday_path_sees_the_spike_a_daily_close_misses():
    s = spread()
    trades = trades_for(s, hours=3)
    minutes = pd.date_range(T0, periods=181, freq="1min")
    path = pd.Series(1.0, index=minutes)
    path.iloc[40] = 1.5  # a spike 40 minutes in
    path.iloc[100] = 0.7  # and a dip
    path.iloc[-1] = 1.2
    intraday = excursion(s, trades, path, intraday=True)
    assert intraday["mfe"] == pytest.approx(0.5 * 2 * 1000) and intraday["mae"] == pytest.approx(-0.3 * 2 * 1000) and "intraday" in intraday["basis"]
    daily = excursion(s, trades, pd.Series([1.2], index=pd.date_range("2026-10-20", periods=1, tz="UTC")))
    assert daily["mfe"] == pytest.approx(0.2 * 2 * 1000) and daily["mae"] == 0.0 and daily["days"] == 1  # the day closed at 1.2: nothing else seen


def test_an_intraday_window_is_cut_to_the_trade():
    s = spread()
    trades = trades_for(s, hours=1)
    path = pd.Series([2.0, 1.1, 1.0, 0.9, 0.0], index=[T0 - timedelta(minutes=30), T0 + timedelta(minutes=10), T0 + timedelta(minutes=20), T0 + timedelta(minutes=30), T0 + timedelta(hours=2)])
    result = excursion(s, trades, path, intraday=True)
    assert result["mfe"] == pytest.approx(0.1 * 2 * 1000) and result["mae"] == pytest.approx(-0.1 * 2 * 1000) and result["points"] == 3
    assert excursion(s, trades, pd.Series([1.0], index=[T0 + timedelta(days=3)]), intraday=True) is None


def test_the_candle_size_follows_the_length_of_the_trade():
    assert intraday_interval(timedelta(minutes=20)) == "1M" and intraday_interval(timedelta(hours=30)) == "1M"
    assert intraday_interval(timedelta(days=3)) == "5M" and intraday_interval(timedelta(days=7)) == "5M"
    assert intraday_interval(timedelta(days=9)) == "1H"


class FakeOhlc:
    def __init__(self, with_z=True):
        self.calls, self.with_z = [], with_z

    def ohlc(self, interval, instruments=None, start=None, end=None, **kwargs):
        self.calls.append((interval, list(instruments), start, end))
        rows = []
        for symbol in instruments:
            if symbol == "CLZ26" and not self.with_z:
                continue
            base = 75.0 if symbol == "CLX26" else 74.0
            for i in range(0, 120, 5):
                stamp = T0 + timedelta(minutes=i)
                if symbol == "CLZ26" and i % 10:
                    continue  # the second leg trades only every ten minutes: the first leg's price is used between
                close = base + (0.5 if (symbol == "CLX26" and i == 60) else 0.0)
                rows.append({"product": symbol, "time": int(stamp.timestamp() * 1000), "open": close, "high": close, "low": close, "close": close, "volume": 1})
        return rows or {"status": "SUCCESS", "data": []}


def test_the_intraday_path_comes_from_one_request_and_keeps_a_quiet_legs_last_price():
    api = FakeOhlc()
    path = intraday_price_path(spread(), api, T0, T0 + timedelta(hours=2))
    assert len(api.calls) == 1 and api.calls[0][0] == "1M" and sorted(api.calls[0][1]) == ["CLX26", "CLZ26"]
    assert path.max() == pytest.approx(1.5) and path.min() == pytest.approx(1.0)  # X rose 0.5 at 60 minutes against a flat Z
    assert intraday_price_path(spread(), FakeOhlc(with_z=False), T0, T0 + timedelta(hours=2)) is None  # a leg with no candles


# ---------- the service decides when to use the API ----------


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


class NoSnapshots:
    def snapshot(self, structure, when):
        return {"at": when.isoformat(), "product": "CL", "prices": {}, "curve": {}, "atr": None}


def closed(repo):
    s = spread()
    for leg in s.legs:
        repo.save_contract(leg.contract)
    repo.save_structure(s)
    for t in trades_for(s):
        repo.save_trade(t.model_copy(update={"structure_id": s.structure_id, "leg_id": s.legs[0].leg_id}))
    return s


def service(repo, spare, api=None):
    api = api or FakeOhlc()
    daily = pd.Series([1.0, 1.1], index=pd.date_range("2026-10-19", periods=2, tz="UTC"))  # the day of the trade closed at 1.1
    svc = JournalService(
        repo, NoSnapshots(), clock=lambda: T0, background=False, price_path=lambda s: daily,
        intraday_path=lambda s, a, b: intraday_price_path(s, api, a, b), spare_requests=lambda: spare[0],
    )
    return svc, api


def test_with_spare_requests_both_measures_are_kept_and_the_intraday_one_leads(repo):
    s = closed(repo)
    spare = [30]
    svc, api = service(repo, spare)
    svc.record_excursion(s)
    exc = svc.journal(s.structure_id)["excursion"]
    assert exc["intraday"]["basis"].startswith("intraday") and exc["daily"]["basis"].startswith("daily") and len(api.calls) == 1
    assert exc["mfe"] == exc["intraday"]["mfe"] == pytest.approx(0.5 * 2 * 1000) and exc["daily"]["mfe"] == pytest.approx(0.1 * 2 * 1000)


def test_without_spare_requests_only_the_daily_measure_is_taken_and_the_backfill_adds_the_rest_later(repo):
    s = closed(repo)
    spare = [3]  # the live pages are using the budget
    svc, api = service(repo, spare)
    svc.record_excursion(s)
    exc = svc.journal(s.structure_id)["excursion"]
    assert api.calls == [] and exc["intraday"] is None and exc["daily"] is not None and exc["mfe"] == exc["daily"]["mfe"]
    svc.backfill()
    assert api.calls == []  # still no room: nothing fetched
    spare[0] = 25
    svc.backfill()
    exc = svc.journal(s.structure_id)["excursion"]
    assert len(api.calls) == 1 and exc["intraday"] is not None and exc["mfe"] == exc["intraday"]["mfe"]
    svc.backfill()
    assert len(api.calls) == 1  # done: not asked again


def test_an_api_failure_keeps_the_daily_figures(repo):
    s = closed(repo)

    class Down:
        def ohlc(self, *a, **k):
            raise RuntimeError("down")

    svc, _api = service(repo, [30], Down())
    svc.record_excursion(s)
    exc = svc.journal(s.structure_id)["excursion"]
    assert exc["daily"] is not None and exc["intraday"] is None
