"""ATR: the true-range maths, the OHLC download, the engine's tables and the ATR tab."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from dash import no_update

from core import atr_data, oi_data
from core.atr_calc import atr_frame, true_range
from core.atr_data import AtrStore, instrument_universe
from core.atr_engine import DTE_MARKS, AtrEngine, AtrParams, load_params, save_params, validate_params
from core.oi_service import OiService
from core.oi_settings import DEFAULT_ATR_HISTORY_DAYS, KEY_ATR_HISTORY_DAYS, load_atr_history_days, validate_atr_history_days
from ui.callbacks import atr_callbacks as acc
from ui.container import Container
from ui.layouts import atr_view as view

TODAY = date(2026, 10, 13)
NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
HISTORY = 400


def ohlc(rows) -> pd.DataFrame:
    """rows: (high, low, close, volume) per consecutive day."""
    index = pd.date_range("2026-01-05", periods=len(rows), freq="B")
    return pd.DataFrame(rows, columns=["high", "low", "close", "volume"], index=index).assign(open=lambda f: f["close"])


# ----------------------------------------------------------------------
# The maths
# ----------------------------------------------------------------------


def test_true_range_is_the_largest_of_the_range_and_the_two_gaps():
    frame = ohlc([(101, 99, 100, 10), (103, 102, 102.5, 10), (101, 98, 99, 10), (99.4, 98.9, 99.2, 10)])
    tr = true_range(frame)
    assert tr.iloc[0] == pytest.approx(2.0)  # first day: no previous close, just high - low
    assert tr.iloc[1] == pytest.approx(3.0)  # gapped up: |103 - 100| beats the 1.0 range
    assert tr.iloc[2] == pytest.approx(4.5)  # gapped down: |98 - 102.5|
    assert tr.iloc[3] == pytest.approx(0.5)  # an ordinary day: the range


def test_atr_is_the_simple_average_of_the_last_period_true_ranges():
    frame = ohlc([(101, 99, 100, 5), (103, 102, 102.5, 5), (101, 98, 99, 5), (99.4, 98.9, 99.2, 5), (102, 99, 101, 5)])
    out = atr_frame(frame, period=3, percent=True)
    assert out["atr"].isna().tolist() == [True, True, False, False, False]
    assert out["atr"].iloc[2] == pytest.approx((2.0 + 3.0 + 4.5) / 3)
    assert out["atr"].iloc[3] == pytest.approx((3.0 + 4.5 + 0.5) / 3)
    assert out["atr_pct"].iloc[2] == pytest.approx(out["atr"].iloc[2] / 99 * 100)


def test_days_without_trading_are_left_out_when_asked():
    frame = ohlc([(101, 99, 100, 5), (100, 100, 100, 0), (100, 100, 100, 0), (103, 101, 102, 7), (104, 102, 103, 9)])
    kept = true_range(frame, skip_idle=True)
    assert len(kept) == 3 and kept.iloc[1] == pytest.approx(3.0)  # the next real day's range includes the gap from the last real close
    assert len(true_range(frame, skip_idle=False)) == 5
    zero_days = atr_frame(frame, period=3, skip_idle=False)["atr"]
    assert atr_frame(frame, period=3, skip_idle=True)["atr"].iloc[-1] > zero_days.iloc[-1]  # idle zero ranges no longer drag it down


def test_percent_of_price_is_only_for_outrights():
    frame = ohlc([(1, 0.5, 0.8, 5)] * 5)
    assert atr_frame(frame, period=2, percent=False)["atr_pct"].isna().all()
    assert atr_frame(frame, period=2, percent=True)["atr_pct"].notna().any()


# ----------------------------------------------------------------------
# The instruments and the download
# ----------------------------------------------------------------------


@pytest.fixture
def small_universe(monkeypatch):
    monkeypatch.setattr(oi_data, "FIRST_CONTRACT_YEAR", 2025)
    monkeypatch.setattr(oi_data, "MONTHS_AHEAD", 4)


def test_instruments_are_outrights_consecutive_spreads_and_consecutive_flies():
    cl = instrument_universe("CL", TODAY)
    by = {f: [i for i in cl if i.family == f] for f in ("outright", "spread", "fly")}
    assert len(by["outright"]) == 130 and len(by["spread"]) == 129 and len(by["fly"]) == 128
    assert by["outright"][0].symbol == "CLF19" and by["spread"][0].symbol == "CLF19-G19" and by["fly"][0].symbol == "CLF19-G19-H19"
    assert by["fly"][0].label == "Jan19/Feb19/Mar19" and by["spread"][0].first.symbol == "CLF19" and by["spread"][0].expiry == by["outright"][0].expiry
    assert instrument_universe("BRN", TODAY)[0].symbol == "COF19"
    assert atr_data.family_of("CLZ26") == "outright" and atr_data.family_of("CLZ26-F27") == "spread" and atr_data.family_of("CLZ26-F27-G27") == "fly"


class FakeOhlcApi:
    """Answers /ohlc/ like the real one: every candle in [start, end] of each instrument, or an empty SUCCESS."""

    def __init__(self, today: date = TODAY, fail_after: int | None = None):
        self.today, self.calls, self.fail_after = today, [], fail_after

    def ohlc(self, interval, instruments=None, start=None, end=None, **kwargs):
        self.calls.append({"instruments": list(instruments), "start": start, "end": end, "interval": interval})
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("boom")
        lookup = {i.symbol: i for p in ("CL", "BRN") for i in instrument_universe(p, self.today)}
        candles = []
        for symbol in instruments:
            expiry = pd.Timestamp(lookup[symbol].expiry)
            days = pd.bdate_range(expiry - pd.Timedelta(days=500), min(expiry, pd.Timestamp(self.today) - pd.Timedelta(days=1)))
            for d in days:
                t = int(d.replace(tzinfo=timezone.utc).timestamp())
                if start <= t <= end:
                    candles.append({"product": symbol, "time": t * 1000, "open": 70.0, "high": 71.0, "low": 69.0, "close": 70.5, "volume": 100})
        return candles if candles else {"status": "SUCCESS", "data": []}


def test_download_batches_by_family_with_unix_dates_and_saves(tmp_path, small_universe):
    api = FakeOhlcApi()
    store = AtrStore(tmp_path, clock=lambda: NOW)
    report = store.update(api, HISTORY, TODAY)
    assert report["requests"] == len(api.calls) > 3 and report["rows"] > 1000
    assert all(c["interval"] == "1D" and isinstance(c["start"], int) and isinstance(c["end"], int) and len(c["instruments"]) <= 50 for c in api.calls)
    assert all({atr_data.family_of(s) for s in c["instruments"]} <= {"outright", "spread", "fly"} and len({atr_data.family_of(s) for s in c["instruments"]}) == 1 for c in api.calls)
    frame = store.frame("CL")
    assert set(frame.columns) == set(atr_data.COLUMNS) and frame.duplicated(["symbol", "date"]).sum() == 0
    assert store.updated_on == "2026-10-13" and store.loaded_history_days == HISTORY and store.has_data()
    clf25 = next(i for i in instrument_universe("CL", TODAY) if i.symbol == "CLF25")
    first = frame[frame["symbol"] == "CLF25"]["date"].min()
    assert (pd.Timestamp(clf25.expiry) - first).days <= HISTORY + 5  # nothing before the requested history


def test_an_empty_reply_is_not_an_error_and_unknown_symbols_are_ignored(tmp_path, small_universe):
    class Empty(FakeOhlcApi):
        def ohlc(self, *a, **k):
            super().ohlc(*a, **k)
            return {"status": "SUCCESS", "data": []}

    store = AtrStore(tmp_path, clock=lambda: NOW)
    store.update(Empty(), HISTORY, TODAY)
    assert store.frame("CL").empty and store.updated_on == "2026-10-13"
    with pytest.raises(Exception):
        AtrStore._fetch(type("A", (), {"ohlc": lambda self, *a, **k: {"status": "ERROR"}})(), ["CLZ26"], TODAY, TODAY)


def test_finished_instruments_are_not_requested_again_and_live_ones_are_topped_up(tmp_path, small_universe):
    api = FakeOhlcApi()
    store = AtrStore(tmp_path, clock=lambda: NOW)
    store.update(api, HISTORY, TODAY)
    api.calls.clear()
    tomorrow = TODAY + timedelta(days=1)
    store.update(api, HISTORY, tomorrow)
    asked = {s for c in api.calls for s in c["instruments"]}
    live = {i.symbol for p in ("CL", "BRN") for i in instrument_universe(p, tomorrow) if i.expiry >= tomorrow - timedelta(days=3)}
    assert asked and asked <= live


def test_a_longer_history_is_downloaded_and_a_failure_resumes(tmp_path, small_universe, monkeypatch):
    api = FakeOhlcApi()
    store = AtrStore(tmp_path, clock=lambda: NOW)
    store.update(api, 120, TODAY)
    before = store.frame("CL").groupby("symbol")["date"].min()
    api.calls.clear()
    store.update(api, 400, TODAY)
    after = store.frame("CL").groupby("symbol")["date"].min()
    assert api.calls and (after <= before).all() and (after < before).any()
    # a failure part-way keeps what was saved
    fresh = AtrStore(tmp_path / "other", clock=lambda: NOW)
    flaky = FakeOhlcApi(fail_after=2)
    with pytest.raises(RuntimeError):
        fresh.update(flaky, HISTORY, TODAY)
    assert fresh._meta["instruments"] and fresh.updated_on is None
    again = FakeOhlcApi()
    fresh.update(again, HISTORY, TODAY)
    assert fresh.updated_on == "2026-10-13" and len(again.calls) < len(flaky.calls) + 30


# ----------------------------------------------------------------------
# The engine, on a synthetic market whose ATR rises towards expiry
# ----------------------------------------------------------------------


def synthetic_ohlc(instrument, market: pd.Series, spike_days: int = 0, spike: float = 4.0) -> pd.DataFrame:
    expiry = pd.Timestamp(instrument.expiry)
    end = min(expiry, pd.Timestamp(TODAY) - pd.Timedelta(days=1))
    days = pd.bdate_range(expiry - pd.Timedelta(days=700), end)
    dte = (expiry - days).days.to_numpy()
    amplitude = (1.0 + 2.0 * np.exp(-dte / 150.0)) * (0.5 if instrument.family != "outright" else 1.0)
    if spike_days:
        amplitude[-spike_days:] *= spike
    close = market.reindex(days).ffill().bfill().to_numpy() + instrument.first.order * 0.05
    if instrument.family != "outright":
        close = 0.3 + 0.02 * (market.reindex(days).ffill().bfill().to_numpy() - 70)
    return pd.DataFrame({
        "symbol": instrument.symbol, "date": days, "open": close, "high": close + amplitude / 2, "low": close - amplitude / 2,
        "close": close, "volume": 100.0,
    })


@pytest.fixture(scope="module")
def market():
    days = pd.bdate_range("2017-01-02", "2026-10-12")
    return pd.Series(70 + np.cumsum(np.random.default_rng(7).normal(0, 0.3, len(days))), index=days)


@pytest.fixture(scope="module")
def frames(market):
    out = {}
    for product in ("CL",):
        parts = [synthetic_ohlc(i, market) for i in instrument_universe(product, TODAY) if i.expiry >= date(2022, 1, 1)]
        out[product] = pd.concat(parts, ignore_index=True)
    return out


LIVE = "CLF27"  # Jan27, about 70 days out; four Januaries behind it


def engine(frame, **overrides) -> AtrEngine:
    return AtrEngine(frame, "CL", TODAY, HISTORY, AtrParams(**overrides))


def spiked_frame(frames, market, days=5):
    parts = []
    for i in instrument_universe("CL", TODAY):
        if i.expiry < date(2022, 1, 1):
            continue
        parts.append(synthetic_ohlc(i, market, spike_days=days) if i.symbol == LIVE else synthetic_ohlc(i, market))
    return pd.concat(parts, ignore_index=True)


def test_params_validate_and_round_trip(tmp_path):
    from db.repository import Repository

    assert (AtrParams().period, AtrParams().contracts, AtrParams().years) == (14, 15, 5)
    assert validate_params({"period": "10", "years": 3}).period == 10 and validate_params({}) == AtrParams()
    for bad in ({"period": 1}, {"period": 500}, {"contracts": 0}, {"years": 1}, {"z_threshold": 0}):
        with pytest.raises(ValueError):
            validate_params(bad)
    repo = Repository(str(tmp_path / "t.db"))
    assert load_params(repo) == AtrParams()
    save_params(repo, AtrParams(period=20, skip_idle=False))
    assert load_params(repo) == AtrParams(period=20, skip_idle=False)


def test_the_live_table_compares_with_the_same_month_only(frames):
    eng = engine(frames["CL"])
    table = eng.live_table("outright")
    assert len(table) == 15 and table["label"].iloc[0] == "Nov26" and table["dte"].is_monotonic_increasing
    assert (table["atr"] > 0).all() and table["atr_pct"].notna().all()
    assert (table["status"] == "").all() and (table["vs_front"].iloc[0] == 1.0) and (table["vs_front"] < 1.0 + 1e-9).all()  # later contracts move less
    row = table[table["symbol"] == LIVE].iloc[0]
    assert row["usual_n"] == 4 and np.isfinite(row["z"]) and abs(row["z"]) < 2.5
    months = {eng.universe[s].first.month for s in eng.same_month(LIVE)}
    assert months == {1} and len(eng.same_month(LIVE)) == 4
    assert eng.live_table("spread")["atr_pct"].isna().all()  # no percent of price for a spread


def test_an_unusually_wide_recent_range_is_flagged_high(frames, market):
    table = engine(spiked_frame(frames, market)).live_table("outright")
    row = table[table["symbol"] == LIVE].iloc[0]
    assert row["status"] == "HIGH" and row["z"] > 2.5 and row["tr_vs_atr"] > 1
    assert (table[table["symbol"] != LIVE]["status"] == "").all()
    assert (engine(spiked_frame(frames, market), z_threshold=50.0).live_table("outright")["status"] == "").all()


def test_atr_rises_as_a_contract_approaches_expiry(frames):
    table = engine(frames["CL"]).month_table("outright", "atr")
    assert table["month"].tolist() == list(range(1, 13))
    jan = table.set_index("month").loc[1]
    assert jan["median_12m"] < jan["median_6m"] < jan["median_3m"] < jan["median_1m"] <= jan["median_1w"] + 1e-9
    assert {f"sd_{label}" for _, label in DTE_MARKS} <= set(table.columns)
    relative = engine(frames["CL"]).month_table("outright", "atr", relative=True).set_index("month").loc[1]
    assert relative["median_1m"] == pytest.approx(1.0) and relative["median_9m"] < 0.95  # a smaller ATR 9 months out than at the front
    pct = engine(frames["CL"]).month_table("outright", "pct")
    assert pct.set_index("month").loc[1, "median_3m"] != jan["median_3m"]


def test_the_cycle_chart_has_the_same_months_past_years_and_the_live_path(frames):
    data = engine(frames["CL"]).chart(LIVE)
    assert data["label"] == "Jan27" and data["unit"] == "points" and len(data["dte"]) == len(data["atr"]) > 100
    usual = data["usual"]
    assert len(usual["labels"]) == 4 and all(label.startswith("Jan") for label in usual["labels"]) and len(usual["lines"]) == 4
    assert (usual["low"] <= usual["median"] + 1e-9).all()
    assert engine(frames["CL"]).chart(LIVE, "pct")["unit"] == "% of price" and engine(frames["CL"]).chart("CLZ99") is None
    figure = view.build_atr_chart(data)
    assert figure.layout.uirevision == LIVE and any("(live)" in (t.name or "") for t in figure.data) and len(view.build_atr_chart(None).data) == 0


def test_hedge_table_compares_later_contracts_with_the_front_for_one_front_month(frames):
    table = engine(frames["CL"]).hedge_table(11)
    assert table["position"].tolist() == list(range(2, 16)) and table["label"].iloc[0] == "Dec26"
    assert table["days"].min() > 20 and (table["median"] < 1.0).all() and table["median"].is_monotonic_decreasing
    assert (table["p25"] <= table["median"]).all() and (table["median"] <= table["p75"]).all() and (table["min"] <= table["p25"]).all()
    assert table["now"].notna().all() and (table["now"].iloc[0] <= 1.0 + 1e-9)
    assert table["hedge"].between(0.2, 1.5).all() and table["corr"].between(0.5, 1.0).all()  # the synthetic contracts share one market
    other = engine(frames["CL"]).hedge_table(3)
    assert other["median"].iloc[0] != table["median"].iloc[0]  # a different front month has its own numbers


def test_idle_days_setting_changes_the_atr(frames):
    idle = frames["CL"].copy()
    mask = (idle["symbol"] == LIVE) & (idle["date"] >= idle["date"].max() - pd.Timedelta(days=12))
    for column in ("open", "high", "low"):
        idle.loc[mask, column] = idle.loc[mask, "close"]  # flat candles ...
    idle.loc[mask, "volume"] = 0.0  # ... with no volume
    with_skip = engine(idle, skip_idle=True).live_table("outright").set_index("symbol").loc[LIVE, "atr"]
    without = engine(idle, skip_idle=False).live_table("outright").set_index("symbol").loc[LIVE, "atr"]
    assert with_skip > without


def test_an_empty_frame_is_handled():
    empty = AtrEngine(pd.DataFrame(columns=atr_data.COLUMNS), "CL", TODAY, HISTORY, AtrParams())
    assert empty.hedge_table(1).empty and empty.chart("CLZ26") is None
    assert "No ATR data" in str(view.atr_body(empty, "outright", "atr", 1, AtrParams()))


# ----------------------------------------------------------------------
# Settings, the service and the tab
# ----------------------------------------------------------------------


def test_atr_history_setting_defaults_to_400_and_is_validated(tmp_path):
    from db.repository import Repository

    assert DEFAULT_ATR_HISTORY_DAYS == 400 and validate_atr_history_days("500") == 500
    for bad in (10, 5000, 400.5, None, "x"):
        with pytest.raises(ValueError):
            validate_atr_history_days(bad)
    repo = Repository(str(tmp_path / "t.db"))
    assert load_atr_history_days(repo) == 400
    repo.set_setting(KEY_ATR_HISTORY_DAYS, 600)
    assert load_atr_history_days(repo) == 600
    repo.set_setting(KEY_ATR_HISTORY_DAYS, "junk")
    assert load_atr_history_days(repo) == 400


def test_settings_page_saves_both_history_lengths(tmp_path, monkeypatch):
    from db.repository import Repository
    from ui.callbacks import settings_callbacks as sc

    repo = Repository(str(tmp_path / "t.db"))
    monkeypatch.setattr(sc, "container", Container(repository=repo))
    assert sc.load_oi_settings("/settings") == (1500, 400)
    assert "ATR" in sc.save_oi_settings(1, 1200, 500) and repo.get_setting("atr_history_days") == 500 and repo.get_setting("oi_history_days") == 1200
    assert "❌" in sc.save_oi_settings(1, 1200, 5) and repo.get_setting("atr_history_days") == 500
    assert "settings-atr-history-days" in str(__import__("ui.layouts.settings", fromlist=["x"]).settings_layout())


class StubAtrStore:
    def __init__(self, frames, updated_on=TODAY.isoformat()):
        self.frames, self.version, self.updated_on, self.loaded_history_days = frames, "a1", updated_on, HISTORY
        self.progress = None

    def frame(self, product):
        return self.frames.get(product, pd.DataFrame(columns=atr_data.COLUMNS))


@pytest.fixture
def service(tmp_path, frames):
    from db.repository import Repository
    repo = Repository(str(tmp_path / "svc.db"))
    repo.set_setting("oi_history_days", 900)
    repo.set_setting("atr_history_days", HISTORY)

    class OiStub:
        def __init__(self):
            self.version, self.updated_on, self.loaded_history_days = "v1", TODAY.isoformat(), 900
            self.progress = type("P", (), {"running": False, "done": 0, "total": 0, "message": "", "error": None, "finished_at": None})()

        def frame(self, product):
            return pd.DataFrame(columns=["contract", "date", "oi", "close", "volume"])

        def has_data(self):
            return True

    atr = StubAtrStore(frames)
    svc = OiService(repo, None, OiStub(), clock=lambda: NOW, background=False, atr_store=atr)
    svc.atr = atr
    return svc


def test_the_service_serves_cached_atr_engines_and_goes_stale_with_the_atr_data(service):
    params = AtrParams()
    first = service.atr_engine("CL", params)
    assert service.atr_engine("CL", params) is first and service.atr_engine("CL", AtrParams(period=10)) is not first
    service.atr.version = "a2"
    assert service.atr_engine("CL", params) is not first
    assert not service.needs_update()
    service.atr.updated_on = "2026-10-12"
    assert service.needs_update()  # ATR data from yesterday: due, even though the OI data is fresh
    service.atr.updated_on, service.atr.loaded_history_days = TODAY.isoformat(), 100
    assert service.needs_update()
    assert service.status()["version"].endswith("|a2")
    service.save_atr_params(AtrParams(contracts=9))
    assert service.atr_params().contracts == 9


def test_the_tab_builds_only_when_open_and_shows_the_tables(service, monkeypatch):
    monkeypatch.setattr(acc, "container", Container(oi_service=service))
    values = [14, 15, 5, 2.5]
    assert acc.render_atr("v", "CL", "outright", "atr", 11, *values, True, "monitor", None) == (no_update,) * 4  # not while another tab shows
    body, options, value, params = acc.render_atr("v", "CL", "outright", "atr", 11, *values, True, "atr", None)
    text = str(body)
    for component_id in ("atr-live-table", "atr-month-table", "atr-front-multiple-table", "atr-hedge-table"):
        assert component_id in text
    assert len(options) == 15 and value == options[0]["value"] and params["period"] == 14
    assert service.atr_params().period == 14  # remembered
    spread_body = acc.render_atr("v", "CL", "spread", "atr", 11, *values, True, "atr", None)[0]
    assert "atr-hedge-table" not in str(spread_body) and "covers outrights" in str(spread_body)
    assert "% of price" in str(acc.render_atr("v", "CL", "outright", "pct", 11, *values, True, "atr", None)[0])
    bad = acc.render_atr("v", "CL", "outright", "atr", 11, 1, 15, 5, 2.5, True, "atr", None)
    assert "ATR period" in bad[0].children and bad[1] is no_update
    assert acc.render_atr(None, "CL", "outright", "atr", 11, *values, True, "atr", None) == (no_update,) * 4
    assert acc.render_atr_chart(value, params, "atr").layout.uirevision == value
    assert "Pick a contract" in acc.render_atr_chart(None, None, "atr").layout.annotations[0].text
    assert acc.product_of("COZ26-F27") == "BRN" and acc.product_of("CLZ26") == "CL" and acc.product_of(None) is None


def test_a_flagged_contract_is_picked_first_and_labelled(service, monkeypatch, market):
    monkeypatch.setattr(acc, "container", Container(oi_service=service))
    service.atr.frames = {"CL": spiked_frame(None, market)}
    service.atr.version = "spiked"
    _, options, value, _ = acc.render_atr("v", "CL", "outright", "atr", 11, 14, 15, 5, 2.5, True, "atr", None)
    assert value == LIVE and "HIGH" in next(o["label"] for o in options if o["value"] == LIVE)
    assert acc.render_atr("v", "CL", "outright", "atr", 11, 14, 15, 5, 2.5, True, "atr", "CLH27")[2] == "CLH27"  # the user's choice is kept


def test_the_layout_has_the_atr_controls_and_the_callbacks_are_wired(service, monkeypatch):
    import dash

    monkeypatch.setattr(view, "container", Container(oi_service=service))
    text = str(view.atr_tab_content())
    for component_id in ("atr-family", "atr-unit", "atr-month", "atr-p-period", "atr-p-contracts", "atr-p-years", "atr-p-z", "atr-p-idle", "atr-wrap", "atr-contract", "atr-chart"):
        assert component_id in text
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    acc.register_atr_callbacks(app)
    wired = " ".join(app.callback_map)
    assert "atr-wrap" in wired and "atr-chart" in wired
    main = next(v for k, v in app.callback_map.items() if "atr-wrap" in k)
    assert {"oi-product", "oi-tabs", "atr-family", "atr-month"} <= {i["id"] for i in main["inputs"]}
