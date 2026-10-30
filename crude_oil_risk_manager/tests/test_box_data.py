"""CL-BZ boxes: candles, settlements, costs, Curve Compare and the stores, all built from the legs when the API has no box series."""

from datetime import date

import pandas as pd
import pytest

from core.atr_data import AtrStore, instrument_universe
from core.box_candles import derive_box_candles, fill_missing_boxes, leg_symbols
from core.box_settlements import fill_missing_box_rows
from core.churn_costs import DEFAULT_COMBO_REBATES, DEFAULT_COSTS, breakeven_ticks, round_trip_cost, tick_of
from core.curve_calendar import OUTRIGHT, SPREAD
from core.curve_compare import SOURCE_DERIVED, build_curve, build_price_book, settlement_curve, symbol_tiers
from core.oi_data import OiStore
from core.settlement_prices import SettlementFirstLoader, SettlementPrices

TODAY = date(2026, 10, 29)
DAY_MS = 86_400_000


def candle(product, day, close, volume=10):
    return {"product": product, "time": (day + 1) * DAY_MS, "open": close, "high": close, "low": close, "close": close, "volume": volume}


def test_leg_symbols_of_a_box():
    assert leg_symbols("WTCLCOF27") == ("CLF27", "BZZF27")
    assert leg_symbols("CLBZZXF27-G27-H27") == ("CLF27-G27-H27", "BZZF27-G27-H27")
    assert leg_symbols("WTCLCOF27-G27") is None  # CL Jan - BZ Feb is two different outrights
    assert leg_symbols("CLF27") is None and leg_symbols("COF27") is None


def test_box_candles_are_the_cl_candles_minus_the_bz_candles():
    cl = [candle("CLF27", d, 70.0 + d) for d in range(5)]
    bz = [candle("BZZF27", d, 72.0 + 0.5 * d, volume=4) for d in range(5)]
    got = fill_missing_boxes(lambda symbols: cl + bz, ["WTCLCOF27"], [], daily=True)
    assert [c["close"] for c in got] == pytest.approx([-2.0, -1.5, -1.0, -0.5, 0.0])
    assert all(c["product"] == "WTCLCOF27" and c["volume"] == 4 for c in got)  # the thinner leg's volume
    assert got[1]["open"] == -2.0 and got[1]["high"] == -1.5 and got[1]["low"] == -2.0  # close to close


def test_a_box_the_api_did_return_is_left_alone_and_nothing_extra_is_asked_for():
    asked = []
    own = [candle("WTCLCOF27", 0, -2.0)]
    assert fill_missing_boxes(lambda symbols: asked.append(symbols) or [], ["WTCLCOF27"], own, daily=True) == []
    assert asked == []


def test_intraday_box_candles_carry_the_last_leg_close_forward():
    cl = pd.DataFrame({"close": [70.0, 71.0], "volume": [5, 5]}, index=[1000, 3000])
    bz = pd.DataFrame({"close": [72.0, 72.5], "volume": [3, 3]}, index=[1000, 2000])
    got = derive_box_candles("WTCLCOF27", cl, bz, daily=False)
    assert [c["time"] for c in got] == [1000, 2000, 3000]
    assert [c["close"] for c in got] == pytest.approx([70.0 - 72.0, 70.0 - 72.5, 71.0 - 72.5])  # CL 70 carried to t=2000, BZ 72.5 to t=3000


def test_box_settlement_rows_are_built_from_the_leg_rows():
    rows = [
        {"qhcode": "CLF27", "datetime": "2026-10-27T00:00:00", "close": 70.0, "volume": 100, "oi": 5},
        {"qhcode": "BZZF27", "datetime": "2026-10-27T00:00:00", "close": 72.0, "volume": 60, "oi": 5},
        {"qhcode": "CLF27", "datetime": "2026-10-28T00:00:00", "close": 71.0, "volume": 100, "oi": 5},
    ]
    got = fill_missing_box_rows(lambda codes: rows, ["WTCLCOF27", "CLF27"], [])
    assert got == [{"qhcode": "WTCLCOF27", "datetime": "2026-10-27T00:00:00", "oi": None, "close": -2.0, "volume": 60.0}]  # only the day both settled
    assert fill_missing_box_rows(lambda codes: rows, ["WTCLCOF27"], [{"qhcode": "WTCLCOF27"}]) == []


# ----------------------------------------------------------------------
# Stores
# ----------------------------------------------------------------------


def test_the_atr_universe_names_box_spreads_with_the_spreadbox_code():
    universe = instrument_universe("CLBZ", TODAY, first_year=2027, months_ahead=6)
    symbols = {i.family: i.symbol for i in universe if i.label.startswith("Jan27")}
    assert symbols["outright"] == "WTCLCOF27" and symbols["spread"] == "CLBZZXF27-G27" and symbols["fly"] == "CLBZZXF27-G27-H27"
    assert [i.symbol for i in instrument_universe("CL", TODAY, 2027, 6) if i.family == "spread"][0] == "CLF27-G27"


class OhlcApi:
    """/ohlc/ answering CL and BZ outrights only (no box series)."""

    def __init__(self):
        self.requests: list[list[str]] = []

    def ohlc(self, interval, instruments=None, start=None, end=None, **_):
        self.requests.append(list(instruments))
        out = []
        for symbol in instruments:
            if symbol in ("CLF27", "BZZF27"):
                base = 70.0 if symbol.startswith("CL") else 72.0
                out += [{"product": symbol, "time": int(pd.Timestamp(2026, 10, d).timestamp() * 1000), "open": base, "high": base, "low": base, "close": base + d / 100, "volume": 9}
                        for d in range(19, 24)]
        return out


def test_the_atr_download_builds_box_candles_from_the_legs(tmp_path):
    api = OhlcApi()
    store = AtrStore(tmp_path)
    store.update(api, history_days=400, today=date(2026, 10, 24))
    frame = store.frame("CLBZ")
    outbox = frame[frame["symbol"] == "WTCLCOF27"].sort_values("date")
    assert not outbox.empty
    assert outbox["close"].iloc[0] == pytest.approx((70.0 + 0.19) - (72.0 + 0.19))
    assert any("BZZF27" in request for request in api.requests)  # the legs were asked for


# ----------------------------------------------------------------------
# Costs
# ----------------------------------------------------------------------


def test_a_box_pays_the_cl_and_bz_costs_on_every_lot():
    # TC only (rebates off): CL 0.89 + BZ 0.89 per lot per side, two sides, on 1 / 2 / 4 / 8 lots of each leg
    assert [round(round_trip_cost("CLBZ", f, DEFAULT_COSTS), 2) for f in ("outright", "spread", "fly", "dfly")] == [3.56, 7.12, 14.24, 28.48]
    # rebates on: the quoted-structure rebates replace the legs' own (CL 0.87, BZ 0.81)
    assert round_trip_cost("CLBZ", "outright", DEFAULT_COSTS, DEFAULT_COMBO_REBATES) == pytest.approx(2 * ((0.89 - 0.87) + (0.89 - 0.81)))
    assert tick_of("CLBZ", DEFAULT_COSTS).tick_value == 10.0
    assert breakeven_ticks("CLBZ", "outright", DEFAULT_COSTS) == pytest.approx(0.356)
    assert round_trip_cost("CL", "dfly", DEFAULT_COSTS) == round_trip_cost("CL", "dfly", DEFAULT_COSTS, DEFAULT_COMBO_REBATES)  # plain products ignore the box rebates


# ----------------------------------------------------------------------
# Curve Compare
# ----------------------------------------------------------------------


def test_curve_compare_quotes_the_box_spreads_under_the_spreadbox_code_and_falls_back_to_the_legs():
    tiers = symbol_tiers("CLBZ", TODAY, 4, SPREAD)
    assert tiers[0][0] == "CLBZZXF27-G27" and "WTCLCOF27" in tiers[1]
    assert set(tiers[-1]) >= {"CLF27", "BZZF27"}
    now = 1_000_000.0
    book = build_price_book({}, {"CLF27": (70.0, now), "BZZF27": (72.0, now), "CLG27": (69.0, now), "BZZG27": (71.6, now)})
    points = build_curve("CLBZ", TODAY, 2, OUTRIGHT, book, now)
    assert points[0].value == pytest.approx(-2.0) and points[0].source == SOURCE_DERIVED
    assert points[1].value == pytest.approx(-2.6)
    spreads = build_curve("CLBZ", TODAY, 2, SPREAD, book, now)
    assert spreads[0].symbol == "CLBZZXF27-G27" and spreads[0].value == pytest.approx(0.6)


def test_settlement_curve_uses_the_legs_when_the_box_has_no_settlement():
    settled = {"CLF27": 70.0, "BZZF27": 72.0, "CLG27": 69.0, "BZZG27": 71.6}
    assert settlement_curve("CLBZ", TODAY, 2, SPREAD, settled) == [pytest.approx(0.6)]
    assert settlement_curve("CLBZ", TODAY, 2, OUTRIGHT, {**settled, "WTCLCOF27": -1.0})[0] == -1.0


# ----------------------------------------------------------------------
# Settlements
# ----------------------------------------------------------------------


class StoreStub:
    version = "v1"

    def __init__(self, frames):
        self.frames = frames

    def frame(self, product):
        return self.frames.get(product, pd.DataFrame(columns=["contract", "date", "oi", "close", "volume"]))


def test_box_history_for_the_correlation_page_comes_from_the_stored_box_settlements():
    days = pd.bdate_range("2026-09-01", periods=40)
    frame = pd.concat([
        pd.DataFrame({"contract": code, "date": days, "oi": None, "close": [base + 0.1 * i for i in range(40)], "volume": 5.0})
        for code, base in (("WTCLCOF27", -2.0), ("WTCLCOG27", -2.5), ("WTCLCOH27", -2.8))
    ])
    loader = SettlementFirstLoader("unused", None, SettlementPrices(StoreStub({"CLBZ": frame})))
    assert loader.covered_by_settlements("CLBZF27")
    spreadbox = loader._derived_series("CLBZSF27-G27")
    assert spreadbox is not None and spreadbox.iloc[0] == pytest.approx(0.5)  # -2.0 - (-2.5): no BZ proxy needed
    flybox = loader._derived_series("CLBZSF27-G27-H27")
    assert flybox.iloc[0] == pytest.approx(-2.0 - 2 * -2.5 + -2.8)
    assert not loader.covered_by_settlements("CLBZF27-G27")  # a cross-month outbox needs the BZ leg, which is not stored


def test_the_settlement_store_downloads_box_settlements_from_the_legs(tmp_path):
    class DailyApi:
        def daily_market_data(self, qhcode, start, end, fields, limit, offset):
            rows = []
            for code in qhcode.split(","):
                if code.startswith("WTCLCO"):
                    continue  # the API has no box settlements
                base = 70.0 if code.startswith("CL") else 72.0
                rows += [{"qhcode": code, "datetime": f"2026-10-{d:02d}T00:00:00", "close": base, "volume": 7, "oi": 3} for d in (26, 27)]
            return {"results": rows, "count": len(rows)}

    store = OiStore(tmp_path)
    store.update(DailyApi(), history_days=60, today=date(2026, 10, 28))
    frame = store.frame("CLBZ")
    assert not frame.empty and (frame["close"] == -2.0).all() and frame["oi"].isna().all()
    assert frame["contract"].str.startswith("WTCLCO").all()


# ----------------------------------------------------------------------
# The Churn page
# ----------------------------------------------------------------------


def test_the_churn_page_scores_a_box_with_both_legs_costs(tmp_path):
    from core.churn_data import ChurnStore, tracked_instruments
    from core.churn_service import ChurnParams, ChurnService
    from tests.test_churn import make_bars
    from tests.test_churn_live import NOW, PARAMS, FakeRepo

    store = ChurnStore(tmp_path, clock=lambda: NOW)
    frames = []
    for n, instrument in enumerate(tracked_instruments("CLBZ", date(2026, 10, 14), 4)):
        bars = make_bars(days=21, seed=n, volume=50.0)
        bars["symbol"] = instrument.symbol
        frames.append(bars[["symbol", "ts", "open", "high", "low", "close", "volume"]])
    pd.concat(frames).to_parquet(store._path("CLBZ"), index=False)
    service = ChurnService(FakeRepo(), None, store, atr_provider=lambda p: {}, clock=lambda: NOW, background=False)
    rows = service.results(PARAMS).rows
    boxes = rows[rows["product"] == "CLBZ"]
    assert not boxes.empty and set(boxes["symbol"]) >= {"WTCLCOG27", "CLBZZXG27-H27"}
    outbox = boxes[boxes["family"] == "outright"].iloc[0]
    assert outbox["cost_rt"] == pytest.approx(3.56)  # CL + BZ TC on both sides, no rebates
    assert boxes[boxes["family"] == "spread"].iloc[0]["cost_rt"] == pytest.approx(7.12)
    assert any(f == "dfly" for f in boxes["family"])  # d-flybox from two flyboxes


# ----------------------------------------------------------------------
# Seasonality
# ----------------------------------------------------------------------


def test_seasonality_runs_on_the_stored_box_settlements():
    from core.curve_calendar import api_symbol, contract_expiry
    from core.seasonality import SeasonalityService, SeasonOptions

    rows = []
    for year in range(2022, 2029):
        for month in range(1, 13):
            expiry = pd.Timestamp(contract_expiry("CLBZ", month, year))
            days = pd.bdate_range(expiry - pd.Timedelta(days=400), expiry)
            level = -2.0 - 0.1 * month
            rows.append(pd.DataFrame({"contract": api_symbol("CLBZ", month, year), "date": days, "oi": None,
                                      "close": level + 0.01 * ((days - days[0]).days % 30), "volume": 10.0}))

    class Store(StoreStub):
        def has_data(self):
            return True

        loaded_history_days = 400

    service = SeasonalityService(Store({"CLBZ": pd.concat(rows)}), clock=lambda: pd.Timestamp(2026, 10, 29, tz="UTC").to_pydatetime())
    table = service.table("CLBZ", "spread", 3, SeasonOptions())
    assert len(table.rows) == 2 and table.rows[0].season.label == "Jan27/Feb27"
    assert table.rows[0].season.past  # earlier years were found at the same days to expiry


def test_the_oi_page_says_a_box_has_no_oi_but_still_lists_it_for_the_atr_tab():
    from types import SimpleNamespace

    from ui.callbacks import oi_callbacks
    from ui.container import container
    from ui.layouts.oi_tab import oi_layout

    saved = container.oi_service
    container.oi_service = SimpleNamespace()
    try:
        research = oi_callbacks.render_research("v1", "CLBZ", "research", 1)
        body, options, value, _ = oi_callbacks.render_monitor("v1", "CLBZ")
    finally:
        container.oi_service = saved
    assert "no open interest" in research.children and "no open interest" in body.children and options == [] and value is None
    assert "CLBZ" in str(oi_layout())  # the ATR tab shares the product picker
