"""Churn page: d-flies (derived from two flies), the single table with the live part, the start-from-contract filter,
the oscillation column, the reasons for a zero score and the live GO / WAIT verdict."""

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.churn_data import ChurnStore, derive_dfly_bars, dfly_instruments, tracked_instruments
from core.churn_engine import STATES, Path, Settings, analyse, count_oscillations, state_of
from core.churn_service import ChurnParams, ChurnService, validate_params
from tests.test_churn import TODAY, calendar_of, make_bars, make_spec
from ui.callbacks import churn_callbacks as cb
from ui.container import Container
from ui.layouts import churn_tab

NOW = datetime(2026, 10, 14, 12, 30, tzinfo=timezone.utc)  # 13:30 in London: inside 09:00-19:00


def _leg(values, volume=10.0, start="2026-10-12 00:00"):
    index = pd.date_range(start, periods=len(values), freq="5min")
    c = np.asarray(values, float)
    return pd.DataFrame({"ts": index, "open": c, "high": c + 0.01, "low": c - 0.01, "close": c, "volume": volume})


def test_dfly_is_the_first_fly_minus_the_next():
    a, b = _leg([1.00, 1.10, 1.30, 1.20]), _leg([0.40, 0.40, 0.50, 0.60], volume=4.0)
    out = derive_dfly_bars(a, b)
    assert np.allclose(out["close"], [0.60, 0.70, 0.80, 0.60])
    assert np.allclose(out["open"], [0.60, 0.60, 0.70, 0.80])  # the previous close
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all() and (out["low"] <= out[["open", "close"]].min(axis=1)).all()
    assert out["volume"].sum() == pytest.approx(16.0)  # the thinner fly's day volume (4 candles x 4) ...
    assert np.allclose(out["volume"], 4.0)  # ... spread evenly over the day's candles, so it falls inside any trading window


def test_dfly_carries_a_silent_fly_forward():
    a = _leg([1.0, 1.2, 1.4])
    b = _leg([0.5, 0.5])  # the second fly stops printing after two candles
    assert np.allclose(derive_dfly_bars(a, b)["close"], [0.5, 0.7, 0.9])


def test_dflies_are_built_on_consecutive_flies_only():
    flies = [i for i in tracked_instruments("CL", TODAY, 4) if i.family == "fly"]
    dflies = dfly_instruments("CL", flies)
    assert len(dflies) == 3 and dflies[0].label == "Nov26/Dec26/Jan27/Feb27"
    assert dflies[0].legs == (flies[0].symbol, flies[1].symbol) and all(d.family == "dfly" for d in dflies)
    assert dfly_instruments("CL", [flies[0], flies[2]]) == []  # not one month apart


def test_entry_states_are_recorded_and_signed_for_the_trade():
    assert [state_of(z) for z in (-2, -1, 0, 0.99, 1, 3)] == ["favourable", "favourable", "neutral", "neutral", "against", "against"]
    bars = make_bars()
    for direction in ("long", "short"):  # a short is the same trade on the negated price
        states = analyse(bars, make_spec(), direction, [1, 2, 3], 3.0, Settings(), calendar_of(bars))["entry"].states
        assert list(states.index) == list(STATES) and states["trials"].sum() > 0


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------


def test_the_atr_and_holding_settings_are_gone_and_old_saved_values_are_ignored():
    assert not {"atr_threshold", "hold_short", "hold_long", "session_hour"} & set(ChurnParams().__dict__)
    params = validate_params({"atr_threshold": 2, "hold_short": 3, "hold_long": 20, "session_hour": 22, "start_from": 4})
    assert params.start_from == 4 and params.osc_days == 5 and (params.window_start, params.window_end) == (9.0, 19.0)
    assert params.spread_dfly == 0.0 and params.stop_mult == 2.0 and params.patience_min == 60.0
    with pytest.raises(ValueError):
        validate_params({"contracts": 6, "start_from": 7})
    with pytest.raises(ValueError):
        validate_params({"osc_days": 0})
    with pytest.raises(ValueError):
        validate_params({"window_start": 12, "window_end": 10})


def test_only_the_simulation_settings_rebuild_the_simulation():
    base = ChurnParams()
    assert ChurnParams(start_from=8, osc_days=9, min_volume=1).sim_key() == base.sim_key()
    for change in ({"window_end": 18}, {"margin": 2}, {"stop_mult": 4}, {"patience_min": 30}, {"spread_fly": 2}):
        assert ChurnParams(**change).sim_key() != base.sim_key()


# ----------------------------------------------------------------------
# Live verdict
# ----------------------------------------------------------------------


def _base(hit=0.80, net=5.0, states=None, hour_trials=200, hour_hit=0.80):
    index = pd.Index(list(STATES), name="state")
    table = pd.DataFrame({"trials": [300, 300, 300], "hit_prob": [0.9, 0.8, 0.5], "net_trade": [8.0, 5.0, -2.0]}, index=index) if states is None else states
    hours = pd.DataFrame({"trials": [hour_trials], "hit_prob": [hour_hit], "net_trade": [net]}, index=pd.Index([12], name="hour"))
    return SimpleNamespace(entry_hit_prob=hit, net_trade=net, states=table, hour=hours)


def _row(direction="long", **extra):
    base = dict(symbol="X", product="CL", label="Dec26", family="outright", direction=direction, score=70.0, target=2, atr=1.5, cost_rt=1.78, net_hit=18.2, worse_score=0.5,
                position=4, net_day=30.0, volume_day=500.0, days=8, wins_day=3.0, stops_day=1.0, stop_usd=62.0, hit_prob=0.7, pos_days=0.6, blowup=0.0,
                insample_net_day=10.0, split=True)
    base.update(extra)
    return SimpleNamespace(**base)


STATS = {"mean": 70.00, "std": 0.10, "vol_ratio": 1.0, "last_ts": None}


def _verdict(price, stats=STATS, base=None, direction="long", age=5.0, hour=12):
    stamp = NOW.timestamp() - age
    return ChurnService._verdict(_row(direction), make_spec(), base or _base(), None if price is None else (price, stamp), stats, NOW.timestamp(), hour)


def test_go_when_the_price_is_on_the_good_side_and_nothing_is_off():
    row = _verdict(69.85)  # 1.5 standard deviations below the day's average: a dip, for a long
    assert row["verdict"] == "GO" and row["state"] == "favourable" and row["z"] == pytest.approx(-1.5)
    assert row["hit_state"] == pytest.approx(0.9)
    assert row["entry"] == pytest.approx(69.85 + 0.005) and row["target_price"] == pytest.approx(69.855 + 0.02)


def test_a_short_is_signed_the_other_way():
    row = _verdict(70.15, direction="short")  # a spike is the good side for a short
    assert row["state"] == "favourable" and row["entry"] == pytest.approx(70.15 - 0.005) and row["target_price"] == pytest.approx(70.145 - 0.02)
    assert _verdict(69.85, direction="short")["state"] == "against"


def test_wait_when_the_state_has_not_worked_before():
    row = _verdict(70.20)  # buying strength: past entries here hit 50% and lost money
    assert row["verdict"] == "WAIT" and row["state"] == "against" and "50%" in row["reason"]


def test_wait_in_a_quiet_or_a_hot_market_and_in_a_weak_hour():
    assert "quiet market" in _verdict(70.0, {**STATS, "vol_ratio": 0.3})["reason"]
    assert "volatile market" in _verdict(70.0, {**STATS, "vol_ratio": 2.6})["reason"]
    weak = _verdict(70.0, base=_base(hour_hit=0.55))
    assert weak["verdict"] == "WAIT" and "weak hour" in weak["reason"] and "12:00 London" in weak["reason"] and weak["hit_hour"] == pytest.approx(0.55)
    assert _verdict(70.0)["verdict"] == "GO"


def test_no_price_stale_price_and_too_few_candles():
    assert _verdict(None)["verdict"] == "NO PRICE"
    assert _verdict(70.0, age=500)["verdict"] == "STALE"
    thin = _verdict(70.0, stats=None)
    assert thin["verdict"] == "WAIT" and "not enough recent candles" in thin["reason"]


def test_a_state_with_few_past_entries_is_not_trusted():
    few = pd.DataFrame({"trials": [20, 300, 300], "hit_prob": [0.1, 0.8, 0.5], "net_trade": [-9.0, 5.0, -2.0]}, index=pd.Index(list(STATES)))
    assert _verdict(69.85, base=_base(states=few))["verdict"] == "GO"  # 20 trials: ignored


def _live_row(r, inside=True, price=(70.00, None), params=None):
    detail = {"entry": _base(), "path": Path(np.array([70.00, 70.03]), np.array([69.99, 70.00]), np.array([0]), np.array([1]))}
    price = (price[0], NOW.timestamp() - 5) if price else None
    return ChurnService._row_live(r, make_spec(), detail, price, STATS, NOW.timestamp(), 12, inside, params or ChurnParams())


def test_a_negative_score_says_flip_or_avoid_and_why_in_plain_words():
    losing = _live_row(_row(score=-12.0, net_day=-18.7, hit_prob=0.2, other_net_day=-4.0, flip=False))
    assert losing["verdict"] == "AVOID" and "Neither side makes money" in losing["reason"] and "loses $19 per lot a day over the 8 scored days" in losing["reason"]
    assert "3.0 wins a day at +$18.20 against 1.0 stop-outs a day at -$62.00" in losing["reason"] and "needs 77% of trades to win, it got 20%" in losing["reason"]
    assert "The short side loses $4 a day too" in losing["reason"]
    flip = _live_row(_row(score=-83.0, net_day=-18.7, hit_prob=0.2, other_net_day=41.0, flip=True))
    assert flip["verdict"] == "FLIP" and "Churn the short side instead: it scores +83 and makes $41 per lot a day" in flip["reason"]
    assert _live_row(_row("short", score=-83.0, net_day=-18.7, other_net_day=41.0, flip=True))["reason"].startswith("Churn the long side instead")
    thin = _live_row(_row(volume_day=12.0, thin=True))
    assert thin["verdict"] in ("GO", "WAIT") and thin["thin"] and "Thin: only 12 lots a day" in thin["reason"] and "minimum 100" in thin["reason"]
    assert not _live_row(_row())["thin"]


def test_a_scored_row_outside_your_hours_is_closed_and_a_recent_only_profit_is_flagged():
    closed = _live_row(_row(), inside=False)
    assert closed["verdict"] == "CLOSED" and "Outside your trading hours (9:00 to 19:00 London)" in closed["reason"]
    assert closed["osc"] == 1  # still worked out: the market was at the entry level (70.005), then 2 ticks higher, on one day
    lucky = _live_row(_row(insample_net_day=-6.0))
    assert lucky["verdict"] in ("GO", "WAIT") and "Caution: it lost $6 per lot a day on the earlier days" in lucky["reason"]
    assert "Caution" not in _live_row(_row(insample_net_day=4.0))["reason"]
    unsplit = _live_row(_row(insample_net_day=-6.0, split=False))["reason"]
    assert "earlier days" not in unsplit and "only 8 trading days of candles, too few to split" in unsplit
    assert np.isnan(_live_row(_row(), price=None)["osc"])


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------

LIVE: dict = {}
PARAMS = ChurnParams(contracts=4, start_from=1, min_volume=1, min_z=0.0, atr_share=0.0)  # the plain ladder; the new options have their own tests


class FakeRepo:
    def __init__(self):
        self.values = {}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)

    def set_setting(self, key, value):
        self.values[key] = value


def _make_service(directory):
    store = ChurnStore(directory, clock=lambda: NOW)
    for product in ("CL", "BRN"):
        frames = []
        for n, instrument in enumerate(tracked_instruments(product, TODAY, 4)):
            bars = make_bars(days=21, seed=n + (7 if product == "BRN" else 0), volume=0.0 if n == 11 else 50.0)  # the fourth fly never trades
            bars["symbol"] = instrument.symbol
            frames.append(bars[["symbol", "ts", "open", "high", "low", "close", "volume"]])
        pd.concat(frames).to_parquet(store._path(product), index=False)
    atr = lambda p: {i.symbol: 1.5 for i in tracked_instruments(p, TODAY, 4)}
    return ChurnService(FakeRepo(), None, store, atr_provider=atr, clock=lambda: NOW, background=False, live_prices=lambda: LIVE)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    service = _make_service(tmp_path_factory.mktemp("churn"))
    service.results(PARAMS)  # built once for the module: it takes a few seconds
    return service


@pytest.fixture
def service(shared):
    LIVE.clear()
    yield shared
    LIVE.clear()


def _fill_live(service, age=5.0):
    """A live price for every tracked instrument: the last stored close, as if the market had not moved."""
    for product in ("CL", "BRN"):
        for symbol, part in service.store.frame(product).groupby("symbol"):
            LIVE[symbol] = (float(part.sort_values("ts")["close"].iloc[-1]), NOW.timestamp() - age)


def test_every_contract_is_simulated_but_only_those_from_the_start_are_shown(service):
    results = service.results(PARAMS)
    assert set(results.rows["position"]) == {1, 2, 3, 4}  # the front contracts are simulated too
    shown = service.scored(ChurnParams(contracts=4, start_from=3, min_volume=1, min_z=0.0, atr_share=0.0))  # same simulation: the start is a view setting
    assert set(shown["position"]) == {3, 4} and not shown.empty
    everything = service.scored(PARAMS).set_index(["symbol", "direction"])["score"]  # every part is on a fixed scale:
    later = shown.set_index(["symbol", "direction"])["score"]
    assert np.allclose(everything.loc[later.index], later)  # a row's score does not depend on which other rows are shown


def test_positions_count_from_the_front_within_each_curve(service):
    rows = service.results(PARAMS).rows
    cl = rows[(rows["product"] == "CL") & (rows["direction"] == "long")]
    for family in ("outright", "spread", "fly"):
        assert list(cl[cl["family"] == family].sort_values("position")["position"]) == [1, 2, 3, 4]
    assert list(cl[cl["family"] == "dfly"].sort_values("position")["position"]) == [1, 2, 3]
    assert cl[(cl["family"] == "outright") & (cl["position"] == 1)]["label"].iloc[0] == "Nov26"


def test_dflies_are_ranked_with_the_dfly_cost_and_no_spread(service):
    rows = service.results(PARAMS).rows
    dfly = rows[rows["family"] == "dfly"]
    assert set(dfly["product"]) == {"CL", "BRN"} and len(dfly) == 2 * 2 * 3  # three d-flies, both sides, two products
    cl = dfly[dfly["product"] == "CL"].iloc[0]
    assert cl["cost_rt"] == pytest.approx(2 * 8 * 0.89) and cl["label"].count("/") == 3
    spec = service.results(PARAMS).details[(cl["symbol"], cl["direction"])]["spec"]
    assert spec.half_spread == 0.0  # built from the legs: nothing to cross


def test_the_figures_that_rank_a_row_are_out_of_sample(service):
    row = service.results(PARAMS).rows.iloc[0]
    assert row["split"] and row["days"] == 9 and np.isfinite(row["insample_net_day"]) and np.isfinite(row["tstat"])  # 15 weekdays: the last 60% are scored


def test_thin_contracts_are_scored_and_marked_and_no_score_is_zero(service):
    params = ChurnParams(contracts=4, start_from=1, min_volume=10, min_z=0.0, atr_share=0.0)
    rows = service.scored(params)
    thin = rows[rows["thin"]]
    assert len(thin) == 2 * 2 * 2  # the silent fly and the d-fly built on it, both sides, both products
    assert rows["score"].is_monotonic_decreasing and rows["score"].between(-100, 100).all() and (rows["score"].abs() >= 1).all()  # thin rows too
    assert ((rows["score"] > 0) == (rows["net_day"] > 0)).all()  # the sign is the sign of the profit
    both = rows.set_index(["symbol", "direction"])
    for (symbol, direction), r in both.iterrows():
        other = both.loc[(symbol, "short" if direction == "long" else "long")]
        assert r["other_score"] == pytest.approx(other["score"])
        if r["net_day"] <= 0 and other["net_day"] > 0:
            assert r["flip"] and r["score"] == pytest.approx(-other["score"])  # the other side is the churn: minus its score
        elif r["net_day"] <= 0:
            assert not r["flip"] and -30 <= r["score"] <= -1  # neither side works: a small negative


def test_the_score_is_signed_never_zero_and_a_big_negative_means_flip():
    from core.churn_service import _score

    rows = pd.DataFrame({
        "symbol": ["A", "A", "B", "B", "C", "C", "D"], "direction": ["long", "short", "long", "short", "long", "short", "long"], "volume_day": [500.0] * 6 + [5.0],
        "net_day": [40.0, -35.0, -3.0, -20.0, 0.0, 6.0, 12.0], "tstat": [6.0, -5.0, -0.2, -4.0, 0.0, 0.3, 2.0],
        "hit_prob": [0.9, 0.3, 0.6, 0.4, 0.5, 0.6, 0.7], "worse_score": [0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.5], "pos_days": [0.9, 0.2, 0.5, 0.3, 0.5, 0.5, 0.6], "blowup": [0.0] * 7,
    })
    scored = _score(rows, 100.0).set_index(["symbol", "direction"])
    score = scored["score"]
    assert score[("A", "long")] > 85 and score[("A", "short")] == pytest.approx(-score[("A", "long")])  # the short loses, the long is strong: FLIP
    assert scored.loc[("A", "short"), "flip"] and not scored.loc[("A", "long"), "flip"]
    assert -30 <= score[("B", "short")] < score[("B", "long")] <= -1 and not scored.loc[("B", "long"), "flip"]  # both lose: mild, the surer loser lower
    assert score[("C", "long")] == pytest.approx(-score[("C", "short")]) and 1 <= score[("C", "short")] < 50  # a net of exactly 0 is not a 0 score
    assert (score.abs() >= 1).all() and score.between(-100, 100).all()
    assert scored.loc[("D", "long"), "thin"] and score[("D", "long")] > 0 and np.isnan(scored.loc[("D", "long"), "other_score"])  # thin: scored and flagged
    assert scored.loc[("A", "long"), "other_score"] == pytest.approx(score[("A", "short")])
    alone = _score(rows.iloc[[0, 1]].assign(net_day=[40.0, -35.0]), 100.0)  # the score is not a rank: fewer rows, same scores
    assert np.allclose(alone["score"], [score[("A", "long")], score[("A", "short")]])


def test_the_table_is_none_until_the_simulation_exists_then_fills_in(service):
    assert service.live_table(ChurnParams(contracts=5, start_from=1)) is None  # a different simulation setting: not built
    table = service.live_table(PARAMS)
    scored = service.scored(PARAMS)
    assert len(table) == len(scored) and list(table["symbol"]) == list(scored["symbol"])  # same rows, same order
    assert set(table["verdict"]) <= {"NO PRICE", "FLIP", "AVOID"}  # no live prices yet
    _fill_live(service)
    table = service.live_table(PARAMS)
    live = table[table["verdict"].isin(["GO", "WAIT"])]
    assert len(live) > 3 and live["price"].notna().all() and (live["entry"] != live["price"])[live["family"] != "dfly"].all() and np.allclose(live["age"], 5.0)
    assert set(service.live_table(PARAMS, family="dfly")["family"]) == {"dfly"} and set(service.live_table(PARAMS, direction="short")["direction"]) == {"short"}
    LIVE.clear()
    assert set(service.live_table(PARAMS)["verdict"]) <= {"NO PRICE", "FLIP", "AVOID"} and service.live_table(PARAMS)["osc"].isna().all()
    _fill_live(service, age=900)
    assert set(service.live_table(PARAMS)["verdict"]) <= {"STALE", "FLIP", "AVOID"}


def test_outside_your_trading_hours_the_rows_say_closed(service, monkeypatch):
    _fill_live(service)
    monkeypatch.setattr(service, "_clock", lambda: datetime(2026, 10, 14, 19, 30, tzinfo=timezone.utc))  # 20:30 London
    for symbol in LIVE:
        LIVE[symbol] = (LIVE[symbol][0], datetime(2026, 10, 14, 19, 30, tzinfo=timezone.utc).timestamp() - 5)
    table = service.live_table(PARAMS)
    assert "CLOSED" in set(table["verdict"]) and not {"GO", "WAIT"} & set(table["verdict"])


def test_the_oscillation_count_is_that_of_the_live_price_level(service):
    results = service.results(PARAMS)
    _fill_live(service)
    params = ChurnParams(contracts=4, start_from=1, min_volume=1, osc_days=3, min_z=0.0, atr_share=0.0)
    table = service.live_table(params, product="CL", family="outright").set_index(["symbol", "direction"])
    assert len(table) > 0
    for (symbol, direction), row in table.iterrows():
        detail = results.details[(symbol, direction)]
        spec = detail["spec"]
        sign = 1 if direction == "long" else -1
        level = LIVE[symbol][0] + sign * spec.half_spread * spec.tick
        assert row["osc"] == count_oscillations(detail["path"], direction, level, detail["best"], spec.tick, 3)
    wide = service.live_table(ChurnParams(contracts=4, start_from=1, min_volume=1, osc_days=10, min_z=0.0, atr_share=0.0), product="CL", family="outright").set_index(["symbol", "direction"])
    assert (wide["osc"] >= table["osc"]).all()  # a longer range can only find more


def test_a_dfly_is_priced_live_from_its_two_flies(service):
    flies = [i for i in tracked_instruments("CL", TODAY, 4) if i.family == "fly"]
    LIVE.update({flies[0].symbol: (1.30, NOW.timestamp() - 3), flies[1].symbol: (0.80, NOW.timestamp() - 9)})
    price, stamp = service._live_price(make_spec(symbol="x"), (flies[0].symbol, flies[1].symbol))
    assert price == pytest.approx(0.50) and NOW.timestamp() - stamp == pytest.approx(9.0)  # the older of the two stamps
    assert service._live_price(make_spec(symbol="x"), (flies[0].symbol, flies[2].symbol)) is None  # one fly has no price


def test_detail_grid_has_every_target_and_worse_fill(service):
    params = ChurnParams(contracts=4, start_from=1, min_volume=1, max_target=4, min_z=0.0, atr_share=0.0)
    assert service.cached(params) is None
    service.results(params)  # max_target is a simulation setting: this builds a second simulation
    row = service.scored(params).iloc[0]
    grid = service.detail_grid(row["symbol"], row["direction"], params)
    assert set(grid["target"]) == {1, 2, 3, 4} and set(grid["worse"]) == {0, 1, 2, 3}


def test_rebates_are_ignored_unless_switched_on(tmp_path):
    service = _make_service(tmp_path)  # its own copy: this test changes the settings
    params = ChurnParams(contracts=3, start_from=1, min_volume=1)
    off = service.results(params).rows
    cl = off[off["product"] == "CL"]
    expected = {"outright": 2 * 0.89, "spread": 2 * 2 * 0.89, "fly": 2 * 4 * 0.89, "dfly": 2 * 8 * 0.89}
    assert all(r.cost_rt == pytest.approx(expected[r.family]) for r in cl.itertuples())
    service.save_costs(*service.costs(), True)
    on = service.results(params).rows
    assert (on[on["product"] == "CL"]["cost_rt"].to_numpy() < cl["cost_rt"].to_numpy()).all()


def test_params_persist(tmp_path):
    service = _make_service(tmp_path)
    service.save_params(ChurnParams(contracts=5, start_from=2, window_end=18))
    assert service.params().contracts == 5 and service.params().start_from == 2 and service.params().window_end == 18


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------

USER_COLUMNS = ["Contract", "Live Price", "Volume/day", "ATR", "Cost", "Per lot Net Profit", "Number of oscillation", "Score", "Other Side Score", "Side", "Worse Price Score",
                "Week Range (low to high)", "Month Range (low to high)"]


def test_the_table_has_the_columns_asked_for_and_nothing_else_visible():
    shown = [c for c in churn_tab.COLUMNS if c["id"] not in ("verdict", "thin")]
    assert [c["name"] for c in shown] == USER_COLUMNS
    table = churn_tab.churn_table()
    assert table.hidden_columns == ["verdict", "thin"] and table.id == "churn-table"
    assert set(table.tooltip_header) == {c["id"] for c in shown}  # every visible heading explains itself


def test_the_range_columns_are_the_low_and_high_of_the_last_week_and_month(service, monkeypatch):
    now_utc = datetime(2026, 8, 21, 12, 30, tzinfo=timezone.utc)  # the stored candles run from 3 to 23 August
    monkeypatch.setattr(service, "_recent", {})
    _fill_live(service)
    details = service.results(PARAMS).details
    with monkeypatch.context() as patch:  # the ranges are worked out as of that day and kept for this version of the store
        patch.setattr(service, "_clock", lambda: now_utc)
        for detail in details.values():
            spec = detail["spec"]
            service._ranges(spec, tuple(spec.symbol.split("/")) if spec.family == "dfly" else None)
    table = service.live_table(PARAMS, product="CL", family="outright", direction="long")
    row = table.iloc[0]
    bars = service.store.frame("CL")
    bars = bars[bars["symbol"] == row["symbol"]]
    now = pd.Timestamp(now_utc).tz_localize(None)
    assert row["range_days"] == pytest.approx(18.52, abs=0.01)
    for name, days in (("week", 7), ("month", 30)):
        part = bars[bars["ts"] >= now - pd.Timedelta(days=days)]
        assert row[f"{name}_low"] == pytest.approx(part["low"].min()) and row[f"{name}_high"] == pytest.approx(part["high"].max())
    assert row["month_low"] <= row["week_low"] <= row["week_high"] <= row["month_high"]
    data, tips = churn_tab.table_data(table)
    assert data[0]["week_range"] == f"{row['week_low']:.2f} to {row['week_high']:.2f}" and " to " in data[0]["month_range"]
    assert "The live price" in tips[0]["week_range"]["value"] and "Last 30 days" in tips[0]["month_range"]["value"]
    dfly = service.live_table(PARAMS, product="CL", family="dfly").iloc[0]
    assert np.isfinite(dfly["week_low"]) and dfly["week_low"] <= dfly["week_high"]  # a d-fly's range comes from its two flies
    short = churn_tab._range_tip("month", 30, 1.0, 2.0, 1.5, 12.0)
    assert "Only 12 days of candles are stored" in short and "50% of the way" in short
    assert churn_tab._range_text(np.nan, np.nan) is None and "No candles" in churn_tab._range_tip("week", 7, np.nan, np.nan, np.nan, np.nan)


def test_the_rarely_changed_settings_sit_under_advanced(service, monkeypatch):
    controls = churn_tab.churn_controls(ChurnParams())
    text = str(controls)
    advanced = str(controls.children[-1])
    assert controls.children[-1].id == "churn-advanced" and "Advanced settings" in advanced
    for component_id, _, field, _, _ in churn_tab.PARAM_INPUTS:
        assert component_id in text and (component_id in advanced) == (field not in churn_tab.MAIN_FIELDS)
    assert len(churn_tab.MAIN_FIELDS) == 6


def test_page_callbacks_end_to_end(service, monkeypatch):
    monkeypatch.setattr(cb, "container", Container())
    cb.container.churn_service = service
    values = [getattr(PARAMS, f) for f in cb._FIELDS]
    note, options, value, params, session = cb.render_churn("v1", "CL", "all", "long", 0, *values)
    assert options and value == options[0]["value"] and params["contracts"] == 4 and "from #1" in str(note) and "9:00 to 19:00" in str(note) and len(session.data) == 2
    assert all("CL" in o["label"] and "long" in o["label"] for o in options)
    _fill_live(service)
    data, tips = cb.render_table(0, params, "v1", "CL", "all", "long")
    assert len(data) == len(tips) == len(options) and {"contract", "price", "osc", "score", "verdict", "direction", "worse_score"} <= set(data[0])
    assert data[0]["contract"].startswith("CL ") and "Target" in tips[0]["contract"]["value"]
    assert tips[0]["score"]["value"].split(":")[0] in ("GO", "WAIT", "FLIP", "AVOID", "CLOSED", "STALE", "NO PRICE")
    assert "Proof:" in tips[0]["score"]["value"] and "Tradability" in tips[0]["score"]["value"] and {"other_score", "week_range", "month_range", "thin"} <= set(data[0])
    assert "rolling blocks" in str(note) and "-100 to +100" in str(note)
    net, hit = cb.render_detail(value, params, "v1")
    assert len(net.data) == 4 and len(hit.data) == 4
    bad = cb.render_churn("v1", "all", "all", "all", 0, *(["x"] + [None] * (len(values) - 1)))
    assert "⚠️" in str(bad[0])
    skip = cb.render_churn("v1", "all", "all", "all", 0, *[v if f != "start_from" else 9 for f, v in zip(cb._FIELDS, values)])
    assert "cannot be beyond" in str(skip[0])
    hours = cb.render_churn("v1", "all", "all", "all", 0, *[v if f != "window_end" else 8 for f, v in zip(cb._FIELDS, values)])
    assert "must end after they start" in str(hours[0])
    assert cb.render_table(0, None, "v1", "all", "all", "all") == (cb.no_update, cb.no_update)


def test_the_costs_panel_and_the_table_wait_for_the_simulation(tmp_path, monkeypatch):
    service = _make_service(tmp_path)
    monkeypatch.setattr(cb, "container", Container())
    cb.container.churn_service = service
    assert cb.render_table(0, ChurnParams(contracts=4).__dict__, "v", "all", "all", "all") == (cb.no_update, cb.no_update)  # not built yet: leave the table alone
    n_cost = len(cb._COST_IDS)
    assert "$1.78" in str(cb.render_breakeven(0))  # rebates are off by default: the full 0.89 TC on both sides
    msg, bump = cb.save_costs_callback(1, *([None] * n_cost), 0.5, None, True)
    assert "Saved" in str(msg) and bump == 1 and service.costs()[1]["CL"] == 0.5 and service.use_rebates()
    assert "$1.38" in str(cb.render_breakeven(1))
    assert "⚠️" in str(cb.save_costs_callback(2, "x", *([None] * (n_cost - 1)), None, None, True)[0])


def test_layout_ids_exist_for_every_callback(service, monkeypatch):
    import dash

    monkeypatch.setattr(churn_tab, "container", Container())
    churn_tab.container.churn_service = service
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    layout = churn_tab.churn_layout()
    app.layout = layout
    cb.register_churn_callbacks(app)
    ids = set()

    def walk(component):
        if getattr(component, "id", None):
            ids.add(component.id)
        children = getattr(component, "children", None)
        for child in children if isinstance(children, (list, tuple)) else ([children] if children is not None else []):
            if hasattr(child, "_prop_names"):
                walk(child)

    walk(layout)
    used = {i["id"] for spec in app.callback_map.values() for i in spec["inputs"] + spec.get("state", [])}
    used |= {part.rsplit(".", 1)[0] for key in app.callback_map for part in key.strip(".").split("...")}  # keys look like "..a.children...b.value.."
    assert sorted(u for u in used if u not in ids) == []
    assert "churn-live-wrap" not in ids and "churn-wrap" not in ids  # the second table is gone
    fields = {f for _, _, f, _, _ in churn_tab.PARAM_INPUTS}
    assert {"window_start", "window_end", "stop_mult", "patience_min", "start_from", "osc_days"} <= fields and "session_hour" not in fields


def test_the_store_refreshes_the_newest_candles_during_the_day(tmp_path):
    clock = {"now": datetime(2026, 10, 14, 9, 0, tzinfo=timezone.utc)}
    store = ChurnStore(tmp_path, clock=lambda: clock["now"])

    class Api:
        def ohlc(self, *a, **k):
            return []

    store.update(Api(), 6, 3, TODAY)
    now = clock["now"]
    assert store.is_current(TODAY, 6, 3, now, 10) and store.daily_version == "2026-10-14|6|3"
    later = datetime(2026, 10, 14, 9, 25, tzinfo=timezone.utc)
    assert not store.is_current(TODAY, 6, 3, later, 10) and store.is_current(TODAY, 6, 3, later, 60)
    assert store.is_current(TODAY, 6, 3)  # the daily check ignores the age


def test_tradability_does_not_depend_on_profit_and_proof_says_whether_a_profit_is_backed_by_the_earlier_days():
    from core.churn_service import _score

    rows = pd.DataFrame({
        "volume_day": [500.0, 500.0, 500.0, 5.0], "net_day": [30.0, 30.0, -50.0, -5.0], "insample_net_day": [10.0, -4.0, -9.0, 1.0], "split": [True, True, True, True],
        "hit_prob": [0.6, 0.6, 0.5, 0.9], "worse_score": [0.8] * 4, "pos_days": [0.7] * 4, "blowup": [0.0] * 4,
        "trips_day": [3.0, 3.0, 2.0, 5.0], "cost_rt": [4.0] * 4, "target_usd": [60.0] * 4,
    })
    scored = _score(rows, 100.0)
    assert list(scored["proof"]) == ["Proven", "Recent only", "No edge", "No edge"]
    assert scored["score"][2] < 0 and scored["tradable"][2] > 0  # a loser scores below zero but is still workable
    assert scored["tradable"][3] > 0 and scored["thin"][3]  # a thin row is judged too, and flagged
    assert scored["tradable"].between(0, 100).all()


def test_saved_settings_that_still_hold_the_old_defaults_take_the_new_ones_but_changed_ones_stay():
    from core.churn_service import KEY_CHURN_PARAMS, load_params

    repo = FakeRepo()
    repo.set_setting(KEY_CHURN_PARAMS, {"window_days": 60, "max_target": 6, "stop_mult": 3.0, "patience_min": 45})
    moved = load_params(repo)
    assert (moved.window_days, moved.max_target, moved.stop_mult, moved.patience_min) == (120, 15, 2.0, 45.0)
    repo.set_setting(KEY_CHURN_PARAMS, {"window_days": 60, "max_target": 8, "stop_mult": 3.0})
    kept = load_params(repo)
    assert (kept.window_days, kept.max_target, kept.stop_mult) == (60, 8, 3.0)


def test_the_calibration_runs_the_ladder_on_what_you_traded_at_your_own_target(service, monkeypatch):
    from datetime import timedelta
    from types import SimpleNamespace

    from core.models.trade import TradeEventType as E

    fly = next(i for i in tracked_instruments("CL", TODAY, 4) if i.family == "fly")
    structure = SimpleNamespace(structure_id="s", name="CL fly", legs=[SimpleNamespace(contract=SimpleNamespace(symbol=fly.symbol), ratio=1)])
    opened = NOW - timedelta(days=2)
    trades = [SimpleNamespace(event_type=E.TRADE, lots=12, direction="buy", realized_pnl=None, transaction_cost=3.56, timestamp=opened)]
    trades += [SimpleNamespace(event_type=E.PARTIAL_EXIT, lots=1, direction="sell", realized_pnl=20.0, transaction_cost=3.56, timestamp=opened + timedelta(minutes=5 * n)) for n in range(1, 13)]
    monkeypatch.setattr(service._repository, "get_all_structures", lambda: [structure], raising=False)
    monkeypatch.setattr(service._repository, "get_trades_for_structure", lambda _id: trades, raising=False)
    monkeypatch.setattr(service, "_fit", None)
    assert service.calibration(ChurnParams(contracts=5)) is None  # that simulation is not built: nothing is run
    fit = service.calibration(PARAMS)
    (family,) = fit["families"]
    assert (family["family"], family["trips"], family["fitted"], family["spread_now"], family["target"]) == ("fly", 12, True, 1.0, 2.0)  # 2 ticks a round trip: your target
    assert family["ticks"] == pytest.approx(2.0) and family["overnight"] == 0.0 and np.isfinite(family["sim_now"]) and np.isfinite(family["sim_fit"])
    assert abs(family["sim_fit"] - 2.0) <= abs(family["sim_now"] - 2.0) and set(fit["changes"]) <= {"spread_fly", "stop_mult"}
    assert service.calibration(PARAMS) is fit  # kept until the candles, the settings or the trades change
    trades.append(SimpleNamespace(event_type=E.FULL_EXIT, lots=1, direction="sell", realized_pnl=20.0, transaction_cost=3.56, timestamp=NOW))
    assert service.calibration(PARAMS)["families"][0]["trips"] == 13
