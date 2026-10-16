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
    assert params.spread_dfly == 0.0 and params.stop_mult == 3.0 and params.patience_min == 60.0
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


def test_a_row_with_no_score_says_why_in_plain_words():
    losing = _live_row(_row(score=0.0, net_day=-18.7, hit_prob=0.2))
    assert losing["verdict"] == "NOT SCORED" and "loses $19 per lot a day over the last 8 days" in losing["reason"]
    assert "3.0 wins a day at +$18.20 against 1.0 stop-outs a day at -$62.00" in losing["reason"] and "needs 77% of trades to win, it got 20%" in losing["reason"]
    thin = _live_row(_row(score=0.0, volume_day=12.0))
    assert "only 12 lots a day" in thin["reason"] and "minimum 100" in thin["reason"]
    parts = _live_row(_row(score=0.0, net_day=5.0, pos_days=0.4))
    assert "makes money but a part of the score is zero" in parts["reason"]


def test_a_scored_row_outside_your_hours_is_closed_and_a_recent_only_profit_is_flagged():
    closed = _live_row(_row(), inside=False)
    assert closed["verdict"] == "CLOSED" and "Outside your trading hours (9:00 to 19:00 London)" in closed["reason"]
    assert closed["osc"] == 1  # still worked out: the market was at the entry level (70.005), then 2 ticks higher, on one day
    lucky = _live_row(_row(insample_net_day=-6.0))
    assert lucky["verdict"] in ("GO", "WAIT") and "Caution: it lost $6 per lot a day on the earlier days" in lucky["reason"]
    assert "Caution" not in _live_row(_row(insample_net_day=4.0))["reason"] and "Caution" not in _live_row(_row(insample_net_day=-6.0, split=False))["reason"]
    assert np.isnan(_live_row(_row(), price=None)["osc"])


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------

LIVE: dict = {}
PARAMS = ChurnParams(contracts=4, start_from=1, min_volume=1)


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
    shown = service.scored(ChurnParams(contracts=4, start_from=3, min_volume=1))  # same simulation: the start is a view setting
    assert set(shown["position"]) == {3, 4} and not shown.empty
    everything = service.scored(PARAMS).set_index(["symbol", "direction"])["score"]  # the edge part ranks among the rows shown
    later = shown.set_index(["symbol", "direction"])["score"]
    assert not np.allclose(everything.loc[later.index], later)


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
    assert row["split"] and row["days"] == 6 and np.isfinite(row["insample_net_day"])  # 15 weekdays: the last 40% are scored


def test_thin_contracts_are_shown_but_not_scored(service):
    params = ChurnParams(contracts=4, start_from=1, min_volume=10)
    rows = service.scored(params)
    thin = rows[~rows["eligible"]]
    assert len(thin) == 2 * 2 * 2 and (thin["score"] == 0).all()  # the silent fly and the d-fly built on it, both sides, both products
    assert rows["score"].is_monotonic_decreasing and rows["score"].between(0, 100).all()


def test_a_contract_that_makes_money_never_scores_zero():
    from core.churn_service import _score

    rows = pd.DataFrame({"volume_day": [500.0] * 3, "net_day": [30.0, 10.0, -5.0], "hit_prob": [0.8, 0.5, 0.9], "worse_score": [0.0, 0.0, 0.5], "pos_days": [0.0, 0.6, 0.9], "blowup": [0.3, 0.0, 0.0]})
    scored = _score(rows, 100.0)
    assert scored["score"][0] > 0 and scored["score"][1] > 0 and scored["score"][2] == 0  # weak parts are floored, a loser is still 0
    assert scored["score"][1] > scored["score"][0]  # the better-rounded one wins even though its net per day is lower


def test_the_table_is_none_until_the_simulation_exists_then_fills_in(service):
    assert service.live_table(ChurnParams(contracts=5, start_from=1)) is None  # a different simulation setting: not built
    table = service.live_table(PARAMS)
    scored = service.scored(PARAMS)
    assert len(table) == len(scored) and list(table["symbol"]) == list(scored["symbol"])  # same rows, same order
    assert set(table["verdict"]) <= {"NO PRICE", "NOT SCORED"}  # no live prices yet
    _fill_live(service)
    table = service.live_table(PARAMS)
    live = table[table["verdict"].isin(["GO", "WAIT"])]
    assert len(live) > 3 and live["price"].notna().all() and (live["entry"] != live["price"]).all() and np.allclose(live["age"], 5.0)
    assert set(service.live_table(PARAMS, family="dfly")["family"]) == {"dfly"} and set(service.live_table(PARAMS, direction="short")["direction"]) == {"short"}
    LIVE.clear()
    assert set(service.live_table(PARAMS)["verdict"]) <= {"NO PRICE", "NOT SCORED"} and service.live_table(PARAMS)["osc"].isna().all()
    _fill_live(service, age=900)
    assert set(service.live_table(PARAMS)["verdict"]) <= {"STALE", "NOT SCORED"}


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
    params = ChurnParams(contracts=4, start_from=1, min_volume=1, osc_days=3)
    table = service.live_table(params, product="CL", family="outright").set_index(["symbol", "direction"])
    assert len(table) > 0
    for (symbol, direction), row in table.iterrows():
        detail = results.details[(symbol, direction)]
        spec = detail["spec"]
        sign = 1 if direction == "long" else -1
        level = LIVE[symbol][0] + sign * spec.half_spread * spec.tick
        assert row["osc"] == count_oscillations(detail["path"], direction, level, detail["best"], spec.tick, 3)
    wide = service.live_table(ChurnParams(contracts=4, start_from=1, min_volume=1, osc_days=10), product="CL", family="outright").set_index(["symbol", "direction"])
    assert (wide["osc"] >= table["osc"]).all()  # a longer range can only find more


def test_a_dfly_is_priced_live_from_its_two_flies(service):
    flies = [i for i in tracked_instruments("CL", TODAY, 4) if i.family == "fly"]
    LIVE.update({flies[0].symbol: (1.30, NOW.timestamp() - 3), flies[1].symbol: (0.80, NOW.timestamp() - 9)})
    price, stamp = service._live_price(make_spec(symbol="x"), (flies[0].symbol, flies[1].symbol))
    assert price == pytest.approx(0.50) and NOW.timestamp() - stamp == pytest.approx(9.0)  # the older of the two stamps
    assert service._live_price(make_spec(symbol="x"), (flies[0].symbol, flies[2].symbol)) is None  # one fly has no price


def test_detail_grid_has_every_target_and_worse_fill(service):
    params = ChurnParams(contracts=4, start_from=1, min_volume=1, max_target=4)
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

USER_COLUMNS = ["Contract", "Live Price", "Volume/day", "ATR", "Cost", "Per lot Net Profit", "Number of oscillation", "Score", "Side", "Worse Price Score"]


def test_the_table_has_the_columns_asked_for_and_nothing_else_visible():
    shown = [c for c in churn_tab.COLUMNS if c["id"] not in ("verdict",)]
    assert [c["name"] for c in shown] == USER_COLUMNS
    table = churn_tab.churn_table()
    assert table.hidden_columns == ["verdict"] and table.id == "churn-table"


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
    assert tips[0]["score"]["value"].split(":")[0] in ("GO", "WAIT", "NOT SCORED", "CLOSED", "STALE", "NO PRICE")
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
