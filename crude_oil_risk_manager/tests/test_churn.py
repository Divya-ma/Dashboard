"""Churn page: costs on net lots, the candle store, the intraday simulation in your trading hours (against plain loops),
choosing the target out of sample, and oscillations."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core import churn_costs as cc
from core.churn_data import ChurnStore, tracked_instruments
from core.churn_engine import InstrumentSpec, Path, Settings, analyse, count_oscillations, day_ladder, prepare, simulate, trading_calendar

TODAY = date(2026, 10, 14)
COSTS = dict(cc.DEFAULT_COSTS)


# ----------------------------------------------------------------------
# Costs
# ----------------------------------------------------------------------


def test_costs_are_charged_on_net_lots_with_the_given_rates():
    assert cc.structure_side_cost("CL", "outright", COSTS) == pytest.approx(0.69)
    assert cc.round_trip_cost("CL", "outright", COSTS) == pytest.approx(1.38)
    assert cc.round_trip_cost("CL", "spread", COSTS) == pytest.approx(2 * 2 * 0.69)  # two lots
    assert cc.round_trip_cost("BRN", "fly", COSTS) == pytest.approx(2 * 4 * 0.70)  # +1 -2 +1 = four lots
    assert cc.round_trip_cost("CL", "dfly", COSTS) == pytest.approx(2 * 8 * 0.69)
    assert COSTS["BZ"].net_per_side == pytest.approx(0.12)
    assert COSTS["WBS"].net_per_side == pytest.approx(0.32)
    assert COSTS["G"].net_per_side == pytest.approx(0.80) and COSTS["G"].tick_size == 0.25


def test_offsetting_months_net_down_and_combo_rebates_apply():
    net = cc.net_lots([("CL", "Jan", 1), ("CL", "Feb", -1), ("CL", "Feb", 1), ("CL", "Mar", -1)])  # Feb nets to zero
    assert cc.side_cost(net, COSTS) == pytest.approx(2 * 0.69)
    pair = cc.net_lots([("CL", "Jan", 1), ("BZ", "Jan", -1)])
    assert cc.side_cost(pair, COSTS) == pytest.approx(0.69 + 0.12)
    assert cc.side_cost(pair, COSTS, combo=True) == pytest.approx((0.89 - 0.87) + (0.89 - 0.81))


def test_breakeven_ticks_and_validation():
    assert cc.breakeven_ticks("CL", "outright", COSTS) == pytest.approx(0.138)
    assert cc.validate_costs({"CL": {"tc": "1.5"}})["CL"].tc == 1.5
    with pytest.raises(ValueError):
        cc.validate_costs({"CL": {"tick_value": 0}})
    with pytest.raises(ValueError):
        cc.validate_costs({"CL": {"tc": "abc"}})


# ----------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------


class FakeApi:
    def __init__(self):
        self.calls = []

    def ohlc(self, interval, instruments=None, start=None, end=None, **_):
        self.calls.append((interval, list(instruments), start, end))
        rows = []
        t = start
        while t < end:
            for symbol in instruments:
                rows.append({"product": symbol, "time": t * 1000, "open": 70.0, "high": 70.1, "low": 69.9, "close": 70.0, "volume": 5})
            t += 3600 * 6
        return rows


def test_tracked_instruments_are_live_and_front_first():
    out = tracked_instruments("CL", TODAY, 4)
    assert [i.family for i in out] == ["outright"] * 4 + ["spread"] * 4 + ["fly"] * 4
    assert out[0].label == "Nov26" and all(i.expiry > TODAY for i in out)


def test_store_downloads_in_small_windows_and_only_new_days_the_next_day(tmp_path):
    store = ChurnStore(tmp_path, clock=lambda: datetime(2026, 10, 14, 12, tzinfo=timezone.utc))
    api = FakeApi()
    report = store.update(api, 6, 3, TODAY)
    assert report["requests"] == len(api.calls) > 2 and all(call[0] == "5M" and len(call[1]) <= 8 for call in api.calls)
    assert store.is_current(TODAY, 6, 3) and set(store.frame("CL")["symbol"]) == {i.symbol for i in tracked_instruments("CL", TODAY, 3)}
    assert store.frame("CL")["ts"].min() >= pd.Timestamp(TODAY - timedelta(days=6))
    tomorrow = TODAY + timedelta(days=1)
    api2 = FakeApi()
    ChurnStore(tmp_path, clock=lambda: datetime(2026, 10, 15, 12, tzinfo=timezone.utc)).update(api2, 6, 3, tomorrow)
    assert api2.calls and min(c[2] for c in api2.calls) >= int(datetime(2026, 10, 12, tzinfo=timezone.utc).timestamp())  # two days back, not the window


# ----------------------------------------------------------------------
# Trading hours
# ----------------------------------------------------------------------

BARS_PER_DAY = 288
IN_WINDOW = 120  # 09:00-19:00 London in summer is 08:00-18:00 UTC: ten hours of five-minute candles


def make_bars(days=14, seed=2, step=0.02, volume=50.0, start="2026-08-03 00:00", close=None):
    """Five-minute candles around the clock (weekends too) from a Monday, on a random walk, or on the given closes."""
    rng = np.random.default_rng(seed)
    n = days * BARS_PER_DAY
    if close is None:
        close = 70 + np.cumsum(rng.normal(0, step, n))
    close = np.asarray(close, float)
    high, low = close + np.abs(rng.normal(0, 0.01, n)), close - np.abs(rng.normal(0, 0.01, n))
    return pd.DataFrame({"ts": pd.date_range(start, periods=n, freq="5min"), "open": close, "high": high, "low": low, "close": close, "volume": volume})


def make_spec(**kw):
    base = dict(symbol="CLX26", product="CL", family="outright", label="Nov26", tick=0.01, tick_value=10.0, cost_rt=1.38, half_spread=0.5, atr=1.0)
    base.update(kw)
    return InstrumentSpec(**base)


def calendar_of(bars, settings=Settings()):
    return trading_calendar(pd.DatetimeIndex(bars["ts"]), settings.start_hour, settings.end_hour)


def test_the_calendar_has_the_weekdays_the_market_traded_in_your_hours():
    bars = make_bars(days=14)  # two whole weeks from Monday 3 August
    calendar = calendar_of(bars)
    assert len(calendar) == 10 and str(calendar[0]) == "2026-08-03" and str(calendar[-1]) == "2026-08-14"
    assert all(pd.Timestamp(d).weekday() < 5 for d in calendar)
    cut = bars[bars["ts"] <= "2026-08-14 12:00"]  # Friday the 14th is still going on
    assert len(calendar_of(cut)) == 9
    quiet = bars[~((bars["ts"] >= "2026-08-05") & (bars["ts"] < "2026-08-06") & (bars.index % 10 != 0))]  # a holiday: one candle in ten
    assert len(calendar_of(quiet)) == 9


def test_london_summer_and_winter_time_are_both_handled():
    bars = make_bars()
    summer = prepare(bars, 0.01, "long", 9, 19, calendar_of(bars))
    assert summer.hour.min() == 9.0 and summer.hour.max() < 19.0 and summer.ts.hour.min() == 8  # 09:00 London is 08:00 UTC in August
    winter_bars = make_bars(start="2026-12-07 00:00")
    winter = prepare(winter_bars, 0.01, "long", 9, 19, calendar_of(winter_bars))
    assert winter.hour.min() == 9.0 and winter.ts.hour.min() == 9  # and 09:00 UTC in December
    assert winter.bars_per_day == IN_WINDOW


def test_only_candles_inside_the_window_are_used_and_days_with_nothing_are_flat():
    bars = make_bars()
    ctx = prepare(bars, 0.01, "long", 9, 19, calendar_of(bars))
    assert len(ctx.ts) == 10 * IN_WINDOW and (ctx.end >= ctx.start).all()
    sparse = bars[~((bars["ts"] >= "2026-08-06") & (bars["ts"] < "2026-08-07"))]  # the instrument printed nothing on the 6th
    ctx = prepare(sparse, 0.01, "long", 9, 19, calendar_of(bars))
    empty = np.flatnonzero(ctx.end < ctx.start)
    assert len(empty) == 1 and str(calendar_of(bars)[empty[0]]) == "2026-08-06"
    arrays = day_ladder(ctx, 2, 6, 60, 0.5, 0, 10.0, 1.38)
    assert arrays.trades[empty[0]] == 0 and arrays.total[empty[0]] == 0  # a flat day, counted
    assert prepare(make_bars(days=4), 0.01, "long", 9, 19, calendar_of(make_bars(days=4))) is None  # fewer than five days


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------


def _entries_brute(ctx, T, S, hs, k):
    h, l, c = ctx.high / ctx.tick, ctx.low / ctx.tick, ctx.close / ctx.tick
    out = []
    for i, last in zip(ctx.entries, ctx.stop):
        fill = c[i] + hs + k
        outcome, end = "time", last
        for j in range(i + 1, last + 1):
            if l[j] <= fill - S:
                outcome, end = "stop", j
                break
            if h[j] >= fill + T:
                outcome, end = "hit", j
                break
        pnl = T if outcome == "hit" else (-S - hs if outcome == "stop" else c[end] - hs - fill)
        out.append((outcome == "hit", outcome == "stop", end, pnl))
    return np.array(out, float)


@pytest.mark.parametrize("direction", ["long", "short"])
@pytest.mark.parametrize("case", [(3, 9, 0.5, 0), (5, 10, 0.5, 2), (1, 2, 0.0, 0)])
def test_every_entry_matches_a_plain_loop(direction, case):
    T, S, hs, k = case
    bars = make_bars()
    ctx = prepare(bars, 0.01, direction, 9, 19, calendar_of(bars))
    result = simulate(ctx, T, S, hs, k)
    expected = _entries_brute(ctx, T, S, hs, k)
    assert np.array_equal(result.hit, expected[:, 0].astype(bool)) and np.array_equal(result.stopped, expected[:, 1].astype(bool))
    assert np.array_equal(result.end, expected[:, 2]) and np.allclose(result.pnl_ticks, expected[:, 3])
    assert (result.end <= ctx.stop).all() and result.stopped.sum() > 0 and result.hit.sum() > 0  # the cases exercise both


def _ladder_brute(ctx, T, S, patience_min, hs, k, tick_value, cost):
    """The day ladder candle by candle: buy at the open, sell on the target, wait for the level, move on after a stop-out or a long wait."""
    h, l, c, t = ctx.high / ctx.tick, ctx.low / ctx.tick, ctx.close / ctx.tick, ctx.t_ns
    patience = int(patience_min * 60e9)
    out = []
    for s, e in zip(ctx.start, ctx.end):
        if e <= s:
            out.append((0.0, 0, 0, 0))
            continue
        level = c[s] + hs
        fill = level + k
        total, trades, wins, stops, holding, hit_bar, j = 0.0, 1, 0, 0, True, None, s + 1
        while j <= e:
            if holding:
                if l[j] <= fill - S:
                    total += (-S - hs) * tick_value - cost
                    stops += 1
                    if j >= e:
                        holding = False
                        break
                    level, fill = c[j] + hs, c[j] + hs + k
                    trades += 1
                elif h[j] >= fill + T:
                    total += T * tick_value - cost
                    wins += 1
                    holding, hit_bar = False, j
                    if j >= e:
                        break
                j += 1
            elif t[j] - t[hit_bar] > patience:  # the wait is over: a new position at the last candle inside it, which then sees candle j
                m = j - 1
                level, fill = c[m] + hs, c[m] + hs + k
                trades += 1
                holding = True
            else:
                if l[j] <= level and j < e:
                    holding, fill = True, level + k
                    trades += 1
                j += 1
        if holding:
            total += (c[e] - hs - fill) * tick_value - cost
        out.append((total, trades, wins, stops))
    return np.array(out, float)


@pytest.mark.parametrize("direction", ["long", "short"])
@pytest.mark.parametrize("case", [(2, 6, 30, 0.5, 0), (3, 9, 60, 0.5, 1), (1, 3, 15, 0.0, 0), (4, 8, 240, 0.5, 2)])
def test_the_day_ladder_matches_a_candle_by_candle_loop(direction, case):
    T, S, patience, hs, k = case
    bars = make_bars(seed=4, step=0.015)
    ctx = prepare(bars, 0.01, direction, 9, 19, calendar_of(bars))
    arrays = day_ladder(ctx, T, S, patience, hs, k, 10.0, 1.38)
    expected = _ladder_brute(ctx, T, S, patience, hs, k, 10.0, 1.38)
    assert np.allclose(arrays.total, expected[:, 0]) and np.array_equal(arrays.trades, expected[:, 1])
    assert np.array_equal(arrays.wins, expected[:, 2]) and np.array_equal(arrays.stops, expected[:, 3])
    assert (arrays.trades >= 1).all() and (arrays.wins <= arrays.trades).all() and arrays.stops.sum() > 0 and arrays.wins.sum() > 0


def test_the_stop_and_the_patience_change_what_the_ladder_does():
    bars = make_bars(seed=4, step=0.015)
    ctx = prepare(bars, 0.01, "long", 9, 19, calendar_of(bars))
    tight = day_ladder(ctx, 3, 3, 60, 0.5, 0, 10.0, 1.38)
    wide = day_ladder(ctx, 3, 30, 60, 0.5, 0, 10.0, 1.38)
    assert tight.stops.sum() > wide.stops.sum()  # a close stop is hit more often
    impatient = day_ladder(ctx, 3, 9, 0, 0.5, 0, 10.0, 1.38)
    patient = day_ladder(ctx, 3, 9, 600, 0.5, 0, 10.0, 1.38)
    assert impatient.trades.sum() != patient.trades.sum()


# ----------------------------------------------------------------------
# Choosing the target out of sample, and what the figures mean
# ----------------------------------------------------------------------


def test_the_target_is_chosen_on_the_early_days_and_scored_on_the_late_ones():
    bars = make_bars(days=35)  # 25 weekdays
    result = analyse(bars, make_spec(), "long", [1, 2, 3, 4], 3.0, Settings(), calendar_of(bars))
    assert result["split"] and result["worse"][0].days == 10 and result["test_days"] == 10  # the last 40% of 25 days
    assert result["insample"].days == 15 and result["best"] in result["grid"]
    assert result["grid"][result["best"]].days == 25  # the charts use every day
    short = make_bars(days=14)
    result = analyse(short, make_spec(), "long", [1, 2, 3, 4], 3.0, Settings(), calendar_of(short))
    assert not result["split"] and result["worse"][0].days == 10  # fewer than 15 days: no split


def test_a_worse_fill_cannot_help_and_the_figures_are_consistent():
    bars = make_bars(days=35)
    result = analyse(bars, make_spec(), "long", [1, 2, 3, 4], 3.0, Settings(), calendar_of(bars))
    nets = [result["worse"][k].net_day for k in range(4)]
    assert nets == sorted(nets, reverse=True)
    base = result["worse"][0]
    assert base.net_day == pytest.approx(base.trips_per_day * base.net_trade) and 0 <= base.hit_prob <= 1 and 0 <= base.blowup <= 1
    assert result["volume_day"] == pytest.approx(50.0 * IN_WINDOW) and result["stop_loss_usd"] == pytest.approx((3 * result["best"] + 0.5) * 10 + 1.38)
    entry = result["entry"]
    assert entry.trials > 1000 and 0 < entry.entry_hit_prob < 1 and entry.hour.index.min() == 9 and entry.hour.index.max() == 18


def test_targets_that_do_not_cover_the_cost_plus_margin_are_not_tried_and_short_data_gives_nothing():
    bars = make_bars()
    assert analyse(bars, make_spec(cost_rt=25.0), "long", [1, 2], 3.0, Settings(), calendar_of(bars)) is None  # $10 and $20 targets below $28
    few = make_bars(days=4)
    assert analyse(few, make_spec(), "long", [1, 2, 3], 3.0, Settings(), calendar_of(few)) is None


def test_a_market_that_keeps_coming_back_to_its_level_earns_and_a_drift_does_not():
    n = 14 * BARS_PER_DAY
    wave = 70 + 0.06 * ((np.arange(n) // 12) % 2)  # 70.00 for an hour, 70.06 for an hour; the window opens at the low
    drift = 70 - np.arange(n) * 0.0004  # 0.048 down over the window: no target, no stop, a forced exit every day
    out = {}
    for name, close in (("wave", wave), ("drift", drift)):
        bars = make_bars(close=close)
        out[name] = analyse(bars, make_spec(), "long", [3], 3.0, Settings(patience_minutes=90), calendar_of(bars))["worse"][0]
    assert out["wave"].net_day == pytest.approx(5 * (3 * 10 - 1.38)) and out["wave"].trips_per_day == 5 and out["wave"].pos_days == 1.0 and out["wave"].stops_per_day == 0  # one round trip every two hours
    assert out["drift"].net_day < 0 and out["drift"].trips_per_day == 1 and out["drift"].hit_prob == 0 and out["drift"].pos_days == 0


# ----------------------------------------------------------------------
# Oscillations
# ----------------------------------------------------------------------


def _path(highs, lows, per_day=None):
    n = len(highs)
    per_day = per_day or n
    starts = np.arange(0, n, per_day)
    return Path(np.array(highs, float), np.array(lows, float), starts, np.minimum(starts + per_day - 1, n - 1))


def test_an_oscillation_is_a_return_to_the_level_followed_by_a_move_of_the_target():
    # level 70.00, target 2 ticks = 0.02: dip to 70.00, rally to 70.02, dip again, rally again, then dip with no rally
    lows = [70.01, 70.00, 70.01, 70.01, 70.00, 70.01, 70.00]
    highs = [70.02, 70.01, 70.02, 70.02, 70.01, 70.02, 70.01]
    assert count_oscillations(_path(highs, lows), "long", 70.00, 2, 0.01, 1) == 2
    # a candle that touches the level and the target together does not count: the order inside a candle is unknown
    assert count_oscillations(_path([70.03], [69.99]), "long", 70.00, 2, 0.01, 1) == 0
    # after a count the price must come back to the level before the next one counts
    assert count_oscillations(_path([70.01, 70.02, 70.03, 70.04], [70.00, 70.01, 70.02, 70.03]), "long", 70.00, 2, 0.01, 1) == 1


def test_a_short_mirrors_a_long_and_only_the_chosen_days_are_counted():
    highs, lows = [70.00, 69.99, 70.00, 69.99], [69.99, 69.98, 69.99, 69.98]
    assert count_oscillations(_path(highs, lows), "short", 70.00, 1, 0.01, 1) == 2
    two_days = _path([70.01, 70.02, 70.01, 70.02], [70.00, 70.01, 70.00, 70.01], per_day=2)  # one oscillation each day
    assert count_oscillations(two_days, "long", 70.00, 2, 0.01, 2) == 2 and count_oscillations(two_days, "long", 70.00, 2, 0.01, 1) == 1


def test_an_oscillation_cannot_start_on_one_day_and_finish_on_the_next():
    two_days = _path([70.00, 70.00, 70.02, 70.02], [70.00, 70.00, 70.01, 70.01], per_day=2)  # arms on day one, the rally is on day two
    assert count_oscillations(two_days, "long", 70.00, 2, 0.01, 2) == 0


def test_a_day_with_no_candles_is_skipped():
    path = Path(np.array([70.02, 70.02]), np.array([70.00, 70.00]), np.array([0, 2]), np.array([1, 1]))  # the second day: end < start
    assert count_oscillations(path, "long", 70.00, 1, 0.01, 2) == 1
