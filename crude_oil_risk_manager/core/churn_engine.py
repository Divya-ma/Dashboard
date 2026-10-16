"""Churn simulation (intraday): how often a small target is reached inside your trading hours, what it earns after
costs, and how fragile that is.

THE TRADE (long; a short is the same on the negated prices). Enter at the close of a five-minute candle by CROSSING the
spread: the fill is the close plus half the spread (plus `worse` ticks when the entry is worse than planned). The target
is `target` ticks above the fill and the stop `stop` ticks below it (a multiple of the target). A later candle's high at
the target is a WIN (+target, no spread to pay, the trailing order only adds to it); its low at the stop is a STOP-OUT
(-stop and the spread to cross; if one candle holds both, the stop is assumed first). A trade open at the end of the
trading window is closed by crossing at the close. Nothing is held outside the window.

TRADING WINDOW. Your hours, in LONDON time (default 09:00 to 19:00; the candles are UTC and are converted, so summer
time is handled). Each weekday is one "day"; days when the market did not trade are left out, and so is the day still in
progress. An instrument that printed nothing on a day counts as a flat day (zero trades), not a missing one.

THE DAY LADDER, the headline figures. At the start of the window buy at the market; that fill is the LEVEL. Each win: sell
and wait up to `patience` minutes for the price to come back to the level, then buy there again. If it does not come back
in time, or after a stop-out, MOVE ON: buy at the market and take that fill as the new level. Net per day, trades per day,
hit probability, positive days and the blow-up rate come from this, day by day.

EVERY ENTRY, for the live verdict: each candle close in the window as a possible entry, independent of the others. The
hit probability and net per trade after entries made at a given hour or in a given price state.

OUT OF SAMPLE. Choosing the best target on the same days it is scored on flatters it. The first 60% of the days pick the
target; the figures that rank the contract (net per day, positive days, hit probability, worse-fill survival) come from
the last 40% only. With fewer than 15 days there is no split.

  worse fill      the same with every entry fill 1, 2, 3 ticks above the level (below it for a short)
  blow-up rate    share of days that lost more than three stop-outs' worth

ENTRY STATE. For each entry the price is compared with its average over the last day of candles: z = (close - mean) /
standard deviation, signed so that it is NEGATIVE when the price is on the good side for this trade (below the average
for a long, above it for a short). z <= -1 is "favourable" (buying a dip), z >= 1 "against" (buying strength), else
"neutral".

OSCILLATIONS (`count_oscillations`). From a given price level, how many times in the last few days did the market come
back to that level (or better for the trade) and then move the target in the trade's favour, within the same day: the
number of round trips a ladder at THIS level would have made, which is what the live price asks.

The first candle at or after a level is found for every entry at once with a range table and a binary search.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

LONDON = "Europe/London"
BLOWUP_STOPS = 3.0
WORSE_FILLS = (0, 1, 2, 3)
MIN_TRIALS = 50
MIN_DAYS = 5
SPLIT_MIN_DAYS = 15
TRAIN_SHARE = 0.6
STATES = ("favourable", "neutral", "against")
STATE_Z = 1.0  # |z| at which a state starts
BIG = 1 << 60


# ----------------------------------------------------------------------
# Range tables
# ----------------------------------------------------------------------


def _table(values: np.ndarray, op) -> list[np.ndarray]:
    table = [values]
    span = 1
    while 2 * span <= len(values):
        previous = table[-1]
        table.append(op(previous[:-span], previous[span:]))
        span *= 2
    return table


def _query(table: list[np.ndarray], low: np.ndarray, high: np.ndarray, op) -> np.ndarray:
    """op over values[low..high] inclusive, for arrays of ranges (high >= low)."""
    length = high - low + 1
    level = np.floor(np.log2(length)).astype(int)
    out = np.empty(len(low))
    for k in np.unique(level):
        pick = level == k
        a = table[k][low[pick]]
        b = table[k][high[pick] - (1 << int(k)) + 1]
        out[pick] = op(a, b)
    return out


def _first_cross(table: list[np.ndarray], first: np.ndarray, stop: np.ndarray, level: np.ndarray, upward: bool) -> np.ndarray:
    """For each range, the first index in first..stop where the value reaches `level` (>= if upward, else <=); BIG if never."""
    op = np.maximum if upward else np.minimum
    reached = (_query(table, first, stop, op) >= level) if upward else (_query(table, first, stop, op) <= level)
    lo, hi = first.copy(), stop.copy()
    while True:
        active = reached & (lo < hi)
        if not active.any():
            break
        mid = (lo + hi) // 2
        agg = _query(table, first[active], mid[active], op)
        ok = np.zeros(len(first), dtype=bool)
        ok[active] = (agg >= level[active]) if upward else (agg <= level[active])
        hi = np.where(active & ok, mid, hi)
        lo = np.where(active & ~ok, mid + 1, lo)
    return np.where(reached, lo, BIG)


# ----------------------------------------------------------------------
# Trading days
# ----------------------------------------------------------------------


def london(ts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC candle times as London local times."""
    return ts.tz_localize("UTC").tz_convert(LONDON)


def _hours(local: pd.DatetimeIndex) -> np.ndarray:
    return (local.hour + local.minute / 60.0).to_numpy()


def trading_calendar(ts: pd.DatetimeIndex, start_hour: float, end_hour: float) -> np.ndarray:
    """The weekdays (numpy datetime64[D], ascending) on which the market traded inside the window, from the candle times of
    a whole product. A day with far fewer candles than usual (a holiday) is left out, and so is the day still in progress."""
    if len(ts) == 0:
        return np.array([], dtype="datetime64[D]")
    local = london(ts)
    hour = _hours(local)
    keep = (hour >= start_hour) & (hour < end_hour)
    days = pd.Series(local[keep].tz_localize(None).normalize())
    counts = days.value_counts()
    if counts.empty:
        return np.array([], dtype="datetime64[D]")
    typical = counts.median()
    good = sorted(d for d, n in counts.items() if n >= 0.4 * typical and d.weekday() < 5)
    last = local.max()
    if good and good[-1] == last.tz_localize(None).normalize() and last.hour + last.minute / 60.0 < end_hour - 0.5:
        good = good[:-1]  # today, still in progress
    return np.array(good, dtype="datetime64[ns]").astype("datetime64[D]")


@dataclass
class Context:
    """One instrument and direction, prepared for simulating many targets."""

    ts: pd.DatetimeIndex  # UTC times of the candles inside the window
    t_ns: np.ndarray
    hour: np.ndarray  # London hour of each candle
    high: np.ndarray  # directional, price units
    low: np.ndarray
    close: np.ndarray
    tick: float
    tables: tuple
    start: np.ndarray  # per calendar day: index of its first candle
    end: np.ndarray  # per calendar day: index of its last candle (end < start: no candle that day)
    entries: np.ndarray  # candles that can be entered (a later candle the same day)
    stop: np.ndarray  # for each entry, the last candle of its day
    z: np.ndarray
    bars_per_day: float


def prepare(bars: pd.DataFrame, tick: float, direction: str, start_hour: float, end_hour: float, calendar: np.ndarray) -> Context | None:
    """Candles of one instrument inside the window, in the frame of one direction (a short is a long on the negated prices)."""
    if len(calendar) < MIN_DAYS or len(bars) < 100:
        return None
    ts_all = pd.DatetimeIndex(bars["ts"])
    local = london(ts_all)
    hour = _hours(local)
    day = local.tz_localize(None).normalize().values.astype("datetime64[D]")
    code = np.searchsorted(calendar, day)
    in_calendar = (code < len(calendar)) & (calendar[np.minimum(code, len(calendar) - 1)] == day)
    keep = (hour >= start_hour) & (hour < end_hour) & in_calendar
    if keep.sum() < MIN_TRIALS:
        return None
    code, hour, ts = code[keep], hour[keep], ts_all[keep]
    high, low, close = (bars[c].to_numpy(float)[keep] for c in ("high", "low", "close"))
    if direction == "short":
        high, low, close = -low, -high, -close
    days = np.arange(len(calendar))
    start = np.searchsorted(code, days, "left")
    end = np.searchsorted(code, days, "right") - 1
    index = np.arange(len(ts))
    last_of_day = end[code]
    entries = index[last_of_day > index]
    if len(entries) < MIN_TRIALS:
        return None
    counts = (end - start + 1)[end >= start]
    per_day = float(np.median(counts))
    tables = (_table(high / tick, np.maximum), _table(low / tick, np.minimum))
    return Context(ts, ts.as_unit("ns").asi8, hour, high, low, close, tick, tables, start, end, entries, last_of_day[entries],
                   entry_z(close / tick, max(int(round(per_day)), 20)), per_day)


@dataclass
class Trials:
    entry: np.ndarray  # index of the entry candle
    hit: np.ndarray  # bool: the target was reached first
    stopped: np.ndarray  # bool: the stop was reached first
    end: np.ndarray
    pnl_ticks: np.ndarray  # price result per lot in ticks, before transaction costs and rebates


def simulate(ctx: Context, target: float, stop_ticks: float, half_spread: float, worse: float) -> Trials:
    """Long trades entered at every usable candle close: ended at the target, the stop, or the end of that candle's day."""
    c = ctx.close / ctx.tick
    max_table, min_table = ctx.tables
    entries, last = ctx.entries, ctx.stop
    fill = c[entries] + half_spread + worse
    first = entries + 1
    when_hit = _first_cross(max_table, first, last, fill + target, True)
    when_stop = _first_cross(min_table, first, last, fill - stop_ticks, False)
    stopped = when_stop <= when_hit  # a candle with both: the stop first
    stopped &= when_stop < BIG
    hit = ~stopped & (when_hit < BIG)
    end = np.where(hit, when_hit, np.where(stopped, when_stop, last))
    pnl = np.where(hit, float(target), np.where(stopped, -stop_ticks - half_spread, c[end] - half_spread - fill))
    return Trials(entries, hit, stopped, end, pnl)


def state_of(z: float) -> str:
    """The entry state of a (direction-signed) z-score."""
    return STATES[0] if z <= -STATE_Z else STATES[2] if z >= STATE_Z else STATES[1]


def entry_z(close: np.ndarray, window: int) -> np.ndarray:
    """z of every candle's close against the mean and standard deviation of the last `window` closes (NaN at first)."""
    series = pd.Series(close)
    roll = series.rolling(window, min_periods=max(window // 2, 10))
    std = roll.std().replace(0, np.nan)
    return ((series - roll.mean()) / std).to_numpy()


# ----------------------------------------------------------------------
# The day ladder
# ----------------------------------------------------------------------


@dataclass
class DayArrays:
    """Per calendar day, what the day ladder did."""

    total: np.ndarray  # net $ after cost
    trades: np.ndarray
    wins: np.ndarray
    stops: np.ndarray


def day_ladder(ctx: Context, target: float, stop_ticks: float, patience_minutes: float, half_spread: float, worse: float, tick_value: float, cost_rt: float) -> DayArrays:
    """The day ladder (see the module docstring) for every calendar day."""
    h, l, c = ctx.high / ctx.tick, ctx.low / ctx.tick, ctx.close / ctx.tick
    t = ctx.t_ns
    patience = int(patience_minutes * 60e9)
    n_days = len(ctx.start)
    total, trades, wins, stops = np.zeros(n_days), np.zeros(n_days, dtype=int), np.zeros(n_days, dtype=int), np.zeros(n_days, dtype=int)
    win_value = target * tick_value - cost_rt
    stop_value = (-stop_ticks - half_spread) * tick_value - cost_rt
    for d in range(n_days):
        s, e = int(ctx.start[d]), int(ctx.end[d])
        if e <= s:  # no candle, or only one: nothing can be traded
            continue
        i = s
        level = c[i] + half_spread
        fill = level + worse
        value, count, won, stopped = 0.0, 0, 0, 0
        while True:  # i < e here: the position was opened on candle i, with at least one candle left
            count += 1
            hit = np.flatnonzero(h[i + 1:e + 1] >= fill + target)
            stop = np.flatnonzero(l[i + 1:e + 1] <= fill - stop_ticks)
            jt = hit[0] if len(hit) else BIG
            js = stop[0] if len(stop) else BIG
            if jt == BIG and js == BIG:  # neither before the window ends: close by crossing at the last close
                value += (c[e] - half_spread - fill) * tick_value - cost_rt
                break
            if js <= jt:  # stopped out (the stop first if one candle holds both): move on to a new price at that candle's close
                value += stop_value
                stopped += 1
                i = i + 1 + int(js)
                if i >= e:
                    break
                level = c[i] + half_spread
                fill = level + worse
                continue
            j = i + 1 + int(jt)
            value += win_value
            won += 1
            if j >= e:
                break
            back = np.flatnonzero(l[j + 1:e] <= level)  # the price comes back to the level, before the last candle
            if len(back) and t[j + 1 + back[0]] - t[j] <= patience:
                i = j + 1 + int(back[0])
                fill = level + worse
                continue
            beyond = np.flatnonzero(t[j + 1:e + 1] > t[j] + patience)  # it did not come back in time: move on at the end of the wait
            if not len(beyond):  # the window ends before the wait does: stay flat
                break
            i = j + int(beyond[0])
            level = c[i] + half_spread
            fill = level + worse
        total[d], trades[d], wins[d], stops[d] = value, count, won, stopped
    return DayArrays(total, trades, wins, stops)


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------


@dataclass
class Result:
    target: int
    worse: int
    days: int
    trades: int
    hit_prob: float  # wins / trades of the ladder
    trips_per_day: float
    wins_per_day: float
    stops_per_day: float
    net_trade: float
    net_day: float
    pos_days: float
    worst_day: float
    blowup: float  # share of days that lost more than BLOWUP_STOPS stop-outs' worth


def result_of(arrays: DayArrays, days: slice, target: int, worse: int, stop_loss_usd: float) -> Result | None:
    total, trades, wins, stops = (a[days] for a in (arrays.total, arrays.trades, arrays.wins, arrays.stops))
    if len(total) == 0 or trades.sum() == 0:
        return None
    return Result(
        target, worse, int(len(total)), int(trades.sum()), float(wins.sum() / trades.sum()), float(trades.mean()), float(wins.mean()), float(stops.mean()),
        float(total.sum() / trades.sum()), float(total.mean()), float((total > 0).mean()), float(total.min()), float((total < -BLOWUP_STOPS * stop_loss_usd).mean()),
    )


@dataclass
class Entry:
    """Statistics of every entry (independent of the ladder), used by the live verdict and the best-hour chart."""

    entry_hit_prob: float
    net_trade: float
    trials: int
    hour: pd.DataFrame  # by London entry hour: trials, hit_prob, net_trade
    states: pd.DataFrame  # by entry state: trials, hit_prob, net_trade


def entry_stats(ctx: Context, trials: Trials, tick_value: float, cost_rt: float) -> Entry:
    net = trials.pnl_ticks * tick_value - cost_rt
    hour = np.floor(ctx.hour[trials.entry]).astype(int)
    hours = pd.DataFrame({"hour": hour, "hit": trials.hit, "net": net}).groupby("hour").agg(trials=("hit", "size"), hit_prob=("hit", "mean"), net_trade=("net", "mean"))
    zs = ctx.z[trials.entry]
    known = np.isfinite(zs)
    names = np.where(zs <= -STATE_Z, STATES[0], np.where(zs >= STATE_Z, STATES[2], STATES[1]))
    frame = pd.DataFrame({"state": names[known], "hit": trials.hit[known], "net": net[known]})
    states = frame.groupby("state").agg(trials=("hit", "size"), hit_prob=("hit", "mean"), net_trade=("net", "mean")).reindex(list(STATES))
    return Entry(float(trials.hit.mean()), float(net.mean()), int(len(net)), hours, states)


@dataclass
class InstrumentSpec:
    symbol: str
    product: str
    family: str
    label: str
    tick: float
    tick_value: float
    cost_rt: float
    half_spread: float  # ticks
    atr: float | None  # daily ATR in points, for display only


@dataclass
class Path:
    """What the oscillation count needs: the candles inside the window (not signed for a direction) and where each day starts and ends."""

    high: np.ndarray
    low: np.ndarray
    start: np.ndarray
    end: np.ndarray


@dataclass
class Settings:
    """Everything the simulation depends on besides the candles and the costs."""

    start_hour: float = 9.0
    end_hour: float = 19.0
    stop_mult: float = 3.0
    patience_minutes: float = 60.0


def analyse(bars: pd.DataFrame, spec: InstrumentSpec, direction: str, targets: list[int], margin: float, settings: Settings, calendar: np.ndarray) -> dict | None:
    """One instrument and direction: every allowed target on all days (for the charts), the best target chosen on the first
    days, and its figures on the last days (with the worse fills). None when there is too little data."""
    ctx = prepare(bars, spec.tick, direction, settings.start_hour, settings.end_hour, calendar)
    if ctx is None:
        return None
    allowed = [t for t in targets if t * spec.tick_value > spec.cost_rt + margin]
    if not allowed:
        return None

    def ladder(t: int, k: int) -> DayArrays:
        return day_ladder(ctx, t, settings.stop_mult * t, settings.patience_minutes, spec.half_spread, k, spec.tick_value, spec.cost_rt)

    def stop_loss(t: int) -> float:
        return (settings.stop_mult * t + spec.half_spread) * spec.tick_value + spec.cost_rt

    n_days = len(calendar)
    split = int(round(n_days * TRAIN_SHARE)) if n_days >= SPLIT_MIN_DAYS else 0
    train, test = slice(0, split) if split else slice(0, n_days), slice(split, n_days)
    runs = {t: ladder(t, 0) for t in allowed}
    grid = {t: result_of(runs[t], slice(0, n_days), t, 0, stop_loss(t)) for t in allowed}
    grid = {t: r for t, r in grid.items() if r is not None}
    if not grid:
        return None
    best = max(grid, key=lambda t: float(runs[t].total[train].mean()))
    worse = {0: result_of(runs[best], test, best, 0, stop_loss(best))}
    for k in WORSE_FILLS[1:]:
        worse[k] = result_of(ladder(best, k), test, best, k, stop_loss(best))
    if worse[0] is None:
        return None
    worse = {k: r for k, r in worse.items() if r is not None}
    trials = simulate(ctx, best, settings.stop_mult * best, spec.half_spread, 0)
    mask = _window_mask(bars, settings, calendar)
    volume = bars["volume"].fillna(0).to_numpy(float)[mask]
    path = Path(bars["high"].to_numpy(float)[mask], bars["low"].to_numpy(float)[mask], ctx.start, ctx.end)
    return {
        "spec": spec, "direction": direction, "best": best, "grid": grid, "worse": worse, "insample": result_of(runs[best], train, best, 0, stop_loss(best)),
        "entry": entry_stats(ctx, trials, spec.tick_value, spec.cost_rt), "bars_per_day": ctx.bars_per_day, "path": path, "volume_day": float(volume.sum()) / len(calendar),
        "active": float((volume > 0).mean()), "last_ts": ctx.ts[-1], "stop_loss_usd": stop_loss(best), "test_days": int(test.stop - test.start), "split": bool(split),
    }


def _window_mask(bars: pd.DataFrame, settings: Settings, calendar: np.ndarray) -> np.ndarray:
    local = london(pd.DatetimeIndex(bars["ts"]))
    hour = _hours(local)
    day = local.tz_localize(None).normalize().values.astype("datetime64[D]")
    code = np.searchsorted(calendar, day)
    in_calendar = (code < len(calendar)) & (calendar[np.minimum(code, len(calendar) - 1)] == day)
    return (hour >= settings.start_hour) & (hour < settings.end_hour) & in_calendar


def count_oscillations(path: Path, direction: str, level: float, target_ticks: float, tick: float, days: int) -> int:
    """How many times, in the last `days` trading days, price came back to `level` (or better for the trade) and then
    moved `target_ticks` in the trade's favour, within the same day. A long arms when a candle's low is at or below the
    level and counts when a LATER candle's high reaches level + target; after a count it must arm again. A short mirrors it."""
    count = 0
    for s, e in zip(path.start[-days:], path.end[-days:]):
        if e < s:
            continue
        high, low = path.high[s:e + 1], path.low[s:e + 1]
        if direction == "long":
            arm, hit = np.flatnonzero(low <= level + 1e-9), np.flatnonzero(high >= level + target_ticks * tick - 1e-9)
        else:
            arm, hit = np.flatnonzero(high >= level - 1e-9), np.flatnonzero(low <= level - target_ticks * tick + 1e-9)
        position = -1
        while True:
            a = np.searchsorted(arm, position, side="right")
            if a >= len(arm):
                break
            b = np.searchsorted(hit, arm[a], side="right")
            if b >= len(hit):
                break
            count += 1
            position = hit[b]
    return count


def london_now_hour(now_utc) -> float:
    """The current London hour (with minutes as a fraction), for 'is it inside my trading window'."""
    local = pd.Timestamp(now_utc).tz_convert(LONDON)
    return local.hour + local.minute / 60.0
