"""The Churn page's engine: keeps the five-minute candles fresh, runs the intraday simulation of every live instrument
and ranks them by churn score, with the live price and the oscillation count of each in the same table.

`ensure_fresh()` is called by the page while it is open (it starts a background download when the data was last
refreshed more than a few minutes ago, or the page's window or number of contracts grew). `results()` builds, and caches
until the stored data, the simulation settings or the cost table change, one row per instrument and direction.
`live_table()` is cheap: it scores those rows (among the contracts from the chosen one onwards) and adds the live price.

INTRADAY, IN YOUR HOURS. Trades are taken only inside the trading window (London time, default 09:00-19:00) and closed
by its end (see core.churn_engine). There is no holding limit in days and no ATR rule; the daily ATR is only shown.
ALL front contracts are downloaded and simulated; `start_from` only decides which ones the page shows and scores.

CHURN SCORE (-100 to +100, never 0). All figures are OUT OF SAMPLE, from rolling blocks (see core.churn_engine).
A row that MAKES money scores +1 to +100: the geometric mean of five 0-1 parts times 100, so a weak part pulls the whole
score down (no part is taken below 0.05, so it cannot erase it). Every part is on a fixed scale, so a row's score does not
change with which other rows are shown:
  edge        how sure the profit is: tanh(t / 2), t being the net $ per day over its standard error
  hit         share of the day ladder's trades that reached their target
  worse fill  how much of the net per day survives an entry 1 and 2 ticks worse (average of the two shares)
  stability   share of days with a positive net
  tail        1 - blow-up rate (days that lost more than three stop-outs' worth)
A row that LOSES money scores below zero, and the size says what to do about it:
  the other side of the same contract makes money   minus that side's score: near -100 means FLIP, churn it the other way
  the other side loses too                          -1 to -30 by how surely this side loses: AVOID, neither side works
A row whose volume per day is below `min_volume` is scored like the others and marked thin.

RANGE COLUMNS. The low and the high of the last 7 and the last 30 calendar days, from the stored five-minute candles (all
hours, not only yours). When the stored window is shorter than that, the range covers what is stored and the tooltip says so.

D-FLIES are simulated too, from candles derived out of two flies (core.churn_data.derive_dfly_bars) and priced with no
bid-ask (they are built from the legs); their ATR comes from the same derived candles (daily high, low and close, 14 days).
Building them from /taq/ quotes instead was tested on 2026-11-12 and dropped: /taq/ has one row per TRADE (its rows and
volume match the five-minute candles exactly), so the d-fly made from the two flies' quotes moves over the same range as the
one made from their candles. What decides a d-fly's score is the D-fly spread setting, not the price source.

YOUR REAL TRADES (`archive_comparison`, core.churn_archive): the round trips booked in the app for a simulated contract, beside
the simulation's figures for the same contract and side, so the assumptions can be checked against what you actually made.
`calibration` then fits the spread per curve and the stop to those trades (the simulation run at your own targets) and
names the settings to change; it only suggests, the page applies them when you press the button.

OSCILLATIONS. For each row, from the live price (the entry fill) how many times in the last `osc_days` trading days the
market came back to that level and then moved the target in the trade's favour, in one day (core.churn_engine).

LIVE VERDICT (a tooltip and the colour of the Score, not a column). The live price (the curve poll of /fairvalue/, a
d-fly = fly - next fly) is placed against what history says:
  state       how far the live price is from its average over the last day of candles, signed for the trade (negative =
              good side). The hit probability and net per trade after entries made in the SAME state are looked up.
  volatility  the last day's candle-to-candle volatility against the whole window's: a quiet market means slow trips, a
              hot one means a trend that can run through the stop.
  hour        the hit probability of entries made in the current London hour against the all-hours figure.
  GO if the state is not against the trade, volatility is within 0.5-2x usual, the hour is not clearly weak and the price
  is fresh; otherwise WAIT with the reasons. Outside your trading window: CLOSED. No price or a price older than 2
  minutes: no verdict.
"""

import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd

from core.atr_calc import atr_frame
from core.churn_archive import compare, fit_settings, real_trips, row_key
from core.churn_costs import ProductCost, load_costs, load_use_rebates, round_trip_cost, save_costs, tick_of, without_rebates
from core.churn_data import ChurnStore, derive_dfly_bars, dfly_instruments, tracked_instruments
from core.churn_engine import (
    SPLIT_MIN_DAYS, TRAIN_SHARE, WORSE_FILLS, InstrumentSpec, Settings, analyse, hours_for, target_grid, count_oscillations, day_ladder, london_now_hour, prepare, result_of, state_of, trading_calendar,
)
from core.curve_calendar import PRODUCTS
from core.user_settings import parse_number

logger = logging.getLogger(__name__)

KEY_CHURN_PARAMS = "churn_params"
RETRY_SECONDS = 300
MAX_RESULTS = 6
LIVE_REFRESH_MINUTES = 10  # newest candles are re-fetched this often while the page is open
MIN_STATE_TRIALS = 100
VOL_LOW, VOL_HIGH = 0.5, 2.0
MAX_PRICE_AGE_SECONDS = 120
STATE_HIT_DROP = 0.10  # hit probability in this state this far below the all-states figure counts as against
HOUR_HIT_DROP = 0.15
MIN_HOUR_TRIALS = 50
DIRECTIONS = ("long", "short")
PART_FLOOR = 0.05  # lowest value of a part of the score
T_SCALE = 2.0  # the edge part is tanh(t / T_SCALE): a net per day two standard errors above zero gives 0.76
MIN_SCORE = 1.0  # no score is closer to zero than this
BOTH_LOSE_MAX = 30.0  # the most negative score of a row whose other side loses too
RANGE_DAYS = {"week": 7, "month": 30}  # calendar days of the range columns


@dataclass
class ChurnParams:
    contracts: int = 12  # live instruments per family and product
    window_days: int = 120  # calendar days of five-minute candles
    margin: float = 3.0  # dollars per lot a target must earn above the round-trip cost
    max_target: int = 15  # ticks
    spread_outright: float = 1.0  # quoted bid-ask spread in ticks, paid half on entry (and on a forced exit)
    spread_spread: float = 1.0
    spread_fly: float = 1.0
    spread_dfly: float = 0.0  # a d-fly is built from its legs, so there is no bid-ask to cross
    window_start: float = 9.0  # your trading hours, London time
    window_end: float = 19.0
    stop_mult: float = 2.0  # the stop is this many targets away from the fill
    patience_min: float = 60.0  # minutes to wait for the price to come back to the level before moving on
    min_volume: float = 100.0  # lots per day
    start_from: int = 4  # first contract (front = 1) shown and scored
    osc_days: int = 5  # trading days the oscillations are counted over
    min_z: float = 1.0  # > 0: take a new price only when it is this many standard deviations below (above, for a short) its recent average; 0 = any price
    best_hours: int = 0  # > 0: trade only this many best London hours (picked on the days the target is picked on); 0 = all hours
    atr_share: float = 0.5  # > 0: also try targets up to this share of the daily ATR (beyond the largest target); 0 = ticks only

    SIM_FIELDS = ("contracts", "window_days", "margin", "max_target", "spread_outright", "spread_spread", "spread_fly", "spread_dfly", "window_start", "window_end", "stop_mult", "patience_min", "min_z", "best_hours", "atr_share")

    def key(self) -> tuple:
        return tuple(asdict(self).values())

    def sim_key(self) -> tuple:
        """What the simulation depends on; the other settings only change what is shown, so they never rebuild it."""
        return tuple(getattr(self, name) for name in self.SIM_FIELDS)

    def spread_ticks(self, family: str) -> float:
        return {"outright": self.spread_outright, "spread": self.spread_spread, "fly": self.spread_fly, "dfly": self.spread_dfly}[family]


PARAM_RULES = {  # name: (label, minimum, maximum, integer)
    "contracts": ("Contracts per curve", 3, 36, True),
    "window_days": ("Window (calendar days)", 20, 180, True),
    "margin": ("Margin over cost ($ per lot)", 0.0, 1000.0, False),
    "max_target": ("Largest target (ticks)", 1, 30, True),
    "spread_outright": ("Outright spread (ticks)", 0.0, 100.0, False),
    "spread_spread": ("Spread spread (ticks)", 0.0, 100.0, False),
    "spread_fly": ("Fly spread (ticks)", 0.0, 100.0, False),
    "spread_dfly": ("D-fly spread (ticks)", 0.0, 100.0, False),
    "window_start": ("Trading hours start (London)", 0.0, 23.0, False),
    "window_end": ("Trading hours end (London)", 1.0, 24.0, False),
    "stop_mult": ("Stop (targets)", 0.5, 50.0, False),
    "patience_min": ("Wait for the level (minutes)", 0.0, 600.0, False),
    "min_volume": ("Minimum volume per day", 0.0, 1e7, False),
    "start_from": ("Start from contract", 1, 36, True),
    "osc_days": ("Oscillation range (days)", 1, 30, True),
    "min_z": ("Entry stretch (z)", 0.0, 5.0, False),
    "best_hours": ("Best hours only", 0, 12, True),
    "atr_share": ("Targets up to share of ATR", 0.0, 2.0, False),
}


def validate_params(raw: dict) -> ChurnParams:
    defaults = ChurnParams()
    values = {}
    for name, (label, minimum, maximum, integer) in PARAM_RULES.items():
        value = raw.get(name)
        values[name] = getattr(defaults, name) if value in (None, "") else parse_number(value, label, minimum=minimum, maximum=maximum, integer=integer)
    if values["window_end"] <= values["window_start"]:
        raise ValueError("The trading hours must end after they start.")
    if values["start_from"] > values["contracts"]:
        raise ValueError("'Start from contract' cannot be beyond the number of contracts per curve.")
    return ChurnParams(**values)


OLD_DEFAULTS = {"window_days": 60, "max_target": 6, "stop_mult": 3.0}  # before the wider target grid, the 2x stop and the longer history


def load_params(repository) -> ChurnParams:
    """The saved settings; a saved set that still has all of the OLD defaults (never changed by hand) takes the new ones."""
    try:
        raw = repository.get_setting(KEY_CHURN_PARAMS, None)
        if not isinstance(raw, dict):
            return ChurnParams()
        if all(raw.get(name) == value for name, value in OLD_DEFAULTS.items()):
            raw = {**raw, **{name: getattr(ChurnParams(), name) for name in OLD_DEFAULTS}}
        return validate_params(raw)
    except (ValueError, TypeError):
        return ChurnParams()


def save_params(repository, params: ChurnParams) -> None:
    if repository.get_setting(KEY_CHURN_PARAMS, None) != asdict(params):
        repository.set_setting(KEY_CHURN_PARAMS, asdict(params))


@dataclass
class ChurnResults:
    rows: pd.DataFrame  # one row per instrument and direction, not yet scored
    details: dict  # (symbol, direction) -> analysis dict from core.churn_engine.analyse
    as_of: str


def _rank(values: pd.Series, mask: pd.Series) -> pd.Series:
    """Percentile rank (0-1) of the values among the masked rows; 0 elsewhere."""
    out = pd.Series(0.0, index=values.index)
    if mask.any():
        out[mask] = values[mask].rank(pct=True)
    return out


def _proof(row) -> str:
    """Whether a profit is backed by the earlier days too: Proven (profitable on the scored days AND on the first days, which only picked
    a target), Recent only (profitable only on the scored days, perhaps a market phase), Unchecked (too few days to split), No edge."""
    if row.net_day <= 0:
        return "No edge"
    if not getattr(row, "split", False) or not np.isfinite(getattr(row, "insample_net_day", np.nan)):
        return "Unchecked"
    return "Proven" if row.insample_net_day > 0 else "Recent only"


def _quality(rows: pd.DataFrame) -> pd.Series:
    """0-100 for a row that makes money: the geometric mean of the five parts (see the module docstring), each floored so that
    a weak part drags the score down without erasing it. Every part is on a fixed scale: nothing is ranked against other rows."""
    tstat = rows["tstat"] if "tstat" in rows else np.sign(rows["net_day"]) * T_SCALE
    parts = pd.DataFrame({"edge": np.tanh(pd.Series(tstat, index=rows.index).clip(lower=0) / T_SCALE), "hit": rows["hit_prob"], "worse": rows["worse_score"],
                          "stability": rows["pos_days"], "tail": 1.0 - rows["blowup"]}).clip(0, 1).fillna(0).clip(lower=PART_FLOOR)
    return 100.0 * np.exp(np.log(parts).mean(axis=1))


def _score(rows: pd.DataFrame, min_volume: float) -> pd.DataFrame:
    """The signed churn score (-100 to +100, never 0), the other side's score, the TRADABILITY figure (0-100, does not depend on
    profit) and the proof. A row below `min_volume` is scored like any other and flagged `thin`."""
    rows = rows.copy()
    rows["thin"] = rows["volume_day"] < min_volume
    rows["eligible"] = ~rows["thin"]
    profit = rows["net_day"] > 0
    quality = _quality(rows).clip(lower=MIN_SCORE)
    tstat = pd.Series(rows["tstat"] if "tstat" in rows else 0.0, index=rows.index).fillna(0.0)
    mild = -(MIN_SCORE + (BOTH_LOSE_MAX - MIN_SCORE) * np.tanh(tstat.abs() / T_SCALE))  # neither side works: a small negative, by how surely it loses

    other_quality = pd.Series(np.nan, index=rows.index)
    other_net = pd.Series(np.nan, index=rows.index)
    if {"symbol", "direction"} <= set(rows.columns):
        by_key = {(r.symbol, r.direction): i for i, r in zip(rows.index, rows.itertuples())}
        for i, r in zip(rows.index, rows.itertuples()):
            j = by_key.get((r.symbol, "short" if r.direction == "long" else "long"))
            if j is not None:
                other_net[i] = rows.at[j, "net_day"]
                if profit[j]:
                    other_quality[i] = quality[j]
    rows["flip"] = ~profit & other_quality.notna()  # this side loses and the other side makes money
    rows["score"] = np.where(profit, quality, np.where(rows["flip"], -other_quality.fillna(0.0), mild))
    rows["other_net_day"] = other_net
    rows["other_score"] = np.nan
    if {"symbol", "direction"} <= set(rows.columns):
        lookup = {(r.symbol, r.direction): r.score for r in rows.itertuples()}
        rows["other_score"] = [lookup.get((r.symbol, "short" if r.direction == "long" else "long"), np.nan) for r in rows.itertuples()]

    zero = pd.Series(0.0, index=rows.index)
    cost_share = (1.0 - rows.get("cost_rt", zero) / rows.get("target_usd", zero).where(rows.get("target_usd", zero) > 0)).clip(0, 1).fillna(0)  # how much of a win the costs leave
    everyone = pd.Series(True, index=rows.index)
    tradable = pd.DataFrame({
        "reach": rows["hit_prob"], "trips": _rank(rows.get("trips_day", zero), everyone), "volume": _rank(rows["volume_day"], everyone), "cost": cost_share,
    }).clip(0, 1).fillna(0).clip(lower=PART_FLOOR)
    rows["tradable"] = 100.0 * np.exp(np.log(tradable).mean(axis=1))
    rows["proof"] = [_proof(r) for r in rows.itertuples()]
    return rows


def _hours_text(hours: list[int] | None) -> str:
    return "" if not hours else ", ".join(f"{h:02d}:00" for h in hours)


def _worse_score(worse: dict) -> float:
    base = worse[0].net_day
    if base <= 0:
        return 0.0
    shares = [min(max(worse[k].net_day / base, 0.0), 1.0) if k in worse else 0.0 for k in (1, 2)]
    return float(np.mean(shares))


class ChurnService:
    def __init__(self, repository, api, store: ChurnStore, atr_provider: Callable[[str], dict[str, float]] | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc), background: bool = True,
                 monotonic: Callable[[], float] = time.monotonic, live_prices: Callable[[], dict] | None = None):
        self._repository = repository
        self._api = api
        self.store = store
        self._atr_provider = atr_provider
        self._clock = clock
        self._background = background
        self._monotonic = monotonic
        self._live_prices = live_prices  # {API symbol: (price, unix seconds)} of the live curve poll
        self._recent: dict[tuple, dict | None] = {}
        self._lock = threading.Lock()
        self._build_lock = threading.Lock()
        self._retry_after = 0.0
        self._results: dict[tuple, ChurnResults] = {}
        self._fit: tuple | None = None  # (key, the last calibration)

    # -- settings -------------------------------------------------------

    def params(self) -> ChurnParams:
        return load_params(self._repository)

    def save_params(self, params: ChurnParams) -> None:
        save_params(self._repository, params)

    def costs(self):
        return load_costs(self._repository)

    def use_rebates(self) -> bool:
        return load_use_rebates(self._repository)

    def effective_costs(self) -> dict[str, ProductCost]:
        """The costs the simulation uses: the saved table, with the rebates removed unless they are switched on."""
        costs = self.costs()[0]
        return costs if self.use_rebates() else without_rebates(costs)

    def combo_rebates(self) -> dict[str, float] | None:
        """The rebates a quoted CL/BZ structure (a CL-BZ box) earns, or None while rebates are off (the box then pays the full TC)."""
        return self.costs()[1] if self.use_rebates() else None

    def save_costs(self, costs, combo, use_rebates: bool | None = None) -> None:
        save_costs(self._repository, costs, combo, use_rebates)

    # -- downloading ----------------------------------------------------

    def needs_update(self, params: ChurnParams) -> bool:
        now = self._clock()
        return not self.store.is_current(now.date(), params.window_days, params.contracts, now, LIVE_REFRESH_MINUTES)

    def _run(self, params: ChurnParams) -> None:
        progress = self.store.progress
        try:
            report = self.store.update(self._api, params.window_days, params.contracts, self._clock().date())
            progress.message = f"Updated: {report['instruments']} instruments, {report['rows']:,} candles, {report['requests']} requests"
            progress.error = None
        except Exception as exc:  # noqa: BLE001 - shown on the page; partial progress is already saved
            logger.exception("Churn candle update failed")
            progress.error = f"{type(exc).__name__}: {exc}"
            self._retry_after = self._monotonic() + RETRY_SECONDS
        finally:
            progress.running = False
            progress.finished_at = self._clock().isoformat()

    def refresh_now(self, params: ChurnParams | None = None) -> bool:
        params = params or self.params()
        with self._lock:
            if self.store.progress.running:
                return False
            self.store.progress.running = True
            self.store.progress.error = None
        if self._background:
            threading.Thread(target=self._run, args=(params,), name="churn-update", daemon=True).start()
        else:
            self._run(params)
        return True

    def ensure_fresh(self, params: ChurnParams | None = None) -> None:
        """Start the update if it is due (never raises)."""
        if self._monotonic() < self._retry_after or self.store.progress.running:
            return
        try:
            params = params or self.params()
            if self.needs_update(params):
                self.refresh_now(params)
        except Exception:  # noqa: BLE001
            logger.exception("Churn freshness check failed")

    def status(self) -> dict:
        progress = self.store.progress
        return {"running": progress.running, "done": progress.done, "total": progress.total, "message": progress.message, "error": progress.error,
                "updated_on": self.store.updated_on, "has_data": self.store.has_data(), "version": self.store.daily_version}

    # -- results --------------------------------------------------------

    @staticmethod
    def settings_of(params: ChurnParams) -> Settings:
        return Settings(params.window_start, params.window_end, params.stop_mult, params.patience_min, params.min_z, params.best_hours, params.atr_share)

    def _spec(self, instrument, costs: dict[str, ProductCost], params: ChurnParams, atr: float | None) -> InstrumentSpec:
        cost = tick_of(instrument.product, costs)
        return InstrumentSpec(
            instrument.symbol, instrument.product, instrument.family, instrument.label, cost.tick_size, cost.tick_value,
            round_trip_cost(instrument.product, instrument.family, costs, self.combo_rebates()), params.spread_ticks(instrument.family) / 2.0, atr,
        )

    def _atr(self, product: str) -> dict[str, float]:
        if self._atr_provider is None:
            return {}
        try:
            return self._atr_provider(product)
        except Exception:  # noqa: BLE001 - the ATR column is blank
            logger.exception("ATR lookup for the churn table failed")
            return {}

    def _key(self, params: ChurnParams, costs, today: date) -> tuple:
        combo = self.combo_rebates()
        return (self.store.daily_version, params.sim_key(), tuple(sorted((p, c.tick_size, c.tick_value, c.tc, c.rebate) for p, c in costs.items())),
                None if combo is None else tuple(sorted(combo.items())), today)

    def cached(self, params: ChurnParams) -> ChurnResults | None:
        """The simulation if it has already been built for these inputs (never builds it)."""
        return self._results.get(self._key(params, self.effective_costs(), self._clock().date()))

    def results(self, params: ChurnParams) -> ChurnResults:
        costs = self.effective_costs()
        today = self._clock().date()
        key = self._key(params, costs, today)
        with self._build_lock:
            if key in self._results:
                return self._results[key]
            results = self._build(params, costs, today)
            if len(self._results) >= MAX_RESULTS:
                self._results.pop(next(iter(self._results)))
            self._results[key] = results
            return results

    def _instruments(self, params: ChurnParams, today: date):
        """(instrument, its candles, the product's trading calendar, its position front = 1) of everything simulated."""
        for product in PRODUCTS:
            frame = self.store.frame(product)
            if frame.empty:
                continue
            calendar = trading_calendar(pd.DatetimeIndex(frame["ts"]), params.window_start, params.window_end)
            by_symbol = {s: part.sort_values("ts") for s, part in frame.groupby("symbol")}
            live = tracked_instruments(product, today, params.contracts)
            positions: dict[str, int] = {}
            for instrument in live:  # front = 1, within each family
                positions[instrument.symbol] = sum(1 for i in live if i.family == instrument.family and i.order <= instrument.order)
            dflies = dfly_instruments(product, [i for i in live if i.family == "fly"])
            positions.update({d.symbol: n + 1 for n, d in enumerate(dflies)})
            for instrument in [*live, *dflies]:
                bars = self._bars(instrument, by_symbol)
                if bars is not None:
                    yield instrument, bars, calendar, positions[instrument.symbol]

    def _build(self, params: ChurnParams, costs, today: date) -> ChurnResults:
        rows, details = [], {}
        atr: dict[str, dict[str, float]] = {}
        for instrument, bars, calendar, position in self._instruments(params, today):
            if instrument.product not in atr:
                atr[instrument.product] = self._atr(instrument.product)
            spec = self._spec(instrument, costs, params, self._instrument_atr(instrument, bars, atr[instrument.product]))
            for direction in DIRECTIONS:
                result = analyse(bars, spec, direction, target_grid(params.max_target, params.atr_share, spec.atr, spec.tick), params.margin, self.settings_of(params), calendar)
                if result is None:
                    continue
                details[(instrument.symbol, direction)] = result
                rows.append(self._row(result, spec, direction, position))
        return ChurnResults(pd.DataFrame(rows), details, today.isoformat())

    @staticmethod
    def _bars(instrument, by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        """The stored candles of an instrument; a d-fly's are derived from its two flies."""
        if instrument.family != "dfly":
            return by_symbol.get(instrument.symbol)
        first, second = (by_symbol.get(s) for s in instrument.legs)
        if first is None or second is None or first.empty or second.empty:
            return None
        return derive_dfly_bars(first, second)

    @staticmethod
    def _instrument_atr(instrument, bars: pd.DataFrame, atr: dict[str, float]) -> float | None:
        """Daily ATR (points): the ATR page's for a real instrument, from its own daily candles for a d-fly."""
        if instrument.family != "dfly":
            return atr.get(instrument.symbol)
        day = bars["ts"].dt.normalize()
        daily = pd.DataFrame({"high": bars["high"].groupby(day).max(), "low": bars["low"].groupby(day).min(), "close": bars["close"].groupby(day).last()})
        series = atr_frame(daily.assign(volume=1.0), 14, True, percent=False)["atr"].dropna()
        return float(series.iloc[-1]) if len(series) else None

    @staticmethod
    def _row(result: dict, spec: InstrumentSpec, direction: str, position: int) -> dict:
        best, worse = result["best"], result["worse"]
        base = worse[0]  # the figures that rank the contract: out of sample
        row = {
            "symbol": spec.symbol, "product": spec.product, "family": spec.family, "label": spec.label, "position": position, "direction": direction,
            "atr": spec.atr if spec.atr is not None else np.nan, "target": best, "target_usd": best * spec.tick_value, "cost_rt": spec.cost_rt,
            "net_hit": best * spec.tick_value - spec.cost_rt, "hit_prob": base.hit_prob, "trips_day": base.trips_per_day, "wins_day": base.wins_per_day,
            "stops_day": base.stops_per_day, "stop_usd": result["stop_loss_usd"], "net_trade": base.net_trade, "net_day": base.net_day,
            "insample_net_day": result["insample"].net_day if result["insample"] is not None else np.nan, "worse_score": _worse_score(worse),
            "pos_days": base.pos_days, "blowup": base.blowup, "tstat": base.tstat, "worst_day": base.worst_day, "volume_day": result["volume_day"], "days": base.days,
            "split": result["split"], "hours": _hours_text(result.get("hours")),
        }
        for k in WORSE_FILLS[1:]:
            row[f"hit_w{k}"] = worse[k].hit_prob if k in worse else np.nan
            row[f"net_day_w{k}"] = worse[k].net_day if k in worse else np.nan
        return row

    # -- the table ------------------------------------------------------

    def scored(self, params: ChurnParams) -> pd.DataFrame | None:
        """The rows from the chosen contract onwards, scored among themselves and sorted; None until the simulation exists."""
        results = self.cached(params)
        if results is None:
            return None
        rows = results.rows
        if rows.empty:
            return rows
        rows = _score(rows[rows["position"] >= params.start_from], params.min_volume)
        return rows.sort_values(["score", "net_day"], ascending=False).reset_index(drop=True)

    def archive_comparison(self, params: ChurnParams) -> tuple[pd.DataFrame, list[str]]:
        """Your real round trips (every structure in the book that is one simulated instrument) beside the simulation's figures of
        the same contract and side, and the names of traded structures the simulation has nothing for (see core.churn_archive).
        The simulation side is blank until it has been built; every contract is looked up, not only those from `start_from`."""
        costs = self.effective_costs()
        ticks = {product: tick_of(product, costs).tick_value for product in PRODUCTS}
        real, unmatched = real_trips(self._repository.get_all_structures(), self._repository.get_trades_for_structure, ticks)
        results = self.cached(params)
        scored = _score(results.rows, params.min_volume) if results is not None and not results.rows.empty else None
        return compare(real, scored), unmatched

    def calibration(self, params: ChurnParams) -> dict | None:
        """The spread per curve and the stop that bring the simulation closest to your real round trips (core.churn_archive.
        fit_settings): the day ladder is run on each traded contract and side at your own target, with your hours, wait and
        entry stretch, over a grid of spreads and stops. None until the simulation is built or when nothing traded is simulated."""
        if self.cached(params) is None:
            return None
        costs, today = self.effective_costs(), self._clock().date()
        ticks = {product: tick_of(product, costs).tick_value for product in PRODUCTS}
        real, _ = real_trips(self._repository.get_all_structures(), self._repository.get_trades_for_structure, ticks)
        if real.empty:
            return None
        key = (self._key(params, costs, today), tuple(real[["product", "key", "direction", "trips", "lots", "gross"]].itertuples(index=False, name=None)))
        if self._fit is not None and self._fit[0] == key:
            return self._fit[1]
        wanted = {(r.product, r.key) for r in real.itertuples()}
        held = {}
        for instrument, bars, calendar, _ in self._instruments(params, today):
            found = row_key(instrument.product, instrument.symbol)
            if found in wanted:
                held[found] = (instrument, bars, calendar)
        contexts: dict[tuple, tuple] = {}

        def simulate(product, months, direction, target, spread, stop):
            if (product, months, direction) not in contexts:
                instrument, bars, calendar = held[(product, months)]
                spec = self._spec(instrument, costs, params, None)
                contexts[(product, months, direction)] = (prepare(bars, spec.tick, direction, params.window_start, params.window_end, calendar), spec)
            ctx, spec = contexts[(product, months, direction)]
            if ctx is None:
                return None
            days = day_ladder(ctx, target, stop * target, params.patience_min, spread / 2.0, 0, spec.tick_value, spec.cost_rt, None, params.min_z)
            trades = int(days.trades.sum())
            if not trades:
                return None
            return (float(days.total.sum()) / trades + spec.cost_rt) / spec.tick_value, float(days.wins.sum()) / trades

        spreads = {family: params.spread_ticks(family) for family in ("outright", "spread", "fly", "dfly")}
        fit = fit_settings(real, lambda product, months: held[(product, months)][0].family if (product, months) in held else None, simulate, spreads, params.stop_mult)
        fit = fit if fit["families"] else None
        self._fit = (key, fit)
        return fit

    @staticmethod
    def sessions(rows: pd.DataFrame, details: dict, top: int = 20) -> pd.DataFrame:
        """By UTC entry hour, pooled over the best scored rows: entries, hit probability and net per trade."""
        parts = [details[(r.symbol, r.direction)]["entry"].hour.reset_index() for r in rows[rows["score"] > 0].head(top).itertuples()]
        if not parts:
            return pd.DataFrame(columns=["hour", "trials", "hit_prob", "net_trade"])
        pooled = pd.concat(parts)
        pooled["w_hit"] = pooled["hit_prob"] * pooled["trials"]
        pooled["w_net"] = pooled["net_trade"] * pooled["trials"]
        grouped = pooled.groupby("hour").agg(trials=("trials", "sum"), w_hit=("w_hit", "sum"), w_net=("w_net", "sum"))
        return pd.DataFrame({"trials": grouped["trials"], "hit_prob": grouped["w_hit"] / grouped["trials"], "net_trade": grouped["w_net"] / grouped["trials"]}).reset_index()

    # -- live -----------------------------------------------------------

    def _symbol_bars(self, spec: InstrumentSpec, legs: tuple[str, ...] | None) -> pd.DataFrame | None:
        """The stored candles of one instrument (all hours), a d-fly's built from its two flies."""
        frame = self.store.frame(spec.product)
        by_symbol = {s: frame[frame["symbol"] == s].sort_values("ts") for s in (legs or (spec.symbol,))}
        return self._bars(SimpleNamespace(family="dfly" if legs else spec.family, symbol=spec.symbol, legs=legs), by_symbol)

    def _recent_stats(self, spec: InstrumentSpec, legs: tuple[str, ...] | None, bars_per_day: float) -> dict | None:
        """Mean and standard deviation of the last day of closes, the volatility against the window's, and the last candle's time."""
        key = (self.store.version, spec.symbol)
        if key not in self._recent:
            if len(self._recent) > 800:
                self._recent.clear()
            bars = self._symbol_bars(spec, legs)
            window = max(int(round(bars_per_day)), 20)
            stats = None
            if bars is not None and len(bars) > 3 * window:
                close = bars["close"].to_numpy(float)
                last, changes = close[-window:], np.diff(close)
                full = changes.std()
                stats = {"mean": float(last.mean()), "std": float(last.std()), "vol_ratio": float(changes[-window:].std() / full) if full > 0 else np.nan, "last_ts": bars["ts"].iloc[-1]}
            self._recent[key] = stats
        return self._recent[key]

    def _ranges(self, spec: InstrumentSpec, legs: tuple[str, ...] | None) -> dict:
        """Low and high of the last week and the last month of stored candles, and how many calendar days the store goes back."""
        key = (self.store.version, spec.symbol, "range")
        if key not in self._recent:
            bars = self._symbol_bars(spec, legs)
            out: dict = {}
            if bars is not None and len(bars):
                now = pd.Timestamp(self._clock()).tz_convert("UTC").tz_localize(None)
                out["range_days"] = float((now - bars["ts"].iloc[0]).total_seconds() / 86400.0)
                for name, days in RANGE_DAYS.items():
                    part = bars[bars["ts"] >= now - pd.Timedelta(days=days)]
                    if len(part):
                        out[f"{name}_low"], out[f"{name}_high"] = float(part["low"].min()), float(part["high"].max())
            self._recent[key] = out
        return self._recent[key]

    def _live_price(self, spec: InstrumentSpec, legs: tuple[str, ...] | None) -> tuple[float, float] | None:
        """(price, unix seconds) of the live curve poll; a d-fly is its first fly minus the next one."""
        prices = self._live_prices() if self._live_prices else {}
        if legs:
            a, b = prices.get(legs[0]), prices.get(legs[1])
            return (a[0] - b[0], min(a[1], b[1])) if a and b else None
        return prices.get(spec.symbol)

    def live_table(self, params: ChurnParams, product: str = "all", family: str = "all", direction: str = "all") -> pd.DataFrame | None:
        """The table of the page: every scored row with its live price, oscillation count and verdict. None until the
        simulation has been built (it is built by `results`, never here)."""
        rows = self.scored(params)
        if rows is None:
            return None
        results = self.cached(params)
        for column, value in (("product", product), ("family", family), ("direction", direction)):
            if value and value != "all" and not rows.empty:
                rows = rows[rows[column] == value]
        out = []
        now = self._clock()
        hour = london_now_hour(now)
        inside = params.window_start <= hour < params.window_end
        for r in rows.itertuples():
            detail = results.details[(r.symbol, r.direction)]
            spec: InstrumentSpec = detail["spec"]
            legs = tuple(r.symbol.split("/")) if r.family == "dfly" else None
            price = self._live_price(spec, legs)
            stats = self._recent_stats(spec, legs, detail["bars_per_day"])
            out.append(self._row_live(r, spec, detail, price, stats, now.timestamp(), int(hour), inside, params, self._ranges(spec, legs)))
        return pd.DataFrame(out)

    @staticmethod
    def _why_negative(r, params: ChurnParams) -> tuple[str, str]:
        """(verdict, reason in plain words) of a row that loses money: FLIP when the other side makes money, else AVOID."""
        other = "short" if r.direction == "long" else "long"
        needed = r.stop_usd / (r.net_hit + r.stop_usd) if r.net_hit + r.stop_usd > 0 else np.nan
        lost = (f"This side loses ${-r.net_day:.0f} per lot a day over the {r.days} scored days: {r.wins_day:.1f} wins a day at +${r.net_hit:.2f} against "
                f"{r.stops_day:.1f} stop-outs a day at -${r.stop_usd:.2f}; it needs {needed:.0%} of trades to win, it got {r.hit_prob:.0%}.")
        other_net = getattr(r, "other_net_day", np.nan)
        if getattr(r, "flip", False):
            return "FLIP", f"Churn the {other} side instead: it scores {-r.score:+.0f} and makes ${other_net:.0f} per lot a day. " + lost
        both = f" The {other} side loses ${-other_net:.0f} a day too." if np.isfinite(other_net) else ""
        return "AVOID", "Neither side makes money after costs. " + lost + both

    @staticmethod
    def _row_live(r, spec: InstrumentSpec, detail: dict, price, stats, now_ts: float, hour: int, inside: bool, params: ChurnParams, ranges: dict | None = None) -> dict:
        row = ChurnService._verdict(r, spec, detail["entry"], price, stats, now_ts, hour)
        row.update(hours=getattr(r, "hours", ""), tradable=getattr(r, "tradable", 0.0), proof=getattr(r, "proof", ""), volume_day=r.volume_day, atr=r.atr, cost_rt=r.cost_rt, net_hit=r.net_hit, worse_score=r.worse_score, position=r.position, net_day=r.net_day, osc=np.nan,
                   other_score=getattr(r, "other_score", np.nan), thin=bool(getattr(r, "thin", False)), days=getattr(r, "days", 0), split=bool(getattr(r, "split", False)),
                   **{name: (ranges or {}).get(name, np.nan) for name in ("week_low", "week_high", "month_low", "month_high", "range_days")})
        if price is not None:
            sign = 1 if r.direction == "long" else -1
            row["osc"] = count_oscillations(detail["path"], r.direction, price[0] + sign * spec.half_spread * spec.tick, r.target, spec.tick, params.osc_days)
        if r.score <= 0:
            verdict, reason = ChurnService._why_negative(r, params)
            row.update(verdict=verdict, reason=reason)
        elif not inside:
            row.update(verdict="CLOSED", reason=f"Outside your trading hours ({params.window_start:g}:00 to {params.window_end:g}:00 London). " + row["reason"])
        if r.score > 0 and r.split and np.isfinite(r.insample_net_day) and r.insample_net_day <= 0:
            row["reason"] += (f" Caution: it lost ${-r.insample_net_day:.0f} per lot a day on the earlier days the target was picked on, so the recent profit may be a market phase and not an edge.")
        if not getattr(r, "split", True):
            row["reason"] += f" Caution: only {r.days} trading days of candles, too few to split, so the target was picked on the same days it is scored on. Widen the window."
        if row["thin"]:
            row["reason"] += f" Thin: only {r.volume_day:,.0f} lots a day trade here (minimum {params.min_volume:,.0f}), so its moves cannot be relied on."
        return row

    @staticmethod
    def _verdict(r, spec: InstrumentSpec, base, price, stats, now_ts: float, hour: int) -> dict:
        sign = 1 if r.direction == "long" else -1
        row = {"symbol": r.symbol, "product": r.product, "label": r.label, "family": r.family, "direction": r.direction, "score": r.score, "target": r.target,
               "price": np.nan, "age": np.nan, "entry": np.nan, "target_price": np.nan, "z": np.nan, "state": "", "hit_state": np.nan, "hit_all": base.entry_hit_prob,
               "net_state": np.nan, "net_all": base.net_trade, "vol_ratio": np.nan, "hit_hour": np.nan, "verdict": "NO PRICE", "reason": "No live price for this instrument in the curve poll."}
        if price is None:
            return row
        live, stamp = price
        age = max(now_ts - stamp, 0.0)
        entry = live + sign * spec.half_spread * spec.tick
        row.update(price=live, age=age, entry=entry, target_price=entry + sign * r.target * spec.tick)
        if age > MAX_PRICE_AGE_SECONDS:
            row.update(verdict="STALE", reason=f"The live price is {age:.0f}s old.")
            return row
        reasons = []
        if stats and stats["std"] > 0:
            z = sign * (live - stats["mean"]) / stats["std"]  # negative: the price is on the good side for this trade
            state = state_of(z)
            cond = base.states.loc[state] if base.states is not None and state in base.states.index else None
            row.update(z=z, state=state)
            if cond is not None and cond["trials"] >= MIN_STATE_TRIALS:
                row.update(hit_state=float(cond["hit_prob"]), net_state=float(cond["net_trade"]))
                if cond["net_trade"] <= 0 or cond["hit_prob"] < base.entry_hit_prob - STATE_HIT_DROP:
                    reasons.append(f"after entries in this state ({state}, z {z:+.1f}) only {cond['hit_prob']:.0%} reached the target (all states {base.entry_hit_prob:.0%}); net {cond['net_trade']:+.2f}$ per trade")
            ratio = stats["vol_ratio"]
            row["vol_ratio"] = ratio
            if np.isfinite(ratio) and ratio < VOL_LOW:
                reasons.append(f"quiet market: {ratio:.1f}x the usual volatility, trips will be slow")
            elif np.isfinite(ratio) and ratio > VOL_HIGH:
                reasons.append(f"volatile market: {ratio:.1f}x the usual, a trend can run through the end of the day")
        else:
            reasons.append("not enough recent candles to judge the state")
        hours = base.hour
        if hour in hours.index and hours.loc[hour, "trials"] >= MIN_HOUR_TRIALS:
            row["hit_hour"] = float(hours.loc[hour, "hit_prob"])
            if row["hit_hour"] < base.entry_hit_prob - HOUR_HIT_DROP:
                reasons.append(f"a weak hour: entries at {hour:02d}:00 London reached the target {row['hit_hour']:.0%} of the time (all hours {base.entry_hit_prob:.0%})")
        row.update(verdict="WAIT" if reasons else "GO", reason="; ".join(reasons) if reasons else "Price, volatility and hour are all in line with the entries that worked.")
        return row

    def detail_grid(self, symbol: str, direction: str, params: ChurnParams) -> pd.DataFrame | None:
        """Every target (1..max) against every worse fill for one instrument, on all days: hit probability and net per day."""
        results = self.results(params)
        entry = results.details.get((symbol, direction))
        if entry is None:
            return None
        spec: InstrumentSpec = entry["spec"]
        frame = self.store.frame(spec.product)
        by_symbol = {k: part.sort_values("ts") for k, part in frame.groupby("symbol")}
        bars = self._bars(SimpleNamespace(family=spec.family, symbol=symbol, legs=tuple(symbol.split("/"))), by_symbol)
        settings = self.settings_of(params)
        calendar = trading_calendar(pd.DatetimeIndex(frame["ts"]), params.window_start, params.window_end)
        ctx = prepare(bars, spec.tick, direction, settings.start_hour, settings.end_hour, calendar)
        if ctx is None:
            return None
        out = []
        n_days = len(calendar)
        train_days = int(round(n_days * TRAIN_SHARE)) if n_days >= SPLIT_MIN_DAYS else n_days
        for t in target_grid(params.max_target, params.atr_share, spec.atr, spec.tick):
            hour_ok = hours_for(ctx, t, settings.stop_mult * t, spec.half_spread, spec.tick_value, spec.cost_rt, train_days, settings.best_hours)
            for k in WORSE_FILLS:
                arrays = day_ladder(ctx, t, settings.stop_mult * t, settings.patience_minutes, spec.half_spread, k, spec.tick_value, spec.cost_rt, hour_ok, settings.min_z)
                res = result_of(arrays, slice(0, len(calendar)), t, k, (settings.stop_mult * t + spec.half_spread) * spec.tick_value + spec.cost_rt)
                if res is not None:
                    out.append({"target": t, "worse": k, "hit_prob": res.hit_prob, "net_day": res.net_day, "net_trade": res.net_trade, "trips_per_day": res.trips_per_day})
        return pd.DataFrame(out)
