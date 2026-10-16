"""The Churn page's engine: keeps the five-minute candles fresh, runs the intraday simulation of every live instrument
and ranks them by churn score, with the live price and the oscillation count of each in the same table.

`ensure_fresh()` is called by the page while it is open (it starts a background download when the data was last
refreshed more than a few minutes ago, or the page's window or number of contracts grew). `results()` builds, and caches
until the stored data, the simulation settings or the cost table change, one row per instrument and direction.
`live_table()` is cheap: it scores those rows (among the contracts from the chosen one onwards) and adds the live price.

INTRADAY, IN YOUR HOURS. Trades are taken only inside the trading window (London time, default 09:00-19:00) and closed
by its end (see core.churn_engine). There is no holding limit in days and no ATR rule; the daily ATR is only shown.
ALL front contracts are downloaded and simulated; `start_from` only decides which ones the page shows and scores.

CHURN SCORE (0 to 100), the geometric mean of five 0-1 parts times 100, so a weak part pulls the whole score down (no part
is taken below 0.05, so a contract that makes money always scores above 0 and a weak part cannot erase it). All
figures are OUT OF SAMPLE (the target is picked on the first 60% of the days, the figures come from the last 40%):
  edge        net $ per day per lot, ranked among the eligible rows shown (rows that lose money score 0)
  hit         share of the day ladder's trades that reached their target
  worse fill  how much of the net per day survives an entry 1 and 2 ticks worse (average of the two shares)
  stability   share of days with a positive net
  tail        1 - blow-up rate (days that lost more than three stop-outs' worth)
A row whose volume per day is below `min_volume` is shown but not scored. A row that scores 0 says why in the tooltip.

D-FLIES are simulated too, from candles derived out of two flies (core.churn_data.derive_dfly_bars) and priced with no
bid-ask (they are built from the legs); their ATR comes from the same derived candles (daily high, low and close, 14 days).

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
from core.churn_costs import ProductCost, load_costs, load_use_rebates, round_trip_cost, save_costs, without_rebates
from core.churn_data import ChurnStore, derive_dfly_bars, dfly_instruments, tracked_instruments
from core.churn_engine import (
    WORSE_FILLS, InstrumentSpec, Settings, analyse, count_oscillations, day_ladder, london_now_hour, prepare, result_of, state_of, trading_calendar,
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
PART_FLOOR = 0.05  # lowest value of the hit, worse-fill, stability and tail parts of the score


@dataclass
class ChurnParams:
    contracts: int = 12  # live instruments per family and product
    window_days: int = 60  # calendar days of five-minute candles
    margin: float = 3.0  # dollars per lot a target must earn above the round-trip cost
    max_target: int = 6  # ticks
    spread_outright: float = 1.0  # quoted bid-ask spread in ticks, paid half on entry (and on a forced exit)
    spread_spread: float = 1.0
    spread_fly: float = 1.0
    spread_dfly: float = 0.0  # a d-fly is built from its legs, so there is no bid-ask to cross
    window_start: float = 9.0  # your trading hours, London time
    window_end: float = 19.0
    stop_mult: float = 3.0  # the stop is this many targets away from the fill
    patience_min: float = 60.0  # minutes to wait for the price to come back to the level before moving on
    min_volume: float = 100.0  # lots per day
    start_from: int = 4  # first contract (front = 1) shown and scored
    osc_days: int = 5  # trading days the oscillations are counted over

    SIM_FIELDS = ("contracts", "window_days", "margin", "max_target", "spread_outright", "spread_spread", "spread_fly", "spread_dfly", "window_start", "window_end", "stop_mult", "patience_min")

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


def load_params(repository) -> ChurnParams:
    try:
        raw = repository.get_setting(KEY_CHURN_PARAMS, None)
        return validate_params(raw) if isinstance(raw, dict) else ChurnParams()
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


def _score(rows: pd.DataFrame, min_volume: float) -> pd.DataFrame:
    rows = rows.copy()
    rows["eligible"] = rows["volume_day"] >= min_volume
    eligible = rows["eligible"] & (rows["net_day"] > 0)
    edge = pd.Series(0.0, index=rows.index)
    if eligible.any():
        edge[eligible] = rows.loc[eligible, "net_day"].rank(pct=True)
    parts = pd.DataFrame({"edge": edge, "hit": rows["hit_prob"], "worse": rows["worse_score"], "stability": rows["pos_days"], "tail": 1.0 - rows["blowup"]}).clip(0, 1).fillna(0)
    parts.loc[:, ["hit", "worse", "stability", "tail"]] = parts[["hit", "worse", "stability", "tail"]].clip(lower=PART_FLOOR)  # a weak part drags the score down, it does not erase it
    geometric = np.exp(np.log(parts.clip(lower=1e-9)).mean(axis=1))
    rows["score"] = np.where(eligible, 100.0 * geometric, 0.0)
    return rows


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
        return Settings(params.window_start, params.window_end, params.stop_mult, params.patience_min)

    def _spec(self, instrument, costs: dict[str, ProductCost], params: ChurnParams, atr: float | None) -> InstrumentSpec:
        cost = costs[instrument.product]
        return InstrumentSpec(
            instrument.symbol, instrument.product, instrument.family, instrument.label, cost.tick_size, cost.tick_value,
            round_trip_cost(instrument.product, instrument.family, costs), params.spread_ticks(instrument.family) / 2.0, atr,
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
        return (self.store.daily_version, params.sim_key(), tuple(sorted((p, c.tick_size, c.tick_value, c.tc, c.rebate) for p, c in costs.items())), today)

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

    def _build(self, params: ChurnParams, costs, today: date) -> ChurnResults:
        rows, details = [], {}
        targets = list(range(1, params.max_target + 1))
        for product in PRODUCTS:
            frame = self.store.frame(product)
            if frame.empty:
                continue
            atr = self._atr(product)
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
                if bars is None:
                    continue
                spec = self._spec(instrument, costs, params, self._instrument_atr(instrument, bars, atr))
                for direction in DIRECTIONS:
                    result = analyse(bars, spec, direction, targets, params.margin, self.settings_of(params), calendar)
                    if result is None:
                        continue
                    details[(instrument.symbol, direction)] = result
                    rows.append(self._row(result, spec, direction, positions[instrument.symbol]))
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
            "pos_days": base.pos_days, "blowup": base.blowup, "worst_day": base.worst_day, "volume_day": result["volume_day"], "days": base.days,
            "split": result["split"],
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

    def _recent_stats(self, spec: InstrumentSpec, legs: tuple[str, ...] | None, bars_per_day: float) -> dict | None:
        """Mean and standard deviation of the last day of closes, the volatility against the window's, and the last candle's time."""
        key = (self.store.version, spec.symbol)
        if key not in self._recent:
            if len(self._recent) > 400:
                self._recent.clear()
            frame = self.store.frame(spec.product)
            if legs:
                first, second = (frame[frame["symbol"] == s].sort_values("ts") for s in legs)
                bars = derive_dfly_bars(first, second) if len(first) and len(second) else None
            else:
                bars = frame[frame["symbol"] == spec.symbol].sort_values("ts")
            window = max(int(round(bars_per_day)), 20)
            stats = None
            if bars is not None and len(bars) > 3 * window:
                close = bars["close"].to_numpy(float)
                last, changes = close[-window:], np.diff(close)
                full = changes.std()
                stats = {"mean": float(last.mean()), "std": float(last.std()), "vol_ratio": float(changes[-window:].std() / full) if full > 0 else np.nan, "last_ts": bars["ts"].iloc[-1]}
            self._recent[key] = stats
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
            out.append(self._row_live(r, spec, detail, price, stats, now.timestamp(), int(hour), inside, params))
        return pd.DataFrame(out)

    @staticmethod
    def _why_zero(r, params: ChurnParams) -> str:
        """In plain words, why a row has no score."""
        if r.volume_day < params.min_volume:
            return f"Not scored: only {r.volume_day:,.0f} lots a day trade here (minimum {params.min_volume:,.0f}), so its moves cannot be relied on."
        if r.net_day <= 0:
            needed = r.stop_usd / (r.net_hit + r.stop_usd) if r.net_hit + r.stop_usd > 0 else np.nan
            return (f"Not scored: loses ${-r.net_day:.0f} per lot a day over the last {r.days} days (the {r.target}-tick target was picked on the days before). "
                    f"{r.wins_day:.1f} wins a day at +${r.net_hit:.2f} against {r.stops_day:.1f} stop-outs a day at -${r.stop_usd:.2f}; it needs {needed:.0%} of trades to win, it got {r.hit_prob:.0%}.")
        return f"Not scored: it makes money but a part of the score is zero (positive days {r.pos_days:.0%}, worse-fill survival {r.worse_score:.0%}, blow-up {r.blowup:.0%})."

    @staticmethod
    def _row_live(r, spec: InstrumentSpec, detail: dict, price, stats, now_ts: float, hour: int, inside: bool, params: ChurnParams) -> dict:
        row = ChurnService._verdict(r, spec, detail["entry"], price, stats, now_ts, hour)
        row.update(volume_day=r.volume_day, atr=r.atr, cost_rt=r.cost_rt, net_hit=r.net_hit, worse_score=r.worse_score, position=r.position, net_day=r.net_day, osc=np.nan)
        if price is not None:
            sign = 1 if r.direction == "long" else -1
            row["osc"] = count_oscillations(detail["path"], r.direction, price[0] + sign * spec.half_spread * spec.tick, r.target, spec.tick, params.osc_days)
        if r.score <= 0:
            row.update(verdict="NOT SCORED", reason=ChurnService._why_zero(r, params))
        elif not inside:
            row.update(verdict="CLOSED", reason=f"Outside your trading hours ({params.window_start:g}:00 to {params.window_end:g}:00 London). " + row["reason"])
        if r.score > 0 and r.split and np.isfinite(r.insample_net_day) and r.insample_net_day <= 0:
            row["reason"] += (f" Caution: it lost ${-r.insample_net_day:.0f} per lot a day on the earlier days the target was picked on, so the recent profit may be a market phase and not an edge.")
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
        for t in range(1, params.max_target + 1):
            for k in WORSE_FILLS:
                arrays = day_ladder(ctx, t, settings.stop_mult * t, settings.patience_minutes, spec.half_spread, k, spec.tick_value, spec.cost_rt)
                res = result_of(arrays, slice(0, len(calendar)), t, k, (settings.stop_mult * t + spec.half_spread) * spec.tick_value + spec.cost_rt)
                if res is not None:
                    out.append({"target": t, "worse": k, "hit_prob": res.hit_prob, "net_day": res.net_day, "net_trade": res.net_trade, "trips_per_day": res.trips_per_day})
        return pd.DataFrame(out)
