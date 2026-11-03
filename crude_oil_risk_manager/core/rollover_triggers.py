"""Trigger study for the Strategy Rollover page: why does a line move one way around an expiry in some months and not in others?

The Strategy Rollover page draws one LINE per front expiry (the Oct26 line, the Sep26 line ...) against trading days from that
expiry (0 on the expiry day, negative before it). A WINDOW is two points of that axis, "from X to Y" (default -10 to +10). Each
earlier expiry gives one MOVE: how far its line went between X and Y. The study also reads TRIGGERS on the entry day (day X): candidates
that might explain why one expiry moved up, another down and another not at all. Triggers are lined up against the moves
(core.trigger_stats) and the window is called one of:

    TREND         the expiries agree on a direction; triggers (if any) explain which were strong and which weak
    MIXED REGIMES they do not agree, but a trigger splits them into groups that move in opposite directions
    NO MECHANISM  they do not agree and no trigger explains it (could be noise, only a few expiries)

Triggers are read on the entry day with data known THEN; a trigger that compares with other expiries uses only EARLIER ones.

ROLL DAYS. The CL-BZ line is the FRONT box, spliced from different CL/BZ pairs: it switches pair on the first trading day after the CL
expiry and again after the Brent expiry. The price jump on those days is the switch, not a market move (it makes CL|BZ look as if it
drops after every expiry), so a line's move is the sum of its daily changes WITHOUT those days (`Prepared.adjusted`), and the days that
were left out are listed with the jump that was ignored. A generic line follows ONE fixed contract through the expiry and never
switches. The market-movement triggers use front-month series, which switch the same way, and leave the same days out (core.roll_days).

BRENT. BZ has no stored settlements, so the Brent triggers use the ICE Brent (BRN) contracts; the same stand-in the rest of the app uses.
"""

import math
import uuid
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import pandas as pd

from core.curve_calendar import FAMILY_LABELS, api_symbol, product_spec
from core.rollover import MAX_BEFORE, RolloverLine, RolloverOptions, RolloverService
from core.roll_days import ChangeSeries, LegData, RollEvent, build_leg_data, difference, total_between
from core.trigger_stats import (
    CONSTANT, MIN_STUDY_YEARS, NO_EVIDENCE, POSSIBLE, STRONG, TOO_FEW, TREND, RuleFit, RuleScore, Screen, WindowStats, finish_screens,
    fit_rule, score_rule, screen, window_stats,
)

DEFAULT_FROM, DEFAULT_TO = -10, 10
MIN_X, MAX_X = -MAX_BEFORE, 300
DEFAULT_LOOKBACK = 10
MIN_LOOKBACK, MAX_LOOKBACK = 3, 60
DEFAULT_LINES, MAX_LINES = 36, 72
OWN_DAYS = 60
MIN_EARLIER = 3  # earlier expiries needed for a trigger that compares with other expiries
VALUE_TOLERANCE_DAYS = 5
SCAN_STEP = 2
SCAN_LENGTHS = (5, 10, 15, 20, 30)
SCAN_SHOW = 15
SCAN_CHANCE_REPS = 60
RULES_SETTING = "rollover_rules"

DONE, OPEN, AHEAD, MISSING = "done", "open", "ahead", "missing"
CASE_TREND, CASE_MIXED, CASE_NONE, CASE_TOO_FEW = "trend", "mixed", "none", "too few"

GROUP_LINE, GROUP_CURVE, GROUP_PRICE, GROUP_OI, GROUP_MARKET = "This line", "Curve type", "Outright prices", "Open interest", "Market movement"
GROUP_ORDER = (GROUP_LINE, GROUP_CURVE, GROUP_PRICE, GROUP_OI, GROUP_MARKET)
LEG_NAMES = {"CL": "CL", "BRN": "Brent (BRN)", "BZ": "Brent (BZ)", "WBS": "WTI (WBS)", "G": "Gasoil"}

# Triggers that matter in this market but have no data feed in the app yet.
EXTERNAL_TRIGGERS = [
    ("Freight / transatlantic arbitrage", "US Gulf-to-Europe and Asia freight against the WTI-Brent spread: when the arbitrage is shut, exports slow."),
    ("US stocks and exports (EIA weekly)", "Cushing and total crude stocks, and crude exports: the direct supply and demand signal for WTI against Brent."),
    ("Refinery runs and turnarounds", "Strongly seasonal, and it hits WTI and Brent differently."),
    ("Positioning (CFTC COT, ICE)", "Crowded positioning can reverse a move around an expiry."),
    ("Event calendar", "OPEC+ meetings, SPR releases, hurricane season, FOMC and the dollar."),
    ("Implied volatility (OVX)", "Option-implied volatility, as a regime marker."),
]


@dataclass(frozen=True)
class TriggerSpec:
    key: str
    label: str
    group: str
    why: str


@dataclass(frozen=True)
class RollSkip:
    """A day left out of a line's move because the line switched what it prices that day."""

    day: date
    label: str  # the line
    before: str  # what was priced the day before
    after: str
    jump: float | None  # the price change that day, which is not counted


@dataclass
class LineObs:
    """One expiry's line over the window: the move and the trigger readings on the entry day."""

    label: str
    anchor: date
    live: bool
    status: str  # DONE (the window has finished), OPEN (running), AHEAD (not started), MISSING (the line does not cover it)
    entry_date: date | None = None
    exit_date: date | None = None
    start_price: float | None = None
    end_price: float | None = None
    move: float | None = None  # roll days not counted
    raw_move: float | None = None  # the plain price difference
    so_far: float | None = None  # open window: the move to the latest settlement
    skipped: list[RollSkip] = field(default_factory=list)
    triggers: dict[str, float | None] = field(default_factory=dict)
    attribution: dict[str, float] | None = None  # CL-BZ: how much of the move came from CL and from Brent
    raw: dict[str, float] = field(default_factory=dict, repr=False)


@dataclass
class TriggerScreen:
    spec: TriggerSpec
    result: Screen
    current: float | None  # the same reading for the line being drawn now


@dataclass
class RuleCheck:
    spec: TriggerSpec
    fit: RuleFit
    score: RuleScore | None
    now_value: float | None
    now_direction: int | None


@dataclass
class RolloverStudy:
    product: str
    options: RolloverOptions
    title: str
    x_from: int
    x_to: int
    lookback: int
    obs: list[LineObs]  # every line the page draws, newest first
    pending: list[LineObs]  # lines whose window has not finished (open or not started)
    current: LineObs | None  # the newest of them
    stats: WindowStats | None
    screens: list[TriggerScreen]
    case: str
    case_text: str
    rule: RuleCheck | None
    skips: list[RollSkip]  # days left out of the lines' moves
    rolls: list[RollEvent]  # front-month roll days left out of the market-movement triggers
    notes: list[str]
    unavailable: list[tuple[str, str]]
    tested: int

    @property
    def window_days(self) -> int:
        return self.x_to - self.x_from

    @property
    def finished(self) -> list[LineObs]:
        return [o for o in self.obs if o.status == DONE]


@dataclass(frozen=True)
class ScanRow:
    x_from: int
    x_to: int
    n: int
    mean: float
    hit_rate: float
    direction: int
    t_stat: float | None
    verdict: str


@dataclass
class ScanResult:
    rows: list[ScanRow]
    windows: int
    trends: int
    weak: int
    chance_trends: float | None
    note: str = ""


# ----------------------------------------------------------------------
# A line as a series that does not count its contract switches
# ----------------------------------------------------------------------


@dataclass
class Prepared:
    """A rollover line as arrays. `adjusted` is the running sum of the daily changes WITHOUT the switch days, so the move between two
    days is a plain difference of it."""

    x: np.ndarray
    price: np.ndarray
    day: pd.DatetimeIndex
    pair: np.ndarray
    is_switch: np.ndarray
    change: np.ndarray  # raw daily change, NaN on a switch day and on the first day
    adjusted: np.ndarray

    def pos(self, x: int) -> int | None:
        at = int(np.searchsorted(self.x, x))
        return at if at < len(self.x) and int(self.x[at]) == x else None

    def skips(self, first: int, last: int, label: str) -> list[RollSkip]:
        """The switch days after row `first` up to row `last`."""
        return [
            RollSkip(self.day[r].date(), label, str(self.pair[r - 1]), str(self.pair[r]), float(self.price[r] - self.price[r - 1]))
            for r in range(first + 1, last + 1) if self.is_switch[r]
        ]


def prepare(line: RolloverLine) -> Prepared:
    frame = line.frame
    x = frame.index.to_numpy(dtype=int)
    price = frame["price"].to_numpy(dtype=float)
    pair = frame["pair"].astype(str).to_numpy()
    is_switch = np.zeros(len(x), dtype=bool)
    is_switch[1:] = pair[1:] != pair[:-1]
    raw = np.full(len(x), np.nan)
    raw[1:] = price[1:] - price[:-1]
    counted = np.where(is_switch, 0.0, np.nan_to_num(raw))
    counted[:1] = 0.0
    change = np.where(is_switch, np.nan, raw)
    return Prepared(x, price, pd.DatetimeIndex(frame["date"]), pair, is_switch, change, np.cumsum(counted))


# ----------------------------------------------------------------------
# The trigger catalogue
# ----------------------------------------------------------------------


def leg_products(product: str) -> tuple[str, ...]:
    """The products whose own settlements feed the triggers: a CL-BZ box has CL and Brent (BRN stands in for BZ)."""
    spec = product_spec(product)
    return tuple("BRN" if leg == "BZ" else leg for leg in spec.legs) if spec.is_box else (product,)


def trigger_specs(product: str, lookback: int) -> list[TriggerSpec]:
    n = lookback
    box = product_spec(product).is_box
    first = "front contract" if box else "structure's first month"
    specs = [
        TriggerSpec("level_hist", "This line's price vs earlier expiries at this point (z)", GROUP_LINE,
                    "Rich or cheap against the same point of earlier expiries (only earlier ones are used, so it needs 3)."),
        TriggerSpec("momentum", f"This line's change over the prior {n} days", GROUP_LINE,
                    "A move already under way can continue or be exhausted. Contract switches are not counted."),
        TriggerSpec("vol", f"This line's daily volatility over the prior {n} days", GROUP_LINE,
                    "A quiet market and a volatile one can behave differently. Contract switches are not counted."),
    ]
    for p in leg_products(product):
        name = LEG_NAMES.get(p, p)
        if not box:
            specs.append(TriggerSpec(f"curve_leg_{p}", f"{name} spread of the {first} (M - M+1)", GROUP_CURVE,
                                     f"{name} backwardation (+) or contango (-) at the line's own month."))
        specs += [
            TriggerSpec(f"curve_mkt_{p}", f"{name} front-month spread (M1 - M2)", GROUP_CURVE,
                        f"The {name} market's overall curve shape on the entry day."),
            TriggerSpec(f"price_{p}", f"{name} outright price ({first})", GROUP_PRICE,
                        f"{name} flat price: high-price and low-price regimes change how spreads behave."),
            TriggerSpec(f"oi_chg_{p}", f"{name} open interest change over the prior {n} days", GROUP_OI,
                        f"Open interest of the {first}: money entering or leaving {name}."),
            TriggerSpec(f"oi_own_{p}", f"{name} open interest vs its own last {OWN_DAYS} days", GROUP_OI,
                        f"Is {name} open interest high or low against its own recent level?"),
            TriggerSpec(f"oi_hist_{p}", f"{name} open interest vs earlier expiries at this point", GROUP_OI,
                        f"{name} open interest against the same point of earlier expiries (needs 3 earlier ones)."),
            TriggerSpec(f"mkt_move_{p}", f"{name} front-month move over the prior {n} days", GROUP_MARKET,
                        f"Market direction. The first trading day after each {name} expiry is not counted."),
            TriggerSpec(f"mkt_vol_{p}", f"{name} front-month daily volatility over the prior {n} days", GROUP_MARKET,
                        f"Market volatility. The first trading day after each {name} expiry is not counted."),
        ]
    if box:
        specs.append(TriggerSpec("mkt_move_box", f"CL front minus Brent front, move over the prior {n} days", GROUP_MARKET,
                                 "How the headline WTI-Brent spread moved. The first trading day after a CL or a Brent expiry is not counted."))
    return sorted(specs, key=lambda s: GROUP_ORDER.index(s.group))


def _month_year(index: int) -> tuple[int, int]:
    return index % 12 + 1, index // 12


def value_at(series: pd.Series | None, when: pd.Timestamp) -> float | None:
    """The series' value on `when`, or its last value within a few days before it; None when there is none."""
    if series is None or series.empty:
        return None
    pos = int(series.index.searchsorted(when, side="right")) - 1
    if pos < 0 or (when - series.index[pos]).days > VALUE_TOLERANCE_DAYS:
        return None
    value = series.iloc[pos]
    return float(value) if np.isfinite(value) else None


# ----------------------------------------------------------------------
# Reading the lines
# ----------------------------------------------------------------------


class LineReader:
    """Reads every line of one selection: its move over the window and the triggers on its entry day."""

    def __init__(self, product: str, options: RolloverOptions, legs: dict[str, LegData | None], lookback: int):
        self.product, self.options, self.legs, self.lookback = product, options, legs, lookback
        self.box = product_spec(product).is_box
        self.leg_products = leg_products(product)
        self.specs = trigger_specs(product, lookback)
        self._clean: dict[tuple[str, str, str], pd.Series] = {}
        self.rolls_seen: dict[tuple[str, date], RollEvent] = {}
        cl, brent = legs.get("CL"), legs.get("BRN")
        self.box_series: ChangeSeries | None = difference(cl.series, brent.series, "CL front - Brent front") if self.box and cl and brent else None

    # -- helpers -----------------------------------------------------------

    def _series(self, product: str, kind: str, symbol: str) -> pd.Series | None:
        leg = self.legs.get(product)
        if leg is None or symbol not in leg.close.columns:
            return None
        key = (product, kind, symbol)
        if key not in self._clean:
            self._clean[key] = (leg.close if kind == "close" else leg.oi)[symbol].dropna()
        return self._clean[key]

    def _contracts(self, product: str, line: RolloverLine, entry: pd.Timestamp) -> tuple[str, str] | None:
        """The contract and the one after it that the leg's triggers are read on: the front pair for a CL-BZ line (whose own pair
        changes), otherwise the line's own first month."""
        leg = self.legs.get(product)
        if leg is None:
            return None
        if self.box:
            pos = int(leg.index.searchsorted(entry, side="right")) - 1
            if pos < 0 or (entry - leg.index[pos]).days > VALUE_TOLERANCE_DAYS or leg.front[pos] < 0 or leg.front[pos] + 1 >= len(leg.ordered):
                return None
            return leg.ordered[leg.front[pos]], leg.ordered[leg.front[pos] + 1]
        first = line.base + self.options.generic - 1
        return api_symbol(product, *_month_year(first)), api_symbol(product, *_month_year(first + 1))

    def _note_rolls(self, rolls) -> None:
        for stamp in rolls:
            for product, leg in self.legs.items():
                event = leg.event_on(stamp.date()) if leg is not None else None
                if event is not None:
                    self.rolls_seen[(product, event.day)] = event

    # -- triggers ----------------------------------------------------------

    def _leg_triggers(self, product: str, line: RolloverLine, entry: pd.Timestamp, out: dict, raw: dict) -> None:
        leg = self.legs.get(product)
        contracts = self._contracts(product, line, entry)
        if leg is None or contracts is None:
            return
        first, second = contracts
        price_first = value_at(self._series(product, "close", first), entry)
        price_second = value_at(self._series(product, "close", second), entry)
        out[f"price_{product}"] = price_first
        if not self.box:
            out[f"curve_leg_{product}"] = None if price_first is None or price_second is None else price_first - price_second

        pos = int(leg.index.searchsorted(entry, side="right")) - 1
        if pos >= 0 and (entry - leg.index[pos]).days <= VALUE_TOLERANCE_DAYS and leg.front[pos] >= 0 and leg.front[pos] + 1 < len(leg.ordered):
            near, far = leg.close.iloc[pos, leg.front[pos]], leg.close.iloc[pos, leg.front[pos] + 1]
            out[f"curve_mkt_{product}"] = float(near - far) if np.isfinite(near) and np.isfinite(far) else None

        oi = self._series(product, "oi", first)
        if oi is not None:
            at = int(oi.index.searchsorted(entry, side="right")) - 1
            if at >= 0 and (entry - oi.index[at]).days <= VALUE_TOLERANCE_DAYS:
                now = float(oi.iloc[at])
                raw[f"oi_{product}"] = now
                if at - self.lookback >= 0 and oi.iloc[at - self.lookback] > 0:
                    out[f"oi_chg_{product}"] = float(now / oi.iloc[at - self.lookback] - 1)
                before = oi.iloc[max(0, at - OWN_DAYS):at]
                if len(before) >= OWN_DAYS // 2 and before.mean() > 0:
                    out[f"oi_own_{product}"] = float(now / before.mean() - 1)

        if leg.series is not None:
            read = leg.series.read(entry, self.lookback)
            if read is not None:
                out[f"mkt_move_{product}"], out[f"mkt_vol_{product}"] = read.total, read.std
                self._note_rolls(read.rolls)

    def _triggers(self, line: RolloverLine, prep: Prepared, i: int) -> tuple[dict, dict]:
        entry = pd.Timestamp(prep.day[i])
        out: dict[str, float | None] = {spec.key: None for spec in self.specs}
        raw: dict[str, float] = {"level": float(prep.price[i])}
        n = self.lookback
        if i - n >= 0:
            out["momentum"] = float(prep.adjusted[i] - prep.adjusted[i - n])
            window = prep.change[i - n + 1:i + 1]
            valid = window[np.isfinite(window)]
            if len(valid) >= max(3, math.ceil(0.6 * n)):
                out["vol"] = float(valid.std(ddof=1))
        for product in self.leg_products:
            self._leg_triggers(product, line, entry, out, raw)
        if self.box_series is not None:
            read = self.box_series.read(entry, n)
            if read is not None:
                out["mkt_move_box"] = read.total
                self._note_rolls(read.rolls)
        return out, raw

    def _attribution(self, entry: pd.Timestamp, exit_: pd.Timestamp, move: float) -> dict[str, float] | None:
        cl, brent = self.legs.get("CL"), self.legs.get("BRN")
        if not self.box or cl is None or brent is None:
            return None
        cl_move, brent_move = total_between(cl.series, entry, exit_), total_between(brent.series, entry, exit_)
        if cl_move is None or brent_move is None:
            return None
        return {"CL": cl_move, "Brent": brent_move, "residual": move - (cl_move - brent_move)}

    # -- one line, and all of them -------------------------------------------

    def observe_line(self, line: RolloverLine, x_from: int, x_to: int) -> LineObs:
        obs = LineObs(line.label, line.anchor, line.live, MISSING, triggers={spec.key: None for spec in self.specs})
        prep = prepare(line)
        if len(prep.x) == 0:
            return obs
        if x_from > int(prep.x[-1]):
            obs.status = AHEAD
            return obs
        i = prep.pos(x_from)
        if i is None:
            return obs
        finished = x_to <= int(prep.x[-1])
        j = prep.pos(x_to) if finished else None
        if finished and j is None or not finished and not line.live:
            return obs
        obs.entry_date, obs.start_price = prep.day[i].date(), float(prep.price[i])
        if finished:
            obs.status, obs.exit_date, obs.end_price = DONE, prep.day[j].date(), float(prep.price[j])
            obs.move, obs.raw_move = float(prep.adjusted[j] - prep.adjusted[i]), float(prep.price[j] - prep.price[i])
            obs.skipped = prep.skips(i, j, line.label)
            obs.attribution = self._attribution(pd.Timestamp(prep.day[i]), pd.Timestamp(prep.day[j]), obs.move)
        else:
            obs.status, obs.so_far = OPEN, float(prep.adjusted[-1] - prep.adjusted[i])
        obs.triggers, obs.raw = self._triggers(line, prep, i)
        return obs

    def observe(self, lines: list[RolloverLine], x_from: int, x_to: int) -> list[LineObs]:
        """A LineObs per line, in the order given (newest first); triggers that compare with other expiries use only earlier ones."""
        observations = [self.observe_line(line, x_from, x_to) for line in lines]
        self._cross_line(observations)
        return observations

    def _cross_line(self, observations: list[LineObs]) -> None:
        for obs in observations:
            if "level" not in obs.raw:
                continue
            earlier = [o for o in observations if o.anchor < obs.anchor and "level" in o.raw]
            levels = np.array([o.raw["level"] for o in earlier])
            if len(levels) >= MIN_EARLIER and levels.std(ddof=1) > 0:
                obs.triggers["level_hist"] = float((obs.raw["level"] - levels.mean()) / levels.std(ddof=1))
            for product in self.leg_products:
                values = [o.raw[f"oi_{product}"] for o in earlier if f"oi_{product}" in o.raw]
                if f"oi_{product}" in obs.raw and len(values) >= MIN_EARLIER and np.median(values) > 0:
                    obs.triggers[f"oi_hist_{product}"] = float(obs.raw[f"oi_{product}"] / np.median(values) - 1)


# ----------------------------------------------------------------------
# Assembling a study
# ----------------------------------------------------------------------

_VERDICT_ORDER = {STRONG: 0, POSSIBLE: 1, NO_EVIDENCE: 2, CONSTANT: 3, TOO_FEW: 4}


def _classify(stats: WindowStats | None, screens: list[TriggerScreen]) -> str:
    if stats is None or stats.n < MIN_STUDY_YEARS:
        return CASE_TOO_FEW
    if stats.verdict == TREND:
        return CASE_TREND
    if any(s.result.verdict in (STRONG, POSSIBLE) and s.result.split.flips for s in screens):
        return CASE_MIXED
    return CASE_NONE


def direction_word(sign: int) -> str:
    return "up" if sign > 0 else "down" if sign < 0 else "flat"


def describe_case(case: str, stats: WindowStats | None, screens: list[TriggerScreen], tested: int) -> str:
    good = [s for s in screens if s.result.verdict in (STRONG, POSSIBLE)]
    if case == CASE_TOO_FEW:
        n = 0 if stats is None else stats.n
        return f"Only {n} earlier expiries have a finished window here; at least {MIN_STUDY_YEARS} are needed before triggers can be screened."
    if case == CASE_TREND:
        text = f"A consistent move: {stats.up} up and {stats.down} down in {stats.n} expiries, usually {direction_word(stats.direction)}."
        if good:
            return text + f" What separates strong expiries from weak ones: {', '.join(s.spec.label for s in good[:3])}."
        return text + " No trigger explains why some expiries moved more than others."
    if case == CASE_MIXED:
        best = next(s for s in good if s.result.split.flips)
        split, result = best.result.split, best.result
        how = "explains it" if result.verdict == STRONG else "may explain it"
        return (
            f"The expiries do not agree ({stats.up} up, {stats.down} down), but '{best.spec.label}' {how} (p = {result.p_value:.3f}, "
            f"q = {result.q_value:.2f}): when it was low the window moved {split.low_mean:+.3f} on average ({split.low_n} expiries), when it "
            f"was high {split.high_mean:+.3f} ({split.high_n} expiries). That would be a move that depends on this trigger rather than a "
            "missing one; with so few expiries, treat it as a hypothesis to forward-test, not a finding."
        )
    if good:
        best = good[0].result
        return (
            f"No consistent move ({stats.up} up, {stats.down} down). '{good[0].spec.label}' stands out ({best.verdict}: p = {best.p_value:.3f}, "
            f"q = {best.q_value:.2f}) but it does not split the expiries into groups that move in opposite directions: at most it separates "
            f"bigger moves from smaller ones. With {tested} triggers tried, that is as likely to be chance as a mechanism."
        )
    return (
        f"No consistent move ({stats.up} up, {stats.down} down) and none of the {tested} triggers tested explains the difference between "
        "expiries. With so few expiries that is as likely to be noise as a mechanism."
    )


def _rule_check(screens: list[TriggerScreen], pairs: dict[str, tuple[list[float], list[float]]]) -> RuleCheck | None:
    good = [s for s in screens if s.result.verdict in (STRONG, POSSIBLE)]
    if not good:
        return None
    best = min(good, key=lambda s: (s.result.p_value, -abs(s.result.rho)))
    x, y = pairs[best.spec.key]
    fit = fit_rule(x, y)
    if fit is None:
        return None
    return RuleCheck(best.spec, fit, score_rule(x, y), best.current, None if best.current is None else fit.direction(best.current))


def study_title(product: str, options: RolloverOptions) -> str:
    spec = product_spec(product)
    if spec.is_box:
        box = {"outright": "outbox", "spread": "spreadbox", "fly": "flybox", "dfly": "dflybox"}[options.family]
        what = f"front {box}" if options.generic == 1 else f"generic {options.generic} {box}"
    else:
        what = f"generic {options.generic} {FAMILY_LABELS[options.family].lower()}"
    return f"{spec.name}: {what}"


def assemble_study(
    reader: LineReader, lines: list[RolloverLine], x_from: int, x_to: int, notes: list[str],
) -> RolloverStudy:
    observations = reader.observe(lines, x_from, x_to)
    done = [o for o in observations if o.status == DONE]
    pending = [o for o in observations if o.status in (OPEN, AHEAD)]
    current = pending[0] if pending else None
    stats = window_stats([o.move for o in done])

    screens: list[TriggerScreen] = []
    pairs: dict[str, tuple[list[float], list[float]]] = {}
    for spec in reader.specs:
        used = [(o.triggers.get(spec.key), o.move) for o in done if o.triggers.get(spec.key) is not None]
        x, y = [u[0] for u in used], [u[1] for u in used]
        pairs[spec.key] = (x, y)
        screens.append(TriggerScreen(spec, screen(x, y), None if current is None else current.triggers.get(spec.key)))
    finish_screens([s.result for s in screens])
    screens.sort(key=lambda s: (_VERDICT_ORDER[s.result.verdict], s.result.p_value if s.result.p_value is not None else 2.0))
    tested = sum(1 for s in screens if s.result.p_value is not None)

    case = _classify(stats, screens)
    notes = list(notes)
    if product_spec(reader.product).is_box:
        notes.append("Brent triggers use ICE Brent (BRN) settlements; BZ has no stored settlements.")
    skipped = [o.label for o in observations if o.status == MISSING]
    if skipped:
        notes.append(f"The line does not cover this window for {', '.join(skipped[:8])}{' ...' if len(skipped) > 8 else ''}; those expiries are left out.")
    if tested:
        notes.append(
            f"{tested} triggers were tested, so about {tested * 0.05:.1f} would look significant at p <= 0.05 by chance alone. "
            "Trust the q column (false-discovery corrected) and the leave-one-out check, and forward-test anything you keep."
        )
    skips = sorted((s for o in done for s in o.skipped), key=lambda s: s.day, reverse=True)
    return RolloverStudy(
        reader.product, reader.options, study_title(reader.product, reader.options), x_from, x_to, reader.lookback, observations, pending,
        current, stats, screens, case, describe_case(case, stats, screens, tested), _rule_check(screens, pairs), skips,
        sorted(reader.rolls_seen.values(), key=lambda e: e.day, reverse=True), notes, EXTERNAL_TRIGGERS, tested,
    )


# ----------------------------------------------------------------------
# Scanning for windows
# ----------------------------------------------------------------------


def scan_windows(lines: list[RolloverLine], seed: int = 20261102) -> ScanResult:
    """Every window of a few lengths across the lines' x-axis, ranked by how consistently the expiries agree. Contract switches are
    not counted in any move. The number of windows that look like trends is set against how many do when each line is given a random
    sign (the same windows, no real pattern): that is what to expect by luck."""
    preps = [prepare(line) for line in lines if len(line.frame)]
    if len(preps) < MIN_STUDY_YEARS:
        return ScanResult([], 0, 0, 0, None, f"Only {len(preps)} expiries have data; scanning needs {MIN_STUDY_YEARS}.")
    low, high = min(int(p.x[0]) for p in preps), max(int(p.x[-1]) for p in preps)
    matrix = np.full((high - low + 1, len(preps)), np.nan)
    for column, prep in enumerate(preps):
        matrix[prep.x - low, column] = prep.adjusted
    windows = [(a, a + length) for length in SCAN_LENGTHS for a in range(low, high - length + 1, SCAN_STEP)]

    def evaluate(values: np.ndarray, sign_test: bool) -> list[tuple[int, int, WindowStats]]:
        found = []
        for a, b in windows:
            moves = values[b - low] - values[a - low]
            finite = moves[np.isfinite(moves)]
            stats = window_stats(finite, sign_test) if len(finite) >= MIN_STUDY_YEARS else None
            if stats is not None:
                found.append((a, b, stats))
        return found

    found = evaluate(matrix, True)
    trends = [f for f in found if f[2].verdict == TREND]
    rng = np.random.default_rng(seed)
    chance = float(np.mean([
        sum(1 for f in evaluate(matrix * rng.choice([-1.0, 1.0], size=matrix.shape[1]), False) if f[2].verdict == TREND)
        for _ in range(SCAN_CHANCE_REPS)
    ]))
    ranked = sorted(trends, key=lambda f: -abs(f[2].t_stat if f[2].t_stat is not None else 99.0))[:SCAN_SHOW]
    rows = [ScanRow(a, b, s.n, s.mean, s.hit_rate, s.direction, s.t_stat, s.verdict) for a, b, s in ranked]
    return ScanResult(rows, len(found), len(trends), sum(1 for f in found if f[2].verdict == "weak"), chance)


# ----------------------------------------------------------------------
# Saved rules and the forward test
# ----------------------------------------------------------------------


class RuleBook:
    """The rules the user chose to track, kept in the settings table."""

    def __init__(self, repository):
        self._repository = repository

    def all(self) -> list[dict]:
        rules = self._repository.get_setting(RULES_SETTING, [])
        return rules if isinstance(rules, list) else []

    def add(self, rule: dict) -> None:
        self._repository.set_setting(RULES_SETTING, [*self.all(), rule])

    def remove(self, rule_id: str) -> bool:
        kept = [r for r in self.all() if r.get("id") != rule_id]
        changed = len(kept) != len(self.all())
        if changed:
            self._repository.set_setting(RULES_SETTING, kept)
        return changed


def rule_from_study(study: RolloverStudy, today: date) -> dict:
    """A snapshot of the study's rule, frozen on `today`. It applies to every later expiry of the same selection."""
    if study.rule is None:
        raise ValueError("This study has no rule to save: no trigger passed the screening")
    options, rule = study.options, study.rule
    return {
        "id": uuid.uuid4().hex[:10], "product": study.product, "family": options.family, "generic": options.generic,
        "expiries": options.expiries, "x_from": study.x_from, "x_to": study.x_to, "lookback": study.lookback,
        "trigger": rule.spec.key, "trigger_label": rule.spec.label, "cut": rule.fit.cut, "high_dir": rule.fit.high_dir,
        "low_dir": rule.fit.low_dir, "created_on": today.isoformat(), "label": study.title,
    }


def forward_rows(rule: dict, observations: list[LineObs]) -> list[dict]:
    """How a saved rule did on the expiries since it was saved: open (or not started) lines, and finished ones that began after it was saved.

    A line whose entry day came before the rule was saved is marked "no": it was known when the rule was written, so it is not a test."""
    created = date.fromisoformat(rule["created_on"])
    rows = []
    for obs in sorted(observations, key=lambda o: o.anchor):
        if obs.status == MISSING or (obs.status == DONE and (obs.entry_date is None or obs.entry_date < created)):
            continue
        reading = obs.triggers.get(rule["trigger"])
        call = 0 if reading is None else (rule["high_dir"] if reading > rule["cut"] else rule["low_dir"])
        move = obs.move if obs.status == DONE else obs.so_far
        row = {
            "rule": rule["id"], "expiry": obs.label, "forward": "", "entry": obs.entry_date.isoformat() if obs.entry_date else None,
            "reading": reading, "expect": "", "status": "waiting: the window has not started", "move": move, "result": "",
        }
        if obs.status != AHEAD:
            row.update(
                forward="yes" if obs.entry_date >= created else "no (entry before the rule was saved)",
                expect=direction_word(call) if call else "no call", status="finished" if obs.status == DONE else "open",
            )
            if obs.status == DONE and call:
                row["result"] = "HIT" if call * obs.move > 0 else "MISS"
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------


class RolloverStudyService:
    """Trigger studies, window scans and the forward test on the lines of the Strategy Rollover page."""

    def __init__(self, rollover: RolloverService, repository=None):
        self._rollover = rollover
        self._rules = RuleBook(repository) if repository is not None else None
        self._legs: dict[str, tuple[str, LegData | None]] = {}
        self._studies: dict[tuple, RolloverStudy] = {}

    # -- data ----------------------------------------------------------------

    def _leg(self, product: str) -> LegData | None:
        version, frame = self._rollover.settlements(product)
        cached = self._legs.get(product)
        if cached is None or cached[0] != version:
            self._legs[product] = (version, build_leg_data(product, frame))
            self._studies.clear()
        return self._legs[product][1]

    def _versions(self, product: str) -> tuple:
        return tuple(self._rollover.settlements(p)[0] for p in dict.fromkeys((product, *leg_products(product))))

    @staticmethod
    def study_options(options: RolloverOptions, x_from: int, lookback: int, lines: int) -> RolloverOptions:
        """The page's selection, drawn for the study: more expiries, no zero base (it restarts the price at every later expiry line),
        the expiry that has not come yet included, and enough days before the expiry for the entry day's lookback."""
        before = min(MAX_BEFORE, max(options.before_days, lookback - x_from + 2))
        return replace(options, comparisons=lines, before_days=before, zero_base=False, upcoming=True)

    @staticmethod
    def check(x_from: int, x_to: int, lookback: int, lines: int) -> None:
        if not MIN_X <= x_from < x_to <= MAX_X:
            raise ValueError(f"From must be earlier than To, both between {MIN_X} and {MAX_X} trading days from the front expiry: got {x_from} and {x_to}")
        if not MIN_LOOKBACK <= lookback <= MAX_LOOKBACK:
            raise ValueError(f"Lookback must be a whole number from {MIN_LOOKBACK} to {MAX_LOOKBACK} days")
        if not 1 <= lines <= MAX_LINES:
            raise ValueError(f"Expiries studied must be a whole number from 1 to {MAX_LINES}")

    def _lines(self, product: str, options: RolloverOptions, x_from: int, lookback: int, lines: int):
        used = self.study_options(options, x_from, lookback, lines)
        result = self._rollover.view(product, used)
        return used, result

    # -- public ----------------------------------------------------------------

    def study(self, product: str, options: RolloverOptions, x_from: int, x_to: int, lookback: int, lines: int) -> RolloverStudy:
        self.check(x_from, x_to, lookback, lines)
        used, result = self._lines(product, options, x_from, lookback, lines)
        legs = {p: self._leg(p) for p in leg_products(product)}
        key = (self._versions(product), self._rollover.today(), product, used.key(), x_from, x_to, lookback)
        study = self._studies.get(key)
        if study is None:
            if len(self._studies) > 60:
                self._studies.clear()
            study = self._studies[key] = assemble_study(LineReader(product, used, legs, lookback), result.lines, x_from, x_to, result.notes)
        return study

    def scan(self, product: str, options: RolloverOptions, lookback: int, lines: int) -> ScanResult:
        self.check(DEFAULT_FROM, DEFAULT_TO, lookback, lines)
        _, result = self._lines(product, options, DEFAULT_FROM, lookback, lines)
        return scan_windows(result.lines)

    # -- saved rules -----------------------------------------------------------

    def rules(self) -> list[dict]:
        return self._rules.all() if self._rules is not None else []

    def save_rule(self, study: RolloverStudy) -> dict:
        if self._rules is None:
            raise ValueError("Saving rules needs the settings database")
        rule = rule_from_study(study, self._rollover.today())
        self._rules.add(rule)
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        return self._rules.remove(rule_id) if self._rules is not None else False

    def forward_test(self, lines: int = DEFAULT_LINES) -> list[dict]:
        rows = []
        for rule in self.rules():
            try:
                options = RolloverOptions(generic=int(rule["generic"]), expiries=int(rule["expiries"]), family=rule["family"])
                study = self.study(rule["product"], options, int(rule["x_from"]), int(rule["x_to"]), int(rule["lookback"]), lines)
                label = f"{rule['label']}: {rule['trigger_label']}, {rule['x_from']}→{rule['x_to']} days"
                rows.extend({**row, "rule_label": label} for row in forward_rows(rule, study.obs))
            except (KeyError, ValueError):
                continue
        return rows
