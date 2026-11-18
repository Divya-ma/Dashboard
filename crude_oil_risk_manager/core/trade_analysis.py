"""Trade Analyzer: how likely is a trade in a structure to reach its target before its stop, and where to put them.

You give a structure (legs and ratios, as when building one; a leg may itself be a spread such as BRNJ27-K27), a side, lots and your risk per trade. The
analysis never looks at anything but daily settlements (core.settlement_prices) and is built in four parts.

1. ANALOGS. The same structure of EARLIER delivery years (BRNJ27-K27 -> BRNJ26-K26, BRNJ25-K25 ...) is priced from its outrights'
   settlements. Each earlier year contributes every day whose number of trading days before the front leg's expiry is within
   `window` of today's. From each such day the trade is replayed forward day by day, as bought or sold (`paths`: the profit in
   points after 1, 2 ... horizon days). An earlier year is never today's own contract, so nothing here peeks at the future.
2. OUTCOME. For a stop S and a target T (points), a replayed trade WINS if its profit reaches +T before it reaches -S, LOSES if
   it reaches -S first, and is UNRESOLVED if neither happens within the horizon or the front leg expires first. Settlements are
   daily, so a day that reaches both counts as a loss. P(win) is wins / (wins + losses): the target must be met before the stop,
   however long that takes (up to the horizon); the unresolved share is shown next to it.
3. CONTEXT. A structure behaves differently when crude is 40 or 120 dollars, when WTI is cheap to Brent, or when the gasoil crack
   is wide. Each replayed day carries the front CL, BRN and gasoil settlements, the front CL-BZ box (WTI minus Brent) and the
   gasoil crack (gasoil / 7.45 - Brent); a day whose context is close to today's gets more weight (a Gaussian kernel on the
   standardised distance). The result is shown with and without that weighting, with the effective number of days behind it.
4. UNCERTAINTY. Replayed days of one year overlap (the next day's trade is nearly the same trade), so the confidence interval
   resamples whole YEARS (a block bootstrap), not days. A year-level interval is wide by design: few years, few independent
   experiments.

The stop is capped so that lots x stop x point value fits the risk per trade. Suggested stops come from how far
replayed trades went against you (percentiles of the worst adverse move), targets from how far they went your way and from
reward:risk multiples; each pair is scored by its expectancy in dollars. A driftless random walk with the same daily volatility
is simulated as a benchmark: where history and the random walk disagree, the structure is not behaving like a coin flip.

Portfolio: the candidate's daily P&L (lots x side x point value) is correlated with the open book's, and the book's 1-day
historical VaR is recomputed with the candidate added, so you see whether the trade adds to or takes away from the risk you hold.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from core.contract_specs import spec_for_symbol
from core.curve_calendar import contract_expiry, product_spec
from core.exceptions import CrudeOilRiskError
from core.positions import dollar_weights
from core.seasonality import trading_days_to
from core.structure_utils import compose_structure_symbol, decompose_to_outrights, normalize_symbol, parse_symbol
from core.var import historical_var, pnl_series

logger = logging.getLogger(__name__)

DEFAULT_HORIZON = 60
MAX_HORIZON = 250
DEFAULT_WINDOW = 7  # trading days either side of today's days-to-expiry that an earlier year may be entered on
MAX_YEARS_BACK = 12
MIN_YEARS = 3  # earlier years with replayed trades before any probability is given
MIN_PATHS = 20
BOOTSTRAP = 400
RANDOM_WALK_PATHS = 20_000
KERNEL_BANDWIDTH = 1.0
GASOIL_BARRELS_PER_TONNE = 7.45
STOP_QUANTILES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TARGET_QUANTILES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
REWARD_RISK = (1.0, 1.5, 2.0, 3.0)
MAX_GRID_ROWS = 60
MIN_RESOLVED_SHARE = 0.5  # a pair is only recommended if at least this share of replayed trades hit the stop or the target
VAR_LOOKBACK = 250
CONTEXT_NAMES = {
    "cl": "Front CL", "brn": "Front BRN", "go": "Front gasoil", "wti_brent": "CL-BZ (WTI minus Brent)", "crack": "Gasoil crack ($/bbl)",
}


KEY_RISK_PER_TRADE = "trade_analyzer_risk_per_trade"


def load_risk_per_trade(repository) -> float:
    """Risk per trade in dollars: the value saved from the Trade Analyzer page, else the trade-plan risk per trade in Settings."""
    saved = repository.get_setting(KEY_RISK_PER_TRADE, None)
    if isinstance(saved, (int, float)) and saved > 0:
        return float(saved)
    from core.curve_settings import load_params

    return float(load_params(repository).risk_per_trade)


def save_risk_per_trade(repository, risk_trade: float) -> None:
    repository.set_setting(KEY_RISK_PER_TRADE, float(risk_trade))


class TradeAnalysisError(CrudeOilRiskError):
    """A problem with the inputs or too little history; the message is safe to show."""


@dataclass(frozen=True)
class TradeSpec:
    legs: tuple  # ((leg symbol, ratio), ...): a leg may itself be a spread or a fly (BRNJ27-K27); ratios as on the Structures page
    side: int  # +1 buy the legs as entered, -1 sell them
    lots: float
    risk_trade: float  # dollars
    horizon: int = DEFAULT_HORIZON
    window: int = DEFAULT_WINDOW
    entry: float | None = None
    stop: float | None = None  # points away from the entry, optional: your own pair is scored too
    target: float | None = None

    @property
    def symbol(self) -> str:
        """The exchange-style symbol of the legs (BRNJ27-K27), else the legs written out."""
        symbols, ratios = [s for s, _ in self.legs], [r for _, r in self.legs]
        return compose_structure_symbol(symbols, [int(r) for r in ratios]) or " / ".join(f"{r:+g}x{s}" for s, r in self.legs)


@dataclass
class Paths:
    """Replayed trades: profit in points after 1..horizon days, one row per (earlier year, entry day)."""

    profit: np.ndarray  # shape (n, horizon), NaN where the structure no longer exists
    year: np.ndarray  # delivery year of the front leg of the analog
    entry_date: list
    context: pd.DataFrame  # one row per path
    labels: dict[int, str] = field(default_factory=dict)
    daily_sd: float = 0.0


@dataclass
class Outcome:
    stop: float
    target: float
    p_win: float  # wins / (wins + losses)
    p_low: float
    p_high: float
    win_share: float  # of all replayed trades
    loss_share: float
    unresolved_share: float
    expectancy: float  # dollars per trade for the lots entered (unresolved trades closed at their last price)
    expectancy_low: float  # the low end (5th percentile) of its interval
    reward_risk: float
    max_lots: int


# ----------------------------------------------------------------------
# The structure
# ----------------------------------------------------------------------


def outright_weights(legs) -> dict[str, float]:
    """The outright contracts of the legs with their weights per lot bought: each leg's own outrights (a spread +1, -1; a fly +1, -2, +1)
    times its ratio. A leg is ((symbol, ratio)) as entered on the page."""
    parts: dict[str, float] = {}
    for symbol, ratio in legs:
        try:
            split = decompose_to_outrights(normalize_symbol(symbol), float(ratio))
        except ValueError as exc:
            raise TradeAnalysisError(str(exc)) from None
        for outright, weight in split.items():
            parts[outright] = parts.get(outright, 0.0) + weight
    parts = {k: v for k, v in parts.items() if abs(v) > 1e-12}
    if not parts:
        raise TradeAnalysisError("These legs cancel out: there is no net outright exposure.")
    return parts


def _leg(outright: str) -> dict:
    return parse_symbol(outright)[0]


def _delivery(outright: str) -> tuple[int, int]:
    from adapters.base import LETTER_TO_MONTH

    leg = _leg(outright)
    return 2000 + int(leg["year"]), LETTER_TO_MONTH[leg["letter"]]


def shift_years(outright: str, years: int) -> str:
    """The same outright `years` earlier (CLF27 -> CLF26)."""
    match = re.fullmatch(r"(.*?)(\d{2})", outright)
    if not match:
        raise TradeAnalysisError(f"Cannot read the year of {outright}.")
    return f"{match.group(1)}{int(match.group(2)) - years:02d}"


def front_outright(parts: dict[str, float]) -> str:
    """The outright of the structure that expires first (its 'front leg')."""
    def expiry(outright):
        year, month = _delivery(outright)
        return contract_expiry(_leg(outright)["product"], month, year)

    return min(parts, key=expiry)


def point_value(parts: dict[str, float]) -> float:
    """Dollars per 1.0 point of the structure's price per lot."""
    return spec_for_symbol(front_outright(parts)).multiplier


def days_to_front_expiry(parts: dict[str, float], today: date) -> int:
    outright = front_outright(parts)
    year, month = _delivery(outright)
    product = _leg(outright)["product"]
    expiry = contract_expiry(product, month, year)
    index = pd.DatetimeIndex([pd.Timestamp(today)])
    return int(trading_days_to(index, expiry, product_spec(product).calendar)[0])


def _closes(settlements, outright: str) -> pd.Series | None:
    series, _proxied = settlements.closes(outright)
    if series is None or series.empty:
        return None
    series = series.copy()
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    return series


def structure_price(settlements, parts: dict[str, float]) -> pd.Series | None:
    """The structure's settlement price on every day all its outrights settled."""
    levels = []
    for outright, weight in parts.items():
        series = _closes(settlements, outright)
        if series is None:
            return None
        levels.append(series * weight)
    price = pd.concat(levels, axis=1, join="inner").dropna().sum(axis=1).sort_index()
    return price if len(price) >= 2 else None


def default_entry(settlements, parts: dict[str, float], live: dict[str, float] | None) -> tuple[float | None, str]:
    """(entry price, where it came from): the live price when every outright has one, else the last settlement."""
    if live and all(o in live for o in parts):
        return sum(w * live[o] for o, w in parts.items()), "live price"
    price = structure_price(settlements, parts)
    if price is None:
        return None, "none"
    return float(price.iloc[-1]), f"settlement of {price.index[-1]:%Y-%m-%d}"


# ----------------------------------------------------------------------
# Context
# ----------------------------------------------------------------------


def context_frame(settlements) -> pd.DataFrame:
    """Daily front-month context: CL, BRN, gasoil, the CL-BZ box (WTI minus Brent) and the gasoil crack. Empty if there is none."""
    def front(product):
        series = settlements.front_prices(product)
        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize() if len(series) and series.index.tz is not None else series.index
        return series

    cl, brn, go = front("CL"), front("BRN"), front("G")
    if cl.empty or brn.empty:
        return pd.DataFrame()
    frame = pd.DataFrame({"cl": cl, "brn": brn})
    box = front("CLBZ")
    frame["wti_brent"] = box if not box.empty else frame["cl"] - frame["brn"]
    if not go.empty:
        frame["go"] = go
        frame["crack"] = go / GASOIL_BARRELS_PER_TONNE - frame["brn"]
    return frame.sort_index().ffill(limit=3).dropna()


# ----------------------------------------------------------------------
# Replaying earlier years
# ----------------------------------------------------------------------


def build_paths(settlements, parts: dict[str, float], side: int, today: date, horizon: int, window: int, context: pd.DataFrame) -> Paths:
    """Replay the trade from every comparable day of every earlier year (see the module docstring, part 1)."""
    td_now = days_to_front_expiry(parts, today)
    if td_now < 1:
        raise TradeAnalysisError("The front leg of this structure has expired or expires today.")
    front = front_outright(parts)
    product = _leg(front)["product"]
    calendar = product_spec(product).calendar
    rows, years, dates, ctx_rows, labels, daily_changes = [], [], [], [], {}, []
    for k in range(1, MAX_YEARS_BACK + 1):
        shifted = {shift_years(o, k): w for o, w in parts.items()}
        price = structure_price(settlements, shifted)
        front_closes = _closes(settlements, shift_years(front, k))
        if price is None or front_closes is None:
            continue
        expiry = front_closes.index.max().date()  # a finished contract expires on its last settlement day
        td = trading_days_to(price.index, expiry, calendar)
        values = price.to_numpy(dtype=float)
        daily_changes.append(np.diff(values))
        picked = [i for i in range(len(values) - 1) if td[i] >= 1 and abs(td[i] - td_now) <= window]
        if not picked:
            continue
        year = _delivery(shift_years(front, k))[0]
        labels[year] = "/".join(shifted)
        for i in picked:
            ahead = side * (values[i + 1:i + 1 + horizon] - values[i])
            padded = np.full(horizon, np.nan)
            padded[:len(ahead)] = ahead
            rows.append(padded)
            years.append(year)
            dates.append(price.index[i])
            ctx_rows.append(context.loc[price.index[i]] if (not context.empty and price.index[i] in context.index) else None)
    if not rows:
        raise TradeAnalysisError("No earlier year of this structure has settlements around the same time to expiry.")
    ctx = pd.DataFrame([r if r is not None else pd.Series(dtype=float) for r in ctx_rows], columns=list(context.columns)) if not context.empty else pd.DataFrame(index=range(len(rows)))
    sd = float(np.nanstd(np.concatenate(daily_changes))) if daily_changes else 0.0
    return Paths(np.array(rows), np.array(years), dates, ctx, labels, sd)


def context_weights(paths: Paths, now: pd.Series | None) -> tuple[np.ndarray, float]:
    """(weight per replayed trade, effective number of trades): trades entered in a context like today's weigh more."""
    n = len(paths.profit)
    if now is None or paths.context.empty:
        return np.ones(n), float(n)
    columns = [c for c in paths.context.columns if c in now.index]
    if not columns:
        return np.ones(n), float(n)
    ctx = paths.context[columns].to_numpy(dtype=float)
    valid = ~np.isnan(ctx).any(axis=1)  # a day with no context data gets no weight
    if not valid.any():
        return np.ones(n), float(n)
    scale = ctx[valid].std(axis=0)
    scale[scale == 0] = 1.0
    distance = np.full(n, np.inf)
    distance[valid] = np.sqrt((((ctx[valid] - now[columns].to_numpy(dtype=float)) / scale) ** 2).mean(axis=1))
    weight = np.exp(-0.5 * (distance / KERNEL_BANDWIDTH) ** 2)
    total = weight.sum()
    if total <= 0:
        return np.ones(n), float(n)
    return weight, float(total ** 2 / (weight ** 2).sum())


# ----------------------------------------------------------------------
# Outcomes
# ----------------------------------------------------------------------


def resolve(profit: np.ndarray, stop: float, target: float) -> tuple[np.ndarray, np.ndarray]:
    """(won, lost) per path: target reached before the stop / stop reached first or on the same day. NaN never hits."""
    with np.errstate(invalid="ignore"):
        hit_target, hit_stop = profit >= target, profit <= -stop
    never = profit.shape[1]
    t_target = np.where(hit_target.any(axis=1), hit_target.argmax(axis=1), never)
    t_stop = np.where(hit_stop.any(axis=1), hit_stop.argmax(axis=1), never)
    won = (t_target < never) & (t_target < t_stop)
    lost = (t_stop < never) & (t_stop <= t_target)
    return won, lost


def _ratio(won_w: float, lost_w: float) -> float:
    return won_w / (won_w + lost_w) if won_w + lost_w > 0 else float("nan")


def _last_profit(profit: np.ndarray) -> np.ndarray:
    """Profit of each path on its last day with data (0 for a path with none): where an unresolved trade is closed."""
    finite = np.isfinite(profit)
    last = profit.shape[1] - 1 - np.argmax(finite[:, ::-1], axis=1)
    values = profit[np.arange(len(profit)), last]
    return np.where(finite.any(axis=1), values, 0.0)


def score(paths: Paths, weight: np.ndarray, stop: float, target: float, dollars_per_point: float, lots: float, risk_limit: float) -> Outcome:
    """P(win) and the expectancy for one stop / target pair, each with a year-block bootstrap interval.

    A trade that neither wins nor loses within the horizon is closed at its last available price, so the expectancy counts it."""
    won, lost = resolve(paths.profit, stop, target)
    result = np.where(won, target, np.where(lost, -stop, _last_profit(paths.profit)))
    total = float(weight.sum())
    w_won, w_lost = float((weight * won).sum()), float((weight * lost).sum())
    p = _ratio(w_won, w_lost)
    keys = np.unique(paths.year)
    table = np.array([[
        float((weight[paths.year == y] * won[paths.year == y]).sum()), float((weight[paths.year == y] * lost[paths.year == y]).sum()),
        float(weight[paths.year == y].sum()), float((weight[paths.year == y] * result[paths.year == y]).sum()),
    ] for y in keys])
    rng = np.random.default_rng(7)
    p_draws, e_draws = [], []
    for _ in range(BOOTSTRAP):
        s = table[rng.integers(0, len(keys), len(keys))].sum(axis=0)
        if s[0] + s[1] > 0 and s[2] > 0:
            p_draws.append(s[0] / (s[0] + s[1]))
            e_draws.append(s[3] / s[2] * dollars_per_point * lots)
    low, high = np.percentile(p_draws, [5, 95]) if p_draws else (float("nan"), float("nan"))
    e_low = float(np.percentile(e_draws, 5)) if e_draws else float("nan")
    expectancy = float((weight * result).sum() / total * dollars_per_point * lots)
    per_lot_stop = stop * dollars_per_point
    return Outcome(
        stop, target, p, float(low), float(high), w_won / total, w_lost / total, 1 - (w_won + w_lost) / total, expectancy, e_low,
        target / stop if stop else float("nan"), int(risk_limit // per_lot_stop) if per_lot_stop > 0 else 0,
    )


def random_walk(sd: float, horizon: int, stop: float, target: float) -> float:
    """P(win) of a driftless random walk with daily volatility `sd`, wins / (wins + losses); the benchmark."""
    if sd <= 0:
        return float("nan")
    steps = np.random.default_rng(11).normal(0.0, sd, (RANDOM_WALK_PATHS, horizon))
    won, lost = resolve(np.cumsum(steps, axis=1), stop, target)
    return _ratio(float(won.sum()), float(lost.sum()))


def _round(value: float) -> float:
    return float(np.round(value, 2 if abs(value) >= 1 else 3))


def candidate_grid(paths: Paths, max_stop: float) -> list[tuple[float, float]]:
    """Stop / target pairs worth scoring: stops from the worst adverse moves of replayed trades (never wider than the risk allows),
    targets from their best favourable moves and from reward:risk multiples."""
    with np.errstate(all="ignore"):
        adverse = -np.nanmin(np.where(np.isnan(paths.profit), np.inf, paths.profit), axis=1)
        favourable = np.nanmax(np.where(np.isnan(paths.profit), -np.inf, paths.profit), axis=1)
    adverse = adverse[np.isfinite(adverse) & (adverse > 0)]
    favourable = favourable[np.isfinite(favourable) & (favourable > 0)]
    stops = {_round(max_stop)}
    if len(adverse):
        stops |= {_round(q) for q in np.quantile(adverse, STOP_QUANTILES)}
    if paths.daily_sd > 0:
        stops |= {_round(m * paths.daily_sd) for m in (1.0, 1.5, 2.0, 3.0)}
    stops = sorted(s for s in stops if 0 < s <= max_stop)
    if not stops:
        stops = [_round(max_stop)]
    pairs = []
    for stop in stops:
        targets = {_round(stop * r) for r in REWARD_RISK}
        if len(favourable):
            targets |= {_round(q) for q in np.quantile(favourable, TARGET_QUANTILES)}
        pairs += [(stop, t) for t in sorted(targets) if t > 0]
    return pairs[:MAX_GRID_ROWS * 2]


@dataclass
class Analysis:
    spec: TradeSpec
    parts: dict[str, float]
    entry: float | None
    entry_source: str
    dollars_per_point: float
    max_stop: float
    days_to_expiry: int
    horizon: int
    n_paths: int
    n_years: int
    effective_n: float
    years: list[dict]
    context_now: dict[str, float]
    context_rank: dict[str, float]  # percentile of today's value among the replayed days
    unweighted: Outcome | None  # for the headline pair, every replayed day counted equally
    weighted: Outcome | None  # for the headline pair, days with a context like today's counted more
    headline: Outcome | None
    grid: list[Outcome]
    random_walk_p: float | None
    notes: list[str]
    fan: dict[str, list[float]]


def analyse(settlements, spec: TradeSpec, today: date, live: dict[str, float] | None = None) -> Analysis:
    """The whole analysis for one trade. Raises TradeAnalysisError for bad inputs or too little history."""
    if spec.side not in (1, -1):
        raise TradeAnalysisError("Choose buy or sell.")
    for name, value in (("Lots", spec.lots), ("Risk per trade", spec.risk_trade)):
        if not value or not np.isfinite(value) or value <= 0:
            raise TradeAnalysisError(f"{name} must be a number above zero.")
    horizon = int(spec.horizon)
    if not 1 <= horizon <= MAX_HORIZON:
        raise TradeAnalysisError(f"Days to give the trade must be a whole number from 1 to {MAX_HORIZON}.")
    parts = outright_weights(spec.legs)
    dollars = point_value(parts)
    risk_limit = spec.risk_trade
    max_stop = risk_limit / (spec.lots * dollars)
    entry, source = (spec.entry, "typed") if spec.entry is not None else default_entry(settlements, parts, live)
    context = context_frame(settlements)
    paths = build_paths(settlements, parts, spec.side, today, horizon, spec.window, context)
    n_years = len(np.unique(paths.year))
    if n_years < MIN_YEARS or len(paths.profit) < MIN_PATHS:
        raise TradeAnalysisError(
            f"Only {n_years} earlier year(s) with {len(paths.profit)} comparable days: too little history for a probability "
            f"(needs {MIN_YEARS} years and {MIN_PATHS} days)."
        )
    now = context.iloc[-1] if not context.empty else None
    weight, effective = context_weights(paths, now)
    flat = np.ones(len(weight))
    notes = []
    td_now = days_to_front_expiry(parts, today)
    if horizon > td_now:
        notes.append(f"The front leg expires in {td_now} trading days, shorter than the {horizon}-day horizon; trades that need longer count as unresolved.")
    if spec.stop is not None and spec.stop > max_stop + 1e-9:
        notes.append(f"Your stop of {spec.stop:g} points risks more than your limit; the widest stop that fits {spec.lots:g} lots is {max_stop:.3g} points.")

    pairs = candidate_grid(paths, max_stop)
    if spec.stop is not None and spec.target is not None:
        pairs.insert(0, (float(spec.stop), float(spec.target)))
    grid = [score(paths, weight, s, t, dollars, spec.lots, risk_limit) for s, t in dict.fromkeys(pairs)]
    grid = [g for g in grid if np.isfinite(g.p_win)]
    if not grid:
        raise TradeAnalysisError("No stop and target pair resolved in any replayed trade; try more days.")
    grid.sort(key=lambda g: g.expectancy, reverse=True)
    mine = next((g for g in grid if spec.stop is not None and spec.target is not None and (g.stop, g.target) == (float(spec.stop), float(spec.target))), None)
    resolving = [g for g in grid if 1 - g.unresolved_share >= MIN_RESOLVED_SHARE]
    safe = [g for g in resolving if g.expectancy_low > 0]
    headline = mine or (safe[0] if safe else resolving[0] if resolving else grid[0])
    if not mine and not safe:
        notes.append("No stop and target pair has a positive expectancy even at the low end of its interval: no edge found in this history.")
    if not mine and not resolving:
        notes.append(f"Fewer than {MIN_RESOLVED_SHARE:.0%} of replayed trades reach either level in {horizon} days; give the trade more days.")
    unweighted = score(paths, flat, headline.stop, headline.target, dollars, spec.lots, risk_limit)
    rank = {}
    if now is not None and not paths.context.empty:
        for name in paths.context.columns:
            rank[name] = float((paths.context[name] <= now[name]).mean() * 100)
    years = []
    for year in sorted(np.unique(paths.year)):
        m = paths.year == year
        won, lost = resolve(paths.profit[m], headline.stop, headline.target)
        years.append({
            "year": int(year), "structure": paths.labels.get(int(year), ""), "days": int(m.sum()), "won": int(won.sum()), "lost": int(lost.sum()),
            "unresolved": int(m.sum() - won.sum() - lost.sum()), "weight": float(weight[m].sum() / weight.sum()),
        })
    with np.errstate(all="ignore"):
        fan = {f"p{q}": [float(np.nanpercentile(paths.profit[:, j], q)) if np.isfinite(paths.profit[:, j]).any() else float("nan") for j in range(horizon)] for q in (10, 50, 90)}
    return Analysis(
        spec, parts, entry, source, dollars, max_stop, td_now, horizon, len(paths.profit), n_years, effective, years,
        {} if now is None else {k: float(v) for k, v in now.items()}, rank, unweighted, score(paths, weight, headline.stop, headline.target, dollars, spec.lots, risk_limit),
        headline, grid[:MAX_GRID_ROWS], random_walk(paths.daily_sd, horizon, headline.stop, headline.target), notes, fan,
    )


# ----------------------------------------------------------------------
# The open book
# ----------------------------------------------------------------------


@dataclass
class PortfolioEffect:
    open_structures: int
    correlation: float | None
    var_before: dict[float, float]
    var_after: dict[float, float]
    observations: int
    message: str


def portfolio_effect(spec: TradeSpec, open_structures: list, data_loader, get_contract=None, lookback: int = VAR_LOOKBACK) -> PortfolioEffect:
    """Correlation of the trade's daily P&L with the open book's and the change in the book's 1-day historical VaR if it is added."""
    book = dollar_weights(open_structures)
    if not book:
        return PortfolioEffect(0, None, {}, {}, 0, "No open positions, so there is nothing to correlate with.")
    candidate: dict[str, float] = {}  # dollars per point, per leg as quoted (as the open book is)
    for leg_symbol, ratio in spec.legs:
        symbol = normalize_symbol(leg_symbol)
        contract = get_contract(symbol) if get_contract is not None else None
        multiplier = contract.multiplier if contract is not None else spec_for_symbol(symbol).multiplier
        candidate[symbol] = candidate.get(symbol, 0.0) + spec.side * spec.lots * float(ratio) * multiplier
    try:
        combined = dict(book)
        for key, value in candidate.items():
            combined[key] = combined.get(key, 0.0) + value
        before = historical_var(book, lookback, data_loader)
        after = historical_var({k: v for k, v in combined.items() if abs(v) > 1e-9}, lookback, data_loader)
        book_pnl, _ = pnl_series(book, data_loader)
        trade_pnl, skipped = pnl_series(candidate, data_loader)
    except Exception:  # noqa: BLE001 - a broken history must give a message, not a stack trace
        logger.exception("Portfolio effect failed")
        return PortfolioEffect(len(open_structures), None, {}, {}, 0, "Could not work out the effect on the portfolio; see the log.")
    if trade_pnl is None or book_pnl is None:
        return PortfolioEffect(len(open_structures), None, {}, {}, 0, "No price history for this structure, so its portfolio effect cannot be measured.")
    joined = pd.concat([book_pnl, trade_pnl], axis=1, join="inner").dropna().iloc[-lookback:]
    corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1])) if len(joined) >= 10 else None
    if before.error or after.error:
        return PortfolioEffect(len(open_structures), corr, {}, {}, len(joined), before.error or after.error)
    return PortfolioEffect(len(open_structures), corr, dict(before.var), dict(after.var), len(joined), "")
