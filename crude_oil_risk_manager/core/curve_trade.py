"""Trade plan for a kink: entry, lots, stop, target and ranked hedges.

Everything here is a rule applied to numbers the engine already has (live price, the methods'
fair values, daily history); nothing is predicted beyond the kink closing. The structure is
traded at its live price in the direction that fades the kink (core.curve_kinks.trade_side).

ENTRY   live price of the structure when the alert fires.
STOP    distance = min(vol_stop_mult x the structure's daily move, risk_per_lot / point_value):
        the volatility stop, but never wider than your per-lot risk allows. When the cap wins
        the plan says so (the stop is then tighter than normal noise).
LOTS    floor(risk_per_trade / (stop distance x point_value)), capped at max_lots. Because the
        stop distance is capped by risk_per_lot, no lot ever risks more than that.
TARGET  fair value = median of the fair values of the methods that flagged the kink, but only the
        share of the gap that kinks like it have historically closed within `reversion_horizon`
        days (from the backtest, floored at min_reversion_fraction):
        target = entry + (fair - entry) x share.
TIME    half-life of the residual (AR(1) on the neighbour-residual history); time stop ~ 2 half-lives.
HEDGE   candidates are the curve types allowed for this kink type (Settings), same product, not
        sharing a contract with the kinked structure (optional). Each is scored on the daily
        changes of the last `hedge_lookback` days (roll days excluded). The ratio h (hedge units
        per trade unit, + buy / - sell) is the one minimising the 1-day historical 95% VaR of
        trade + h x hedge over a grid; the minimum-variance ratio -cov/var is shown beside it.
        The best `hedge_count` with |correlation| >= hedge_min_corr are listed, ranked by VaR
        reduction; if none qualifies the single best is listed anyway, with a warning.
"""

import math
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import pandas as pd

from core.curve_calendar import FAMILIES, FLY, CurveStructure, parse_contract
from core.curve_history import CurveHistoryStore, generic_code
from core.curve_settings import CurveParams

MIN_TICK = 0.01
MIN_HEDGE_ROWS = 30
MIN_VOL_ROWS = 10
MIN_REVERSION_EVENTS = 5
VAR_PERCENTILE = 5.0  # 95% one-day VaR
_RATIO_GRID = np.linspace(-3.0, 3.0, 241)
WEAK_CORRELATION_WARNING = 0.3


def trade_legs(structure: CurveStructure, side: str) -> str:
    """The legs that make up the trade, e.g. SELL a fly -> "sell Feb27, buy 2 Mar27, sell Apr27".

    The structure's leg weights describe BUYING it (a fly is +1/-2/+1), so SELL flips every sign."""
    parts = []
    for symbol, weight in zip(structure.legs, structure.leg_weights):
        _, month, year = parse_contract(symbol)
        sign = weight if side == "BUY" else -weight
        quantity = "" if abs(weight) == 1 else f"{abs(weight)} "
        parts.append(f"{'buy' if sign > 0 else 'sell'} {quantity}{date(year, month, 1).strftime('%b%y')}")
    return ", ".join(parts)


def structure_generic(product: str, structure: CurveStructure) -> str:
    """History code of a structure ("CO1-2"); a dfly is shown as its two flies."""
    if structure.generic_code:
        return structure.generic_code
    return f"{generic_code(product, FLY, structure.position)} - {generic_code(product, FLY, structure.position + 1)}"


@dataclass
class HedgeOption:
    family: str
    label: str
    generic: str
    side: str  # BUY / SELL of the hedge structure
    ratio: float  # hedge units per 1 unit of the trade (VaR-minimising), always positive; side gives the sign
    ratio_minvar: float  # minimum-variance ratio, signed like the hedge side (+ buy / - sell)
    lots: int
    entry: float | None  # hedge structure's live price
    corr: float
    var_reduction: float  # 0..1 fall in 1-day 95% VaR of the combined position
    legs: str
    warning: str = ""


@dataclass
class TradePlan:
    side: str
    entry: float
    stop: float
    stop_distance: float
    stop_basis: str  # "volatility" | "per-lot risk cap" | "per-lot risk cap (no volatility history)"
    sigma_day: float | None
    lots: int
    dollar_risk: float
    target: float | None = None
    target_distance: float | None = None
    rr: float | None = None
    dollar_reward: float | None = None
    fair_value: float | None = None
    reversion_share: float | None = None
    reversion_events: int = 0
    half_life_days: float | None = None
    time_stop_days: int | None = None
    hedges: list[HedgeOption] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class HedgeCandidate:
    family: str
    structure: CurveStructure
    live: float | None


# ----------------------------------------------------------------------
# History helpers
# ----------------------------------------------------------------------


def change_frame(store: CurveHistoryStore, product: str, n: int) -> pd.DataFrame:
    """Daily changes of every family/position (columns (family, position - 1)), NaN on roll days.

    A generic position holds a different contract after a roll, so the jump on that day is not
    a P&L of anything tradable and is blanked out."""
    parts = []
    for family in FAMILIES:
        matrix = store.matrix(product, family, n)
        if matrix is None:
            continue
        index = pd.DatetimeIndex(matrix.dates)
        values = pd.DataFrame(matrix.values, index=index)
        months = pd.DataFrame(matrix.months, index=index)
        diff = values.diff().mask(months.ne(months.shift()))
        diff.columns = pd.MultiIndex.from_product([[family], range(matrix.values.shape[1])])
        parts.append(diff)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


def daily_move(frame: pd.DataFrame, family: str, k0: int, window: int) -> float | None:
    """Standard deviation of the structure's daily changes over the last `window` days."""
    if (family, k0) not in frame.columns:
        return None
    series = frame[(family, k0)].tail(window).dropna()
    return float(series.std(ddof=1)) if len(series) >= MIN_VOL_ROWS else None


def half_life(resid_hist: np.ndarray, lookback: int) -> float | None:
    """Half-life in days of the neighbour residual: AR(1) fitted on all positions pooled."""
    window = np.asarray(resid_hist, dtype=float)[-lookback:]
    if window.shape[0] < 20:
        return None
    previous, current = window[:-1].ravel(), window[1:].ravel()
    ok = np.isfinite(previous) & np.isfinite(current)
    denominator = float(np.sum(previous[ok] ** 2))
    if ok.sum() < 50 or denominator <= 0:
        return None
    phi = float(np.sum(previous[ok] * current[ok]) / denominator)
    return math.log(2) / -math.log(phi) if 0 < phi < 1 else None


# ----------------------------------------------------------------------
# Hedges
# ----------------------------------------------------------------------


def _var(pnl: np.ndarray) -> np.ndarray:
    """1-day historical VaR (a positive loss) along the last axis."""
    return -np.percentile(pnl, VAR_PERCENTILE, axis=-1)


def rank_hedges(
    frame: pd.DataFrame, family: str, k0: int, side: str, product: str, candidates: list[HedgeCandidate],
    params: CurveParams,
) -> tuple[list[HedgeOption], list[str]]:
    """Hedge alternatives ranked by VaR reduction, with ratios from the VaR search. See module docstring."""
    warnings: list[str] = []
    if (family, k0) not in frame.columns:
        return [], ["No history for this curve, so no hedge could be computed"]
    window = frame.tail(params.hedge_lookback)
    sign = 1.0 if side == "BUY" else -1.0
    scored: list[tuple[float, HedgeOption]] = []
    for candidate in candidates:
        key = (candidate.family, candidate.structure.position - 1)
        if key not in window.columns:
            continue
        pair = pd.concat([window[(family, k0)], window[key]], axis=1).dropna()
        if len(pair) < MIN_HEDGE_ROWS:
            continue
        x = sign * pair.iloc[:, 0].to_numpy()  # P&L per unit of the trade
        b = pair.iloc[:, 1].to_numpy()  # P&L per unit BOUGHT of the candidate
        if np.std(b) < 1e-9 or np.std(x) < 1e-9:
            continue
        correlation = float(np.corrcoef(x, b)[0, 1])
        base_var = float(_var(x))
        portfolio = x[None, :] + _RATIO_GRID[:, None] * b[None, :]
        var_curve = _var(portfolio) + 1e-9 * np.abs(_RATIO_GRID)  # ties go to the smaller hedge
        best = int(np.argmin(var_curve))
        ratio = float(_RATIO_GRID[best])
        reduction = 1.0 - float(var_curve[best]) / base_var if base_var > 0 else 0.0
        h_side = "BUY" if ratio > 0 else "SELL"
        hedge = HedgeOption(
            family=candidate.family, label=candidate.structure.label,
            generic=structure_generic(product, candidate.structure), side=h_side, ratio=abs(ratio),
            ratio_minvar=float(-np.cov(x, b)[0, 1] / np.var(b, ddof=1)), lots=0, entry=candidate.live,
            corr=correlation, var_reduction=max(0.0, reduction), legs=trade_legs(candidate.structure, h_side),
        )
        if abs(ratio) < 0.05:
            hedge.warning = "no meaningful hedge ratio found"
        scored.append((hedge.var_reduction, hedge))
    if not scored:
        return [], ["No hedge candidate had enough overlapping history"]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    qualified = [h for _, h in scored if abs(h.corr) >= params.hedge_min_corr]
    chosen = qualified[: params.hedge_count]
    if not chosen:
        chosen = [scored[0][1]]
        chosen[0].warning = chosen[0].warning or f"weak hedge (|correlation| {abs(chosen[0].corr):.2f} below {params.hedge_min_corr:.2f})"
        warnings.append("No hedge reached the minimum correlation; the best available is shown")
    for hedge in chosen:
        if not hedge.warning and abs(hedge.corr) < WEAK_CORRELATION_WARNING:
            hedge.warning = f"weak correlation {hedge.corr:+.2f}"
    return chosen, warnings


# ----------------------------------------------------------------------
# The plan
# ----------------------------------------------------------------------


def fair_value(fair: dict[str, float | None], flags: dict[str, bool]) -> float | None:
    """Median fair value of the methods that flagged the kink (any method's if none has one)."""
    values = [v for m, v in fair.items() if flags.get(m) and v is not None]
    if not values:
        values = [v for v in (fair.get("neighbour"), fair.get("fit")) if v is not None]
    return float(np.median(values)) if values else None


def build_plan(
    *, product: str, structure: CurveStructure, side: str, live: float, fair: dict, flags: dict,
    frame: pd.DataFrame, half_life_days: float | None, reversion_share: float | None, reversion_events: int,
    candidates: list[HedgeCandidate], params: CurveParams, hedge_cache: dict | None = None,
) -> TradePlan:
    """Entry, lots, stop, target, time stop and hedges for fading one kink. See module docstring.

    `hedge_cache` (optional) keeps the ranking between polls: it depends only on history, so only
    the hedges' live entry prices are refreshed each time."""
    family, k0 = structure.family, structure.position - 1
    warnings: list[str] = []
    sigma = daily_move(frame, family, k0, params.vol_window) if len(frame) else None
    cap = params.risk_per_lot / params.point_value
    if sigma is None:
        distance, basis = cap, "per-lot risk cap (no volatility history)"
    elif params.vol_stop_mult * sigma <= cap:
        distance, basis = params.vol_stop_mult * sigma, "volatility"
    else:
        distance, basis = cap, "per-lot risk cap"
        warnings.append(
            f"The per-lot risk cap puts the stop at {cap:.2f}, tighter than the {params.vol_stop_mult:g}-sigma "
            f"volatility stop ({params.vol_stop_mult * sigma:.2f}): normal noise may stop this trade out"
        )
    distance = max(distance, MIN_TICK)
    stop = live - distance if side == "BUY" else live + distance

    loss_per_lot = distance * params.point_value
    lots = int(params.risk_per_trade // loss_per_lot)
    if lots < 1:
        warnings.append(
            f"Risk per trade (${params.risk_per_trade:,.0f}) is below the ${loss_per_lot:,.0f} that 1 lot risks at this stop: no lots"
        )
        lots = 0
    elif lots > params.max_lots:
        warnings.append(f"Lots capped at {params.max_lots} (risk allows {lots})")
        lots = params.max_lots
    plan = TradePlan(
        side=side, entry=live, stop=stop, stop_distance=distance, stop_basis=basis, sigma_day=sigma, lots=lots,
        dollar_risk=lots * loss_per_lot, half_life_days=half_life_days,
        time_stop_days=None if half_life_days is None else max(1, math.ceil(2 * half_life_days)),
    )

    fair_price = fair_value(fair, flags)
    share = params.min_reversion_fraction if reversion_share is None else max(params.min_reversion_fraction, min(1.0, reversion_share))
    plan.reversion_events = reversion_events
    if reversion_events < MIN_REVERSION_EVENTS:
        warnings.append(f"Only {reversion_events} similar past kinks to estimate how much closes; using {share:.0%}")
    plan.reversion_share, plan.fair_value = share, fair_price
    if fair_price is None:
        warnings.append("No fair value available, so no target")
    else:
        target = live + (fair_price - live) * share
        moving_right = (target < live) if side == "SELL" else (target > live)
        if not moving_right:
            warnings.append("The fair value is on the wrong side of the entry for this direction: no target")
        else:
            plan.target, plan.target_distance = target, abs(target - live)
            plan.rr = plan.target_distance / distance
            plan.dollar_reward = lots * plan.target_distance * params.point_value
            if plan.rr < 1.0:
                warnings.append(f"Reward:risk is {plan.rr:.2f}, below 1")

    cache_key = (
        product, family, k0, side, tuple((c.family, c.structure.position) for c in candidates),
        params.hedge_lookback, params.hedge_min_corr, params.hedge_count,
        len(frame), None if frame.empty else frame.index[-1],
    )
    if hedge_cache is not None and cache_key in hedge_cache:
        ranked, hedge_warnings = hedge_cache[cache_key]
    else:
        ranked, hedge_warnings = rank_hedges(frame, family, k0, side, product, candidates, params)
        if hedge_cache is not None:
            hedge_cache.clear() if len(hedge_cache) > 500 else None
            hedge_cache[cache_key] = (ranked, hedge_warnings)
    live_by_candidate = {(c.family, c.structure.label): c.live for c in candidates}
    plan.hedges = [replace(h, entry=live_by_candidate.get((h.family, h.label), h.entry)) for h in ranked]
    warnings.extend(hedge_warnings)
    for hedge in plan.hedges:
        hedge.lots = max(1, round(hedge.ratio * lots)) if lots else 0
    plan.warnings = warnings
    return plan


# ----------------------------------------------------------------------
# Text
# ----------------------------------------------------------------------


def _price(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def plan_summary(label: str, family_name: str, legs: str, plan: TradePlan) -> str:
    """One-paragraph plan for an alert."""
    parts = [f"{plan.side} {family_name} {label} ({legs}) @ {_price(plan.entry)}, {plan.lots} lots"]
    parts.append(
        f"Stop {_price(plan.stop)} ({plan.stop_distance:.2f} pts, risk ${plan.dollar_risk:,.0f}, {plan.stop_basis})"
    )
    if plan.target is not None:
        parts.append(
            f"Target {_price(plan.target)} ({plan.target_distance:.2f} pts, {plan.reversion_share:.0%} of the gap to "
            f"fair {_price(plan.fair_value)}), R:R {plan.rr:.1f}"
        )
    else:
        parts.append("No target")
    if plan.time_stop_days:
        parts.append(f"time stop ~{plan.time_stop_days}d (half-life {plan.half_life_days:.1f}d)")
    text = " | ".join(parts) + "."
    if plan.hedges:
        options = [
            f"{h.side} {h.lots} x {h.family} {h.label} @ {_price(h.entry)} [{h.legs}] "
            f"(ratio {h.ratio:.2f}, corr {h.corr:+.2f}, VaR -{h.var_reduction:.0%}"
            + (f", {h.warning}" if h.warning else "") + ")"
            for h in plan.hedges
        ]
        text += " HEDGE: " + "; or ".join(options) + "."
    return text
