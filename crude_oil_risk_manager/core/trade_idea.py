"""Trade Idea Analyzer: risk/reward, historical context and portfolio correlation of a hypothetical trade.

An idea is 1-3 legs (exchange-quoted symbol, buy/sell, lots) with an entry, stop and
target price for the structure. Everything is built from local daily Parquet price
DIFFERENCES; nothing is fetched from the API.

The idea's daily PnL series is sum(leg_diff * side * lots) with side +1 for a buy and
-1 for a sell (the same convention as the Correlation tab's structure series). Risk and
reward are quoted in structure points, so the historical context and the chart use the
series per 1 structure unit (the series divided by the gcd of the leg lots); the
correlation against open structures is scale-free and uses the full series.

This is separate from core.regression.run_trade_idea_analysis (mean reversion / sigma
moves), which answers a different question and is not used here.
"""

import math
from dataclasses import dataclass, field

import pandas as pd

from core.correlation import (
    MIN_OBSERVATIONS,
    _load_local_differences,
    build_correlation_matrix_from_series,
    normalize_instrument_symbol,
    structure_difference_series,
)
from core.data_loader import DataLoader
from core.exceptions import CrudeOilRiskError, InsufficientDataError
from core.models import Structure

# Dollar value of a 1.0 point move per lot, used for the estimated $ risk/reward. One
# constant so it is easy to change (CL is really $1,000/pt; $10 is a 0.01 tick).
DOLLARS_PER_POINT_PER_LOT = 10.0

STRUCTURE_LEG_COUNTS = {"outright": 1, "spread": 2, "fly": 3}
# Legs of a new idea per structure type: (direction, lots), the usual ratio pattern.
DEFAULT_LEGS = {
    "outright": [("buy", 1)],
    "spread": [("buy", 1), ("sell", 1)],
    "fly": [("buy", 1), ("sell", 2), ("buy", 1)],
}

HIGH_CORRELATION = 0.7
MODERATE_CORRELATION = 0.4


@dataclass
class TradeIdea:
    legs: list[dict]  # [{"symbol", "direction", "lots"}]
    entry: float
    stop: float
    target: float

    @property
    def total_lots(self) -> int:
        return sum(leg["lots"] for leg in self.legs)

    @property
    def unit_lots(self) -> int:
        """Greatest common divisor of the leg lots: how many structure units the idea holds."""
        return math.gcd(*[leg["lots"] for leg in self.legs])

    @property
    def side(self) -> str:
        """The idea's side, taken from the first leg (the structure price is quoted from it)."""
        return self.legs[0]["direction"]


@dataclass
class IdeaAnalysis:
    idea: TradeIdea | None = None
    risk: float = 0.0
    reward: float = 0.0
    rr_ratio: float = 0.0
    dollar_risk: float = 0.0
    dollar_reward: float = 0.0
    stats: dict = field(default_factory=dict)
    unit_series: pd.Series | None = None
    observations: int = 0
    requested: int = 0
    correlations: list[dict] = field(default_factory=list)  # [{"name", "correlation"}]
    skipped_structures: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Input parsing
# ----------------------------------------------------------------------


def _number(value) -> float | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_idea_inputs(
    structure_type: str, symbols: list, directions: list, lots: list, entry, stop, target
) -> tuple[TradeIdea | None, list[str], list[str]]:
    """Validate the form values. Returns (idea or None, blocking errors, non-blocking warnings).

    Errors: a missing symbol or non-whole lots, a missing entry/stop/target, or a stop/target
    equal to the entry or to each other. A stop or target on the wrong side of the entry for
    the first leg's direction is only a warning.
    """
    errors: list[str] = []
    warnings: list[str] = []
    count = STRUCTURE_LEG_COUNTS.get(structure_type)
    if count is None:
        return None, ["Choose a structure type."], []

    legs = []
    for number in range(1, count + 1):
        symbol = normalize_instrument_symbol(symbols[number - 1] if number <= len(symbols) else None)
        direction = directions[number - 1] if number <= len(directions) else None
        leg_lots = _number(lots[number - 1] if number <= len(lots) else None)
        if not symbol:
            errors.append(f"Enter a symbol for leg {number}.")
        if leg_lots is None or leg_lots <= 0 or leg_lots != int(leg_lots):
            errors.append(f"Leg {number} lots must be a positive whole number.")
        legs.append({"symbol": symbol, "direction": direction if direction in ("buy", "sell") else "buy",
                     "lots": int(leg_lots) if leg_lots and leg_lots > 0 and leg_lots == int(leg_lots) else 0})

    levels = {"entry": _number(entry), "stop": _number(stop), "target": _number(target)}
    for label, value in levels.items():
        if value is None:
            errors.append(f"{label.capitalize()} price is required.")
    if not any(v is None for v in levels.values()):
        if levels["stop"] == levels["entry"]:
            errors.append("Stop cannot equal entry price")
        if levels["target"] == levels["entry"]:
            errors.append("Target cannot equal entry price")
        if levels["stop"] == levels["target"] and levels["stop"] != levels["entry"]:
            errors.append("Stop and target cannot be the same price")
    if errors:
        return None, errors, warnings

    idea = TradeIdea(legs=legs, entry=levels["entry"], stop=levels["stop"], target=levels["target"])
    long_side = idea.side == "buy"
    if (idea.stop > idea.entry) == long_side:
        warnings.append(
            f"Illogical stop for a {idea.side} idea: it should be {'below' if long_side else 'above'} the entry price."
        )
    if (idea.target < idea.entry) == long_side:
        warnings.append(
            f"Illogical target for a {idea.side} idea: it should be {'above' if long_side else 'below'} the entry price."
        )
    return idea, errors, warnings


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------


def idea_daily_series(idea: TradeIdea, data_loader: DataLoader) -> pd.Series:
    """Daily PnL of the idea in points x lots: sum(leg_diff * side * lots), legs inner-joined on date."""
    terms = []
    for leg in idea.legs:
        side = 1 if leg["direction"] == "buy" else -1
        terms.append(_load_local_differences(leg["symbol"], data_loader) * (side * leg["lots"]))
    return pd.concat(terms, axis=1, join="inner").sum(axis=1)


def historical_context(unit_series: pd.Series, risk: float, reward: float) -> dict:
    """Daily-move statistics per structure unit, and how often one day beat the stop or the target."""
    values = unit_series.to_numpy(dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "max_gain": float(values.max()),
        "max_loss": float(values.min()),
        "pct_beyond_risk": float((values < -risk).mean() * 100.0),
        "pct_beyond_reward": float((values > reward).mean() * 100.0),
    }


def correlation_label(correlation: float | None) -> str:
    """'high' (|c| > 0.7), 'moderate' (0.4 <= |c| <= 0.7), 'low' (< 0.4) or 'n/a'."""
    if correlation is None or correlation != correlation:
        return "n/a"
    size = abs(correlation)
    if size > HIGH_CORRELATION:
        return "high"
    return "moderate" if size >= MODERATE_CORRELATION else "low"


def analyze_trade_idea(
    idea: TradeIdea, lookback_days: int, open_structures: list[Structure], data_loader: DataLoader
) -> IdeaAnalysis:
    """Risk/reward, historical context, portfolio correlation and the hypothetical PnL series.

    Symbols with no local history stop the analysis ("No historical data for X"). Open
    structures without data are left out of the correlation table and listed in
    `skipped_structures`.
    """
    result = IdeaAnalysis(idea=idea, requested=lookback_days)
    for leg in idea.legs:
        try:
            missing = data_loader.get_available_date_range(leg["symbol"]) is None
        except CrudeOilRiskError:
            missing = True
        if missing:
            result.errors.append(f"No historical data for {leg['symbol']}")
    if result.errors:
        return result

    result.risk = abs(idea.entry - idea.stop)
    result.reward = abs(idea.target - idea.entry)
    result.rr_ratio = result.reward / result.risk
    result.dollar_risk = result.risk * idea.total_lots * DOLLARS_PER_POINT_PER_LOT
    result.dollar_reward = result.reward * idea.total_lots * DOLLARS_PER_POINT_PER_LOT

    try:
        full = idea_daily_series(idea, data_loader).dropna().sort_index()
    except CrudeOilRiskError as exc:
        result.errors.append(str(exc))
        return result
    window = full.iloc[-lookback_days:]
    if len(window) < MIN_OBSERVATIONS:
        result.errors.append(
            f"Only {len(window)} common trading days across the legs; at least {MIN_OBSERVATIONS} are needed."
        )
        return result
    result.observations = len(window)
    result.unit_series = window / idea.unit_lots
    result.stats = historical_context(result.unit_series, result.risk, result.reward)

    for structure in open_structures:
        try:
            other = structure_difference_series(structure, data_loader)
            matrix, _ = build_correlation_matrix_from_series({"idea": window, "other": other}, lookback_days)
            value = matrix.loc["idea", "other"]
            result.correlations.append(
                {"name": structure.name, "correlation": None if value != value else float(value)}
            )
        except (CrudeOilRiskError, InsufficientDataError, ValueError) as exc:
            result.skipped_structures[structure.name] = str(exc)
    return result
