"""Historical-simulation Value-at-Risk (VaR).

Scenario P&L for each historical day is price_difference * net_lots *
multiplier; VaR is the loss at the (1 - confidence) percentile of that
distribution, always expressed as a POSITIVE number.

Known weaknesses of this approach (read before relying on the numbers):

1. Historical simulation assumes the past return distribution repeats. It may
   understate tail risk when the market moves into a new regime.
2. Thin history on newly listed contract months reduces reliability. In a
   portfolio VaR the inner join on dates means the shortest-history symbol
   limits the scenario count for the whole portfolio.
3. It uses price differences, not percentage returns. That suits spread and
   fly structures, but absolute dollar moves — not percentage moves — drive
   VaR.
4. Price differences are assumed stationary. Verify this periodically.
5. VaR is not a complete risk measure: it does not capture liquidity risk or
   gap risk at the open.

TODO: Marginal VaR (accounting for cross-structure correlation) is not
implemented; per-structure VaR here is standalone (the structure in
isolation).
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.data_loader import DataLoader
from core.exceptions import CrudeOilRiskError, DataNotAvailableError, InsufficientDataError
from core.exposure import filter_open_structures, get_structure_net_lots
from core.models import Contract, Structure

_TERM_STRUCTURE_HORIZONS = [1, 2, 3, 5, 10]
_RELIABLE_SCENARIOS = 252
_MIN_SCENARIOS = 20

_MARGINAL_VAR_NOTE = (
    "TODO: Marginal VaR (accounting for cross-structure correlation) not yet "
    "implemented. Confirm aggregation method with user before building."
)


def _validate_inputs(confidence: float, horizon_days: int) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1 (exclusive), got {confidence}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")


def _scale_var(var_1day: float, horizon_days: int) -> float:
    """Square-root-of-time scaling: VaR_N = VaR_1 * sqrt(N)."""
    return float(var_1day * np.sqrt(horizon_days))


def _var_from_scenarios(scenario_pnl: np.ndarray, confidence: float) -> tuple[float, float]:
    """Return (var, percentile_cutoff); var = max(0, -percentile), so never negative."""
    cutoff = float(np.percentile(scenario_pnl, (1.0 - confidence) * 100.0))
    return max(0.0, -cutoff), cutoff


def _data_warning(n_scenarios: int) -> str | None:
    if n_scenarios < _RELIABLE_SCENARIOS:
        return (
            f"Only {n_scenarios} scenarios available (recommend minimum "
            f"{_RELIABLE_SCENARIOS}). VaR estimate may be unreliable."
        )
    return None


def _combine_scenarios(pnl_by_symbol: dict[str, pd.Series]) -> pd.Series:
    """Sum per-symbol scenario P&L on dates where ALL symbols have data (inner join)."""
    frame = pd.concat(list(pnl_by_symbol.values()), axis=1, join="inner", keys=list(pnl_by_symbol))
    return frame.sum(axis=1)


def calculate_historical_var(
    symbol: str,
    net_lots: float,
    multiplier: float,
    confidence: float,
    horizon_days: int,
    data_loader: DataLoader,
) -> dict:
    """Historical-simulation VaR for a single position (one symbol).

    scenario_pnl = price_diff * net_lots * multiplier for each historical day.
    var_1day = -np.percentile(scenario_pnl, (1 - confidence) * 100), floored
    at 0.0 so VaR is always a non-negative loss amount.

    For horizon_days > 1, VaR is scaled by square-root-of-time
    (var_1day * sqrt(horizon_days)). WARNING: square-root scaling assumes
    i.i.d. returns. For spread structures this may understate risk if
    autocorrelation exists.

    Returns a dict with symbol, var_1day, var_Nday (N = horizon_days),
    confidence, scenarios_used, percentile_cutoff (the raw scenario P&L at
    the percentile, normally negative), min/max/mean scenario P&L, and
    data_warning (set if fewer than 252 scenarios are available).
    A zero net_lots position has a VaR of 0.0.

    Raises DataNotAvailableError / InsufficientDataError from the loader if
    the symbol's price history cannot be provided.
    """
    _validate_inputs(confidence, horizon_days)
    if multiplier <= 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")

    diffs = data_loader.load_price_differences(symbol)
    scenario_pnl = diffs.to_numpy(dtype=float) * net_lots * multiplier
    var_1day, cutoff = _var_from_scenarios(scenario_pnl, confidence)
    n = len(scenario_pnl)

    return {
        "symbol": symbol,
        "var_1day": var_1day,
        "var_Nday": _scale_var(var_1day, horizon_days),
        "confidence": confidence,
        "scenarios_used": n,
        "percentile_cutoff": cutoff,
        "min_scenario_pnl": float(np.min(scenario_pnl)),
        "max_scenario_pnl": float(np.max(scenario_pnl)),
        "mean_scenario_pnl": float(np.mean(scenario_pnl)),
        "data_warning": _data_warning(n),
    }


def calculate_portfolio_var(
    structures: list[Structure],
    exposure_summary: dict,
    confidence: float,
    horizon_days: int,
    data_loader: DataLoader,
    contracts: dict[str, Contract],
) -> dict:
    """Portfolio-level historical-simulation VaR.

    Portfolio VaR: per-symbol scenario P&L (price_diff * net_lots *
    multiplier) for every symbol with non-zero net_lots in
    exposure_summary["by_product"], summed on dates where ALL symbols have
    data (inner join), then the confidence percentile is applied to the
    combined series. Horizons > 1 day use square-root-of-time scaling, which
    assumes i.i.d. returns and may understate risk for autocorrelated spread
    structures.

    per_structure_var is STANDALONE VaR (each open structure in isolation,
    over its own symbols' common dates), not marginal contribution.
    standalone values are None (with a data_warning) if none of a structure's
    positions have price data or too few common dates exist; a structure
    with no net position has 0.0.

    A symbol with no price data (or no contract multiplier) is excluded from
    portfolio VaR and reported in data_warnings — missing data is never
    replaced by zeros. This understates VaR, so consumers must surface the
    warnings. Raises DataNotAvailableError if positions exist but none has
    data, and InsufficientDataError if fewer than 20 common dates remain.
    """
    _validate_inputs(confidence, horizon_days)

    open_structures = filter_open_structures(structures)
    data_warnings: list[str] = []

    multipliers = {symbol: contract.multiplier for symbol, contract in contracts.items()}
    for structure in open_structures:
        for leg in structure.legs:
            multipliers.setdefault(leg.contract.symbol, leg.contract.multiplier)

    portfolio_lots: dict[str, float] = {}
    for symbols in exposure_summary.get("by_product", {}).values():
        for symbol, info in symbols.items():
            if info["net_lots"] != 0:
                portfolio_lots[symbol] = info["net_lots"]

    structure_lots: dict[str, dict[str, float]] = {
        s.structure_id: {sym: lots for sym, lots in get_structure_net_lots(s).items() if lots != 0}
        for s in open_structures
    }

    needed = set(portfolio_lots)
    for lots in structure_lots.values():
        needed.update(lots)

    diffs: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}
    for symbol in sorted(needed):
        if symbol not in multipliers:
            failures[symbol] = "no contract specification (multiplier) available"
            continue
        try:
            diffs[symbol] = data_loader.load_price_differences(symbol)
        except CrudeOilRiskError as exc:
            failures[symbol] = str(exc)

    def scenario_pnl(symbol: str, lots: float) -> pd.Series:
        return diffs[symbol] * lots * multipliers[symbol]

    # ---- portfolio-level VaR ----
    included = [s for s in portfolio_lots if s in diffs]
    for symbol in portfolio_lots:
        if symbol in failures:
            data_warnings.append(
                f"{symbol} excluded from portfolio VaR (VaR is understated): {failures[symbol]}"
            )

    if not portfolio_lots:
        var_1day, n_scenarios = 0.0, 0
    else:
        if not included:
            raise DataNotAvailableError(
                "Portfolio VaR cannot be calculated: no price data for any position. "
                + "; ".join(f"{s}: {m}" for s, m in failures.items())
            )
        combined = _combine_scenarios({s: scenario_pnl(s, portfolio_lots[s]) for s in included})
        n_scenarios = len(combined)
        if n_scenarios < _MIN_SCENARIOS:
            raise InsufficientDataError(
                f"Portfolio VaR: only {n_scenarios} common dates across {included}, "
                f"minimum required is {_MIN_SCENARIOS}."
            )
        var_1day, _ = _var_from_scenarios(combined.to_numpy(dtype=float), confidence)
        warning = _data_warning(n_scenarios)
        if warning:
            data_warnings.append(f"Portfolio: {warning}")

    # ---- per-structure standalone VaR ----
    per_structure_var: dict[str, dict] = {}
    for structure in open_structures:
        lots_by_symbol = structure_lots[structure.structure_id]
        have = [s for s in lots_by_symbol if s in diffs]
        missing = [s for s in lots_by_symbol if s in failures]
        notes: list[str] = []
        if missing:
            notes.append(f"Excluded symbols with no usable data: {', '.join(missing)}.")

        standalone_1day: float | None
        if not lots_by_symbol:
            standalone_1day = 0.0
        elif not have:
            standalone_1day = None
            notes.append("No price data for any of this structure's positions.")
        else:
            combined = _combine_scenarios({s: scenario_pnl(s, lots_by_symbol[s]) for s in have})
            if len(combined) < _MIN_SCENARIOS:
                standalone_1day = None
                notes.append(
                    f"Only {len(combined)} common dates (minimum {_MIN_SCENARIOS}); VaR not calculated."
                )
            else:
                standalone_1day, _ = _var_from_scenarios(combined.to_numpy(dtype=float), confidence)
                warning = _data_warning(len(combined))
                if warning:
                    notes.append(warning)

        per_structure_var[structure.structure_id] = {
            "standalone_var_1day": standalone_1day,
            "standalone_var_Nday": None if standalone_1day is None else _scale_var(standalone_1day, horizon_days),
            "symbols_included": have,
            "data_warning": " ".join(notes) if notes else None,
        }

    return {
        "portfolio_var_1day": var_1day,
        "portfolio_var_Nday": _scale_var(var_1day, horizon_days),
        "confidence": confidence,
        "horizon_days": horizon_days,
        "scenarios_used": n_scenarios,
        "common_dates_used": n_scenarios,
        "per_structure_var": per_structure_var,
        "marginal_var": None,
        "marginal_var_note": _MARGINAL_VAR_NOTE,
        "data_warnings": data_warnings,
        "calculated_at": datetime.now(timezone.utc),
    }


def get_var_term_structure(
    symbol: str,
    net_lots: float,
    multiplier: float,
    confidence: float,
    data_loader: DataLoader,
) -> dict[int, float]:
    """VaR for horizons [1, 2, 3, 5, 10] days, as {horizon_days: var_amount}.

    The 1-day VaR is computed once from history and scaled by
    square-root-of-time for longer horizons (assumes i.i.d. returns; may
    understate risk for autocorrelated spread structures).
    """
    result = calculate_historical_var(symbol, net_lots, multiplier, confidence, 1, data_loader)
    return {h: _scale_var(result["var_1day"], h) for h in _TERM_STRUCTURE_HORIZONS}
