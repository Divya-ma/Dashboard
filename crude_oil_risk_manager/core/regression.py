"""Beta of one series on another, and the position size it implies.

This is the regression the Correlation tab's Beta column, the VaR tab's beta-vs-portfolio panel and the New Structure
builder's position-sizing panel share. It works on any two already-built price-difference / P&L series (a structure's
derived series, a watchlist item, a portfolio P&L series), so none of them re-reads price files.

SIGN AND UNITS. A = alpha + beta x B  (ordinary least squares on the most recent `window` shared days). beta is units of A
per unit of B, so to HEDGE one unit of A you trade beta units of B the OPPOSITE way (`lots_to_hedge = -beta`), and to
MATCH it you trade beta units the same way (`lots_to_match`). Functions raise InsufficientDataError rather than returning
NaN/inf.
"""

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core.exceptions import InsufficientDataError

_MIN_BETA_OBS = 30


def _ensure_finite(values: dict[str, float], context: str) -> None:
    bad = [name for name, value in values.items() if not math.isfinite(value)]
    if bad:
        raise InsufficientDataError(
            f"{context}: regression produced non-finite {', '.join(bad)} "
            f"(degenerate data, e.g. zero variance or a perfect fit)."
        )


def beta_from_series(series_a: pd.Series, series_b: pd.Series, window: int) -> dict:
    """OLS beta of two pre-built price-difference/P&L series: A = alpha + beta * B + eps.

    Works on any two already-built series — a structure's derived diff series
    (core.correlation.structure_difference_series), a watchlist item's series
    (core.correlation.build_watchlist_series), or a portfolio P&L series
    (core.var.portfolio_pnl_series) — so the same regression machinery serves the
    Correlation tab's Beta column, the VaR tab's beta-vs-portfolio panel, and the
    Structure Builder's position-sizing panel without re-touching Parquet.

    Uses the most recent `window` aligned observations (fewer if fewer are available,
    never fewer than _MIN_BETA_OBS). Raises InsufficientDataError if fewer than
    _MIN_BETA_OBS aligned observations exist, B has zero variance, or the fit is
    degenerate.
    """
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna().sort_index()
    if len(aligned) < _MIN_BETA_OBS:
        raise InsufficientDataError(
            f"Insufficient common observations for beta: {len(aligned)} available, "
            f"minimum required is {_MIN_BETA_OBS}."
        )
    n = min(window, len(aligned))
    y = aligned.iloc[-n:, 0]
    x = aligned.iloc[-n:, 1]
    if np.std(x.to_numpy(dtype=float)) == 0:
        raise InsufficientDataError("Beta is undefined: the reference series has zero variance.")

    result = sm.OLS(y.to_numpy(dtype=float), sm.add_constant(x.to_numpy(dtype=float))).fit()
    alpha, beta = (float(p) for p in result.params)
    stats = {
        "alpha": alpha,
        "beta": beta,
        "r_squared": float(result.rsquared),
        "beta_std_error": float(result.bse[1]),
        "beta_p_value": float(result.pvalues[1]),
    }
    _ensure_finite(stats, "Beta")
    return {
        "beta": beta,
        "alpha": alpha,
        "r_squared": stats["r_squared"],
        "beta_std_error": stats["beta_std_error"],
        "beta_p_value": stats["beta_p_value"],
        "observations": int(result.nobs),
        "window_used": n,
    }


def position_sizing_from_beta(beta_result: dict, current_lots: float = 1.0) -> dict:
    """Lots of the candidate needed to match or hedge the reference series' moves.

    `beta` is dollars-of-reference-move per dollar-of-candidate-move-at-`current_lots`,
    so beta / current_lots is dollars-of-reference-move per dollar-of-candidate-move
    PER LOT — i.e. the lot count at which the candidate's daily P&L would track the
    reference series one-for-one. Trading that many lots in the SAME direction as the
    candidate was sized MATCHES the reference's move; trading it in the OPPOSITE
    direction HEDGES (offsets) it. current_lots defaults to 1.0 for a per-lot candidate
    series (the Structure Builder's outright decomposition, or a single leg).
    """
    if current_lots == 0:
        raise ValueError("current_lots must be non-zero")
    lots_to_match = beta_result["beta"] / current_lots
    return {
        **beta_result,
        "current_lots": current_lots,
        "lots_to_match": lots_to_match,
        "lots_to_hedge": -lots_to_match,
    }
