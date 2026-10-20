"""The minimum-variance hedge ratio and the correlation of daily changes: the one definition the whole app uses.

For the daily changes y of a position and x of the instrument you would hedge it with:

    beta = cov(y, x) / var(x)        (the OLS slope; sample statistics, ddof = 1)

beta units of x move for every unit of y, so to HEDGE one unit of y you trade beta units of x the OPPOSITE way, and to
match it the same way. The correlation of the changes is Pearson; its square is the share of y's variance the hedge removes.

Used by the ATR page's pair hedge and hedge table, the correlation page's rolling hedge ratio and Curve Kinks' hedge
(which negates it: its ratio is the signed quantity to trade). core.regression.beta_from_series is the same slope with
OLS statistics (standard error, p-value) for the panels that show them.
"""

import numpy as np
import pandas as pd

MIN_OBS = 3  # a slope needs at least this many pairs of changes


def slope_and_corr(y, x, min_obs: int = MIN_OBS) -> tuple[float, float]:
    """(beta of y on x, correlation of y and x) for two equal-length arrays of changes; NaN when undefined."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if len(x) < max(min_obs, 3) or np.var(x, ddof=1) <= 0:
        return np.nan, np.nan
    beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    corr = float(np.corrcoef(y, x)[0, 1]) if np.var(y, ddof=1) > 0 else np.nan
    return beta, corr


def rolling_slope(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """beta of y on x over the last `window` pairs, for every day with a full window (NaN where x is flat)."""
    variance = x.rolling(window).var()
    return (y.rolling(window).cov(x) / variance.where(variance > 0)).replace([np.inf, -np.inf], np.nan)
