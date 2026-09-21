"""Correlation analysis between price series (contract-vs-contract).

Price DIFFERENCES (close_t - close_{t-1}) are used throughout, never log or
simple returns. Futures spreads and flies are stationary in price-difference
space and can be zero or negative, so return-based measures (which assume
non-stationary, strictly positive, equity-like prices) are undefined or
misleading for them. Every series here is an exchange-quoted symbol with its
own price history — spreads/flies are never derived from their legs.

All data access goes through DataLoader; this module does no file I/O.
"""

import logging
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from core.data_loader import DataLoader
from core.exceptions import CrudeOilRiskError, InsufficientDataError

logger = logging.getLogger(__name__)

_HIGH_THRESHOLD = 0.7


def _validate_window(window: int) -> None:
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")


def _load_aligned(
    symbol_a: str,
    symbol_b: str,
    window: int,
    data_loader: DataLoader,
    start: date | None,
    end: date | None,
) -> tuple[pd.Series, pd.Series]:
    diffs_a = data_loader.load_price_differences(symbol_a, start, end, min_rows=window)
    diffs_b = data_loader.load_price_differences(symbol_b, start, end, min_rows=window)
    return data_loader.align_series(diffs_a, diffs_b)


def _pearson_last_window(
    aligned_a: pd.Series, aligned_b: pd.Series, window: int, symbol_a: str, symbol_b: str
) -> float:
    """Pearson correlation over the most recent `window` aligned observations."""
    if len(aligned_a) < window:
        raise InsufficientDataError(
            f"Insufficient data for correlation of {symbol_a} and {symbol_b}: "
            f"{len(aligned_a)} aligned observations available, window requires {window}."
        )
    a = aligned_a.iloc[-window:].to_numpy(dtype=float)
    b = aligned_b.iloc[-window:].to_numpy(dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        raise InsufficientDataError(
            f"Correlation of {symbol_a} and {symbol_b} is undefined: "
            f"a series has zero variance over the last {window} observations."
        )
    r, _ = pearsonr(a, b)
    if not np.isfinite(r):
        raise InsufficientDataError(
            f"Correlation of {symbol_a} and {symbol_b} is not finite over the last {window} observations."
        )
    return float(min(1.0, max(-1.0, r)))


def calculate_correlation(
    symbol_a: str,
    symbol_b: str,
    window: int,
    data_loader: DataLoader,
    start: date | None = None,
    end: date | None = None,
) -> float:
    """Pearson correlation of daily price differences over the latest `window` common days.

    Raises InsufficientDataError if fewer than `window` observations are
    available after alignment (or the correlation is undefined). Never
    returns NaN. Result is in [-1.0, 1.0].
    """
    _validate_window(window)
    aligned_a, aligned_b = _load_aligned(symbol_a, symbol_b, window, data_loader, start, end)
    return _pearson_last_window(aligned_a, aligned_b, window, symbol_a, symbol_b)


def calculate_rolling_correlation(
    symbol_a: str,
    symbol_b: str,
    window: int,
    data_loader: DataLoader,
    start: date | None = None,
    end: date | None = None,
) -> pd.Series:
    """Rolling `window`-day Pearson correlation of price differences over aligned dates.

    Leading NaN values (the first window-1 observations) and any undefined
    windows (e.g. zero variance) are dropped. Raises InsufficientDataError if
    no values remain.
    """
    _validate_window(window)
    aligned_a, aligned_b = _load_aligned(symbol_a, symbol_b, window, data_loader, start, end)
    rolling = aligned_a.rolling(window).corr(aligned_b)
    rolling = rolling.replace([np.inf, -np.inf], np.nan).dropna().clip(-1.0, 1.0)
    if rolling.empty:
        raise InsufficientDataError(
            f"Insufficient data for rolling correlation of {symbol_a} and {symbol_b}: "
            f"{len(aligned_a)} aligned observations, window is {window}."
        )
    rolling.name = f"{symbol_a}|{symbol_b}"
    return rolling


def build_correlation_matrix(
    symbols: list[str],
    window: int,
    data_loader: DataLoader,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """N x N correlation matrix (diagonal 1.0, symmetric) for the Correlation Heatmap.

    Best-effort: if a pair fails (missing/insufficient data), that cell and
    its mirror are NaN and a warning is logged; individual pair failures
    never raise.
    """
    n = len(symbols)
    values = np.full((n, n), np.nan)
    for i in range(n):
        values[i, i] = 1.0
        for j in range(i + 1, n):
            try:
                corr = calculate_correlation(symbols[i], symbols[j], window, data_loader, start, end)
            except (CrudeOilRiskError, ValueError) as exc:
                logger.warning("Correlation failed for %s / %s: %s", symbols[i], symbols[j], exc)
                continue
            values[i, j] = corr
            values[j, i] = corr
    return pd.DataFrame(values, index=symbols, columns=symbols)


def classify_correlation(correlation: float | None) -> str:
    """Classify a correlation coefficient.

    >= 0.7 -> "highly_correlated"; <= -0.7 -> "negatively_correlated";
    otherwise "uncorrelated"; NaN/None -> "insufficient_data".
    Used by the structure builder for exposure-duplication warnings.
    """
    if correlation is None or np.isnan(correlation):
        return "insufficient_data"
    if correlation >= _HIGH_THRESHOLD:
        return "highly_correlated"
    if correlation <= -_HIGH_THRESHOLD:
        return "negatively_correlated"
    return "uncorrelated"


def get_correlation_with_portfolio(
    candidate_symbol: str,
    portfolio_symbols: list[str],
    window: int,
    data_loader: DataLoader,
) -> dict[str, dict]:
    """Correlation of a candidate symbol against every portfolio symbol.

    Returns {existing_symbol: {"correlation": float | None, "classification": str,
    "window_used": int, "common_dates": int}}. Every portfolio symbol is
    always present as a key; correlation is None when the calculation failed
    (window_used is then 0). window_used is the number of observations
    actually used in the correlation. Interface for the Structure Builder
    live correlation check.
    """
    _validate_window(window)
    results: dict[str, dict] = {}

    try:
        candidate_diffs = data_loader.load_price_differences(candidate_symbol, min_rows=window)
    except CrudeOilRiskError as exc:
        logger.warning("Cannot load candidate %s: %s", candidate_symbol, exc)
        candidate_diffs = None

    for existing in portfolio_symbols:
        entry = {
            "correlation": None,
            "classification": "insufficient_data",
            "window_used": 0,
            "common_dates": 0,
        }
        results[existing] = entry
        if candidate_diffs is None:
            continue

        try:
            existing_diffs = data_loader.load_price_differences(existing, min_rows=window)
            common_dates = len(candidate_diffs.index.intersection(existing_diffs.index))
            entry["common_dates"] = common_dates
            aligned_a, aligned_b = data_loader.align_series(candidate_diffs, existing_diffs)
            corr = _pearson_last_window(aligned_a, aligned_b, window, candidate_symbol, existing)
        except (CrudeOilRiskError, ValueError) as exc:
            logger.warning("Correlation failed for %s / %s: %s", candidate_symbol, existing, exc)
            continue

        entry["correlation"] = corr
        entry["classification"] = classify_correlation(corr)
        entry["window_used"] = window

    return results


# TODO: Structure-vs-structure correlation.
# NOTE: Structure-vs-structure correlation aggregation method not yet confirmed
# by user. Will be implemented when method is defined. See conversation notes
# from Phase 1 design.
