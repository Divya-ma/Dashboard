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
import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from adapters.base import PRODUCT_TO_API_CODE, SymbolTranslator
from core.data_loader import DataLoader
from core.exceptions import CrudeOilRiskError, InsufficientDataError
from core.models import Structure
from core.structure_utils import normalize_symbol

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


# ----------------------------------------------------------------------
# Watchlist correlation (Correlation Heatmap tab)
# ----------------------------------------------------------------------

# Fewest common daily observations a heatmap will be computed from.
MIN_OBSERVATIONS = 5

_TENOR_RE = re.compile(r"[FGHJKMNQUVXZ]\d{2}")


def normalize_instrument_symbol(symbol: str | None) -> str:
    """Upper-case an instrument symbol and accept the long spread form.

    "clz25-clh26" -> "CLZ25-H26": a later leg may repeat the product code, which is
    dropped so the symbol matches the exchange-quoted format used everywhere else.
    """
    sym = normalize_symbol(symbol)
    try:
        product, rest = SymbolTranslator._match_product_prefix(sym, PRODUCT_TO_API_CODE)
    except ValueError:
        return sym
    parts = re.split(r"([-+])", rest)
    out = [parts[0]]
    for separator, token in zip(parts[1::2], parts[2::2]):
        if token.startswith(product) and _TENOR_RE.fullmatch(token[len(product):]):
            token = token[len(product):]
        out += [separator, token]
    return product + "".join(out)


def build_correlation_matrix_from_series(
    series_by_label: dict[str, pd.Series], window: int
) -> tuple[pd.DataFrame, int]:
    """Pearson correlation matrix of pre-built price-difference series.

    Series are inner-joined on date and the most recent `window` common days are used
    (fewer if that is all there is). Returns (matrix, observations_used). A pair whose
    series has zero variance over the window is NaN, like the symbol-based matrix.
    Raises InsufficientDataError if there are fewer than 2 series or fewer than
    MIN_OBSERVATIONS common days.
    """
    _validate_window(window)
    if len(series_by_label) < 2:
        raise InsufficientDataError("Need at least 2 valid series to compute a correlation matrix.")
    aligned = pd.concat(list(series_by_label.values()), axis=1, join="inner", keys=list(series_by_label))
    aligned = aligned.dropna().sort_index()
    if len(aligned) < MIN_OBSERVATIONS:
        raise InsufficientDataError(
            f"Only {len(aligned)} common days across the selected series; at least {MIN_OBSERVATIONS} are needed."
        )
    used = aligned.iloc[-window:]
    n = len(used)
    labels = list(series_by_label)
    values = np.full((len(labels), len(labels)), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j < i:
                values[i, j] = values[j, i]
            elif i == j:
                values[i, j] = 1.0 if np.std(used[a].to_numpy(dtype=float)) > 0 else np.nan
            else:
                try:
                    values[i, j] = _pearson_last_window(used[a], used[b], n, a, b)
                except InsufficientDataError:
                    pass  # zero variance: leave NaN
    return pd.DataFrame(values, index=labels, columns=labels), n


def structure_difference_series(structure: Structure, data_loader: DataLoader) -> pd.Series:
    """Daily synthetic PnL-difference series of a structure, in point x lots.

    structure_diff[t] = sum(leg_diff[t] * ratio * side * lots) over the legs, where side is
    +1 for a buy and -1 for a sell. `ratio` keeps the leg's own sign and size (a fly's -2
    body) so it agrees with the PnL engine. Legs are inner-joined on date. Raises
    CrudeOilRiskError if any leg has no usable local price history.
    """
    terms = []
    for leg in structure.legs:
        diffs = _load_local_differences(leg.contract.symbol, data_loader)
        side = 1 if leg.direction == "buy" else -1
        terms.append(diffs * (leg.ratio * side * leg.lots))
    combined = pd.concat(terms, axis=1, join="inner").sum(axis=1)
    combined.name = structure.name
    return combined


def _load_local_differences(symbol: str, data_loader: DataLoader) -> pd.Series:
    """Price differences from the local Parquet only; never triggers an API backfill."""
    if data_loader.get_available_date_range(symbol) is None:
        raise CrudeOilRiskError(f"no local price data for {symbol}")
    return data_loader.load_price_differences(symbol, min_rows=MIN_OBSERVATIONS)


def _safe_range(symbol: str, data_loader: DataLoader):
    try:
        return data_loader.get_available_date_range(symbol)
    except CrudeOilRiskError:
        return None


def watchlist_item_warning(
    item: dict, structures_by_id: dict[str, Structure], data_loader: DataLoader
) -> str | None:
    """Why a watchlist item cannot be computed (missing data, unknown symbol), or None if it can."""
    if item["type"] == "structure":
        structure = structures_by_id.get(item["key"])
        if structure is None:
            return "structure not found or no longer open"
        missing = [leg.contract.symbol for leg in structure.legs if _safe_range(leg.contract.symbol, data_loader) is None]
        return f"missing price data for {', '.join(missing)}" if missing else None
    return None if _safe_range(item["key"], data_loader) is not None else "no price data found for this symbol"


@dataclass
class WatchlistResult:
    """Outcome of a watchlist correlation run: a matrix, or an error message."""

    matrix: pd.DataFrame | None = None
    observations: int = 0
    requested: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def compute_watchlist_correlation(
    items: list[dict],
    structures_by_id: dict[str, Structure],
    lookback_days: int,
    data_loader: DataLoader,
) -> WatchlistResult:
    """Correlation matrix of watchlist items ({"type": instrument|structure, "key", "label"}).

    Instruments use their own exchange-quoted price differences; structures use
    `structure_difference_series`. Items with missing data are skipped (and reported in
    `skipped`) rather than failing the run.
    """
    result = WatchlistResult(requested=lookback_days)
    if len(items) < 2:
        result.error = "Add at least 2 items to compute"
        return result

    series: dict[str, pd.Series] = {}
    for item in items:
        label = item["label"]
        try:
            if item["type"] == "structure":
                structure = structures_by_id.get(item["key"])
                if structure is None:
                    raise CrudeOilRiskError("structure not found or no longer open")
                series[label] = structure_difference_series(structure, data_loader)
            else:
                series[label] = _load_local_differences(item["key"], data_loader)
        except (CrudeOilRiskError, ValueError) as exc:
            result.skipped[label] = str(exc)

    if not series:
        result.error = "No valid data to compute — check symbols"
    elif len(series) < 2:
        result.error = "Need at least 2 valid series"
    else:
        try:
            result.matrix, result.observations = build_correlation_matrix_from_series(series, lookback_days)
        except InsufficientDataError as exc:
            result.error = str(exc)
    return result


# TODO: Structure-vs-structure correlation.
# NOTE: Structure-vs-structure correlation aggregation method not yet confirmed
# by user. Will be implemented when method is defined. See conversation notes
# from Phase 1 design.
