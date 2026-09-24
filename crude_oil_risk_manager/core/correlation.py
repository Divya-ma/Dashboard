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
from core.structure_utils import net_outright_equivalent, normalize_symbol

logger = logging.getLogger(__name__)

_HIGH_THRESHOLD = 0.7

# Floor for the portfolio-correlation-check fallback below: below this many aligned
# observations no correlation is attempted at all, matching DataLoader.align_series'
# own _MIN_COMMON_DATES floor.
_MIN_CORRELATION_OBSERVATIONS = 20


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


# ----------------------------------------------------------------------
# Structure correlation via outright decomposition (Structure Builder correlation check)
# ----------------------------------------------------------------------
#
# A structure's correlation is derived from the OUTRIGHT contracts it decomposes to, never
# from a "composed" exchange-quoted symbol's own price history. That composed symbol often
# doesn't exist at all (custom ratios, cross-product legs, a structure nested inside another
# one's leg — compose_structure_symbol returns None for all of these), and even when it does,
# the vendor may have no direct history for that exact spread/fly. Decomposing to outrights
# means only the underlying, almost-always-already-cached contracts need price data.
#
# Each structure's own derived price-DIFFERENCE series is S = sum(w_i * P_i) — a structure
# is a linear combination of its outright legs' prices, and differencing commutes with that
# combination, so building the derived series from differenced legs is equivalent to
# differencing the structure's own (unobserved) price series. This is the "Direct" method:
# build each of the two structures' own derived series (inner-joined on ITS OWN legs' common
# dates only) and correlate them directly with Pearson. A covariance matrix shared across
# every active structure at once (Cov(A,B) = a^T Sigma b, computed once for all pairs) is
# mathematically equivalent and cheaper for many structures, but requires ALL structures'
# legs to be simultaneously date-aligned — one illiquid leg on an unrelated structure then
# starves every other comparison's common-date window too. For the handful of structures a
# correlation check compares against, Direct avoids that cross-contamination.


def outright_weights(legs: list[dict]) -> dict[str, float]:
    """A structure's weight vector over outright contracts, per 1 unit of the structure.

    Thin wrapper over core.structure_utils.net_outright_equivalent — the same decomposition
    already used for the "Net Outright Equivalent" preview — so a custom ratio, a
    cross-product leg, or a leg whose own symbol is itself a spread/fly is handled exactly
    the same way here as everywhere else in the app. Legs that don't parse are silently
    dropped (see net_outright_equivalent); an empty result means nothing usable was entered.
    """
    weights, _ignored = net_outright_equivalent(legs)
    return weights


def structure_return_series(weights: dict[str, float], data_loader: DataLoader) -> pd.Series | None:
    """A structure's own derived price-difference series: sum(weight_i * diff_i) over its
    outright legs (see outright_weights), inner-joined on the dates where ALL of ITS OWN
    legs have data.

    None if no leg has usable local data, or fewer than _MIN_CORRELATION_OBSERVATIONS common
    dates remain across just its own legs. A leg with no usable data (never backfilled here
    — see DataLoader.ensure_cached) or whose read fails unexpectedly (an adapter/network
    fluke) is silently dropped rather than failing the whole series.
    """
    terms = []
    for symbol, weight in weights.items():
        try:
            terms.append(data_loader.load_price_differences(symbol, min_rows=_MIN_CORRELATION_OBSERVATIONS) * weight)
        except CrudeOilRiskError as exc:
            logger.info("Correlation: no usable price history for %s: %s", symbol, exc)
        except Exception as exc:  # noqa: BLE001 - one bad leg must not fail the whole derived series
            logger.warning("Correlation: unexpected error loading price history for %s: %s", symbol, exc)
    if not terms:
        return None

    combined = pd.concat(terms, axis=1, join="inner").dropna().sum(axis=1).sort_index()
    if len(combined) < _MIN_CORRELATION_OBSERVATIONS:
        return None
    return combined


def correlate_structures(
    weights_a: dict[str, float], weights_b: dict[str, float], window: int, data_loader: DataLoader
) -> dict:
    """Best-effort Pearson correlation between two structures via their own derived
    difference series (see structure_return_series). Never raises.

    Returns {"correlation": float | None, "classification": str, "window_used": int,
    "error": str | None}. Falls back to min(window, observations available) instead of
    requiring the full window up front (align_series' own floor of 20 common dates still
    applies below that); "error" then carries a caveat, not a failure.
    """
    entry = {"correlation": None, "classification": "insufficient_data", "window_used": 0, "error": None}
    series_a = structure_return_series(weights_a, data_loader)
    series_b = structure_return_series(weights_b, data_loader)
    if series_a is None or series_b is None:
        entry["error"] = "Not enough local price history for one of the structures' underlying contracts."
        return entry

    try:
        aligned_a, aligned_b = data_loader.align_series(series_a, series_b)
        effective_window = min(window, len(aligned_a))
        corr = _pearson_last_window(aligned_a, aligned_b, effective_window, "candidate", "structure")
    except (CrudeOilRiskError, ValueError) as exc:
        entry["error"] = str(exc)
        return entry

    entry["correlation"] = corr
    entry["classification"] = classify_correlation(corr)
    entry["window_used"] = effective_window
    if effective_window < window:
        entry["error"] = f"Only {effective_window} of the requested {window}-day window available."
    return entry


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
    series_by_label: dict[str, pd.Series], window: int, as_of: date | None = None
) -> tuple[pd.DataFrame, int]:
    """Pearson correlation matrix of pre-built price-difference series.

    Series are inner-joined on date; if `as_of` is given, dates after it are dropped
    first (so the matrix reflects the portfolio as it stood on that date), then the
    most recent `window` common days are used (fewer if that is all there is). Returns
    (matrix, observations_used). A pair whose series has zero variance over the window
    is NaN, like the symbol-based matrix. Raises InsufficientDataError if there are
    fewer than 2 series or fewer than MIN_OBSERVATIONS common days (at or before
    `as_of`, when given).
    """
    _validate_window(window)
    if len(series_by_label) < 2:
        raise InsufficientDataError("Need at least 2 valid series to compute a correlation matrix.")
    aligned = pd.concat(list(series_by_label.values()), axis=1, join="inner", keys=list(series_by_label))
    aligned = aligned.dropna().sort_index()
    if as_of is not None:
        aligned = aligned[aligned.index <= pd.Timestamp(as_of, tz="UTC")]
    if len(aligned) < MIN_OBSERVATIONS:
        as_of_desc = f" as of {as_of}" if as_of is not None else ""
        raise InsufficientDataError(
            f"Only {len(aligned)} common days across the selected series{as_of_desc}; "
            f"at least {MIN_OBSERVATIONS} are needed."
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


def watchlist_item_missing_symbols(
    item: dict, structures_by_id: dict[str, Structure], data_loader: DataLoader
) -> list[str]:
    """Exchange-quoted symbols this watchlist item needs but has no local price history for.

    Empty if the item can already be computed (or its structure/symbol is simply gone,
    which no backfill can fix). Used to offer a one-click "Backfill Missing Data" action
    instead of leaving the item stuck on "insufficient data" forever.
    """
    if item["type"] == "structure":
        structure = structures_by_id.get(item["key"])
        if structure is None:
            return []
        return [leg.contract.symbol for leg in structure.legs if _safe_range(leg.contract.symbol, data_loader) is None]
    return [] if _safe_range(item["key"], data_loader) is not None else [item["key"]]


def watchlist_item_warning(
    item: dict, structures_by_id: dict[str, Structure], data_loader: DataLoader
) -> str | None:
    """Why a watchlist item cannot be computed (missing data, unknown symbol), or None if it can."""
    if item["type"] == "structure":
        structure = structures_by_id.get(item["key"])
        if structure is None:
            return "structure not found or no longer open"
        missing = watchlist_item_missing_symbols(item, structures_by_id, data_loader)
        return f"missing price data for {', '.join(missing)}" if missing else None
    return None if _safe_range(item["key"], data_loader) is not None else "no price data found for this symbol"


def build_watchlist_series(
    items: list[dict], structures_by_id: dict[str, Structure], data_loader: DataLoader
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Price-difference series for every watchlist item ({"type", "key", "label"}), by label.

    Instruments use their own exchange-quoted price differences; structures use
    `structure_difference_series`. Local Parquet only — never backfills (see
    `_load_local_differences`), so this is safe to call on every render/compute, not just
    on an explicit user action. Items with missing/invalid data are skipped rather than
    failing the run; the second dict is {label: reason}. Shared by every watchlist view
    (Heatmap, Time Series, Year Overlay, Summary Table) so they always agree on what each
    item's series actually is.
    """
    series: dict[str, pd.Series] = {}
    skipped: dict[str, str] = {}
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
            skipped[label] = str(exc)
    return series, skipped


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
    as_of: date | None = None,
) -> WatchlistResult:
    """Correlation matrix of watchlist items ({"type": instrument|structure, "key", "label"}).

    `as_of` restricts the matrix to data at or before that date (the Heatmap tab's as-of
    slider); omit it for "as of the latest available date" (the default).
    """
    result = WatchlistResult(requested=lookback_days)
    if len(items) < 2:
        result.error = "Add at least 2 items to compute"
        return result

    series, result.skipped = build_watchlist_series(items, structures_by_id, data_loader)

    if not series:
        result.error = "No valid data to compute — check symbols"
    elif len(series) < 2:
        result.error = "Need at least 2 valid series"
    else:
        try:
            result.matrix, result.observations = build_correlation_matrix_from_series(series, lookback_days, as_of)
        except InsufficientDataError as exc:
            result.error = str(exc)
    return result


# ----------------------------------------------------------------------
# Time Series / Year Overlay / Summary Table (Correlation tab)
# ----------------------------------------------------------------------


def rolling_correlation_from_series(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
    """Rolling `window`-day Pearson correlation between two pre-built price-difference series.

    Series are inner-joined on date first (no forward-fill). Leading NaNs (the first
    window-1 observations) and any undefined windows (zero variance) are dropped. Raises
    InsufficientDataError if the two series share no common dates at all, or if nothing
    survives the rolling computation (e.g. window larger than the shared history).
    """
    _validate_window(window)
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna().sort_index()
    if aligned.empty:
        raise InsufficientDataError(f"{series_a.name} and {series_b.name} share no common trading days.")
    a, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    rolling = a.rolling(window).corr(b).replace([np.inf, -np.inf], np.nan).dropna().clip(-1.0, 1.0)
    if rolling.empty:
        raise InsufficientDataError(
            f"No {window}-day rolling correlation available for {series_a.name} vs {series_b.name}: "
            f"only {len(aligned)} common days."
        )
    rolling.name = f"{series_a.name}|{series_b.name}"
    return rolling


def year_overlay_frame(rolling: pd.Series) -> pd.DataFrame:
    """A rolling-correlation series reshaped into columns (value, doy, year) for a seasonal overlay."""
    frame = pd.DataFrame({"value": rolling.to_numpy(dtype=float)}, index=rolling.index)
    frame["doy"] = frame.index.dayofyear
    frame["year"] = frame.index.year
    return frame


def watchlist_pair_summary(series_by_label: dict[str, pd.Series], window: int) -> list[dict]:
    """Mean/std/min/max/last rolling correlation, at `window`, for every pair in the watchlist.

    Base/target pairing mirrors the heatmap: every label paired with every label after it
    (so each pair appears once). A pair with no usable rolling correlation (e.g. no shared
    history) is left out rather than raising.
    """
    _validate_window(window)
    labels = list(series_by_label)
    rows = []
    for i, base in enumerate(labels):
        for target in labels[i + 1:]:
            try:
                s = rolling_correlation_from_series(series_by_label[base], series_by_label[target], window)
            except InsufficientDataError:
                continue
            rows.append(
                {
                    "base": base,
                    "target": target,
                    "mean": float(s.mean()),
                    "std": float(s.std()) if len(s) > 1 else 0.0,
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "last": float(s.iloc[-1]),
                    "n_obs": int(len(s)),
                }
            )
    return rows


# TODO: Structure-vs-structure correlation.
# NOTE: Structure-vs-structure correlation aggregation method not yet confirmed
# by user. Will be implemented when method is defined. See conversation notes
# from Phase 1 design.
