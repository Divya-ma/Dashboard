"""Correlation analysis between price series (contract-vs-contract).

Price DIFFERENCES (close_t - close_{t-1}) are used throughout, never log or
simple returns. Futures spreads and flies are stationary in price-difference
space and can be zero or negative, so return-based measures (which assume
non-stationary, strictly positive, equity-like prices) are undefined or
misleading for them. Every series here is an exchange-quoted symbol with its
own price history — spreads/flies are never derived from their legs.

All data access goes through DataLoader; this module does no file I/O.

HOW "THE" CORRELATION OF TWO SERIES IS MEASURED (everywhere a single number is shown: the
Structure Builder check, the Heatmap, the Summary table, Monte Carlo VaR's matrix):

  1. ROLLING WINDOW (R days). For each day, the Pearson correlation of the two series' price
     differences over the R days ending that day. That gives one correlation value per day.
  2. LOOKBACK PERIOD (L days). The plain average of those daily values over the last L days.

Defaults are R = 5 and L = 20 (Settings > Analysis Defaults), so with the defaults a pair needs
R + L - 1 = 24 common days. R must be at least 2 (a correlation needs two points). A day whose
window has no variance in a series has no value and is left out of the average. If fewer than L
days of values exist, the average uses what there is and reports how many days that was.
The Time Series and Year Overlay views plot the daily values of step 1 themselves.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from adapters.base import PRODUCT_TO_API_CODE, SymbolTranslator
from core.data_loader import DataLoader
from core.excel_correlation import ExcelCorrelationStore
from core.exceptions import CrudeOilRiskError, InsufficientDataError
from core.models import Structure
from core.structure_utils import net_outright_equivalent, normalize_symbol

logger = logging.getLogger(__name__)

_HIGH_THRESHOLD = 0.7

# Floor for the portfolio-correlation-check fallback below: below this many aligned
# observations no correlation is attempted at all, matching DataLoader.align_series'
# own _MIN_COMMON_DATES floor.
MIN_COMMON_DAYS = 20  # fewest shared daily changes any correlation in the app is computed from
_MIN_CORRELATION_OBSERVATIONS = MIN_COMMON_DAYS


def _validate_window(window: int) -> None:
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")


def _validate_rolling(rolling: int) -> None:
    if rolling < 2:
        raise ValueError(f"rolling window must be >= 2 (a correlation needs at least two days), got {rolling}")


def _validate_lookback(lookback: int) -> None:
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")


def beta_window(rolling: int, lookback: int) -> int:
    """Observations the beta panels regress over: the span the correlation uses, but never fewer
    than the 30 a reliable OLS fit needs (core.regression)."""
    return max(30, rolling + lookback - 1)


def daily_correlations(a: pd.Series, b: pd.Series, rolling: int) -> pd.Series:
    """Step 1: per-day Pearson correlation over the last `rolling` days; NaN where undefined."""
    return a.rolling(rolling).corr(b).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)


def lookback_average_correlation(
    series_a: pd.Series, series_b: pd.Series, rolling: int, lookback: int
) -> tuple[float, int]:
    """Step 2: average of the daily `rolling`-day correlations over the last `lookback` days.

    Series are inner-joined on date first (no forward-fill). Returns (average, days_used):
    days_used is how many of the last `lookback` days had a defined value (fewer than `lookback`
    if the history is short or a window had no variance). Raises InsufficientDataError if the
    series share no days or no day has a defined value.
    """
    _validate_rolling(rolling)
    _validate_lookback(lookback)
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna().sort_index()
    if aligned.empty:
        raise InsufficientDataError(f"{series_a.name} and {series_b.name} share no common trading days.")
    daily = daily_correlations(aligned.iloc[:, 0], aligned.iloc[:, 1], rolling).iloc[-lookback:].dropna()
    if daily.empty:
        raise InsufficientDataError(
            f"No {rolling}-day correlation is defined for {series_a.name} vs {series_b.name} "
            f"({len(aligned)} common days; a window needs variance in both series)."
        )
    return float(daily.mean()), int(len(daily))


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


def calculate_correlation(
    symbol_a: str,
    symbol_b: str,
    rolling: int,
    lookback: int,
    data_loader: DataLoader,
    start: date | None = None,
    end: date | None = None,
) -> float:
    """Correlation of two symbols: the average over the last `lookback` days of the daily
    `rolling`-day Pearson correlation of their price differences (see the module docstring).

    The two symbols are aligned on their own common dates (at least MIN_COMMON_DAYS of them); if fewer than
    `lookback` days have a value the average uses the days there are, exactly as every other page does
    (lookback_average_correlation). Raises InsufficientDataError when there is nothing to average. Never
    returns NaN. Result is in [-1.0, 1.0].
    """
    _validate_rolling(rolling)
    _validate_lookback(lookback)
    aligned_a, aligned_b = _load_aligned(symbol_a, symbol_b, MIN_COMMON_DAYS, data_loader, start, end)
    value, _ = lookback_average_correlation(aligned_a, aligned_b, rolling, lookback)
    return value


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
    rolling: int,
    lookback: int,
    data_loader: DataLoader,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """N x N correlation matrix (diagonal 1.0, symmetric), each cell calculate_correlation().

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
                corr = calculate_correlation(symbols[i], symbols[j], rolling, lookback, data_loader, start, end)
            except (CrudeOilRiskError, ValueError) as exc:
                logger.warning("Correlation failed for %s / %s: %s", symbols[i], symbols[j], exc)
                continue
            values[i, j] = corr
            values[j, i] = corr
    return pd.DataFrame(values, index=symbols, columns=symbols)


def classify_correlation(correlation: float | None, threshold: float | None = None) -> str:
    """Classify a correlation coefficient.

    >= threshold -> "highly_correlated"; <= -threshold -> "negatively_correlated";
    otherwise "uncorrelated"; NaN/None -> "insufficient_data". The threshold defaults to 0.7; the
    Structure Builder passes its own (core.structure_builder.CORRELATION_FLAG_THRESHOLD).
    """
    limit = _HIGH_THRESHOLD if threshold is None else threshold
    if correlation is None or np.isnan(correlation):
        return "insufficient_data"
    if correlation >= limit:
        return "highly_correlated"
    if correlation <= -limit:
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


@dataclass
class StructureSeries:
    """A structure's derived price-difference series, or why there is none."""

    series: pd.Series | None
    missing: list[str] = field(default_factory=list)  # legs with no usable price history
    proxied: list[str] = field(default_factory=list)  # legs priced from a proxy (see core.settlement_prices)


def _leg_closes(symbol: str, data_loader: DataLoader, settlements) -> tuple[pd.Series | None, bool]:
    """Daily closes of one outright: its settlements when a settlement source is given and has them, else the DataLoader's."""
    if settlements is not None:
        try:
            series, proxied = settlements.closes(symbol)
        except Exception as exc:  # noqa: BLE001 - fall back to the loader rather than fail the series
            logger.warning("Correlation: settlement lookup failed for %s: %s", symbol, exc)
            series, proxied = None, False
        if series is not None and not series.empty:
            return series, proxied
    try:
        return data_loader.load_close_series(symbol, min_rows=_MIN_CORRELATION_OBSERVATIONS + 1), False
    except CrudeOilRiskError as exc:
        logger.info("Correlation: no usable price history for %s: %s", symbol, exc)
    except Exception as exc:  # noqa: BLE001 - one bad leg must not fail the whole derived series
        logger.warning("Correlation: unexpected error loading price history for %s: %s", symbol, exc)
    return None, False


def structure_series(weights: dict[str, float], data_loader: DataLoader, settlements=None) -> StructureSeries:
    """A structure's own price-difference series from its outright legs (see outright_weights).

    The structure's PRICE is built first, S_t = sum(weight_i * P_i,t) on the dates where ALL of ITS OWN legs have a
    close, and the daily difference S_t - S_(t-1) is taken afterwards, over those same dates. (Differencing each leg
    first and adding the differences would mix a one-day change of one leg with a two-day change of another wherever
    a leg has a missing day.) Every leg is required: dropping one would silently turn the structure into a different
    one, so a leg without history makes the whole series None and is named in `missing`. Fewer than
    _MIN_CORRELATION_OBSERVATIONS differences also gives None.

    `settlements` (core.settlement_prices.SettlementPrices) supplies gap-free settlement closes for CL and BRN
    outrights (BZ from the ICE Brent contract of the same month); anything it does not have is read from `data_loader`.
    """
    levels, missing, proxied = {}, [], []
    for symbol, weight in weights.items():
        closes, via_proxy = _leg_closes(symbol, data_loader, settlements)
        if closes is None or closes.empty:
            missing.append(symbol)
            continue
        levels[symbol] = closes * weight
        if via_proxy:
            proxied.append(symbol)
    if missing or not levels:
        return StructureSeries(None, missing, proxied)
    price = pd.concat(levels.values(), axis=1, join="inner").dropna().sum(axis=1).sort_index()
    diffs = price.diff().dropna()
    if len(diffs) < _MIN_CORRELATION_OBSERVATIONS:
        return StructureSeries(None, [], proxied)
    return StructureSeries(diffs, [], proxied)


def structure_return_series(weights: dict[str, float], data_loader: DataLoader, settlements=None) -> pd.Series | None:
    """The price-difference series of `structure_series`, or None (see there)."""
    return structure_series(weights, data_loader, settlements).series


def correlate_structures(
    weights_a: dict[str, float], weights_b: dict[str, float], rolling: int, lookback: int, data_loader: DataLoader,
    settlements=None, threshold: float | None = None,
) -> dict:
    """Best-effort correlation between two structures via their own derived difference series
    (see structure_series), measured as in the module docstring. Never raises.

    Returns {"correlation": float | None, "classification": str, "window_used": int,
    "rolling": int, "error": str | None, "note": str | None}. `window_used` is how many days of the lookback
    were averaged. Falls back to the days available instead of requiring the full lookback up front
    (align_series' own floor of 20 common dates still applies); "error" then carries a caveat, not a failure.
    `note` says when a leg was priced from a proxy. `threshold` is passed to classify_correlation.
    """
    entry = {"correlation": None, "classification": "insufficient_data", "window_used": 0, "rolling": rolling, "error": None, "note": None}
    built_a = structure_series(weights_a, data_loader, settlements)
    built_b = structure_series(weights_b, data_loader, settlements)
    proxied = sorted({*built_a.proxied, *built_b.proxied})
    if proxied:
        from core.settlement_prices import PROXY_NOTE  # local import: keeps the maths module free of the data source

        entry["note"] = PROXY_NOTE
    if built_a.series is None or built_b.series is None:
        missing = sorted({*built_a.missing, *built_b.missing})
        entry["error"] = (
            f"No price history for {', '.join(missing)}." if missing
            else "Not enough shared price history across the underlying contracts of one of the structures."
        )
        return entry

    try:
        aligned_a, aligned_b = data_loader.align_series(built_a.series, built_b.series)
        corr, used = lookback_average_correlation(aligned_a, aligned_b, rolling, lookback)
    except (CrudeOilRiskError, ValueError) as exc:
        entry["error"] = str(exc)
        return entry

    entry["correlation"] = corr
    entry["classification"] = classify_correlation(corr, threshold)
    entry["window_used"] = used
    if used < lookback:
        entry["error"] = f"Only {used} of the requested {lookback}-day lookback available."
    return entry


# ----------------------------------------------------------------------
# Watchlist correlation (Correlation Heatmap tab)
# ----------------------------------------------------------------------

# Fewest common daily observations a heatmap will be computed from.
MIN_OBSERVATIONS = MIN_COMMON_DAYS

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
    series_by_label: dict[str, pd.Series], rolling: int, lookback: int, as_of: date | None = None
) -> tuple[pd.DataFrame, int]:
    """Correlation matrix of pre-built price-difference series (see the module docstring).

    Each cell is lookback_average_correlation of that PAIR (the two series aligned on their own common dates), so a
    cell does not change when another item is added to the watchlist and the Heatmap agrees with the Summary table
    and the Time Series. If `as_of` is given, dates after it are dropped first (the matrix as the portfolio stood
    then). Returns (matrix, lookback_days_used): the fewest days any cell averaged, up to `lookback`. A pair with fewer
    than MIN_COMMON_DAYS common days or no defined value is NaN, as is a series with no variance on the diagonal.
    Raises InsufficientDataError if there are fewer than 2 series or no pair could be computed.
    """
    _validate_rolling(rolling)
    _validate_lookback(lookback)
    if len(series_by_label) < 2:
        raise InsufficientDataError("Need at least 2 valid series to compute a correlation matrix.")
    limit = pd.Timestamp(as_of, tz="UTC") if as_of is not None else None
    series = {label: (s[s.index <= limit] if limit is not None else s).dropna().sort_index() for label, s in series_by_label.items()}
    labels = list(series)
    values = np.full((len(labels), len(labels)), np.nan)
    used_min, reasons = lookback, []
    for i, a in enumerate(labels):
        values[i, i] = 1.0 if len(series[a]) > 1 and np.std(series[a].to_numpy(dtype=float)) > 0 else np.nan
        for j in range(i + 1, len(labels)):
            b = labels[j]
            common = series[a].index.intersection(series[b].index)
            if len(common) < max(MIN_COMMON_DAYS, rolling):
                reasons.append(f"{a} / {b}: {len(common)} common days")
                continue
            try:
                value, used = lookback_average_correlation(series[a], series[b], rolling, lookback)
            except InsufficientDataError as exc:
                reasons.append(str(exc))
                continue
            values[i, j] = values[j, i] = value
            used_min = min(used_min, used)
    if not np.isfinite(values[np.triu_indices(len(labels), 1)]).any():
        as_of_desc = f" as of {as_of}" if as_of is not None else ""
        raise InsufficientDataError(
            f"No pair has enough shared history{as_of_desc}: at least {max(MIN_COMMON_DAYS, rolling)} common days are needed"
            + (f" ({reasons[0]})." if reasons else ".")
        )
    return pd.DataFrame(values, index=labels, columns=labels), used_min


def structure_difference_series(structure: Structure, data_loader: DataLoader) -> pd.Series:
    """Daily synthetic PnL-difference series of a structure, in point x lots.

    The structure's price is built first, S[t] = sum(close[t] * ratio * side * lots) over the legs on the dates where
    EVERY leg has a close, and the daily difference is taken afterwards (core.correlation.structure_series), so a leg
    with a missing day cannot mix a one-day change with a two-day one. `side` is +1 for a buy and -1 for a sell and
    `ratio` keeps the leg's own sign and size (a fly's -2 body), so it agrees with the PnL engine. Raises
    CrudeOilRiskError if any leg has no usable LOCAL price history (nothing is backfilled here).
    """
    weights: dict[str, float] = {}
    for leg in structure.legs:
        side = 1 if leg.direction == "buy" else -1
        weights[leg.contract.symbol] = weights.get(leg.contract.symbol, 0.0) + leg.ratio * side * leg.lots
    for symbol in weights:
        if data_loader.get_available_date_range(symbol) is None:
            raise CrudeOilRiskError(f"no local price data for {symbol}")
    built = structure_series(weights, data_loader)
    if built.series is None:
        raise CrudeOilRiskError(
            f"no price history for {', '.join(built.missing)}" if built.missing
            else f"fewer than {MIN_COMMON_DAYS} shared trading days across the legs of {structure.name}"
        )
    series = built.series.copy()
    series.name = structure.name
    return series


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
    if item["type"] == "excel":
        return []  # nothing to backfill for an uploaded file
    return [] if _safe_range(item["key"], data_loader) is not None else [item["key"]]


def watchlist_item_warning(
    item: dict,
    structures_by_id: dict[str, Structure],
    data_loader: DataLoader,
    excel_store: ExcelCorrelationStore | None = None,
) -> str | None:
    """Why a watchlist item cannot be computed (missing data, unknown symbol), or None if it can."""
    if item["type"] == "structure":
        structure = structures_by_id.get(item["key"])
        if structure is None:
            return "structure not found or no longer open"
        missing = watchlist_item_missing_symbols(item, structures_by_id, data_loader)
        return f"missing price data for {', '.join(missing)}" if missing else None
    if item["type"] == "excel":
        if excel_store is None:
            return "Excel data source not available"
        filename, sheet, column = parse_excel_item_key(item["key"])
        try:
            excel_store.load_series(filename, sheet, column, min_rows=MIN_OBSERVATIONS)
            return None
        except CrudeOilRiskError as exc:
            return str(exc)
    return None if _safe_range(item["key"], data_loader) is not None else "no price data found for this symbol"


_EXCEL_KEY_SEP = "\x1f"  # unit separator: won't collide with realistic filenames/sheets/columns


def excel_item_key(filename: str, sheet: str, column: str) -> str:
    """Pack an Excel-sourced watchlist item's identity into its "key" field."""
    return _EXCEL_KEY_SEP.join((filename, sheet, column))


def parse_excel_item_key(key: str) -> tuple[str, str, str]:
    filename, sheet, column = key.split(_EXCEL_KEY_SEP)
    return filename, sheet, column


def build_watchlist_series(
    items: list[dict],
    structures_by_id: dict[str, Structure],
    data_loader: DataLoader,
    excel_store: ExcelCorrelationStore | None = None,
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Price-difference series for every watchlist item ({"type", "key", "label"}), by label.

    Instruments use their own exchange-quoted price differences; structures use
    `structure_difference_series`; "excel" items (a manual fallback source for when the API
    can't provide a symbol — see core.excel_correlation) read a column from an uploaded
    workbook via `excel_store`. Local Parquet / uploaded files only — never backfills (see
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
            elif item["type"] == "excel":
                if excel_store is None:
                    raise CrudeOilRiskError("Excel data source not available")
                filename, sheet, column = parse_excel_item_key(item["key"])
                series[label] = excel_store.load_series(filename, sheet, column, min_rows=MIN_OBSERVATIONS)
            else:
                series[label] = _load_local_differences(item["key"], data_loader)
        except (CrudeOilRiskError, ValueError) as exc:
            skipped[label] = str(exc)
    return series, skipped


@dataclass
class WatchlistResult:
    """Outcome of a watchlist correlation run: a matrix, or an error message."""

    matrix: pd.DataFrame | None = None
    observations: int = 0  # lookback days actually averaged
    requested: int = 0  # the lookback asked for
    rolling: int = 0
    skipped: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def compute_watchlist_correlation(
    items: list[dict],
    structures_by_id: dict[str, Structure],
    rolling: int,
    lookback_days: int,
    data_loader: DataLoader,
    as_of: date | None = None,
    excel_store: ExcelCorrelationStore | None = None,
) -> WatchlistResult:
    """Correlation matrix of watchlist items ({"type": instrument|structure|excel, "key", "label"}).

    `as_of` restricts the matrix to data at or before that date (the Heatmap tab's as-of
    slider); omit it for "as of the latest available date" (the default).
    """
    result = WatchlistResult(requested=lookback_days, rolling=rolling)
    if len(items) < 2:
        result.error = "Add at least 2 items to compute"
        return result

    series, result.skipped = build_watchlist_series(items, structures_by_id, data_loader, excel_store)

    if not series:
        result.error = "No valid data to compute — check symbols"
    elif len(series) < 2:
        result.error = "Need at least 2 valid series"
    else:
        try:
            result.matrix, result.observations = build_correlation_matrix_from_series(series, rolling, lookback_days, as_of)
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


def watchlist_pair_summary(series_by_label: dict[str, pd.Series], rolling: int, lookback: int) -> list[dict]:
    """Per pair: the lookback-average correlation (`avg`) and the std/min/max/last of the daily
    `rolling`-day correlations over that lookback, plus target-on-base beta.

    Base/target pairing mirrors the heatmap: every label paired with every label after it
    (so each pair appears once). A pair with no usable correlation (e.g. no shared history) is
    left out rather than raising. Beta (target = alpha + beta * base, over beta_window days) is
    best effort: if the regression is underdetermined (see core.regression.beta_from_series),
    beta and r_squared are None but the row is still included.
    """
    from core.regression import beta_from_series  # local import: avoids a cycle at module load

    _validate_rolling(rolling)
    _validate_lookback(lookback)
    labels = list(series_by_label)
    rows = []
    for i, base in enumerate(labels):
        for target in labels[i + 1:]:
            try:
                daily = rolling_correlation_from_series(series_by_label[base], series_by_label[target], rolling)
            except InsufficientDataError:
                continue
            s = daily.iloc[-lookback:]
            beta = r_squared = None
            try:
                beta_result = beta_from_series(
                    series_by_label[target], series_by_label[base], beta_window(rolling, lookback)
                )
                beta, r_squared = beta_result["beta"], beta_result["r_squared"]
            except InsufficientDataError:
                pass
            rows.append(
                {
                    "base": base,
                    "target": target,
                    "avg": float(s.mean()),
                    "std": float(s.std()) if len(s) > 1 else 0.0,
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "last": float(s.iloc[-1]),
                    "n_obs": int(len(s)),
                    "beta": beta,
                    "r_squared": r_squared,
                }
            )
    return rows


# TODO: Structure-vs-structure correlation.
# NOTE: Structure-vs-structure correlation aggregation method not yet confirmed
# by user. Will be implemented when method is defined. See conversation notes
# from Phase 1 design.
