"""Lead-lag analysis between two bar-to-bar price series (Correlation page's Lead-Lag view).

No I/O here: callers supply already-loaded OHLC bar series (see core.lead_lag_data, which decides which
contracts to compare and fetches/caches their intraday candles). Like core.correlation, every series is a
bar-to-bar PRICE DIFFERENCE, never a return (the Correlation module's own reasoning applies here too: a
spread, fly or CL-BZ box can be zero or negative, so return-based math is undefined or misleading for them).

THE MATH

  1. CROSS-CORRELATION FUNCTION (CCF). For two series A and B and a lag k (in bars, positive or negative):

         CCF(k) = Corr( A(t), B(t+k) )

     scanned over k = -max_lag..+max_lag. The k that maximizes |CCF(k)| is the lead-lag estimate: k > 0 means
     B(t+k) lines up with A(t) -- A moves first, B follows k bars later ("A leads, B lags"); k < 0 means B leads;
     k = 0 (or a flat CCF) means no detectable lead-lag at this bar resolution.

  2. PREWHITENING. A raw price-difference series has its own serial correlation (momentum, mean-reversion,
     volatility clustering), and that alone can produce a CCF peak away from zero that has nothing to do with
     either series actually leading the other -- it is each series' own autocorrelation leaking into the cross
     terms. The standard fix (Box-Jenkins): fit a small AR(p) model to each series on its own first, and
     cross-correlate the TWO RESIDUAL series instead of the raw ones. `_prewhiten` does this (AR(1) by default);
     only the residuals are ever passed into the CCF.

  3. SIGNIFICANCE. A correlation r from n (prewhitened, aligned) observations has a two-sided p-value from the
     usual t-test: t = r * sqrt((n-2)/(1-r^2)), df = n-2. Because many lags are scanned at once, the same
     Bonferroni convention used elsewhere in this app (core.journal_analysis.separation) applies: a lag "stands
     out" only if its p-value beats alpha / (number of lags tested); one that only beats alpha alone is "worth a
     look"; anything else is "no sign".

MIN_OBSERVATIONS is the floor below which a pair is not attempted at all; a lag whose own aligned window (after
the shift) falls under it is left undefined rather than computed on too little data.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from core.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)

DEFAULT_MAX_LAG = 12  # bars either side of 0 scanned for the best lag (+-60 min at 5-min bars)
DEFAULT_AR_ORDER = 1  # prewhitening order (see _prewhiten)
MIN_OBSERVATIONS = 60  # fewest aligned bar-differences any pair (or any one lag of it) is computed from

STANDS_OUT, WORTH_A_LOOK, NO_SIGN = "stands out", "worth a look", "no sign"


def bar_differences(bars: pd.DataFrame) -> pd.Series:
    """Bar-to-bar close difference of an OHLC frame (columns "timestamp", "close"), sorted and de-duplicated
    on timestamp (keeping the last row of a duplicate, as a just-refetched bar should win over a stale one)."""
    frame = bars.dropna(subset=["timestamp", "close"]).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    if frame.empty:
        return pd.Series(dtype=float)
    series = pd.Series(frame["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(frame["timestamp"]))
    return series.diff().dropna()


def _prewhiten(series: pd.Series, order: int = DEFAULT_AR_ORDER) -> pd.Series:
    """Residuals of series[t] regressed on its own last `order` values (OLS, with a constant).

    A series with zero variance (nothing happened) prewhitens to all-zero residuals rather than raising, since
    a flat leg is a legitimate (if uninteresting) input; the CCF against it is simply undefined everywhere."""
    values = series.to_numpy(dtype=float)
    if len(values) < order + 5:
        raise InsufficientDataError(f"Need at least {order + 5} bars to prewhiten (AR({order})); got {len(values)}.")
    y = values[order:]
    if np.std(y) == 0:
        return pd.Series(np.zeros_like(y), index=series.index[order:])
    x = np.column_stack([values[order - lag:len(values) - lag] for lag in range(1, order + 1)])
    model = sm.OLS(y, sm.add_constant(x)).fit()
    return pd.Series(np.asarray(model.resid, dtype=float), index=series.index[order:])


def cross_correlation(
    diff_a: pd.Series, diff_b: pd.Series, max_lag: int = DEFAULT_MAX_LAG, ar_order: int = DEFAULT_AR_ORDER,
) -> pd.DataFrame:
    """CCF(k) for k = -max_lag..+max_lag (see the module docstring): a DataFrame indexed by lag with columns
    "correlation" and "n" (the exact aligned observation count that lag's correlation was computed from -- it
    shrinks slightly at the extremes, since shifting by k trims k rows off one end).

    A and B are inner-joined on their own common bar timestamps first. Raises InsufficientDataError if fewer
    than MIN_OBSERVATIONS common bars remain before prewhitening. A lag whose own post-shift window falls under
    MIN_OBSERVATIONS, or has no variance in one side, is left NaN rather than raising.
    """
    aligned = pd.concat([diff_a, diff_b], axis=1, join="inner").dropna().sort_index()
    if len(aligned) < MIN_OBSERVATIONS:
        raise InsufficientDataError(f"Only {len(aligned)} common bars (need at least {MIN_OBSERVATIONS}).")
    a, b = _prewhiten(aligned.iloc[:, 0], ar_order), _prewhiten(aligned.iloc[:, 1], ar_order)
    rows = {}
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat([a, b.shift(-k)], axis=1).dropna()
        n = len(pair)
        if n < MIN_OBSERVATIONS or np.std(pair.iloc[:, 0]) == 0 or np.std(pair.iloc[:, 1]) == 0:
            rows[k] = (np.nan, n)
            continue
        rows[k] = (float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1]), n)
    ccf = pd.DataFrame.from_dict(rows, orient="index", columns=["correlation", "n"]).sort_index()
    if ccf["correlation"].dropna().empty:
        raise InsufficientDataError("No lag has a defined correlation (a window had no variance in one series).")
    return ccf


def _p_value(r: float, n: int) -> float:
    if n <= 2 or abs(r) >= 1.0:
        return 0.0
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    return float(2 * scipy_stats.t.sf(abs(t_stat), df=n - 2))


def best_lag(ccf: pd.DataFrame, alpha: float = 0.05) -> dict:
    """The lag that maximizes |CCF|, with its own p-value and a reading (see the module docstring).

    Positive "lag" means the first series (A, in cross_correlation's own call) leads the second (B); negative
    means B leads A; the magnitude is in bars. Raises InsufficientDataError if every lag is NaN.
    """
    valid = ccf["correlation"].dropna()
    if valid.empty:
        raise InsufficientDataError("No lag has a defined correlation.")
    lag = int(valid.abs().idxmax())
    correlation, n = float(ccf.loc[lag, "correlation"]), int(ccf.loc[lag, "n"])
    p_value = _p_value(correlation, n)
    bonferroni = alpha / len(ccf)
    reading = STANDS_OUT if p_value <= bonferroni else WORTH_A_LOOK if p_value <= alpha else NO_SIGN
    return {"lag": lag, "correlation": correlation, "n_obs": n, "p_value": p_value, "reading": reading, "n_lags_tested": len(ccf)}


def describe_lead_lag(label_a: str, label_b: str, result: dict, bar_minutes: int) -> str:
    """A plain-English sentence for the detail panel (who leads whom, by how long, how sure)."""
    lag, reading = result["lag"], result["reading"]
    if lag == 0 or reading == NO_SIGN:
        why = "they move together with no detectable lead-lag" if lag == 0 else f"no reliable lead-lag at this resolution ({reading})"
        return f"{label_a} and {label_b}: {why} (ρ={result['correlation']:+.2f} at lag 0 of {result['n_obs']} bars)."
    leader, follower = (label_a, label_b) if lag > 0 else (label_b, label_a)
    minutes = abs(lag) * bar_minutes
    return (
        f"{leader} leads {follower} by about {minutes} minute{'s' if minutes != 1 else ''} ({abs(lag)} bar{'s' if abs(lag) != 1 else ''}): "
        f"ρ={result['correlation']:+.2f}, {reading} (p={result['p_value']:.3f} of {result['n_lags_tested']} lags tested), "
        f"{result['n_obs']} aligned bars."
    )


def lead_lag_matrix(
    diffs_by_label: dict[str, pd.Series], max_lag: int = DEFAULT_MAX_LAG, ar_order: int = DEFAULT_AR_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], dict]]:
    """Lag and strength matrices across every pair of `diffs_by_label`, plus the full per-pair detail for a
    click-through explanation. Best-effort: a pair that fails (too few shared bars, no defined lag) is left
    NaN and logged, never raised, matching core.correlation.build_correlation_matrix's own convention.

    Returns (lag_matrix, strength_matrix, detail):
      lag_matrix[i, j]      bars label i leads label j by (negative: j leads i); lag_matrix[j, i] == -lag_matrix[i, j].
      strength_matrix[i, j] |correlation| at that best lag, in [0, 1], symmetric; diagonal is 1.0.
      detail[(i, j)]        {**best_lag(...), "ccf": the DataFrame cross_correlation returned}, i always before
                             j in `diffs_by_label`'s own order, for the click-through panel to read either way.
    """
    labels = list(diffs_by_label)
    n = len(labels)
    lag_values = np.zeros((n, n))
    strength_values = np.full((n, n), np.nan)
    detail: dict[tuple[str, str], dict] = {}
    for i in range(n):
        strength_values[i, i] = 1.0
        for j in range(i + 1, n):
            a, b = labels[i], labels[j]
            try:
                ccf = cross_correlation(diffs_by_label[a], diffs_by_label[b], max_lag, ar_order)
                result = best_lag(ccf)
            except InsufficientDataError as exc:
                logger.warning("Lead-lag failed for %s / %s: %s", a, b, exc)
                continue
            lag_values[i, j], lag_values[j, i] = result["lag"], -result["lag"]
            strength_values[i, j] = strength_values[j, i] = abs(result["correlation"])
            detail[(a, b)] = {**result, "ccf": ccf}
    return (
        pd.DataFrame(lag_values, index=labels, columns=labels),
        pd.DataFrame(strength_values, index=labels, columns=labels),
        detail,
    )
