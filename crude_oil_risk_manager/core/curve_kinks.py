"""Kink detection on a forward curve, by four independent methods plus a seasonal check.

A KINK is a contract (or spread / fly / dfly) that sits away from the rest of its curve. For
each position of one curve (one product, one family) four methods each produce a z-score:

  fit        Robust polynomial fit across the whole curve (Tukey-biweight IRLS, so a kink does
             not drag the fit toward itself); residual / the larger of today's cross-sectional
             robust scale (MAD of 15 points: noisy on its own) and that position's historical
             fit-residual scale.
  neighbour  Leave-one-out interpolation: each point is predicted from its neighbours (cubic
             Lagrange; quadratic at the ends); residual / that position's own historical
             residual scale. Equivalent to a second difference, so it is a fly's logic applied
             to every point.
  pca        The curve is rebuilt from the top principal components of its history (level,
             slope, curvature ...); the leftover residual / the historical residual scale at
             that position. Moves the history says are "normal" are not kinks.
  history    The value against its own history at the same generic position (median / robust
             std). Spread, fly and dfly curves only: an outright's level just follows price.

A method FLAGS a position when |z| exceeds its threshold. Agreement raises the score and the
priority: HIGH needs `min_methods_high` methods, MEDIUM one. SEASONALITY: the neighbour
residual of that delivery month on the same time of year in earlier years gives a seasonal z;
if |seasonal z| is small the contract is "seasonally normal" and its priority drops one level
(it is still reported, with that note). All scales have a floor (`min_scale`) so a very flat
curve cannot produce enormous z-scores.
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from core.curve_settings import PRIORITY_RANK, CurveParams

METHODS = ("fit", "neighbour", "pca", "history")
METHOD_LABELS = {"fit": "Fit", "neighbour": "Neighbour", "pca": "PCA", "history": "History"}
MIN_HISTORY_ROWS = 30
_MAD_TO_STD = 1.4826
_MAX_STRENGTH = 3.0
_PRIORITY_BY_LEVEL = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


def robust_scale(values: np.ndarray, axis=None, floor: float = 0.0) -> np.ndarray:
    """1.4826 * median absolute deviation, never below `floor`."""
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=axis)
    return np.maximum(_MAD_TO_STD * mad, floor)


def neighbour_weights(n: int) -> np.ndarray:
    """W (n x n): row k holds the weights predicting point k from its neighbours (zero diagonal).

    Interior: cubic through k-2,k-1,k+1,k+2 = (-1/6, 2/3, 2/3, -1/6); next to an end: quadratic
    through the three nearest other points; at an end: quadratic extrapolation (3, -3, 1).
    """
    w = np.zeros((n, n))
    for k in range(n):
        if 2 <= k <= n - 3:
            w[k, [k - 2, k - 1, k + 1, k + 2]] = (-1 / 6, 2 / 3, 2 / 3, -1 / 6)
        elif k == 1 and n >= 4:
            w[k, [k - 1, k + 1, k + 2]] = (1 / 3, 1, -1 / 3)
        elif k == n - 2 and n >= 4:
            w[k, [k + 1, k - 1, k - 2]] = (1 / 3, 1, -1 / 3)
        elif k == 0 and n >= 4:
            w[k, [1, 2, 3]] = (3, -3, 1)
        elif k == n - 1 and n >= 4:
            w[k, [n - 2, n - 3, n - 4]] = (3, -3, 1)
        elif 0 < k < n - 1:  # tiny curves: plain average of the two neighbours
            w[k, [k - 1, k + 1]] = (0.5, 0.5)
    return w


def neighbour_residuals(values: np.ndarray) -> np.ndarray:
    """value - interpolation from neighbours, along the last axis. NaN where a needed neighbour is NaN."""
    values = np.asarray(values, dtype=float)
    w = neighbour_weights(values.shape[-1])
    missing = np.isnan(values)
    needed = (w != 0).astype(float)
    bad = (missing.astype(float) @ needed.T) > 0
    residual = np.where(missing, 0.0, values) - np.where(missing, 0.0, values) @ w.T
    return np.where(bad | missing, np.nan, residual)


def robust_fit(y: np.ndarray, degree: int, iterations: int = 12) -> np.ndarray:
    """Tukey-biweight robust polynomial fit of y against position; NaNs are ignored. Returns the fit (NaN where y is NaN)."""
    n = len(y)
    ok = np.isfinite(y)
    degree = min(degree, int(ok.sum()) - 2)
    if degree < 1:
        return np.full(n, np.nan)
    x = np.linspace(-1.0, 1.0, n)
    design = np.vander(x[ok], degree + 1)
    weights = np.ones(int(ok.sum()))
    coefficients = np.zeros(degree + 1)
    for _ in range(iterations):
        weighted = design * weights[:, None]
        try:  # weighted normal equations: far cheaper than a least-squares SVD for this tiny, well-scaled problem
            coefficients = np.linalg.solve(design.T @ weighted + 1e-12 * np.eye(degree + 1), weighted.T @ y[ok])
        except np.linalg.LinAlgError:
            root = np.sqrt(weights)
            coefficients, *_ = np.linalg.lstsq(design * root[:, None], y[ok] * root, rcond=None)
        residual = y[ok] - design @ coefficients
        scale = max(_MAD_TO_STD * float(np.median(np.abs(residual - np.median(residual)))), 1e-9)
        u = residual / (4.685 * scale)
        weights = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0) + 1e-6
    fit = np.full(n, np.nan)
    fit[ok] = design @ coefficients
    return fit


def fit_residuals(values: np.ndarray, degree: int, iterations: int = 12) -> np.ndarray:
    """Robust-fit residual of every row of a T x n matrix (each day fitted on its own).

    Same Tukey-biweight IRLS as robust_fit, run for all complete rows at once (the design matrix
    is the same for every day); rows with a missing value fall back to robust_fit one by one."""
    values = np.asarray(values, dtype=float)
    rows, n = values.shape
    out = np.full(values.shape, np.nan)
    complete = np.isfinite(values).all(axis=1)
    degree_used = min(degree, n - 2)
    if complete.any() and degree_used >= 1:
        v = values[complete]
        design = np.vander(np.linspace(-1.0, 1.0, n), degree_used + 1)
        ridge = 1e-12 * np.eye(degree_used + 1)
        weights = np.ones_like(v)
        for _ in range(iterations):
            normal = np.einsum("tn,np,nq->tpq", weights, design, design) + ridge
            rhs = np.einsum("tn,np,tn->tp", weights, design, v)
            coefficients = np.linalg.solve(normal, rhs[..., None])[..., 0]
            residual = v - coefficients @ design.T
            centre = np.median(residual, axis=1, keepdims=True)
            scale = np.maximum(_MAD_TO_STD * np.median(np.abs(residual - centre), axis=1, keepdims=True), 1e-9)
            u = residual / (4.685 * scale)
            weights = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0) + 1e-6
        out[complete] = residual
    for row in np.flatnonzero(~complete):
        out[row] = values[row] - robust_fit(values[row], degree)
    return out


def circular_day_difference(day_of_year: np.ndarray, target: int) -> np.ndarray:
    diff = np.abs(day_of_year - target)
    return np.minimum(diff, 366 - diff)


# ----------------------------------------------------------------------
# History preparation (computed once per history refresh, then reused every poll)
# ----------------------------------------------------------------------


@dataclass
class PreparedHistory:
    values: np.ndarray  # T x n, full history
    dates: np.ndarray  # T, datetime64[D]
    months: np.ndarray | None  # T x n delivery month (front leg) per date/position
    n: int
    center: np.ndarray  # n: median per position (history method)
    spread: np.ndarray  # n: robust std per position
    resid_scale: np.ndarray  # n: scale of the neighbour residual per position
    resid_hist: np.ndarray  # T x n: neighbour residual history (seasonal baseline)
    fit_scale: np.ndarray  # n: scale of the robust-fit residual per position
    pca_mean: np.ndarray | None = None
    pca_components: np.ndarray | None = None  # m x n
    pca_scale: np.ndarray | None = None  # n
    year_of: np.ndarray = field(default_factory=lambda: np.zeros(0))
    doy_of: np.ndarray = field(default_factory=lambda: np.zeros(0))


def prepare_history(
    values: np.ndarray, dates: np.ndarray, months: np.ndarray | None, params: CurveParams,
    fit_resid: np.ndarray | None = None,
) -> PreparedHistory | None:
    """Scales, PCA and the residual history from a T x n matrix of daily values (oldest first).

    None if there are fewer than MIN_HISTORY_ROWS rows. `values` must be complete (no NaN rows).
    `fit_resid` (fit_residuals(values, degree), same rows) can be passed in to avoid refitting
    every day when many windows of one history are prepared (the backtest does this).
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < MIN_HISTORY_ROWS:
        return None
    n = values.shape[1]
    window = values[-params.lookback_days:]
    resid_hist = neighbour_residuals(values)
    prepared = PreparedHistory(
        values=values, dates=dates, months=months, n=n,
        center=np.median(window, axis=0),
        spread=robust_scale(window, axis=0, floor=params.min_scale),
        resid_scale=robust_scale(resid_hist[-params.lookback_days:], axis=0, floor=params.min_scale),
        resid_hist=resid_hist,
        fit_scale=robust_scale(
            fit_resid[-params.lookback_days:] if fit_resid is not None else fit_residuals(window, params.poly_degree),
            axis=0, floor=params.min_scale,
        ),
    )
    dates_d = np.asarray(dates, dtype="datetime64[D]")
    years = dates_d.astype("datetime64[Y]").astype(int) + 1970
    prepared.year_of = years
    prepared.doy_of = (dates_d - dates_d.astype("datetime64[Y]")).astype(int) + 1

    components = min(params.pca_components, n - 1)
    if components >= 1 and window.shape[0] > components + 5:
        mean = window.mean(axis=0)
        centered = window - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:components]
        leftover = centered - centered @ basis.T @ basis
        prepared.pca_mean, prepared.pca_components = mean, basis
        prepared.pca_scale = robust_scale(leftover, axis=0, floor=params.min_scale)
    return prepared


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------


@dataclass
class PositionResult:
    index: int
    value: float | None
    fitted: float | None  # robust-fit curve value
    implied: float | None  # value implied by the neighbours
    residual: float | None  # value - implied (neighbour residual)
    z: dict[str, float | None]
    flags: dict[str, bool]
    n_flags: int
    score: float
    direction: str  # "rich" (above the curve), "cheap" (below) or ""
    priority: str  # NONE / LOW / MEDIUM / HIGH (after the seasonal adjustment)
    seasonal_z: float | None = None
    seasonal_normal: bool = False
    seasonal_obs: int = 0
    echo_of: int | None = None  # index of the stronger neighbouring kink this one is an interpolation echo of
    fair: dict[str, float | None] = field(default_factory=dict)  # each method's idea of the fair value


def _pca_z(values: np.ndarray, prepared: PreparedHistory) -> tuple[np.ndarray, np.ndarray]:
    """(z, implied value) from rebuilding the curve with the principal components; NaN where unavailable."""
    z = np.full(len(values), np.nan)
    implied = np.full(len(values), np.nan)
    if prepared.pca_components is None:
        return z, implied
    deviation = values - prepared.pca_mean
    ok = np.isfinite(deviation)
    basis = prepared.pca_components
    if ok.sum() <= basis.shape[0] + 1:
        return z, implied
    coefficients, *_ = np.linalg.lstsq(basis[:, ok].T, deviation[ok], rcond=None)
    leftover = deviation - coefficients @ basis
    z[ok] = leftover[ok] / prepared.pca_scale[ok]
    implied[ok] = values[ok] - leftover[ok]
    return z, implied


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def seasonal_context(
    prepared: PreparedHistory, month_numbers: list[int], residuals: np.ndarray, today: date, params: CurveParams
) -> list[tuple[float | None, int]]:
    """Per position: (seasonal z of the neighbour residual or None, observations used).

    Sample = this delivery month's residual on dates within +/- seasonal_window_days of today's
    day of year, in earlier calendar years only; needs >= 10 observations from >= 2 years.
    """
    out: list[tuple[float | None, int]] = []
    if prepared.months is None:
        return [(None, 0)] * len(month_numbers)
    day_ok = (prepared.year_of < today.year) & (
        circular_day_difference(prepared.doy_of, today.timetuple().tm_yday) <= params.seasonal_window_days
    )
    resid_rows = prepared.resid_hist[day_ok]
    month_rows = prepared.months[day_ok]
    year_rows = prepared.year_of[day_ok]
    for index, month in enumerate(month_numbers):
        residual = residuals[index]
        if not np.isfinite(residual):
            out.append((None, 0))
            continue
        pick = (month_rows == month) & np.isfinite(resid_rows)
        sample = resid_rows[pick]
        years = np.unique(np.broadcast_to(year_rows[:, None], resid_rows.shape)[pick])
        if len(sample) < 10 or len(years) < 2:
            out.append((None, len(sample)))
            continue
        scale = float(robust_scale(sample, floor=params.min_scale))
        out.append(((residual - float(np.median(sample))) / scale, len(sample)))
    return out


def _detect_once(
    values: np.ndarray,
    prepared: PreparedHistory | None,
    use_history_method: bool,
    params: CurveParams,
    month_numbers: list[int] | None,
    today: date | None,
) -> list[PositionResult]:
    """One pass of all methods on one live curve (`values`, one entry per position, NaN = no price)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    fitted = robust_fit(values, params.poly_degree)
    fit_residual = values - fitted
    if np.isfinite(fit_residual).sum() >= 3:
        cross_scale = float(robust_scale(fit_residual[np.isfinite(fit_residual)], floor=params.min_scale))
        if prepared is not None and prepared.n == n:
            fit_z = fit_residual / np.maximum(cross_scale, prepared.fit_scale)
        else:
            fit_z = fit_residual / cross_scale
    else:
        fit_z = np.full(n, np.nan)

    residual = neighbour_residuals(values)
    if prepared is not None and prepared.n == n:
        neighbour_z = residual / prepared.resid_scale
        pca_z, pca_implied = _pca_z(values, prepared)
        history_z = (values - prepared.center) / prepared.spread if use_history_method else np.full(n, np.nan)
        history_fair = prepared.center if use_history_method else np.full(n, np.nan)
    else:  # no usable history: cross-sectional fallback for the neighbour method, others unavailable
        prepared = None
        finite = residual[np.isfinite(residual)]
        scale = float(robust_scale(finite, floor=params.min_scale)) if len(finite) >= 3 else np.nan
        neighbour_z = residual / scale
        pca_z = np.full(n, np.nan)
        pca_implied = np.full(n, np.nan)
        history_z = np.full(n, np.nan)
        history_fair = np.full(n, np.nan)

    seasonal = (
        seasonal_context(prepared, month_numbers, residual, today, params)
        if prepared is not None and month_numbers is not None and today is not None
        else [(None, 0)] * n
    )

    thresholds = params.thresholds
    columns = {"fit": fit_z, "neighbour": neighbour_z, "pca": pca_z, "history": history_z}
    results = []
    for i in range(n):
        z = {m: (float(columns[m][i]) if np.isfinite(columns[m][i]) else None) for m in METHODS}
        flags = {m: z[m] is not None and abs(z[m]) > thresholds[m] for m in METHODS}
        flagged = [m for m in METHODS if flags[m]]
        score = sum(min(abs(z[m]) / thresholds[m], _MAX_STRENGTH) for m in flagged)
        direction = ""
        if flagged:
            total = sum(z[m] for m in flagged)
            direction = "rich" if total > 0 else "cheap"
        level = 3 if len(flagged) >= params.min_methods_high else 2 if flagged else 0
        seasonal_z, seasonal_obs = seasonal[i]
        seasonal_normal = seasonal_z is not None and abs(seasonal_z) <= params.seasonal_z
        if seasonal_normal and level:
            level -= 1
        has_value = np.isfinite(values[i])
        results.append(
            PositionResult(
                index=i,
                value=float(values[i]) if has_value else None,
                fitted=float(fitted[i]) if np.isfinite(fitted[i]) else None,
                implied=float(values[i] - residual[i]) if np.isfinite(residual[i]) else None,
                residual=float(residual[i]) if np.isfinite(residual[i]) else None,
                z=z, flags=flags, n_flags=len(flagged), score=float(score), direction=direction,
                fair={
                    "fit": _finite(fitted[i]), "neighbour": _finite(values[i] - residual[i]),
                    "pca": _finite(pca_implied[i]), "history": _finite(history_fair[i]),
                },
                priority=_PRIORITY_BY_LEVEL[level], seasonal_z=seasonal_z,
                seasonal_normal=bool(flagged) and seasonal_normal, seasonal_obs=seasonal_obs,
            )
        )
    return results


MAX_KINKS = 6


def detect(
    values: np.ndarray,
    prepared: PreparedHistory | None,
    use_history_method: bool,
    params: CurveParams,
    month_numbers: list[int] | None = None,
    today: date | None = None,
) -> list[PositionResult]:
    """Find the kinks on one live curve, attributing each to the right contract.

    A kink contaminates the residuals of the points around it (the neighbour method predicts a
    point from its neighbours, so one bad neighbour shifts it; at the curve's ends the weights
    even amplify it), so a single bad contract makes several points look wrong. The strongest
    flagged point is therefore taken as a kink and masked out, every method is re-run on the
    rest, and the process repeats. A point that was flagged at first but is clean once the
    stronger kink is masked is only an ECHO of it (`echo_of`, priority NONE).
    """
    working = np.asarray(values, dtype=float).copy()
    first_pass = _detect_once(working, prepared, use_history_method, params, month_numbers, today)
    results = first_pass
    kinks: dict[int, PositionResult] = {}
    echoes: dict[int, int] = {}
    flagged = {r.index for r in results if r.n_flags}
    for _ in range(MAX_KINKS):
        candidates = [r for r in results if r.n_flags and r.index not in kinks]
        if not candidates:
            break
        top = max(candidates, key=lambda r: r.score)
        kinks[top.index] = top
        working[top.index] = np.nan
        results = _detect_once(working, prepared, use_history_method, params, month_numbers, today)
        still = {r.index for r in results if r.n_flags}
        for index in flagged - still - set(kinks):
            echoes.setdefault(index, top.index)
        flagged = still

    final = []
    for position in results:
        i = position.index
        if i in kinks:
            final.append(kinks[i])
        elif i in echoes:
            echo = first_pass[i]
            echo.echo_of, echo.priority = echoes[i], "NONE"
            final.append(echo)
        else:
            final.append(position)
    return final


def trade_side(direction: str) -> str:
    """The trade that fades a kink: a contract rich against the curve is SOLD (short), a cheap one BOUGHT (long)."""
    return {"rich": "SELL", "cheap": "BUY"}.get(direction, "")


def is_kink(result: PositionResult) -> bool:
    return result.n_flags > 0 and result.echo_of is None


def meets_priority(result: PositionResult, minimum: str) -> bool:
    return is_kink(result) and PRIORITY_RANK[result.priority] >= PRIORITY_RANK[minimum]
