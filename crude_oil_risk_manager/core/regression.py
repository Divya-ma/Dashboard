"""Regression, mean-reversion and sigma-move analytics (Trade Idea Analyzer).

Everything here works on daily closes loaded through DataLoader. Beta and
sigma moves use price DIFFERENCES; the Ornstein-Uhlenbeck fit uses the price
LEVEL series.

Conventions:
- "Days" are observations (trading rows in the local history), not calendar days.
- Functions raise InsufficientDataError rather than returning NaN/inf.
- Where a quantity is genuinely undefined (e.g. no half-life for a series that
  is not mean-reverting, no reversion rate for a direction with no events) the
  result holds None, never NaN. Consumers must handle None.
"""

import math
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from core.data_loader import DataLoader
from core.exceptions import InsufficientDataError

_MIN_BETA_OBS = 30
_MIN_OU_OBS = 60
_MIN_SIGMA_ROWS = 60
_MIN_SIGMA_EVENTS = 5
_FORWARD_DAYS = 30


def _ensure_finite(values: dict[str, float], context: str) -> None:
    bad = [name for name, value in values.items() if not math.isfinite(value)]
    if bad:
        raise InsufficientDataError(
            f"{context}: regression produced non-finite {', '.join(bad)} "
            f"(degenerate data, e.g. zero variance or a perfect fit)."
        )


# ----------------------------------------------------------------------
# Beta analysis
# ----------------------------------------------------------------------


def _load_aligned_diffs(
    symbol_a: str,
    symbol_b: str,
    data_loader: DataLoader,
    start: date | None,
    end: date | None,
) -> tuple[pd.Series, pd.Series]:
    diff_a = data_loader.load_price_differences(symbol_a, start, end, min_rows=_MIN_BETA_OBS)
    diff_b = data_loader.load_price_differences(symbol_b, start, end, min_rows=_MIN_BETA_OBS)
    aligned_a, aligned_b = data_loader.align_series(diff_a, diff_b)
    if len(aligned_a) < _MIN_BETA_OBS:
        raise InsufficientDataError(
            f"Insufficient common observations for beta of {symbol_a} on {symbol_b}: "
            f"{len(aligned_a)} available, minimum required is {_MIN_BETA_OBS}."
        )
    if np.std(aligned_b.to_numpy(dtype=float)) == 0:
        raise InsufficientDataError(
            f"Beta of {symbol_a} on {symbol_b} is undefined: {symbol_b} has zero variance."
        )
    return aligned_a, aligned_b


def calculate_beta(
    symbol_a: str,
    symbol_b: str,
    window: int,
    data_loader: DataLoader,
    start: date | None = None,
    end: date | None = None,
) -> dict:
    """OLS beta of price differences: dP_A = alpha + beta * dP_B + eps.

    Uses the most recent `window` common observations (fewer if fewer are
    available, but never fewer than 30; window itself must be >= 30).
    residual_std is the regression standard error (sqrt of SSR / df_resid).

    Raises ValueError if window < 30 and InsufficientDataError if fewer than
    30 common observations exist or the fit is degenerate (zero variance,
    perfect fit) — NaN/inf are never returned.
    """
    if window < _MIN_BETA_OBS:
        raise ValueError(f"window must be >= {_MIN_BETA_OBS} for a reliable OLS fit, got {window}")

    aligned_a, aligned_b = _load_aligned_diffs(symbol_a, symbol_b, data_loader, start, end)
    n = min(window, len(aligned_a))
    y = aligned_a.iloc[-n:]
    x = aligned_b.iloc[-n:]

    result = sm.OLS(y.to_numpy(dtype=float), sm.add_constant(x.to_numpy(dtype=float))).fit()
    alpha, beta = (float(p) for p in result.params)
    stats = {
        "alpha": alpha,
        "beta": beta,
        "r_squared": float(result.rsquared),
        "r_squared_adj": float(result.rsquared_adj),
        "beta_std_error": float(result.bse[1]),
        "beta_t_stat": float(result.tvalues[1]),
        "beta_p_value": float(result.pvalues[1]),
        "residual_std": float(math.sqrt(result.scale)),
    }
    _ensure_finite(stats, f"Beta of {symbol_a} on {symbol_b}")

    interpretation = (
        f"A 1-point move in {symbol_b} is associated with a {beta:.3f}-point move in "
        f"{symbol_a} (R²={stats['r_squared']:.3f}, p={stats['beta_p_value']:.4f})"
    )
    return {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "beta": beta,
        "alpha": alpha,
        "r_squared": stats["r_squared"],
        "r_squared_adj": stats["r_squared_adj"],
        "beta_std_error": stats["beta_std_error"],
        "beta_t_stat": stats["beta_t_stat"],
        "beta_p_value": stats["beta_p_value"],
        "residual_std": stats["residual_std"],
        "observations": int(result.nobs),
        "window_used": n,
        "start_date": y.index[0].date(),
        "end_date": y.index[-1].date(),
        "interpretation": interpretation,
    }


def calculate_rolling_beta(
    symbol_a: str,
    symbol_b: str,
    window: int,
    data_loader: DataLoader,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Rolling OLS beta of price differences (statsmodels RollingOLS).

    Returns a DataFrame indexed by datetime with columns
    [date, beta, alpha, r_squared, residual_std]. The first window-1 rows and
    any undefined windows are dropped, so it contains no NaN. Raises
    ValueError if window < 30 and InsufficientDataError if there are fewer
    than `window` common observations.
    """
    if window < _MIN_BETA_OBS:
        raise ValueError(f"window must be >= {_MIN_BETA_OBS} for a reliable OLS fit, got {window}")

    aligned_a, aligned_b = _load_aligned_diffs(symbol_a, symbol_b, data_loader, start, end)
    if len(aligned_a) < window:
        raise InsufficientDataError(
            f"Insufficient data for rolling beta of {symbol_a} on {symbol_b}: "
            f"{len(aligned_a)} common observations, window is {window}."
        )

    exog = sm.add_constant(aligned_b.rename("x"))
    fitted = RollingOLS(aligned_a.rename("y"), exog, window=window).fit()
    out = pd.DataFrame(
        {
            "beta": fitted.params["x"],
            "alpha": fitted.params["const"],
            "r_squared": fitted.rsquared,
            "residual_std": np.sqrt(fitted.mse_resid),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        raise InsufficientDataError(
            f"Rolling beta of {symbol_a} on {symbol_b} produced no valid windows "
            f"(window={window}, observations={len(aligned_a)})."
        )
    out.insert(0, "date", [ts.date() for ts in out.index])
    return out


# ----------------------------------------------------------------------
# Ornstein-Uhlenbeck mean reversion
# ----------------------------------------------------------------------


def fit_ou_process(
    symbol: str,
    data_loader: DataLoader,
    start: date | None = None,
    end: date | None = None,
) -> dict:
    """Fit an Ornstein-Uhlenbeck process to the price LEVEL series.

    Model: dX = kappa (mu - X) dt + sigma dW, estimated with dt = 1 day via
    the discretised OLS  X_t - X_{t-1} = a + b X_{t-1} + eps  where
    kappa = -b, mu = -a/b, sigma = residual std, half_life = ln(2) / kappa.

    If kappa <= 0 the series is not mean-reverting: is_mean_reverting is False,
    data_warning is set, and mu / half_life_days are None (a long-run mean and
    half-life are not defined). Raises InsufficientDataError for fewer than 60
    regression observations or a degenerate fit.
    """
    closes = data_loader.load_close_series(symbol, start, end, min_rows=_MIN_OU_OBS + 1)
    levels = closes.to_numpy(dtype=float)
    lagged = levels[:-1]
    changes = np.diff(levels)
    if np.std(lagged) == 0:
        raise InsufficientDataError(f"OU fit for {symbol} is undefined: the price series is constant.")

    result = sm.OLS(changes, sm.add_constant(lagged)).fit()
    a, b = (float(p) for p in result.params)
    kappa = -b
    sigma = float(math.sqrt(result.scale))
    r_squared = float(result.rsquared)
    _ensure_finite({"kappa": kappa, "a": a, "sigma": sigma, "r_squared": r_squared}, f"OU fit for {symbol}")

    is_mean_reverting = kappa > 0
    mu: float | None = None
    half_life: float | None = None
    data_warning: str | None = None
    if is_mean_reverting:
        mu = -a / b
        half_life = math.log(2) / kappa
        _ensure_finite({"mu": mu, "half_life_days": half_life}, f"OU fit for {symbol}")
        interpretation = (
            f"Price is expected to revert halfway to its mean ({mu:.2f}) in "
            f"{half_life:.1f} days. Long-run mean: {mu:.2f}."
        )
    else:
        data_warning = (
            f"Series does not appear mean-reverting over this period (κ={kappa:.4f}). "
            f"Half-life estimate is unreliable."
        )
        interpretation = (
            "No mean reversion detected over this period; no half-life or long-run mean is estimated."
        )

    return {
        "symbol": symbol,
        "kappa": kappa,
        "mu": mu,
        "sigma": sigma,
        "half_life_days": half_life,
        "is_mean_reverting": is_mean_reverting,
        "r_squared": r_squared,
        "observations": int(result.nobs),
        "data_warning": data_warning,
        "interpretation": interpretation,
    }


def _target_result(
    current_price: float | None,
    target_price: float | None,
    mu: float | None,
    expected_days: float | None,
    is_valid: bool,
    note: str,
) -> dict:
    return {
        "current_price": current_price,
        "target_price": target_price,
        "mu": mu,
        "expected_days": expected_days,
        "is_valid": is_valid,
        "validity_note": note,
    }


def estimate_time_to_target(ou_result: dict, current_price: float, target_price: float) -> dict:
    """Expected days for price to travel from current_price to target_price.

    E[T] ~= (1/kappa) * ln((current - mu) / (target - mu)). This is the time
    for the deterministic OU mean path to reach the target; it ignores the
    noise term, so actual times will vary widely around it.

    Only valid when the series is mean-reverting (kappa > 0) and the target
    lies between current_price and mu (on the same side of mu, no further
    from it than current_price, and not equal to mu). Never raises for an
    invalid target: it returns is_valid=False with expected_days=None and an
    explanatory validity_note.
    """
    mu = ou_result.get("mu")
    kappa = ou_result.get("kappa")

    if not (math.isfinite(current_price) and math.isfinite(target_price)):
        return _target_result(None, None, mu, None, False, "Price inputs must be finite numbers.")
    if not ou_result.get("is_mean_reverting") or mu is None or kappa is None or kappa <= 0:
        return _target_result(
            current_price, target_price, mu, None, False,
            "Series is not mean-reverting, so no time to target can be estimated.",
        )

    current_gap = current_price - mu
    target_gap = target_price - mu
    if current_gap * target_gap <= 0 or abs(target_gap) > abs(current_gap):
        return _target_result(
            current_price, target_price, mu, None, False,
            "Target is not in the direction of mean reversion. The price would need to "
            "move away from its mean to reach this target.",
        )

    expected_days = math.log(current_gap / target_gap) / kappa
    return _target_result(
        current_price, target_price, mu, expected_days, True,
        "Estimate follows the deterministic OU mean path (ignores noise); actual time will vary.",
    )


# ----------------------------------------------------------------------
# Sigma-move analysis
# ----------------------------------------------------------------------


def find_sigma_moves(
    symbol: str,
    sigma_threshold: float,
    data_loader: DataLoader,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Scan daily price differences for moves beyond `sigma_threshold` standard deviations.

    Assumption: the full-sample mean and standard deviation of price
    differences are used (no rolling estimate), so event_sigma is
    |diff - mean| / std over the whole loaded range.

    Each event is treated as a FADE trade entered at the event-day close:
    - reversion: the first later day (within 30 observations) whose close is
      back through event_price against the event direction (below it after an
      "up" event, above it after a "down" event); days_to_reversion counts
      observations after the event (NaN if it did not revert).
    - max_extension: largest move further in the event direction over the
      whole 30-observation path (positive, price units).
    - max_adverse_move: largest such move BEFORE the reversion day (the whole
      path if it never reverted) — the fade trade's maximum adverse excursion.
    Events with fewer than 30 subsequent observations are excluded because
    their outcome is not yet known (including them would bias the reversion
    rate down).

    Returns columns [event_date, event_price, event_diff, event_sigma,
    direction, days_to_reversion, reverted_in_30d, max_extension,
    max_adverse_move]; the threshold is stored in df.attrs["sigma_threshold"].
    Raises InsufficientDataError if fewer than 5 events are found.
    """
    if sigma_threshold <= 0:
        raise ValueError(f"sigma_threshold must be > 0, got {sigma_threshold}")

    closes = data_loader.load_close_series(symbol, start, end, min_rows=_MIN_SIGMA_ROWS)
    prices = closes.to_numpy(dtype=float)
    diffs = np.diff(prices)
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    if not math.isfinite(std) or std == 0:
        raise InsufficientDataError(
            f"Sigma moves for {symbol} are undefined: price differences have zero variance."
        )

    z_scores = (diffs - mean) / std
    last_complete_position = len(prices) - 1 - _FORWARD_DAYS
    rows = []
    for j in np.flatnonzero(np.abs(z_scores) > sigma_threshold):
        position = int(j) + 1
        if position > last_complete_position:
            continue

        event_price = float(prices[position])
        is_up = z_scores[j] > 0
        future = prices[position + 1 : position + 1 + _FORWARD_DAYS]
        reverted_mask = future < event_price if is_up else future > event_price
        reverted_at = np.flatnonzero(reverted_mask)

        if reverted_at.size:
            reversion_index = int(reverted_at[0])
            days_to_reversion = float(reversion_index + 1)
            pre_reversion = future[:reversion_index]
        else:
            days_to_reversion = float("nan")
            pre_reversion = future

        if is_up:
            extension = max(0.0, float(future.max()) - event_price)
            adverse = max(0.0, float(pre_reversion.max()) - event_price) if pre_reversion.size else 0.0
        else:
            extension = max(0.0, event_price - float(future.min()))
            adverse = max(0.0, event_price - float(pre_reversion.min())) if pre_reversion.size else 0.0

        rows.append(
            {
                "event_date": closes.index[position],
                "event_price": event_price,
                "event_diff": float(diffs[j]),
                "event_sigma": float(abs(z_scores[j])),
                "direction": "up" if is_up else "down",
                "days_to_reversion": days_to_reversion,
                "reverted_in_30d": bool(reverted_at.size),
                "max_extension": extension,
                "max_adverse_move": adverse,
            }
        )

    if len(rows) < _MIN_SIGMA_EVENTS:
        raise InsufficientDataError(
            f"Only {len(rows)} sigma events found for {symbol} at {sigma_threshold}σ threshold. "
            f"Consider lowering the threshold or extending the date range."
        )

    events = pd.DataFrame(
        rows,
        columns=[
            "event_date", "event_price", "event_diff", "event_sigma", "direction",
            "days_to_reversion", "reverted_in_30d", "max_extension", "max_adverse_move",
        ],
    )
    events.attrs["sigma_threshold"] = sigma_threshold
    return events


def analyze_sigma_moves(
    sigma_events: pd.DataFrame,
    stop_loss_ticks: float,
    tick_size: float,
    sigma_threshold: float | None = None,
) -> dict:
    """Trading statistics for the fade trades in a find_sigma_moves DataFrame.

    stop_loss_price = stop_loss_ticks * tick_size (price distance). A stop is
    triggered when an event's max_adverse_move exceeds it. Rates are
    fractions in [0, 1]. avg/median reversion days are None if no event
    reverted; a direction's reversion rate is None if it had no events.

    sigma_threshold is reported in the result; if not passed it is taken from
    sigma_events.attrs, and failing that the smallest observed event sigma.
    Raises ValueError for non-positive stop/tick inputs and
    InsufficientDataError for an empty events frame.
    """
    if stop_loss_ticks <= 0 or tick_size <= 0:
        raise ValueError("stop_loss_ticks and tick_size must be > 0")
    if sigma_events.empty:
        raise InsufficientDataError("No sigma events to analyze.")

    if sigma_threshold is None:
        sigma_threshold = sigma_events.attrs.get("sigma_threshold")
    if sigma_threshold is None:
        sigma_threshold = float(sigma_events["event_sigma"].min())

    total = len(sigma_events)
    reverted = sigma_events["reverted_in_30d"].to_numpy(dtype=bool)
    days = sigma_events["days_to_reversion"].to_numpy(dtype=float)[reverted]
    adverse = sigma_events["max_adverse_move"].to_numpy(dtype=float)
    is_up = (sigma_events["direction"] == "up").to_numpy()

    hit_rate = float(reverted.mean())
    avg_days = float(np.mean(days)) if days.size else None
    median_days = float(np.median(days)) if days.size else None

    stop_price = stop_loss_ticks * tick_size
    trigger_probability = float(np.mean(adverse > stop_price))
    survival_probability = 1.0 - trigger_probability

    if avg_days is None:
        reversion_text = f"{hit_rate:.0%} of moves reverted within {_FORWARD_DAYS} days"
    else:
        reversion_text = (
            f"{hit_rate:.0%} of moves reverted within {_FORWARD_DAYS} days (avg {avg_days:.1f} days)"
        )
    interpretation = (
        f"At {sigma_threshold:g}σ, {reversion_text}. Stop of {stop_loss_ticks:g} ticks has "
        f"{survival_probability:.0%} survival probability."
    )

    return {
        "total_events": total,
        "reversion_hit_rate": hit_rate,
        "avg_reversion_days": avg_days,
        "median_reversion_days": median_days,
        "avg_max_adverse_move": float(np.mean(adverse)),
        "stop_loss_trigger_probability": trigger_probability,
        "stop_loss_survival_probability": survival_probability,
        "avg_extension_after_event": float(sigma_events["max_extension"].mean()),
        "up_events": int(is_up.sum()),
        "down_events": int((~is_up).sum()),
        "up_reversion_rate": float(reverted[is_up].mean()) if is_up.any() else None,
        "down_reversion_rate": float(reverted[~is_up].mean()) if (~is_up).any() else None,
        "stop_loss_ticks": stop_loss_ticks,
        "stop_loss_price": stop_price,
        "sigma_threshold": float(sigma_threshold),
        "interpretation": interpretation,
    }


# ----------------------------------------------------------------------
# Combined trade idea analyzer
# ----------------------------------------------------------------------


def run_trade_idea_analysis(
    symbol: str,
    current_price: float,
    sigma_threshold: float,
    stop_loss_ticks: float,
    target_ticks: float,
    tick_size: float,
    tick_value: float,
    data_loader: DataLoader,
    beta_benchmark: str | None = None,
    window: int = 60,
    start: date | None = None,
    end: date | None = None,
) -> dict:
    """Run every Trade Idea Analyzer component and return one unified dict.

    Components run independently: a failure in one is reported as a
    "<component>_error" string (ExceptionType: message) while the others
    still run. This function never raises.

    The target price is current_price moved target_ticks * tick_size toward
    the OU long-run mean (below current if current > mu, else above). If the
    OU fit failed, time_to_target is None. If the series is not
    mean-reverting or the target inputs are invalid, time_to_target is an
    is_valid=False result. beta_analysis and beta_error are both None when no
    beta_benchmark is given. tick_value is echoed for the UI's dollar maths.
    """
    ou_analysis: dict | None = None
    ou_error: str | None = None
    try:
        ou_analysis = fit_ou_process(symbol, data_loader, start, end)
    except Exception as exc:  # noqa: BLE001 - components must never raise
        ou_error = f"{type(exc).__name__}: {exc}"

    time_to_target: dict | None = None
    if ou_analysis is not None:
        if not (target_ticks > 0 and tick_size > 0 and math.isfinite(current_price)):
            time_to_target = _target_result(
                None, None, ou_analysis["mu"], None, False,
                "target_ticks and tick_size must be positive and current_price finite.",
            )
        else:
            offset = target_ticks * tick_size
            mu = ou_analysis["mu"]
            target_price = current_price - offset if (mu is not None and current_price > mu) else current_price + offset
            time_to_target = estimate_time_to_target(ou_analysis, current_price, target_price)

    sigma_analysis: dict | None = None
    sigma_error: str | None = None
    try:
        events = find_sigma_moves(symbol, sigma_threshold, data_loader, start, end)
        sigma_analysis = analyze_sigma_moves(events, stop_loss_ticks, tick_size, sigma_threshold)
    except Exception as exc:  # noqa: BLE001
        sigma_error = f"{type(exc).__name__}: {exc}"

    beta_analysis: dict | None = None
    beta_error: str | None = None
    if beta_benchmark is not None:
        try:
            beta_analysis = calculate_beta(symbol, beta_benchmark, window, data_loader, start, end)
        except Exception as exc:  # noqa: BLE001
            beta_error = f"{type(exc).__name__}: {exc}"

    return {
        "symbol": symbol,
        "current_price": current_price,
        "ou_analysis": ou_analysis,
        "ou_error": ou_error,
        "time_to_target": time_to_target,
        "sigma_analysis": sigma_analysis,
        "sigma_error": sigma_error,
        "beta_analysis": beta_analysis,
        "beta_error": beta_error,
        "stop_loss_ticks": stop_loss_ticks,
        "target_ticks": target_ticks,
        "tick_size": tick_size,
        "tick_value": tick_value,
        "analyzed_at": datetime.now(timezone.utc),
    }
