"""Backtest of the kink signals on stored generic history.

For each of the last `test_days` days the detector is run exactly as it runs live, but fitted
only on the history BEFORE that day (no look-ahead). Every kink it flags (echoes excluded) is
then followed forward: the same CONTRACT is found again `h` days later (positions shift as
contracts roll, so it is matched by delivery month) and its neighbour residual - how far it sits
from the curve implied by its neighbours - is compared with the residual on the flag day.

  reverted   the residual shrank to less than half its flag-day size within h days
  reduction  how many points the residual shrank by (negative = it widened)

`reversion` is the median share of the flag-day residual that had closed after h days, over
every flagged kink (the trade plan uses it to place the target).

Reversion is partly mechanical: a large residual tends to shrink even by chance, so read the
figures across groups (1 vs 2 vs 3+ agreeing methods) rather than as absolute proof. Seasonality
is not applied here (it needs the delivery-month history of each past day); the backtest answers
"do consensus kinks close more often", which is what tunes the thresholds.
"""

from dataclasses import dataclass, field

import numpy as np

from core.curve_calendar import OUTRIGHT
from core.curve_history import HistoryMatrix
from core.curve_kinks import (
    METHOD_LABELS, METHODS, detect, fit_residuals, is_kink, neighbour_residuals, prepare_history,
)
from core.curve_settings import CurveParams

MIN_TRAIN_ROWS = 120
DEFAULT_TEST_DAYS = 250
DEFAULT_HORIZONS = (5, 10)


@dataclass
class BacktestResult:
    product: str
    family: str
    days_tested: int = 0
    rows: list[dict] = field(default_factory=list)  # one row per (group, horizon)
    reversion: dict[int, float | None] = field(default_factory=dict)  # horizon -> median share of the residual closed
    reversion_events: dict[int, int] = field(default_factory=dict)  # horizon -> kinks it is based on
    error: str | None = None


def _track(months: np.ndarray, t: int, k: int, later: int) -> int | None:
    """Position holding at `later` the same contract that sat at position k on day t (it can roll down one)."""
    for j in (k, k - 1):
        if 0 <= j < months.shape[1] and months[later, j] == months[t, k]:
            return j
    return None


def run_backtest(
    product: str, family: str, history: HistoryMatrix | None, params: CurveParams,
    test_days: int = DEFAULT_TEST_DAYS, horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> BacktestResult:
    result = BacktestResult(product=product, family=family)
    longest = max(horizons)
    if history is None or history.rows < MIN_TRAIN_ROWS + longest + 5:
        result.error = "Not enough history for a backtest yet"
        return result

    values, dates, months = history.values, history.dates, history.months
    residuals = neighbour_residuals(values)
    fit_all = fit_residuals(values, params.poly_degree)  # each day is fitted on its own: do it once
    first = max(MIN_TRAIN_ROWS, history.rows - longest - test_days)
    last = history.rows - longest  # exclusive: every flag needs `longest` days of follow-up
    groups: dict[str, dict[int, list[tuple[float, float]]]] = {}
    everything: dict[int, list[tuple[float, float]]] = {}

    def add(group: str, horizon: int, before: float, after: float) -> None:
        groups.setdefault(group, {}).setdefault(horizon, []).append((before, after))

    for t in range(first, last):
        prepared = prepare_history(values[:t], dates[:t], months[:t], params, fit_resid=fit_all[:t])
        if prepared is None:
            continue
        result.days_tested += 1
        for position in detect(values[t], prepared, family != OUTRIGHT, params):
            if not is_kink(position):
                continue
            k = position.index
            before = abs(residuals[t, k])
            if not np.isfinite(before):
                continue
            labels = [
                "1 method" if position.n_flags == 1 else "2 methods" if position.n_flags == 2 else "3+ methods"
            ] + [f"{METHOD_LABELS[m]} flagged" for m in METHODS if position.flags[m]]
            for horizon in horizons:
                j = _track(months, t, k, t + horizon)
                after = abs(residuals[t + horizon, j]) if j is not None else np.nan
                if not np.isfinite(after):
                    continue
                everything.setdefault(horizon, []).append((before, after))
                for label in labels:
                    add(label, horizon, before, after)

    for horizon in horizons:
        pairs = everything.get(horizon, [])
        result.reversion_events[horizon] = len(pairs)
        shares = [(b - a) / b for b, a in pairs if b > 0]
        result.reversion[horizon] = float(np.median(shares)) if shares else None

    order = ["1 method", "2 methods", "3+ methods"] + [f"{METHOD_LABELS[m]} flagged" for m in METHODS]
    for label in order:
        for horizon in horizons:
            pairs = groups.get(label, {}).get(horizon, [])
            if not pairs:
                continue
            before = np.array([p[0] for p in pairs])
            after = np.array([p[1] for p in pairs])
            result.rows.append(
                {
                    "group": label, "horizon": horizon, "events": len(pairs),
                    "reverted_pct": float(np.mean(after < 0.5 * before) * 100.0),
                    "narrower_pct": float(np.mean(after < before) * 100.0),
                    "avg_residual": float(before.mean()),
                    "avg_reduction": float((before - after).mean()),
                }
            )
    if not result.rows:
        result.error = "No kinks were flagged in the tested period at the current thresholds"
    return result
