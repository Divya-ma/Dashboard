"""Small-sample statistics for the Seasonality trigger study. Pure functions, no I/O.

A seasonal window gives ONE number per delivery year (the move between two days before expiry), so a study has at most about
8-12 observations. Everything here is chosen for that size:

* the years, not the days, are the observations (days inside one year are not independent);
* a trigger is judged by its RANK correlation with the move and by a PERMUTATION test (exact when there are 8 years or fewer),
  not by a normal-theory p-value;
* a trigger only counts when it keeps its sign when any single year is left out;
* p-values are corrected for the number of triggers tried (Benjamini-Hochberg), because the more triggers are screened the more
  look good by chance;
* a trading rule built on a trigger is scored leave-one-year-out: each year is judged by a rule learned from the OTHER years.
"""

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats as scipy_stats

MIN_STUDY_YEARS = 5  # years needed to screen a trigger (and to call a window anything but "too few years")
PERM_EXACT_MAX_N = 8  # up to this many years every ordering is enumerated (8! = 40,320)
PERM_DRAWS = 5000
PERM_SEED = 20261102

HIT_TREND, T_TREND = 0.70, 2.0
HIT_WEAK, T_WEAK = 0.60, 1.5
P_STRONG, P_POSSIBLE, Q_STRONG = 0.05, 0.10, 0.25

TREND, WEAK, NO_TREND = "trend", "weak", "no trend"
STRONG, POSSIBLE, NO_EVIDENCE, CONSTANT, TOO_FEW = "strong", "possible", "no evidence", "constant", "too few years"


# ----------------------------------------------------------------------
# The window as a whole
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class WindowStats:
    n: int
    mean: float
    median: float
    sd: float | None
    t_stat: float | None
    direction: int  # +1 the usual move is up, -1 down, 0 none (median and mean both zero)
    up: int
    down: int
    hit_rate: float  # share of years that moved in the usual direction
    sign_p: float | None  # two-sided binomial test of that share against 50% (None when it was not asked for)
    verdict: str  # TREND / WEAK / NO_TREND


def window_stats(moves, sign_test: bool = True) -> WindowStats | None:
    """Mean, median, hit rate and a verdict for one window's moves (one per year); None for fewer than two years.

    TREND: at least 70% of the years moved the usual way AND |t| >= 2 (a constant move counts as significant). WEAK: 60% of
    the years or |t| >= 1.5. Otherwise NO_TREND. The usual direction is the sign of the MEDIAN, so one extreme year cannot set it.
    `sign_test=False` skips the binomial test (a window scan calls this thousands of times and does not show it).
    """
    values = np.array([m for m in moves if m is not None and np.isfinite(m)], dtype=float)
    n = len(values)
    if n < 2:
        return None
    mean, median = float(values.mean()), float(np.median(values))
    sd = float(values.std(ddof=1))
    t_stat = None if sd == 0 else mean / (sd / math.sqrt(n))
    direction = int(np.sign(median)) or int(np.sign(mean))
    up, down = int((values > 0).sum()), int((values < 0).sum())
    hits = up if direction > 0 else down if direction < 0 else 0
    hit_rate = hits / n
    sign_p = (float(scipy_stats.binomtest(hits, n, 0.5).pvalue) if direction else 1.0) if sign_test else None
    t_big = (t_stat is None and mean != 0) or (t_stat is not None and abs(t_stat) >= T_TREND)
    t_some = (t_stat is None and mean != 0) or (t_stat is not None and abs(t_stat) >= T_WEAK)
    if direction and hit_rate >= HIT_TREND and t_big:
        verdict = TREND
    elif direction and (hit_rate >= HIT_WEAK or t_some):
        verdict = WEAK
    else:
        verdict = NO_TREND
    return WindowStats(n, mean, median, sd, t_stat, direction, up, down, hit_rate, sign_p, verdict)


# ----------------------------------------------------------------------
# One trigger against the moves
# ----------------------------------------------------------------------


def _ranks(values) -> np.ndarray:
    return scipy_stats.rankdata(np.asarray(values, dtype=float), method="average")


def rank_correlation(x, y) -> float | None:
    """Spearman correlation of two equally long lists; None for fewer than 3 points or when either is constant."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = _ranks(x), _ranks(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


@lru_cache(maxsize=8)
def _all_orderings(n: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(n))), dtype=np.int16)


def permutation_p_value(x, y) -> float | None:
    """Two-sided p-value of the rank correlation: how often does a random pairing of the years correlate at least as strongly?

    Exact (every ordering) for up to 8 years, otherwise 5,000 seeded random orderings with the usual +1 correction."""
    observed = rank_correlation(x, y)
    if observed is None:
        return None
    n = len(x)
    rx, ry = _ranks(x), _ranks(y)
    if n <= PERM_EXACT_MAX_N:
        orderings, exact = _all_orderings(n), True
    else:
        rng = np.random.default_rng(PERM_SEED)
        orderings, exact = np.array([rng.permutation(n) for _ in range(PERM_DRAWS)]), False
    shuffled = ry[orderings]
    sx = rx - rx.mean()
    sy = shuffled - shuffled.mean(axis=1, keepdims=True)
    rho = (sy @ sx) / np.sqrt((sx**2).sum() * (sy**2).sum(axis=1))
    extreme = int(np.sum(np.abs(rho) >= abs(observed) - 1e-12))
    return extreme / len(orderings) if exact else (extreme + 1) / (len(orderings) + 1)


def leave_one_out_stable(x, y, rho: float | None) -> bool:
    """True when the correlation keeps its sign after any single year is removed."""
    if rho is None or rho == 0:
        return False
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    for i in range(len(x)):
        kept = np.arange(len(x)) != i
        again = rank_correlation(x[kept], y[kept])
        if again is None or again * rho <= 0:
            return False
    return True


@dataclass(frozen=True)
class GroupSplit:
    low_mean: float | None
    mid_mean: float | None
    high_mean: float | None
    low_n: int
    mid_n: int
    high_n: int

    @property
    def flips(self) -> bool:
        """The lowest and highest thirds moved in opposite directions: the trigger separates two regimes."""
        return self.low_mean is not None and self.high_mean is not None and self.low_mean * self.high_mean < 0


def split_groups(x, y) -> GroupSplit:
    """Mean move of the years with the lowest third, the middle and the highest third of the trigger values."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return GroupSplit(None, None, None, 0, 0, 0)
    order = np.argsort(x, kind="stable")
    k = max(1, n // 3)
    low, high, mid = order[:k], order[n - k:], order[k:n - k]
    mean = lambda idx: float(y[idx].mean()) if len(idx) else None  # noqa: E731
    return GroupSplit(mean(low), mean(mid), mean(high), len(low), len(mid), len(high))


def benjamini_hochberg(p_values) -> list[float | None]:
    """False-discovery-rate adjusted p-values (q); None entries are left out of the count."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    q: list[float | None] = [None] * len(p_values)
    m = len(indexed)
    running = 1.0
    for rank, (i, p) in sorted(enumerate(sorted(indexed, key=lambda item: item[1])), key=lambda pair: -pair[0]):
        running = min(running, p * m / (rank + 1))
        q[i] = running
    return q


@dataclass
class Screen:
    n: int
    rho: float | None
    p_value: float | None
    q_value: float | None
    loo_stable: bool
    split: GroupSplit
    verdict: str


def screen(x, y) -> Screen:
    """One trigger's evidence against the moves (x and y hold only the years that have both). q and the verdict are
    finalised by `finish_screens`, which needs every trigger's p-value."""
    n = len(x)
    if n < MIN_STUDY_YEARS:
        return Screen(n, None, None, None, False, split_groups(x, y) if n >= 3 else GroupSplit(None, None, None, 0, 0, 0), TOO_FEW)
    rho = rank_correlation(x, y)
    if rho is None:
        return Screen(n, None, None, None, False, split_groups(x, y), CONSTANT)
    return Screen(n, rho, permutation_p_value(x, y), None, leave_one_out_stable(x, y, rho), split_groups(x, y), NO_EVIDENCE)


def finish_screens(screens: list[Screen]) -> None:
    """Adds the false-discovery q to every screen and sets each verdict (strong / possible / no evidence)."""
    for item, q in zip(screens, benjamini_hochberg([s.p_value for s in screens])):
        item.q_value = q
        if item.verdict not in (NO_EVIDENCE,):
            continue
        if item.p_value <= P_STRONG and q <= Q_STRONG and item.loo_stable:
            item.verdict = STRONG
        elif item.p_value <= P_POSSIBLE and item.loo_stable:
            item.verdict = POSSIBLE


# ----------------------------------------------------------------------
# A rule built on a trigger, tested leave-one-year-out
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RuleFit:
    """The rule learned from a set of years: above `cut` expect `high_dir`, at or below it expect `low_dir` (+1 up, -1 down, 0 none)."""

    cut: float
    high_dir: int
    low_dir: int

    def direction(self, value: float) -> int:
        return self.high_dir if value > self.cut else self.low_dir


def fit_rule(x, y) -> RuleFit | None:
    """Cut at the median trigger value; each side expects the sign of its own mean move (0 when a side has fewer than 2 years)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 4:
        return None
    cut = float(np.median(x))
    high, low = y[x > cut], y[x <= cut]
    side = lambda values: int(np.sign(values.mean())) if len(values) >= 2 else 0  # noqa: E731
    return RuleFit(cut, side(high), side(low))


@dataclass(frozen=True)
class RuleScore:
    years: int
    trades: int  # years in which the rule had an opinion
    hits: int
    mean_pnl: float | None  # average signed move over the trades
    baseline_hits: int  # "always the usual direction", learned from the other years
    baseline_mean_pnl: float | None
    thin: bool  # too few trades to trust


def score_rule(x, y) -> RuleScore | None:
    """Leave-one-year-out score: each year is predicted by a rule fitted to the OTHER years, so no year judges itself.

    Compared with the baseline of always taking the usual direction (also learned from the other years)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < MIN_STUDY_YEARS:
        return None
    pnl, base = [], []
    for i in range(n):
        kept = np.arange(n) != i
        fit = fit_rule(x[kept], y[kept])
        usual = int(np.sign(np.median(y[kept]))) or int(np.sign(np.mean(y[kept])))  # the same fallback as window_stats
        base.append(usual * y[i])
        if fit is not None:
            call = fit.direction(x[i])
            if call:
                pnl.append(call * y[i])
    trades = len(pnl)
    return RuleScore(
        n, trades, int(sum(1 for p in pnl if p > 0)), float(np.mean(pnl)) if pnl else None,
        int(sum(1 for b in base if b > 0)), float(np.mean(base)) if base else None,
        trades < max(4, math.ceil(0.6 * n)),
    )
