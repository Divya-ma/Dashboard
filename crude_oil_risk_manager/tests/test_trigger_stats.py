"""Small-sample statistics of the trigger study: window verdicts, rank correlation, permutation test, leave-one-out, FDR and rules."""

import numpy as np
import pytest

from core.trigger_stats import (
    CONSTANT, NO_TREND, POSSIBLE, STRONG, TOO_FEW, TREND, benjamini_hochberg, finish_screens, fit_rule, leave_one_out_stable,
    permutation_p_value, rank_correlation, score_rule, screen, split_groups, window_stats,
)


def test_window_stats_calls_a_consistent_move_a_trend_and_a_coin_flip_no_trend():
    trend = window_stats([0.4, 0.5, 0.3, 0.6, 0.45, 0.2, -0.1])
    assert trend.verdict == TREND and trend.direction == 1 and (trend.up, trend.down) == (6, 1)
    assert trend.hit_rate == pytest.approx(6 / 7) and trend.sign_p < 0.2
    flat = window_stats([0.4, -0.5, 0.3, -0.6, 0.45, -0.2, 0.1, -0.1])
    assert flat.verdict == NO_TREND and flat.hit_rate == 0.5


def test_window_stats_edge_cases():
    assert window_stats([1.0]) is None and window_stats([]) is None
    assert window_stats([float("nan"), 1.0]) is None  # a missing expiry is not an expiry
    constant = window_stats([0.5] * 5)  # no spread at all: still a trend
    assert constant.verdict == TREND and constant.t_stat is None
    assert window_stats([-0.5] * 5).direction == -1
    assert window_stats([0.4, 0.5, 0.3], sign_test=False).sign_p is None


def test_the_usual_direction_is_the_median_so_one_huge_move_cannot_set_it():
    stats = window_stats([-0.1, -0.2, -0.15, -0.1, 50.0])
    assert stats.direction == -1 and stats.mean > 0 and stats.hit_rate == 0.8


def test_rank_correlation():
    assert rank_correlation([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert rank_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert rank_correlation([1, 1, 1, 1], [1, 2, 3, 4]) is None  # a constant trigger has no correlation
    assert rank_correlation([1, 2], [1, 2]) is None


def test_permutation_p_value_is_exact_for_few_expiries_and_deterministic_for_many():
    perfect = permutation_p_value(list(range(6)), list(range(6)))
    assert perfect == pytest.approx(2 / 720)  # the identity and the exact reverse ordering are as extreme
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=6), rng.normal(size=6)
    assert 0.0 < permutation_p_value(x, y) <= 1.0
    big = list(range(11))
    assert permutation_p_value(big, big) == permutation_p_value(big, big) < 0.01  # seeded
    assert permutation_p_value([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) is None


def test_leave_one_out_rejects_a_correlation_that_one_expiry_carries():
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 3, 4, 6, 5, 2]  # a mildly positive rank correlation that turns negative once a particular expiry is removed
    rho = rank_correlation(x, y)
    assert 0 < rho < 0.4 and not leave_one_out_stable(x, y, rho)
    assert leave_one_out_stable(x, [float(v) for v in x], 1.0)  # a clean relationship survives every removal
    assert not leave_one_out_stable(x, y, None) and not leave_one_out_stable(x, y, 0.0)


def test_benjamini_hochberg_matches_the_textbook_values_and_skips_missing():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5, None])
    assert q[0] == pytest.approx(0.04) and q[2] == pytest.approx(0.04 * 4 / 3) and q[1] == pytest.approx(0.04 * 4 / 3)
    assert q[3] == pytest.approx(0.5) and q[4] is None


def test_split_groups_means_of_the_lowest_and_highest_thirds():
    split = split_groups([1, 2, 3, 4, 5, 6], [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0])
    assert (split.low_mean, split.mid_mean, split.high_mean) == (-1.0, 0.0, 1.0) and split.flips
    assert not split_groups([1, 2, 3, 4, 5, 6], [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]).flips


def test_screening_one_clean_trigger_is_strong_and_a_constant_or_short_one_is_not():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [-0.5, -0.4, -0.45, -0.3, 0.3, 0.45, 0.4, 0.5]
    good, constant, short = screen(x, y), screen([1.0] * 8, y), screen([1, 2, 3], [1.0, 2.0, 3.0])
    finish_screens([good, constant, short])
    assert good.verdict == STRONG and good.rho > 0.9 and good.loo_stable and good.q_value == good.p_value
    assert constant.verdict == CONSTANT and short.verdict == TOO_FEW


def test_many_triggers_make_each_p_value_count_for_less():
    rng = np.random.default_rng(11)
    y = list(rng.normal(size=8))
    lucky = [screen(list(rng.normal(size=8)), y) for _ in range(40)]
    finish_screens(lucky)
    for item in lucky:
        if item.p_value is not None:
            assert item.q_value >= item.p_value  # the correction can only make a p-value larger
    assert sum(1 for item in lucky if item.verdict in (STRONG, POSSIBLE)) <= 3  # random triggers are not promoted


def test_a_rule_scored_leave_one_out_beats_the_usual_direction_when_the_trigger_is_real():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [-0.5, -0.4, -0.45, -0.3, 0.3, 0.45, 0.4, 0.5]
    fit = fit_rule(x, y)
    assert fit.cut == 4.5 and (fit.low_dir, fit.high_dir) == (-1, 1) and fit.direction(2) == -1 and fit.direction(7) == 1
    score = score_rule(x, y)
    assert score.hits == score.trades == 8 and score.mean_pnl > 0 and not score.thin
    assert score.baseline_hits < score.hits  # "always the usual direction" is right about half the time
    assert score_rule([1, 2, 3], [1.0, 2.0, 3.0]) is None


def test_the_baseline_falls_back_to_the_mean_when_the_median_is_zero():
    x = [1, 2, 3, 4, 5, 6]
    y = [0.0, 0.0, 0.0, 0.5, 0.4, 0.6]  # the median of the others is often 0: the usual direction is then the mean's
    score = score_rule(x, y)
    assert score.baseline_hits >= 3 and score.baseline_mean_pnl > 0
