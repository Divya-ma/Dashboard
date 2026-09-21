"""Tests for core.regression."""

import math

import numpy as np
import pandas as pd
import pytest

from adapters.mock.mock_historical import MockHistoricalAdapter
from core.data_loader import DataLoader
from core.exceptions import InsufficientDataError
from core.regression import (
    _ensure_finite,
    analyze_sigma_moves,
    calculate_beta,
    calculate_rolling_beta,
    estimate_time_to_target,
    find_sigma_moves,
    fit_ou_process,
    run_trade_idea_analysis,
)

END_DATE = pd.Timestamp("2026-06-30", tz="UTC")


@pytest.fixture
def loader(tmp_path):
    return DataLoader(str(tmp_path), MockHistoricalAdapter())


def create_ou_series(create_synthetic_parquet, tmp_path, symbol, kappa, mu, sigma, n_days, seed):
    """Write a synthetic OU path (Euler-Maruyama, dt = 1) and return its closes."""
    rng = np.random.default_rng(seed)
    closes = np.empty(n_days)
    closes[0] = mu
    for t in range(1, n_days):
        closes[t] = closes[t - 1] + kappa * (mu - closes[t - 1]) + sigma * rng.normal()
    create_synthetic_parquet(tmp_path, symbol, closes=closes)
    return closes


def create_related_pair(create_synthetic_parquet, tmp_path, symbol_x, symbol_y, slope, n_days, seed):
    """dY = slope * dX + noise, so beta of Y on X is ~slope."""
    rng = np.random.default_rng(seed)
    dx = rng.normal(0.0, 1.0, n_days)
    dy = slope * dx + rng.normal(0.0, 0.3, n_days)
    create_synthetic_parquet(tmp_path, symbol_x, closes=75.0 + np.cumsum(dx))
    create_synthetic_parquet(tmp_path, symbol_y, closes=75.0 + np.cumsum(dy))


def spike_closes(n=300, positions=(50, 100, 150, 200, 250), seed=5):
    """Quiet series with five +5 spikes that follow a known path: 80.5, 81, 79.5, then back to ~75."""
    rng = np.random.default_rng(seed)
    closes = 75.0 + rng.normal(0.0, 0.05, n)
    for p in positions:
        closes[p : p + 4] = [80.0, 80.5, 81.0, 79.5]
    return closes


# ---------- calculate_beta ----------


def test_beta_recovers_known_slope(tmp_path, loader, create_synthetic_parquet):
    create_related_pair(create_synthetic_parquet, tmp_path, "CLZ26", "CLF27", 2.0, 300, seed=1)
    result = calculate_beta("CLF27", "CLZ26", 100, loader)
    assert result["beta"] == pytest.approx(2.0, abs=0.15)
    assert result["symbol_a"] == "CLF27"
    assert result["symbol_b"] == "CLZ26"
    assert result["window_used"] == 100
    assert result["observations"] == 100
    assert result["start_date"] < result["end_date"]


def test_beta_r_squared_between_zero_and_one(tmp_path, loader, create_synthetic_parquet):
    create_related_pair(create_synthetic_parquet, tmp_path, "CLZ26", "CLF27", 2.0, 300, seed=1)
    result = calculate_beta("CLF27", "CLZ26", 100, loader)
    assert 0.0 <= result["r_squared"] <= 1.0
    assert 0.0 <= result["r_squared_adj"] <= 1.0
    assert result["r_squared"] > 0.9


def test_beta_all_values_finite(tmp_path, loader, create_synthetic_parquet):
    create_related_pair(create_synthetic_parquet, tmp_path, "CLZ26", "CLF27", 2.0, 300, seed=1)
    result = calculate_beta("CLF27", "CLZ26", 100, loader)
    numeric = [v for v in result.values() if isinstance(v, float)]
    assert numeric and all(math.isfinite(v) for v in numeric)


def test_beta_fewer_than_30_observations_raises(tmp_path, loader, create_synthetic_parquet):
    create_related_pair(create_synthetic_parquet, tmp_path, "CLZ26", "CLF27", 2.0, 25, seed=1)
    with pytest.raises(InsufficientDataError):
        calculate_beta("CLF27", "CLZ26", 60, loader)


def test_beta_window_below_minimum_raises_value_error(tmp_path, loader, create_synthetic_parquet):
    create_related_pair(create_synthetic_parquet, tmp_path, "CLZ26", "CLF27", 2.0, 300, seed=1)
    with pytest.raises(ValueError):
        calculate_beta("CLF27", "CLZ26", 10, loader)


def test_beta_uses_available_history_when_shorter_than_window(tmp_path, loader, create_synthetic_parquet):
    create_related_pair(create_synthetic_parquet, tmp_path, "CLZ26", "CLF27", 2.0, 51, seed=1)
    result = calculate_beta("CLF27", "CLZ26", 200, loader)
    assert result["window_used"] == 50


def test_beta_interpretation_is_non_empty(tmp_path, loader, create_synthetic_parquet):
    create_related_pair(create_synthetic_parquet, tmp_path, "CLZ26", "CLF27", 2.0, 300, seed=1)
    result = calculate_beta("CLF27", "CLZ26", 100, loader)
    assert result["interpretation"]
    assert "CLZ26" in result["interpretation"] and "CLF27" in result["interpretation"]


def test_beta_near_perfect_fit_stays_finite(tmp_path, loader, create_synthetic_parquet):
    base = 75.0 + np.cumsum(np.random.default_rng(3).normal(size=100))
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    create_synthetic_parquet(tmp_path, "CLF27", closes=2 * base)
    result = calculate_beta("CLF27", "CLZ26", 60, loader)
    assert result["beta"] == pytest.approx(2.0)
    assert math.isfinite(result["beta_t_stat"])


def test_ensure_finite_raises_on_nan_or_inf():
    with pytest.raises(InsufficientDataError):
        _ensure_finite({"beta": 1.0, "t": float("inf")}, "test")
    with pytest.raises(InsufficientDataError):
        _ensure_finite({"beta": float("nan")}, "test")
    _ensure_finite({"beta": 1.0}, "test")


# ---------- calculate_rolling_beta ----------


def test_rolling_beta_columns_and_length(tmp_path, loader, create_synthetic_parquet):
    create_related_pair(create_synthetic_parquet, tmp_path, "CLZ26", "CLF27", 2.0, 150, seed=2)
    result = calculate_rolling_beta("CLF27", "CLZ26", 60, loader)
    assert list(result.columns) == ["date", "beta", "alpha", "r_squared", "residual_std"]
    assert isinstance(result.index, pd.DatetimeIndex)
    # 150 closes -> 149 diffs -> 149 - 60 + 1 windows
    assert len(result) == 90


def test_rolling_beta_has_no_nan(tmp_path, loader, create_synthetic_parquet):
    create_related_pair(create_synthetic_parquet, tmp_path, "CLZ26", "CLF27", 2.0, 150, seed=2)
    result = calculate_rolling_beta("CLF27", "CLZ26", 60, loader)
    assert not result[["beta", "alpha", "r_squared", "residual_std"]].isna().any().any()
    assert (result["beta"].between(1.5, 2.5)).all()


def test_rolling_beta_window_larger_than_data_raises(tmp_path, loader, create_synthetic_parquet):
    create_related_pair(create_synthetic_parquet, tmp_path, "CLZ26", "CLF27", 2.0, 50, seed=2)
    with pytest.raises(InsufficientDataError):
        calculate_rolling_beta("CLF27", "CLZ26", 120, loader)


# ---------- fit_ou_process ----------


def test_ou_mean_reverting_series_detected(tmp_path, loader, create_synthetic_parquet):
    create_ou_series(create_synthetic_parquet, tmp_path, "CLZ26", 0.1, 75.0, 1.0, 1500, seed=7)
    result = fit_ou_process("CLZ26", loader)
    assert result["is_mean_reverting"] is True
    assert result["data_warning"] is None
    assert 0.05 < result["kappa"] < 0.15
    assert result["observations"] == 1499


def test_ou_half_life_positive(tmp_path, loader, create_synthetic_parquet):
    create_ou_series(create_synthetic_parquet, tmp_path, "CLZ26", 0.1, 75.0, 1.0, 1500, seed=7)
    result = fit_ou_process("CLZ26", loader)
    assert result["half_life_days"] > 0
    assert result["half_life_days"] == pytest.approx(math.log(2) / result["kappa"])
    assert 3.0 < result["half_life_days"] < 14.0


def test_ou_mu_close_to_true_mean(tmp_path, loader, create_synthetic_parquet):
    create_ou_series(create_synthetic_parquet, tmp_path, "CLZ26", 0.1, 75.0, 1.0, 1500, seed=7)
    result = fit_ou_process("CLZ26", loader)
    assert result["mu"] == pytest.approx(75.0, abs=2.0)
    assert result["sigma"] == pytest.approx(1.0, abs=0.1)
    assert "revert halfway" in result["interpretation"]


def test_ou_non_mean_reverting_series_has_no_half_life(tmp_path, loader, create_synthetic_parquet):
    exponential_growth = 75.0 * 1.01 ** np.arange(200)  # kappa = -0.01
    create_synthetic_parquet(tmp_path, "CLZ26", closes=exponential_growth)
    result = fit_ou_process("CLZ26", loader)
    assert result["is_mean_reverting"] is False
    assert result["kappa"] < 0
    assert result["mu"] is None
    assert result["half_life_days"] is None
    assert "does not appear mean-reverting" in result["data_warning"]


def test_ou_fewer_than_60_observations_raises(tmp_path, loader, create_synthetic_parquet):
    create_ou_series(create_synthetic_parquet, tmp_path, "CLZ26", 0.1, 75.0, 1.0, 50, seed=7)
    with pytest.raises(InsufficientDataError):
        fit_ou_process("CLZ26", loader)


# ---------- estimate_time_to_target ----------

OU_FIT = {"kappa": 0.1, "mu": 75.0, "is_mean_reverting": True}


def test_time_to_target_valid_between_current_and_mu():
    result = estimate_time_to_target(OU_FIT, current_price=80.0, target_price=77.0)
    assert result["is_valid"] is True
    assert result["expected_days"] == pytest.approx(10.0 * math.log(5.0 / 2.0))
    assert result["mu"] == 75.0


def test_time_to_target_valid_below_mu():
    result = estimate_time_to_target(OU_FIT, current_price=70.0, target_price=73.0)
    assert result["is_valid"] is True
    assert result["expected_days"] == pytest.approx(10.0 * math.log(5.0 / 2.0))


def test_time_to_target_beyond_mu_is_invalid():
    result = estimate_time_to_target(OU_FIT, current_price=80.0, target_price=73.0)
    assert result["is_valid"] is False
    assert result["expected_days"] is None
    assert "not in the direction of mean reversion" in result["validity_note"]


def test_time_to_target_away_from_mu_is_invalid():
    result = estimate_time_to_target(OU_FIT, current_price=80.0, target_price=85.0)
    assert result["is_valid"] is False
    assert result["expected_days"] is None


def test_time_to_target_exactly_at_mu_is_invalid():
    result = estimate_time_to_target(OU_FIT, current_price=80.0, target_price=75.0)
    assert result["is_valid"] is False


def test_time_to_target_not_mean_reverting_is_invalid():
    result = estimate_time_to_target(
        {"kappa": -0.01, "mu": None, "is_mean_reverting": False}, 80.0, 77.0
    )
    assert result["is_valid"] is False
    assert result["expected_days"] is None
    assert "not mean-reverting" in result["validity_note"]


def test_time_to_target_non_finite_input_does_not_raise():
    result = estimate_time_to_target(OU_FIT, float("nan"), 77.0)
    assert result["is_valid"] is False
    assert result["current_price"] is None


# ---------- find_sigma_moves ----------


def write_random_walk(create_synthetic_parquet, tmp_path, symbol="CLZ26", n_days=600, seed=3):
    create_synthetic_parquet(tmp_path, symbol, n_days=n_days, daily_vol=0.02, seed=seed)


def test_sigma_moves_returns_expected_columns(tmp_path, loader, create_synthetic_parquet):
    write_random_walk(create_synthetic_parquet, tmp_path)
    events = find_sigma_moves("CLZ26", 2.0, loader)
    assert list(events.columns) == [
        "event_date", "event_price", "event_diff", "event_sigma", "direction",
        "days_to_reversion", "reverted_in_30d", "max_extension", "max_adverse_move",
    ]
    assert len(events) >= 5
    assert set(events["direction"]) <= {"up", "down"}


def test_sigma_moves_all_events_exceed_threshold(tmp_path, loader, create_synthetic_parquet):
    write_random_walk(create_synthetic_parquet, tmp_path)
    events = find_sigma_moves("CLZ26", 2.0, loader)
    assert (events["event_sigma"] > 2.0).all()
    assert (events["days_to_reversion"].dropna().between(1, 30)).all()
    assert (events["max_adverse_move"] >= 0).all()
    assert (events["max_extension"] >= events["max_adverse_move"]).all()


def test_sigma_moves_too_few_events_raises(tmp_path, loader, create_synthetic_parquet):
    write_random_walk(create_synthetic_parquet, tmp_path)
    with pytest.raises(InsufficientDataError, match="sigma events found"):
        find_sigma_moves("CLZ26", 10.0, loader)


def test_sigma_moves_excludes_events_without_full_forward_window(tmp_path, loader, create_synthetic_parquet):
    closes = spike_closes()
    closes[-5:] = [75.0, 75.0, 75.0, 75.0, 90.0]  # huge move in the last 5 days
    create_synthetic_parquet(tmp_path, "CLZ26", closes=closes)
    events = find_sigma_moves("CLZ26", 3.0, loader)
    assert (events["event_price"] < 85).all()


def test_sigma_moves_known_spike_path(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=spike_closes())
    events = find_sigma_moves("CLZ26", 3.0, loader)

    spike_date = END_DATE - pd.Timedelta(days=300 - 1 - 100)
    row = events[events["event_date"] == spike_date].iloc[0]
    assert row["direction"] == "up"
    assert row["event_price"] == 80.0
    assert row["reverted_in_30d"]
    assert row["days_to_reversion"] == 3.0
    assert row["max_adverse_move"] == pytest.approx(1.0)
    assert row["max_extension"] == pytest.approx(1.0)


# ---------- analyze_sigma_moves ----------


def test_analyze_survival_plus_trigger_equals_one(tmp_path, loader, create_synthetic_parquet):
    write_random_walk(create_synthetic_parquet, tmp_path)
    events = find_sigma_moves("CLZ26", 2.0, loader)
    stats = analyze_sigma_moves(events, stop_loss_ticks=100, tick_size=0.01)
    assert stats["stop_loss_survival_probability"] + stats["stop_loss_trigger_probability"] == pytest.approx(1.0)
    assert stats["stop_loss_price"] == pytest.approx(1.0)


def test_analyze_hit_rate_between_zero_and_one(tmp_path, loader, create_synthetic_parquet):
    write_random_walk(create_synthetic_parquet, tmp_path)
    events = find_sigma_moves("CLZ26", 2.0, loader)
    stats = analyze_sigma_moves(events, stop_loss_ticks=100, tick_size=0.01)
    assert 0.0 <= stats["reversion_hit_rate"] <= 1.0
    assert stats["total_events"] == len(events)
    assert stats["up_events"] + stats["down_events"] == stats["total_events"]
    assert stats["sigma_threshold"] == 2.0
    assert "2σ" in stats["interpretation"]


def test_analyze_stop_probability_on_known_spikes(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=spike_closes())
    events = find_sigma_moves("CLZ26", 3.0, loader)
    # up spikes have a 1.0 adverse excursion; the down retraces stay far below 0.8
    stats = analyze_sigma_moves(events, stop_loss_ticks=80, tick_size=0.01)
    assert stats["up_events"] == 5
    assert stats["down_events"] == 5
    assert stats["stop_loss_trigger_probability"] == pytest.approx(0.5)
    assert stats["up_reversion_rate"] == 1.0
    assert stats["avg_reversion_days"] is not None


def test_analyze_handles_no_reversions_without_nan():
    events = pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=5, tz="UTC"),
            "event_price": 75.0,
            "event_diff": 3.0,
            "event_sigma": 3.5,
            "direction": "up",
            "days_to_reversion": float("nan"),
            "reverted_in_30d": False,
            "max_extension": 2.0,
            "max_adverse_move": 2.0,
        }
    )
    stats = analyze_sigma_moves(events, stop_loss_ticks=100, tick_size=0.01)
    assert stats["reversion_hit_rate"] == 0.0
    assert stats["avg_reversion_days"] is None
    assert stats["median_reversion_days"] is None
    assert stats["down_reversion_rate"] is None
    assert stats["stop_loss_trigger_probability"] == 1.0
    assert stats["interpretation"]


def test_analyze_invalid_inputs_raise():
    events = pd.DataFrame({"event_sigma": [3.0]})
    with pytest.raises(ValueError):
        analyze_sigma_moves(events, stop_loss_ticks=0, tick_size=0.01)
    with pytest.raises(InsufficientDataError):
        analyze_sigma_moves(pd.DataFrame(), stop_loss_ticks=10, tick_size=0.01)


# ---------- run_trade_idea_analysis ----------

REQUIRED_KEYS = {
    "symbol", "current_price", "ou_analysis", "ou_error", "time_to_target",
    "sigma_analysis", "sigma_error", "beta_analysis", "beta_error",
    "stop_loss_ticks", "target_ticks", "tick_size", "tick_value", "analyzed_at",
}


def run_analysis(loader, **overrides):
    kwargs = dict(
        symbol="CLZ26",
        current_price=78.0,
        sigma_threshold=2.0,
        stop_loss_ticks=100,
        target_ticks=200,
        tick_size=0.01,
        tick_value=10.0,
        data_loader=loader,
        beta_benchmark="CLF27",
        window=60,
    )
    kwargs.update(overrides)
    return run_trade_idea_analysis(**kwargs)


def test_trade_idea_returns_all_required_keys(tmp_path, loader, create_synthetic_parquet):
    create_ou_series(create_synthetic_parquet, tmp_path, "CLZ26", 0.1, 75.0, 1.0, 800, seed=7)
    create_ou_series(create_synthetic_parquet, tmp_path, "CLF27", 0.1, 74.0, 1.0, 800, seed=8)
    result = run_analysis(loader)

    assert REQUIRED_KEYS <= set(result)
    assert result["ou_error"] is None
    assert result["sigma_error"] is None
    assert result["beta_error"] is None
    assert result["ou_analysis"]["is_mean_reverting"] is True
    assert result["sigma_analysis"]["total_events"] >= 5
    assert result["beta_analysis"]["symbol_b"] == "CLF27"
    # current 78 > mu ~75, so the 200-tick (2.0) target is below current
    assert result["time_to_target"]["target_price"] == pytest.approx(76.0)
    assert result["time_to_target"]["is_valid"] is True


def test_trade_idea_without_benchmark_skips_beta(tmp_path, loader, create_synthetic_parquet):
    create_ou_series(create_synthetic_parquet, tmp_path, "CLZ26", 0.1, 75.0, 1.0, 800, seed=7)
    result = run_analysis(loader, beta_benchmark=None)
    assert result["beta_analysis"] is None
    assert result["beta_error"] is None


def test_trade_idea_component_failure_populates_error_and_continues(tmp_path, loader, create_synthetic_parquet):
    create_ou_series(create_synthetic_parquet, tmp_path, "CLZ26", 0.1, 75.0, 1.0, 800, seed=7)
    # No CLF27 data -> beta fails; threshold of 10 sigma -> sigma analysis fails.
    result = run_analysis(loader, sigma_threshold=10.0)

    assert REQUIRED_KEYS <= set(result)
    assert result["ou_analysis"] is not None and result["ou_error"] is None
    assert result["sigma_analysis"] is None and "InsufficientDataError" in result["sigma_error"]
    assert result["beta_analysis"] is None and "DataNotAvailableError" in result["beta_error"]


def test_trade_idea_never_raises_when_symbol_has_no_data(loader):
    result = run_analysis(loader)
    assert REQUIRED_KEYS <= set(result)
    assert result["ou_analysis"] is None and result["ou_error"]
    assert result["sigma_analysis"] is None and result["sigma_error"]
    assert result["beta_analysis"] is None and result["beta_error"]
    assert result["time_to_target"] is None


def test_trade_idea_invalid_target_inputs_do_not_raise(tmp_path, loader, create_synthetic_parquet):
    create_ou_series(create_synthetic_parquet, tmp_path, "CLZ26", 0.1, 75.0, 1.0, 800, seed=7)
    result = run_analysis(loader, target_ticks=0, tick_size=0.0, beta_benchmark=None)
    assert result["time_to_target"]["is_valid"] is False
    assert result["sigma_error"] is not None
