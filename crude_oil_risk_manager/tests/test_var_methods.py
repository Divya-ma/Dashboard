"""core.var: historical simulation and the Monte Carlo (each symbol's own volatility by default)."""

import numpy as np
import pandas as pd
import pytest

from adapters.mock.mock_historical import MockHistoricalAdapter
from core.data_loader import DataLoader
from core.var import (
    VOL_FIXED,
    VOL_HISTORY,
    historical_var,
    historical_volatilities,
    monte_carlo_var,
    pnl_series,
    simulate_correlated_pnl,
)


@pytest.fixture
def loader(tmp_path):
    return DataLoader(str(tmp_path), MockHistoricalAdapter())


def walk(n, seed, vol):
    return 75 + np.cumsum(np.random.default_rng(seed).normal(0, vol, n))


def matrix(symbols, rho=0.0):
    values = np.full((len(symbols), len(symbols)), rho)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=symbols, columns=symbols)


# ---------- historical simulation ----------


def test_pnl_series_combines_symbols_on_common_dates_and_reports_the_ones_left_out(tmp_path, loader, create_synthetic_parquet):
    a, b = walk(80, 1, 0.5), walk(60, 2, 0.5)
    create_synthetic_parquet(tmp_path, "CLZ26", closes=a)
    create_synthetic_parquet(tmp_path, "CLF27", closes=b)
    series, skipped = pnl_series({"CLZ26": 1000.0, "CLF27": -2000.0, "CLG27": 500.0}, loader)  # CLG27 has no file
    assert list(skipped) == ["CLG27"] and len(series) == 59
    expected = np.diff(a)[-59:] * 1000.0 + np.diff(b) * -2000.0
    assert series.to_numpy() == pytest.approx(expected)
    assert pnl_series({}, loader) == (None, {})


def test_historical_var_is_the_percentile_of_the_last_n_days(tmp_path, loader, create_synthetic_parquet):
    closes = walk(150, 3, 0.4)
    create_synthetic_parquet(tmp_path, "CLZ26", closes=closes)
    result = historical_var({"CLZ26": 1000.0}, 60, loader)
    pnl = np.diff(closes)[-60:] * 1000.0
    assert result.error is None and result.observations == 60
    assert result.var[0.95] == pytest.approx(max(0.0, -np.percentile(pnl, 5))) and result.var[0.99] >= result.var[0.95]
    short = historical_var({"CLZ26": -1000.0}, 60, loader)  # the other side loses on the other tail
    assert short.var[0.95] == pytest.approx(max(0.0, -np.percentile(-pnl, 5)))


def test_historical_var_says_why_it_has_no_number(tmp_path, loader):
    assert historical_var({}, 60, loader).error == "No open structures — nothing to analyze"
    assert "No price data for any open leg" in historical_var({"CLZ26": 1000.0}, 60, loader).error
    with pytest.raises(ValueError):
        historical_var({"CLZ26": 1.0}, 0, loader)


def test_historical_var_warns_about_skipped_legs_and_short_history(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=walk(8, 1, 0.3))
    result = historical_var({"CLZ26": 1000.0, "CLF27": 1000.0}, 60, loader)
    assert result.skipped and any("understated" in w for w in result.warnings) and any("Insufficient data" in w for w in result.warnings)


# ---------- the simulation ----------


def test_each_symbol_moves_by_its_own_sigma_and_the_correlation_is_imposed():
    symbols = ["A", "B"]
    pnl = simulate_correlated_pnl(matrix(symbols, 0.6), {"A": 1.0, "B": 1.0}, "normal", {"mean": 0.0, "std": 1.0}, 100_000, seed=1, sigma={"A": 2.0, "B": 0.5})
    expected_sd = np.sqrt(2.0 ** 2 + 0.5 ** 2 + 2 * 0.6 * 2.0 * 0.5)  # var(A + B) with correlation 0.6
    assert pnl.std() == pytest.approx(expected_sd, rel=0.02) and pnl.mean() == pytest.approx(0.0, abs=0.03)


def test_the_mean_is_a_shift_of_each_move_not_something_the_correlation_mixes():
    pnl = simulate_correlated_pnl(matrix(["A", "B"], 0.9), {"A": 1.0, "B": 0.0}, "normal", {"mean": 0.3, "std": 0.1}, 50_000, seed=2)
    assert pnl.mean() == pytest.approx(0.3, abs=0.01)  # before, the correlation transformed the mean (0.3 -> about 0.57)


def test_uniform_draws_have_the_requested_spread_in_both_modes():
    fixed = simulate_correlated_pnl(matrix(["A"]), {"A": 1.0}, "uniform", {"low": -0.2, "high": 0.4}, 100_000, seed=3)
    assert fixed.mean() == pytest.approx(0.1, abs=0.01) and fixed.min() >= -0.2 - 1e-9 and fixed.max() <= 0.4 + 1e-9
    own = simulate_correlated_pnl(matrix(["A"]), {"A": 1.0}, "uniform", {"low": -9.0, "high": 9.0}, 100_000, seed=3, sigma={"A": 0.2})
    assert own.std() == pytest.approx(0.2, rel=0.02) and abs(own.mean()) < 0.01  # uniform scaled to the symbol's own sigma


def test_seed_reproduces_and_bad_inputs_are_refused():
    args = (matrix(["A"]), {"A": 1.0}, "normal", {"mean": 0.0, "std": 1.0})
    assert np.array_equal(simulate_correlated_pnl(*args, 500, seed=7), simulate_correlated_pnl(*args, 500, seed=7))
    with pytest.raises(ValueError):
        simulate_correlated_pnl(*args, 5)
    with pytest.raises(ValueError):
        simulate_correlated_pnl(matrix(["A"]), {"A": 1.0}, "normal", {"std": -1.0}, 500)
    with pytest.raises(ValueError):
        simulate_correlated_pnl(matrix(["A"]), {"A": 1.0}, "weird", {}, 500)
    assert len(simulate_correlated_pnl(matrix(["A"]), {}, "normal", {}, 500)) == 0


# ---------- the Monte Carlo VaR ----------


def test_history_mode_measures_each_symbols_own_volatility_over_the_lookback(tmp_path, loader, create_synthetic_parquet):
    quiet, loud = walk(200, 1, 0.05), walk(200, 2, 0.5)
    create_synthetic_parquet(tmp_path, "CLZ26", closes=quiet)
    create_synthetic_parquet(tmp_path, "CLF27", closes=loud)
    sigma, missing = historical_volatilities(["CLZ26", "CLF27", "CLG27"], loader, 60)
    assert sigma["CLZ26"] == pytest.approx(np.diff(quiet)[-60:].std(ddof=1)) and sigma["CLF27"] == pytest.approx(np.diff(loud)[-60:].std(ddof=1))
    assert list(missing) == ["CLG27"]
    result = monte_carlo_var({"CLZ26": 1000.0, "CLF27": 1000.0}, loader, "normal", {"mean": 0.0, "std": 0.5}, n_simulations=20_000, seed=4, vol_lookback=60)
    assert result.error is None and result.vol_mode == VOL_HISTORY and result.sigma["CLF27"] > 5 * result.sigma["CLZ26"]
    own = np.sqrt(sum((1000 * s) ** 2 for s in result.sigma.values()))  # roughly uncorrelated legs
    assert result.var[0.95] == pytest.approx(1.645 * own, rel=0.35)


def test_fixed_mode_is_the_one_size_for_every_symbol(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=walk(200, 1, 0.05))
    result = monte_carlo_var({"CLZ26": 1000.0}, loader, "normal", {"mean": 0.0, "std": 0.5}, n_simulations=50_000, seed=5, vol_mode=VOL_FIXED)
    assert result.sigma == {} and result.var[0.95] == pytest.approx(1.645 * 500, rel=0.03)


def test_a_symbol_without_history_falls_back_to_the_fixed_size_with_a_warning(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=walk(200, 1, 0.1))
    result = monte_carlo_var({"CLZ26": 1000.0, "CLF27": 1000.0}, loader, "normal", {"mean": 0.0, "std": 0.5}, n_simulations=10_000, seed=6)
    assert any("CLF27" in w and "fixed size" in w for w in result.warnings) and any("uncorrelated" in w for w in result.warnings)
    assert "CLF27" not in result.sigma


def test_monte_carlo_errors_are_returned_not_raised(loader):
    assert monte_carlo_var({}, loader, "normal", {}).error == "No positions — nothing to simulate"
    assert "std must be" in monte_carlo_var({"CLZ26": 1.0}, loader, "normal", {"std": -1.0}, vol_mode=VOL_FIXED).error
    with pytest.raises(ValueError):
        monte_carlo_var({"CLZ26": 1.0}, loader, "normal", {"std": 1.0}, confidence_levels=(1.5,))
