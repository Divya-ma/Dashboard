"""Tests for core.regression: beta of one series on another, and the position sizing it implies."""

import math

import numpy as np
import pandas as pd
import pytest

from core.exceptions import InsufficientDataError
from core.regression import _ensure_finite, beta_from_series, position_sizing_from_beta


def series_pair(slope, n=120, noise=0.3, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=n, freq="B", tz="UTC")
    b = pd.Series(rng.normal(0, 1, n), index=index)
    a = slope * b + rng.normal(0, noise, n)
    return a, b


def test_beta_recovers_the_known_slope_and_reports_its_fit():
    a, b = series_pair(1.7)
    result = beta_from_series(a, b, 60)
    assert result["beta"] == pytest.approx(1.7, abs=0.15) and 0.8 < result["r_squared"] <= 1.0
    assert result["observations"] == 60 == result["window_used"] and result["beta_p_value"] < 0.01 and result["beta_std_error"] > 0
    assert all(math.isfinite(result[k]) for k in ("beta", "alpha", "r_squared"))


def test_a_window_longer_than_the_history_uses_what_there_is():
    a, b = series_pair(1.0, n=45)
    assert beta_from_series(a, b, 500)["window_used"] == 45


def test_fewer_than_30_shared_days_or_a_flat_reference_is_refused():
    a, b = series_pair(1.0, n=29)
    with pytest.raises(InsufficientDataError, match="minimum required is 30"):
        beta_from_series(a, b, 60)
    a, b = series_pair(1.0)
    with pytest.raises(InsufficientDataError, match="zero variance"):
        beta_from_series(a, b * 0.0, 60)


def test_only_the_shared_dates_are_used():
    a, b = series_pair(2.0, n=100)
    result = beta_from_series(a.iloc[:80], b.iloc[20:], 100)  # 60 days overlap
    assert result["observations"] == 60


def test_position_sizing_matches_with_beta_and_hedges_with_minus_beta():
    a, b = series_pair(0.8)
    beta = beta_from_series(a, b, 60)
    sized = position_sizing_from_beta(beta, 1.0)
    assert sized["lots_to_match"] == pytest.approx(beta["beta"]) and sized["lots_to_hedge"] == pytest.approx(-beta["beta"])
    per_lot = position_sizing_from_beta(beta, 4.0)  # a candidate series already scaled to 4 lots
    assert per_lot["lots_to_match"] == pytest.approx(beta["beta"] / 4) and per_lot["current_lots"] == 4.0
    with pytest.raises(ValueError):
        position_sizing_from_beta(beta, 0)


def test_ensure_finite_raises_on_nan_or_inf():
    _ensure_finite({"a": 1.0, "b": -2.0}, "ctx")
    with pytest.raises(InsufficientDataError, match="non-finite b"):
        _ensure_finite({"a": 1.0, "b": float("nan")}, "ctx")
    with pytest.raises(InsufficientDataError, match="a"):
        _ensure_finite({"a": float("inf")}, "ctx")
