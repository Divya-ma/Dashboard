"""core.hedge: the one slope / correlation definition behind every hedge ratio."""

import numpy as np
import pandas as pd
import pytest

from core.atr_engine import pair_hedge
from core.corr_matrix import rolling_beta
from core.hedge import rolling_slope, slope_and_corr
from core.regression import beta_from_series


def pair(slope=1.7, n=120, noise=0.2, seed=1):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=n, freq="B", tz="UTC")
    x = pd.Series(rng.normal(0, 1, n), index=index)
    return slope * x + rng.normal(0, noise, n), x


def test_slope_and_corr_recover_the_relationship():
    y, x = pair()
    beta, corr = slope_and_corr(y.to_numpy(), x.to_numpy())
    assert beta == pytest.approx(1.7, abs=0.1) and 0.95 < corr <= 1.0


def test_degenerate_inputs_give_nan_not_errors():
    assert all(np.isnan(v) for v in slope_and_corr([1, 2], [1, 2]))  # too few pairs
    assert all(np.isnan(v) for v in slope_and_corr([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]))  # a flat hedge instrument
    beta, corr = slope_and_corr([2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0])  # a flat position: beta 0, correlation undefined
    assert beta == 0.0 and np.isnan(corr)


def test_every_page_gets_the_same_slope_for_the_same_data():
    y, x = pair(seed=4)
    beta, corr = slope_and_corr(y.to_numpy(), x.to_numpy())
    assert beta_from_series(y, x, 200)["beta"] == pytest.approx(beta)  # the OLS panels
    assert rolling_slope(y, x, 120).iloc[-1] == pytest.approx(beta)  # the correlation page's rolling hedge ratio
    assert rolling_beta(y, x, 120)["beta"].iloc[-1] == pytest.approx(beta) and rolling_beta(y, x, 120)["corr"].iloc[-1] == pytest.approx(corr)


def test_the_atr_pair_hedge_uses_the_same_slope():
    y, x = pair(seed=5)
    frame = lambda s: pd.DataFrame({"close": s.cumsum() + 70.0, "atr": s.abs().rolling(14).mean()})
    result = pair_hedge(frame(y), frame(x), "A", "B", 60)
    tail = pd.concat([frame(y)["close"].diff(), frame(x)["close"].diff()], axis=1).dropna().iloc[-60:]
    assert result.beta_window == pytest.approx(slope_and_corr(tail.iloc[:, 0].to_numpy(), tail.iloc[:, 1].to_numpy())[0])
