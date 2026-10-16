"""Structure correlation matrix: structures from settlements, the lookback average, the historical range, hedge ratio."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core import corr_matrix as cm
from core.oi_data import contract_universe
from ui.callbacks import corr_matrix_callbacks as cb
from ui.container import Container

TODAY = date(2026, 10, 13)


def _frame(product="CL", days=300, seed=3):
    """Random-walk settlements for every contract from 2025 on, the contracts sharing one common factor."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=TODAY - timedelta(days=1), periods=days)
    common = rng.normal(0, 1, len(index)).cumsum()
    rows = []
    for i, contract in enumerate(c for c in contract_universe(product, TODAY) if c.year >= 2025):
        idio = rng.normal(0, 0.3, len(index)).cumsum()
        close = 70 + common * (1 + 0.02 * i) + idio - 0.2 * i
        alive = index <= pd.Timestamp(contract.expiry)
        rows.append(pd.DataFrame({"contract": contract.symbol, "date": index[alive], "oi": 1.0, "close": close[alive], "volume": 1.0}))
    return pd.concat(rows, ignore_index=True)


def test_structure_prices_use_the_formulas():
    frame = _frame()
    for family, weights in cm.WEIGHTS.items():
        s = cm.build_structure_set(frame, "CL", family, 5, TODAY, TODAY)
        prices = frame.pivot(index="date", columns="contract", values="close")
        legs = s.contracts[0]
        expected = sum(w * prices[sym] for sym, w in zip(legs, weights))
        assert np.allclose(s.levels[s.labels[0]].dropna(), expected.dropna())
        assert len(s.labels) == 5 and len(legs) == len(weights)
    assert cm.build_structure_set(frame, "CL", "outright", 3, TODAY, TODAY).labels[0] == "Nov26"  # CLX26 is the front


def test_matrix_is_symmetric_with_unit_diagonal_for_one_curve():
    s = cm.build_structure_set(_frame(), "CL", "spread", 6, TODAY, TODAY)
    values, days = cm.correlation_matrix(s, s, 5, 20)
    assert values.shape == (6, 6)
    assert np.allclose(np.diag(values), 1.0) and np.allclose(values, values.T, equal_nan=True)
    # one cell by hand: the average of the last 20 five-day correlations of the two changes
    a, b = s.changes[s.labels[0]], s.changes[s.labels[2]]
    both = pd.concat([a, b], axis=1).dropna()
    expected = both.iloc[:, 0].rolling(5).corr(both.iloc[:, 1]).dropna().iloc[-20:].mean()
    assert values.iloc[0, 2] == pytest.approx(expected)
    assert days.iloc[0, 2] == 20


def test_target_and_base_may_be_different_curves():
    frame = _frame()
    target = cm.build_structure_set(frame, "CL", "fly", 4, TODAY, TODAY)
    base = cm.build_structure_set(frame, "CL", "spread", 6, TODAY, TODAY)
    values, _ = cm.correlation_matrix(target, base, 5, 20)
    assert values.shape == (4, 6) and values.notna().to_numpy().all()


def test_historical_range_averages_every_day_inside_it():
    frame = _frame(days=400)
    as_of = TODAY - timedelta(days=120)
    s = cm.build_structure_set(frame, "CL", "outright", 3, as_of, TODAY)
    start = as_of - timedelta(days=60)
    values, days = cm.correlation_matrix(s, s, 5, 20, start)
    daily = cm.pair_daily(s.changes[s.labels[0]], s.changes[s.labels[1]], 5)
    inside = daily[(daily.index >= pd.Timestamp(start)) & (daily.index <= pd.Timestamp(as_of))]
    assert days.iloc[0, 1] == len(inside) > 20  # the range, not the 20-day lookback
    assert values.iloc[0, 1] == pytest.approx(inside.mean())
    assert s.levels.index.max() <= pd.Timestamp(as_of)  # nothing after the range end


def test_rolling_beta_recovers_a_known_ratio():
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2026-01-01", periods=120)
    base = pd.Series(rng.normal(0, 1, 120), index=index)
    target = 1.7 * base + rng.normal(0, 0.05, 120)
    result = cm.rolling_beta(target, base, 30)
    assert result["beta"].iloc[-1] == pytest.approx(1.7, abs=0.05) and result["corr"].iloc[-1] > 0.99
    assert cm.rolling_beta(target.iloc[:10], base.iloc[:10], 30).empty


def test_range_validation():
    cm.validate_range("current", None, None)
    with pytest.raises(ValueError):
        cm.validate_range("historical", None, date(2024, 1, 1))
    with pytest.raises(ValueError):
        cm.validate_range("historical", date(2024, 2, 1), date(2024, 1, 1))


@pytest.fixture
def page(monkeypatch):
    frame = _frame()
    store = SimpleNamespace(version="v1", frame=lambda product: frame if product == "CL" else frame.iloc[0:0], has_data=lambda: True)
    service = SimpleNamespace(store=store, ensure_fresh=lambda: None)
    monkeypatch.setattr(cb, "container", Container())
    cb.container.oi_service = service
    monkeypatch.setattr(cb, "_today", lambda: TODAY)
    cb._CACHE.clear()
    return service


def test_callbacks_end_to_end(page):
    figure, status, params, opt_t, opt_b, ht, hb = cb.render_matrix("CL|fly", "CL|spread", 6, 5, 20, "current", None, None, None, None)
    assert params["n"] == 6 and len(figure.data[0].y) == 6 and len(figure.data[0].x) == 6
    assert ht == opt_t[0]["value"] and hb == opt_b[1]["value"]
    click = {"points": [{"x": params and figure.data[0].x[1], "y": figure.data[0].y[0]}]}
    chart = cb.render_click_chart(click, params, 30)
    assert len(chart.data[0].y) == 30
    text, hedge = cb.render_hedge(ht, hb, 40, None, params)
    assert "lot(s)" in str(text) and len(hedge.data[0].y) == 40
    assert "Click a cell" in cb.render_click_chart(None, params, 30).layout.annotations[0].text


def test_callbacks_report_bad_input(page):
    _, status, params, *_ = cb.render_matrix("CL|fly", "CL|fly", 6, 1, 20, "current", None, None, None, None)
    assert "Rolling window" in str(status)
    _, status, *_ = cb.render_matrix("CL|fly", "CL|fly", 6, 5, 20, "historical", None, None, None, None)
    assert "date range" in str(status)
    figure, status, *_ = cb.render_matrix("BRN|fly", "BRN|fly", 6, 5, 20, "current", None, None, None, None)
    assert "no settlement data" in str(status)
