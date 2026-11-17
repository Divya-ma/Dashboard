"""core.lead_lag (the cross-correlation math), core.lead_lag_data (which contracts go in a matrix) and the
Correlation tab's Lead-Lag callbacks (ui.callbacks.lead_lag_callbacks)."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.exceptions import InsufficientDataError
from core.lead_lag import NO_SIGN, STANDS_OUT, bar_differences, best_lag, cross_correlation, describe_lead_lag, lead_lag_matrix
from core.lead_lag_data import ALL_GROUPS, GROUP_SIZE, group_symbols, load_bar_diffs
from ui.callbacks import lead_lag_callbacks as cb
from ui.container import Container


def ar1_series(n: int, phi: float = 0.6, seed: int = 1) -> pd.Series:
    """A stationary AR(1) series (momentum), 5-minute-apart timestamps: a believable price-difference series."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = phi * values[t - 1] + rng.normal()
    index = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    return pd.Series(values, index=index)


# ---------- bar_differences ----------


def test_bar_differences_sorts_dedupes_and_diffs():
    frame = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-05 10:05", tz="UTC"), pd.Timestamp("2026-01-05 10:00", tz="UTC"),
                      pd.Timestamp("2026-01-05 10:05", tz="UTC")],  # out of order, with a duplicate bar
        "close": [101.0, 100.0, 102.0],  # the duplicate's LATER value (102.0) should win
    })
    diffs = bar_differences(frame)
    assert list(diffs.to_numpy()) == [2.0]  # 102.0 - 100.0, the one real step


def test_bar_differences_on_empty_frame_is_an_empty_series():
    assert bar_differences(pd.DataFrame(columns=["timestamp", "close"])).empty


# ---------- cross_correlation / best_lag ----------


def test_cross_correlation_needs_min_observations():
    a, b = ar1_series(10), ar1_series(10, seed=2)
    with pytest.raises(InsufficientDataError):
        cross_correlation(a, b)


def test_a_pure_lag_is_detected_with_the_right_sign():
    """B is exactly A delayed by 3 bars: A should be found to LEAD B by 3 (positive lag)."""
    a = ar1_series(300)
    b = a.shift(3).dropna()
    ccf = cross_correlation(a, b, max_lag=10)
    result = best_lag(ccf)
    assert result["lag"] == 3
    assert result["correlation"] > 0.5
    assert result["reading"] == STANDS_OUT

    # swapping the call order flips which side "leads": now B is passed first, so the same relationship reads as -3.
    reversed_result = best_lag(cross_correlation(b, a, max_lag=10))
    assert reversed_result["lag"] == -3


def test_independent_series_show_no_reliable_lead_lag():
    a, b = ar1_series(200, seed=10), ar1_series(200, seed=11)
    result = best_lag(cross_correlation(a, b, max_lag=12))
    assert result["reading"] == NO_SIGN


def test_describe_lead_lag_reads_in_both_directions_and_for_no_signal():
    leading = {"lag": 2, "correlation": 0.8, "reading": STANDS_OUT, "p_value": 0.0001, "n_obs": 500, "n_lags_tested": 25}
    text = describe_lead_lag("CLZ26", "CLF27", leading, bar_minutes=5)
    assert text.startswith("CLZ26 leads CLF27 by about 10 minutes")
    lagging = {**leading, "lag": -2}
    assert describe_lead_lag("CLZ26", "CLF27", lagging, bar_minutes=5).startswith("CLF27 leads CLZ26")
    flat = {"lag": 0, "correlation": 0.01, "reading": NO_SIGN, "p_value": 0.9, "n_obs": 500, "n_lags_tested": 25}
    assert "move together" in describe_lead_lag("CLZ26", "CLF27", flat, bar_minutes=5)


# ---------- lead_lag_matrix ----------


def test_lead_lag_matrix_shape_and_antisymmetric_lag():
    a = ar1_series(300, seed=5)
    b = a.shift(2).dropna()
    c = ar1_series(300, seed=99)  # unrelated third series
    lag_matrix, strength_matrix, detail = lead_lag_matrix({"A": a, "B": b, "C": c}, max_lag=8)
    assert list(lag_matrix.index) == list(lag_matrix.columns) == ["A", "B", "C"]
    assert lag_matrix.loc["A", "B"] == 2 and lag_matrix.loc["B", "A"] == -2
    assert strength_matrix.loc["A", "B"] == strength_matrix.loc["B", "A"]
    assert strength_matrix.loc["A", "A"] == strength_matrix.loc["B", "B"] == 1.0
    assert ("A", "B") in detail and detail[("A", "B")]["lag"] == 2


def test_lead_lag_matrix_skips_a_pair_with_too_little_data_instead_of_raising():
    a, tiny = ar1_series(300), ar1_series(5, seed=7)
    _, strength_matrix, detail = lead_lag_matrix({"A": a, "Tiny": tiny}, max_lag=5)
    assert strength_matrix.loc["A", "Tiny"] != strength_matrix.loc["A", "Tiny"]  # NaN
    assert detail == {}


# ---------- core.lead_lag_data ----------


AS_OF = date(2026, 11, 13)


def test_group_symbols_gives_four_consecutive_contracts_for_one_product():
    symbols = group_symbols("CL", 1, AS_OF)
    assert len(symbols) == GROUP_SIZE
    assert all(sym.startswith("CL") for sym in symbols.values())


def test_group_symbols_starting_further_out_shifts_the_window():
    front = group_symbols("CL", 1, AS_OF)
    later = group_symbols("CL", 3, AS_OF)
    assert set(front) & set(later)  # months 3-4 are shared between a 1-4 and a 3-6 window
    assert set(front) != set(later)


def test_group_symbols_all_groups_gives_one_per_product_with_prefixed_labels():
    symbols = group_symbols(ALL_GROUPS, 1, AS_OF)
    assert len(symbols) == 4  # CL, BRN, G, CLBZ
    assert all(" " in label for label in symbols)  # "<product> <month label>"
    products = {label.split(" ")[0] for label in symbols}
    assert products == {"CL", "BRN", "G", "CLBZ"}


class _FakeAdapter:
    """Stands in for adapters.base.HistoricalDataAdapter: returns canned bars, no API or cache involved."""

    def __init__(self, bars_by_symbol: dict[str, pd.DataFrame]):
        self._bars = bars_by_symbol

    def get_ohlc_bulk(self, symbols, interval, start=None, end=None, count=None):
        return {s: self._bars.get(s, pd.DataFrame(columns=["timestamp", "close"])) for s in symbols}


def _bars(prices: list[float], offset_bars: int = 0) -> pd.DataFrame:
    """OHLC-ish bars (just "timestamp"/"close") of 5-minute candles, `offset_bars` bars after a fixed start --
    so a series built from another one's values, with an offset, really does lag it by that many bars."""
    start = datetime(2026, 11, 10, tzinfo=timezone.utc) + pd.Timedelta(minutes=5 * offset_bars)
    return pd.DataFrame({
        "timestamp": [start + pd.Timedelta(minutes=5 * i) for i in range(len(prices))],
        "close": prices,
    })


def test_load_bar_diffs_skips_symbols_with_no_bars():
    adapter = _FakeAdapter({"CLZ26": _bars([100.0, 101.0, 100.5])})
    diffs = load_bar_diffs({"Front": "CLZ26", "No data": "CLF27"}, adapter)
    assert list(diffs) == ["Front"]
    assert list(diffs["Front"].to_numpy()) == pytest.approx([1.0, -0.5])


# ---------- ui.callbacks.lead_lag_callbacks ----------


CB_AS_OF = date(2026, 11, 13)


def _price_levels(n: int, seed: int) -> list[float]:
    return list(100.0 + ar1_series(n, seed=seed).cumsum().to_numpy())


@pytest.fixture
def page(monkeypatch):
    """Stubs cb.container with a fake historical adapter whose bars encode a known lead-lag: Jan leads Feb by
    2 bars, Mar and Apr (same AR process, independent draws) have no reliable relationship to either."""
    symbols = group_symbols("CL", 1, CB_AS_OF)  # the real 4 live CL contracts as of CB_AS_OF, to key the fake bars by
    ordered = list(symbols.values())
    front = _price_levels(400, seed=21)
    bars = {
        ordered[0]: _bars(front),
        ordered[1]: _bars(front[:-2], offset_bars=2),  # the same series, 2 bars later: ordered[0] leads ordered[1] by 2
        ordered[2]: _bars(_price_levels(400, seed=22)),
        ordered[3]: _bars(_price_levels(400, seed=23)),
    }
    adapter = _FakeAdapter(bars)
    monkeypatch.setattr(cb, "container", Container())
    cb.container.historical_adapter = adapter
    monkeypatch.setattr(cb, "_today", lambda: CB_AS_OF)
    cb._CACHE.clear()
    return symbols


def test_render_matrix_end_to_end_and_click_detail(page):
    figure, status, params = cb.render_matrix(1, "CL", 1, 5)
    assert params["group"] == "CL" and len(figure.data[0].x) == 4
    assert "4 consecutive months" in str(status)

    labels = list(figure.data[0].x)
    click = {"points": [{"x": labels[1], "y": labels[0]}]}  # the pair we rigged: labels[0] leads labels[1] by 2 bars
    text, ccf_figure = cb.render_click_detail(click, params)
    assert "leads" in str(text) and len(ccf_figure.data) >= 2

    same_cell = {"points": [{"x": labels[0], "y": labels[0]}]}
    no_lead_text, _ = cb.render_click_detail(same_cell, params)
    assert "no lead-lag" in str(no_lead_text)


def test_render_matrix_reports_bad_input(monkeypatch):
    monkeypatch.setattr(cb, "container", Container())
    cb.container.historical_adapter = _FakeAdapter({})
    figure, status, params = cb.render_matrix(1, None, 1, 5)
    assert "Pick a group" in str(status)


def test_render_matrix_without_an_adapter_is_a_friendly_message(monkeypatch):
    monkeypatch.setattr(cb, "container", Container())  # historical_adapter stays None
    figure, status, params = cb.render_matrix(1, "CL", 1, 5)
    assert "not available" in str(status)


def test_render_click_detail_before_a_compute_prompts_to_click():
    text, figure = cb.render_click_detail(None, None)
    assert "Click a cell" in str(text)
