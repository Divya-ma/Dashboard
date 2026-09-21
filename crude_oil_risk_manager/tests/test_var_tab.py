"""Tests for the VaR & Scenarios tab: core.var extensions, core.scenarios, layout builders, callbacks."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dash.exceptions import PreventUpdate

from adapters.base import API_CODE_TO_PRODUCT, SymbolTranslator
from core.data_loader import DataLoader
from core.exceptions import InsufficientDataError
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from core.scenarios import KIND_PORTFOLIO, build_scenarios, load_previous_day_atr, previous_day_true_range
from core.var import (
    RELIABLE_OBSERVATIONS,
    leg_dollar_weights,
    portfolio_pnl_var,
    var_from_pnl_series,
)
from db.repository import Repository
from ui.callbacks import var_callbacks as vc
from ui.container import Container
from ui.layouts.shell import COLORS
from ui.layouts.var_tab import (
    ATR_UNAVAILABLE,
    NOTHING_TO_ANALYZE,
    build_histogram,
    build_scenario_note,
    build_scenario_table_rows,
    build_var_cards,
    build_var_warnings,
    var_scenario_layout,
)


class NoBackfillAdapter:
    def backfill_symbol(self, *args, **kwargs):
        raise AssertionError("the VaR tab must not backfill")


@pytest.fixture
def loader(tmp_path):
    return DataLoader(str(tmp_path), NoBackfillAdapter())


@pytest.fixture
def write_ohlc(tmp_path):
    def _write(symbol, closes, highs=None, lows=None, end="2026-09-18"):
        closes = np.asarray(closes, dtype=float)
        df = pd.DataFrame(
            {
                "symbol": symbol,
                "timestamp": pd.date_range(end=end, periods=len(closes), freq="D", tz="UTC"),
                "open": closes,
                "high": closes + 0.5 if highs is None else np.asarray(highs, dtype=float),
                "low": closes - 0.5 if lows is None else np.asarray(lows, dtype=float),
                "close": closes,
                "volume": 1000.0,
            }
        )
        api = SymbolTranslator.internal_to_api(symbol)
        code, _ = SymbolTranslator._match_product_prefix(api, API_CODE_TO_PRODUCT)
        path = Path(tmp_path) / code / f"{api}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    return _write


def random_closes(n, seed):
    return 75 + np.cumsum(np.random.default_rng(seed).normal(0, 1.5, n))


def structure(name, legs, status=StructureStatus.OPEN):
    """legs: [(symbol, ratio, direction, lots)]"""
    built = [
        Leg(contract=Contract(product="CL", contract_month=12, contract_year=2026, symbol=s, multiplier=1000,
                              tick_size=0.01, tick_value=10), ratio=r, direction=d, lots=lots, entry_price=1.0)
        for s, r, d, lots in legs
    ]
    return Structure(name=name, structure_type=StructureType.CUSTOM, products=["CL"], legs=built, status=status)


# ---------- dollar weights ----------


def test_leg_dollar_weights_apply_ratio_side_lots_and_multiplier():
    weights = leg_dollar_weights([
        structure("a", [("CLZ26", 1, "buy", 2), ("CLF27", -1, "buy", 2)]),
        structure("b", [("CLZ26", 1, "sell", 1)]),
        structure("fly", [("CLG27", 1, "sell", 1), ("CLH27", -2, "sell", 1)]),
        structure("spread symbol", [("CLJ27-K27", 1, "buy", 3)]),
    ])
    assert weights["CLZ26"] == 2000 - 1000 and weights["CLF27"] == -2000
    assert weights["CLG27"] == -1000 and weights["CLH27"] == 2000
    assert weights["CLJ27-K27"] == 3000  # exchange-quoted spread leg used as it is


def test_leg_dollar_weights_ignore_closed_shell_and_cancelling_legs():
    weights = leg_dollar_weights([
        structure("closed", [("CLZ26", 1, "buy", 5)], StructureStatus.CLOSED),
        structure("shell", [("CLZ26", 1, "buy", 0)], StructureStatus.SHELL),
        structure("up", [("CLF27", 1, "buy", 2)]),
        structure("down", [("CLF27", 1, "sell", 2)]),
    ])
    assert weights == {}


# ---------- VaR from a pre-built series ----------


def test_var_from_pnl_series_uses_the_percentile_rule():
    pnl = pd.Series(np.arange(-50.0, 50.0))
    var, cutoff = var_from_pnl_series(pnl)
    assert cutoff[0.95] == pytest.approx(np.percentile(pnl, 5)) and var[0.95] == pytest.approx(-cutoff[0.95])
    assert var[0.99] > var[0.95] > 0
    assert var_from_pnl_series(pd.Series([5.0, 6.0, 7.0]))[0][0.95] == 0.0  # all profits: floored at zero
    with pytest.raises(InsufficientDataError):
        var_from_pnl_series(pd.Series(dtype=float))


def test_portfolio_var_matches_a_hand_computed_series(write_ohlc, loader):
    closes = random_closes(120, 1)
    write_ohlc("CLZ26", closes)
    result = portfolio_pnl_var([structure("a", [("CLZ26", 1, "buy", 2)])], 60, loader)
    expected = np.diff(closes)[-60:] * 2000
    assert result.error is None and result.observations == 60 and result.requested == 60
    assert np.allclose(result.pnl.to_numpy(), expected)
    assert result.var[0.95] == pytest.approx(-np.percentile(expected, 5))
    assert result.var[0.99] == pytest.approx(-np.percentile(expected, 1))


def test_portfolio_var_sell_direction_flips_the_distribution(write_ohlc, loader):
    write_ohlc("CLZ26", random_closes(80, 2))
    buy = portfolio_pnl_var([structure("a", [("CLZ26", 1, "buy", 1)])], 60, loader)
    sell = portfolio_pnl_var([structure("a", [("CLZ26", 1, "sell", 1)])], 60, loader)
    assert np.allclose(sell.pnl.to_numpy(), -buy.pnl.to_numpy())


def test_portfolio_var_combines_legs_on_common_dates(write_ohlc, loader):
    a, b = random_closes(100, 1), random_closes(70, 2)
    write_ohlc("CLZ26", a)
    write_ohlc("CLF27", b)
    result = portfolio_pnl_var([structure("spread", [("CLZ26", 1, "buy", 1), ("CLF27", -1, "buy", 1)])], 252, loader)
    assert result.observations == 69 < result.requested
    assert np.allclose(result.pnl.to_numpy(), (np.diff(a)[-69:] - np.diff(b)) * 1000)


def test_portfolio_var_skips_missing_symbols_with_a_warning(write_ohlc, loader):
    write_ohlc("CLZ26", random_closes(60, 1))
    result = portfolio_pnl_var([structure("s", [("CLZ26", 1, "buy", 1), ("CLF27", 1, "buy", 1)])], 30, loader)
    assert list(result.skipped) == ["CLF27"] and result.error is None
    assert any("CLF27" in w and "understated" in w for w in result.warnings)


def test_portfolio_var_errors(write_ohlc, loader):
    assert portfolio_pnl_var([], 30, loader).error == NOTHING_TO_ANALYZE
    closed = structure("c", [("CLZ26", 1, "buy", 1)], StructureStatus.CLOSED)
    assert portfolio_pnl_var([closed], 30, loader).error == NOTHING_TO_ANALYZE
    result = portfolio_pnl_var([structure("s", [("CLZ26", 1, "buy", 1)])], 30, loader)
    assert "No price data" in result.error and "CLZ26" in result.skipped
    with pytest.raises(ValueError):
        portfolio_pnl_var([], 0, loader)


def test_portfolio_var_warns_below_ten_observations_but_still_returns_a_result(write_ohlc, loader):
    write_ohlc("CLZ26", random_closes(RELIABLE_OBSERVATIONS - 3 + 1, 1))  # 7 differences
    result = portfolio_pnl_var([structure("s", [("CLZ26", 1, "buy", 1)])], 60, loader)
    assert result.error is None and result.observations == RELIABLE_OBSERVATIONS - 3
    assert "Insufficient data for reliable VaR" in result.warnings


# ---------- previous-day True Range and scenarios ----------


def ohlc_frame(rows):
    """rows: [(date, high, low, close)]"""
    index = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d, *_ in rows])
    return pd.DataFrame({"open": [r[3] for r in rows], "high": [r[1] for r in rows], "low": [r[2] for r in rows],
                         "close": [r[3] for r in rows]}, index=index)


def test_true_range_takes_the_largest_of_the_three_measures():
    frame = ohlc_frame([("2026-09-16", 76, 74, 75), ("2026-09-17", 77, 76.5, 76.8)])
    assert previous_day_true_range(frame, date(2026, 9, 18)) == pytest.approx(2.0)  # |high - prev close|
    gap_down = ohlc_frame([("2026-09-16", 76, 74, 75), ("2026-09-17", 72, 71, 71.5)])
    assert previous_day_true_range(gap_down, date(2026, 9, 18)) == pytest.approx(4.0)  # |low - prev close|
    inside = ohlc_frame([("2026-09-16", 76, 74, 75), ("2026-09-17", 78, 73, 75)])
    assert previous_day_true_range(inside, date(2026, 9, 18)) == pytest.approx(5.0)  # high - low


def test_true_range_ignores_todays_forming_candle_and_needs_two_days():
    frame = ohlc_frame([("2026-09-16", 76, 74, 75), ("2026-09-17", 77, 76.5, 76.8), ("2026-09-18", 99, 1, 50)])
    assert previous_day_true_range(frame, date(2026, 9, 18)) == pytest.approx(2.0)
    assert previous_day_true_range(frame.iloc[:1], date(2026, 9, 18)) is None
    assert previous_day_true_range(frame.iloc[1:], date(2026, 9, 18)) is None  # only one completed candle
    assert previous_day_true_range(None) is None


def test_load_previous_day_atr_reads_parquet_and_marks_missing_symbols(write_ohlc, loader):
    closes = [75.0, 76.0, 77.0]
    write_ohlc("CLZ26", closes, highs=[75.5, 76.5, 78.0], lows=[74.5, 75.5, 76.0])
    atr = load_previous_day_atr(["CLZ26", "CLF27"], loader, date(2026, 9, 19))
    assert atr["CLZ26"] == pytest.approx(2.0) and atr["CLF27"] is None
    assert loader.load_daily_ohlc("CLF27") is None


def test_build_scenarios_single_and_portfolio_wide_impacts():
    weights = {"CLZ26": 2000.0, "CLF27": -3000.0}
    rows = build_scenarios(weights, {"CLZ26": 1.5, "CLF27": 0.5})
    by_key = {(r["symbol"], r["sign"]): r for r in rows}
    assert by_key[("CLZ26", 1)]["pnl_impact"] == pytest.approx(3000.0)
    assert by_key[("CLZ26", -1)]["pnl_impact"] == pytest.approx(-3000.0)
    assert by_key[("CLF27", 1)]["pnl_impact"] == pytest.approx(-1500.0)
    up, down = [r for r in rows if r["kind"] == KIND_PORTFOLIO]
    assert up["pnl_impact"] == pytest.approx(3000.0 - 1500.0) and down["pnl_impact"] == pytest.approx(-1500.0)
    assert len(rows) == 6


def test_build_scenarios_excludes_instruments_without_atr_from_portfolio_wide():
    rows = build_scenarios({"CLZ26": 1000.0, "CLF27": 1000.0}, {"CLZ26": 2.0, "CLF27": None})
    missing = [r for r in rows if r["symbol"] == "CLF27"]
    assert all(r["shock"] is None and r["pnl_impact"] is None for r in missing)
    up = next(r for r in rows if r["kind"] == KIND_PORTFOLIO and r["sign"] == 1)
    assert up["pnl_impact"] == pytest.approx(2000.0) and up["included"] == ["CLZ26"] and up["excluded"] == ["CLF27"]
    none_usable = build_scenarios({"CLZ26": 1000.0}, {"CLZ26": None})
    assert [r["pnl_impact"] for r in none_usable if r["kind"] == KIND_PORTFOLIO] == [None, None]


# ---------- layout builders ----------


class Result:
    """Just enough of PortfolioVarResult for the builders."""


def make_result(pnl, requested=60, warnings=None):
    var, cutoff = var_from_pnl_series(pd.Series(pnl))
    result = Result()
    result.pnl, result.var, result.cutoff = pd.Series(pnl), var, cutoff
    result.observations, result.requested, result.error = len(pnl), requested, None
    result.warnings = warnings or []
    return result


def test_var_cards_show_dollars_lookback_and_observation_count():
    result = make_result(np.random.default_rng(1).normal(0, 1000, 60), requested=60)
    c95, c99, lookback, obs = build_var_cards(result, 60)
    assert c95.startswith("$") and c99.startswith("$") and lookback == "60d" and obs == "60"
    short = make_result(np.random.default_rng(1).normal(0, 1000, 20), requested=90)
    assert build_var_cards(short, 90)[3] == "20 (of 90 requested)"
    assert build_var_cards(None, 60) == ("—", "—", "60d", "—")


def test_histogram_colors_zero_edge_vlines_and_title():
    pnl = np.random.default_rng(3).normal(0, 1000, 100)
    figure = build_histogram(make_result(pnl), 60)
    bar = figure.data[0]
    assert figure.layout.title.text == "Portfolio PnL Distribution (60d Historical)"
    assert sum(bar.y) == 100
    colors = dict(zip(bar.x, bar.marker.color))
    assert all(c == (COLORS["ACCENT_RED"] if x < 0 else COLORS["ACCENT_GREEN"]) for x, c in colors.items())
    assert COLORS["ACCENT_RED"] in colors.values() and COLORS["ACCENT_GREEN"] in colors.values()
    lines = figure.layout.shapes
    assert len(lines) == 2 and all(line.line.dash == "dash" for line in lines)
    assert sorted(a.text.split(":")[0] for a in figure.layout.annotations) == ["VaR 95%", "VaR 99%"]
    assert figure.layout.xaxis.title.text == "PnL ($)" and figure.layout.yaxis.title.text == "Frequency"


def test_histogram_handles_constant_and_missing_series():
    assert len(build_histogram(make_result([500.0] * 12), 30).data[0].x) >= 1
    error = Result()
    error.error = "No price data for any open leg — cannot compute VaR"
    assert build_histogram(error, 60).layout.annotations[0].text == error.error
    assert build_histogram(None, 60).layout.annotations[0].text == NOTHING_TO_ANALYZE


def test_var_warning_banner_lists_each_warning():
    banner = build_var_warnings(make_result([1.0, -2.0, 3.0], warnings=["Skipped: CLF27", "Insufficient data for reliable VaR"]))
    assert [a.children for a in banner] == ["⚠️ Skipped: CLF27", "⚠️ Insufficient data for reliable VaR"]
    assert all(a.color == "warning" for a in banner) and build_var_warnings(None) == []


def test_scenario_table_rows_labels_and_unavailable_atr():
    rows = build_scenarios({"CLZ26": 1000.0, "CLF27": 1000.0}, {"CLZ26": 1.25, "CLF27": None})
    table = build_scenario_table_rows(rows)
    assert [r["scenario"] for r in table] == ["CLF27 +ATR", "CLF27 −ATR", "CLZ26 +ATR", "CLZ26 −ATR",
                                              "All Instruments +ATR", "All Instruments −ATR"]
    assert table[0]["shock"] == ATR_UNAVAILABLE and table[0]["pnl_impact"] is None
    assert table[2] == {"scenario": "CLZ26 +ATR", "shock": "1.25", "pnl_impact": 1250.0, "direction": "▲ Up"}
    assert table[3]["direction"] == "▼ Down" and table[3]["pnl_impact"] == -1250.0
    assert "CLF27" in build_scenario_note(rows) and build_scenario_note(build_scenarios({"A": 1.0}, {"A": 1.0})) == ""


def find(node, component_id):
    if getattr(node, "id", None) == component_id:
        return node
    children = getattr(node, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and not isinstance(child, str):
            found = find(child, component_id)
            if found is not None:
                return found
    return None


def test_layout_components_defaults_and_table_styling():
    layout = var_scenario_layout()
    for component_id in ("var-lookback", "var-card-95", "var-card-99", "var-card-lookback", "var-card-observations",
                         "var-histogram", "var-warnings", "scenario-refresh-btn", "scenario-table", "scenario-message"):
        assert find(layout, component_id) is not None, component_id
    assert find(layout, "var-lookback").value == "60"
    assert [o["value"] for o in find(layout, "var-lookback").options] == ["30", "60", "90", "252"]
    text = str(layout)
    assert "Scenario Analysis — ATR-Based Shocks" in text and "Shock size = previous day's True Range per instrument" in text
    table = find(layout, "scenario-table")
    assert [c["name"] for c in table.columns] == ["Scenario", "Shock Size (pts)", "PnL Impact ($)", "Direction"]
    colors = {c["if"]["filter_query"]: c["color"] for c in table.style_data_conditional if c["if"].get("column_id") == "pnl_impact"}
    assert colors == {"{pnl_impact} > 0": COLORS["ACCENT_GREEN"], "{pnl_impact} < 0": COLORS["ACCENT_RED"]}


# ---------- callbacks ----------


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture
def env(repo, loader, monkeypatch):
    monkeypatch.setattr(vc, "container", Container(repository=repo, data_loader=loader))


def save(repo, s):
    for leg in s.legs:
        repo.save_contract(leg.contract)
    repo.save_structure(s)


def test_callbacks_only_run_on_the_var_tab(env):
    with pytest.raises(PreventUpdate):
        vc.update_var_section("/structures", "60")
    with pytest.raises(PreventUpdate):
        vc.update_scenarios("/structures", None)


def test_placeholders_without_open_structures(env):
    c95, c99, lookback, obs, figure, warnings = vc.update_var_section("/var-scenario", "60")
    assert (c95, c99, obs) == ("—", "—", "—") and lookback == "60d"
    assert figure.layout.annotations[0].text == NOTHING_TO_ANALYZE
    assert vc.update_scenarios("/var-scenario", None) == ([], NOTHING_TO_ANALYZE, "")


def test_var_and_scenarios_from_open_structures(env, repo, write_ohlc):
    closes = random_closes(100, 5)
    write_ohlc("CLZ26", closes, end="2026-09-18")
    save(repo, structure("Long", [("CLZ26", 1, "buy", 2)]))
    save(repo, structure("Missing", [("CLF27", 1, "sell", 1)]))
    c95, c99, lookback, obs, figure, warnings = vc.update_var_section("/var-scenario", "30")
    expected = -np.percentile(np.diff(closes)[-30:] * 2000, 5)
    assert c95 == f"${expected:,.0f}" and obs == "30" and lookback == "30d"
    assert "30d Historical" in figure.layout.title.text
    assert "CLF27" in warnings[0].children

    table, message, note = vc.update_scenarios("/var-scenario", 1)
    by_name = {r["scenario"]: r for r in table}
    assert message == "" and by_name["CLF27 +ATR"]["shock"] == ATR_UNAVAILABLE
    z_up = by_name["CLZ26 +ATR"]
    assert float(z_up["shock"]) > 0 and z_up["pnl_impact"] == pytest.approx(float(z_up["shock"]) * 2000, rel=1e-2)
    assert by_name["All Instruments +ATR"]["pnl_impact"] == pytest.approx(z_up["pnl_impact"])
    assert "CLF27" in note


def test_refresh_rereads_the_previous_day_true_range(env, repo, write_ohlc):
    write_ohlc("CLZ26", [75.0, 76.0, 77.0], highs=[75.5, 76.5, 78.0], lows=[74.5, 75.5, 76.0], end="2026-09-18")
    save(repo, structure("Long", [("CLZ26", 1, "buy", 1)]))
    first = {r["scenario"]: r["shock"] for r in vc.update_scenarios("/var-scenario", 0)[0]}
    write_ohlc("CLZ26", [75.0, 76.0, 77.0], highs=[75.5, 76.5, 80.0], lows=[74.5, 75.5, 76.0], end="2026-09-18")
    second = {r["scenario"]: r["shock"] for r in vc.update_scenarios("/var-scenario", 1)[0]}
    assert first["CLZ26 +ATR"] == "2.00" and second["CLZ26 +ATR"] == "4.00"
