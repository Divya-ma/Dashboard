"""Tests for core.trade_idea, the Trade Idea Analyzer layout builders and callbacks."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from dash.exceptions import PreventUpdate

from core.data_loader import DataLoader
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from core.trade_idea import (
    DOLLARS_PER_POINT_PER_LOT,
    TradeIdea,
    analyze_trade_idea,
    correlation_label,
    historical_context,
    idea_daily_series,
    parse_idea_inputs,
)
from db.repository import Repository
from ui.callbacks import idea_callbacks as ic
from ui.container import Container
from ui.layouts.idea_tab import (
    PLACEHOLDER_TEXT,
    build_correlation_section,
    build_pnl_chart,
    build_risk_reward,
    trade_analyzer_layout,
)
from ui.layouts.shell import COLORS


class NoBackfillAdapter:
    def backfill_symbol(self, *args, **kwargs):
        raise AssertionError("the analyzer must not backfill")


@pytest.fixture
def loader(tmp_path):
    return DataLoader(str(tmp_path), NoBackfillAdapter())


@pytest.fixture
def make_data(tmp_path, create_synthetic_parquet):
    def _make(**closes_by_symbol):
        for symbol, closes in closes_by_symbol.items():
            create_synthetic_parquet(tmp_path, symbol, closes=np.asarray(closes, dtype=float))

    return _make


def random_closes(n, seed):
    return 75 + np.cumsum(np.random.default_rng(seed).normal(0, 1.5, n))


def leg(symbol="CLZ26", direction="buy", lots=1):
    return {"symbol": symbol, "direction": direction, "lots": lots}


def idea(legs=None, entry=75.0, stop=74.0, target=77.5):
    return TradeIdea(legs=legs or [leg()], entry=entry, stop=stop, target=target)


def parse(structure_type="outright", symbols=("CLZ26", None, None), directions=("buy", "buy", "buy"),
          lots=(1, 1, 1), entry=75.0, stop=74.0, target=77.0):
    return parse_idea_inputs(structure_type, list(symbols), list(directions), list(lots), entry, stop, target)


# ---------- input validation ----------


def test_parse_valid_outright_and_spread():
    parsed, errors, warnings = parse()
    assert errors == [] and warnings == [] and parsed.legs == [leg()]
    parsed, errors, warnings = parse("spread", ("clz26", "CLF27", None), ("buy", "sell", "buy"), (2, 2, 1))
    assert errors == [] and [l["symbol"] for l in parsed.legs] == ["CLZ26", "CLF27"]
    assert parsed.total_lots == 4 and parsed.unit_lots == 2 and parsed.side == "buy"


def test_parse_normalizes_symbols_and_only_reads_needed_legs():
    parsed, _, _ = parse("spread", ("clz25-clh26", "clf27", "ignored ???"), lots=(1, 1, "junk"))
    assert parsed.legs[0]["symbol"] == "CLZ25-H26" and len(parsed.legs) == 2


def test_parse_blocking_errors():
    _, errors, _ = parse(symbols=(" ", None, None), lots=(0, 1, 1))
    assert "Enter a symbol for leg 1." in errors and "Leg 1 lots must be a positive whole number." in errors
    _, errors, _ = parse(lots=(1.5, 1, 1))
    assert "Leg 1 lots must be a positive whole number." in errors
    _, errors, _ = parse(entry=None, stop="", target="x")
    assert {"Entry price is required.", "Stop price is required.", "Target price is required."} <= set(errors)
    assert parse(stop=75.0)[1] == ["Stop cannot equal entry price"]
    assert parse(target=75.0)[1] == ["Target cannot equal entry price"]
    assert parse(stop=76.0, target=76.0)[1] == ["Stop and target cannot be the same price"]
    assert parse("hedge")[1] == ["Choose a structure type."]


def test_parse_illogical_levels_warn_but_do_not_block():
    parsed, errors, warnings = parse(stop=76.0, target=74.0)  # buy with stop above and target below
    assert parsed is not None and errors == []
    assert len(warnings) == 2 and "Illogical stop for a buy" in warnings[0] and "Illogical target" in warnings[1]
    parsed, _, warnings = parse(directions=("sell", "buy", "buy"), stop=74.0, target=77.0)  # sell: reversed
    assert len(warnings) == 2 and "for a sell" in warnings[0]
    assert parse(directions=("sell", "buy", "buy"), stop=76.0, target=73.0)[2] == []


# ---------- series and statistics ----------


def test_idea_series_is_signed_lot_weighted_sum(make_data, loader):
    a, b = random_closes(60, 1), random_closes(60, 2)
    make_data(CLZ26=a, CLF27=b)
    series = idea_daily_series(idea([leg("CLZ26", "buy", 2), leg("CLF27", "sell", 2)]), loader)
    assert np.allclose(series.to_numpy(), (np.diff(a) - np.diff(b)) * 2)


def test_historical_context_values_and_threshold_percentages():
    series = pd.Series([1.0, -3.0, 2.0, -0.5, 4.0, -1.5, 0.5, 3.0, -2.5, 0.0])
    stats = historical_context(series, risk=2.0, reward=2.5)
    assert stats["mean"] == pytest.approx(0.3) and stats["std"] == pytest.approx(series.std(ddof=1))
    assert (stats["max_gain"], stats["max_loss"]) == (4.0, -3.0)
    assert stats["pct_beyond_risk"] == pytest.approx(20.0)  # -3.0 and -2.5 fall below -2.0
    assert stats["pct_beyond_reward"] == pytest.approx(20.0)  # 4.0 and 3.0 exceed 2.5


@pytest.mark.parametrize("value,label", [(0.71, "high"), (-0.9, "high"), (0.7, "moderate"), (-0.4, "moderate"),
                                         (0.39, "low"), (0.0, "low"), (None, "n/a"), (float("nan"), "n/a")])
def test_correlation_label_bands(value, label):
    assert correlation_label(value) == label


# ---------- analyze_trade_idea ----------


def test_analysis_risk_reward_dollars_and_unit_series(make_data, loader):
    a, b = random_closes(80, 1), random_closes(80, 2)
    make_data(CLZ26=a, CLF27=b)
    spread = idea([leg("CLZ26", "buy", 2), leg("CLF27", "sell", 2)], entry=1.0, stop=0.5, target=2.25)
    result = analyze_trade_idea(spread, 30, [], loader)
    assert result.errors == []
    assert (result.risk, result.reward, result.rr_ratio) == (0.5, 1.25, 2.5)
    assert result.dollar_risk == 0.5 * 4 * DOLLARS_PER_POINT_PER_LOT
    assert result.dollar_reward == 1.25 * 4 * DOLLARS_PER_POINT_PER_LOT
    expected_unit = (np.diff(a) - np.diff(b))[-30:]  # lots 2/2 -> gcd 2 -> per-unit series
    assert result.observations == 30 and np.allclose(result.unit_series.to_numpy(), expected_unit)
    assert result.stats["max_gain"] == pytest.approx(expected_unit.max())


def test_analysis_missing_symbols_are_reported(make_data, loader):
    make_data(CLZ26=random_closes(40, 1))
    result = analyze_trade_idea(idea([leg("CLZ26"), leg("CLF27"), leg("NOPE")]), 30, [], loader)
    assert result.errors == ["No historical data for CLF27", "No historical data for NOPE"]
    assert result.unit_series is None


def test_analysis_short_history_and_too_little_history(make_data, loader):
    make_data(CLZ26=random_closes(16, 1))
    result = analyze_trade_idea(idea(), 90, [], loader)
    assert result.errors == [] and result.observations == 15 and result.requested == 90
    make_data(CLF27=random_closes(4, 1))
    assert "Insufficient data for CLF27" in analyze_trade_idea(idea([leg("CLF27")]), 30, [], loader).errors[0]


def test_analysis_legs_without_overlapping_days(tmp_path, create_synthetic_parquet, loader):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=random_closes(30, 1), end_date=date(2026, 3, 30))
    create_synthetic_parquet(tmp_path, "CLF27", closes=random_closes(30, 2), end_date=date(2026, 9, 30))
    result = analyze_trade_idea(idea([leg("CLZ26"), leg("CLF27", "sell")]), 30, [], loader)
    assert "common trading days" in result.errors[0]


def open_structure(name, legs):
    built = [Leg(contract=Contract(product="CL", contract_month=12, contract_year=2026, symbol=s, multiplier=1000,
                                   tick_size=0.01, tick_value=10), ratio=r, direction=d, lots=lots, entry_price=1.0)
             for s, r, d, lots in legs]
    return Structure(name=name, structure_type=StructureType.CUSTOM, products=["CL"], legs=built, status=StructureStatus.OPEN)


def test_correlation_with_open_structures_and_skipped_ones(make_data, loader):
    a, b = random_closes(80, 1), random_closes(80, 2)
    make_data(CLZ26=a, CLF27=b)
    same = open_structure("Same as idea", [("CLZ26", 1, "buy", 3)])
    opposite = open_structure("Opposite", [("CLZ26", 1, "sell", 1)])
    unrelated = open_structure("Other leg", [("CLF27", 1, "buy", 1)])
    broken = open_structure("No data", [("CLH27", 1, "buy", 1)])
    result = analyze_trade_idea(idea(), 30, [same, opposite, unrelated, broken], loader)
    by_name = {c["name"]: c["correlation"] for c in result.correlations}
    assert by_name["Same as idea"] == pytest.approx(1.0) and by_name["Opposite"] == pytest.approx(-1.0)
    assert abs(by_name["Other leg"]) < 0.9
    assert list(result.skipped_structures) == ["No data"]
    assert analyze_trade_idea(idea(), 30, [], loader).correlations == []


# ---------- layout builders ----------


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


def make_analysis(make_data, loader, open_structures=(), **idea_kwargs):
    make_data(CLZ26=random_closes(80, 1))
    return analyze_trade_idea(idea(**idea_kwargs), 30, list(open_structures), loader)


def test_risk_reward_cards(make_data, loader):
    analysis = make_analysis(make_data, loader, stop=74.0, target=77.5, entry=75.0)
    text = str(build_risk_reward(analysis))
    for expected in ("Risk (pts)", "1.00", "Reward (pts)", "2.50", "2.5 : 1", "Est. $ Risk", "$10", "Est. $ Reward", "$25"):
        assert expected in text, expected


def test_correlation_section_colors_and_messages():
    class Analysis:
        correlations = [{"name": "High", "correlation": 0.85}, {"name": "Mid", "correlation": -0.5},
                        {"name": "Low", "correlation": 0.1}, {"name": "Flat", "correlation": None}]
        skipped_structures = {"Broken": "no local price data for CLH27"}

    text = str(build_correlation_section(Analysis, True))
    for name, color in (("+0.85", COLORS["ACCENT_RED"]), ("-0.50", COLORS["ACCENT_YELLOW"]), ("+0.10", COLORS["ACCENT_GREEN"])):
        assert f"children='{name}'" in text and color in text
    assert "n/a" in text and "Skipped (missing data): Broken" in text
    assert "No open positions to correlate against" in str(build_correlation_section(Analysis, False))


def test_pnl_chart_lines_title_and_series(make_data, loader):
    analysis = make_analysis(make_data, loader)
    figure = build_pnl_chart(analysis, 30)
    assert figure.layout.title.text == "Hypothetical Historical PnL (30d)"
    assert np.allclose(figure.data[0].y, analysis.unit_series.cumsum().to_numpy())
    levels = sorted(shape.y0 for shape in figure.layout.shapes)
    assert levels == [-analysis.risk, 0, analysis.reward] and all(s.line.dash == "dash" for s in figure.layout.shapes)
    assert figure.layout.yaxis.title.text == "Cumulative PnL (pts)"


def test_layout_defaults_and_hidden_leg_rows():
    layout = trade_analyzer_layout()
    for component_id in ("idea-structure-type", "idea-leg-1-symbol", "idea-leg-3-direction", "idea-leg-2-lots", "idea-entry",
                         "idea-stop", "idea-target", "idea-lookback", "idea-analyze-btn", "idea-output", "idea-open-structures"):
        assert find(layout, component_id) is not None, component_id
    assert find(layout, "idea-lookback").value == "30" and find(layout, "idea-structure-type").value == "outright"
    assert find(layout, "idea-leg-row-1").style == {"marginBottom": "16px"}
    assert find(layout, "idea-leg-row-2").style["display"] == "none" and find(layout, "idea-leg-row-3").style["display"] == "none"
    assert PLACEHOLDER_TEXT in str(find(layout, "idea-output"))


# ---------- callbacks ----------


def test_toggle_leg_inputs_shows_rows_and_sets_default_pattern():
    out = ic.toggle_leg_inputs("fly")
    assert out[:3] == ({}, {}, {}) and out[3:6] == ("buy", "sell", "buy") and out[6:] == (1, 2, 1)
    out = ic.toggle_leg_inputs("spread")
    assert out[:3] == ({}, {}, {"display": "none"}) and out[3:5] == ("buy", "sell") and out[6:8] == (1, 1)
    assert ic.toggle_leg_inputs("outright")[1:3] == ({"display": "none"}, {"display": "none"})


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture
def env(repo, loader, monkeypatch):
    monkeypatch.setattr(ic, "container", Container(repository=repo, data_loader=loader))


def save(repo, s):
    for l in s.legs:
        repo.save_contract(l.contract)
    repo.save_structure(s)


def run_analyze(**overrides):
    args = dict(n_clicks=1, structure_type="outright", symbol_1="CLZ26", direction_1="buy", lots_1=1,
                symbol_2=None, direction_2="buy", lots_2=1, symbol_3=None, direction_3="buy", lots_3=1,
                entry=75.0, stop=74.0, target=77.0, lookback="30", open_structures=[])
    args.update(overrides)
    return ic.analyze(**args)


def test_load_open_structures_on_tab_open(env, repo):
    save(repo, open_structure("Open", [("CLZ26", 1, "buy", 2)]))
    save(repo, open_structure("Closed", [("CLF27", 1, "buy", 0)]).model_copy(update={"status": StructureStatus.CLOSED}))
    stored = ic.load_open_structures("/trade-analyzer")
    assert [s["name"] for s in stored] == ["Open"] and stored[0]["structure_id"]
    with pytest.raises(PreventUpdate):
        ic.load_open_structures("/structures")


def test_analyze_validation_errors_show_inline(env):
    text = str(run_analyze(stop=75.0))
    assert "Stop cannot equal entry price" in text
    assert "Target cannot equal entry price" in str(run_analyze(target=75.0))
    assert "Enter a symbol for leg 1." in str(run_analyze(symbol_1=""))
    with pytest.raises(PreventUpdate):
        run_analyze(n_clicks=None)


def test_analyze_missing_symbol_error(env):
    assert "No historical data for CLZ26" in str(run_analyze())


def test_analyze_full_output_with_open_structure_correlation(env, repo, make_data):
    make_data(CLZ26=random_closes(80, 1))
    same = open_structure("Held outright", [("CLZ26", 1, "buy", 2)])
    save(repo, same)
    output = run_analyze(open_structures=[{"structure_id": same.structure_id, "name": same.name}])
    text = str(output)
    for section in ("A. Risk / Reward Summary", "B. Historical Context", "C. Correlation with Portfolio", "D. Price Chart"):
        assert section in text
    assert "Held outright" in text and "+1.00" in text and "2.0 : 1" in text
    assert "Illogical" not in text


def test_analyze_shows_warning_but_still_computes(env, make_data):
    make_data(CLZ26=random_closes(80, 1))
    text = str(run_analyze(stop=76.0, target=74.0))
    assert "Illogical stop for a buy" in text and "A. Risk / Reward Summary" in text
    assert "No open positions to correlate against" in text
