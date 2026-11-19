"""core.trade_analysis (replaying earlier years, stop/target outcomes, portfolio effect) and the Trade Analyzer page callbacks."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from adapters.base import MONTH_TO_LETTER
from core.curve_calendar import contract_expiry
from core.trade_analysis import (
    Paths, TradeAnalysisError, TradeSpec, analyse, context_weights, front_outright, outright_weights, point_value, portfolio_effect,
    resolve, shift_years,
)
from ui.callbacks import trade_analyzer_callbacks as cb
from ui.container import Container

TODAY = date(2026, 11, 17)
SPREAD = (("BRNJ27", 1), ("BRNK27", -1))
FLY = (("BRNJ27", 1), ("BRNK27", -2), ("BRNM27", 1))


class FakeSettlements:
    """Random-walk BRN outrights J/K of 2019..2027, each settling until its rule-based expiry, plus fronts for the context."""

    def __init__(self, seed=3, years=range(2019, 2028)):
        rng = np.random.default_rng(seed)
        days = pd.bdate_range("2018-06-01", "2026-11-17")
        self._series = {}
        base = pd.Series(70 + np.cumsum(rng.normal(0, 0.8, len(days))), index=days)
        for year in years:
            for month in (4, 5):
                expiry = contract_expiry("BRN", month, year)
                alive = days[(days <= pd.Timestamp(expiry)) & (days >= pd.Timestamp(expiry) - pd.Timedelta(days=400))]
                carry = (month - 4) * -0.4 + np.cumsum(rng.normal(0, 0.05, len(alive)))
                s = base.reindex(alive) + carry
                s.index = pd.DatetimeIndex(alive).tz_localize("UTC")
                self._series[f"BRN{MONTH_TO_LETTER[month]}{year % 100:02d}"] = s

    def closes(self, symbol):
        return self._series.get(symbol), False

    def front_prices(self, product):
        days = pd.bdate_range("2018-06-01", "2026-11-17")
        rng = np.random.default_rng({"CL": 1, "BRN": 2, "G": 3, "CLBZ": 4}.get(product, 5))
        level = {"CL": 68, "BRN": 72, "G": 750, "CLBZ": -4}[product]
        return pd.Series(level + np.cumsum(rng.normal(0, 0.3, len(days))), index=days)


def test_outright_weights_and_year_shift():
    assert outright_weights(SPREAD) == {"BRNJ27": 1.0, "BRNK27": -1.0}
    assert outright_weights(FLY) == {"BRNJ27": 1.0, "BRNK27": -2.0, "BRNM27": 1.0}
    assert outright_weights((("BRNJ27-K27", 1),)) == {"BRNJ27": 1.0, "BRNK27": -1.0}  # a leg that is itself a spread
    assert outright_weights((("BRNJ27-K27", 2), ("BRNK27", 2))) == {"BRNJ27": 2.0}  # legs that partly cancel
    assert TradeSpec(SPREAD, 1, 5, 1e4).symbol == "BRNJ27-K27"
    assert shift_years("BRNJ27", 2) == "BRNJ25"
    assert front_outright({"BRNK27": -1.0, "BRNJ27": 1.0}) == "BRNJ27"
    assert point_value({"BRNJ27": 1.0}) == 1000.0
    with pytest.raises(TradeAnalysisError):
        outright_weights((("NOTASYMBOL", 1),))
    with pytest.raises(TradeAnalysisError):
        outright_weights((("BRNJ27", 1), ("BRNJ27", -1)))


def test_resolve_target_before_stop():
    profit = np.array([
        [0.5, 1.0, 1.5, np.nan],  # reaches +1.0 on day 2 first: win
        [-1.0, 2.0, 2.0, 2.0],  # hits the stop on day 1: loss
        [0.2, 0.3, 0.1, 0.2],  # neither: unresolved
        [1.0, -1.0, 1.0, 1.0],  # target on day 1 first: win
        [np.nan, np.nan, np.nan, np.nan],  # no data at all: unresolved
    ])
    won, lost = resolve(profit, stop=1.0, target=1.0)
    assert list(won) == [True, False, False, True, False]
    assert list(lost) == [False, True, False, False, False]


def test_a_day_that_reaches_the_stop_and_the_target_counts_as_a_loss():
    # one daily value cannot reach both, but the stop on the same day index as the target is a tie: the stop wins it
    won, lost = resolve(np.array([[1.0, 1.0]]), stop=-1.0, target=1.0)  # stop level +1 and target level 1 are both met on day 1
    assert list(lost) == [True] and list(won) == [False]


def test_context_weights_favour_similar_days():
    ctx = pd.DataFrame({"cl": [60.0, 61.0, 100.0, 101.0]})
    paths = Paths(np.zeros((4, 3)), np.array([2019, 2020, 2021, 2022]), [None] * 4, ctx)
    weight, effective = context_weights(paths, pd.Series({"cl": 60.5}))
    assert weight[0] > weight[2] * 5 and 1 < effective < 4
    flat, eff = context_weights(paths, None)
    assert list(flat) == [1, 1, 1, 1] and eff == 4


@pytest.fixture(scope="module")
def settlements():
    return FakeSettlements()


def test_analyse_gives_a_probability_a_stop_within_the_risk_and_the_shares_add_up(settlements):
    spec = TradeSpec(SPREAD, -1, 5, risk_trade=10000, horizon=40)
    result = analyse(settlements, spec, TODAY)
    assert result.max_stop == pytest.approx(10000 / (5 * 1000))
    assert result.n_years >= 3 and result.n_paths >= 20 and result.days_to_expiry > 0
    head = result.headline
    assert 0 <= head.p_win <= 1
    assert head.stop <= result.max_stop + 1e-9
    assert head.win_share + head.loss_share + head.unresolved_share == pytest.approx(1.0)
    assert result.grid[0].expectancy >= result.grid[-1].expectancy
    assert set(result.context_now) == {"cl", "brn", "wti_brent", "go", "crack"} and set(result.fan) == {"p10", "p50", "p90"}
    assert sum(y["days"] for y in result.years) == result.n_paths


def test_a_tight_risk_limit_caps_every_stop_and_my_own_pair_is_scored(settlements):
    spec = TradeSpec(SPREAD, 1, 10, risk_trade=2000, horizon=30, stop=0.1, target=0.3)
    result = analyse(settlements, spec, TODAY)
    assert result.max_stop == pytest.approx(0.2)
    assert all(g.stop <= 0.2 + 1e-9 for g in result.grid)
    assert (result.headline.stop, result.headline.target) == (0.1, 0.3)
    wide = analyse(settlements, TradeSpec(SPREAD, 1, 10, 2000, horizon=30, stop=0.5, target=1.0), TODAY)
    assert any("risks more than your limit" in n for n in wide.notes)


def test_too_little_history_or_bad_inputs_are_refused(settlements):
    with pytest.raises(TradeAnalysisError):
        analyse(FakeSettlements(years=[2025]), TradeSpec(SPREAD, 1, 5, 1e4), TODAY)
    bad_specs = (
        TradeSpec(SPREAD, 1, 0, 1e4), TradeSpec(SPREAD, 1, 5, -1), TradeSpec(SPREAD, 2, 5, 1e4), TradeSpec(SPREAD, 1, 5, 1e4, horizon=999),
    )
    for bad in bad_specs:
        with pytest.raises(TradeAnalysisError):
            analyse(settlements, bad, TODAY)


def test_portfolio_effect_without_positions_says_so():
    effect = portfolio_effect(TradeSpec(SPREAD, 1, 5, 1e4), [], data_loader=None)
    assert effect.open_structures == 0 and effect.correlation is None and "nothing to correlate" in effect.message


# ---------------- page callbacks ----------------


SYMS, RATIOS = ["BRNJ27", "BRNK27"], [1, -1]


def test_build_spec_reads_the_leg_rows_like_the_structure_builder():
    spec = cb.build_spec("spread", [" brnj27 ", "brnk27"], RATIOS, -1, 5, None, 10000, 60, None, None, None)
    assert spec.legs == (("BRNJ27", 1), ("BRNK27", -1)) and spec.side == -1 and spec.horizon == 60 and spec.window == 7 and spec.entry is None
    fly = cb.build_spec("fly", ["BRNJ27-K27", "BRNK27-M27", "BRNM27"], [1, -1, 1], 1, 2, None, 5000, 30, None, None, None)
    assert len(fly.legs) == 3  # a leg may itself be a spread
    custom = cb.build_spec("custom", ["BRNJ27"], [3], 1, 1, None, 5000, 30, None, None, None)
    assert custom.legs == (("BRNJ27", 3),)
    bad_forms = (
        ("spread", ["BRNJ27"], [1], 1, 5, None, 1, 60, None, None, None),  # a spread needs two legs
        ("spread", ["BRNJ27", ""], RATIOS, 1, 5, None, 1, 60, None, None, None),  # a leg has no symbol
        ("spread", ["BRNJ27", "NOPE"], RATIOS, 1, 5, None, 1, 60, None, None, None),
        ("spread", SYMS, [1, 0], 1, 5, None, 1, 60, None, None, None),
        ("spread", SYMS, [1, 1.5], 1, 5, None, 1, 60, None, None, None),
        ("spread", SYMS, [1, None], 1, 5, None, 1, 60, None, None, None),
        ("spread", SYMS, RATIOS, 0, 5, None, 1, 60, None, None, None),
        ("spread", SYMS, RATIOS, 1, 0, None, 1, 60, None, None, None),
        ("spread", SYMS, RATIOS, 1, 5, None, 1, 2.5, None, None, None),
        ("spread", SYMS, RATIOS, 1, 5, None, 1, 999, None, None, None),
        ("spread", SYMS, RATIOS, 1, 5, None, 1, 60, None, 1, None),
        ("nonsense", SYMS, RATIOS, 1, 5, None, 1, 60, None, None, None),
    )
    for bad in bad_forms:
        with pytest.raises(ValueError):
            cb.build_spec(*bad)


def test_leg_rows_follow_the_template_and_custom_can_grow_and_shrink(monkeypatch):
    class Ctx:
        triggered_id = "ta-template"
        triggered = [{"prop_id": "ta-template.value", "value": "fly"}]

    monkeypatch.setattr(cb, "callback_context", Ctx())
    rows = cb.update_leg_rows("fly", None, [], [], [])
    assert len(rows) == 3 and "ta-remove" not in str(rows)  # a fly has fixed legs
    rows = cb.update_leg_rows("custom", None, [], ["BRNJ27"], [1])
    assert len(rows) == 1 and "ta-remove" in str(rows)
    Ctx.triggered_id, Ctx.triggered = "ta-add-leg", [{"prop_id": "ta-add-leg.n_clicks", "value": 1}]
    assert len(cb.update_leg_rows("custom", 1, [], ["BRNJ27"], [1])) == 2
    Ctx.triggered_id, Ctx.triggered = {"type": "ta-remove", "index": 0}, [{"prop_id": "x.n_clicks", "value": 1}]
    assert len(cb.update_leg_rows("custom", 1, [1], ["BRNJ27", "BRNK27"], [1, -1])) == 1
    Ctx.triggered = [{"prop_id": "x.n_clicks", "value": None}]
    with pytest.raises(cb.PreventUpdate):
        cb.update_leg_rows("custom", 1, [None], ["BRNJ27", "BRNK27"], [1, -1])


def test_run_analysis_fills_every_panel_and_reports_errors(monkeypatch, settlements):
    monkeypatch.setattr(cb, "container", Container(settlements=settlements))
    monkeypatch.setattr(cb, "_today", lambda: TODAY)
    out = cb.run_analysis(1, "spread", SYMS, RATIOS, -1, 5, None, 10000, 40, None, None, None)
    assert len(out) == 8 and "comparable days" in str(out[0]) and "met before the stop" in str(out[1]) and "ta-grid-table" in str(out[3])
    bad = cb.run_analysis(1, "spread", ["", ""], RATIOS, -1, 5, None, 10000, 40, None, None, None)
    assert "Type the symbol of leg 1" in str(bad[0]) and len(bad) == 8
    with pytest.raises(cb.PreventUpdate):
        cb.run_analysis(None, "spread", SYMS, RATIOS, 1, 1, None, 1, 1, None, None, None)


def test_risk_per_trade_is_saved_and_defaults_from_the_settings(tmp_db_path):
    from core.trade_analysis import load_risk_per_trade, save_risk_per_trade
    from db.repository import Repository

    repo = Repository(str(tmp_db_path))
    assert load_risk_per_trade(repo) > 0
    save_risk_per_trade(repo, 12000)
    assert load_risk_per_trade(repo) == 12000.0


def test_front_prices_pick_the_earliest_unexpired_delivery_not_a_far_contract():
    """Every live contract ends on the same stored day; the front must still be the nearest delivery month."""
    from core.settlement_prices import SettlementPrices

    days = pd.bdate_range("2026-09-01", "2026-11-17")
    rows = []
    for symbol, level in (("COF27", 80.0), ("COG27", 79.0), ("COZ28", 68.0), ("COZ26", 86.0)):
        for day in days:
            rows.append({"contract": symbol, "date": day, "oi": 1.0, "close": level, "volume": 1.0})

    class Store:
        version = "v1"

        def frame(self, product):
            return pd.DataFrame(rows)

    front = SettlementPrices(Store()).front_prices("BRN")
    assert not front.empty
    assert front.iloc[-1] == 86.0 or front.iloc[-1] == 80.0  # a near contract, never the Dec 2028 level of 68
    assert front.iloc[-1] != 68.0
