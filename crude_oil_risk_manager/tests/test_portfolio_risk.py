"""Portfolio stop risk, sqrt(r C r'), on Home: the formula, the stop risk of a structure, the edge cases and the card."""

import numpy as np
import pandas as pd
import pytest

from core import portfolio_risk as pr
from core.correlation import WatchlistResult
from core.data_loader import DataLoader
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from core.portfolio_risk import PortfolioRisk, combine, structure_stop_risk
from core.trade_entry import enter_trade
from ui.callbacks import home_callbacks as hc
from ui.callbacks import shell_callbacks as sc
from ui.container import Container
from ui.layouts.home import home_layout

LIVE = {"CLZ26": 75.0, "CLF27": 75.0, "CLG27": 75.0, "CLH27": 75.0}


def contract(symbol):
    month = "FGHJKMNQUVXZ".index(symbol[2]) + 1
    return Contract(product="CL", contract_month=month, contract_year=2000 + int(symbol[3:]), symbol=symbol, multiplier=1000, tick_size=0.01, tick_value=10)


def position(name, symbol="CLZ26", price=75.0, lots=2, stop=70.0, direction="buy", symbols=None, ratios=None):
    """An open structure entered through the real engine; returns (structure, its trades)."""
    symbols, ratios = symbols or (symbol,), ratios or (1,)
    shell = Structure(name=name, structure_type=StructureType.SPREAD, products=["CL"], status=StructureStatus.SHELL,
                      legs=[Leg(contract=contract(s), ratio=r) for s, r in zip(symbols, ratios)])
    result = enter_trade(shell, price, lots, direction, None, LIVE, stop_loss_price=stop)
    return shell.model_copy(update={"legs": result.legs, "status": result.status}), [result.trade]


# ---------- the formula ----------


def test_the_formula_is_the_root_of_r_c_r_transpose():
    r = np.array([3.0, 4.0])
    assert combine(r, np.eye(2)) == pytest.approx(5.0)  # no correlation: root of the sum of squares
    assert combine(r, np.ones((2, 2))) == pytest.approx(7.0)  # fully correlated: the plain sum
    assert combine(r, np.array([[1, -1], [-1, 1]])) == pytest.approx(1.0)  # offsetting positions
    assert combine(r, np.array([[1, 0.5], [0.5, 1]])) == pytest.approx(np.sqrt(9 + 16 + 2 * 0.5 * 12))
    assert combine(np.array([5.0]), np.ones((1, 1))) == 5.0


def test_a_matrix_that_is_not_a_valid_correlation_matrix_is_repaired_not_rooted_negative():
    broken = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])  # pairwise averages that no real data could give
    assert np.linalg.eigvalsh(broken).min() < 0
    value = combine(np.array([1.0, 1.0, 1.0]), broken)
    assert np.isfinite(value) and value >= 0


# ---------- the stop risk of one structure ----------


def test_the_stop_risk_is_the_loss_at_the_stop_for_the_lots_held():
    buy, trades = position("buy", price=75.0, lots=2, stop=70.0)
    risk = structure_stop_risk(buy, trades)
    assert (risk.entry, risk.stop, risk.lots, risk.side) == (75.0, 70.0, 2.0, "BUY") and risk.risk == pytest.approx(5.0 * 2 * 1000)
    sell, sell_trades = position("sell", price=75.0, lots=3, stop=77.0, direction="sell")
    assert structure_stop_risk(sell, sell_trades).risk == pytest.approx(2.0 * 3 * 1000)
    spread, spread_trades = position("spread", symbols=("CLZ26", "CLF27"), ratios=(1, -1), price=0.50, lots=5, stop=0.30)
    assert structure_stop_risk(spread, spread_trades).risk == pytest.approx(0.20 * 5 * 1000)
    inverse, inverse_trades = position("inverse", symbols=("CLZ26", "CLF27"), ratios=(-1, 1), price=0.50, lots=5, stop=0.30)
    assert structure_stop_risk(inverse, inverse_trades).risk == pytest.approx(0.20 * 5 * 1000)  # a stop is read by buy / sell, as it is entered and alerted


def test_no_stop_or_a_stop_already_past_the_entry_risks_nothing():
    open_, trades = position("no stop", stop=None)
    assert structure_stop_risk(open_, trades).risk is None
    locked, locked_trades = position("locked", price=75.0, stop=74.0)
    locked_trades = [locked_trades[0].model_copy(update={"stop_loss_price": 76.0})]  # trailed above the entry: a profit is locked in
    assert structure_stop_risk(locked, locked_trades).risk == 0.0


def test_the_stop_is_the_latest_one_typed_and_the_entry_is_that_of_the_lots_still_open():
    structure, trades = position("moved", price=75.0, lots=2, stop=70.0)
    later = trades[0].model_copy(update={"trade_id": "later", "timestamp": trades[0].timestamp.replace(year=trades[0].timestamp.year + 1), "stop_loss_price": 73.0})
    assert structure_stop_risk(structure, [*trades, later]).stop == 73.0
    assert structure_stop_risk(structure, [*trades, later]).risk == pytest.approx(2.0 * 2 * 1000)


# ---------- the whole book (correlation stubbed so every number is exact) ----------


def stub(monkeypatch, correlation, calls=None):
    def fake(items, structures, rolling, lookback, loader, *args, **kwargs):
        if calls is not None:
            calls.append((len(items), rolling, lookback))
        ids = [i["key"] for i in items if i["key"] not in (correlation.get("missing") or [])]
        values = np.eye(len(ids))
        for a, i in enumerate(ids):
            for b, j in enumerate(ids):
                if a != b:
                    values[a, b] = correlation["value"]
        return WatchlistResult(matrix=pd.DataFrame(values, index=ids, columns=ids), observations=lookback, requested=lookback, rolling=rolling)

    monkeypatch.setattr(pr, "compute_watchlist_correlation", fake)


def book(*positions):
    structures = [s for s, _ in positions]
    return structures, {s.structure_id: t for s, t in positions}


def test_two_positions_combine_with_their_correlation(monkeypatch):
    structures, trades = book(position("a", price=75, lots=2, stop=70), position("b", symbol="CLF27", price=75, lots=2, stop=72))  # 10,000 and 6,000
    for rho, expected in ((0.0, np.sqrt(10_000**2 + 6_000**2)), (1.0, 16_000.0), (-1.0, 4_000.0), (0.5, np.sqrt(10_000**2 + 6_000**2 + 2 * 0.5 * 10_000 * 6_000))):
        stub(monkeypatch, {"value": rho})
        result = PortfolioRisk(lambda: None).compute(structures, trades, 5, 20)
        assert result.value == pytest.approx(expected) and result.sum_of_stops == pytest.approx(16_000) and result.counted == 2
    assert result.diversification == pytest.approx(1 - result.value / 16_000)


def test_a_single_position_needs_no_correlation_and_none_with_a_stop_gives_no_figure(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("one position has nothing to correlate with")

    monkeypatch.setattr(pr, "compute_watchlist_correlation", boom)
    structures, trades = book(position("only", stop=70), position("no stop", symbol="CLF27", stop=None))
    result = PortfolioRisk(lambda: None).compute(structures, trades, 5, 20)
    assert result.value == pytest.approx(10_000) and result.counted == 1 and result.without_stop == ["no stop"] and result.open_structures == 2
    nothing = PortfolioRisk(lambda: None).compute(*book(position("a", stop=None)), 5, 20)
    assert nothing.value is None and nothing.without_stop == ["a"]
    assert PortfolioRisk(lambda: None).compute([], {}, 5, 20).value is None


def test_a_position_without_price_history_is_assumed_fully_correlated(monkeypatch):
    structures, trades = book(position("a", stop=70), position("b", symbol="CLF27", stop=70), position("c", symbol="CLG27", stop=70))  # 10,000 each
    stub(monkeypatch, {"value": 0.0, "missing": [structures[2].structure_id]})
    result = PortfolioRisk(lambda: None).compute(structures, trades, 5, 20)
    assert result.assumed_pairs == 2  # c with a, and c with b
    ones = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1.0]])
    assert result.value == pytest.approx(combine(np.array([10_000.0] * 3), ones))


def test_the_correlation_is_kept_until_the_book_the_data_or_the_windows_change(monkeypatch):
    calls, version = [], ["v1"]
    stub(monkeypatch, {"value": 0.3}, calls)
    engine = PortfolioRisk(lambda: None, lambda: version[0])
    structures, trades = book(position("a"), position("b", symbol="CLF27"))
    engine.compute(structures, trades, 5, 20)
    engine.compute(structures, trades, 5, 20)
    assert len(calls) == 1  # a Home refresh every 5 seconds does not recompute it
    moved = {s.structure_id: [t.model_copy(update={"stop_loss_price": 71.0}) for t in ts] for s, ts in zip(structures, trades.values())}
    assert engine.compute(structures, moved, 5, 20).sum_of_stops == pytest.approx(16_000) and len(calls) == 1  # a new stop only changes r
    engine.compute(structures, trades, 10, 20)
    version[0] = "v2"
    engine.compute(structures, trades, 10, 20)
    assert [c[1] for c in calls] == [5, 10, 10]
    engine.compute(*book(position("a"), position("b", symbol="CLF27"), position("c", symbol="CLG27")), 10, 20)
    assert len(calls) == 4


def test_a_failing_correlation_falls_back_to_the_cautious_sum_and_says_so(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no data")

    monkeypatch.setattr(pr, "compute_watchlist_correlation", broken)
    result = PortfolioRisk(lambda: None).compute(*book(position("a"), position("b", symbol="CLF27")), 5, 20)
    assert result.value == pytest.approx(20_000) and result.assumed_pairs == 1 and "correlation unavailable" in result.error


# ---------- through the real correlation engine ----------


def test_a_long_and_a_short_in_correlated_structures_offset_each_other(tmp_path, create_synthetic_parquet):
    class NoBackfill:
        def backfill_symbol(self, *args, **kwargs):
            raise AssertionError("no backfill")

    base = 75.0 + np.cumsum(np.random.default_rng(1).normal(0.0, 1.5, 80))
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    create_synthetic_parquet(tmp_path, "CLF27", closes=base + np.cumsum(np.random.default_rng(2).normal(0.0, 0.05, 80)))
    create_synthetic_parquet(tmp_path, "CLG27", closes=75.0 + np.cumsum(np.random.default_rng(3).normal(0.0, 1.5, 80)))
    loader = DataLoader(str(tmp_path), NoBackfill())
    engine = PortfolioRisk(lambda: loader)
    hedged = engine.compute(*book(position("long", stop=70), position("short", symbol="CLF27", direction="sell", stop=80)), 5, 30)
    assert hedged.sum_of_stops == pytest.approx(20_000) and hedged.value < 0.5 * hedged.sum_of_stops  # the short offsets the long
    same_way = engine.compute(*book(position("long", stop=70), position("also long", symbol="CLF27", stop=70)), 5, 30)
    assert same_way.value > 0.9 * same_way.sum_of_stops  # two longs in one market are almost one position
    three = engine.compute(*book(position("long", stop=70), position("short", symbol="CLF27", direction="sell", stop=80), position("other", symbol="CLG27", stop=70)), 5, 30)
    assert three.counted == 3 and 0 < three.value < three.sum_of_stops and three.assumed_pairs == 0


# ---------- Home ----------


def info(**overrides):
    base = {"value": 12_340.0, "sum_of_stops": 18_000.0, "counted": 3, "open_structures": 4, "without_stop": [], "assumed_pairs": 0,
            "rolling": 5, "lookback": 20, "error": None, "diversification": 1 - 12_340 / 18_000}
    base.update(overrides)
    return base


def test_the_home_card_shows_the_figure_the_sum_and_what_to_be_aware_of():
    text = str(hc.update_stop_risk_card({"stop_risk": info(without_stop=["Jun d-fly"], assumed_pairs=2)}))
    assert "$12,340" in text and "sum of the stops $18,000" in text and "correlation takes off 31%" in text
    assert "No stop, left out: Jun d-fly" in text and "2 pair(s) assumed fully correlated" in text and "5d rolling correlation, averaged over 20d" in text
    single = str(hc.update_stop_risk_card({"stop_risk": info(value=10_000.0, sum_of_stops=10_000.0, counted=1, diversification=0.0)}))
    assert "$10,000" in single and "correlation takes off" not in single and "rolling" not in single
    assert hc.update_stop_risk_card({}) == "—" and hc.update_stop_risk_card(None) == "—"
    assert "Could not be calculated" in str(hc.update_stop_risk_card({"stop_risk": None}))
    assert "Set a stop" in str(hc.update_stop_risk_card({"stop_risk": info(value=None, counted=0, without_stop=["a"])}))
    assert "No open structures" in str(hc.update_stop_risk_card({"stop_risk": info(value=None, counted=0, open_structures=0)}))


def test_the_card_is_on_the_home_layout_and_the_refresh_feeds_it(monkeypatch):
    def ids(component):
        found, stack = set(), [component]
        while stack:
            item = stack.pop()
            if getattr(item, "id", None):
                found.add(item.id)
            children = getattr(item, "children", None)
            stack.extend(children if isinstance(children, (list, tuple)) else [children] if children is not None else [])
        return found

    assert "home-card-stop-risk" in ids(home_layout())
    structures, trades = book(position("a"), position("b", symbol="CLF27"))
    stub(monkeypatch, {"value": 0.0})
    box = Container(repository=type("R", (), {"get_setting": lambda self, key, default=None: default})(), data_loader=None)
    monkeypatch.setattr(sc, "container", box)
    monkeypatch.setattr(sc, "_portfolio_risk", PortfolioRisk(lambda: None))
    summary = sc._stop_risk(structures, trades)
    assert summary["value"] == pytest.approx(np.sqrt(2) * 10_000) and summary["rolling"] == 5 and summary["lookback"] == 20
    monkeypatch.setattr(sc, "_portfolio_risk", None)  # anything that goes wrong leaves Home with a dash, not an error
    assert sc._stop_risk(structures, trades) is None
