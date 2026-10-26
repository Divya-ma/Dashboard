"""A partial exit or dissolve is charged only the transaction cost of the lots it takes off; the rest stays with the lots left."""

from datetime import datetime, timedelta, timezone

import pytest

from core.dissolve import dissolve_structures
from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType
from core.pnl import calculate_portfolio_pnl, entry_cost_for_lots, split_transaction_costs
from core.trade_entry import exit_structure
from db.repository import Repository
from ui.callbacks import structure_detail_callbacks as dc
from ui.callbacks import structures_callbacks as sc
from ui.container import Container

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
GRID = ("active", "all", "name", {}, {})


def contract(symbol="CLZ26", month=12):
    return Contract(product="CL", contract_month=month, contract_year=2026, symbol=symbol, multiplier=1000.0, tick_size=0.01, tick_value=10.0)


def held(name="s", lots=10.0, tc_per_lot=1.5, entry=70.0):
    leg = Leg(contract=contract(), ratio=1, lots=lots, entry_price=entry, average_entry_price=entry, transaction_cost_per_lot=tc_per_lot)
    return Structure(name=name, structure_type=StructureType.OUTRIGHT, products=["CL"], legs=[leg], status=StructureStatus.OPEN)


def trade(s, kind, lots, price, tc=0.0, realized=None, day=0, direction="buy"):
    return Trade(structure_id=s.structure_id, leg_id=s.legs[0].leg_id, event_type=kind, lots=lots, price=price, direction=direction,
                 timestamp=T0 + timedelta(days=day), transaction_cost=tc, realized_pnl=realized)


def test_the_entry_cost_belongs_to_the_lots_that_were_entered():
    s = held()
    entry = trade(s, TradeEventType.TRADE, 10, 70.0, tc=15.0)
    assert entry_cost_for_lots([entry], 4) == pytest.approx(6.0) and entry_cost_for_lots([entry], 10) == 15.0
    assert entry_cost_for_lots([entry], 99) == 15.0 and entry_cost_for_lots([], 4) == 0.0
    # a later add at a different cost: the earliest lots go first (first in, first out), each at its own entry's cost per lot
    add = trade(s, TradeEventType.ADD, 10, 71.0, tc=25.0, day=1)
    assert entry_cost_for_lots([entry, add], 5) == pytest.approx(7.5)  # 5 of the first entry's lots at 1.50
    assert entry_cost_for_lots([entry, add], 12) == pytest.approx(15.0 + 2 * 2.5)  # all 10 of them, then 2 of the add's at 2.50


def test_costs_split_between_the_exited_lots_and_the_lots_still_open():
    s = held()
    entry = trade(s, TradeEventType.TRADE, 10, 70.0, tc=15.0)
    exit4 = trade(s, TradeEventType.STRUCTURE_PARTIAL_EXIT, 4, 72.0, tc=6.0, realized=8000.0, day=1, direction="sell")
    realized_tc, open_tc = split_transaction_costs([entry, exit4], open_lots=6)
    assert open_tc == pytest.approx(9.0)  # 6 of the 10 entered lots
    assert realized_tc == pytest.approx(6.0 + 6.0)  # their exit cost plus the entry cost of the 4 lots exited
    assert realized_tc + open_tc == pytest.approx(15.0 + 6.0)  # always the structure's whole cost


def test_homes_realized_takes_only_the_exited_lots_cost_and_the_rest_stays_open():
    s = held(lots=6.0)
    trades = [trade(s, TradeEventType.TRADE, 10, 70.0, tc=15.0),
              trade(s, TradeEventType.STRUCTURE_PARTIAL_EXIT, 4, 72.0, tc=6.0, realized=8000.0, day=1, direction="sell")]
    summary = calculate_portfolio_pnl([s], {s.structure_id: trades}, {"CLZ26": 73.0}, [])
    assert summary["total_realized_all_time"] == pytest.approx(8000.0 - 12.0)  # not 8000 - 21: the 6 open lots' entry cost is not realized
    assert summary["total_unrealized"] == pytest.approx(3000.0 * 6) and summary["total_unrealized_net_of_tc"] == pytest.approx(18000.0 - 9.0)
    assert summary["total_pnl"] == pytest.approx(18000.0 + 8000.0 - 21.0)  # the position as a whole still carries every cost
    info = summary["per_structure"][s.structure_id]
    assert info["realized_costs"] == pytest.approx(12.0) and info["open_costs"] == pytest.approx(9.0)


def test_an_untouched_position_keeps_all_its_entry_cost_with_its_lots():
    s = held(lots=10.0)
    summary = calculate_portfolio_pnl([s], {s.structure_id: [trade(s, TradeEventType.TRADE, 10, 70.0, tc=15.0)]}, {"CLZ26": 70.0}, [])
    assert summary["total_realized_all_time"] == 0.0 and summary["total_pnl"] == pytest.approx(-15.0)
    assert summary["total_unrealized_net_of_tc"] == pytest.approx(-15.0)


# ---------- the exit forms ----------


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture(autouse=True)
def env(repo, monkeypatch):
    fresh = Container(repository=repo)
    for module in (dc, sc):
        monkeypatch.setattr(module, "container", fresh)
    return fresh


def save_held(repo, s, trades):
    for leg in s.legs:
        repo.save_contract(leg.contract)
    repo.save_structure(s)
    for t in trades:
        repo.save_trade(t)


def test_the_exit_dialog_and_toast_charge_only_the_lots_being_exited(repo):
    s = held()
    save_held(repo, s, [trade(s, TradeEventType.TRADE, 10, 70.0, tc=15.0)])
    is_open, body, disabled = dc.open_exit_confirmation(1, 72.0, 4, s.structure_id)
    text = str(body)
    assert disabled is False and "Transaction cost of these 4 lots (entry + exit): -$12" in text  # 6 entry + 6 exit, not 15 + 6
    assert "+$7,988" in text  # 8,000 gross - 12
    out = dc.execute_full_exit(1, 72.0, 4, None, s.structure_id, *GRID)
    assert "net of TC for these lots): +$7,988" in out[3] and "6 lots remain" in out[3]
    left = repo.get_structure(s.structure_id)
    trades = repo.get_trades_for_structure(s.structure_id)
    summary = calculate_portfolio_pnl([left], {s.structure_id: trades}, {"CLZ26": 72.0}, [])
    assert summary["total_realized_all_time"] == pytest.approx(8000.0 - 12.0)
    assert summary["per_structure"][s.structure_id]["open_costs"] == pytest.approx(9.0)  # stays with the 6 lots left
    out = dc.execute_full_exit(1, 73.0, 6, None, s.structure_id, *GRID)  # the rest: its own 9 entry + 9 exit
    assert "net of TC): +$17,982" in out[3]


def test_a_partial_dissolve_takes_only_the_dissolved_lots_entry_cost():
    a = held("a", lots=10.0, tc_per_lot=1.0)
    b = held("b", lots=10.0, tc_per_lot=1.0)
    b = b.model_copy(update={"legs": [b.legs[0].model_copy(update={"ratio": 1, "direction": "sell"})]})
    trades = {a.structure_id: [trade(a, TradeEventType.TRADE, 10, 70.0, tc=10.0)],
              b.structure_id: [trade(b, TradeEventType.TRADE, 10, 70.0, tc=10.0, direction="sell")]}
    result = dissolve_structures([a, b], lots={a.structure_id: 4, b.structure_id: 4}, leg_prices={"CLZ26": 71.0}, trades_by_structure=trades)
    assert result.entry_costs == pytest.approx(4.0 + 4.0)  # 4 of 10 lots of each, not the whole 20
    assert result.net_pnl == pytest.approx(result.gross_pnl - 8.0)
    assert [i.legs[0].lots for i in result.items] == [6.0, 6.0]  # and the other 6 lots keep the rest of their cost
    typed = dissolve_structures([a, b], lots={a.structure_id: 5, b.structure_id: 5}, typed_gross=100.0, trades_by_structure=trades)
    assert typed.entry_costs == pytest.approx(10.0) and typed.net_pnl == pytest.approx(90.0)
