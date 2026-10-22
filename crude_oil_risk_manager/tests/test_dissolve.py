"""Dissolve: structures cut against each other close together with no exit cost (core.dissolve + its modal)."""

from datetime import datetime, timedelta, timezone

import pytest

from core.dissolve import DissolveError, dissolve_structures, leg_symbols
from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType
from core.pnl import calculate_net_realized_pnl, calculate_portfolio_pnl, calculate_structure_transaction_costs
from core.trade_entry import recompute_structure_from_trades
from db.repository import Repository
from ui.callbacks import dissolve_callbacks as dcb
from ui.callbacks import structures_callbacks as sc
from ui.container import Container

MONTHS = {"F": 1, "G": 2, "H": 3, "J": 4}


def contract(letter):
    return Contract(product="CL", contract_month=MONTHS[letter], contract_year=2027, symbol=f"CL{letter}27",
                    multiplier=1000.0, tick_size=0.01, tick_value=10.0)


def structure(name, letters, ratios, direction, lots, entries, tc=1.0):
    legs = [Leg(contract=contract(l), ratio=r, lots=lots, entry_price=e, average_entry_price=e, direction=direction, transaction_cost_per_lot=tc)
            for l, r, e in zip(letters, ratios, entries)]
    kind = {1: StructureType.OUTRIGHT, 2: StructureType.SPREAD, 3: StructureType.FLY}.get(len(legs), StructureType.CUSTOM)
    return Structure(name=name, structure_type=kind, products=["CL"], legs=legs, status=StructureStatus.OPEN)


def entry_trade(s, lots, price, when):
    return Trade(structure_id=s.structure_id, leg_id=s.legs[0].leg_id, event_type=TradeEventType.TRADE, lots=lots, price=price,
                 direction=s.legs[0].direction, timestamp=when, transaction_cost=sum(l.transaction_cost_per_lot * lots for l in s.legs))


T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.fixture
def group():
    """A long Jan d-fly (2 lots) cut by selling fly 1 and buying fly 2: nothing is left in any contract."""
    dfly = structure("dfly", "FGHJ", (1, -3, 3, -1), "buy", 2, (80.0, 79.0, 78.0, 77.0), tc=1.0)
    fly1 = structure("fly1", "FGH", (1, -2, 1), "sell", 2, (80.5, 79.2, 78.1), tc=1.0)
    fly2 = structure("fly2", "GHJ", (1, -2, 1), "buy", 2, (79.3, 78.2, 77.4), tc=1.0)
    trades = {s.structure_id: [entry_trade(s, 2, 0.0, T0 + timedelta(days=i))] for i, s in enumerate((dfly, fly1, fly2))}
    return [dfly, fly1, fly2], trades


PRICES = {"CLF27": 80.2, "CLG27": 79.1, "CLH27": 78.15, "CLJ27": 77.3}


def expected_gross(structures, prices):
    total = 0.0
    for s in structures:
        for leg in s.legs:
            sign = 1 if leg.direction == "buy" else -1
            total += sign * leg.ratio * leg.lots * leg.contract.multiplier * (prices[leg.contract.symbol] - leg.entry_price)
    return total


def test_the_group_nets_to_zero_and_its_gross_pnl_comes_from_the_leg_prices(group):
    structures, trades = group
    assert leg_symbols(structures) == ["CLF27", "CLG27", "CLH27", "CLJ27"]
    result = dissolve_structures(structures, leg_prices=PRICES, trades_by_structure=trades)
    assert result.method == "legs" and result.warnings == []  # every contract nets to zero
    assert result.gross_pnl == pytest.approx(expected_gross(structures, PRICES))
    assert sum(i.realized_pnl for i in result.items) == pytest.approx(result.gross_pnl)
    assert result.entry_costs == pytest.approx(4 * 2 + 3 * 2 + 3 * 2)  # entry costs only: 20, no exit cost anywhere
    assert result.net_pnl == pytest.approx(result.gross_pnl - 20.0)
    assert all(i.status == StructureStatus.CLOSED and all(l.lots == 0 for l in i.legs) for i in result.items)
    assert all(i.trade.event_type == TradeEventType.DISSOLVE and i.trade.transaction_cost == 0.0 for i in result.items)


def test_a_typed_gross_pnl_is_split_but_adds_up_exactly(group):
    structures, trades = group
    result = dissolve_structures(structures, typed_gross=1500.0, trades_by_structure=trades)
    assert result.method == "typed" and result.gross_pnl == pytest.approx(1500.0)
    shares = {i.structure.name: i.realized_pnl for i in result.items}
    assert shares["dfly"] == pytest.approx(1500.0 * 8 / 20) and shares["fly1"] == pytest.approx(1500.0 * 6 / 20)  # by entry cost
    assert result.net_pnl == pytest.approx(1500.0 - 20.0)


def test_part_of_the_lots_can_be_dissolved(group):
    structures, trades = group
    lots = {s.structure_id: 1.0 for s in structures}
    result = dissolve_structures(structures, lots=lots, leg_prices=PRICES, trades_by_structure=trades)
    assert all(i.status == StructureStatus.OPEN and [l.lots for l in i.legs] == [1.0] * len(i.legs) for i in result.items)
    assert result.gross_pnl == pytest.approx(expected_gross(structures, PRICES) / 2)


def test_problems_are_listed_and_a_group_that_does_not_net_is_flagged(group):
    structures, trades = group
    with pytest.raises(DissolveError) as exc:
        dissolve_structures(structures[:1], leg_prices=PRICES)
    assert "at least two" in " ".join(exc.value.errors)
    with pytest.raises(DissolveError, match="Price needed for: CLJ27"):
        dissolve_structures(structures, leg_prices={k: v for k, v in PRICES.items() if k != "CLJ27"})
    with pytest.raises(DissolveError, match="either the leg prices or a typed"):
        dissolve_structures(structures, leg_prices=PRICES, typed_gross=5.0)
    with pytest.raises(DissolveError, match="Give the price of every leg"):
        dissolve_structures(structures)
    with pytest.raises(DissolveError, match="cannot dissolve 9 lots"):
        dissolve_structures(structures, lots={structures[0].structure_id: 9}, leg_prices=PRICES)
    lopsided = dissolve_structures(structures[:2], typed_gross=0.0, trades_by_structure=trades)
    assert any("do not net to zero" in w and "CLJ27" in w for w in lopsided.warnings)


def test_the_book_counts_only_the_entry_costs_and_the_gross_pnl(group):
    structures, trades = group
    result = dissolve_structures(structures, leg_prices=PRICES, trades_by_structure=trades)
    booked = []
    for item in result.items:
        booked.append(item.trade)
    all_trades = {i.structure.structure_id: trades[i.structure.structure_id] + [i.trade] for i in result.items}
    net = sum(calculate_net_realized_pnl(t) for t in all_trades.values())
    assert net == pytest.approx(result.gross_pnl - 20.0)  # gross minus the entry costs, nothing extra for an exit
    assert sum(calculate_structure_transaction_costs(t) for t in all_trades.values()) == pytest.approx(20.0)
    closed = [i.structure.model_copy(update={"legs": i.legs, "status": i.status}) for i in result.items]
    assert calculate_portfolio_pnl(closed, all_trades, PRICES, [], closed, all_trades)["total_realized_all_time"] == pytest.approx(net)


def test_a_dissolve_is_replayed_when_an_earlier_trade_is_deleted(group):
    structures, trades = group
    dfly = structures[0]
    result = dissolve_structures(structures, leg_prices=PRICES, trades_by_structure=trades)
    item = result.items[0]
    entry = Trade(structure_id=dfly.structure_id, leg_id=dfly.legs[0].leg_id, event_type=TradeEventType.TRADE, lots=2, price=1.0,
                  direction="buy", timestamp=T0)
    item.trade.timestamp = T0 + timedelta(days=5)
    replay = recompute_structure_from_trades(dfly.model_copy(update={"status": StructureStatus.OPEN}), [entry, item.trade])
    assert replay.structure.status == StructureStatus.CLOSED and replay.structure.close_trigger == "dissolved"
    assert all(l.lots == 0 for l in replay.structure.legs)


# ---------- the modal's callbacks ----------


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture(autouse=True)
def env(repo, monkeypatch):
    fresh = Container(repository=repo)
    for module in (dcb, sc):
        monkeypatch.setattr(module, "container", fresh)
    return fresh


def save_group(repo, group):
    structures, trades = group
    for s in structures:
        for leg in s.legs:
            repo.save_contract(leg.contract)
        repo.save_structure(s)
        for t in trades[s.structure_id]:
            repo.save_trade(t)
    return [s.structure_id for s in structures]


def form(ids, mode="legs", prices=PRICES, gross=None, lots=None, notes=None):
    symbols = list(prices) if prices else []
    return (ids, [lots] * 0 if lots is None else lots, [{"type": "dissolve-lots", "index": i} for i in ids] if lots else [],
            [prices[s] for s in symbols] if mode == "legs" else [], [{"type": "dissolve-price", "index": s} for s in symbols] if mode == "legs" else [],
            mode, gross, notes)


def test_the_modal_lists_the_open_structures_and_builds_the_inputs(repo, group):
    ids = save_group(repo, group)
    labels = sorted(dcb._label(s) for s in dcb._open_structures())
    assert labels[0].startswith("dfly — BUY 2 lots @ ") and any(l.startswith("fly1 — SELL 2 lots") for l in labels)
    lots_wrap, price_wrap = dcb.build_inputs(ids, {"CLF27": {"price": 80.0}})
    assert "dfly" in str(lots_wrap) and "fly2" in str(lots_wrap)
    text = str(price_wrap)
    for symbol in PRICES:
        assert symbol in text
    assert "80.0" in text  # the live price is the starting value
    assert dcb.build_inputs([], {}) == ([], [])


def test_the_preview_and_the_dissolve_itself(repo, group):
    ids = save_group(repo, group)
    body, disabled = dcb.preview(*form(ids))
    assert disabled is False and "Net P&L" in str(body) and "no exit cost" in str(body)
    assert dcb.preview(*form([]))[1] is True
    body, disabled = dcb.preview(*form(ids, mode="typed", gross=None, prices=None))
    assert disabled is True and "Give the price of every leg" in str(body)

    out = dcb.execute(1, *form(ids), "active", "all", "name", {}, {})
    modal_open, rows, toast_open, message, icon, header = out
    assert modal_open is False and icon == "success" and "Dissolved 3 structures" in message
    for sid in ids:
        saved = repo.get_structure(sid)
        assert saved.status == StructureStatus.CLOSED and saved.close_trigger == "dissolved" and all(l.lots == 0 for l in saved.legs)
        trades = repo.get_trades_for_structure(sid)
        assert trades[-1].event_type == TradeEventType.DISSOLVE and trades[-1].transaction_cost == 0.0
    total = sum(calculate_net_realized_pnl(repo.get_trades_for_structure(sid)) for sid in ids)
    assert total == pytest.approx(expected_gross(group[0], PRICES) - 20.0)


def test_a_failed_dissolve_changes_nothing(repo, group):
    ids = save_group(repo, group)
    out = dcb.execute(1, *form(ids[:1]), "active", "all", "name", {}, {})
    assert out[0] is dcb.no_update and out[4] == "danger"
    assert all(repo.get_structure(sid).status == StructureStatus.OPEN for sid in ids)


def test_the_page_has_the_button_and_the_modal_and_the_callbacks_register():
    import dash

    from ui.layouts.structures import structures_layout

    text = str(structures_layout())
    for component_id in ("btn-dissolve", "modal-dissolve", "dissolve-select", "dissolve-mode", "dissolve-gross", "btn-dissolve-confirm", "btn-new-structure"):
        assert component_id in text, component_id
    app = dash.Dash(__name__)
    dcb.register_dissolve_callbacks(app)
    assert len(app.callback_map) >= 5


def test_one_number_of_lots_can_be_set_for_every_structure(repo, group):
    ids = save_group(repo, group)
    id_dicts = [{"type": "dissolve-lots", "index": i} for i in ids]
    assert dcb.set_all_lots(1, id_dicts) == [1.0, 1.0, 1.0]
    assert dcb.set_all_lots(5, id_dicts) == [2.0, 2.0, 2.0]  # never more than each has open
    assert dcb.set_all_lots(None, id_dicts) == [2.0, 2.0, 2.0]  # blank: all lots again
    from dash.exceptions import PreventUpdate

    with pytest.raises(PreventUpdate):
        dcb.set_all_lots(1, [])
    assert "dissolve-lots-all" in str(__import__("ui.layouts.dissolve", fromlist=["x"]).dissolve_modal())
