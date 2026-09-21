"""Tests for core.trade_entry, core.structure_edit, the detail layout and its callbacks."""

import pytest
from dash.exceptions import PreventUpdate

from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType
from core.pnl import calculate_portfolio_pnl
from core.structure_builder import StructureBuildError
from core.structure_edit import apply_leg_edits
from core.structure_view import structure_entry_price
from core.trade_entry import (
    TradeError, allocate_leg_entry_prices, enter_trade, exit_structure, structure_pnl,
)
from db.repository import Repository
from ui.callbacks import structure_detail_callbacks as dc
from ui.callbacks import structures_callbacks as sc
from ui.container import Container
from ui.layouts.structure_detail import structure_detail_layout


def contract(symbol="CLZ26", product="CL", month=12, multiplier=1000.0):
    return Contract(product=product, contract_month=month, contract_year=2026, symbol=symbol,
                    multiplier=multiplier, tick_size=0.01, tick_value=10.0)


def outright(status=StructureStatus.SHELL, lots=0.0, entry=None):
    leg = Leg(contract=contract(), ratio=1, lots=lots, entry_price=entry, average_entry_price=entry)
    return Structure(name="Outright", structure_type=StructureType.OUTRIGHT, products=["CL"], legs=[leg], status=status)


def spread(status=StructureStatus.SHELL, ratios=(1, -1)):
    legs = [Leg(contract=contract("CLX26", month=11), ratio=ratios[0]), Leg(contract=contract("CLZ26"), ratio=ratios[1])]
    return Structure(name="Spread", structure_type=StructureType.SPREAD, products=["CL"], legs=legs, status=status)


LIVE = {"CLX26": 75.0, "CLZ26": 74.0}


# ---------- core.trade_entry ----------


def test_single_leg_entry_takes_structure_price():
    result = enter_trade(outright(), 75.0, 10, "buy", "  hi ", {})
    leg = result.legs[0]
    assert (leg.lots, leg.entry_price, leg.average_entry_price) == (10, 75.0, 75.0)
    assert result.status == StructureStatus.OPEN and result.trade.event_type == TradeEventType.TRADE
    assert result.trade.notes == "hi" and result.trade.leg_id == leg.leg_id


@pytest.mark.parametrize("ratios", [(1, -1), (-1, 1)])
def test_multi_leg_entry_reproduces_structure_price_and_pnl(ratios):
    s = spread(ratios=ratios)
    result = enter_trade(s, 1.25, 5, "buy", None, LIVE)
    entered = s.model_copy(update={"legs": result.legs, "status": StructureStatus.OPEN})
    assert structure_entry_price(entered) == pytest.approx(1.25)
    # Per-leg engine total == structure-level PnL formula.
    live_price = 1.75
    engine, _ = calculate_portfolio_pnl([entered], {}, {"CLX26": 75.5, "CLZ26": 73.75}, [])["total_unrealized"], None
    direction = 1 if ratios[0] > 0 else -1
    assert engine == pytest.approx(direction * (live_price - 1.25) * 5 * 1000)
    assert structure_pnl(entered, 1.25, live_price, 5) == pytest.approx(engine)


def test_fly_allocation_composite():
    legs = [Leg(contract=contract(f"CL{m}26", month=i), ratio=r) for i, (m, r) in enumerate([("F", 1), ("G", -2), ("H", 1)], 1)]
    prices = {"CLF26": 76.0, "CLG26": 75.0, "CLH26": 74.5}
    entries = allocate_leg_entry_prices(legs, 0.5, prices)
    assert sum(l.ratio * e for l, e in zip(legs, entries)) == pytest.approx(0.5)


def test_multi_leg_entry_needs_live_prices():
    with pytest.raises(TradeError, match="Live prices for CLZ26"):
        enter_trade(spread(), 1.0, 5, "buy", None, {"CLX26": 75.0})


def test_add_averages_the_entry_price():
    s = outright(StructureStatus.OPEN, lots=10, entry=70.0)
    result = enter_trade(s, 76.0, 10, "buy", None, {})
    assert result.trade.event_type == TradeEventType.ADD
    assert result.legs[0].lots == 20 and result.legs[0].average_entry_price == pytest.approx(73.0)
    assert result.legs[0].entry_price == 70.0


def test_entry_validation():
    for price, lots, match in [(None, 5, "price"), (0, 5, "greater than 0"), (75, 0, "Lots"), (75, None, "Lots")]:
        with pytest.raises(TradeError, match=match):
            enter_trade(outright(), price, lots, "buy", None, {})
    with pytest.raises(TradeError, match="buy or sell"):
        enter_trade(outright(), 75, 1, "hold", None, {})
    with pytest.raises(TradeError, match="closed"):
        enter_trade(outright(StructureStatus.CLOSED), 75, 1, "buy", None, {})
    # Spread prices may be zero or negative.
    assert enter_trade(spread(), -0.5, 1, "buy", None, LIVE).trade.price == -0.5


def test_exit_computes_realized_pnl_and_zeroes_legs():
    s = outright(StructureStatus.OPEN, lots=10, entry=75.0)
    result = exit_structure(s, 77.0, 10, "done")
    assert result.realized_pnl == pytest.approx(20000.0)
    assert result.trade.event_type == TradeEventType.FULL_EXIT and result.trade.realized_pnl == pytest.approx(20000.0)
    assert result.trade.direction == "sell" and result.legs[0].lots == 0 and result.legs[0].entry_price == 75.0


def test_exit_short_structure_direction():
    s = outright(StructureStatus.OPEN, lots=1, entry=75.0)
    s = s.model_copy(update={"legs": [s.legs[0].model_copy(update={"ratio": -1})]})
    result = exit_structure(s, 74.0, 1, None)
    assert result.realized_pnl == pytest.approx(1000.0) and result.trade.direction == "buy"


def test_exit_rejects_partial_and_wrong_status():
    s = outright(StructureStatus.OPEN, lots=10, entry=75.0)
    with pytest.raises(TradeError, match="Partial exits are not supported"):
        exit_structure(s, 77.0, 4, None)
    with pytest.raises(TradeError, match="open structure"):
        exit_structure(outright(), 77.0, 10, None)
    with pytest.raises(TradeError, match="Exit price"):
        exit_structure(s, None, 10, None)


# ---------- core.structure_edit ----------


def test_apply_leg_edits_records_changes_and_keeps_position():
    s = outright(StructureStatus.OPEN, lots=10, entry=75.0)
    result = apply_leg_edits(s, ["clf27"], [2], lambda symbol: None)
    assert result.legs[0].contract.symbol == "CLF27" and result.legs[0].ratio == 2
    assert result.legs[0].lots == 10 and result.legs[0].entry_price == 75.0
    assert [c.symbol for c in result.new_contracts] == ["CLF27"]
    assert "symbol CLZ26 -> CLF27" in result.audit_note and "ratio +1 -> +2" in result.audit_note
    assert any("entry price was not changed" in w for w in result.warnings)


def test_apply_leg_edits_validates():
    s = outright()
    with pytest.raises(StructureBuildError) as exc:
        apply_leg_edits(s, ["nope"], [0], lambda symbol: None)
    assert len(exc.value.errors) == 2
    with pytest.raises(StructureBuildError, match="number of legs"):
        apply_leg_edits(s, ["CLZ26", "CLF27"], [1, -1], lambda symbol: None)


# ---------- layout ----------


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


FORM_IDS = [
    "btn-edit-structure", "trade-entry-form", "trade-entry-price", "btn-use-live-price", "trade-entry-lots",
    "trade-direction", "trade-pnl-preview", "trade-entry-notes", "btn-confirm-trade", "trade-exit-form",
    "trade-exit-price", "btn-use-live-exit-price", "trade-exit-lots", "btn-exit-all-lots", "exit-pnl-preview",
    "trade-exit-notes", "btn-confirm-exit",
]


def test_layout_always_contains_every_form_id_and_shows_forms_by_status():
    live = {"CLZ26": {"price": 76.0}}
    for structure in (outright(), outright(StructureStatus.OPEN, 10, 75.0), outright(StructureStatus.CLOSED, 0, 75.0)):
        layout = structure_detail_layout(structure, live, [], None)
        for component_id in FORM_IDS:
            assert find(layout, component_id) is not None, (structure.status, component_id)

    def hidden(structure, form):
        style = find(structure_detail_layout(structure, live, [], None), form).style
        return style == {"display": "none"}

    assert not hidden(outright(), "trade-entry-form") and hidden(outright(), "trade-exit-form")
    opened = outright(StructureStatus.OPEN, 10, 75.0)
    assert not hidden(opened, "trade-exit-form")
    closed = outright(StructureStatus.CLOSED, 0, 75.0)
    assert hidden(closed, "trade-entry-form") and hidden(closed, "trade-exit-form")
    assert find(structure_detail_layout(outright(), live, [], None), "trade-entry-collapse").is_open is True
    assert find(structure_detail_layout(opened, live, [], None), "trade-entry-collapse").is_open is False


def test_layout_pnl_summary_lock_icon_and_history():
    s = outright(StructureStatus.OPEN, 10, 75.0)
    trade = Trade(structure_id=s.structure_id, leg_id=s.legs[0].leg_id, event_type=TradeEventType.TRADE, lots=10, price=75.0, direction="buy")
    text = str(structure_detail_layout(s, {"CLZ26": {"price": 76.0}}, [trade], None))
    assert "+$10,000" in text and "75.00 → 76.00" in text and "🔒 Edit" in text
    assert "Trade History" in text and "exchange-quoted" in text
    plain = str(structure_detail_layout(outright(), {}, [], None))
    assert "✏️ Edit" in plain and "Trade History" not in plain and "Unrealized PnL" not in plain


def test_layout_falls_back_to_saved_pnl_when_prices_missing():
    from core.models import PnLRecord

    s = outright(StructureStatus.OPEN, 10, 75.0)
    record = PnLRecord(structure_id=s.structure_id, unrealized_pnl=500.0, realized_pnl=0.0, total_pnl=500.0)
    text = str(structure_detail_layout(s, {}, [], record))
    assert "+$500" in text and "Live prices missing" in text


def test_layout_edit_mode_has_inline_inputs():
    layout = structure_detail_layout(outright(), {}, [], None, edit_mode=True)
    assert find(layout, {"type": "edit-leg-symbol", "index": 0}) is not None
    assert find(layout, "btn-save-edit") is not None
    assert find(structure_detail_layout(outright(), {}, [], None), "btn-save-edit") is None


# ---------- callbacks ----------


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture(autouse=True)
def env(repo, monkeypatch):
    fresh = Container(repository=repo)
    for module in (dc, sc):
        monkeypatch.setattr(module, "container", fresh)
    return fresh


def save(repo, structure):
    for leg in structure.legs:
        repo.save_contract(leg.contract)
    repo.save_structure(structure)
    return structure.structure_id


LIVE_STORE = {"CLZ26": {"price": 76.0}, "CLX26": {"price": 75.0}}
GRID = ("active", "all", "name", {}, {})


def test_render_and_helpers(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    assert "Outright" in str(dc.render_structure_detail(sid, LIVE_STORE))
    assert dc.render_structure_detail(None, LIVE_STORE) == ""
    assert "not found" in str(dc.render_structure_detail("missing", {}))
    assert dc.fill_live_price(1, sid, LIVE_STORE) == 76.0
    assert dc.fill_exit_all_lots(1, sid) == 10
    assert dc.update_live_labels(LIVE_STORE, sid) == ("Live: 76.00", "Live: 76.00")
    for bad in (lambda: dc.fill_live_price(None, sid, LIVE_STORE), lambda: dc.fill_live_price(1, sid, {}),
                lambda: dc.fill_exit_all_lots(1, "missing"), lambda: dc.toggle_add_trade(None, False)):
        with pytest.raises(PreventUpdate):
            bad()
    assert dc.toggle_add_trade(1, False) is True


def test_pnl_previews(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    assert "+$10,000" in str(dc.preview_entry_pnl(75.0, 10, sid, LIVE_STORE))
    assert dc.preview_entry_pnl(None, 10, sid, LIVE_STORE) == ""
    assert "unavailable" in dc.preview_entry_pnl(75.0, 10, sid, {})
    assert "+$20,000" in str(dc.preview_exit_pnl(77.0, 10, sid))
    assert dc.preview_exit_pnl(77.0, None, sid) == ""


def test_confirm_trade_entry_opens_shell_and_writes_audit(repo):
    sid = save(repo, outright())
    body, rows, is_open, message, icon, header = dc.confirm_trade_entry(1, 75.0, 10, "buy", "n", sid, LIVE_STORE, *GRID)
    saved = repo.get_structure(sid)
    assert saved.status == StructureStatus.OPEN and saved.legs[0].lots == 10 and saved.legs[0].entry_price == 75.0
    assert "trade entered: 10 lots at 75" in saved.notes
    (trade,) = repo.get_trades_for_structure(sid)
    assert trade.event_type == TradeEventType.TRADE and trade.price == 75.0
    assert icon == "success" and "Trade entered: 10 lots at 75" in message and [r["status"] for r in rows] == ["OPEN"]
    assert "Trade History" in str(body)


def test_confirm_trade_entry_validation_error_saves_nothing(repo):
    sid = save(repo, outright())
    body, rows, is_open, message, icon, header = dc.confirm_trade_entry(1, 75.0, 0, "buy", None, sid, LIVE_STORE, *GRID)
    assert body is dc.no_update and rows is dc.no_update and icon == "danger"
    assert repo.get_trades_for_structure(sid) == [] and repo.get_structure(sid).status == StructureStatus.SHELL


def test_multi_leg_trade_entry_persists_consistent_legs(repo):
    sid = save(repo, spread())
    dc.confirm_trade_entry(1, 1.25, 5, "buy", None, sid, LIVE_STORE, *GRID)
    saved = repo.get_structure(sid)
    assert structure_entry_price(saved) == pytest.approx(1.25)
    assert calculate_portfolio_pnl([saved], {}, {"CLX26": 75.0, "CLZ26": 73.75}, [])["total_unrealized"] == pytest.approx(0.0)


def test_exit_confirmation_and_execution(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    is_open, body, disabled = dc.open_exit_confirmation(1, 77.0, 10, sid)
    assert is_open is True and disabled is False and "+$20,000" in str(body) and "cannot be undone" in str(body)
    is_open, body, disabled = dc.open_exit_confirmation(1, 77.0, 4, sid)
    assert disabled is True and "Partial exits" in str(body)
    assert repo.get_structure(sid).status == StructureStatus.OPEN  # confirmation alone never exits

    detail_open, rows, toast_open, message, icon, header, confirm_open = dc.execute_full_exit(1, 77.0, 10, "bye", sid, *GRID)
    closed = repo.get_structure(sid)
    assert closed.status == StructureStatus.CLOSED and closed.close_trigger == "manual" and closed.closed_at
    assert closed.legs[0].lots == 0 and "full exit" in closed.notes
    (trade,) = repo.get_trades_for_structure(sid)
    assert trade.event_type == TradeEventType.FULL_EXIT and trade.realized_pnl == pytest.approx(20000.0)
    assert detail_open is False and confirm_open is False and rows == [] and "+$20,000" in message


def test_execute_full_exit_rejects_partial(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    detail_open, rows, toast_open, message, icon, header, confirm_open = dc.execute_full_exit(1, 77.0, 3, None, sid, *GRID)
    assert detail_open is dc.no_update and icon == "danger" and confirm_open is False
    assert repo.get_structure(sid).status == StructureStatus.OPEN and repo.get_trades_for_structure(sid) == []


def test_edit_flow_without_and_with_trades(repo):
    sid = save(repo, outright())
    confirm_open, body = dc.initiate_edit(1, sid, LIVE_STORE)
    assert confirm_open is dc.no_update and "btn-save-edit" in str(body)  # no trades: edit straight away

    opened = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    trade = Trade(structure_id=opened, leg_id=repo.get_structure(opened).legs[0].leg_id, event_type=TradeEventType.TRADE, lots=10, price=75.0, direction="buy")
    repo.save_trade(trade)
    confirm_open, body = dc.initiate_edit(1, opened, LIVE_STORE)
    assert confirm_open is True and body is dc.no_update  # trades: ask first

    body, confirm_open = dc.proceed_with_edit(1, opened, LIVE_STORE)
    assert confirm_open is False and "btn-save-edit" in str(body)
    assert "edit mode opened after confirmation" in repo.get_structure(opened).notes
    assert "btn-save-edit" not in str(dc.cancel_edit(1, opened, LIVE_STORE))
    assert dc.cancel_edit_confirm(1) is False and dc.cancel_exit(1) is False


def test_save_edit_persists_and_audits(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    body, is_open, message, icon, header = dc.save_edit(1, ["CLF27"], [1], sid, LIVE_STORE)
    saved = repo.get_structure(sid)
    assert saved.legs[0].contract.symbol == "CLF27" and saved.legs[0].lots == 10
    assert "edited: leg 1 symbol CLZ26 -> CLF27" in saved.notes and icon == "success"
    body, is_open, message, icon, header = dc.save_edit(1, ["bad"], [1], sid, LIVE_STORE)
    assert body is dc.no_update and icon == "danger"


def test_audit_notes_can_accumulate_beyond_500_chars(repo):
    sid = save(repo, outright())
    for i in range(12):
        repo.update_structure_legs(sid, repo.get_structure(sid).legs, f"audit line number {i} " + "x" * 30)
    assert len(repo.get_structure(sid).notes) > 500
