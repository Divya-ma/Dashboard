"""Tests for core.trade_entry, core.structure_edit, the detail layout and its callbacks."""

from datetime import datetime, timedelta, timezone

import pytest
from dash.exceptions import PreventUpdate

from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType
from core.pnl import calculate_portfolio_pnl
from core.structure_builder import StructureBuildError
from core.structure_edit import apply_leg_edits
from core.structure_view import structure_entry_price
from core.trade_entry import (
    TradeError, allocate_leg_entry_prices, enter_trade, exit_structure, partial_exit_legs,
    recompute_structure_from_trades, structure_pnl,
)
from core.alerts import AlertManager
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
    assert result.realized_pnl == pytest.approx(1000.0) and result.trade.direction == "sell"  # closes a buy


def test_exit_rejects_partial_and_wrong_status():
    s = outright(StructureStatus.OPEN, lots=10, entry=75.0)
    with pytest.raises(TradeError, match="Partial exits are not supported"):
        exit_structure(s, 77.0, 4, None)
    with pytest.raises(TradeError, match="open structure"):
        exit_structure(outright(), 77.0, 10, None)
    with pytest.raises(TradeError, match="Exit price"):
        exit_structure(s, None, 10, None)


# ---------- core.trade_entry: optional per-leg entry prices ----------


def test_leg_prices_blank_reproduces_todays_allocation():
    """All leg prices left blank (None) behaves exactly like today: the synthetic allocation."""
    s = spread()
    with_none = enter_trade(s, 1.25, 5, "buy", None, LIVE, leg_prices=[None, None])
    without = enter_trade(s, 1.25, 5, "buy", None, LIVE)
    assert [l.entry_price for l in with_none.legs] == [l.entry_price for l in without.legs]


def test_leg_prices_filled_are_used_verbatim():
    s = spread()
    result = enter_trade(s, 1.25, 5, "buy", None, LIVE, leg_prices=[76.5, 75.1])
    assert [l.entry_price for l in result.legs] == [76.5, 75.1]
    assert [l.average_entry_price for l in result.legs] == [76.5, 75.1]
    assert result.trade.price == 1.25  # the structure price stays authoritative on the Trade row


def test_leg_prices_partial_fill_or_bad_value_rejected():
    s = spread()
    with pytest.raises(TradeError, match="Enter a price for every leg"):
        enter_trade(s, 1.25, 5, "buy", None, LIVE, leg_prices=[76.5, None])
    with pytest.raises(TradeError, match="Leg 2 price"):
        enter_trade(s, 1.25, 5, "buy", None, LIVE, leg_prices=[76.5, -1.0])


# ---------- core.trade_entry: partial_exit_legs ----------


def two_leg_open(lots_a=10.0, lots_b=10.0, entry_a=75.0, entry_b=74.0):
    legs = [
        Leg(contract=contract("CLX26", month=11), ratio=1, lots=lots_a, entry_price=entry_a, average_entry_price=entry_a),
        Leg(contract=contract("CLZ26"), ratio=-1, lots=lots_b, entry_price=entry_b, average_entry_price=entry_b, direction="buy"),
    ]
    return Structure(name="Spread", structure_type=StructureType.SPREAD, products=["CL"], legs=legs, status=StructureStatus.OPEN)


def test_partial_exit_one_leg_fully_leaves_the_other_open_and_naked():
    s = two_leg_open()
    leg_a_id = s.legs[0].leg_id
    result = partial_exit_legs(s, {leg_a_id: (78.0, 10.0)}, "closing the front leg")
    assert len(result.trades) == 1 and result.trades[0].event_type == TradeEventType.PARTIAL_EXIT
    assert result.realized_pnl_by_leg[leg_a_id] == pytest.approx((78.0 - 75.0) * 1 * 10 * 1000)
    new_a, new_b = result.legs
    assert new_a.lots == 0 and new_a.is_naked is False  # closed legs are not "naked"
    assert new_b.lots == 10 and new_b.is_naked is True  # its hedge is gone
    assert result.status == StructureStatus.PARTIALLY_CLOSED


def test_partial_exit_partial_lots_of_one_leg():
    s = two_leg_open()
    leg_a_id = s.legs[0].leg_id
    result = partial_exit_legs(s, {leg_a_id: (78.0, 4.0)}, None)
    new_a = result.legs[0]
    assert new_a.lots == pytest.approx(6.0) and new_a.entry_price == 75.0  # entry kept for the remainder
    assert result.status == StructureStatus.OPEN  # both legs still have lots


def test_partial_exit_both_legs_at_different_prices_closes_the_structure():
    s = two_leg_open()
    leg_a_id, leg_b_id = s.legs[0].leg_id, s.legs[1].leg_id
    result = partial_exit_legs(s, {leg_a_id: (78.0, 10.0), leg_b_id: (73.0, 10.0)}, "full close via legs")
    assert {t.leg_id for t in result.trades} == {leg_a_id, leg_b_id}
    assert result.status == StructureStatus.CLOSED
    assert all(leg.lots == 0 for leg in result.legs)
    assert result.realized_pnl == pytest.approx(
        (78.0 - 75.0) * 1 * 10 * 1000 + (73.0 - 74.0) * -1 * 10 * 1000
    )


def test_partial_exit_validation():
    s = two_leg_open()
    leg_a_id = s.legs[0].leg_id
    with pytest.raises(TradeError, match="Select at least one leg"):
        partial_exit_legs(s, {}, None)
    with pytest.raises(TradeError, match="cannot exit more than its open"):
        partial_exit_legs(s, {leg_a_id: (78.0, 11.0)}, None)
    with pytest.raises(TradeError, match="exit price must be"):
        partial_exit_legs(s, {leg_a_id: (0.0, 5.0)}, None)
    with pytest.raises(TradeError, match="open structure"):
        partial_exit_legs(outright(), {leg_a_id: (78.0, 1.0)}, None)


# ---------- core.trade_entry: recompute_structure_from_trades (delete-trade support) ----------


def _mk_trade(structure, event_type, lots, price, direction="buy", timestamp=None, realized_pnl=None):
    kwargs = dict(
        structure_id=structure.structure_id, leg_id=structure.legs[0].leg_id, event_type=event_type,
        lots=lots, price=price, direction=direction, realized_pnl=realized_pnl,
    )
    if timestamp is not None:
        kwargs["timestamp"] = timestamp
    return Trade(**kwargs)


def test_recompute_deleting_the_only_trade_reverts_to_shell():
    s = outright(StructureStatus.OPEN, lots=10, entry=75.0)
    result = recompute_structure_from_trades(s, [])
    assert result.structure.status == StructureStatus.SHELL
    leg = result.structure.legs[0]
    assert leg.lots == 0 and leg.entry_price is None and leg.average_entry_price is None
    assert result.exit_trade_updates == {}


def test_recompute_deleting_the_first_trade_promotes_the_next_add():
    """Delete the original TRADE, leaving only a later ADD: that ADD becomes the new entry."""
    s = outright()
    remaining = [_mk_trade(s, TradeEventType.ADD, 10, 76.0)]
    result = recompute_structure_from_trades(s, remaining)
    assert result.structure.status == StructureStatus.OPEN
    leg = result.structure.legs[0]
    assert leg.lots == 10 and leg.average_entry_price == pytest.approx(76.0)


def test_recompute_deleting_an_add_shifts_the_average():
    s = outright()
    remaining = [_mk_trade(s, TradeEventType.TRADE, 10, 70.0)]  # the later ADD@80 was deleted
    result = recompute_structure_from_trades(s, remaining)
    leg = result.structure.legs[0]
    assert leg.lots == 10 and leg.average_entry_price == pytest.approx(70.0)


def test_recompute_reopens_when_the_exit_trade_is_excluded():
    """The surviving list already excludes the trade being deleted — deleting a FULL_EXIT
    means it's simply absent here, and the structure must come back OPEN, not CLOSED."""
    s = outright()
    remaining = [_mk_trade(s, TradeEventType.TRADE, 10, 70.0)]
    result = recompute_structure_from_trades(s, remaining)
    assert result.structure.status == StructureStatus.OPEN
    assert result.structure.closed_at is None and result.structure.close_trigger is None


_T1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_T2 = _T1 + timedelta(days=1)


def test_recompute_refreshes_a_surviving_exit_trades_realized_pnl():
    """The exit's stored realized_pnl is always recomputed from the replayed entry price,
    not trusted as-is — it may be stale once an earlier trade's history has changed."""
    s = outright()
    entry = _mk_trade(s, TradeEventType.TRADE, 10, 70.0, timestamp=_T1)
    exit_trade = _mk_trade(s, TradeEventType.FULL_EXIT, 10, 90.0, direction="sell", realized_pnl=999999.0, timestamp=_T2)
    result = recompute_structure_from_trades(s, [exit_trade, entry])  # order in the list must not matter
    assert result.structure.status == StructureStatus.CLOSED
    assert result.exit_trade_updates[exit_trade.trade_id][0] == pytest.approx(200000.0)  # (90-70)*10*1000


def test_recompute_lot_mismatch_before_a_surviving_exit_raises():
    """Deleting a trade whose lots the surviving FULL_EXIT still expects is rejected rather
    than silently recorded against the wrong lot count."""
    s = outright()
    entry = _mk_trade(s, TradeEventType.TRADE, 10, 70.0, timestamp=_T1)  # only 10 lots survive
    exit_trade = _mk_trade(s, TradeEventType.FULL_EXIT, 20, 90.0, direction="sell", timestamp=_T2)  # expects 20 lots
    with pytest.raises(TradeError, match="Partial exits are not supported"):
        recompute_structure_from_trades(s, [entry, exit_trade])


def test_recompute_rejects_roll_trades():
    s = outright()
    with pytest.raises(TradeError, match="Cannot recompute past"):
        recompute_structure_from_trades(s, [_mk_trade(s, TradeEventType.ROLL, 10, 70.0)])


def test_recompute_replays_a_surviving_partial_exit():
    s = outright()
    entry = _mk_trade(s, TradeEventType.TRADE, 10, 70.0, timestamp=_T1)
    partial = _mk_trade(s, TradeEventType.PARTIAL_EXIT, 4, 90.0, direction="sell", timestamp=_T2)
    result = recompute_structure_from_trades(s, [entry, partial])
    assert result.structure.status == StructureStatus.OPEN  # a single-leg structure has no "other" leg to go naked
    leg = result.structure.legs[0]
    assert leg.lots == pytest.approx(6.0)
    assert result.exit_trade_updates[partial.trade_id][0] == pytest.approx(80000.0)  # (90-70)*4*1000


def test_recompute_does_not_need_live_prices_for_a_multi_leg_structure():
    """Multi-leg allocation normally needs live prices; recompute must not depend on them."""
    s = spread()
    remaining = [_mk_trade(s, TradeEventType.TRADE, 5, 1.25)]
    result = recompute_structure_from_trades(s, remaining)
    assert result.structure.status == StructureStatus.OPEN
    assert structure_entry_price(result.structure) == pytest.approx(1.25)


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


def test_apply_leg_edits_corrects_entry_price():
    s = outright(StructureStatus.OPEN, lots=10, entry=75.0)
    result = apply_leg_edits(s, ["CLZ26"], [1], lambda symbol: None, entry_prices=[75.35])
    assert result.legs[0].entry_price == 75.35 and result.legs[0].average_entry_price == 75.35
    assert "entry price 75 -> 75.35 (manual correction)" in result.audit_note


def test_apply_leg_edits_entry_price_blank_or_unchanged_is_a_no_op():
    s = outright(StructureStatus.OPEN, lots=10, entry=75.0)
    blank = apply_leg_edits(s, ["CLZ26"], [1], lambda symbol: None, entry_prices=[None])
    assert blank.legs[0].entry_price == 75.0 and blank.audit_note == "edit saved (no changes)"
    same = apply_leg_edits(s, ["CLZ26"], [1], lambda symbol: None, entry_prices=[75.0])
    assert same.audit_note == "edit saved (no changes)"


def test_apply_leg_edits_entry_price_ignored_for_shell_leg():
    """A leg with no position yet has nothing to correct."""
    s = outright()
    result = apply_leg_edits(s, ["CLZ26"], [1], lambda symbol: None, entry_prices=[75.0])
    assert result.legs[0].entry_price is None


def test_apply_leg_edits_rejects_bad_entry_price():
    s = outright(StructureStatus.OPEN, lots=10, entry=75.0)
    with pytest.raises(StructureBuildError, match="entry price must be"):
        apply_leg_edits(s, ["CLZ26"], [1], lambda symbol: None, entry_prices=[-1.0])


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


def test_layout_edit_mode_entry_price_input_disabled_for_shell_leg():
    traded = structure_detail_layout(outright(StructureStatus.OPEN, 10, 75.0), {}, [], None, edit_mode=True)
    price_input = find(traded, {"type": "edit-leg-entry-price", "index": 0})
    assert price_input is not None and price_input.disabled is False and price_input.value == 75.0

    shell = structure_detail_layout(outright(), {}, [], None, edit_mode=True)
    shell_input = find(shell, {"type": "edit-leg-entry-price", "index": 0})
    assert shell_input is not None and shell_input.disabled is True


def test_layout_leg_prices_section_only_for_multi_leg():
    single = structure_detail_layout(outright(), {}, [], None)
    assert find(single, "btn-toggle-leg-prices") is None

    multi = structure_detail_layout(spread(), {"CLX26": {"price": 75.0}, "CLZ26": {"price": 74.0}}, [], None)
    assert find(multi, "btn-toggle-leg-prices") is not None
    assert find(multi, {"type": "trade-leg-price", "index": 0}) is not None
    assert find(multi, {"type": "trade-leg-price", "index": 1}) is not None


def test_layout_leg_exit_section_present_and_hidden_when_not_open():
    s = outright(StructureStatus.OPEN, 10, 75.0)
    opened = structure_detail_layout(s, {}, [], None)
    section = find(opened, "leg-exit-section")
    assert section is not None and section.style != {"display": "none"}
    assert find(opened, {"type": "exit-leg-selected", "index": s.legs[0].leg_id}) is not None

    shell = structure_detail_layout(outright(), {}, [], None)
    assert find(shell, "leg-exit-section").style == {"display": "none"}


def test_layout_exit_form_hides_the_simple_full_exit_once_legs_are_uneven():
    s = two_leg_open(lots_a=6.0, lots_b=10.0)
    text = str(structure_detail_layout(s, {}, [], None))
    assert "btn-confirm-exit" not in text
    assert "Exit Specific Legs" in text
    uniform = structure_detail_layout(two_leg_open(), {}, [], None)
    assert find(uniform, "btn-confirm-exit") is not None


def test_status_badge_partially_closed_is_distinct():
    s = two_leg_open(lots_a=0.0, lots_b=10.0).model_copy(update={"status": StructureStatus.PARTIALLY_CLOSED})
    text = str(structure_detail_layout(s, {}, [], None))
    assert "PARTIALLY CLOSED" in text


def test_legs_table_flags_a_naked_leg():
    s = two_leg_open(lots_a=0.0, lots_b=10.0)
    legs = list(s.legs)
    legs[1] = legs[1].model_copy(update={"is_naked": True})
    s = s.model_copy(update={"legs": legs})
    text = str(structure_detail_layout(s, {}, [], None))
    assert "naked" in text


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
    assert dc.update_live_labels(LIVE_STORE, sid) == ("Live: 76.00", "Live: 76.00", "Current live price: 76.00")
    for bad in (lambda: dc.fill_live_price(None, sid, LIVE_STORE), lambda: dc.fill_live_price(1, sid, {}),
                lambda: dc.fill_exit_all_lots(1, "missing"), lambda: dc.toggle_add_trade(None, False)):
        with pytest.raises(PreventUpdate):
            bad()
    assert dc.toggle_add_trade(1, False) is True


def test_pnl_previews(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    assert "+$10,000" in str(dc.preview_entry_pnl(75.0, 10, "buy", sid, LIVE_STORE))
    assert "-$10,000" in str(dc.preview_entry_pnl(75.0, 10, "sell", sid, LIVE_STORE))
    assert dc.preview_entry_pnl(None, 10, "buy", sid, LIVE_STORE) == ""
    assert "unavailable" in dc.preview_entry_pnl(75.0, 10, "buy", sid, {})
    assert "+$20,000" in str(dc.preview_exit_pnl(77.0, 10, sid))
    assert dc.preview_exit_pnl(77.0, None, sid) == ""


def test_confirm_trade_entry_opens_shell_and_writes_audit(repo):
    sid = save(repo, outright())
    body, rows, is_open, message, icon, header = dc.confirm_trade_entry(1, 75.0, 10, "buy", "n", None, None, None, sid, LIVE_STORE, *GRID)
    saved = repo.get_structure(sid)
    assert saved.status == StructureStatus.OPEN and saved.legs[0].lots == 10 and saved.legs[0].entry_price == 75.0
    assert "trade entered: 10 lots at 75" in saved.notes
    (trade,) = repo.get_trades_for_structure(sid)
    assert trade.event_type == TradeEventType.TRADE and trade.price == 75.0
    assert icon == "success" and "Trade entered: 10 lots at 75" in message and [r["status"] for r in rows] == ["OPEN"]
    assert "Trade History" in str(body)


def test_confirm_trade_entry_validation_error_saves_nothing(repo):
    sid = save(repo, outright())
    body, rows, is_open, message, icon, header = dc.confirm_trade_entry(1, 75.0, 0, "buy", None, None, None, None, sid, LIVE_STORE, *GRID)
    assert body is dc.no_update and rows is dc.no_update and icon == "danger"
    assert repo.get_trades_for_structure(sid) == [] and repo.get_structure(sid).status == StructureStatus.SHELL


def test_multi_leg_trade_entry_persists_consistent_legs(repo):
    sid = save(repo, spread())
    dc.confirm_trade_entry(1, 1.25, 5, "buy", None, None, None, None, sid, LIVE_STORE, *GRID)
    saved = repo.get_structure(sid)
    assert structure_entry_price(saved) == pytest.approx(1.25)
    assert calculate_portfolio_pnl([saved], {}, {"CLX26": 75.0, "CLZ26": 73.75}, [])["total_unrealized"] == pytest.approx(0.0)


def test_confirm_trade_entry_with_custom_leg_prices(repo):
    sid = save(repo, spread())
    dc.confirm_trade_entry(1, 1.25, 5, "buy", None, None, None, [76.5, 75.25], sid, LIVE_STORE, *GRID)
    saved = repo.get_structure(sid)
    assert [leg.entry_price for leg in saved.legs] == [76.5, 75.25]
    assert (saved.legs[0].average_entry_price, saved.legs[1].average_entry_price) == (76.5, 75.25)


def test_confirm_trade_entry_leg_prices_partial_fill_is_rejected(repo):
    sid = save(repo, spread())
    body, rows, is_open, message, icon, header = dc.confirm_trade_entry(
        1, 1.25, 5, "buy", None, None, None, [76.5, None], sid, LIVE_STORE, *GRID
    )
    assert icon == "danger" and "Enter a price for every leg" in str(message)
    assert repo.get_trades_for_structure(sid) == []


def test_toggle_leg_prices_and_fill_live_leg_price(repo, monkeypatch):
    sid = save(repo, spread())
    assert dc.toggle_leg_prices(1, False) is True
    with pytest.raises(PreventUpdate):
        dc.toggle_leg_prices(None, False)

    trigger(monkeypatch, {"type": "btn-use-live-leg-price", "index": 1})
    assert dc.fill_live_leg_price(1, sid, LIVE_STORE) == 76.0  # legs[1] is CLZ26
    trigger(monkeypatch, {"type": "btn-use-live-leg-price", "index": 1})
    with pytest.raises(PreventUpdate):
        dc.fill_live_leg_price(1, sid, {})  # no live price available


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


def test_leg_exit_confirmation_and_execution(repo):
    sid = save(repo, spread())
    dc.confirm_trade_entry(1, 1.25, 5, "buy", None, None, None, None, sid, LIVE_STORE, *GRID)
    leg_a, leg_b = repo.get_structure(sid).legs
    ids = [{"type": "exit-leg-selected", "index": leg_a.leg_id}, {"type": "exit-leg-selected", "index": leg_b.leg_id}]

    # Only leg A selected: exit its full 5 lots at 76.0.
    is_open, body, disabled = dc.open_leg_exit_confirmation(1, [["on"], []], [5, 5], [76.0, None], ids, sid)
    assert is_open is True and disabled is False and "realized" in str(body).lower()
    assert repo.get_structure(sid).legs[0].lots == 5  # confirmation alone never exits

    body, rows, is_open2, message, icon, header, confirm_open = dc.execute_leg_exit(
        1, [["on"], []], [5, 5], [76.0, None], ids, "closing front leg", sid, *GRID
    )
    saved = repo.get_structure(sid)
    assert saved.legs[0].lots == 0 and saved.legs[1].lots == 5
    assert saved.status == StructureStatus.PARTIALLY_CLOSED
    assert saved.legs[1].is_naked is True
    trades = repo.get_trades_for_structure(sid)
    partial = [t for t in trades if t.event_type == TradeEventType.PARTIAL_EXIT]
    assert len(partial) == 1 and partial[0].leg_id == leg_a.leg_id and partial[0].lots == 5
    assert confirm_open is False and icon == "success" and "exited" in message.lower()


def test_leg_exit_closing_every_remaining_leg_closes_the_structure(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    leg_id = repo.get_structure(sid).legs[0].leg_id
    ids = [{"type": "exit-leg-selected", "index": leg_id}]
    dc.execute_leg_exit(1, [["on"]], [10], [77.0], ids, None, sid, *GRID)
    saved = repo.get_structure(sid)
    assert saved.status == StructureStatus.CLOSED and saved.close_trigger == "manual" and saved.closed_at
    assert saved.legs[0].lots == 0


def test_leg_exit_validation_rejects_nothing_selected(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    leg_id = repo.get_structure(sid).legs[0].leg_id
    ids = [{"type": "exit-leg-selected", "index": leg_id}]
    is_open, body, disabled = dc.open_leg_exit_confirmation(1, [[]], [10], [77.0], ids, sid)
    assert disabled is True and "Select at least one leg" in str(body)

    body, rows, is_open2, message, icon, header, confirm_open = dc.execute_leg_exit(
        1, [[]], [10], [77.0], ids, None, sid, *GRID
    )
    assert icon == "danger" and confirm_open is False
    assert repo.get_trades_for_structure(sid) == []


def test_cancel_leg_exit():
    assert dc.cancel_leg_exit(1) is False
    with pytest.raises(PreventUpdate):
        dc.cancel_leg_exit(None)


def test_toggle_leg_exit_and_fill_live_exit_leg_price(repo, monkeypatch):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    leg_id = repo.get_structure(sid).legs[0].leg_id
    assert dc.toggle_leg_exit(1, False) is True
    with pytest.raises(PreventUpdate):
        dc.toggle_leg_exit(None, False)

    trigger(monkeypatch, {"type": "btn-use-live-exit-leg-price", "index": leg_id})
    assert dc.fill_live_exit_leg_price(1, sid, LIVE_STORE) == 76.0  # CLZ26
    with pytest.raises(PreventUpdate):
        dc.fill_live_exit_leg_price(1, sid, {})


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
    body, is_open, message, icon, header = dc.save_edit(1, ["CLF27"], [1], None, sid, LIVE_STORE)
    saved = repo.get_structure(sid)
    assert saved.legs[0].contract.symbol == "CLF27" and saved.legs[0].lots == 10
    assert "edited: leg 1 symbol CLZ26 -> CLF27" in saved.notes and icon == "success"
    body, is_open, message, icon, header = dc.save_edit(1, ["bad"], [1], None, sid, LIVE_STORE)
    assert body is dc.no_update and icon == "danger"


def test_audit_notes_can_accumulate_beyond_500_chars(repo):
    sid = save(repo, outright())
    for i in range(12):
        repo.update_structure_legs(sid, repo.get_structure(sid).legs, f"audit line number {i} " + "x" * 30)
    assert len(repo.get_structure(sid).notes) > 500


# ---------- delete structure / delete trade ----------


def trigger(monkeypatch, triggered_id, value=1):
    """Simulate a pattern-matched-button click by setting dc.callback_context directly."""

    class Ctx:
        pass

    ctx = Ctx()
    ctx.triggered_id = triggered_id
    ctx.triggered = [{"prop_id": "x.n_clicks", "value": value}]
    monkeypatch.setattr(dc, "callback_context", ctx)


def trigger_delete_btn(monkeypatch, trade_id, value=1):
    """Simulate the pattern-matched {"type": "trade-delete-btn", "index": trade_id} click."""

    class Ctx:
        pass

    ctx = Ctx()
    ctx.triggered_id = {"type": "trade-delete-btn", "index": trade_id}
    ctx.triggered = [{"prop_id": "x.n_clicks", "value": value}]
    monkeypatch.setattr(dc, "callback_context", ctx)


DELETE_ARGS = ("active", "all", "name", {})  # status_filter, product_filter, sort_by, portfolio_pnl


def test_open_delete_structure_confirm(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    repo.save_trade(Trade(structure_id=sid, leg_id=repo.get_structure(sid).legs[0].leg_id,
                          event_type=TradeEventType.TRADE, lots=10, price=75.0, direction="buy"))

    is_open, body, pending = dc.open_delete_structure_confirm(1, sid)
    assert is_open is True and "Outright" in str(body) and "1 trade(s)" in str(body)
    assert pending == {"kind": "structure", "id": sid}
    with pytest.raises(PreventUpdate):
        dc.open_delete_structure_confirm(None, sid)


def test_open_delete_trade_confirm(repo, monkeypatch):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    trade = Trade(structure_id=sid, leg_id=repo.get_structure(sid).legs[0].leg_id,
                  event_type=TradeEventType.TRADE, lots=10, price=75.0, direction="buy")
    repo.save_trade(trade)

    trigger_delete_btn(monkeypatch, trade.trade_id)
    is_open, body, pending = dc.open_delete_trade_confirm([1], sid)
    assert is_open is True and "recomputed" in str(body)
    assert pending == {"kind": "trade", "id": trade.trade_id}

    trigger_delete_btn(monkeypatch, trade.trade_id, value=None)  # newly-created button, not a real click
    with pytest.raises(PreventUpdate):
        dc.open_delete_trade_confirm([None], sid)

    trigger_delete_btn(monkeypatch, "unknown-trade-id")
    with pytest.raises(PreventUpdate):
        dc.open_delete_trade_confirm([1], sid)


def test_cancel_delete():
    assert dc.cancel_delete(1) == (False, None)


# ---------- repeating stop-loss / target alerts ----------


def trigger_stop_alert_btn(monkeypatch, trade_id, value=1):
    """Simulate the pattern-matched {"type": "trade-stop-alert-btn", "index": trade_id} click."""

    class Ctx:
        pass

    ctx = Ctx()
    ctx.triggered_id = {"type": "trade-stop-alert-btn", "index": trade_id}
    ctx.triggered = [{"prop_id": "x.n_clicks", "value": value}]
    monkeypatch.setattr(dc, "callback_context", ctx)


def _open_trade_with_stop(repo, stop=73.0):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    trade = Trade(structure_id=sid, leg_id=repo.get_structure(sid).legs[0].leg_id,
                  event_type=TradeEventType.TRADE, lots=10, price=75.0, direction="buy", stop_loss_price=stop)
    repo.save_trade(trade)
    return sid, trade


def test_render_shows_stop_alerts_button_only_while_alert_is_active(repo, monkeypatch):
    sid, trade = _open_trade_with_stop(repo)
    fresh = Container(repository=repo, alert_manager=AlertManager(repo))
    monkeypatch.setattr(dc, "container", fresh)

    live_below_stop = {"CLZ26": {"price": 72.0}}
    body = dc.render_structure_detail(sid, live_below_stop)
    assert "Stop Alerts" in str(body)

    live_above_stop = {"CLZ26": {"price": 76.0}}
    body = dc.render_structure_detail(sid, live_above_stop)
    assert "Stop Alerts" not in str(body)


def test_stop_trade_alert_button_stops_future_resends(repo, monkeypatch):
    sid, trade = _open_trade_with_stop(repo)
    fresh = Container(repository=repo, alert_manager=AlertManager(repo))
    monkeypatch.setattr(dc, "container", fresh)

    trigger_stop_alert_btn(monkeypatch, trade.trade_id)
    live_below_stop = {"CLZ26": {"price": 72.0}}
    body, toast_open, message, icon, header = dc.stop_trade_alert([1], sid, live_below_stop)
    assert toast_open is True and "stopped" in message.lower()
    assert "Stop Alerts" not in str(body)  # button hides once the alert is stopped
    assert repo.get_setting(f"alert_stopped_{trade.trade_id}_stop") is True

    trigger_stop_alert_btn(monkeypatch, trade.trade_id, value=None)  # newly-created button, not a real click
    with pytest.raises(PreventUpdate):
        dc.stop_trade_alert([None], sid, live_below_stop)
    with pytest.raises(PreventUpdate):
        dc.cancel_delete(None)


def test_execute_delete_removes_structure_and_closes_both_modals(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    pending = {"kind": "structure", "id": sid}

    confirm_open, detail_open, body, rows, cleared, toast_open, message, icon, header = dc.execute_delete(
        1, pending, sid, LIVE_STORE, *DELETE_ARGS
    )
    assert confirm_open is False and detail_open is False and cleared is None
    assert icon == "success" and "deleted" in message
    assert repo.get_structure(sid) is None
    assert rows == []


def test_execute_delete_removes_trade_and_recomputes_pnl(repo):
    sid = save(repo, outright())
    dc.confirm_trade_entry(1, 70.0, 10, "buy", None, None, None, None, sid, LIVE_STORE, *GRID)
    (trade_to_delete,) = repo.get_trades_for_structure(sid)
    dc.confirm_trade_entry(1, 80.0, 10, "buy", None, None, None, None, sid, LIVE_STORE, *GRID)  # ADD@80

    pending = {"kind": "trade", "id": trade_to_delete.trade_id}
    confirm_open, detail_open, body, rows, cleared, toast_open, message, icon, header = dc.execute_delete(
        1, pending, sid, LIVE_STORE, *DELETE_ARGS
    )

    assert confirm_open is False and detail_open is dc.no_update and cleared is None
    assert icon == "success" and "recomputed" in message
    saved = repo.get_structure(sid)
    assert saved.legs[0].lots == 10 and saved.legs[0].average_entry_price == pytest.approx(80.0)
    (remaining_trade,) = repo.get_trades_for_structure(sid)
    assert remaining_trade.price == 80.0
    assert "Trade History" in str(body)


def test_execute_delete_no_pending_or_no_click_prevents_update():
    with pytest.raises(PreventUpdate):
        dc.execute_delete(None, {"kind": "trade", "id": "x"}, "sid", {}, *DELETE_ARGS)
    with pytest.raises(PreventUpdate):
        dc.execute_delete(1, None, "sid", {}, *DELETE_ARGS)


def test_execute_delete_surfaces_trade_error_without_deleting(repo):
    """Deleting a trade that would break a surviving exit's lot count is rejected, not silently applied."""
    sid = save(repo, outright())
    leg_id = repo.get_structure(sid).legs[0].leg_id
    entry = Trade(structure_id=sid, leg_id=leg_id, event_type=TradeEventType.TRADE, lots=10, price=70.0, direction="buy")
    exit_trade = Trade(structure_id=sid, leg_id=leg_id, event_type=TradeEventType.FULL_EXIT, lots=20, price=90.0,
                       direction="sell", realized_pnl=200000.0)
    repo.save_trade(entry)
    repo.save_trade(exit_trade)

    pending = {"kind": "trade", "id": entry.trade_id}
    result = dc.execute_delete(1, pending, sid, LIVE_STORE, *DELETE_ARGS)
    assert result[-2] == "danger"  # icon is the second-to-last output; nothing was deleted
    assert [t.trade_id for t in repo.get_trades_for_structure(sid)] == [entry.trade_id, exit_trade.trade_id]


# ---------- direction, stop / target, reuse ----------


def test_sell_entry_stores_direction_on_legs_and_flips_pnl():
    result = enter_trade(outright(), 0.45, 10, "sell", None, {})
    assert result.legs[0].direction == "sell" and result.trade.direction == "sell"
    sold = outright().model_copy(update={"legs": result.legs, "status": StructureStatus.OPEN})
    assert structure_pnl(sold, 0.45, 0.37, 10) == pytest.approx(800.0)
    assert structure_pnl(sold, 0.45, 0.50, 10) == pytest.approx(-500.0)
    engine = calculate_portfolio_pnl([sold], {}, {"CLZ26": 0.37}, [])["total_unrealized"]
    assert engine == pytest.approx(800.0)


def test_sell_exit_realizes_profit_and_closes_with_a_buy():
    result = enter_trade(outright(), 0.45, 10, "sell", None, {})
    sold = outright().model_copy(update={"legs": result.legs, "status": StructureStatus.OPEN})
    closed = exit_structure(sold, 0.37, 10, None)
    assert closed.realized_pnl == pytest.approx(800.0)
    assert closed.trade.direction == "buy" and closed.trade.realized_pnl == pytest.approx(800.0)


def test_add_must_match_the_position_direction():
    s = outright(StructureStatus.OPEN, lots=10, entry=75.0)
    with pytest.raises(TradeError, match="an add must be a buy"):
        enter_trade(s, 76.0, 5, "sell", None, {})


def test_stop_and_target_are_stored_and_validated_by_direction():
    result = enter_trade(outright(), 75.0, 10, "buy", None, {}, stop_loss_price=73.0, target_price=80.0)
    assert (result.trade.stop_loss_price, result.trade.target_price) == (73.0, 80.0)
    sell = enter_trade(outright(), 75.0, 10, "sell", None, {}, stop_loss_price=77.0, target_price=70.0)
    assert (sell.trade.stop_loss_price, sell.trade.target_price) == (77.0, 70.0)
    with pytest.raises(TradeError, match="stop loss must be below"):
        enter_trade(outright(), 75.0, 10, "buy", None, {}, stop_loss_price=76.0)
    with pytest.raises(TradeError, match="target must be below"):
        enter_trade(outright(), 75.0, 10, "sell", None, {}, target_price=76.0)


def test_trade_entry_persists_direction_and_alert_levels(repo):
    sid = save(repo, outright())
    dc.confirm_trade_entry(1, 0.45, 10, "sell", None, 0.5, 0.37, None, sid, LIVE_STORE, *GRID)
    saved = repo.get_structure(sid)
    assert saved.legs[0].direction == "sell"
    (trade,) = repo.get_trades_for_structure(sid)
    assert (trade.stop_loss_price, trade.target_price, trade.direction) == (0.5, 0.37, "sell")


def test_detail_layout_has_price_alert_inputs():
    layout = structure_detail_layout(outright(), {"CLZ26": {"price": 76.0}}, [], None)
    for component_id in ("trade-stop-loss-price", "trade-target-price", "trade-alert-live-reference"):
        assert find(layout, component_id) is not None
    assert "Current live price: 76.00" in str(layout)


def closed_structure(repo):
    s = outright(StructureStatus.CLOSED, lots=0.0, entry=75.0).model_copy(update={"close_trigger": "manual"})
    return save(repo, s)


def test_detail_layout_offers_reuse_only_for_closed_structures():
    closed = outright(StructureStatus.CLOSED, 0, 75.0)
    layout = structure_detail_layout(closed, {}, [], None)
    for component_id in ("btn-reuse-structure", "reuse-structure-name", "btn-confirm-reuse", "reuse-collapse"):
        assert find(layout, component_id) is not None
    assert find(structure_detail_layout(outright(), {}, [], None), "btn-reuse-structure") is None


def test_reuse_structure_from_detail_creates_shell_and_closes_modal(repo):
    sid = closed_structure(repo)
    is_open, toast_open, message, icon, header = dc.reuse_structure(1, sid, "Fresh copy")
    assert is_open is False and icon == "success" and "Fresh copy" in message
    shells = repo.get_all_structures(status_filter=[StructureStatus.SHELL])
    assert [s.name for s in shells] == ["Fresh copy"]
    assert shells[0].structure_id != sid and shells[0].legs[0].lots == 0 and shells[0].legs[0].entry_price is None
    assert repo.get_structure(sid).status == StructureStatus.CLOSED  # the source is untouched


def test_reuse_structure_default_name_and_refuses_open_structures(repo):
    sid = closed_structure(repo)
    dc.reuse_structure(1, sid, None)
    assert [s.name for s in repo.get_all_structures(status_filter=[StructureStatus.SHELL])] == ["Outright (reuse)"]
    open_id = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    is_open, toast_open, message, icon, header = dc.reuse_structure(1, open_id, None)
    assert is_open is dc.no_update and icon == "danger"
    with pytest.raises(PreventUpdate):
        dc.toggle_reuse_panel(None, False)
    assert dc.toggle_reuse_panel(1, False) is True


# ---------- rename ----------


def test_layout_has_rename_ids():
    layout = structure_detail_layout(outright(), {}, [], None)
    for component_id in ("btn-rename-structure", "rename-collapse", "rename-structure-input", "btn-confirm-rename", "btn-cancel-rename"):
        assert find(layout, component_id) is not None


def test_toggle_and_cancel_rename_panel():
    with pytest.raises(PreventUpdate):
        dc.toggle_rename_panel(None, False)
    assert dc.toggle_rename_panel(1, False) is True
    with pytest.raises(PreventUpdate):
        dc.cancel_rename(None)
    assert dc.cancel_rename(1) is False


def test_save_rename_updates_name_and_refreshes_grid(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    is_open, body, rows, toast_open, message, icon, header = dc.save_rename(1, "New Name", sid, LIVE_STORE, *GRID[:4])
    assert is_open is False and icon == "success" and "New Name" in message
    assert repo.get_structure(sid).name == "New Name"
    assert [r["name"] for r in rows] == ["New Name"]
    assert "New Name" in str(body)


def test_save_rename_rejects_empty_or_too_long_name(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    is_open, body, rows, toast_open, message, icon, header = dc.save_rename(1, "   ", sid, LIVE_STORE, *GRID[:4])
    assert icon == "danger" and "cannot be empty" in message
    assert repo.get_structure(sid).name == "Outright"

    is_open, body, rows, toast_open, message, icon, header = dc.save_rename(1, "x" * 101, sid, LIVE_STORE, *GRID[:4])
    assert icon == "danger" and "100 characters" in message


def test_save_rename_noop_when_name_unchanged(repo):
    sid = save(repo, outright(StructureStatus.OPEN, 10, 75.0))
    is_open, body, rows, toast_open, message, icon, header = dc.save_rename(1, "Outright", sid, LIVE_STORE, *GRID[:4])
    assert is_open is False and icon == "success" and "unchanged" in message.lower()
    assert body is dc.no_update and rows is dc.no_update


def test_save_rename_requires_a_click():
    with pytest.raises(PreventUpdate):
        dc.save_rename(None, "New Name", "sid", {}, *GRID[:4])


def test_closed_grid_reuse_action_refreshes_active_grid(repo):
    sid = closed_structure(repo)
    rows, toast_open, message, icon, header = sc.handle_closed_structure_action(
        {"value": {"action": "reuse", "structure_id": sid}, "rowId": sid}, "active", "all", "name", {}, {}
    )
    assert [r["status"] for r in rows] == ["SHELL"] and "ready as new shell" in message
    for bad in (None, {"value": {"action": "other", "structure_id": sid}}, {"value": {"action": "reuse"}}):
        with pytest.raises(PreventUpdate):
            sc.handle_closed_structure_action(bad, "active", "all", "name", {}, {})
    rows, toast_open, message, icon, header = sc.handle_closed_structure_action(
        {"value": {"action": "reuse", "structure_id": "missing"}}, "active", "all", "name", {}, {}
    )
    assert rows is sc.no_update and icon == "danger"
