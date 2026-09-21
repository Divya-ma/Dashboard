"""Tests for core.structure_view, the Structures layout and ui.callbacks.structures_callbacks."""

from datetime import datetime, timedelta, timezone

import dash_bootstrap_components as dbc
import pytest
from dash.exceptions import PreventUpdate

from core.models import (
    Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType,
)
from core.structure_view import (
    build_active_rows, build_closed_rows, prices_from_store, statuses_for_filter,
    structure_entry_price, structure_exit_price, structure_live_price,
)
from db.repository import Repository
from ui.callbacks import structures_callbacks as sc
from ui.container import Container
from ui.layouts.structures import structures_layout


def contract(product="CL", month=12, symbol=None):
    symbol = symbol or f"{product}{'FGHJKMNQUVXZ'[month - 1]}26"
    return Contract(product=product, contract_month=month, contract_year=2026, symbol=symbol,
                    multiplier=1000.0, tick_size=0.01, tick_value=10.0)


def make_structure(name="S", legs=None, status=StructureStatus.OPEN, stype=StructureType.OUTRIGHT, **kw):
    legs = legs or [Leg(contract=contract(), ratio=1, lots=5, entry_price=75.0)]
    return Structure(name=name, structure_type=stype, products=["CL"], legs=legs, status=status, **kw)


def spread(status=StructureStatus.OPEN, ratios=(1, -1), entries=(75.0, 74.0), lots=5):
    legs = [
        Leg(contract=contract(month=11), ratio=ratios[0], lots=lots, entry_price=entries[0]),
        Leg(contract=contract(month=12), ratio=ratios[1], lots=lots, entry_price=entries[1]),
    ]
    return make_structure("Spread", legs, status, StructureType.SPREAD)


def shell_structure(name="shell"):
    return make_structure(name, status=StructureStatus.SHELL, legs=[Leg(contract=contract(), ratio=1)])


# ---------- prices ----------


def test_structure_price_matches_quote_for_spread_either_direction():
    assert structure_entry_price(spread()) == pytest.approx(1.0)
    assert structure_entry_price(spread(ratios=(-1, 1))) == pytest.approx(1.0)  # short spread, same quote


def test_structure_price_fly_and_outright_short():
    legs = [Leg(contract=contract(month=m), ratio=r, lots=1, entry_price=p)
            for m, r, p in [(10, 1, 76.0), (11, -2, 75.0), (12, 1, 74.5)]]
    assert structure_entry_price(make_structure(legs=legs, stype=StructureType.FLY)) == pytest.approx(0.5)
    short = make_structure(legs=[Leg(contract=contract(), ratio=-1, lots=1, entry_price=75.0)])
    assert structure_entry_price(short) == pytest.approx(75.0)


def test_shell_has_no_entry_price_and_missing_live_price_is_none():
    assert structure_entry_price(shell_structure()) is None
    assert structure_live_price(spread(), {"CLX26": 75.5}) is None
    assert structure_live_price(spread(), {"CLX26": 75.5, "CLZ26": 74.0}) == pytest.approx(1.5)


def test_prices_from_store():
    assert prices_from_store({"CLZ26": {"price": 70.0}, "bad": {}}) == {"CLZ26": 70.0}
    assert prices_from_store(None) == {}


# ---------- filters / rows ----------


def test_statuses_for_filter():
    assert StructureStatus.SHELL in statuses_for_filter("active")
    assert statuses_for_filter("shell") == [StructureStatus.SHELL]
    assert StructureStatus.SHELL not in statuses_for_filter("open")
    assert statuses_for_filter("nonsense") == statuses_for_filter("active")


def pnl(unrealized, realized=0.0):
    return {"unrealized": unrealized, "realized": realized, "total": unrealized + realized}


def test_active_rows_use_store_pnl_and_sort_shells_last():
    a, b, shell = make_structure("a"), make_structure("b"), shell_structure()
    store = {a.structure_id: pnl(100.0), b.structure_id: pnl(-50.0, 300.0)}
    rows = build_active_rows([shell, a, b], store, {"CLZ26": 76.0})
    assert [r["name"] for r in rows] == ["b", "a", "shell"]
    assert rows[0]["total"] == 250.0 and rows[0]["live_price"] == 76.0
    assert rows[2]["total"] is None and rows[2]["status"] == "SHELL" and rows[2]["structure_price"] is None
    assert rows[0]["lots"] == 5 and rows[0]["type"] == "Outright" and rows[0]["status"] == "OPEN"


def test_active_rows_sort_options_and_product_filter():
    a, b = make_structure("beta"), make_structure("Alpha")
    brn = make_structure(
        "brn", legs=[Leg(contract=contract("BRN", symbol="BRNZ26"), ratio=1, lots=1, entry_price=80.0)]
    )
    store = {a.structure_id: pnl(1.0), b.structure_id: pnl(2.0), brn.structure_id: pnl(3.0)}

    def names(**kwargs):
        return [r["name"] for r in build_active_rows([a, b, brn], store, {}, **kwargs)]

    assert names(sort_by="name") == ["Alpha", "beta", "brn"]
    assert names(product_filter="BRN") == ["brn"]
    assert names(sort_by="unrealized_pnl") == ["brn", "Alpha", "beta"]
    assert build_active_rows([a], store, {}, product_filter="G") == []


def test_closed_rows():
    now = datetime.now(timezone.utc)
    s = make_structure("done", status=StructureStatus.CLOSED, close_trigger="alert",
                       created_at=now - timedelta(days=9), closed_at=now - timedelta(days=2))
    leg = s.legs[0]
    trades = [
        Trade(structure_id=s.structure_id, leg_id=leg.leg_id, event_type=TradeEventType.TRADE,
              lots=5, price=75.0, direction="buy"),
        Trade(structure_id=s.structure_id, leg_id=leg.leg_id, event_type=TradeEventType.FULL_EXIT,
              lots=5, price=77.0, direction="sell", realized_pnl=10000.0),
    ]
    older = make_structure("older", status=StructureStatus.CLOSED,
                           created_at=now - timedelta(days=30), closed_at=now - timedelta(days=20))
    rows = build_closed_rows([older, s], {s.structure_id: trades})
    assert [r["name"] for r in rows] == ["done", "older"]
    row = rows[0]
    assert (row["entry_price"], row["exit_price"], row["realized"], row["lots"]) == (75.0, 77.0, 10000.0, 5)
    assert row["days_held"] == 7 and row["close_trigger"] == "Alert" and row["closed_at"]
    assert structure_exit_price(s, []) is None


# ---------- layout ----------


def find_component(node, component_id):
    if getattr(node, "id", None) == component_id:
        return node
    children = getattr(node, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and not isinstance(child, str):
            found = find_component(child, component_id)
            if found is not None:
                return found
    return None


def test_layout_has_required_ids_and_closed_collapsed():
    layout = structures_layout()
    for component_id in [
        "btn-new-structure", "filter-structure-status", "filter-structure-product", "filter-structure-sort",
        "structures-active-grid", "structures-closed-grid", "btn-toggle-closed",
        "structures-closed-collapse", "modal-structure-detail", "modal-structure-detail-body",
    ]:
        assert find_component(layout, component_id) is not None, component_id
    assert find_component(layout, "structures-closed-collapse").is_open is False
    assert find_component(layout, "modal-structure-detail").size == "xl"
    assert isinstance(find_component(layout, "structures-closed-collapse"), dbc.Collapse)


# ---------- callbacks ----------


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture(autouse=True)
def env(repo, monkeypatch):
    fresh = Container(repository=repo)
    monkeypatch.setattr(sc, "container", fresh)
    return fresh


def save(repo, structure):
    for leg in structure.legs:
        repo.save_contract(leg.contract)
    repo.save_structure(structure)


def test_update_active_structures_filters_by_status(repo):
    open_s, shell = make_structure("open"), shell_structure()
    closed = make_structure("closed", status=StructureStatus.CLOSED)
    for s in (open_s, shell, closed):
        save(repo, s)
    store = {"per_structure": {open_s.structure_id: pnl(10.0)}}

    def names(status_filter):
        return {r["name"] for r in sc.update_active_structures(store, status_filter, "all", "name", {})}

    assert names("active") == {"open", "shell"}
    assert names("shell") == {"shell"}
    assert names("open") == {"open"}


def test_update_closed_structures_label_and_toggle(repo):
    save(repo, make_structure("c1", status=StructureStatus.CLOSED))
    rows, label = sc.update_closed_structures("/structures", 0, False)
    assert [r["name"] for r in rows] == ["c1"] and label == "📁 Show Closed Structures (1)"
    assert sc.update_closed_structures("/structures", 0, True)[1] == "📁 Hide Closed Structures (1)"
    assert sc.toggle_closed(1, False) is True and sc.toggle_closed(2, True) is False


def cell(action, sid="abc"):
    return {"value": {"action": action, "structure_id": sid}, "rowId": sid, "colId": "actions"}


def test_handle_structure_action():
    assert sc.handle_structure_action(cell("view")) == (True, "abc")
    is_open, sid = sc.handle_structure_action(cell("exit"))
    assert is_open is sc.no_update and sid == "abc"
    assert sc.handle_structure_action({"value": {"action": "view"}, "rowId": "row1"}) == (True, "row1")
    for bad in (None, {}, cell("bogus"), {"value": {"action": "view"}}):
        with pytest.raises(PreventUpdate):
            sc.handle_structure_action(bad)


def test_close_and_render_detail(repo):
    assert sc.close_detail_modal(1) is False
    with pytest.raises(PreventUpdate):
        sc.close_detail_modal(None)
    s = make_structure("Detail me")
    save(repo, s)
    assert "Detail me" in str(sc.render_structure_detail(s.structure_id))
    assert "not found" in str(sc.render_structure_detail("missing"))
    assert sc.render_structure_detail(None) == ""
