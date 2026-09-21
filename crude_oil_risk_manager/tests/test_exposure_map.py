"""Tests for core.exposure_map, the Exposure Map layout builders and its callbacks."""

import logging

import pytest
from dash.exceptions import PreventUpdate

from core.exposure_map import build_exposure_map, concentration_warnings
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from db.repository import Repository
from ui.callbacks import exposure_callbacks as ec
from ui.container import Container
from ui.layouts.exposure_tab import (
    FLAT_TEXT,
    NO_EXPOSURE_TEXT,
    NO_WARNINGS_TEXT,
    build_alerts,
    build_bar_figure,
    build_table_rows,
    exposure_layout,
)
from ui.layouts.shell import COLORS


def contract(symbol):
    product = "BRN" if symbol.startswith("BRN") else "CL"
    return Contract(product=product, contract_month=12, contract_year=2026, symbol=symbol,
                    multiplier=1000, tick_size=0.01, tick_value=10)


def structure(name, legs, status=StructureStatus.OPEN):
    """legs: [(symbol, ratio, direction, lots)]"""
    built = [Leg(contract=contract(s), ratio=r, direction=d, lots=lots, entry_price=1.0) for s, r, d, lots in legs]
    return Structure(name=name, structure_type=StructureType.CUSTOM, products=["CL"], legs=built, status=status)


def by_symbol(rows):
    return {row["symbol"]: row for row in rows}


# ---------- aggregation ----------


def test_buy_and_sell_signs_and_ratio():
    rows = by_symbol(build_exposure_map([
        structure("long", [("CLZ26", 1, "buy", 10)]),
        structure("short", [("CLF27", 1, "sell", 4)]),
        structure("spread", [("CLG27", 1, "buy", 3), ("CLH27", -1, "buy", 3)]),
        structure("fly", [("CLJ27", 1, "sell", 2), ("CLK27", -2, "sell", 2), ("CLM27", 1, "sell", 2)]),
    ]))
    assert rows["CLZ26"]["net_lots"] == 10
    assert rows["CLF27"]["net_lots"] == -4
    assert rows["CLG27"]["net_lots"] == 3 and rows["CLH27"]["net_lots"] == -3
    assert (rows["CLJ27"]["net_lots"], rows["CLK27"]["net_lots"], rows["CLM27"]["net_lots"]) == (-2, 4, -2)


def test_gross_long_short_and_contributing_structures():
    rows = by_symbol(build_exposure_map([
        structure("A", [("CLZ26", 1, "buy", 10)]),
        structure("B", [("CLZ26", 1, "sell", 4)]),
        structure("C", [("CLZ26", 1, "buy", 1), ("CLF27", -1, "buy", 1)]),
    ]))
    z = rows["CLZ26"]
    assert (z["net_lots"], z["gross_long"], z["gross_short"]) == (7, 11, 4)
    assert z["structures"] == ["A", "B", "C"] and rows["CLF27"]["structures"] == ["C"]


def test_spread_and_fly_symbol_legs_are_decomposed_into_outrights():
    rows = by_symbol(build_exposure_map([
        structure("spread leg", [("CLZ26-F27", 1, "buy", 5)]),
        structure("fly leg", [("CLZ26-F27-G27", 1, "sell", 2)]),
    ]))
    assert rows["CLZ26"]["net_lots"] == 5 - 2
    assert rows["CLF27"]["net_lots"] == -5 + 4
    assert rows["CLG27"]["net_lots"] == -2
    assert rows["CLF27"]["gross_long"] == 4 and rows["CLF27"]["gross_short"] == 5


def test_only_open_structures_count_and_empty_structures_are_ignored():
    empty = structure("no legs", [("CLZ26", 1, "buy", 1)]).model_copy(update={"legs": []})
    rows = build_exposure_map([
        structure("open", [("CLZ26", 1, "buy", 5)]),
        structure("shell", [("CLZ26", 1, "buy", 0)], StructureStatus.SHELL),
        structure("closed", [("CLZ26", 1, "buy", 0)], StructureStatus.CLOSED),
        empty,
    ])
    assert [(r["symbol"], r["net_lots"], r["structures"]) for r in rows] == [("CLZ26", 5, ["open"])]
    assert build_exposure_map([]) == []


def test_unparseable_symbol_is_skipped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        rows = build_exposure_map([structure("odd", [("WEIRD", 1, "buy", 1), ("CLZ26", 1, "buy", 2)])])
    assert [r["symbol"] for r in rows] == ["CLZ26"]
    assert "Skipping leg WEIRD" in caplog.text


def test_flat_month_is_kept_with_zero_net():
    rows = build_exposure_map([structure("a", [("CLZ26", 1, "buy", 5)]), structure("b", [("CLZ26", 1, "sell", 5)])])
    assert rows[0]["net_lots"] == 0 and rows[0]["gross_long"] == 5 and rows[0]["gross_short"] == 5


def test_rows_sorted_chronologically_with_month_labels():
    rows = build_exposure_map([
        structure("x", [("CLG27", 1, "buy", 1), ("CLZ26", 1, "buy", 1), ("CLF27", 1, "buy", 1), ("CLH26", 1, "buy", 1)])
    ])
    assert [r["symbol"] for r in rows] == ["CLH26", "CLZ26", "CLF27", "CLG27"]
    assert [r["label"] for r in rows] == ["Mar-26", "Dec-26", "Jan-27", "Feb-27"]


def test_labels_include_the_product_when_several_products_are_held():
    rows = build_exposure_map([structure("x", [("CLZ26", 1, "buy", 1), ("BRNZ26", 1, "buy", 1)])])
    assert sorted(r["label"] for r in rows) == ["BRN Dec-26", "CL Dec-26"]


def test_concentration_warnings_use_absolute_net_and_strict_threshold():
    rows = build_exposure_map([
        structure("big long", [("CLZ26", 1, "buy", 12)]),
        structure("big short", [("CLF27", 1, "sell", 15)]),
        structure("edge", [("CLG27", 1, "buy", 10)]),
    ])
    flagged = concentration_warnings(rows, 10)
    assert [(f["label"], f["side"], f["lots"]) for f in flagged] == [("Dec-26", "long", 12), ("Jan-27", "short", 15)]
    assert flagged[0]["structures"] == ["big long"]
    assert concentration_warnings(rows, 20) == []


# ---------- layout builders ----------


def test_bar_figure_colors_order_and_zero_line():
    rows = build_exposure_map([
        structure("a", [("CLF27", 1, "sell", 4), ("CLZ26", 1, "buy", 6)]),
        structure("flat", [("CLG27", 1, "buy", 2)]),
        structure("flat2", [("CLG27", 1, "sell", 2)]),
    ])
    figure = build_bar_figure(rows)
    bar = figure.data[0]
    assert list(bar.y) == ["Dec-26", "Jan-27"] and list(bar.x) == [6, -4]  # the flat month has no bar
    assert list(bar.marker.color) == [COLORS["ACCENT_GREEN"], COLORS["ACCENT_RED"]]
    assert bar.orientation == "h" and figure.layout.title.text == "Net Exposure by Contract Month"
    assert figure.layout.yaxis.autorange == "reversed" and len(figure.layout.shapes) == 1  # the zero line


def test_bar_figure_placeholders():
    assert build_bar_figure([]).layout.annotations[0].text == NO_EXPOSURE_TEXT
    flat = build_exposure_map([structure("a", [("CLZ26", 1, "buy", 2)]), structure("b", [("CLZ26", 1, "sell", 2)])])
    assert build_bar_figure(flat).layout.annotations[0].text == FLAT_TEXT


def test_table_rows_include_flat_months_and_join_structure_names():
    rows = build_exposure_map([structure("A", [("CLZ26", 1, "buy", 5)]), structure("B", [("CLZ26", 1, "sell", 5)])])
    assert build_table_rows(rows) == [
        {"symbol": "CLZ26", "net_lots": 0, "gross_long": 5, "gross_short": 5, "structures": "A, B"}
    ]


def test_alert_text_and_all_clear():
    (alert,) = build_alerts([{"label": "Dec-26", "side": "long", "lots": 12.0, "structures": ["A", "B"]}])
    assert alert.color == "warning"
    assert alert.children == "Dec-26: net long 12 lots (structures: A, B)"
    (clear,) = build_alerts([])
    assert clear.color == "success" and clear.children == NO_WARNINGS_TEXT


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


def test_layout_components_and_table_styling():
    layout = exposure_layout()
    for component_id in ("exposure-bar-chart", "exposure-table", "exposure-threshold", "exposure-alerts", "exposure-data"):
        assert find(layout, component_id) is not None, component_id
    assert find(layout, "exposure-threshold").value == 10
    table = find(layout, "exposure-table")
    assert [c["name"] for c in table.columns] == ["Contract", "Net Lots", "Gross Long", "Gross Short", "Contributing Structures"]
    colors = {c["if"]["filter_query"]: c["color"] for c in table.style_data_conditional}
    assert colors == {"{net_lots} > 0": COLORS["ACCENT_GREEN"], "{net_lots} < 0": COLORS["ACCENT_RED"],
                      "{net_lots} = 0": COLORS["TEXT_PRIMARY"]}
    assert table.style_cell["backgroundColor"] == COLORS["CARD_BG"]


# ---------- callbacks ----------


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture(autouse=True)
def env(repo, monkeypatch):
    monkeypatch.setattr(ec, "container", Container(repository=repo))


def save(repo, s):
    for leg in s.legs:
        repo.save_contract(leg.contract)
    repo.save_structure(s)


def test_load_exposure_reads_open_structures_from_the_repository(repo):
    save(repo, structure("Open", [("CLZ26", 1, "sell", 12)]))
    save(repo, structure("Closed", [("CLF27", 1, "buy", 0)], StructureStatus.CLOSED))
    figure, table, rows = ec.load_exposure("/exposure")
    assert list(figure.data[0].x) == [-12] and table[0]["symbol"] == "CLZ26" and table[0]["net_lots"] == -12
    assert [r["symbol"] for r in rows] == ["CLZ26"]
    with pytest.raises(PreventUpdate):
        ec.load_exposure("/structures")


def test_load_exposure_with_no_open_structures(repo):
    figure, table, rows = ec.load_exposure("/exposure")
    assert figure.layout.annotations[0].text == NO_EXPOSURE_TEXT and table == [] and rows == []


def test_threshold_change_rerenders_only_the_alert_panel(repo):
    rows = build_exposure_map([structure("A", [("CLZ26", 1, "buy", 12)])])
    assert ec.update_concentration_alerts(10, rows)[0].children == "Dec-26: net long 12 lots (structures: A)"
    assert ec.update_concentration_alerts(15, rows)[0].children == NO_WARNINGS_TEXT
    assert ec.update_concentration_alerts(0, [])[0].children == NO_WARNINGS_TEXT
    for bad in (None, -1, float("nan"), True):
        assert "threshold of 0 or more" in ec.update_concentration_alerts(bad, rows).children
