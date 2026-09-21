"""Tests for core.archive, the Archive layout builders and its (read-only) callbacks."""

from datetime import date, datetime, timezone

import pytest
from dash.exceptions import PreventUpdate

from core.archive import build_archive_rows, filter_archive_rows, summarize
from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType
from db.repository import Repository
from ui.callbacks import archive_callbacks as ac
from ui.container import Container
from ui.layouts.archive_tab import (
    NO_ARCHIVE_TEXT,
    NO_MATCH_TEXT,
    archive_layout,
    build_detail,
    build_stats,
    build_table_rows,
    money,
)
from ui.layouts.shell import COLORS


def utc(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def closed(name, legs, entry=(2026, 8, 1), exit_=(2026, 8, 11), pnl=1000.0, stype=StructureType.SPREAD,
           entry_price=75.0, exit_price=77.0, lots=5.0, notes="", with_trades=True):
    """legs: [(symbol, ratio, direction)]; returns (structure, trades)."""
    built = [
        Leg(contract=Contract(product="CL", contract_month=12, contract_year=2026, symbol=s, multiplier=1000,
                              tick_size=0.01, tick_value=10), ratio=r, direction=d, lots=0.0,
            entry_price=entry_price if i == 0 else 0.0, average_entry_price=entry_price if i == 0 else 0.0)
        for i, (s, r, d) in enumerate(legs)
    ]
    structure = Structure(
        name=name, structure_type=stype, products=["CL"], legs=built, status=StructureStatus.CLOSED,
        close_trigger="manual", closed_at=utc(*exit_) if exit_ else None, notes=notes,
        created_at=utc(*entry) if entry else utc(2026, 7, 1),
    )
    trades = []
    if with_trades and built:
        if entry:
            trades.append(Trade(structure_id=structure.structure_id, leg_id=built[0].leg_id, event_type=TradeEventType.TRADE,
                                lots=lots, price=entry_price, direction="buy", timestamp=utc(*entry)))
        trades.append(Trade(structure_id=structure.structure_id, leg_id=built[0].leg_id, event_type=TradeEventType.FULL_EXIT,
                            lots=lots, price=exit_price, direction="sell", realized_pnl=pnl, timestamp=utc(*exit_) if exit_ else utc(2026, 9, 1)))
    return structure, trades


def build(*items):
    return build_archive_rows([s for s, _ in items], {s.structure_id: t for s, t in items})


# ---------- rows ----------


def test_row_fields_from_structure_and_trades():
    (row,) = build(closed("Dec-Jan", [("CLZ25", 1, "buy"), ("CLH26", -1, "buy")], pnl=1234.5, lots=3))
    assert row["name"] == "Dec-Jan" and row["type"] == "Spread"
    assert row["legs_summary"] == "CLZ25 B3, CLH26 S3"
    assert (row["entry_date"], row["exit_date"], row["days_held"]) == ("2026-08-01", "2026-08-11", 10)
    assert (row["entry_price"], row["exit_price"], row["lots"]) == (75.0, 77.0, 3.0)
    assert row["realized_pnl"] == 1234.5 and row["result"] == "Win"


def test_leg_sides_lots_and_fly_ratios():
    (row,) = build(closed("fly", [("CLF26", 1, "sell"), ("CLG26", -2, "sell"), ("CLH26", 1, "sell")], lots=2, stype=StructureType.FLY))
    assert row["legs_summary"] == "CLF26 S2, CLG26 B4, CLH26 S2"  # a sold fly: short wings, long body
    assert [leg["side"] for leg in row["legs"]] == ["sell", "buy", "sell"]


def test_result_classification_and_sorting_by_exit_date_descending():
    rows = build(
        closed("old win", [("CLZ26", 1, "buy")], exit_=(2026, 7, 1), entry=(2026, 6, 1), pnl=10),
        closed("new loss", [("CLZ26", 1, "buy")], exit_=(2026, 9, 1), pnl=-5),
        closed("flat", [("CLZ26", 1, "buy")], exit_=(2026, 8, 20), pnl=0),
    )
    assert [r["name"] for r in rows] == ["new loss", "flat", "old win"]
    assert [r["result"] for r in rows] == ["Loss", "Flat", "Win"]


def test_missing_dates_and_no_legs_do_not_crash():
    (row,) = build(closed("anomaly", [("CLZ26", 1, "buy")], entry=None, exit_=None))
    assert row["entry_date"] is None and row["days_held"] is None and row["exit_date"] is not None  # falls back to the exit trade
    (bare,) = build(closed("no trades", [("CLZ26", 1, "buy")], with_trades=False, exit_=None))
    assert bare["exit_date"] is None and bare["days_held"] is None and bare["realized_pnl"] == 0.0
    no_legs = closed("no legs", [("CLZ26", 1, "buy")], with_trades=False)[0].model_copy(update={"legs": []})  # anomaly in the DB
    (empty,) = build_archive_rows([no_legs], {})
    assert empty["legs_summary"] == "—" and empty["symbols"] == [] and empty["entry_price"] is None


# ---------- filters ----------


@pytest.fixture
def rows():
    return build(
        closed("Z spread", [("CLZ25", 1, "buy"), ("CLH26", -1, "buy")], exit_=(2026, 8, 10), pnl=500, stype=StructureType.SPREAD),
        closed("F outright", [("CLF26", 1, "buy")], exit_=(2026, 8, 20), pnl=-200, stype=StructureType.OUTRIGHT),
        closed("Flat fly", [("BRNZ25", 1, "buy")], exit_=(2026, 9, 5), pnl=0, stype=StructureType.FLY),
    )


def names(found):
    return {r["name"] for r in found}


def test_filter_date_range_is_inclusive_on_exit_date(rows):
    assert names(filter_archive_rows(rows, start=date(2026, 8, 10), end=date(2026, 8, 20))) == {"Z spread", "F outright"}
    assert names(filter_archive_rows(rows, start=date(2026, 8, 21))) == {"Flat fly"}
    assert names(filter_archive_rows(rows, end=date(2026, 8, 9))) == set()
    assert len(filter_archive_rows(rows)) == 3


def test_filter_type_symbol_and_pnl(rows):
    assert names(filter_archive_rows(rows, structure_type="spread")) == {"Z spread"}
    assert len(filter_archive_rows(rows, structure_type="all")) == 3
    assert names(filter_archive_rows(rows, symbol_contains="  clz ")) == {"Z spread"}  # any leg, case-insensitive
    assert names(filter_archive_rows(rows, symbol_contains="h26")) == {"Z spread"}
    assert names(filter_archive_rows(rows, pnl_filter="winners")) == {"Z spread", "Flat fly"}  # flat shows in both
    assert names(filter_archive_rows(rows, pnl_filter="losers")) == {"F outright", "Flat fly"}


def test_filters_apply_together_and_can_match_nothing(rows):
    assert names(filter_archive_rows(rows, start=date(2026, 8, 1), structure_type="spread", symbol_contains="clz", pnl_filter="winners")) == {"Z spread"}
    assert filter_archive_rows(rows, structure_type="fly", pnl_filter="losers", symbol_contains="CLF") == []
    undated = build(closed("undated", [("CLZ26", 1, "buy")], with_trades=False, exit_=None))
    assert filter_archive_rows(undated, start=date(2026, 1, 1)) == [] and len(filter_archive_rows(undated)) == 1


# ---------- summary ----------


def test_summary_stats_and_flat_excluded_from_win_rate(rows):
    stats = summarize(rows)
    assert (stats["total"], stats["winners"], stats["losers"]) == (3, 1, 1)
    assert stats["win_rate"] == pytest.approx(50.0)  # the flat trade is not in the denominator
    assert stats["total_pnl"] == 300 and stats["avg_pnl"] == pytest.approx(100) and (stats["best"], stats["worst"]) == (500, -200)


def test_summary_of_nothing_is_all_zeros():
    assert summarize([]) == {"total": 0, "winners": 0, "losers": 0, "win_rate": 0.0, "total_pnl": 0.0,
                             "avg_pnl": 0.0, "best": 0.0, "worst": 0.0}


# ---------- layout builders ----------


def test_money_formatting():
    assert money(1234.5) == "$1,234.50" and money(-1234.5) == "-$1,234.50" and money(0) == "$0.00"


def test_stat_card_colors():
    stats = build_stats(summarize([]))
    assert stats["win-rate"][1]["color"] == COLORS["TEXT_PRIMARY"] and stats["total-pnl"][0] == "$0.00"
    good = build_stats({"total": 4, "winners": 3, "losers": 1, "win_rate": 75.0, "total_pnl": 100.0, "avg_pnl": 25.0, "best": 90.0, "worst": -20.0})
    assert good["win-rate"][1]["color"] == COLORS["ACCENT_GREEN"] and good["total-pnl"][1]["color"] == COLORS["ACCENT_GREEN"]
    assert good["win-rate"][0] == "75.0%" and good["worst"][0] == "-$20.00"
    bad = build_stats({"total": 4, "winners": 1, "losers": 3, "win_rate": 25.0, "total_pnl": -100.0, "avg_pnl": -25.0, "best": 10.0, "worst": -90.0})
    assert bad["win-rate"][1]["color"] == COLORS["ACCENT_RED"] and bad["total-pnl"][1]["color"] == COLORS["ACCENT_RED"]
    at_line = build_stats({"total": 2, "winners": 1, "losers": 1, "win_rate": 50.0, "total_pnl": 0.0, "avg_pnl": 0.0, "best": 1.0, "worst": -1.0})
    assert at_line["win-rate"][1]["color"] == COLORS["ACCENT_GREEN"] and at_line["total-pnl"][1]["color"] == COLORS["TEXT_PRIMARY"]


def test_table_rows_use_dashes_for_missing_values():
    (row,) = build(closed("anomaly", [("CLZ26", 1, "buy")], entry=None, with_trades=False, exit_=None))
    (cells,) = build_table_rows([row])
    assert cells["id"] == row["structure_id"] and cells["entry_date"] == "—" and cells["exit_date"] == "—"
    assert cells["days_held"] == "—" and cells["exit_price"] == "—" and cells["lots"] == "—"
    assert cells["realized_pnl"] == 0.0 and cells["result"] == "Flat"


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


def test_detail_panel_contents_and_notes():
    (row,) = build(closed("Dec-Jan", [("CLZ25", 1, "buy"), ("CLH26", -1, "buy")], pnl=-250, notes="hedge book\n[2026-08-11] full exit"))
    text = str(build_detail(row))
    for expected in ("Dec-Jan — Spread", "CLZ25", "CLH26", "Buy", "Sell", "75.00", "77.00", "-$250.00", "Days Held", "hedge book"):
        assert expected in text, expected
    (plain,) = build(closed("plain", [("CLZ25", 1, "buy")], notes="   "))
    assert "Notes" not in str(build_detail(plain)) and build_detail(None) is None


def test_layout_controls_and_table_configuration():
    layout = archive_layout()
    for component_id in ("archive-dates", "archive-type", "archive-symbol", "archive-pnl", "archive-apply-btn", "archive-clear-btn",
                         "archive-table", "archive-message", "archive-rows", "archive-selected", "archive-detail", "archive-detail-card", "archive-detail-close",
                         *[f"archive-stat-{s}" for s in ("total", "winners", "losers", "win-rate", "total-pnl", "avg-pnl", "best", "worst")]):
        assert find(layout, component_id) is not None, component_id
    table = find(layout, "archive-table")
    assert table.page_size == 20 and table.sort_action == "native"
    assert table.sort_by == [{"column_id": "exit_date", "direction": "desc"}]
    assert [c["name"] for c in table.columns][:3] == ["Structure Name", "Type", "Legs"]
    assert [o["label"] for o in find(layout, "archive-type").options] == ["All", "Outright", "Spread", "Fly"]
    assert [o["label"] for o in find(layout, "archive-pnl").options] == ["All", "Winners only", "Losers only"]
    colors = {c["if"].get("filter_query"): c["color"] for c in table.style_data_conditional if "color" in c}
    assert colors["{realized_pnl} > 0"] == COLORS["ACCENT_GREEN"] and colors["{realized_pnl} < 0"] == COLORS["ACCENT_RED"]
    assert colors['{result} = "Win"'] == COLORS["ACCENT_GREEN"] and colors['{result} = "Loss"'] == COLORS["ACCENT_RED"]


# ---------- callbacks (read-only) ----------


class ReadOnlyRepository:
    """Delegates reads to a real Repository and fails the test on any write."""

    def __init__(self, repo):
        self._repo = repo

    def __getattr__(self, name):
        if name.startswith(("save_", "update_", "delete_", "set_", "acknowledge_")):
            raise AssertionError(f"the Archive tab must not call {name}")
        return getattr(self._repo, name)


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture(autouse=True)
def env(repo, monkeypatch):
    monkeypatch.setattr(ac, "container", Container(repository=ReadOnlyRepository(repo)))


def trigger(monkeypatch, triggered_id):
    class Ctx:
        pass

    ctx = Ctx()
    ctx.triggered_id = triggered_id
    monkeypatch.setattr(ac, "callback_context", ctx)


def seed(repo, *items):
    for structure, trades in items:
        for leg in structure.legs:
            repo.save_contract(leg.contract)
        repo.save_structure(structure)
        for trade in trades:
            repo.save_trade(trade)


def refresh(monkeypatch, source="archive-apply-btn", **filters):
    trigger(monkeypatch, source)
    args = dict(pathname="/archive", apply_clicks=1, clear_clicks=None, start=None, end=None,
                structure_type="all", symbol="", pnl_filter="all")
    args.update(filters)
    return ac.refresh_archive(**args)


N_STATS = 8


def test_tab_open_shows_everything_with_stats(monkeypatch, repo):
    seed(repo,
         closed("A", [("CLZ25", 1, "buy")], pnl=400, exit_=(2026, 8, 10)),
         closed("B", [("CLF26", 1, "buy")], pnl=-100, exit_=(2026, 8, 12)))
    out = refresh(monkeypatch, "url")
    table, message, stored, *rest = out
    assert [r["name"] for r in table] == ["B", "A"] and message == "" and len(stored) == 2
    texts, styles = rest[:N_STATS], rest[N_STATS:N_STATS + 2]
    assert texts[0] == "2" and texts[3] == "50.0%" and texts[4] == "$300.00" and styles[1]["color"] == COLORS["ACCENT_GREEN"]
    assert out[3 + N_STATS + 2:] == (ac.no_update,) * 5


def test_apply_filters_and_empty_states(monkeypatch, repo):
    assert refresh(monkeypatch)[1] == NO_ARCHIVE_TEXT  # nothing closed yet
    empty_stats = refresh(monkeypatch)[3:3 + N_STATS]
    assert empty_stats[0] == "0" and empty_stats[3] == "0.0%" and empty_stats[4] == "$0.00"
    seed(repo, closed("A", [("CLZ25", 1, "buy")], pnl=400), closed("B", [("CLF26", 1, "buy")], pnl=-100))
    out = refresh(monkeypatch, symbol="clf", pnl_filter="losers")
    assert [r["name"] for r in out[0]] == ["B"] and out[1] == ""
    out = refresh(monkeypatch, symbol="brn")
    assert out[0] == [] and out[1] == NO_MATCH_TEXT and out[3] == "0"
    out = refresh(monkeypatch, start="2026-08-12T00:00:00", end="2026-08-31")
    assert [r["name"] for r in out[0]] == []  # both exited 2026-08-11
    assert len(refresh(monkeypatch, start="2026-08-11", end="2026-08-11")[0]) == 2


def test_clear_filters_resets_controls_and_reloads_everything(monkeypatch, repo):
    seed(repo, closed("A", [("CLZ25", 1, "buy")], pnl=400), closed("B", [("CLF26", 1, "buy")], pnl=-100))
    out = refresh(monkeypatch, "archive-clear-btn", clear_clicks=1, symbol="clf", pnl_filter="losers", structure_type="fly")
    assert len(out[0]) == 2 and out[3 + N_STATS + 2:] == (None, None, "all", "", "all")


def test_other_paths_do_not_refresh_and_archive_only_lists_closed(monkeypatch, repo):
    trigger(monkeypatch, "url")
    with pytest.raises(PreventUpdate):
        ac.refresh_archive("/structures", None, None, None, None, "all", "", "all")
    open_structure = closed("still open", [("CLZ25", 1, "buy")])[0].model_copy(update={"status": StructureStatus.OPEN})
    seed(repo, (open_structure, []), closed("done", [("CLZ25", 1, "buy")]))
    assert [r["name"] for r in refresh(monkeypatch, "url")[0]] == ["done"]


def test_row_click_selects_and_close_detail_clears(monkeypatch):
    trigger(monkeypatch, "archive-table")
    assert ac.select_row({"row": 0, "column_id": "name", "row_id": "abc"}, None) == ("abc", ac.no_update)
    for bad in (None, {}, {"row": 1, "column_id": "name"}):
        with pytest.raises(PreventUpdate):
            ac.select_row(bad, None)
    trigger(monkeypatch, "archive-detail-close")
    assert ac.select_row(None, 1) == (None, None)
    with pytest.raises(PreventUpdate):
        ac.select_row(None, None)


def test_detail_panel_shows_for_selected_row_and_hides_otherwise():
    (row,) = build(closed("Detail me", [("CLZ25", 1, "buy")]))
    children, style = ac.render_detail(row["structure_id"], [row])
    assert "Detail me" in str(children) and style["display"] == "block"
    assert ac.render_detail("gone", [row])[1]["display"] == "none"  # filtered out of the current set
    assert ac.render_detail(None, [row]) == (None, ac.render_detail(None, [])[1])
