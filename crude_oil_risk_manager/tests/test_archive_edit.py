"""Archive edits: every column of a closed trade can be corrected (structure, trades, legs, market data), and the Archive rows,
their results and the audit notes follow."""

from datetime import datetime, timezone

import pytest

from core.archive import build_archive_rows
from core.archive_edit import (
    ArchiveEditError, apply_excursion_values, apply_snapshot_values, edit_legs, edit_structure, edit_trades, parse_snapshot_cell,
    parse_timestamp, trade_rows,
)
from core.journal_service import JournalService
from core.models import Contract, Leg, Structure, StructureStatus, StructureType, TradeEventType
from core.trade_entry import enter_trade, exit_structure
from db.repository import Repository
from ui.callbacks import archive_callbacks as ac
from ui.container import Container

ENTRY_AT = datetime(2026, 8, 3, 9, 30, 41, tzinfo=timezone.utc)  # with seconds: the table shows minutes only
EXIT_AT = datetime(2026, 8, 13, 15, 5, 9, tzinfo=timezone.utc)
ANCHORS = {"CLZ26": 80.0, "CLF27": 79.5, "CLG27": 79.0}


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


def contract(symbol):
    month = "FGHJKMNQUVXZ".index(symbol[2]) + 1
    return Contract(product="CL", contract_month=month, contract_year=2000 + int(symbol[3:]), symbol=symbol,
                    multiplier=1000, tick_size=0.01, tick_value=10)


def make_closed(repo, name="Dec-Jan", symbols=("CLZ26", "CLF27"), ratios=(1, -1), entry=0.50, exit_=0.80, lots=5.0,
                kind=StructureType.SPREAD, direction="buy", stop=0.30, target=1.00):
    """A closed trade built with the real entry / exit engine and saved: returns its id."""
    legs = [Leg(contract=contract(s), ratio=r) for s, r in zip(symbols, ratios)]
    for leg in legs:
        repo.save_contract(leg.contract)
    shell = Structure(name=name, structure_type=kind, products=["CL"], legs=legs, status=StructureStatus.SHELL, notes="note")
    opened = enter_trade(shell, entry, lots, direction, None, ANCHORS, stop_loss_price=stop, target_price=target)
    entered = shell.model_copy(update={"legs": opened.legs, "status": opened.status})
    out = exit_structure(entered, exit_, lots, None)
    closed = entered.model_copy(update={"legs": out.legs, "status": StructureStatus.CLOSED, "closed_at": EXIT_AT, "close_trigger": "manual"})
    repo.save_structure(closed)
    repo.save_trade(opened.trade.model_copy(update={"timestamp": ENTRY_AT, "transaction_cost": 10.0}))
    repo.save_trade(out.trade.model_copy(update={"timestamp": EXIT_AT, "realized_pnl": out.realized_pnl, "transaction_cost": 10.0}))
    return closed.structure_id


def rows_of(repo, structure_id):
    structure = repo.get_structure(structure_id)
    return trade_rows(repo.get_trades_for_structure(structure_id)), structure


def archive_row(repo, structure_id):
    structure = repo.get_structure(structure_id)
    return build_archive_rows([structure], {structure_id: repo.get_trades_for_structure(structure_id)}, repo.get_journals())[0]


def exit_of(repo, structure_id):
    return next(t for t in repo.get_trades_for_structure(structure_id) if t.event_type == TradeEventType.FULL_EXIT)


def edit(rows, index, **changes):
    rows = [dict(r) for r in rows]
    rows[index].update(changes)
    return rows


# ---------- structure ----------


def test_name_type_and_notes_are_changed_with_an_audit_line(repo):
    sid = make_closed(repo)
    message = edit_structure(repo, sid, "Jan d-fly", "D-Fly", "my notes")
    saved = repo.get_structure(sid)
    assert saved.name == "Jan d-fly" and saved.structure_type is StructureType.DFLY and saved.notes.startswith("my notes\n[")
    assert "archive edit: name 'Dec-Jan' -> 'Jan d-fly'; type Spread -> Dfly; notes edited" in saved.notes
    assert "type Spread -> Dfly" in message
    assert archive_row(repo, sid)["type"] == "Dfly"
    assert edit_structure(repo, sid, "Jan d-fly", "dfly", saved.notes) == "No changes to save."


def test_structure_edit_rejects_bad_input_and_keeps_the_type_when_left_blank(repo):
    sid = make_closed(repo)
    with pytest.raises(ArchiveEditError) as exc:
        edit_structure(repo, sid, " ", "condor", None)
    assert "name is required" in " ".join(exc.value.errors) and "Unknown structure type" in " ".join(exc.value.errors)
    with pytest.raises(ArchiveEditError, match="limited to 100"):
        edit_structure(repo, sid, "x" * 101, None, None)
    edit_structure(repo, sid, "Renamed", "", None)
    assert repo.get_structure(sid).structure_type is StructureType.SPREAD


# ---------- legs ----------


def test_leg_symbol_and_ratio_edits_create_the_contract_and_log_it(repo):
    sid = make_closed(repo)
    message = edit_legs(repo, sid, ["CLZ26", "CLG27"], [1, -1])
    saved = repo.get_structure(sid)
    assert [leg.contract.symbol for leg in saved.legs] == ["CLZ26", "CLG27"] and repo.get_contract("CLG27") is not None
    assert "leg 2 symbol CLF27 -> CLG27" in message and "leg 2 symbol CLF27 -> CLG27" in saved.notes
    assert archive_row(repo, sid)["legs_summary"] == "CLZ26 B5, CLG27 S5"
    assert edit_legs(repo, sid, ["CLZ26", "CLG27"], [1, -1]) == "No changes to save."
    with pytest.raises(ArchiveEditError):
        edit_legs(repo, sid, ["CLZ26", "nonsense"], [1, -1])


# ---------- trades ----------


def test_the_table_rows_save_back_unchanged(repo):
    sid = make_closed(repo)
    rows, _ = rows_of(repo, sid)
    assert [r["event"] for r in rows] == ["trade", "full exit"] and rows[0]["timestamp"] == "2026-08-03 09:30"
    before = repo.get_trades_for_structure(sid)
    assert edit_trades(repo, sid, rows) == "No changes to save."
    assert repo.get_trades_for_structure(sid) == before  # the stored seconds survive a save that changed nothing


def test_changing_the_exit_price_recalculates_the_pnl_and_the_archive_row(repo):
    sid = make_closed(repo)
    rows, _ = rows_of(repo, sid)
    message = edit_trades(repo, sid, edit(rows, 1, price=1.00))
    assert exit_of(repo, sid).price == 1.00 and exit_of(repo, sid).realized_pnl == pytest.approx((1.00 - 0.50) * 5 * 1000)
    row = archive_row(repo, sid)
    assert row["exit_price"] == 1.00 and row["realized_pnl"] == pytest.approx(2500.0 - 20.0) and row["result"] == "Win"
    assert repo.get_structure(sid).status is StructureStatus.CLOSED
    assert "P&L recalculated from the prices" in message and "archive edit:" in repo.get_structure(sid).notes


def test_changing_the_entry_price_moves_the_entry_price_column_and_the_exit_pnl(repo):
    sid = make_closed(repo)
    rows, _ = rows_of(repo, sid)
    with pytest.raises(ArchiveEditError, match="stop loss must be below the entry price"):
        edit_trades(repo, sid, edit(rows, 0, price=0.20))  # the stop (0.30) would now be above a buy at 0.20
    edit_trades(repo, sid, edit(rows, 0, price=0.20, stop_loss_price=0.10))
    row = archive_row(repo, sid)
    assert row["entry_price"] == pytest.approx(0.20)
    assert exit_of(repo, sid).realized_pnl == pytest.approx((0.80 - 0.20) * 5 * 1000)


def test_a_pnl_typed_in_the_same_save_wins_and_so_does_switching_the_recalculation_off(repo):
    sid = make_closed(repo)
    rows, _ = rows_of(repo, sid)
    edit_trades(repo, sid, edit(rows, 1, price=1.00, realized_pnl=123.0))
    assert exit_of(repo, sid).price == 1.00 and exit_of(repo, sid).realized_pnl == 123.0
    other = make_closed(repo, name="Other")
    rows, _ = rows_of(repo, other)
    original = exit_of(repo, other).realized_pnl
    edit_trades(repo, other, edit(rows, 1, price=1.00), recalculate=False)
    assert exit_of(repo, other).price == 1.00 and exit_of(repo, other).realized_pnl == original  # stored exactly as typed


def test_a_price_change_leaves_a_hand_corrected_cost_alone_and_new_lots_scale_the_costs(repo):
    sid = make_closed(repo)
    rows, _ = rows_of(repo, sid)
    edit_trades(repo, sid, edit(rows, 1, price=1.00, transaction_cost=33.0))
    assert exit_of(repo, sid).transaction_cost == 33.0  # typed
    other = make_closed(repo, name="Other")
    rows, _ = rows_of(repo, other)
    edit_trades(repo, other, edit(rows, 1, price=0.90))
    assert exit_of(repo, other).transaction_cost == 10.0  # a price does not touch the cost
    rows, _ = rows_of(repo, other)
    message = edit_trades(repo, other, edit(edit(rows, 0, lots=4), 1, lots=4))
    assert [t.transaction_cost for t in repo.get_trades_for_structure(other)] == [8.0, 8.0] and "cost scaled" in message
    assert archive_row(repo, other)["realized_pnl"] == pytest.approx((0.90 - 0.50) * 4 * 1000 - 16.0)


def test_a_sell_direction_flips_the_pnl(repo):
    sid = make_closed(repo)
    rows, _ = rows_of(repo, sid)
    edit_trades(repo, sid, edit(edit(rows, 0, direction="sell", stop_loss_price=0.70, target_price=0.30), 1, direction="buy"))
    assert exit_of(repo, sid).realized_pnl == pytest.approx(-(0.80 - 0.50) * 5 * 1000)


def test_dates_costs_stops_and_targets_are_stored_and_drive_their_columns(repo):
    sid = make_closed(repo)
    rows, _ = rows_of(repo, sid)
    edit_trades(repo, sid, edit(edit(rows, 0, timestamp="2026-08-01", stop_loss_price=0.40, target_price=0.90, transaction_cost=25),
                                 1, timestamp="2026-08-20 10:00", transaction_cost=15))
    row = archive_row(repo, sid)
    assert (row["entry_date"], row["exit_date"], row["days_held"]) == ("2026-08-01", "2026-08-20", 19)
    assert row["transaction_costs"] == 40.0 and row["realized_pnl"] == pytest.approx(1500.0 - 40.0)
    assert row["rr_detail"]["stop"] == 0.40 and row["rr_detail"]["target"] == 0.90 and row["rr_text"] == "1 : 4.00"
    assert repo.get_structure(sid).closed_at == datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
    assert parse_timestamp("2026-08-01T10:15:30Z") == datetime(2026, 8, 1, 10, 15, 30, tzinfo=timezone.utc)


def test_edits_that_would_reopen_the_trade_or_are_malformed_are_refused_and_change_nothing(repo):
    sid = make_closed(repo)
    rows, _ = rows_of(repo, sid)
    before = repo.get_trades_for_structure(sid)
    with pytest.raises(ArchiveEditError, match="no longer close the whole position"):
        edit_trades(repo, sid, edit(rows, 1, lots=3))
    with pytest.raises(ArchiveEditError) as exc:
        edit_trades(repo, sid, edit(edit(rows, 0, price="abc", lots=0), 1, timestamp="yesterday", transaction_cost=-1, direction="hold"))
    text = " ".join(exc.value.errors)
    for expected in ("price must be a number", "lots must be greater than 0", "use YYYY-MM-DD", "cannot be below 0", "buy or sell"):
        assert expected in text
    with pytest.raises(ArchiveEditError, match="not one of this trade"):
        edit_trades(repo, sid, [{**rows[0], "id": "nope"}])
    assert repo.get_trades_for_structure(sid) == before and repo.get_structure(sid).status is StructureStatus.CLOSED


# ---------- market data and excursion ----------


def test_snapshot_cells_are_parsed_and_validated():
    assert parse_snapshot_cell("px_cl", "71,250.5") == 71250.5 and parse_snapshot_cell("px_cl", "") is None
    assert parse_snapshot_cell("move_day", "bullish") == "Bullish"
    assert parse_snapshot_cell("curve_cl", "contango-backwardation") == "Contango - Backwardation"
    with pytest.raises(ValueError, match="must be one of"):
        parse_snapshot_cell("move_min", "sideways")
    with pytest.raises(ValueError, match="must be a number"):
        parse_snapshot_cell("atr", "big")


def test_snapshot_values_build_the_nested_document_and_empty_input_makes_nothing():
    snapshot = apply_snapshot_values(None, {"px_cl": 71.5, "px_co": "75", "move_day": "Bearish", "curve_brn": "Flat", "atr": 0.12})
    assert snapshot["prices"] == {"CL": 71.5, "CO": 75.0} and snapshot["move_day"] == "Bearish"
    assert snapshot["curve"] == {"BRN": "Flat"} and snapshot["atr"] == 0.12 and snapshot["edited"] is True
    changed = apply_snapshot_values(snapshot, {"px_cl": "", "atr": 0.2})
    assert changed["prices"]["CL"] is None and changed["prices"]["CO"] == 75.0 and changed["atr"] == 0.2
    assert apply_snapshot_values(None, {"px_cl": "", "move_day": None}) is None


def test_excursion_values_keep_intraday_over_daily_and_refuse_wrong_signs():
    result = apply_excursion_values(None, {"mfe_i": 500, "mae_i": -200, "mfe_d": 450, "mae_d": -250})
    assert (result["mfe"], result["mae"]) == (500, -200) and result["daily"]["mfe"] == 450
    only_daily = apply_excursion_values(None, {"mfe_d": 450, "mae_d": -250})
    assert (only_daily["mfe"], only_daily["mae"], only_daily["intraday"]) == (450, -250, None)
    assert apply_excursion_values(result, {}) is None
    with pytest.raises(ValueError, match="cannot be negative"):
        apply_excursion_values(None, {"mfe_i": -1})
    with pytest.raises(ValueError, match="cannot be positive"):
        apply_excursion_values(None, {"mae_i": 5})


def test_the_journal_service_saves_market_cells_and_the_archive_columns_show_them(repo):
    sid = make_closed(repo)
    service = JournalService(repo, None, background=False)
    service.set_market(sid, {"px_cl": 71.5, "move_day": "Bullish"}, {"px_cl": 72.25, "curve_cl": "Backwardation"},
                       {"mfe_i": 900, "mae_i": -300, "mfe_d": None, "mae_d": None})
    row = archive_row(repo, sid)
    assert (row["e_px_cl"], row["x_px_cl"], row["e_move_day"], row["x_curve_cl"]) == (71.5, 72.25, "Bullish", "Backwardation")
    assert (row["mfe_i"], row["mae_i"]) == (900, -300)
    with pytest.raises(ValueError):
        service.set_market(sid, {"move_day": "Sideways"}, None, None)
    assert archive_row(repo, sid)["e_move_day"] == "Bullish"  # a bad cell wrote nothing


# ---------- the panel's callbacks ----------


@pytest.fixture
def panel(repo, monkeypatch):
    box = Container(repository=repo, journal=JournalService(repo, None, background=False))
    monkeypatch.setattr(ac, "container", box)
    return box


def test_the_panel_fills_with_the_stored_values(repo, panel):
    sid = make_closed(repo)
    name, kind, notes, trades, legs, market, *excursion = ac.fill_trade_editor(sid, 0)
    assert (name, kind, notes) == ("Dec-Jan", "spread", "note") and [t["event"] for t in trades] == ["trade", "full exit"]
    assert [(r["symbol"], r["ratio"]) for r in legs] == [("CLZ26", 1), ("CLF27", -1)]
    assert [r["id"] for r in market][:3] == ["px_co", "px_cl", "px_go"] and excursion == [None] * 4
    assert ac.fill_trade_editor(None, 0)[:3] == (None, None, "")


def test_each_save_reports_and_bumps_the_refresh_counter(repo, panel):
    sid = make_closed(repo)
    status, counter = ac.save_structure_edit(1, sid, "New name", "dfly", "n", 4)
    assert counter == 5 and "Saved" in status.children and repo.get_structure(sid).structure_type is StructureType.DFLY
    status, counter = ac.save_structure_edit(1, sid, "", "dfly", "n", 5)
    assert counter is ac.no_update and "Not saved" in status.children and "name is required" in status.children
    rows, _ = rows_of(repo, sid)
    status, counter = ac.save_trades_edit(1, sid, edit(rows, 1, price=0.9), True, 5)
    assert counter == 6 and exit_of(repo, sid).price == 0.9
    status, counter = ac.save_legs_edit(1, sid, [{"symbol": "CLZ26", "ratio": "1"}, {"symbol": "CLG27", "ratio": 1.0}], 6)
    assert counter == 7 and [leg.contract.symbol for leg in repo.get_structure(sid).legs] == ["CLZ26", "CLG27"]
    market = [{"id": "px_cl", "field": "CL price", "entry": 70.0, "exit": 71.0}, {"id": "move_day", "field": "Day move", "entry": "oops", "exit": ""}]
    status, counter = ac.save_market_edit(1, sid, market, 100, -50, None, None, 7)
    assert counter is ac.no_update and "must be one of" in status.children  # the bad cell stops the whole save
    market[1]["entry"] = "Bearish"
    status, counter = ac.save_market_edit(1, sid, market, 100, -50, None, None, 7)
    assert counter == 8 and archive_row(repo, sid)["e_move_day"] == "Bearish"
    for call in (lambda: ac.save_structure_edit(None, sid, "x", "dfly", "", 0), lambda: ac.save_trades_edit(None, sid, [], True, 0)):
        with pytest.raises(ac.PreventUpdate):
            call()
    assert "Select a trade first" in ac.save_trades_edit(1, None, [], True, 0)[0].children
    assert ac.toggle_editor(1, False) is True and ac.toggle_editor(2, True) is False


def test_every_editor_id_used_by_a_callback_is_in_the_archive_layout():
    import dash

    from ui.layouts.archive_tab import archive_layout

    def ids(component):
        found = set()
        stack = [component]
        while stack:
            item = stack.pop()
            if getattr(item, "id", None) is not None:
                found.add(item.id)
            children = getattr(item, "children", None)
            stack.extend(children if isinstance(children, (list, tuple)) else [children] if children is not None else [])
        return found

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = dash.html.Div([dash.dcc.Location(id="url"), archive_layout()])
    ac.register_archive_callbacks(app)
    present = ids(app.layout)
    wanted = {part.rsplit(".", 1)[0] for key in app.callback_map for part in key.strip(".").split("...") if part}
    editor_ids = {i for i in wanted if i.startswith("archive-edit")}
    assert editor_ids and editor_ids <= present, editor_ids - present
