"""Archive rows per exit: a partial exit shows at once, and each exit takes its lots from the earliest entries (first in, first out)."""

from datetime import datetime, timezone

import pytest

from core.archive import build_archive_rows, structure_id_of
from core.archive_edit import ArchiveEditError, edit_trades, trade_rows
from core.fifo import match_exits, open_queue
from core.journal_service import JournalService
from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType
from core.pnl import calculate_net_realized_pnl
from core.trade_entry import enter_trade, exit_structure
from db.repository import Repository
from ui.callbacks import archive_callbacks as ac
from ui.container import Container

ANCHORS = {"CLZ26": 80.0, "CLF27": 79.5}
MON, TUE, WED, THU = (datetime(2026, 8, d, 9, 30, tzinfo=timezone.utc) for d in (3, 4, 5, 6))
COST_PER_LOT = 2.0


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


def contract(symbol):
    month = "FGHJKMNQUVXZ".index(symbol[2]) + 1
    return Contract(product="CL", contract_month=month, contract_year=2000 + int(symbol[3:]), symbol=symbol, multiplier=1000, tick_size=0.01, tick_value=10)


def build_history(repo, entries, exits, name="Dec-Jan"):
    """Enter and exit through the real engine and save it: entries [(lots, price, time, stop, target)], exits [(lots, price, time)]."""
    legs = [Leg(contract=contract("CLZ26"), ratio=1), Leg(contract=contract("CLF27"), ratio=-1)]
    for leg in legs:
        repo.save_contract(leg.contract)
    current = Structure(name=name, structure_type=StructureType.SPREAD, products=["CL"], legs=legs, status=StructureStatus.SHELL)
    trades = []
    for lots, price, when, stop, target in entries:
        result = enter_trade(current, price, lots, "buy", None, ANCHORS, stop_loss_price=stop, target_price=target)
        current = current.model_copy(update={"legs": result.legs, "status": result.status})
        trades.append(result.trade.model_copy(update={"timestamp": when, "transaction_cost": COST_PER_LOT * lots}))
    for lots, price, when in exits:
        result = exit_structure(current, price, lots, None, open_entries=open_queue(trades))
        update = {"legs": result.legs}
        if result.closed:
            update.update(status=StructureStatus.CLOSED, closed_at=when, close_trigger="manual")
        current = current.model_copy(update=update)
        trades.append(result.trade.model_copy(update={"timestamp": when, "realized_pnl": result.realized_pnl, "transaction_cost": COST_PER_LOT * lots}))
    repo.save_structure(current)
    for trade in trades:
        repo.save_trade(trade)
    return current.structure_id


def rows_for(repo, sid):
    structure = repo.get_structure(sid)
    return build_archive_rows([structure], {sid: repo.get_trades_for_structure(sid)}, repo.get_journals())


def trade(event, lots, price, when, **kw):
    return Trade(structure_id="s", leg_id="l", event_type=event, lots=lots, price=price, direction="buy", timestamp=when, **kw)


# ---------- the matching ----------


def test_each_exit_takes_its_lots_from_the_earliest_entries():
    entry1, entry2 = trade(TradeEventType.TRADE, 3, 0.50, MON), trade(TradeEventType.ADD, 2, 0.60, TUE)
    exit1, exit2 = trade(TradeEventType.STRUCTURE_PARTIAL_EXIT, 4, 0.80, WED), trade(TradeEventType.FULL_EXIT, 1, 0.90, THU)
    first, second = match_exits([exit2, entry2, exit1, entry1])  # any order in, time order out
    assert [(s.trade.trade_id, s.lots) for s in first.entries] == [(entry1.trade_id, 3), (entry2.trade_id, 1)]
    assert first.entry_price == pytest.approx((3 * 0.50 + 1 * 0.60) / 4) and first.first_entry is entry1 and first.index == 1
    assert [(s.trade.trade_id, s.lots) for s in second.entries] == [(entry2.trade_id, 1)]
    assert second.entry_price == 0.60 and second.unmatched == 0 and second.index == 2
    assert [t.lots for t in first.entry_trades()] == [3, 1]  # copies with just the matched lots


def test_entry_costs_are_the_share_of_the_matched_lots_and_a_missing_entry_is_flagged():
    entry = trade(TradeEventType.TRADE, 4, 0.50, MON, transaction_cost=8.0)
    (match,) = match_exits([entry, trade(TradeEventType.FULL_EXIT, 1, 0.9, WED)])
    assert match.entry_cost == 2.0 and match.matched_lots == 1
    (orphan,) = match_exits([trade(TradeEventType.FULL_EXIT, 2, 0.9, WED)])
    assert orphan.entries == [] and orphan.unmatched == 2 and orphan.entry_price is None
    assert match_exits([entry]) == []


def test_per_leg_exits_and_rolls_cannot_be_matched_lot_by_lot():
    entry = trade(TradeEventType.TRADE, 2, 0.5, MON)
    assert match_exits([entry, trade(TradeEventType.PARTIAL_EXIT, 1, 0.9, WED)]) is None
    assert match_exits([entry, trade(TradeEventType.ROLL, 1, 0.9, WED)]) is None


# ---------- rows ----------


def test_a_partial_exit_is_in_the_archive_at_once_and_the_rest_adds_a_second_row(repo):
    sid = build_history(repo, [(5, 0.50, MON, 0.30, 1.00)], [(2, 0.80, WED)])
    (first,) = rows_for(repo, sid)
    assert repo.get_structure(sid).status is StructureStatus.OPEN
    assert (first["id"], first["structure_id"], first["lots"], first["open_lots"]) == (sid, sid, 2, 3.0)
    assert (first["entry_price"], first["exit_price"], first["entry_date"], first["exit_date"]) == (0.50, 0.80, "2026-08-03", "2026-08-05")
    assert first["realized_pnl"] == pytest.approx((0.80 - 0.50) * 2 * 1000 - 2 * COST_PER_LOT * 2)  # gross - exit cost - the 2 lots' entry cost
    assert first["legs_summary"] == "CLZ26 B2, CLF27 S2" and first["result"] == "Win" and first["rr_text"] == "1 : 2.50"

    structure = repo.get_structure(sid)
    result = exit_structure(structure, 0.70, 3, None, open_entries=open_queue(repo.get_trades_for_structure(sid)))
    repo.update_structure_legs(sid, result.legs, result.audit_note)
    repo.save_trade(result.trade.model_copy(update={"timestamp": THU, "realized_pnl": result.realized_pnl, "transaction_cost": COST_PER_LOT * 3}))
    repo.update_structure_status(sid, StructureStatus.CLOSED, close_trigger="manual")
    rows = sorted(rows_for(repo, sid), key=lambda r: r["exit_number"])
    assert [r["id"] for r in rows] == [sid, f"{sid}#2"] and [r["lots"] for r in rows] == [2, 3] and rows[1]["open_lots"] == 0
    assert all(r["exits_of_structure"] == 2 and structure_id_of(r["id"]) == sid for r in rows)
    assert sum(r["realized_pnl"] for r in rows) == pytest.approx(calculate_net_realized_pnl(repo.get_trades_for_structure(sid)))


def test_with_adds_at_other_prices_the_exits_use_the_earliest_entries(repo):
    sid = build_history(
        repo, [(3, 0.50, MON, 0.30, 1.00), (2, 0.60, TUE, 0.40, 1.10)], [(4, 0.80, WED), (1, 0.90, THU)],
    )
    first, second = sorted(rows_for(repo, sid), key=lambda r: r["exit_number"])
    assert first["entry_price"] == pytest.approx(0.525) and first["entry_date"] == "2026-08-03" and first["lots"] == 4
    assert second["entry_price"] == pytest.approx(0.60) and second["entry_date"] == "2026-08-04" and second["lots"] == 1
    # risk:reward from just the entries each exit used (stops and targets are simply averaged, as everywhere): 0.30 / 1.00 and 0.40 / 1.10
    assert first["rr_detail"]["stop"] == pytest.approx(0.35) and first["rr_detail"]["target"] == pytest.approx(1.05)
    assert second["rr_detail"]["stop"] == pytest.approx(0.40) and second["rr_detail"]["target"] == pytest.approx(1.10)
    assert first["transaction_costs"] == pytest.approx(COST_PER_LOT * 4 + COST_PER_LOT * 4)  # exit cost + 3 + 1 lots of entry cost
    # P&L is first in, first out too: 3 lots from 0.50 and 1 from 0.60 against 0.80, then the last lot from 0.60 against 0.90
    assert first["realized_pnl"] == pytest.approx((3 * 0.30 + 1 * 0.20) * 1000 - 16.0)
    assert second["realized_pnl"] == pytest.approx(0.30 * 1000 - 4.0)
    assert second["days_held"] == 2 and first["days_held"] == 2
    assert sum(r["realized_pnl"] for r in (first, second)) == pytest.approx(calculate_net_realized_pnl(repo.get_trades_for_structure(sid)))


def test_a_structure_that_has_not_exited_has_no_row_and_a_closed_single_exit_is_one_row_as_before(repo):
    untouched = build_history(repo, [(5, 0.50, MON, 0.30, 1.00)], [], name="Untouched")
    assert rows_for(repo, untouched) == []
    whole = build_history(repo, [(5, 0.50, MON, 0.30, 1.00)], [(5, 0.80, WED)], name="Whole")
    (row,) = rows_for(repo, whole)
    assert row["id"] == whole and row["exits_of_structure"] == 1 and row["open_lots"] == 0 and row["lots"] == 5


def test_a_closed_structure_with_a_per_leg_exit_keeps_its_single_older_row(repo):
    sid = build_history(repo, [(5, 0.50, MON, 0.30, 1.00)], [(5, 0.80, WED)])
    stored = repo.get_trades_for_structure(sid)
    leg_exit = next(t for t in stored if t.event_type == TradeEventType.FULL_EXIT)
    repo.delete_trade(leg_exit.trade_id)
    repo.save_trade(leg_exit.model_copy(update={"event_type": TradeEventType.PARTIAL_EXIT}))
    (row,) = rows_for(repo, sid)
    assert row["id"] == sid and row["lots"] == 5 and row["exits_of_structure"] == 1


# ---------- the market snapshot of each entry and exit ----------


def snapshot(price):
    return {"prices": {"CL": price}, "at": "2026-08-03T09:30:00+00:00"}


def test_each_row_reads_the_snapshot_of_the_entry_it_took_lots_from_and_of_its_own_exit(repo):
    sid = build_history(repo, [(3, 0.50, MON, 0.30, 1.00), (2, 0.60, TUE, 0.40, 1.10)], [(4, 0.80, WED), (1, 0.90, THU)])
    trades = sorted(repo.get_trades_for_structure(sid), key=lambda t: t.timestamp)
    entry1, entry2, exit1, exit2 = trades
    journal = {
        "entry": snapshot(70.0), "exit": snapshot(74.0),
        "entries": {entry2.trade_id: snapshot(71.0)}, "exits": {exit1.trade_id: snapshot(72.0)},
    }
    repo.save_journal(sid, journal)
    first, second = sorted(rows_for(repo, sid), key=lambda r: r["exit_number"])
    assert (first["e_px_cl"], first["x_px_cl"]) == (70.0, 72.0)  # the first entry (legacy `entry`), exit 1's own
    assert (second["e_px_cl"], second["x_px_cl"]) == (71.0, 74.0)  # the add's own, and the last exit falls back to `exit`


def test_the_journal_service_keeps_a_snapshot_per_trade_and_the_legacy_first_entry_and_last_exit(repo):
    class Shooter:
        def __init__(self):
            self.calls = 0

        def snapshot(self, structure, when):
            self.calls += 1
            return snapshot(60.0 + self.calls)

    sid = build_history(repo, [(3, 0.50, MON, 0.30, 1.00), (2, 0.60, TUE, 0.40, 1.10)], [(5, 0.80, WED)])
    entry1, entry2, exit1 = sorted(repo.get_trades_for_structure(sid), key=lambda t: t.timestamp)
    service = JournalService(repo, Shooter(), background=False)
    structure = repo.get_structure(sid)
    service.record(structure, "entry", MON, trade_id=entry1.trade_id)
    service.record(structure, "entry", TUE, trade_id=entry2.trade_id, set_legacy=False)  # an add never becomes THE entry
    service.record(structure, "exit", WED, overwrite=True, trade_id=exit1.trade_id)
    journal = service.journal(sid)
    assert journal["entry"]["prices"]["CL"] == 61.0 and journal["entries"][entry1.trade_id]["prices"]["CL"] == 61.0
    assert journal["entries"][entry2.trade_id]["prices"]["CL"] == 62.0 and journal["exits"][exit1.trade_id]["prices"]["CL"] == 63.0


def test_the_backfill_adopts_the_old_snapshots_and_fetches_only_what_is_missing_for_open_structures_too(repo):
    class Shooter:
        def __init__(self):
            self.calls = 0

        def snapshot(self, structure, when):
            self.calls += 1
            return snapshot(80.0 + self.calls)

    sid = build_history(repo, [(3, 0.50, MON, 0.30, 1.00), (2, 0.60, TUE, 0.40, 1.10)], [(2, 0.80, WED)])  # still 3 lots open
    repo.save_journal(sid, {"entry": snapshot(70.0)})  # only the first entry was ever recorded
    shooter = Shooter()
    service = JournalService(repo, shooter, background=False)
    assert service.backfill() == {"filled": 1, "left": 0}
    entry1, entry2, exit1 = sorted(repo.get_trades_for_structure(sid), key=lambda t: t.timestamp)
    journal = service.journal(sid)
    assert journal["entries"][entry1.trade_id]["prices"]["CL"] == 70.0  # adopted, no request
    assert shooter.calls == 2 and set(journal["entries"]) == {entry1.trade_id, entry2.trade_id} and set(journal["exits"]) == {exit1.trade_id}
    assert service.backfill() == {"filled": 0, "left": 0} and shooter.calls == 2


# ---------- the Archive page ----------


@pytest.fixture
def page(repo, monkeypatch):
    box = Container(repository=repo, journal=JournalService(repo, None, background=False))
    monkeypatch.setattr(ac, "container", box)
    return box


def test_the_page_loads_partly_exited_structures_and_selects_one_exit_row(repo, page):
    sid = build_history(repo, [(5, 0.50, MON, 0.30, 1.00)], [(2, 0.80, WED)])
    rows = ac._load_rows()
    assert [r["id"] for r in rows] == [sid] and rows[0]["open_lots"] == 3
    detail, style = ac.render_detail(sid, rows)
    assert "Exit 1 of 1" in str(detail) and "3 lots are still open" in str(detail) and style["display"] == "block"
    assert ac.render_detail("nope", rows)[1]["display"] == "none"
    assert ac.fill_trade_editor(f"{sid}#2", 0)[0] == "Dec-Jan"  # the editor works on the structure behind any of its rows


def test_deleting_keeps_a_structure_that_still_has_lots_open_and_counts_a_closed_one_once(repo, page):
    open_sid = build_history(repo, [(5, 0.50, MON, 0.30, 1.00)], [(2, 0.80, WED)], name="Still open")
    closed_sid = build_history(repo, [(3, 0.50, MON, 0.30, 1.00), (2, 0.60, TUE, 0.40, 1.10)], [(4, 0.80, WED), (1, 0.90, THU)], name="Done")
    table = [{"id": open_sid}, {"id": closed_sid}, {"id": f"{closed_sid}#2"}]
    assert ac._ticked_ids(table, [0, 1, 2]) == [open_sid, closed_sid]
    opened, body, ids, _wrap, confirm_disabled = ac.open_delete(1, table, [0, 1, 2])
    assert ids == [closed_sid] and confirm_disabled is False and "Still open" in str(body) and "lots still open" in str(body)
    assert "all of its exits" in str(body)
    _opened, body, ids, _wrap, confirm_disabled = ac.open_delete(1, table, [0])
    assert ids == [] and confirm_disabled is True and "still has lots open" in str(body)


# ---------- the editor on a trade with lots still open ----------


def test_the_editor_changes_an_exit_of_an_open_trade_but_will_not_close_it(repo):
    sid = build_history(repo, [(5, 0.50, MON, 0.30, 1.00)], [(2, 0.80, WED)])
    rows = trade_rows(repo.get_trades_for_structure(sid))
    exit_row = next(r for r in rows if r["event"] == "structure partial exit")
    edit_trades(repo, sid, [{**r, "price": 0.90} if r is exit_row else r for r in rows])
    stored = next(t for t in repo.get_trades_for_structure(sid) if t.event_type == TradeEventType.STRUCTURE_PARTIAL_EXIT)
    assert stored.price == 0.90 and stored.realized_pnl == pytest.approx(0.40 * 2 * 1000)
    assert repo.get_structure(sid).status is StructureStatus.OPEN and repo.get_structure(sid).legs[0].lots == 3
    with pytest.raises(ArchiveEditError, match="still open"):
        edit_trades(repo, sid, [{**r, "lots": 5} if r is exit_row else r for r in rows])
    assert repo.get_structure(sid).status is StructureStatus.OPEN
