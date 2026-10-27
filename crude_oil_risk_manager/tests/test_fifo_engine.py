"""First in, first out in the engine: exit P&L, the entry price of the lots left open, costs, the replay, dissolve and the migration."""

from datetime import datetime, timezone

import pytest

from core.dissolve import dissolve_structures
from core.fifo import composite_entry, open_queue, queue_price, reconciles, take, with_entry_price
from core.fifo_migration import KEY_FIFO_MIGRATED, migrate_to_fifo, migrate_to_fifo_once
from core.models import Contract, Leg, Structure, StructureStatus, StructureType, TradeEventType
from core.pnl import calculate_portfolio_pnl, calculate_structure_unrealized_pnl, entry_cost_for_lots, split_transaction_costs
from core.structure_view import structure_entry_price
from core.trade_entry import enter_trade, exit_structure, recompute_structure_from_trades
from db.repository import Repository

ANCHORS = {"CLZ26": 80.0, "CLF27": 79.5}
MON, TUE, WED, THU = (datetime(2026, 8, d, 9, 30, tzinfo=timezone.utc) for d in (3, 4, 5, 6))
COST = 2.0


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


def contract(symbol):
    month = "FGHJKMNQUVXZ".index(symbol[2]) + 1
    return Contract(product="CL", contract_month=month, contract_year=2000 + int(symbol[3:]), symbol=symbol, multiplier=1000, tick_size=0.01, tick_value=10)


def shell(name="s", symbols=("CLZ26", "CLF27"), ratios=(1, -1)):
    return Structure(name=name, structure_type=StructureType.SPREAD, products=["CL"], status=StructureStatus.SHELL,
                     legs=[Leg(contract=contract(sym), ratio=r) for sym, r in zip(symbols, ratios)])


class Book:
    """A structure driven through the real entry / exit engine, the way the pages do it (first in, first out unless fifo=False)."""

    def __init__(self, fifo=True, **kw):
        self.structure, self.trades, self.fifo = shell(**kw), [], fifo

    def enter(self, lots, price, when, stop=None, target=None):
        result = enter_trade(self.structure, price, lots, "buy", None, ANCHORS, stop_loss_price=stop, target_price=target)
        self.structure = self.structure.model_copy(update={"legs": result.legs, "status": result.status})
        self.trades.append(result.trade.model_copy(update={"timestamp": when, "transaction_cost": COST * lots}))
        return result

    def exit(self, lots, price, when):
        result = exit_structure(self.structure, price, lots, None, open_entries=open_queue(self.trades) if self.fifo else None)
        update = {"legs": result.legs}
        if result.closed:
            update.update(status=StructureStatus.CLOSED, closed_at=when, close_trigger="manual")
        self.structure = self.structure.model_copy(update=update)
        self.trades.append(result.trade.model_copy(update={"timestamp": when, "realized_pnl": result.realized_pnl, "transaction_cost": COST * lots}))
        return result

    def save(self, repo):
        for leg in self.structure.legs:
            repo.save_contract(leg.contract)
        repo.save_structure(self.structure)
        for trade in self.trades:
            repo.save_trade(trade)
        return self.structure.structure_id


def two_entries(fifo=True):
    book = Book(fifo=fifo)
    book.enter(3, 0.50, MON)
    book.enter(2, 0.60, TUE)
    return book


# ---------- exit P&L and the entry price of the lots left open ----------


def test_an_exit_is_pnl_of_the_earliest_lots_and_the_rest_keeps_their_entry_price():
    book = two_entries()
    assert structure_entry_price(book.structure) == pytest.approx(0.54)  # 5 lots: the average of the whole position
    first = book.exit(4, 0.80, WED)
    assert first.realized_pnl == pytest.approx((3 * 0.30 + 1 * 0.20) * 1000)  # 3 lots from 0.50 and 1 from 0.60
    assert not first.closed and first.remaining_lots == 1
    assert structure_entry_price(book.structure) == pytest.approx(0.60)  # the one lot left was bought at 0.60
    assert "entry price of 0.6" in first.audit_note
    last = book.exit(1, 0.90, THU)
    assert last.closed and last.realized_pnl == pytest.approx(0.30 * 1000)


def test_the_old_average_cost_figure_is_still_available_and_the_total_is_the_same_either_way():
    average, fifo = two_entries(fifo=False), two_entries(fifo=True)
    a1, f1 = average.exit(4, 0.80, WED), fifo.exit(4, 0.80, WED)
    assert a1.realized_pnl == pytest.approx((0.80 - 0.54) * 4 * 1000) and f1.realized_pnl == pytest.approx(1100.0)
    assert structure_entry_price(average.structure) == pytest.approx(0.54)
    a2, f2 = average.exit(1, 0.90, THU), fifo.exit(1, 0.90, THU)
    assert a1.realized_pnl + a2.realized_pnl == pytest.approx(f1.realized_pnl + f2.realized_pnl)  # 1400 either way


def test_realized_plus_unrealized_does_not_depend_on_the_method():
    average, fifo = two_entries(fifo=False), two_entries(fifo=True)
    prices = {"CLZ26": 80.35, "CLF27": 79.50}  # the spread is 0.85
    totals = []
    for book in (average, fifo):
        realized = book.exit(4, 0.80, WED).realized_pnl
        unrealized, _, _ = calculate_structure_unrealized_pnl(book.structure.legs, prices)
        totals.append(realized + unrealized)
    assert totals[0] == pytest.approx(totals[1]) == pytest.approx(1350.0)
    # ...but FIFO shows more of it as realized: 1100 + 250 against 1040 + 310
    assert fifo.trades[-1].realized_pnl == pytest.approx(1100.0) and average.trades[-1].realized_pnl == pytest.approx(1040.0)


def test_a_sell_and_a_single_leg_position_work_the_same_way():
    book = Book(symbols=("CLZ26",), ratios=(1,))
    book.enter(3, 70.0, MON)
    book.enter(1, 72.0, TUE)
    result = book.exit(3, 71.0, WED)
    assert result.realized_pnl == pytest.approx(3 * 1.0 * 1000) and structure_entry_price(book.structure) == pytest.approx(72.0)


def test_with_one_entry_price_nothing_changes_and_unreconciled_legs_fall_back_to_the_average():
    one = Book()
    one.enter(5, 0.50, MON)
    assert one.exit(2, 0.80, WED).realized_pnl == pytest.approx(0.30 * 2 * 1000) and structure_entry_price(one.structure) == pytest.approx(0.50)
    # real per-leg fills typed at entry: the legs' composite is not the quoted price, so the average-cost figure is kept
    fills = Book()
    result = enter_trade(fills.structure, 0.50, 4, "buy", None, ANCHORS, leg_prices=[80.30, 79.70])  # composite 0.60, quoted 0.50
    fills.structure = fills.structure.model_copy(update={"legs": result.legs, "status": result.status})
    fills.trades.append(result.trade.model_copy(update={"timestamp": MON}))
    assert not reconciles(fills.structure.legs, open_queue(fills.trades))
    out = exit_structure(fills.structure, 0.80, 2, None, open_entries=open_queue(fills.trades))
    assert out.realized_pnl == pytest.approx((0.80 - 0.60) * 2 * 1000)


def test_helpers_take_lots_from_the_front_and_move_the_entry_price_to_a_target():
    book = two_entries()
    queue = open_queue(book.trades)
    taken, left, short = take(queue, 4)
    assert [(i.trade.price, i.lots) for i in taken] == [(0.50, 3), (0.60, 1)] and [(i.trade.price, i.lots) for i in left] == [(0.60, 1)] and short == 0
    assert queue_price(queue) == pytest.approx(0.54) and len(queue) == 2  # the queue given is not changed
    assert take(queue, 9)[2] == 4  # more than are open: 4 lots could not be found
    legs = with_entry_price(book.structure.legs, 0.75)
    assert composite_entry(legs) == pytest.approx(0.75) and legs[1] == book.structure.legs[1]


# ---------- costs ----------


def test_entry_cost_follows_the_same_lots():
    book = Book()
    book.enter(3, 0.50, MON)
    book.trades[-1] = book.trades[-1].model_copy(update={"transaction_cost": 6.0})  # 2 a lot
    book.enter(2, 0.60, TUE)
    book.trades[-1] = book.trades[-1].model_copy(update={"transaction_cost": 15.0})  # 7.5 a lot
    assert entry_cost_for_lots(book.trades, 4) == pytest.approx(6.0 + 7.5)  # 3 lots at 2 and 1 at 7.5
    book.exit(4, 0.80, WED)
    assert entry_cost_for_lots(book.trades, 1) == pytest.approx(7.5)  # what is left is the add's lot
    realized_cost, open_cost = split_transaction_costs(book.trades, open_lots=1)
    assert open_cost == pytest.approx(7.5) and realized_cost + open_cost == pytest.approx(6 + 15 + 8)  # exit cost 2 x 4 lots


def test_homes_realized_and_unrealized_follow_first_in_first_out():
    book = two_entries()
    book.exit(4, 0.80, WED)
    summary = calculate_portfolio_pnl([book.structure], {book.structure.structure_id: book.trades}, {"CLZ26": 80.35, "CLF27": 79.5}, [])
    entry_cost, exit_cost = COST * 4, COST * 4  # the 4 exited lots: 3 + 1 of entry cost, and the exit
    assert summary["total_realized_all_time"] == pytest.approx(1100.0 - entry_cost - exit_cost)
    assert summary["total_unrealized"] == pytest.approx(250.0)  # the last lot was bought at 0.60, now 0.85
    assert summary["total_open_costs"] == pytest.approx(COST * 1)
    assert summary["total_pnl"] == pytest.approx(1350.0 - COST * 5 - COST * 4)


# ---------- replay (delete a trade, the Archive editor) ----------


def test_the_replay_gives_the_stored_figures_and_the_old_average_replay_gives_the_old_ones():
    book = two_entries()
    first = book.exit(4, 0.80, WED)
    fifo = recompute_structure_from_trades(book.structure, book.trades)
    assert fifo.exit_trade_updates[book.trades[-1].trade_id][0] == pytest.approx(first.realized_pnl)
    assert structure_entry_price(fifo.structure) == pytest.approx(0.60) and fifo.structure.legs[0].lots == 1
    average = recompute_structure_from_trades(book.structure, book.trades, fifo=False)
    assert average.exit_trade_updates[book.trades[-1].trade_id][0] == pytest.approx(1040.0)
    assert structure_entry_price(average.structure) == pytest.approx(0.54)


def test_deleting_the_first_entry_replays_the_exit_against_what_is_left():
    book = two_entries()
    book.exit(2, 0.80, WED)
    survivors = [t for t in book.trades if t.event_type != TradeEventType.TRADE]  # the 3 lots at 0.50 are gone
    replay = recompute_structure_from_trades(book.structure, survivors)
    assert replay.exit_trade_updates[survivors[-1].trade_id][0] == pytest.approx(0.20 * 2 * 1000)


# ---------- dissolve ----------


def test_a_partial_dissolve_leaves_the_lots_that_were_bought_last():
    book = two_entries()
    other = Book(name="other", symbols=("CLG27",), ratios=(1,))
    result = enter_trade(other.structure, 70.0, 2, "buy", None, {"CLG27": 70.0})
    other.structure = other.structure.model_copy(update={"legs": result.legs, "status": result.status})
    outcome = dissolve_structures(
        [book.structure, other.structure], lots={book.structure.structure_id: 4, other.structure.structure_id: 2},
        typed_gross=500.0, trades_by_structure={book.structure.structure_id: book.trades, other.structure.structure_id: [result.trade]},
    )
    item = next(i for i in outcome.items if i.structure.structure_id == book.structure.structure_id)
    assert item.status == StructureStatus.OPEN and composite_entry(item.legs) == pytest.approx(0.60) and item.legs[0].lots == 1
    assert item.entry_costs == pytest.approx(COST * 4)  # the entry cost of the 3 + 1 lots dissolved, first in, first out


# ---------- the one-time migration ----------


def test_the_migration_moves_engine_made_average_figures_to_fifo_and_leaves_typed_ones_alone(repo):
    old_open = two_entries(fifo=False)
    old_open.exit(4, 0.80, WED)
    open_id = old_open.save(repo)
    old_closed = two_entries(fifo=False)
    old_closed.exit(4, 0.80, WED)
    old_closed.exit(1, 0.90, THU)
    old_closed.structure = old_closed.structure.model_copy(update={"name": "closed"})
    closed_id = old_closed.save(repo)
    typed = two_entries(fifo=False)
    typed.exit(4, 0.80, WED)
    typed.trades[-1] = typed.trades[-1].model_copy(update={"realized_pnl": 999.0})  # a figure typed in the Archive editor
    typed.structure = typed.structure.model_copy(update={"name": "typed"})
    typed_id = typed.save(repo)

    report = migrate_to_fifo(repo)
    assert sorted(report["structures"]) == ["closed", "s"] and report["exits"] == 3 and report["positions"] == 1  # 1 + 2 exits, 1 open position
    open_exit = next(t for t in repo.get_trades_for_structure(open_id) if t.event_type == TradeEventType.STRUCTURE_PARTIAL_EXIT)
    assert open_exit.realized_pnl == pytest.approx(1100.0) and open_exit.transaction_cost == COST * 4  # the cost stays as stored
    assert structure_entry_price(repo.get_structure(open_id)) == pytest.approx(0.60)
    assert "re-allocated first in, first out" in repo.get_structure(open_id).notes
    closed_exits = sorted((t for t in repo.get_trades_for_structure(closed_id) if t.event_type in (TradeEventType.STRUCTURE_PARTIAL_EXIT, TradeEventType.FULL_EXIT)), key=lambda t: t.timestamp)
    assert [round(t.realized_pnl, 6) for t in closed_exits] == [1100.0, 300.0] and sum(t.realized_pnl for t in closed_exits) == pytest.approx(1400.0)
    typed_exit = next(t for t in repo.get_trades_for_structure(typed_id) if t.event_type == TradeEventType.STRUCTURE_PARTIAL_EXIT)
    assert typed_exit.realized_pnl == 999.0 and structure_entry_price(repo.get_structure(typed_id)) == pytest.approx(0.54)  # nothing of it was touched
    assert migrate_to_fifo(repo) == {"structures": [], "exits": 0, "positions": 0}  # nothing left to change


def test_an_open_structure_whose_legs_carry_a_corrected_price_moves_by_the_same_shift_and_keeps_its_total(repo):
    book = two_entries(fifo=False)
    book.exit(4, 0.80, WED)
    first_leg, *others = book.structure.legs
    corrected = first_leg.model_copy(update={"average_entry_price": first_leg.average_entry_price + 0.05})  # a hand-corrected entry price
    book.structure = book.structure.model_copy(update={"legs": [corrected, *others]})
    sid = book.save(repo)
    assert migrate_to_fifo(repo) == {"structures": ["s"], "exits": 1, "positions": 1}
    exit_trade = next(t for t in repo.get_trades_for_structure(sid) if t.event_type == TradeEventType.STRUCTURE_PARTIAL_EXIT)
    assert exit_trade.realized_pnl == pytest.approx(1100.0)  # +60 realized ...
    assert structure_entry_price(repo.get_structure(sid)) == pytest.approx(0.59 + 0.06)  # ... and the open lot's price up by the same 60 / 1000, the correction kept
    assert migrate_to_fifo(repo) == {"structures": [], "exits": 0, "positions": 0}


def test_the_migration_runs_once_and_skips_single_entry_and_per_leg_histories(repo):
    single = Book(name="single")
    single.enter(5, 0.50, MON)
    single.exit(2, 0.80, WED)
    single.save(repo)
    legacy = two_entries(fifo=False)
    legacy.structure = legacy.structure.model_copy(update={"name": "per leg"})
    legacy.exit(4, 0.80, WED)
    legacy.trades[-1] = legacy.trades[-1].model_copy(update={"event_type": TradeEventType.PARTIAL_EXIT})
    legacy.save(repo)
    first = migrate_to_fifo_once(repo)
    assert first == {"structures": [], "exits": 0, "positions": 0} and repo.get_setting(KEY_FIFO_MIGRATED) is True
    assert migrate_to_fifo_once(repo) is None
