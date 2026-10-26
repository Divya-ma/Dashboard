"""First in, first out: which entry lots each exit of a structure closed.

A structure is entered once (TRADE) and may be added to (ADD); each exit (a partial or the final one, or a dissolve) takes its lots
from the EARLIEST entry lots still open. The match decides the entry-side facts of the exit's Archive row: its entry price (the
lots-weighted structure price of the entries it took lots from), the entry date, the entry cost of exactly those lots, the stops and
targets typed on those entries (risk:reward) and which entry's market snapshot belongs to it.

    entries   3 lots at 0.50 (Mon), 2 lots at 0.60 (Tue)
    exit 1    4 lots  ->  3 lots of Mon's + 1 lot of Tue's   (entry price 0.525, entry date Mon)
    exit 2    1 lot   ->  the last lot of Tue's              (entry price 0.60,  entry date Tue)

EVERYTHING FOLLOWS THE SAME LOTS. The realized P&L of an exit is worked out from the entry lots it takes (core.trade_entry.
exit_structure), the entry transaction cost that goes with it (core.pnl.entry_cost_for_lots) is theirs, and what stays open keeps
the entry price and the entry cost of the lots NOT yet taken (`with_entry_price`, `open_queue`). With every entry at one price this is
the same number as an average-cost figure; with adds at other prices it differs per exit, never in total.

Structures with a per-leg exit (PARTIAL_EXIT) or a roll cannot be matched lot by lot; `match_exits` / `open_queue` return None
for them and the engine keeps its older average-cost figure.
"""

from dataclasses import dataclass, field

from core.models import Leg, Trade, TradeEventType

ENTRY_EVENTS = (TradeEventType.TRADE, TradeEventType.ADD)
EXIT_EVENTS = (TradeEventType.STRUCTURE_PARTIAL_EXIT, TradeEventType.FULL_EXIT, TradeEventType.DISSOLVE)
UNMATCHABLE_EVENTS = (TradeEventType.PARTIAL_EXIT, TradeEventType.ROLL)
_TOLERANCE = 1e-9


@dataclass
class EntrySlice:
    """`lots` of one entry trade taken by an exit."""

    trade: Trade
    lots: float

    @property
    def cost(self) -> float:
        """The share of the entry trade's transaction cost that belongs to these lots."""
        return self.trade.transaction_cost * self.lots / self.trade.lots if self.trade.lots else 0.0


@dataclass
class ExitMatch:
    exit: Trade
    index: int  # 1-based, in the order the structure's exits happened
    entries: list[EntrySlice] = field(default_factory=list)
    unmatched: float = 0.0  # exit lots with no entry lot behind them (only if the history is inconsistent)

    @property
    def lots(self) -> float:
        return self.exit.lots

    @property
    def matched_lots(self) -> float:
        return sum(s.lots for s in self.entries)

    @property
    def entry_price(self) -> float | None:
        """Lots-weighted structure price of the matched entry lots."""
        total = self.matched_lots
        return sum(s.trade.price * s.lots for s in self.entries) / total if total > 0 else None

    @property
    def entry_cost(self) -> float:
        return sum(s.cost for s in self.entries)

    @property
    def first_entry(self) -> Trade | None:
        return self.entries[0].trade if self.entries else None

    def entry_trades(self) -> list[Trade]:
        """The matched entries as trades of just the matched lots (for core.journal.risk_reward)."""
        return [s.trade.model_copy(update={"lots": s.lots}) for s in self.entries]


def _replay(trades: list[Trade]) -> tuple[list[ExitMatch], list[EntrySlice]] | None:
    """Walk the trades in time order: every exit's matches, and the entry lots still open at the end; None if unmatchable."""
    ordered = sorted(trades, key=lambda t: t.timestamp)
    if any(t.event_type in UNMATCHABLE_EVENTS for t in ordered):
        return None
    queue: list[EntrySlice] = []  # oldest first
    matches: list[ExitMatch] = []
    for trade in ordered:
        if trade.event_type in ENTRY_EVENTS:
            queue.append(EntrySlice(trade, trade.lots))
        elif trade.event_type in EXIT_EVENTS:
            taken, queue, short = take(queue, trade.lots)
            matches.append(ExitMatch(exit=trade, index=len(matches) + 1, entries=taken, unmatched=short))
    return matches, queue


def take(queue: list[EntrySlice], lots: float) -> tuple[list[EntrySlice], list[EntrySlice], float]:
    """Take `lots` from the front of an entry queue: (the slices taken, the queue left, lots that could not be found). The
    queue given is not changed."""
    taken: list[EntrySlice] = []
    left: list[EntrySlice] = []
    needed = lots
    for item in queue:
        if needed > _TOLERANCE:
            part = min(item.lots, needed)
            taken.append(EntrySlice(item.trade, part))
            needed -= part
            if item.lots - part > _TOLERANCE:
                left.append(EntrySlice(item.trade, item.lots - part))
        else:
            left.append(item)
    return taken, left, max(needed, 0.0)


def match_exits(trades: list[Trade]) -> list[ExitMatch] | None:
    """Every exit of the structure, in time order, with the entry lots it closed (FIFO); None if the trades cannot be matched."""
    replayed = _replay(trades)
    return None if replayed is None else replayed[0]


def open_queue(trades: list[Trade]) -> list[EntrySlice] | None:
    """The entry lots still open after every exit so far, oldest first; None when the history cannot be matched lot by lot
    (a per-leg exit or a roll, or an exit with no entry behind it)."""
    replayed = _replay(trades)
    if replayed is None or any(m.unmatched > _TOLERANCE for m in replayed[0]):
        return None
    return replayed[1]


def queue_lots(queue: list[EntrySlice]) -> float:
    return sum(item.lots for item in queue)


def queue_price(queue: list[EntrySlice]) -> float | None:
    """Lots-weighted structure price of the entry lots in the queue."""
    total = queue_lots(queue)
    return sum(item.trade.price * item.lots for item in queue) / total if total > _TOLERANCE else None


def queue_cost(queue: list[EntrySlice]) -> float:
    """The entry transaction cost that belongs to exactly these lots."""
    return sum(item.cost for item in queue)


# ----------------------------------------------------------------------
# The legs' entry price
# ----------------------------------------------------------------------


def _entry_of(leg: Leg) -> float | None:
    return leg.average_entry_price if leg.average_entry_price is not None else leg.entry_price


def composite_entry(legs: list[Leg]) -> float | None:
    """The structure's entry price as quoted, from its legs' entry prices (None if a leg has none): sign(first ratio) x sum(ratio x price)."""
    total = 0.0
    for leg in legs:
        price = _entry_of(leg)
        if price is None:
            return None
        total += leg.ratio * price
    return total if legs[0].ratio > 0 else -total


def with_entry_price(legs: list[Leg], target: float) -> list[Leg]:
    """The legs with the first leg's entry price moved so the structure's composite entry price is `target`.

    Only the weighted sum of the leg prices matters for P&L (an entry's leg prices are an allocation of the structure price,
    see core.trade_entry), so the first leg takes the whole difference and the others stay."""
    current = composite_entry(legs)
    if current is None:
        return legs
    shift = (target - current) / abs(legs[0].ratio)
    first = legs[0]
    moved = _entry_of(first) + shift
    return [first.model_copy(update={"average_entry_price": moved}), *legs[1:]]


def reconciles(legs: list[Leg], queue: list[EntrySlice], tolerance: float = 1e-6) -> bool:
    """True when the entry lots in the queue add up to the position the legs show: the same lots and the same entry price.
    Real per-leg fills typed at entry, or a hand-corrected entry price, make the legs differ from the entry trades; the engine
    then keeps the older average-cost figure rather than guess."""
    price, composite = queue_price(queue), composite_entry(legs)
    return price is not None and composite is not None and abs(price - composite) <= tolerance * max(1.0, abs(price))         and abs(queue_lots(queue) - legs[0].lots) <= 1e-6
