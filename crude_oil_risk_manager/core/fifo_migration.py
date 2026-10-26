"""One-time move of the stored history from average-cost to first-in-first-out figures.

Before first in, first out, an exit's realized P&L was worked out against the AVERAGE entry price of the whole position and a partly
exited position kept that average for the lots left. With every entry at one price that is the same number as FIFO and nothing
changes. Where lots were added at another price, the profit is split differently between the exits (and between realized and
unrealized while lots stay open); the structure's total is the same either way.

For each structure with exits this replays its trades both ways (core.trade_entry.recompute_structure_from_trades with fifo True and
False) and changes only what the ENGINE itself had produced:

  * an exit's realized P&L, if the stored figure equals what the old average-cost replay gives;
  * the entry price of the lots still open. An open structure whose legs do NOT show what the old replay gives (real per-leg fills, a
    hand-corrected entry price) is left entirely alone: its exits and its open lots must move together or total P&L would shift.

A structure with ANY exit whose stored P&L differs from the old replay (typed in the Archive editor, or from real per-leg fills) is
left entirely alone, so a typed figure is never combined with a moved entry price.

Structures with a per-leg exit or a roll cannot be matched lot by lot and are skipped. Every change adds an audit line to the
structure's notes. The move runs once (a setting remembers it); running it again changes nothing.
"""

import logging

from core.fifo import composite_entry, open_queue, with_entry_price
from core.models import StructureStatus, TradeEventType
from core.trade_entry import TradeError, recompute_structure_from_trades

logger = logging.getLogger(__name__)

KEY_FIFO_MIGRATED = "fifo_migrated"
_OPEN_STATUSES = (StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED)
_EXIT_EVENTS = (TradeEventType.STRUCTURE_PARTIAL_EXIT, TradeEventType.FULL_EXIT, TradeEventType.DISSOLVE)
_TOLERANCE = 1e-6


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= _TOLERANCE * max(1.0, abs(a), abs(b))


def migrate_to_fifo(repository) -> dict:
    """Re-allocate the stored figures first in, first out. Returns {"structures": [names changed], "exits": n, "positions": n}."""
    changed, exits_changed, positions_changed = [], 0, 0
    for structure in repository.get_all_structures():
        trades = repository.get_trades_for_structure(structure.structure_id)
        if not any(t.event_type in _EXIT_EVENTS for t in trades) or open_queue(trades) is None:
            continue
        try:
            fifo = recompute_structure_from_trades(structure, trades, fifo=True)
            average = recompute_structure_from_trades(structure, trades, fifo=False)
        except TradeError:
            logger.info("FIFO migration: %s cannot be replayed, left as it is", structure.name)
            continue
        if fifo.structure.status != structure.status or not _same(fifo.structure.legs[0].lots, structure.legs[0].lots):
            continue  # the stored legs are not what its trades give: do not guess
        by_id = {t.trade_id: t for t in trades}
        pnl_updates, hand_typed = {}, False
        for trade_id, (new_pnl, _cost) in fifo.exit_trade_updates.items():
            old_pnl, stored = average.exit_trade_updates.get(trade_id, (None, None))[0], by_id[trade_id].realized_pnl
            if old_pnl is None or stored is None:
                continue
            if not _same(stored, old_pnl):
                hand_typed = True  # a figure typed in the Archive editor, or real per-leg fills: not the engine's
            elif not _same(new_pnl, stored):  # the engine's own average-cost figure: now FIFO
                pnl_updates[trade_id] = (stored, new_pnl)
        if hand_typed:
            continue  # leave the whole structure as it is: a typed P&L and a moved entry price were never meant to go together
        legs = None
        if structure.status in _OPEN_STATUSES:
            stored_c, old_c, new_c = (composite_entry(x) for x in (structure.legs, average.structure.legs, fifo.structure.legs))
            if None in (stored_c, old_c, new_c) or not _same(stored_c, old_c):
                continue  # the legs are not what the trades give (real per-leg fills, a corrected price): the exits and the lots left open must move together, so neither does
            if not _same(new_c, old_c):
                legs = with_entry_price(structure.legs, new_c)
        if not pnl_updates and legs is None:
            continue
        for trade_id, (_old, new_pnl) in pnl_updates.items():
            repository.update_trade_realized_pnl(trade_id, new_pnl, by_id[trade_id].transaction_cost)  # the cost stays as stored
        parts = [f"exit P&L {old:,.2f} -> {new:,.2f}" for old, new in pnl_updates.values()]
        if legs is not None:
            parts.append(f"entry price of the open lots {composite_entry(structure.legs):g} -> {composite_entry(legs):g}")
        audit = "re-allocated first in, first out (was average cost; the total is unchanged): " + "; ".join(parts)
        if legs is not None:
            repository.update_structure_legs(structure.structure_id, legs, audit)
        else:
            repository.update_structure_fields(structure.structure_id, audit_note=audit)
        changed.append(structure.name)
        exits_changed += len(pnl_updates)
        positions_changed += legs is not None
    return {"structures": changed, "exits": exits_changed, "positions": positions_changed}


def migrate_to_fifo_once(repository) -> dict | None:
    """Run migrate_to_fifo the first time only (the setting is saved after it succeeds); None if it had already run."""
    if repository.get_setting(KEY_FIFO_MIGRATED, False):
        return None
    report = migrate_to_fifo(repository)
    repository.set_setting(KEY_FIFO_MIGRATED, True)
    if report["structures"]:
        logger.info("Moved to first in, first out: %d structure(s), %d exit(s), %d open position(s): %s",
                    len(report["structures"]), report["exits"], report["positions"], ", ".join(report["structures"]))
    return report
