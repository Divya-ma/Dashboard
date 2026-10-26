"""Archive rows, filters and summary statistics for exited trades (read-only).

Nothing here touches the database: callers pass the structures that are CLOSED or have exited some of their lots, with their trades.

ONE ROW PER EXIT. Exiting 2 of 5 lots puts a row in the Archive at once; exiting the other 3 adds a second row. Each exit takes its
lots from the earliest entries first (core.fifo), and that match gives the row its entry price, entry date, entry cost, stops and
targets (risk:reward) and entry snapshot. The row's realized PnL is the exit trade's saved figure (persisted at exit, never
recomputed) net of the exit's own transaction cost and the entry cost of the lots it closed, so a structure's rows add up to
core.pnl.calculate_net_realized_pnl of the whole structure. With per-leg exits, or a history that cannot be matched (no entry),
a closed structure keeps the older single row for the whole structure.
"""

from datetime import date, datetime

from core.fifo import match_exits
from core.journal import PRODUCT_LABELS, format_risk_reward, risk_reward, with_defaults
from core.models import Structure, StructureStatus, Trade, TradeEventType
from core.pnl import calculate_net_realized_pnl, calculate_structure_transaction_costs
from core.structure_view import structure_entry_price, structure_exit_price

RESULT_WIN, RESULT_LOSS, RESULT_FLAT = "Win", "Loss", "Flat"
PNL_ALL, PNL_WINNERS, PNL_LOSERS = "all", "winners", "losers"

_ENTRY_EVENTS = {TradeEventType.TRADE, TradeEventType.ADD}
_EXIT_EVENTS = {TradeEventType.PARTIAL_EXIT, TradeEventType.STRUCTURE_PARTIAL_EXIT, TradeEventType.FULL_EXIT, TradeEventType.DISSOLVE}


def _leg_side(leg) -> str:
    """Effective side of a leg: the ratio's own sign flipped by a sold position."""
    return "buy" if (leg.ratio > 0) == (leg.direction == "buy") else "sell"


def _legs_detail(structure: Structure, entered_lots: float | None) -> list[dict]:
    """Legs with symbol, side and lots. Lots are the entered lots x |ratio| (legs are zeroed at exit)."""
    details = []
    for leg in structure.legs:
        lots = (entered_lots if entered_lots is not None else leg.lots) * abs(leg.ratio)
        details.append({"symbol": leg.contract.symbol, "side": _leg_side(leg), "ratio": leg.ratio, "lots": lots})
    return details


def _legs_summary(details: list[dict]) -> str:
    """'CLZ25 B1, CLH26 S1', or a dash for a structure with no legs."""
    if not details:
        return "—"
    return ", ".join(f"{d['symbol']} {'B' if d['side'] == 'buy' else 'S'}{d['lots']:g}" for d in details)


def _date_or_none(value: datetime | None) -> date | None:
    return value.date() if value else None


def result_for(pnl: float) -> str:
    return RESULT_WIN if pnl > 0 else RESULT_LOSS if pnl < 0 else RESULT_FLAT


# the journal columns: id prefix "e_" / "x_" for the entry / exit snapshot
CURVE_PRODUCTS = ("CL", "BRN", "G")  # the curves are listed CL, BRN, GO
SNAPSHOT_FIELDS = (
    ("px_co", "prices", "CO"), ("px_cl", "prices", "CL"), ("px_go", "prices", "GO"),
    ("move_day", "move_day", None), ("move_min", "move_min", None),
    ("curve_cl", "curve", "CL"), ("curve_brn", "curve", "BRN"), ("curve_go", "curve", "G"),
    ("atr", "atr", None),
    ("vol_ratio", "vol_ratio", None), ("volume_ratio", "volume_ratio", None), ("spread_ticks", "spread_ticks", None),
    ("brent_wti", "brent_wti", None), ("crack", "crack", None), ("kink_z", "kink_z", None),
)


def journal_columns(journal: dict | None, trades: list[Trade]) -> dict:
    """The journal's part of an archive row: risk:reward, the entry and exit snapshots, trade ideas and payups.

    Every figure the journal does not have is None. Payups are ticks typed by hand (entry and exit, whole structure; the
    per-leg figures are kept in `payup_legs`)."""
    journal = with_defaults(journal)
    rr = risk_reward(trades)
    out = {
        "rr": rr["ratio"] if rr else None, "rr_text": format_risk_reward(rr), "rr_detail": rr,
        "ideas": journal["ideas"], "ideas_text": ", ".join(journal["ideas"]),
        "payup_entry": journal["payups"]["entry"]["total"], "payup_exit": journal["payups"]["exit"]["total"],
        "payup_legs": {"entry": journal["payups"]["entry"]["legs"], "exit": journal["payups"]["exit"]["legs"]},
        "snapshots": {"entry": journal["entry"], "exit": journal["exit"]},
        "conviction": journal["conviction"], "exit_reason": journal["exit_reason"],
        "mistakes": journal["mistakes"], "mistakes_text": ", ".join(journal["mistakes"]),
        "mfe": (journal["excursion"] or {}).get("mfe"), "mae": (journal["excursion"] or {}).get("mae"),  # the intraday figures if there are any
    }
    exc = journal["excursion"] or {}
    daily = exc.get("daily") or (exc if str(exc.get("basis", "")).startswith("daily") else None) or {}
    intraday = exc.get("intraday") or {}
    out.update(mfe_d=daily.get("mfe"), mae_d=daily.get("mae"), mfe_i=intraday.get("mfe"), mae_i=intraday.get("mae"))
    for prefix, kind in (("e_", "entry"), ("x_", "exit")):
        snap = journal[kind] or {}
        for column, key, sub_key in SNAPSHOT_FIELDS:
            value = snap.get(key)
            out[prefix + column] = (value or {}).get(sub_key) if sub_key is not None else value
    return out


def structure_id_of(row_id) -> str:
    """The structure an Archive row belongs to (a structure's first exit row has its id; later ones are '<id>#2', '<id>#3' ...)."""
    return str(row_id).split("#", 1)[0]


def _row_id(structure_id: str, index: int) -> str:
    return structure_id if index == 1 else f"{structure_id}#{index}"


def _finish_row(row: dict, structure: Structure, extra: dict, rr: dict | None, lots: float | None, pnl: float) -> dict:
    """The columns every kind of row shares: risk $, R, the curve edge of the entry's z-score, and the journal columns."""
    multiplier = structure.legs[0].contract.multiplier if structure.legs else 1000.0
    risk_usd = rr["risk"] * (lots or 0.0) * multiplier if rr and lots else None
    z = extra.get("e_kink_z")
    long_curve = bool(structure.legs) and ((structure.legs[0].direction == "buy") == (structure.legs[0].ratio > 0))
    extra["kink_edge"] = None if z is None else (-z if long_curve else z)  # a long structure that was cheap (z < 0), or a short one that was rich
    extra["risk_usd"] = risk_usd if risk_usd and risk_usd > 0 else None
    extra["r_multiple"] = pnl / extra["risk_usd"] if extra["risk_usd"] else None
    row.update(extra)
    return row


def _exit_row(structure: Structure, match, exits_in_total: int, journal: dict | None, last_exit_id: str, first_entry_id: str | None) -> dict:
    """The row of ONE exit: its lots, the entry lots they closed (FIFO) and the snapshots of that entry and that exit."""
    exit_trade, entry = match.exit, match.first_entry
    closed = structure.status == StructureStatus.CLOSED
    journal = with_defaults(journal)
    entry_snapshot = journal["entries"].get(entry.trade_id) or (journal["entry"] if entry.trade_id == first_entry_id else None)
    exit_snapshot = journal["exits"].get(exit_trade.trade_id) or (journal["exit"] if exit_trade.trade_id == last_exit_id else None)
    view = {**journal, "entry": entry_snapshot, "exit": exit_snapshot, "excursion": journal["excursion"] if closed else None}
    extra = journal_columns(view, match.entry_trades())
    transaction_costs = exit_trade.transaction_cost + match.entry_cost
    pnl = (exit_trade.realized_pnl or 0.0) - transaction_costs
    entry_date, exit_date = entry.timestamp.date(), exit_trade.timestamp.date()
    legs = _legs_detail(structure, match.lots)
    row = {
        "id": _row_id(structure.structure_id, match.index),
        "structure_id": structure.structure_id,
        "exit_trade_id": exit_trade.trade_id,
        "exit_number": match.index,
        "exits_of_structure": exits_in_total,
        "open_lots": 0.0 if closed else (structure.legs[0].lots if structure.legs else 0.0),
        "name": structure.name,
        "type": structure.structure_type.label,
        "legs_summary": _legs_summary(legs),
        "legs": legs,
        "symbols": [leg["symbol"] for leg in legs],
        "entry_date": entry_date.isoformat(),
        "exit_date": exit_date.isoformat(),
        "days_held": (exit_date - entry_date).days,
        "entry_price": match.entry_price,
        "exit_price": exit_trade.price,
        "lots": match.lots,
        "realized_pnl": pnl,
        "transaction_costs": transaction_costs,
        "result": result_for(pnl),
        "notes": structure.notes or "",
        "products": list(structure.products),
        "entry_at": entry.timestamp.isoformat(),
        "exit_at": exit_trade.timestamp.isoformat(),
    }
    return _finish_row(row, structure, extra, extra["rr_detail"], match.lots, pnl)


def _closed_structure_row(structure: Structure, trades: list[Trade], journals: dict[str, dict] | None) -> dict:
    """The older row: one for a whole closed structure (per-leg exits, or trades that cannot be matched lot by lot)."""
    entries = sorted((t for t in trades if t.event_type in _ENTRY_EVENTS), key=lambda t: t.timestamp)
    exits = sorted((t for t in trades if t.event_type in _EXIT_EVENTS), key=lambda t: t.timestamp)
    entry_date = _date_or_none(entries[0].timestamp if entries else None)
    exit_date = _date_or_none(structure.closed_at or (exits[-1].timestamp if exits else None))
    entered_lots = sum(t.lots for t in entries) if entries else None
    pnl = calculate_net_realized_pnl(trades)
    legs = _legs_detail(structure, entered_lots)
    extra = journal_columns((journals or {}).get(structure.structure_id), trades)
    exit_at = structure.closed_at or (exits[-1].timestamp if exits else None)
    row = {
        "id": structure.structure_id,
        "structure_id": structure.structure_id,
        "exit_trade_id": exits[-1].trade_id if exits else None,
        "exit_number": 1,
        "exits_of_structure": 1,
        "open_lots": 0.0,
        "name": structure.name,
        "type": structure.structure_type.label,
        "legs_summary": _legs_summary(legs),
        "legs": legs,
        "symbols": [leg["symbol"] for leg in legs],
        "entry_date": entry_date.isoformat() if entry_date else None,
        "exit_date": exit_date.isoformat() if exit_date else None,
        "days_held": (exit_date - entry_date).days if entry_date and exit_date else None,
        "entry_price": structure_entry_price(structure) if structure.legs else None,
        "exit_price": structure_exit_price(structure, trades),
        "lots": entered_lots,
        "realized_pnl": pnl,
        "transaction_costs": calculate_structure_transaction_costs(trades),
        "result": result_for(pnl),
        "notes": structure.notes or "",
        "products": list(structure.products),
        "entry_at": entries[0].timestamp.isoformat() if entries else None,
        "exit_at": exit_at.isoformat() if exit_at else None,
    }
    return _finish_row(row, structure, extra, extra["rr_detail"], entered_lots, pnl)


def build_archive_rows(
    structures: list[Structure], trades_by_structure: dict[str, list[Trade]], journals: dict[str, dict] | None = None,
) -> list[dict]:
    """One row per exit of the given structures, most recent exit first (see the module docstring).

    A structure that has not exited anything has no row. Missing dates (data anomalies) stay None. A closed structure whose
    exits cannot be matched to entries (or that has none, or no trades at all) still gets its single older row. `journals`
    (by structure id) adds the trade-journal columns (see journal_columns)."""
    rows = []
    for structure in structures:
        trades = trades_by_structure.get(structure.structure_id, [])
        matches = match_exits(trades)
        matched = [m for m in matches or [] if m.entries]
        clean = bool(matches) and len(matched) == len(matches) and not any(m.unmatched for m in matches)
        if not clean and structure.status == StructureStatus.CLOSED:
            rows.append(_closed_structure_row(structure, trades, journals))
            continue
        entries = sorted((t for t in trades if t.event_type in _ENTRY_EVENTS), key=lambda t: t.timestamp)
        exits = sorted((t for t in trades if t.event_type in _EXIT_EVENTS), key=lambda t: t.timestamp)
        journal = (journals or {}).get(structure.structure_id)
        for match in matched:  # an open structure with an inconsistent history still shows the exits that can be matched
            rows.append(_exit_row(structure, match, len(matches), journal, exits[-1].trade_id, entries[0].trade_id if entries else None))
    rows.sort(key=lambda r: r["exit_date"] or "", reverse=True)  # stable: the exits of one day keep their order
    return rows


def filter_archive_rows(
    rows: list[dict],
    start: date | None = None,
    end: date | None = None,
    structure_type: str | None = None,
    symbol_contains: str | None = None,
    pnl_filter: str | None = PNL_ALL,
) -> list[dict]:
    """Apply the archive filters together; each is optional.

    The date range is on the exit date (inclusive; rows with no exit date are dropped when a
    range is set). "Winners" keeps PnL >= 0 and "Losers" PnL <= 0, so a flat trade shows up
    under both and is easy to find.
    """
    needle = (symbol_contains or "").strip().lower()
    kept = []
    for row in rows:
        if start or end:
            exit_date = date.fromisoformat(row["exit_date"]) if row["exit_date"] else None
            if exit_date is None or (start and exit_date < start) or (end and exit_date > end):
                continue
        if structure_type and structure_type != "all" and row["type"].lower().replace(" ", "") != structure_type.lower():
            continue
        if needle and not any(needle in symbol.lower() for symbol in row["symbols"]):
            continue
        if pnl_filter == PNL_WINNERS and row["realized_pnl"] < 0:
            continue
        if pnl_filter == PNL_LOSERS and row["realized_pnl"] > 0:
            continue
        kept.append(row)
    return kept


def summarize(rows: list[dict]) -> dict:
    """Summary stats of a set of rows. Flat trades count in the total but not in the win rate."""
    pnls = [row["realized_pnl"] for row in rows]
    winners = sum(1 for p in pnls if p > 0)
    losers = sum(1 for p in pnls if p < 0)
    decided = winners + losers
    return {
        "total": len(pnls),
        "winners": winners,
        "losers": losers,
        "win_rate": winners / decided * 100.0 if decided else 0.0,
        "total_pnl": sum(pnls),
        "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
        "best": max(pnls) if pnls else 0.0,
        "worst": min(pnls) if pnls else 0.0,
    }


# ----------------------------------------------------------------------
# Deleting archived structures
# ----------------------------------------------------------------------


def delete_archived_structures(repository, structure_ids: list[str], shift_reset_baseline: bool = False) -> dict:
    """Permanently delete closed structures (with their legs, trades and PnL history).

    Home's realized PnL is the sum over the open and the CLOSED structures' trades, so deleting a closed structure takes its
    net realized PnL (after transaction costs) out of it; so do today's realized PnL and the archive. Only structures that
    are CLOSED are deleted; anything else is skipped and named.

    An Account Reset (Settings) shows the totals RELATIVE to a baseline captured at the reset, so a structure that was
    already closed at the reset is in that baseline: deleting it would push the shown figure below zero. With
    `shift_reset_baseline` both baselines move by the same amount, so the shown figures are unchanged by that deletion.

    Returns {"deleted": [names], "skipped": [names], "removed_pnl": net realized PnL taken out}.
    """
    from core.user_settings import KEY_PNL_BASELINE, KEY_PNL_REALIZED_BASELINE

    deleted, skipped, removed = [], [], 0.0
    for structure_id in structure_ids:
        structure = repository.get_structure(structure_id)
        if structure is None:
            skipped.append(structure_id)
            continue
        if structure.status != StructureStatus.CLOSED:
            skipped.append(structure.name)
            continue
        removed += calculate_net_realized_pnl(repository.get_trades_for_structure(structure_id))
        repository.delete_structure(structure_id)
        deleted.append(structure.name)
    if shift_reset_baseline and deleted:
        for key in (KEY_PNL_BASELINE, KEY_PNL_REALIZED_BASELINE):
            current = repository.get_setting(key, 0.0) or 0.0
            if current:
                repository.set_setting(key, current - removed)
    return {"deleted": deleted, "skipped": skipped, "removed_pnl": removed}
