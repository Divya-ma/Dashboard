"""Archive rows, filters and summary statistics for closed structures (read-only).

Nothing here touches the database: callers pass CLOSED structures and their trades.
Realized PnL is the sum of the structure's saved exit trades (the value persisted at
exit, never recomputed). Dates come from the trades (entry) and the structure's
closed_at (exit); prices from core.structure_view.
"""

from datetime import date, datetime

from core.models import Structure, Trade, TradeEventType
from core.pnl import calculate_structure_realized_pnl
from core.structure_view import structure_entry_price, structure_exit_price

RESULT_WIN, RESULT_LOSS, RESULT_FLAT = "Win", "Loss", "Flat"
PNL_ALL, PNL_WINNERS, PNL_LOSERS = "all", "winners", "losers"

_ENTRY_EVENTS = {TradeEventType.TRADE, TradeEventType.ADD}
_EXIT_EVENTS = {TradeEventType.PARTIAL_EXIT, TradeEventType.FULL_EXIT}


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


def build_archive_rows(structures: list[Structure], trades_by_structure: dict[str, list[Trade]]) -> list[dict]:
    """One row per closed structure, most recently closed first.

    Missing entry/exit dates (data anomalies) stay None; days_held is then None too.
    A structure without legs still gets a row (legs shown as a dash).
    """
    rows = []
    for structure in structures:
        trades = trades_by_structure.get(structure.structure_id, [])
        entries = sorted((t for t in trades if t.event_type in _ENTRY_EVENTS), key=lambda t: t.timestamp)
        exits = sorted((t for t in trades if t.event_type in _EXIT_EVENTS), key=lambda t: t.timestamp)
        entry_date = _date_or_none(entries[0].timestamp if entries else None)
        exit_date = _date_or_none(structure.closed_at or (exits[-1].timestamp if exits else None))
        entered_lots = sum(t.lots for t in entries) if entries else None
        pnl = calculate_structure_realized_pnl(trades)
        legs = _legs_detail(structure, entered_lots)
        rows.append(
            {
                "structure_id": structure.structure_id,
                "name": structure.name,
                "type": structure.structure_type.value.capitalize(),
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
                "result": result_for(pnl),
                "notes": structure.notes or "",
            }
        )
    rows.sort(key=lambda r: r["exit_date"] or "", reverse=True)
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
        if structure_type and structure_type != "all" and row["type"].lower() != structure_type.lower():
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
