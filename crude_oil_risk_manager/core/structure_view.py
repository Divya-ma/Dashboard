"""Row building for the Structures tab: filtering, sorting and display values.

Pure functions over domain objects and the already-computed PnL numbers, so the
Structures callbacks stay thin. P&L is never recalculated here: unrealized,
realized and total come from the portfolio PnL summary (calculate_portfolio_pnl).

Structure price is the exchange-style composite of the leg prices:
sign(first leg ratio) * sum(ratio * leg price). That reproduces the quoted
spread (front - back) and fly (front - 2*mid + back) from the legs' outright
prices. The live poll only carries outright contracts, so this is derived; when
exchange-quoted structure symbols are polled, swap it in `_composite`'s callers.
"""

from datetime import datetime

from core.models import Leg, Structure, StructureStatus, Trade, TradeEventType
from core.pnl import calculate_structure_realized_pnl

FILTER_ACTIVE = "active"
FILTER_SHELL = "shell"
FILTER_OPEN = "open"

# PARTIALLY_CLOSED is legacy (no partial exits are offered); treat it as open so old rows stay visible.
_OPEN_STATUSES = [StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED]

_STATUS_FILTERS: dict[str, list[StructureStatus]] = {
    FILTER_ACTIVE: [StructureStatus.SHELL, *_OPEN_STATUSES],
    FILTER_SHELL: [StructureStatus.SHELL],
    FILTER_OPEN: _OPEN_STATUSES,
}

SORT_TOTAL_PNL = "total_pnl"
SORT_UNREALIZED_PNL = "unrealized_pnl"
SORT_DAYS_HELD = "days_held"
SORT_NAME = "name"

# sort key -> (row field, descending)
_SORTS = {
    SORT_TOTAL_PNL: ("total", True),
    SORT_UNREALIZED_PNL: ("unrealized", True),
    SORT_DAYS_HELD: ("days_held", True),
    SORT_NAME: ("name", False),
}

_EXIT_EVENTS = {TradeEventType.PARTIAL_EXIT, TradeEventType.FULL_EXIT}
_ENTRY_EVENTS = {TradeEventType.TRADE, TradeEventType.ADD}


def statuses_for_filter(status_filter: str | None) -> list[StructureStatus]:
    """Statuses to load for the status filter; unknown values fall back to all active."""
    return list(_STATUS_FILTERS.get(status_filter or FILTER_ACTIVE, _STATUS_FILTERS[FILTER_ACTIVE]))


def prices_from_store(live_prices: dict | None) -> dict[str, float]:
    """{symbol: price} from the `store-live-prices` payload."""
    return {symbol: data["price"] for symbol, data in (live_prices or {}).items() if "price" in data}


def _composite(legs: list[Leg], price_of) -> float | None:
    """Structure-level price from leg prices; None if any leg has no price."""
    total = 0.0
    for leg in legs:
        price = price_of(leg)
        if price is None:
            return None
        total += leg.ratio * price
    return total if legs[0].ratio > 0 else -total


def _entry_price_of(leg: Leg) -> float | None:
    return leg.average_entry_price if leg.average_entry_price is not None else leg.entry_price


def structure_entry_price(structure: Structure) -> float | None:
    """Composite entry price; None for a shell (no trade entered yet)."""
    return _composite(structure.legs, _entry_price_of)


def structure_live_price(structure: Structure, prices: dict[str, float]) -> float | None:
    """Composite live price; None if any leg's live price is missing."""
    return _composite(structure.legs, lambda leg: prices.get(leg.contract.symbol))


def structure_exit_price(structure: Structure, trades: list[Trade]) -> float | None:
    """Composite exit price from each leg's latest exit trade; None if a leg has none."""
    latest: dict[str, Trade] = {}
    for trade in sorted(trades, key=lambda t: t.timestamp):
        if trade.event_type in _EXIT_EVENTS:
            latest[trade.leg_id] = trade
    return _composite(
        structure.legs, lambda leg: latest[leg.leg_id].price if leg.leg_id in latest else None
    )


def _products(structure: Structure) -> list[str]:
    return sorted({leg.contract.product for leg in structure.legs})


def _type_label(structure: Structure) -> str:
    return structure.structure_type.value.capitalize()


def _sort_rows(rows: list[dict], sort_by: str | None) -> list[dict]:
    field, descending = _SORTS.get(sort_by or SORT_TOTAL_PNL, _SORTS[SORT_TOTAL_PNL])
    present = [r for r in rows if r[field] is not None]
    absent = [r for r in rows if r[field] is None]  # shells have no PnL / days held; keep them last
    key = (lambda r: r[field].lower()) if field == "name" else (lambda r: r[field])
    return sorted(present, key=key, reverse=descending) + absent


def build_active_rows(
    structures: list[Structure],
    per_structure_pnl: dict[str, dict] | None,
    live_prices: dict[str, float],
    product_filter: str | None = "all",
    sort_by: str | None = SORT_TOTAL_PNL,
) -> list[dict]:
    """Grid rows for shell/open structures. PnL comes from `per_structure_pnl` (shells have none)."""
    per_structure_pnl = per_structure_pnl or {}
    rows = []
    for structure in structures:
        if product_filter not in (None, "all") and product_filter not in _products(structure):
            continue
        pnl = per_structure_pnl.get(structure.structure_id)
        is_shell = structure.status == StructureStatus.SHELL
        rows.append(
            {
                "structure_id": structure.structure_id,
                "name": structure.name,
                "products": ", ".join(_products(structure)),
                "type": _type_label(structure),
                "structure_price": structure_entry_price(structure),
                "live_price": structure_live_price(structure, live_prices),
                "unrealized": pnl["unrealized"] if pnl else None,
                "realized": pnl["realized"] if pnl else None,
                "total": pnl["total"] if pnl else None,
                "lots": structure.legs[0].lots,
                "days_held": structure.days_held,
                "status": "SHELL" if is_shell else "OPEN",
            }
        )
    return _sort_rows(rows, sort_by)


def _closed_days_held(structure: Structure) -> int | None:
    if structure.closed_at is None:
        return structure.days_held
    return (structure.closed_at - structure.created_at).days


def _format_closed_at(closed_at: datetime | None) -> str | None:
    return closed_at.strftime("%Y-%m-%d %H:%M") if closed_at else None


def build_closed_rows(
    structures: list[Structure], trades_by_structure: dict[str, list[Trade]]
) -> list[dict]:
    """Grid rows for closed structures, most recently closed first."""
    rows = []
    for structure in structures:
        trades = trades_by_structure.get(structure.structure_id, [])
        first_leg_id = structure.legs[0].leg_id
        entered_lots = sum(t.lots for t in trades if t.leg_id == first_leg_id and t.event_type in _ENTRY_EVENTS)
        rows.append(
            {
                "structure_id": structure.structure_id,
                "name": structure.name,
                "products": ", ".join(_products(structure)),
                "type": _type_label(structure),
                "entry_price": structure_entry_price(structure),
                "exit_price": structure_exit_price(structure, trades),
                "realized": calculate_structure_realized_pnl(trades),
                "lots": entered_lots or structure.legs[0].lots,
                "days_held": _closed_days_held(structure),
                "closed_at": _format_closed_at(structure.closed_at),
                "close_trigger": (structure.close_trigger or "").capitalize() or None,
                "_closed_sort": structure.closed_at.isoformat() if structure.closed_at else "",
            }
        )
    rows.sort(key=lambda r: r["_closed_sort"], reverse=True)
    for row in rows:
        del row["_closed_sort"]
    return rows
