"""Pure P&L calculation engine.

Every function here is stateless and side-effect-free: it takes prices and
positions as input and returns P&L numbers as output. This module has no
dependency on Dash, adapters, or the database — only on core.models and the
standard library.
"""

import math
from datetime import date, datetime, timezone

from core.models import Leg, PnLRecord, Structure, StructureStatus, Trade, TradeEventType

_EXIT_EVENT_TYPES = {TradeEventType.PARTIAL_EXIT, TradeEventType.FULL_EXIT, TradeEventType.ROLL}


# ----------------------------------------------------------------------
# Leg PnL
# ----------------------------------------------------------------------


def calculate_leg_unrealized_pnl(
    entry_price: float,
    current_price: float,
    ratio: int,
    lots: float,
    multiplier: float,
    direction: str = "buy",
) -> float:
    """Unrealized P&L for a single leg.

    Formula: (current_price - entry_price) * direction_multiplier * ratio * lots * multiplier

    The ratio sign gives the leg's side within the structure (+1 long leg, -1 short
    leg). `direction` is the side the position was entered on: "buy" (multiplier +1)
    or "sell" (multiplier -1, so a falling price is a profit).

    Raises ValueError if lots < 0, multiplier <= 0, ratio == 0, direction is not
    "buy"/"sell", or if the result is not finite (NaN/inf).
    """
    if direction not in ("buy", "sell"):
        raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")
    direction_multiplier = 1 if direction == "buy" else -1
    if lots < 0:
        raise ValueError(f"lots must be >= 0, got {lots}")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")
    if ratio == 0:
        raise ValueError("ratio must be non-zero")

    pnl = (current_price - entry_price) * direction_multiplier * ratio * lots * multiplier
    if not math.isfinite(pnl):
        raise ValueError(f"calculated P&L is not finite: {pnl}")
    return pnl


def calculate_leg_realized_pnl(
    entry_price: float,
    exit_price: float,
    ratio: int,
    lots: float,
    multiplier: float,
    direction: str = "buy",
) -> float:
    """Realized P&L for a single leg at exit.

    Same formula and validation as calculate_leg_unrealized_pnl (including the
    `direction` sign), but using exit_price. Called once at trade exit.
    """
    return calculate_leg_unrealized_pnl(entry_price, exit_price, ratio, lots, multiplier, direction)


def calculate_average_entry_price(
    existing_lots: float,
    existing_avg_price: float,
    new_lots: float,
    new_price: float,
) -> float:
    """Weighted average entry price after adding to an existing position.

    Formula:
    ((existing_lots * existing_avg_price) + (new_lots * new_price))
    / (existing_lots + new_lots)

    Raises ValueError if existing_lots < 0, new_lots <= 0, or the resulting
    total lots is 0 (division-by-zero guard).
    """
    if existing_lots < 0:
        raise ValueError(f"existing_lots must be >= 0, got {existing_lots}")
    if new_lots <= 0:
        raise ValueError(f"new_lots must be > 0, got {new_lots}")

    total_lots = existing_lots + new_lots
    if total_lots == 0:
        raise ValueError("total lots is 0; cannot compute average entry price")

    avg_price = ((existing_lots * existing_avg_price) + (new_lots * new_price)) / total_lots
    if not math.isfinite(avg_price):
        raise ValueError(f"calculated average entry price is not finite: {avg_price}")
    return avg_price


# ----------------------------------------------------------------------
# Structure PnL
# ----------------------------------------------------------------------


def calculate_structure_unrealized_pnl(
    legs: list[Leg],
    live_prices: dict[str, float],
) -> tuple[float, dict[str, float], list[str]]:
    """Aggregate unrealized P&L across all traded legs of a structure.

    Only legs where leg.is_traded is True are considered; legs with
    lots == 0 are skipped silently. The entry price used is
    leg.average_entry_price if set, else leg.entry_price.

    Returns (total_unrealized_pnl, leg_pnl_breakdown, missing_prices).
    Legs whose symbol is not present in live_prices are excluded from the
    total (not counted as zero) and their symbol is added to missing_prices
    — the caller must inspect that list.
    """
    total = 0.0
    breakdown: dict[str, float] = {}
    missing: list[str] = []

    for leg in legs:
        if not leg.is_traded:
            continue

        symbol = leg.contract.symbol
        if symbol not in live_prices:
            missing.append(symbol)
            continue

        entry_price = leg.average_entry_price if leg.average_entry_price is not None else leg.entry_price
        leg_pnl = calculate_leg_unrealized_pnl(
            entry_price=entry_price,
            current_price=live_prices[symbol],
            ratio=leg.ratio,
            lots=leg.lots,
            multiplier=leg.contract.multiplier,
            direction=leg.direction,
        )
        breakdown[leg.leg_id] = leg_pnl
        total += leg_pnl

    return total, breakdown, missing


def calculate_structure_realized_pnl(trades: list[Trade]) -> float:
    """Sum realized_pnl across all exit/roll Trade events for a structure.

    Only trades where event_type is PARTIAL_EXIT, FULL_EXIT, or ROLL and
    realized_pnl is not None are counted. Returns 0.0 if no such trades exist.
    """
    total = 0.0
    for trade in trades:
        if trade.event_type in _EXIT_EVENT_TYPES and trade.realized_pnl is not None:
            total += trade.realized_pnl
    return total


def build_pnl_record(
    structure_id: str,
    legs: list[Leg],
    trades: list[Trade],
    live_prices: dict[str, float],
    stale_symbols: list[str],
) -> PnLRecord:
    """Build a complete PnLRecord for a structure.

    Combines calculate_structure_unrealized_pnl and
    calculate_structure_realized_pnl, records only the live prices actually
    used by this structure's legs, and flags is_stale if any of the
    structure's leg symbols appear in stale_symbols.

    Raises ValueError if the unrealized + realized total is not finite.
    """
    unrealized, _, _ = calculate_structure_unrealized_pnl(legs, live_prices)
    realized = calculate_structure_realized_pnl(trades)
    total = unrealized + realized
    if not math.isfinite(total):
        raise ValueError(f"total P&L is not finite: {total}")

    leg_symbols = {leg.contract.symbol for leg in legs}
    last_price_used = {
        symbol: price for symbol, price in live_prices.items() if symbol in leg_symbols
    }
    structure_stale_symbols = sorted(leg_symbols & set(stale_symbols))
    is_stale = bool(structure_stale_symbols)

    return PnLRecord(
        structure_id=structure_id,
        unrealized_pnl=unrealized,
        realized_pnl=realized,
        total_pnl=total,
        last_price_used=last_price_used,
        is_stale=is_stale,
        stale_symbols=structure_stale_symbols,
    )


# ----------------------------------------------------------------------
# Portfolio PnL
# ----------------------------------------------------------------------

_OPEN_STATUSES = {StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED}


def calculate_portfolio_pnl(
    structures: list[Structure],
    trades_by_structure: dict[str, list[Trade]],
    live_prices: dict[str, float],
    stale_symbols: list[str],
    closed_structures: list[Structure] | None = None,
    closed_trades_by_structure: dict[str, list[Trade]] | None = None,
) -> dict:
    """Calculate a portfolio-level P&L summary across all open structures.

    Only structures with status OPEN or PARTIALLY_CLOSED contribute unrealized P&L,
    positions and per-structure rows. `total_realized` is the realized P&L of those
    open structures only. Realized P&L booked by CLOSED structures is added to
    `total_realized_all_time` (open + closed), so it does not vanish when a structure
    closes; pass `closed_structures` and their trades to include it.
    net_lots_by_product sums ratio * lots per product across all legs of
    all included structures. largest_winner/largest_loser are the
    structures with the highest/lowest total P&L (unrealized + realized),
    or None if there are no open structures.
    """
    open_structures = [s for s in structures if s.status in _OPEN_STATUSES]

    total_unrealized = 0.0
    total_realized = 0.0
    per_structure: dict[str, dict] = {}
    open_leg_count = 0
    net_lots_by_product: dict[str, float] = {}
    pnl_by_structure: dict[str, float] = {}

    for structure in open_structures:
        trades = trades_by_structure.get(structure.structure_id, [])
        unrealized, _, missing_prices = calculate_structure_unrealized_pnl(structure.legs, live_prices)
        realized = calculate_structure_realized_pnl(trades)
        total = unrealized + realized

        leg_symbols = {leg.contract.symbol for leg in structure.legs}
        is_stale = bool(leg_symbols & set(stale_symbols))

        per_structure[structure.structure_id] = {
            "unrealized": unrealized,
            "realized": realized,
            "total": total,
            "is_stale": is_stale,
            "missing_prices": missing_prices,
        }
        pnl_by_structure[structure.structure_id] = total

        total_unrealized += unrealized
        total_realized += realized
        open_leg_count += structure.leg_count

        for leg in structure.legs:
            if not leg.is_traded:
                continue
            product = leg.contract.product
            net_lots_by_product[product] = net_lots_by_product.get(product, 0.0) + (leg.ratio * leg.lots)

    closed_realized = sum(
        calculate_structure_realized_pnl((closed_trades_by_structure or {}).get(s.structure_id, []))
        for s in (closed_structures or [])
        if s.status == StructureStatus.CLOSED
    ) or 0.0

    if pnl_by_structure:
        winner_id = max(pnl_by_structure, key=lambda sid: pnl_by_structure[sid])
        loser_id = min(pnl_by_structure, key=lambda sid: pnl_by_structure[sid])
        largest_winner = {"structure_id": winner_id, "pnl": pnl_by_structure[winner_id]}
        largest_loser = {"structure_id": loser_id, "pnl": pnl_by_structure[loser_id]}
    else:
        largest_winner = None
        largest_loser = None

    return {
        "total_unrealized": total_unrealized,
        "total_realized": total_realized,
        "total_realized_all_time": total_realized + closed_realized,
        "total_pnl": total_unrealized + total_realized,
        "per_structure": per_structure,
        "open_structure_count": len(open_structures),
        "open_leg_count": open_leg_count,
        "net_lots_by_product": net_lots_by_product,
        "largest_winner": largest_winner,
        "largest_loser": largest_loser,
        "has_stale_data": any(s["is_stale"] for s in per_structure.values()),
        "stale_symbols": list(stale_symbols),
        "calculated_at": datetime.now(timezone.utc),
    }


# ----------------------------------------------------------------------
# Today's PnL
# ----------------------------------------------------------------------


def calculate_todays_pnl(
    pnl_records: list[PnLRecord],
    reference_date: date | None = None,
) -> float:
    """P&L change since the start of the reference date (default: today UTC).

    Uses the earliest PnLRecord of the day as the baseline and the most
    recent as the current value; returns the difference in total_pnl.
    Returns 0.0 if fewer than 2 records fall on the reference date.
    """
    ref_date = reference_date if reference_date is not None else datetime.now(timezone.utc).date()
    todays_records = sorted(
        (r for r in pnl_records if r.timestamp.date() == ref_date),
        key=lambda r: r.timestamp,
    )
    if len(todays_records) < 2:
        return 0.0
    return todays_records[-1].total_pnl - todays_records[0].total_pnl


def calculate_todays_realized_pnl(
    all_trades: list[Trade],
    reference_date: date | None = None,
) -> float:
    """Realized P&L booked on `reference_date` (default: today, UTC).

    Sums realized_pnl of FULL_EXIT trades whose timestamp falls on that UTC date.
    Returns 0.0 if there were no exits that day.
    """
    ref_date = reference_date if reference_date is not None else datetime.now(timezone.utc).date()
    return sum(
        trade.realized_pnl
        for trade in all_trades
        if trade.event_type == TradeEventType.FULL_EXIT
        and trade.realized_pnl is not None
        and trade.timestamp.astimezone(timezone.utc).date() == ref_date
    ) or 0.0


# ----------------------------------------------------------------------
# Margin metrics
# ----------------------------------------------------------------------


def calculate_margin_efficiency(total_pnl: float, margin_used: float) -> float | None:
    """Margin efficiency: total_pnl / margin_used.

    This is informational only — it is NOT a risk constraint and must not be
    used to gate trading decisions. Returns None if margin_used == 0.
    """
    if margin_used == 0:
        return None
    efficiency = total_pnl / margin_used
    if not math.isfinite(efficiency):
        raise ValueError(f"calculated margin efficiency is not finite: {efficiency}")
    return efficiency


# ----------------------------------------------------------------------
# Net exposure rollup
# ----------------------------------------------------------------------


def calculate_net_exposure(structures: list[Structure]) -> dict[str, dict[str, float]]:
    """Net lots per contract month, summed across ALL structures.

    Returns {product_code: {contract_symbol: net_lots, ...}, ...}, summing
    ratio * lots for every traded leg (leg.is_traded == True) sharing that
    contract symbol, across every structure passed in (regardless of
    structure status).

    This is the single source of truth for the exposure map and VaR — do
    not reimplement this calculation elsewhere.
    """
    exposure: dict[str, dict[str, float]] = {}
    for structure in structures:
        for leg in structure.legs:
            if not leg.is_traded:
                continue
            product = leg.contract.product
            symbol = leg.contract.symbol
            product_map = exposure.setdefault(product, {})
            product_map[symbol] = product_map.get(symbol, 0.0) + (leg.ratio * leg.lots)
    return exposure
