"""Structure-level trade entry, full exit, and per-leg partial exit.

Traders quote ONE structure price (the exchange-quoted spread/fly/outright price)
and ONE lot count. The P&L engine, however, works per leg
((live - entry) * ratio * lots * multiplier), so this module translates:

* Entry: leg entry prices are chosen so that the per-leg engine reproduces the
  structure-level P&L exactly. Single-leg structures simply take the structure
  price. For multi-leg structures every leg but the first is anchored at its
  current live price and the first leg absorbs the difference, so
  sign * sum(ratio * entry) == the entered structure price. Only the sum matters
  for P&L; individual leg entry prices are an allocation, not fills — UNLESS the
  caller supplies real per-leg prices via enter_trade's optional `leg_prices`,
  which are then used verbatim instead of being allocated.
* P&L: (live - entry) * side * lots * multiplier on structure prices, using
  core.pnl.calculate_leg_unrealized_pnl. `side` is the sign of the first leg's ratio
  (matching how core.structure_view builds structure prices) and the position's
  buy/sell direction (stored on every leg) flips the result for a sell.
* Exit: `exit_structure` is a full exit only (one structure-level trade against the
  first leg, all legs' lots zeroed). `partial_exit_legs` exits a subset of legs
  (and/or partial lots of a leg), each at its own price, computing realized P&L
  per leg via core.pnl.calculate_leg_realized_pnl and recording one Trade row per
  leg touched. Entry prices are kept for history either way (a closed/partially
  closed leg no longer counts towards exposure via Leg.is_traded).

Pure functions: nothing here touches the database.
"""

import math
from dataclasses import dataclass

from core.exceptions import CrudeOilRiskError
from core.models import Leg, Structure, StructureStatus, Trade, TradeEventType
from core.pnl import calculate_average_entry_price, calculate_leg_realized_pnl, calculate_leg_unrealized_pnl
from core.structure_utils import get_structure_type_from_symbol
from core.structure_view import structure_entry_price

_LOT_TOLERANCE = 1e-9
_OPEN_STATUSES = (StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED)


class TradeError(CrudeOilRiskError):
    """Blocking problems with a trade entry/exit; `errors` are safe to show the user."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class EntryResult:
    legs: list[Leg]
    trade: Trade
    status: StructureStatus
    audit_note: str


@dataclass
class ExitResult:
    legs: list[Leg]
    trade: Trade
    realized_pnl: float
    audit_note: str


@dataclass
class PartialExitResult:
    """Result of exiting a subset of legs (and/or partial lots of a leg), each at its own price."""

    legs: list[Leg]
    trades: list[Trade]  # one per leg acted on, event_type PARTIAL_EXIT
    realized_pnl_by_leg: dict[str, float]  # leg_id -> realized PnL for this exit
    status: StructureStatus
    audit_note: str

    @property
    def realized_pnl(self) -> float:
        return sum(self.realized_pnl_by_leg.values())


# ----------------------------------------------------------------------
# Structure-level helpers
# ----------------------------------------------------------------------


def structure_direction(legs: list[Leg]) -> int:
    """+1 if the first leg is long, -1 if short (the structure price is quoted from that side)."""
    return 1 if legs[0].ratio > 0 else -1


def structure_open_lots(structure: Structure) -> float:
    """Open lots of the structure (every leg carries the same base lots)."""
    return structure.legs[0].lots


def structure_pnl(
    structure: Structure, entry_price: float, current_price: float, lots: float, direction: str | None = None
) -> float:
    """P&L of `lots` of the structure between two structure prices, via core.pnl.

    `direction` ("buy"/"sell") defaults to the position's stored direction; pass it
    explicitly to preview a trade that has not been entered yet.
    """
    return calculate_leg_unrealized_pnl(
        entry_price=entry_price,
        current_price=current_price,
        ratio=structure_direction(structure.legs),
        lots=lots,
        multiplier=structure.legs[0].contract.multiplier,
        direction=direction or structure.legs[0].direction,
    )


def _leg_entry(leg: Leg) -> float | None:
    return leg.average_entry_price if leg.average_entry_price is not None else leg.entry_price


def structure_transaction_cost(structure: Structure, lots: float) -> float:
    """Sum of leg.transaction_cost_per_lot * lots across the structure's legs (one side)."""
    return sum(leg.transaction_cost_per_lot * lots for leg in structure.legs)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_price_and_lots(structure: Structure, price, lots, price_label: str) -> list[str]:
    errors = []
    if not _finite(price):
        errors.append(f"{price_label} is required.")
    elif len(structure.legs) == 1 and get_structure_type_from_symbol(structure.legs[0].contract.symbol) == "outright":
        if price <= 0:
            errors.append(f"{price_label} must be greater than 0.")
    # Spread/fly prices can legitimately be zero or negative, so only outrights must be positive.
    if not _finite(lots) or lots <= 0:
        errors.append("Lots must be greater than 0.")
    return errors


# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------


def allocate_leg_entry_prices(legs: list[Leg], structure_price: float, live_prices: dict[str, float]) -> list[float]:
    """Leg entry prices whose composite equals `structure_price` (see module docstring)."""
    if len(legs) == 1:
        return [structure_price / abs(legs[0].ratio)]

    missing = [leg.contract.symbol for leg in legs[1:] if leg.contract.symbol not in live_prices]
    if missing:
        raise TradeError(
            [
                f"Live prices for {', '.join(missing)} are needed to split the structure price across legs. "
                "Wait for the next price refresh and try again."
            ]
        )
    direction = structure_direction(legs)
    others = sum(leg.ratio * live_prices[leg.contract.symbol] for leg in legs[1:])
    return [(direction * structure_price - others) / legs[0].ratio] + [live_prices[leg.contract.symbol] for leg in legs[1:]]


def _validate_alert_levels(price, direction: str, stop_loss, target) -> list[str]:
    """Stop and target must be finite and on the correct side of the entry for the direction."""
    errors = []
    for label, level in (("Stop loss", stop_loss), ("Target", target)):
        if level is not None and not _finite(level):
            errors.append(f"{label} price must be a number.")
    if errors or not _finite(price) or direction not in ("buy", "sell"):
        return errors
    buy = direction == "buy"
    if stop_loss is not None and (stop_loss >= price if buy else stop_loss <= price):
        errors.append(f"For a {direction}, the stop loss must be {'below' if buy else 'above'} the entry price.")
    if target is not None and (target <= price if buy else target >= price):
        errors.append(f"For a {direction}, the target must be {'above' if buy else 'below'} the entry price.")
    return errors


def _validate_leg_prices(legs: list[Leg], leg_prices: list | None) -> list[str]:
    """All-or-nothing per-leg price override: every leg filled, or none.

    Each filled price must be finite and > 0 — every leg is itself always a single
    outright contract month, so the same positivity rule as a single-leg structure
    price applies (see _validate_price_and_lots).
    """
    if leg_prices is None:
        return []
    if len(leg_prices) != len(legs):
        return ["Leg price count does not match the number of legs."]
    filled = [p for p in leg_prices if p is not None]
    if not filled:
        return []
    if len(filled) != len(legs):
        return ["Enter a price for every leg, or leave all leg prices blank to auto-allocate."]
    errors = []
    for number, price in enumerate(leg_prices, start=1):
        if not _finite(price) or price <= 0:
            errors.append(f"Leg {number} price must be a number greater than 0.")
    return errors


def enter_trade(
    structure: Structure, price, lots, direction: str, notes: str | None, live_prices: dict[str, float],
    stop_loss_price: float | None = None, target_price: float | None = None,
    leg_prices: list[float | None] | None = None,
) -> EntryResult:
    """First entry (SHELL -> OPEN, TRADE event) or an add to an open position (ADD event, average price).

    `direction` is stored on every leg and flips the PnL sign for a sell. An add must
    use the position's existing direction. Optional stop/target levels are structure
    prices and are stored with the trade for the price alerts.

    `leg_prices`, if given with every element filled in (all-or-nothing), is used
    verbatim as each leg's entry price instead of the synthetic allocation from
    allocate_leg_entry_prices — a real recorded fill rather than an algebraic split.
    The typed `price` stays the structure-level Trade price regardless (what stop/
    target alerts and structure_entry_price compare against); the per-leg prices
    only affect what's written into each Leg's entry_price/average_entry_price and
    don't need to reconcile back to `price` algebraically.
    """
    errors = []
    if structure.status not in (StructureStatus.SHELL, *_OPEN_STATUSES):
        errors.append("A closed structure cannot take new trades. Create a new structure instead.")
    errors += _validate_price_and_lots(structure, price, lots, "Structure price")
    if direction not in ("buy", "sell"):
        errors.append("Direction must be buy or sell.")
    elif structure.status in _OPEN_STATUSES and structure.legs[0].lots > 0 and direction != structure.legs[0].direction:
        errors.append(
            f"This position is a {structure.legs[0].direction}; an add must be a {structure.legs[0].direction} too. "
            "Exit it and create a new structure to trade the other side."
        )
    errors += _validate_alert_levels(price, direction, stop_loss_price, target_price)
    errors += _validate_leg_prices(structure.legs, leg_prices)
    notes = (notes or "").strip()
    if len(notes) > 200:
        errors.append("Trade notes are limited to 200 characters.")
    if errors:
        raise TradeError(errors)

    custom_prices = leg_prices if leg_prices and all(p is not None for p in leg_prices) else None
    new_entries = custom_prices if custom_prices is not None else allocate_leg_entry_prices(structure.legs, price, live_prices)
    is_add = structure.status in _OPEN_STATUSES and structure.legs[0].lots > 0
    legs = []
    for leg, entry in zip(structure.legs, new_entries):
        if is_add:
            average = calculate_average_entry_price(leg.lots, _leg_entry(leg), lots, entry)
            legs.append(leg.model_copy(update={"lots": leg.lots + lots, "average_entry_price": average}))
        else:
            legs.append(
                leg.model_copy(
                    update={"lots": lots, "entry_price": entry, "average_entry_price": entry, "direction": direction}
                )
            )

    event = TradeEventType.ADD if is_add else TradeEventType.TRADE
    trade = Trade(
        structure_id=structure.structure_id,
        leg_id=structure.legs[0].leg_id,
        event_type=event,
        lots=lots,
        price=price,
        direction=direction,
        notes=notes,
        stop_loss_price=stop_loss_price,
        target_price=target_price,
        transaction_cost=structure_transaction_cost(structure, lots),
    )
    verb = "added" if is_add else "entered"
    return EntryResult(legs=legs, trade=trade, status=StructureStatus.OPEN, audit_note=f"trade {verb}: {lots:g} lots at {price:g}")


# ----------------------------------------------------------------------
# Exit
# ----------------------------------------------------------------------


def exit_structure(structure: Structure, price, lots, notes: str | None) -> ExitResult:
    """Full exit at one structure price. Partial exits are rejected."""
    errors = []
    if structure.status not in _OPEN_STATUSES:
        errors.append("Only an open structure can be exited.")
    errors += _validate_price_and_lots(structure, price, lots, "Exit price")
    notes = (notes or "").strip()
    if len(notes) > 200:
        errors.append("Trade notes are limited to 200 characters.")
    open_lots = structure_open_lots(structure)
    if not errors and abs(lots - open_lots) > _LOT_TOLERANCE:
        errors.append(f"Partial exits are not supported: exit all {open_lots:g} lots (or create a new structure).")
    entry_price = structure_entry_price(structure)
    if not errors and entry_price is None:
        errors.append("This structure has no entry price; it cannot be exited.")
    if errors:
        raise TradeError(errors)

    realized = structure_pnl(structure, entry_price, price, lots)
    exit_tc = structure_transaction_cost(structure, lots)
    trade = Trade(
        structure_id=structure.structure_id,
        leg_id=structure.legs[0].leg_id,
        event_type=TradeEventType.FULL_EXIT,
        lots=lots,
        price=price,
        direction="sell" if structure.legs[0].direction == "buy" else "buy",  # the closing side
        realized_pnl=realized,
        notes=notes,
        transaction_cost=exit_tc,
    )
    legs = [leg.model_copy(update={"lots": 0.0}) for leg in structure.legs]
    return ExitResult(
        legs=legs, trade=trade, realized_pnl=realized,
        audit_note=f"full exit: {lots:g} lots at {price:g}, realized {realized:+,.0f} gross"
        + (f" (exit cost {exit_tc:,.0f})" if exit_tc else ""),
    )


# ----------------------------------------------------------------------
# Per-leg partial exit
# ----------------------------------------------------------------------


def _recompute_naked(legs: list[Leg]) -> list[Leg]:
    """A leg is naked when it still has lots but at least one other leg in the
    structure is now at zero — its counterpart hedge is gone. Never naked in a
    single-leg (outright) structure."""
    if len(legs) < 2:
        return [leg.model_copy(update={"is_naked": False}) for leg in legs]
    any_zero = any(leg.lots <= _LOT_TOLERANCE for leg in legs)
    return [leg.model_copy(update={"is_naked": leg.lots > _LOT_TOLERANCE and any_zero}) for leg in legs]


def partial_exit_legs(
    structure: Structure, leg_exits: dict[str, tuple[float, float]], notes: str | None
) -> PartialExitResult:
    """Exit a subset of legs (and/or partial lots of a leg), each at its own price.

    `leg_exits` is {leg_id: (price, lots)}. Each leg is validated independently: lots
    must be > 0 and <= that leg's currently open lots; price must be finite and > 0
    (a leg is always a single contract month, so the same positivity rule as a
    single-leg outright structure price applies). Realized P&L is computed per leg via
    core.pnl.calculate_leg_realized_pnl against that leg's current average/entry price
    — genuinely per-leg, unlike exit_structure's single structure-level formula. One
    Trade row (event_type PARTIAL_EXIT) is produced per leg acted on. Entry prices are
    kept for history, like exit_structure. The resulting status is CLOSED once every
    leg is at zero, PARTIALLY_CLOSED if some but not all are, otherwise unchanged.
    """
    errors = []
    if structure.status not in _OPEN_STATUSES:
        errors.append("Only an open structure can be exited.")
    notes = (notes or "").strip()
    if len(notes) > 200:
        errors.append("Trade notes are limited to 200 characters.")
    if not leg_exits:
        errors.append("Select at least one leg to exit.")

    by_id = {leg.leg_id: leg for leg in structure.legs}
    for leg_id, (price, lots) in leg_exits.items():
        leg = by_id.get(leg_id)
        if leg is None:
            errors.append(f"Unknown leg {leg_id}.")
            continue
        if not _finite(lots) or lots <= 0:
            errors.append(f"Leg {leg.contract.symbol}: lots to exit must be greater than 0.")
        elif lots - leg.lots > _LOT_TOLERANCE:
            errors.append(f"Leg {leg.contract.symbol}: cannot exit more than its open {leg.lots:g} lots.")
        if not _finite(price) or price <= 0:
            errors.append(f"Leg {leg.contract.symbol}: exit price must be a number greater than 0.")
    if errors:
        raise TradeError(errors)

    trades: list[Trade] = []
    realized_by_leg: dict[str, float] = {}
    new_legs = list(structure.legs)
    note_parts = []
    for leg_id, (price, lots) in leg_exits.items():
        index = next(i for i, leg in enumerate(new_legs) if leg.leg_id == leg_id)
        leg = new_legs[index]
        entry = _leg_entry(leg)
        realized = calculate_leg_realized_pnl(
            entry_price=entry, exit_price=price, ratio=leg.ratio, lots=lots,
            multiplier=leg.contract.multiplier, direction=leg.direction,
        )
        tc = leg.transaction_cost_per_lot * lots
        trades.append(
            Trade(
                structure_id=structure.structure_id,
                leg_id=leg_id,
                event_type=TradeEventType.PARTIAL_EXIT,
                lots=lots,
                price=price,
                direction="sell" if leg.direction == "buy" else "buy",
                realized_pnl=realized,
                notes=notes,
                transaction_cost=tc,
            )
        )
        realized_by_leg[leg_id] = realized
        new_legs[index] = leg.model_copy(update={"lots": leg.lots - lots})
        note_parts.append(f"{leg.contract.symbol} {lots:g} lots at {price:g} (realized {realized:+,.0f})")

    new_legs = _recompute_naked(new_legs)
    if all(leg.lots <= _LOT_TOLERANCE for leg in new_legs):
        status = StructureStatus.CLOSED
    elif any(leg.lots <= _LOT_TOLERANCE for leg in new_legs):
        status = StructureStatus.PARTIALLY_CLOSED
    else:
        status = StructureStatus.OPEN

    return PartialExitResult(
        legs=new_legs, trades=trades, realized_pnl_by_leg=realized_by_leg, status=status,
        audit_note="partial exit: " + "; ".join(note_parts),
    )


# ----------------------------------------------------------------------
# Delete a trade: recompute everything downstream of it
# ----------------------------------------------------------------------


@dataclass
class RecomputeResult:
    """Result of replaying a structure's surviving trades after one was deleted."""

    structure: Structure
    # trade_id -> (realized_pnl, transaction_cost) for every FULL_EXIT trade whose stored
    # figures must be updated (they depend on the entry price, which can shift once an
    # earlier trade is removed from the history).
    exit_trade_updates: dict[str, tuple[float, float]]
    audit_note: str


def recompute_structure_from_trades(structure: Structure, trades: list[Trade]) -> RecomputeResult:
    """Rebuild leg state, status and every FULL_EXIT trade's realized_pnl from scratch.

    Used after deleting a trade: `trades` is the SURVIVING history (the deleted one already
    excluded), replayed in chronological order against a fresh SHELL copy of the structure,
    using the exact same enter_trade/exit_structure functions a live trade entry would use.
    This guarantees every derived figure (lots, entry/average price, direction, status,
    closed_at, and any surviving exit's realized PnL) lands exactly where it would be had
    the deleted trade never happened, instead of reversing one trade's math in place —
    which is much easier to get subtly wrong once average-price adds are involved.

    TRADE, ADD, FULL_EXIT and PARTIAL_EXIT are all replayed; a ROLL trade in `trades`
    still raises TradeError, since there is no pure function here to replay it.

    Multi-leg entry-price allocation (see allocate_leg_entry_prices) needs an "other legs"
    price purely to split a structure price across legs algebraically — only the weighted
    SUM matters for P&L (see that function's docstring), so each leg's own currently-stored
    entry price is reused as a stable placeholder. This means recomputing never depends on
    the live feed being available.
    """
    ordered = sorted(trades, key=lambda t: t.timestamp)
    anchors = {leg.contract.symbol: (_leg_entry(leg) or 0.0) for leg in structure.legs}
    shell_legs = [
        leg.model_copy(
            update={"lots": 0.0, "entry_price": None, "average_entry_price": None, "direction": "buy", "is_naked": False}
        )
        for leg in structure.legs
    ]
    current = structure.model_copy(
        update={"legs": shell_legs, "status": StructureStatus.SHELL, "closed_at": None, "close_trigger": None}
    )
    exit_updates: dict[str, tuple[float, float]] = {}

    for trade in ordered:
        if trade.event_type in (TradeEventType.TRADE, TradeEventType.ADD):
            result = enter_trade(
                current, trade.price, trade.lots, trade.direction, trade.notes, anchors,
                stop_loss_price=trade.stop_loss_price, target_price=trade.target_price,
            )
            current = current.model_copy(update={"legs": result.legs, "status": result.status})
        elif trade.event_type == TradeEventType.FULL_EXIT:
            result = exit_structure(current, trade.price, trade.lots, trade.notes)
            current = current.model_copy(
                update={
                    "legs": result.legs, "status": StructureStatus.CLOSED,
                    "closed_at": trade.timestamp, "close_trigger": "manual",
                }
            )
            exit_updates[trade.trade_id] = (result.realized_pnl, result.trade.transaction_cost)
        elif trade.event_type == TradeEventType.PARTIAL_EXIT:
            leg_result = partial_exit_legs(current, {trade.leg_id: (trade.price, trade.lots)}, trade.notes)
            updates = {"legs": leg_result.legs, "status": leg_result.status}
            if leg_result.status == StructureStatus.CLOSED:
                updates["closed_at"], updates["close_trigger"] = trade.timestamp, "manual"
            current = current.model_copy(update=updates)
            exit_updates[trade.trade_id] = (leg_result.realized_pnl_by_leg[trade.leg_id], leg_result.trades[0].transaction_cost)
        else:
            raise TradeError(
                [f"Cannot recompute past a {trade.event_type.value} trade; delete newer trades first."]
            )

    return RecomputeResult(
        structure=current,
        exit_trade_updates=exit_updates,
        audit_note=f"recomputed after a trade was deleted ({len(ordered)} remaining trade(s) replayed)",
    )
