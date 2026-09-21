"""Structure-level trade entry and full exit.

Traders quote ONE structure price (the exchange-quoted spread/fly/outright price)
and ONE lot count. The P&L engine, however, works per leg
((live - entry) * ratio * lots * multiplier), so this module translates:

* Entry: leg entry prices are chosen so that the per-leg engine reproduces the
  structure-level P&L exactly. Single-leg structures simply take the structure
  price. For multi-leg structures every leg but the first is anchored at its
  current live price and the first leg absorbs the difference, so
  sign * sum(ratio * entry) == the entered structure price. Only the sum matters
  for P&L; individual leg entry prices are an allocation, not fills.
* P&L: (live - entry) * direction * lots * multiplier on structure prices, using
  core.pnl.calculate_leg_unrealized_pnl. direction is the sign of the first leg's
  ratio, matching how core.structure_view builds structure prices.
* Exit: full exit only. It is recorded as one structure-level trade against the
  first leg, and the legs' lots are zeroed (entry prices are kept for history) so
  a closed structure no longer counts towards exposure.

Pure functions: nothing here touches the database.
"""

import math
from dataclasses import dataclass

from core.exceptions import CrudeOilRiskError
from core.models import Leg, Structure, StructureStatus, Trade, TradeEventType
from core.pnl import calculate_average_entry_price, calculate_leg_unrealized_pnl
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


# ----------------------------------------------------------------------
# Structure-level helpers
# ----------------------------------------------------------------------


def structure_direction(legs: list[Leg]) -> int:
    """+1 if the first leg is long, -1 if short (the structure price is quoted from that side)."""
    return 1 if legs[0].ratio > 0 else -1


def structure_open_lots(structure: Structure) -> float:
    """Open lots of the structure (every leg carries the same base lots)."""
    return structure.legs[0].lots


def structure_pnl(structure: Structure, entry_price: float, current_price: float, lots: float) -> float:
    """P&L of `lots` of the structure between two structure prices, via core.pnl."""
    return calculate_leg_unrealized_pnl(
        entry_price=entry_price,
        current_price=current_price,
        ratio=structure_direction(structure.legs),
        lots=lots,
        multiplier=structure.legs[0].contract.multiplier,
    )


def _leg_entry(leg: Leg) -> float | None:
    return leg.average_entry_price if leg.average_entry_price is not None else leg.entry_price


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


def enter_trade(
    structure: Structure, price, lots, direction: str, notes: str | None, live_prices: dict[str, float]
) -> EntryResult:
    """First entry (SHELL -> OPEN, TRADE event) or an add to an open position (ADD event, average price)."""
    errors = []
    if structure.status not in (StructureStatus.SHELL, *_OPEN_STATUSES):
        errors.append("A closed structure cannot take new trades. Create a new structure instead.")
    errors += _validate_price_and_lots(structure, price, lots, "Structure price")
    if direction not in ("buy", "sell"):
        errors.append("Direction must be buy or sell.")
    notes = (notes or "").strip()
    if len(notes) > 200:
        errors.append("Trade notes are limited to 200 characters.")
    if errors:
        raise TradeError(errors)

    new_entries = allocate_leg_entry_prices(structure.legs, price, live_prices)
    is_add = structure.status in _OPEN_STATUSES and structure.legs[0].lots > 0
    legs = []
    for leg, entry in zip(structure.legs, new_entries):
        if is_add:
            average = calculate_average_entry_price(leg.lots, _leg_entry(leg), lots, entry)
            legs.append(leg.model_copy(update={"lots": leg.lots + lots, "average_entry_price": average}))
        else:
            legs.append(leg.model_copy(update={"lots": lots, "entry_price": entry, "average_entry_price": entry}))

    event = TradeEventType.ADD if is_add else TradeEventType.TRADE
    trade = Trade(
        structure_id=structure.structure_id,
        leg_id=structure.legs[0].leg_id,
        event_type=event,
        lots=lots,
        price=price,
        direction=direction,
        notes=notes,
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
    trade = Trade(
        structure_id=structure.structure_id,
        leg_id=structure.legs[0].leg_id,
        event_type=TradeEventType.FULL_EXIT,
        lots=lots,
        price=price,
        direction="sell" if structure_direction(structure.legs) > 0 else "buy",
        realized_pnl=realized,
        notes=notes,
    )
    legs = [leg.model_copy(update={"lots": 0.0}) for leg in structure.legs]
    return ExitResult(
        legs=legs, trade=trade, realized_pnl=realized,
        audit_note=f"full exit: {lots:g} lots at {price:g}, realized {realized:+,.0f}",
    )
