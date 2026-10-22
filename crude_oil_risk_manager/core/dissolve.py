"""Dissolve: close several structures against EACH OTHER, with no exit transaction cost.

Example: you hold a Jan d-fly and cut it by trading two flies. The three positions net to nothing, but nothing was
traded to "exit" the d-fly itself, so exiting each one normally would charge an exit cost that was never paid. Dissolving
the group closes the lots of every selected structure at once, books ONLY the entry costs already incurred (they stay in
each structure's trades), and records the gross P&L of the whole group.

Gross P&L comes from one of two sources:
  leg prices   the price each leg contract was actually traded at when the group was cut (one price per contract, shared by
               every structure that has it, so legs that cancel between structures cancel in the P&L too); each structure
               then gets its own P&L by core.pnl.calculate_leg_realized_pnl.
  typed total  a gross P&L worked out elsewhere (when the leg prices are not known). It is split between the structures in
               proportion to their entry costs (equally if none has any), so the group total is exactly what was typed.
The group's net P&L is the gross figure minus the entry costs of the DISSOLVED lots (a partial dissolve takes only its share of each
structure's entry cost; the rest stays with the lots left open), which the P&L engine already deducts.

Pure functions: nothing here touches the database.
"""

import math
from dataclasses import dataclass, field

from core.exceptions import CrudeOilRiskError
from core.models import Leg, Structure, StructureStatus, Trade, TradeEventType
from core.pnl import calculate_leg_realized_pnl, entry_cost_for_lots
from core.positions import leg_outrights
from core.structure_view import structure_entry_price, structure_live_price

_LOT_TOLERANCE = 1e-9
_OPEN_STATUSES = (StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED)
CLOSE_TRIGGER = "dissolved"


class DissolveError(CrudeOilRiskError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class DissolvedStructure:
    structure: Structure
    lots: float  # lots dissolved
    legs: list[Leg]  # legs after
    status: StructureStatus
    trade: Trade
    realized_pnl: float  # this structure's share of the group's gross P&L
    entry_costs: float  # entry costs of the dissolved lots (already booked; deducted from its P&L). The rest stays with the lots left open


@dataclass
class DissolveResult:
    items: list[DissolvedStructure]
    gross_pnl: float
    entry_costs: float
    method: str  # "legs" | "typed"
    warnings: list[str] = field(default_factory=list)

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.entry_costs


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def dissolve_lots(structure: Structure, lots: float) -> tuple[list[Leg], StructureStatus]:
    """The legs and status after taking `lots` off every leg (no P&L: that is booked by the group)."""
    legs = [leg.model_copy(update={"lots": max(leg.lots - lots, 0.0), "is_naked": False}) for leg in structure.legs]
    if all(leg.lots <= _LOT_TOLERANCE for leg in legs):
        legs = [leg.model_copy(update={"lots": 0.0}) for leg in legs]
        return legs, StructureStatus.CLOSED
    return legs, StructureStatus.OPEN


def leg_symbols(structures: list[Structure]) -> list[str]:
    """Every distinct leg contract of the structures, in order of first appearance."""
    return list(dict.fromkeys(leg.contract.symbol for s in structures for leg in s.legs if leg.is_traded))


def _structure_realized(structure: Structure, lots: float, prices: dict[str, float]) -> float:
    total = 0.0
    for leg in structure.legs:
        if not leg.is_traded:
            continue
        entry = leg.average_entry_price if leg.average_entry_price is not None else leg.entry_price
        total += calculate_leg_realized_pnl(
            entry_price=entry, exit_price=prices[leg.contract.symbol], ratio=leg.ratio, lots=lots,
            multiplier=leg.contract.multiplier, direction=leg.direction,
        )
    return total


def net_position_warnings(structures: list[Structure], lots: dict[str, float]) -> list[str]:
    """Outright contracts the group does NOT net to zero in (a sign the selection is not one cut position)."""
    net: dict[str, float] = {}
    for s in structures:
        for leg in s.legs:
            if not leg.is_traded:
                continue
            scale = lots[s.structure_id] / leg.lots if leg.lots else 0.0
            for symbol, amount in leg_outrights(leg, s.name).items():
                net[symbol] = net.get(symbol, 0.0) + amount * scale
    left = {symbol: amount for symbol, amount in net.items() if abs(amount) > 1e-6}
    if not left:
        return []
    shown = ", ".join(f"{symbol} {amount:+g}" for symbol, amount in sorted(left.items())[:6])
    return [f"These structures do not net to zero in every contract (left over: {shown}); dissolving removes that position without any exit."]


def dissolve_structures(
    structures: list[Structure],
    lots: dict[str, float] | None = None,
    leg_prices: dict[str, float] | None = None,
    typed_gross: float | None = None,
    notes: str | None = None,
    trades_by_structure: dict[str, list[Trade]] | None = None,
) -> DissolveResult:
    """Close `lots` (default: all open lots) of each structure together, with no exit cost.

    Give EITHER `leg_prices` ({leg contract symbol: price} covering every leg contract) OR `typed_gross` (the group's
    gross P&L). Raises DissolveError listing every problem. `trades_by_structure` supplies each structure's booked
    entry costs (for the net figure and for splitting a typed gross P&L).
    """
    errors: list[str] = []
    if len(structures) < 2:
        errors.append("Select at least two structures to dissolve against each other.")
    notes = (notes or "").strip()
    lots = dict(lots or {})
    chosen: dict[str, float] = {}
    for s in structures:
        if s.status not in _OPEN_STATUSES:
            errors.append(f"{s.name}: only an open structure can be dissolved.")
            continue
        open_lots = s.legs[0].lots
        if any(abs(leg.lots - open_lots) > _LOT_TOLERANCE for leg in s.legs):
            errors.append(f"{s.name}: its legs are no longer at equal lots; exit them with the per-leg exit instead.")
            continue
        want = lots.get(s.structure_id, open_lots)
        if not _finite(want) or want <= 0:
            errors.append(f"{s.name}: lots to dissolve must be greater than 0.")
        elif want - open_lots > _LOT_TOLERANCE:
            errors.append(f"{s.name}: cannot dissolve {want:g} lots, only {open_lots:g} are open.")
        else:
            chosen[s.structure_id] = float(want)

    prices = {k: v for k, v in (leg_prices or {}).items() if v is not None and v != ""}
    use_legs = bool(prices)
    if use_legs and typed_gross not in (None, ""):
        errors.append("Give either the leg prices or a typed gross P&L, not both.")
    if not use_legs:
        if typed_gross in (None, ""):
            errors.append("Give the price of every leg contract, or type the gross P&L.")
        elif not _finite(typed_gross):
            errors.append("The gross P&L must be a number.")
    else:
        missing = [symbol for symbol in leg_symbols(structures) if symbol not in prices]
        bad = [symbol for symbol, price in prices.items() if not _finite(price)]
        if missing:
            errors.append(f"Price needed for: {', '.join(missing)} (or type the gross P&L instead).")
        if bad:
            errors.append(f"Not a number: {', '.join(bad)}.")
    if len(notes) > 120:
        errors.append("Notes are limited to 120 characters.")
    if errors:
        raise DissolveError(errors)

    trades_by_structure = trades_by_structure or {}
    costs = {s.structure_id: entry_cost_for_lots(trades_by_structure.get(s.structure_id, []), chosen[s.structure_id]) for s in structures}
    if use_legs:
        shares = {s.structure_id: _structure_realized(s, chosen[s.structure_id], prices) for s in structures}
        method = "legs"
    else:
        weight = {sid: costs[sid] for sid in costs}
        total_weight = sum(weight.values())
        shares = {
            sid: float(typed_gross) * ((weight[sid] / total_weight) if total_weight > 0 else 1.0 / len(structures))
            for sid in weight
        }
        method = "typed"

    names = ", ".join(s.name for s in structures)
    items = []
    for s in structures:
        legs, status = dissolve_lots(s, chosen[s.structure_id])
        if use_legs:
            price = structure_live_price(s, prices)
        else:
            price = structure_entry_price(s)
        trade = Trade(
            structure_id=s.structure_id, leg_id=s.legs[0].leg_id, event_type=TradeEventType.DISSOLVE, lots=chosen[s.structure_id],
            price=price if price is not None else 0.0, direction="sell" if s.legs[0].direction == "buy" else "buy",
            realized_pnl=shares[s.structure_id], transaction_cost=0.0,
            notes=(f"dissolved with {names}" + (f": {notes}" if notes else ""))[:200],
        )
        items.append(DissolvedStructure(s, chosen[s.structure_id], legs, status, trade, shares[s.structure_id], costs[s.structure_id]))
    warnings = net_position_warnings(structures, chosen)
    return DissolveResult(items, sum(shares.values()), sum(costs.values()), method, warnings)
