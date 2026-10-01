"""Inline leg edits (symbol, ratio, entry price) for an existing structure.

Lots and leg ids are kept; the symbol, ratio and (for a traded leg) entry price of
each leg can change. Editing a structure that has trades is allowed (the UI warns
and asks for confirmation first) and always produces an audit line describing
exactly what changed.
"""

import math
from dataclasses import dataclass, field

from core.models import Contract, Leg, Structure
from core.structure_builder import StructureBuildError, _contract_for_symbol
from core.structure_utils import normalize_symbol, split_validation_messages, validate_structure_legs


@dataclass
class EditResult:
    legs: list[Leg]
    new_contracts: list[Contract]
    audit_note: str
    warnings: list[str] = field(default_factory=list)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def apply_leg_edits(
    structure: Structure, symbols: list, ratios: list, get_contract, entry_prices: list | None = None
) -> EditResult:
    """Validate the edited symbols/ratios/entry prices and build the updated legs.

    New contracts copy the multiplier and tick specs of the structure's first leg.
    `entry_prices` (optional, one slot per leg) is a manual correction of a traded
    leg's recorded fill: a non-None, non-blank value there overwrites both
    entry_price and average_entry_price for that leg (a leg with no position yet
    has nothing to correct, so a value there is ignored). Raises StructureBuildError
    with every blocking problem.
    """
    rows = [{"symbol": normalize_symbol(s), "ratio": r} for s, r in zip(symbols or [], ratios or [])]
    errors, warnings = split_validation_messages(validate_structure_legs(rows))
    if len(rows) != len(structure.legs):
        errors.append("The number of legs cannot be changed here.")
    entry_prices = list(entry_prices or [])
    entry_prices += [None] * (len(structure.legs) - len(entry_prices))
    for number, (leg, price) in enumerate(zip(structure.legs, entry_prices), start=1):
        if price in (None, ""):
            continue
        if leg.is_traded and not (_finite(price) and price > 0):
            errors.append(f"Leg {number} entry price must be a number greater than 0.")
    if errors:
        raise StructureBuildError(errors, warnings)

    template = structure.legs[0].contract
    legs, new_contracts, changes = [], [], []
    for number, (leg, row, price) in enumerate(zip(structure.legs, rows, entry_prices), start=1):
        symbol, ratio = row["symbol"], int(row["ratio"])
        contract = leg.contract
        if symbol != leg.contract.symbol:
            contract = get_contract(symbol)
            if contract is None:
                contract = _contract_for_symbol(symbol, template.multiplier, template.tick_size, template.tick_value)
                new_contracts.append(contract)
            changes.append(f"leg {number} symbol {leg.contract.symbol} -> {symbol}")
            if leg.is_traded:
                warnings.append(f"Leg {number} already has a position; its entry price was not changed.")
        if ratio != leg.ratio:
            changes.append(f"leg {number} ratio {leg.ratio:+d} -> {ratio:+d}")

        update = {"contract": contract, "ratio": ratio}
        if leg.is_traded and price not in (None, ""):
            new_price = float(price)
            old_price = leg.average_entry_price if leg.average_entry_price is not None else leg.entry_price
            if new_price != old_price:
                changes.append(f"leg {number} entry price {old_price:g} -> {new_price:g} (manual correction)")
                update["entry_price"] = new_price
                update["average_entry_price"] = new_price
        legs.append(leg.model_copy(update=update))

    audit = "edited: " + "; ".join(changes) if changes else "edit saved (no changes)"
    return EditResult(legs=legs, new_contracts=new_contracts, audit_note=audit, warnings=warnings)
