"""Inline leg edits (symbol and ratio) for an existing structure.

Lots, entry prices and leg ids are kept; only the symbol and ratio of each leg can
change. Editing a structure that has trades is allowed (the UI warns and asks for
confirmation first) and always produces an audit line describing exactly what changed.
"""

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


def apply_leg_edits(structure: Structure, symbols: list, ratios: list, get_contract) -> EditResult:
    """Validate the edited symbols/ratios and build the updated legs.

    New contracts copy the multiplier and tick specs of the structure's first leg.
    Raises StructureBuildError with every blocking problem.
    """
    rows = [{"symbol": normalize_symbol(s), "ratio": r} for s, r in zip(symbols or [], ratios or [])]
    errors, warnings = split_validation_messages(validate_structure_legs(rows))
    if len(rows) != len(structure.legs):
        errors.append("The number of legs cannot be changed here.")
    if errors:
        raise StructureBuildError(errors, warnings)

    template = structure.legs[0].contract
    legs, new_contracts, changes = [], [], []
    for number, (leg, row) in enumerate(zip(structure.legs, rows), start=1):
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
        legs.append(leg.model_copy(update={"contract": contract, "ratio": ratio}))

    audit = "edited: " + "; ".join(changes) if changes else "edit saved (no changes)"
    return EditResult(legs=legs, new_contracts=new_contracts, audit_note=audit, warnings=warnings)
