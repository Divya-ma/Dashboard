"""How a book's traded legs become lots and dollar weights: the ONE place this is worked out.

Every page that needs "what am I long or short" (Home's net lots, the Exposure Map, VaR, the scenario shocks, the beta panels,
the New Structure builder) reads it from here, so they cannot disagree about a sold leg.

THE RULE. A traded leg contributes  ratio x side x lots,  with side +1 for a buy and -1 for a sell. The ratio carries the
leg's own side inside its structure (a fly's body is -2), and `Leg.direction` flips a position that was entered as a sell.
Only structures that carry risk (status OPEN or PARTIALLY_CLOSED) and only legs that are traded (lots and an entry price)
count.

TWO VIEWS of the same book, both here:
  as quoted    each leg symbol as it is (a fly stays a fly). This is what the VaR page uses: a fly has its own price history.
  outrights    each leg split into the outright contracts it is made of (core.structure_utils.decompose_to_outrights). This
               is what the Exposure Map shows and what beta / correlation against a book use, because outrights are the
               one thing every structure has in common.
Dollar weights multiply by the leg's contract multiplier: dollars of P&L per 1.0-point move.
"""

import logging
from typing import Callable, Iterator

from core.models import Leg, Structure, StructureStatus
from core.structure_utils import decompose_to_outrights, normalize_symbol

logger = logging.getLogger(__name__)

OPEN_STATUSES = {StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED}

# Dollars per point of one lot when a contract has not been saved and no multiplier has been typed in yet (1,000 barrels).
FALLBACK_MULTIPLIER = 1000.0
_EPSILON = 1e-9


def filter_open_structures(structures: list[Structure]) -> list[Structure]:
    """Structures that currently carry risk (status OPEN or PARTIALLY_CLOSED)."""
    return [s for s in structures if s.status in OPEN_STATUSES]


def side(leg: Leg) -> int:
    return 1 if leg.direction == "buy" else -1


def signed_lots(leg: Leg) -> float:
    """ratio x side x lots of one leg (0.0 for a leg that is not traded)."""
    return leg.ratio * side(leg) * leg.lots if leg.is_traded else 0.0


def _traded_legs(structures: list[Structure]) -> Iterator[tuple[Structure, Leg]]:
    for structure in filter_open_structures(structures):
        for leg in structure.legs:
            if leg.is_traded:
                yield structure, leg


def leg_outrights(leg: Leg, structure_name: str = "") -> dict[str, float]:
    """The outright contracts one traded leg is made of, with their signed lots ({} if the symbol cannot be split)."""
    try:
        return decompose_to_outrights(leg.contract.symbol, signed_lots(leg))
    except ValueError as exc:
        logger.warning("Skipping leg %s of %s (it cannot be split into outrights): %s", leg.contract.symbol, structure_name or "a structure", exc)
        return {}


def net_lots(structures: list[Structure], decompose: bool = False) -> dict[str, float]:
    """Net lots per symbol over the open structures' traded legs; legs that cancel exactly are dropped."""
    lots: dict[str, float] = {}
    for structure, leg in _traded_legs(structures):
        parts = leg_outrights(leg, structure.name) if decompose else {leg.contract.symbol: signed_lots(leg)}
        for symbol, amount in parts.items():
            lots[symbol] = lots.get(symbol, 0.0) + amount
    return {symbol: amount for symbol, amount in lots.items() if abs(amount) > _EPSILON}


def net_lots_by_product(structures: list[Structure]) -> dict[str, float]:
    """Net lots per product (CL, BRN, ...) over the traded legs of the structures passed in (not filtered by status)."""
    totals: dict[str, float] = {}
    for structure in structures:
        for leg in structure.legs:
            if leg.is_traded:
                totals[leg.contract.product] = totals.get(leg.contract.product, 0.0) + signed_lots(leg)
    return totals


def net_lots_by_product_and_symbol(structures: list[Structure]) -> dict[str, dict[str, float]]:
    """{product: {symbol: net lots}} over every structure passed in (regardless of status), as quoted."""
    exposure: dict[str, dict[str, float]] = {}
    for structure in structures:
        for leg in structure.legs:
            if leg.is_traded:
                book = exposure.setdefault(leg.contract.product, {})
                book[leg.contract.symbol] = book.get(leg.contract.symbol, 0.0) + signed_lots(leg)
    return exposure


def dollar_weights(structures: list[Structure], decompose: bool = False) -> dict[str, float]:
    """Dollar P&L per 1.0-point move, per symbol, over the open structures' traded legs (as quoted, or split into outrights)."""
    weights: dict[str, float] = {}
    for structure, leg in _traded_legs(structures):
        parts = leg_outrights(leg, structure.name) if decompose else {leg.contract.symbol: signed_lots(leg)}
        for symbol, amount in parts.items():
            weights[symbol] = weights.get(symbol, 0.0) + amount * leg.contract.multiplier
    return {symbol: weight for symbol, weight in weights.items() if abs(weight) > _EPSILON}


def candidate_dollar_weights(
    legs: list[dict], get_contract: Callable[[str], object], default_multiplier: float | None = None, decompose: bool = False,
) -> dict[str, float]:
    """Dollar P&L per 1.0-point move, per symbol, for 1 lot BOUGHT of a structure that has not been saved yet.

    `legs` are the builder's rows [{"symbol", "ratio"}, ...]. The multiplier is the saved contract's, else the one typed in
    the builder, else FALLBACK_MULTIPLIER. Same two views as `dollar_weights`; symbols that cancel are dropped.
    """
    weights: dict[str, float] = {}
    for row in legs:
        symbol = normalize_symbol(row.get("symbol"))
        try:
            ratio = float(row.get("ratio") or 0)
        except (TypeError, ValueError):
            ratio = 0.0
        if not symbol or ratio == 0:
            continue
        contract = get_contract(symbol)
        multiplier = contract.multiplier if contract is not None else (default_multiplier or FALLBACK_MULTIPLIER)
        if decompose:
            try:
                parts = decompose_to_outrights(symbol, ratio)
            except ValueError:
                continue
        else:
            parts = {symbol: ratio}
        for name, amount in parts.items():
            part_contract = get_contract(name) if decompose else contract
            part_multiplier = part_contract.multiplier if part_contract is not None else multiplier
            weights[name] = weights.get(name, 0.0) + amount * part_multiplier
    return {symbol: weight for symbol, weight in weights.items() if abs(weight) > _EPSILON}
