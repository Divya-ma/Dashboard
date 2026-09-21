"""Net exposure with dollar equivalents: the single input to VaR and the exposure map.

Net lots come from core.pnl.calculate_net_exposure (the single source of truth
for the lot rollup); this module only adds dollar values on top of it. Symbols
are exchange-quoted, so spreads and flies are their own symbols and are never
decomposed into legs here.
"""

import math
from datetime import datetime, timezone

from core.models import Contract, Structure, StructureStatus
from core.pnl import calculate_net_exposure

OPEN_STATUSES = {StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED}

# 1 basis point is defined as 0.01 price units.
_BP_IN_PRICE_UNITS = 0.01


def filter_open_structures(structures: list[Structure]) -> list[Structure]:
    """Structures that currently carry risk (status OPEN or PARTIALLY_CLOSED)."""
    return [s for s in structures if s.status in OPEN_STATUSES]


def get_structure_net_lots(structure: Structure) -> dict[str, float]:
    """Flat {symbol: net_lots} for a single structure, via calculate_net_exposure."""
    flat: dict[str, float] = {}
    for symbols in calculate_net_exposure([structure]).values():
        flat.update(symbols)
    return flat


def calculate_dollar_exposure(
    net_exposure: dict[str, dict[str, float]],
    live_prices: dict[str, float],
    contracts: dict[str, Contract],
) -> dict[str, dict[str, dict]]:
    """Extend net exposure with dollar values.

    Returns {product: {symbol: {"net_lots", "price", "dollar_exposure",
    "dollar_per_bp", "price_is_stale"}}}.

    Formulas:
        dollar_exposure = net_lots * price * multiplier
        dollar_per_bp   = net_lots * tick_value * (0.01 / tick_size)

    Assumption: dollar_per_bp assumes 1bp = 0.01 price units, i.e. it is the
    P&L of a 0.01 move in the quoted price, for any product.

    Every symbol is always included. If no (finite) live price was supplied,
    price and dollar_exposure are None and price_is_stale is True; a missing
    price is never treated as zero. dollar_per_bp does not depend on price,
    so it is still computed. If a symbol has no Contract in `contracts`, its
    dollar_exposure and dollar_per_bp are both None (multiplier/tick unknown).
    price_is_stale here means "no usable price was supplied"; adapter-level
    staleness flags are not visible to this pure function.
    """
    result: dict[str, dict[str, dict]] = {}
    for product, symbols in net_exposure.items():
        product_map: dict[str, dict] = {}
        for symbol, net_lots in symbols.items():
            price = live_prices.get(symbol)
            if price is not None and not math.isfinite(price):
                price = None
            contract = contracts.get(symbol)

            dollar_exposure = None
            dollar_per_bp = None
            if contract is not None:
                dollar_per_bp = net_lots * contract.tick_value * (_BP_IN_PRICE_UNITS / contract.tick_size)
                if price is not None:
                    dollar_exposure = net_lots * price * contract.multiplier

            product_map[symbol] = {
                "net_lots": net_lots,
                "price": price,
                "dollar_exposure": dollar_exposure,
                "dollar_per_bp": dollar_per_bp,
                "price_is_stale": price is None,
            }
        result[product] = product_map
    return result


def get_exposure_summary(
    structures: list[Structure],
    live_prices: dict[str, float],
    contracts: dict[str, Contract],
) -> dict:
    """Single entry point for all exposure data.

    Only structures with status OPEN or PARTIALLY_CLOSED contribute, so
    closed structures never inflate exposure. Combines calculate_net_exposure
    and calculate_dollar_exposure. Consumed by the Exposure Map tab, the VaR
    engine (position weights) and the Home tab (net lots by product).

    totals_by_product dollar_exposure and grand_total_dollar_exposure are
    None if any component symbol has no dollar exposure (missing price or
    contract); they are never silently summed over partial data.
    """
    live_structures = filter_open_structures(structures)
    net_exposure = calculate_net_exposure(live_structures)
    dollar = calculate_dollar_exposure(net_exposure, live_prices, contracts)

    by_product: dict[str, dict[str, dict]] = {}
    totals_by_product: dict[str, dict] = {}
    has_stale_prices = False
    grand_total: float | None = 0.0

    for product, symbols in dollar.items():
        by_product[product] = {
            symbol: {
                "net_lots": info["net_lots"],
                "dollar_exposure": info["dollar_exposure"],
                "dollar_per_bp": info["dollar_per_bp"],
                "price_is_stale": info["price_is_stale"],
            }
            for symbol, info in symbols.items()
        }
        has_stale_prices = has_stale_prices or any(i["price_is_stale"] for i in symbols.values())

        product_net_lots = sum(i["net_lots"] for i in symbols.values())
        exposures = [i["dollar_exposure"] for i in symbols.values()]
        product_dollar = None if any(e is None for e in exposures) else sum(exposures)
        totals_by_product[product] = {"net_lots": product_net_lots, "dollar_exposure": product_dollar}

        if product_dollar is None or grand_total is None:
            grand_total = None
        else:
            grand_total += product_dollar

    return {
        "by_product": by_product,
        "totals_by_product": totals_by_product,
        "grand_total_dollar_exposure": grand_total,
        "has_stale_prices": has_stale_prices,
        "calculated_at": datetime.now(timezone.utc),
    }
