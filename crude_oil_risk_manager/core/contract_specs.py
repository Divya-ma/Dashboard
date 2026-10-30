"""Contract size and tick of every product the dashboard trades: the one place the numbers live.

    CL, BRN, BZ, WBS, CLBZ, CLBZS   1,000 barrels  -> $1,000 per 1.0 point, tick 0.01 = $10
    G (ICE gasoil, API GO)   100 tonnes  -> $100 per 1.0 point, tick 0.25 = $25

`multiplier` is the dollar value of a 1.0 move of the quoted price for one lot, `tick_size` the smallest price step and
`tick_value` the dollars per tick (tick_size x multiplier). Anything that turns a price move into dollars, or a tick into
a price, reads it from here by PRODUCT (internal code: CL, BRN, BZ, WBS, G), so a new instrument is one entry.
The curve products (CL, BRN, G) take theirs from core.curve_calendar.PRODUCTS.
"""

from dataclasses import dataclass

from adapters.base import PRODUCT_TO_API_CODE
from core.curve_calendar import PRODUCTS


@dataclass(frozen=True)
class ContractSpec:
    product: str
    multiplier: float  # $ per 1.0 price point per lot
    tick_size: float
    tick_value: float  # $ per tick per lot


_EXTRA = {
    "BZ": ContractSpec("BZ", 1000.0, 0.01, 10.0),
    "WBS": ContractSpec("WBS", 1000.0, 0.01, 10.0),
    # CL-BZ boxes: both legs are 1,000 barrels quoted in dollars, so a box ticks like either leg
    "CLBZ": ContractSpec("CLBZ", 1000.0, 0.01, 10.0),
    "CLBZS": ContractSpec("CLBZS", 1000.0, 0.01, 10.0),
}
SPECS: dict[str, ContractSpec] = {
    **{code: ContractSpec(code, p.multiplier, p.tick_size, p.tick_value) for code, p in PRODUCTS.items()},
    **_EXTRA,
}
DEFAULT_SPEC = ContractSpec("", 1000.0, 0.01, 10.0)  # for a product the app does not know


def product_of_internal_symbol(symbol: str | None) -> str | None:
    """CL, BRN, BZ, WBS or G from an internal symbol such as 'GZ26' or 'CLZ26-F27' (longest product code first)."""
    if not symbol:
        return None
    text = str(symbol).strip().upper()
    for code in sorted(PRODUCT_TO_API_CODE, key=len, reverse=True):
        if text.startswith(code):
            return code
    return None


def spec_for_product(product: str | None) -> ContractSpec:
    return SPECS.get(product or "", DEFAULT_SPEC)


def spec_for_symbol(symbol: str | None) -> ContractSpec:
    """The contract spec of an internal symbol's product (the default 1,000-barrel spec if it is not recognised)."""
    return spec_for_product(product_of_internal_symbol(symbol))
