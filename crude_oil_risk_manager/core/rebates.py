"""Exchange rebates: what comes back per lot and per side on a structure, matched by its products and its type.

A rate row is (product, structure type, monthly, yearly), in dollars per lot per side of the whole structure (as the exchange fee
sheet quotes them). Every month of a structure uses the same row, so only the type matters: a CL spread earns the CL spread rate
whichever months it is. A trade pays and earns on two sides (entry and exit), so its rebate is rate x lots x 2.

    products   the legs' products, sorted and joined with "+" (CL, BRN+BZ, CLBZ)
    type       core.models.StructureType's value; on a box product the plain types mean the box types (a CLBZ "spread" is a spreadbox)
    count      which programmes are added: both (the default), the monthly one only, or the yearly one only

The rows are edited in Settings and saved as a list in the settings table; until then DEFAULT_RATES (the fee sheet) applies.
A structure with no matching row earns no rebate. The Archive adds the rebate to a trade's PnL as a second figure and never
replaces the stored realized PnL.
"""

from core.models import Structure, StructureType
from core.money import cents
from core.models.structure import parse_structure_type

KEY_REBATE_RATES = "rebate_rates"
KEY_REBATE_COUNT = "rebate_count"
SIDES_PER_TRADE = 2  # entry and exit

COUNT_BOTH, COUNT_MONTHLY, COUNT_YEARLY = "both", "monthly", "yearly"
COUNT_CHOICES = {COUNT_BOTH: "Monthly + yearly", COUNT_MONTHLY: "Monthly only", COUNT_YEARLY: "Yearly only"}

_BOX_TYPE = {
    StructureType.OUTRIGHT.value: StructureType.OUTRIGHTBOX.value, StructureType.SPREAD.value: StructureType.SPREADBOX.value,
    StructureType.FLY.value: StructureType.FLYBOX.value, StructureType.DFLY.value: StructureType.DFLYBOX.value,
}


_LEG_TYPE = {0: StructureType.OUTRIGHT.value, 1: StructureType.SPREAD.value, 2: StructureType.FLY.value, 3: StructureType.DFLY.value}


def _row(product: str, kind: StructureType, monthly: float, yearly: float) -> dict:
    return {"product": product, "type": kind.value, "monthly": monthly, "yearly": yearly}


def _family(product: str, *rates: tuple[float, float], kinds=(StructureType.OUTRIGHT, StructureType.SPREAD, StructureType.FLY, StructureType.DFLY)) -> list[dict]:
    return [_row(product, kind, monthly, yearly) for kind, (monthly, yearly) in zip(kinds, rates)]


# Dollars per lot per side. Brent is the front-four-months rate; the rest months (outright 0.40, spread 0.80, fly 1.60 monthly) are
# not separated, since a structure's months are not looked at.
DEFAULT_RATES: list[dict] = [
    *_family("CL", (0.25, 0.04), (0.50, 0.08), (1.00, 0.16), (1.60, 0.16)),
    *_family("BRN", (0.25, 0.06), (0.50, 0.12), (1.00, 0.24), (2.00, 0.24)),
    *_family("BZ", (0.77, 0.0), (1.54, 0.0), (3.08, 0.0), (6.16, 0.0)),
    *_family("WBS", (0.70, 0.0), (1.40, 0.0), (2.80, 0.0), (5.60, 0.0)),
    *_family("G", (0.25, 0.0), (0.50, 0.0), (1.00, 0.0), (2.00, 0.0)),
    *_family("CLBZ", (1.68, 0.04), (3.36, 0.08), (6.72, 0.08), (13.44, 0.08),
             kinds=(StructureType.OUTRIGHTBOX, StructureType.SPREADBOX, StructureType.FLYBOX, StructureType.DFLYBOX)),
    *_family("CL+WBS", (1.80, 0.08), (3.60, 0.16), kinds=(StructureType.SPREADDIFF, StructureType.FLYDIFF)),
    *_family("BRN+BZ", (2.04, 0.12), (4.08, 0.24), kinds=(StructureType.SPREADDIFF, StructureType.FLYDIFF)),
]


_BOX_ALIASES = {"BZ+CL": "CLBZ", "CL+BZ": "CLBZ", "CLBZS": "CLBZ"}  # a CL|BZ structure holds CL and BZ legs; it is the CLBZ box


def products_key(products) -> str:
    key = "+".join(sorted({str(p).strip().upper() for p in products if str(p).strip()}))
    return _BOX_ALIASES.get(key, key)


def structure_products(structure: Structure) -> str:
    return products_key(leg.contract.product for leg in structure.legs)


def clean_rows(rows) -> list[dict]:
    """Rate rows from the Settings table: product upper-cased, type parsed ("Spread box" -> spreadbox), rates numbers >= 0.

    ValueError names the first bad row. Blank rows are dropped; a repeated product and type is an error."""
    cleaned, seen = [], set()
    for number, row in enumerate(rows or [], start=1):
        product, kind = products_key(str(row.get("product") or "").replace("_", "+").replace("|", "+").replace("-", "+").split("+")), row.get("type")
        if not product and not kind and row.get("monthly") in (None, "") and row.get("yearly") in (None, ""):
            continue
        try:
            parsed = parse_structure_type(kind)
        except ValueError as exc:
            raise ValueError(f"Rebates row {number}: {exc}") from None
        if not product or parsed is None:
            raise ValueError(f"Rebates row {number}: the product and the structure type are both needed")
        rates = []
        for name in ("monthly", "yearly"):
            value = row.get(name)
            try:
                number_value = 0.0 if value in (None, "") else float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Rebates row {number}: the {name} rebate must be a number") from None
            if not number_value >= 0 or number_value == float("inf"):
                raise ValueError(f"Rebates row {number}: the {name} rebate must be zero or more")
            rates.append(number_value)
        if (product, parsed.value) in seen:
            raise ValueError(f"Rebates row {number}: {product} {parsed.label.lower()} is listed twice")
        seen.add((product, parsed.value))
        cleaned.append({"product": product, "type": parsed.value, "monthly": rates[0], "yearly": rates[1]})
    return cleaned


class RebateBook:
    """The rate rows and which programmes count, with the lookup the Archive uses."""

    def __init__(self, rows: list[dict] | None = None, count: str = COUNT_BOTH):
        self.rows = [dict(r) for r in (DEFAULT_RATES if rows is None else rows)]
        self.count = count if count in COUNT_CHOICES else COUNT_BOTH
        self._by_key = {(r["product"], r["type"]): r for r in self.rows}

    @classmethod
    def load(cls, repository) -> "RebateBook":
        try:
            rows = repository.get_setting(KEY_REBATE_RATES, None)
            return cls(None if rows is None else clean_rows(rows), repository.get_setting(KEY_REBATE_COUNT, COUNT_BOTH))
        except (ValueError, TypeError, AttributeError):  # a damaged saved value must not stop the Archive
            return cls()

    def _type_key(self, structure: Structure, products: str) -> str:
        kind = structure.structure_type.value
        return _BOX_TYPE.get(kind, kind) if products.startswith("CLBZ") else kind

    def _row_rate(self, products: str, kind: str) -> float | None:
        row = self._by_key.get((products, kind))
        if row is None:
            return None
        monthly, yearly = row["monthly"], row["yearly"]
        return monthly if self.count == COUNT_MONTHLY else yearly if self.count == COUNT_YEARLY else monthly + yearly

    def _leg_rate(self, structure: Structure) -> float | None:
        """A structure with no row of its own (a BRN fly - fly) earns the rates of its legs, each leg taken as the outright, spread,
        fly or dfly its symbol is, times its ratio. None when any leg has no rate."""
        total = 0.0
        for leg in structure.legs:
            kind = _LEG_TYPE.get(leg.contract.symbol.count("-") + leg.contract.symbol.count("+"))
            rate = None if kind is None else self._row_rate(products_key([leg.contract.product]), kind)
            if rate is None:
                return None
            total += rate * abs(leg.ratio or 1)
        return total

    def per_side(self, structure: Structure) -> float | None:
        """Dollars per lot per side for the structure; None when no row matches its products and type (nor its legs')."""
        if not structure.legs:
            return None
        products = structure_products(structure)
        rate = self._row_rate(products, self._type_key(structure, products))
        return self._leg_rate(structure) if rate is None else rate

    def for_trade(self, structure: Structure, lots: float | None) -> float | None:
        """The rebate of a trade of `lots` (entry and exit): None when the structure has no rate."""
        rate = self.per_side(structure)
        return None if rate is None else cents(rate * (lots or 0.0) * SIDES_PER_TRADE)
