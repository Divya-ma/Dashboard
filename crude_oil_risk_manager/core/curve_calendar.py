"""Contract calendar for the Curve Kinks page: expiries, rolls and the 15-month structure ladder.

Dates are standard-rule approximations (see below), not exchange-published expiry dates.
Because of that, a contract is treated as rolled out ONE BUSINESS DAY BEFORE its rule-based
expiry (the "buffer day"): a contract is live on `as_of` only while as_of < buffer day. On the
buffer day itself the curve has already moved on, and the dashboard flags that the buffer was
used (`buffer_rolls`), so a wrong rule never leaves an expired contract on the curve.

Expiry rules
  CL  (NYMEX WTI): 3 business days before the 25th calendar day of the month preceding the
      delivery month; if the 25th is not a business day, 3 business days before the business
      day preceding the 25th.
  BRN (ICE Brent): the last business day of the second month preceding the delivery month.

Business days use an approximate holiday calendar per exchange (NYMEX: US holidays + Good
Friday; ICE Brent: UK bank holidays). One-off closures are not modelled.

Structures per product and family, for N months (default 15):
  outright N, spread N-1, fly N-2, dfly N-3 (all consecutive one-month spacing).
  spread = M1 - M2; fly = M1 - 2*M2 + M3; dfly = fly(k) - fly(k+1) = M1 - 3*M2 + 3*M3 - M4.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

from dateutil.easter import easter

from adapters.base import LETTER_TO_MONTH, MONTH_TO_LETTER

N_MONTHS = 15

OUTRIGHT, SPREAD, FLY, DFLY = "outright", "spread", "fly", "dfly"
FAMILIES = (OUTRIGHT, SPREAD, FLY, DFLY)
FAMILY_LABELS = {OUTRIGHT: "Outright", SPREAD: "Spread", FLY: "Fly", DFLY: "Dfly"}
# (leg count, weights on the consecutive outright legs)
FAMILY_WEIGHTS = {OUTRIGHT: (1,), SPREAD: (1, -1), FLY: (1, -2, 1), DFLY: (1, -3, 3, -1)}


@dataclass(frozen=True)
class ProductSpec:
    code: str  # internal product code, e.g. "BRN"
    api_code: str  # API code, e.g. "CO"
    name: str
    calendar: str  # "nymex" | "ice_uk"
    expiry_rule: str  # "cl" | "brent"


PRODUCTS: dict[str, ProductSpec] = {
    "CL": ProductSpec("CL", "CL", "WTI (CL)", "nymex", "cl"),
    "BRN": ProductSpec("BRN", "CO", "Brent (BRN)", "ice_uk", "brent"),
}


# ----------------------------------------------------------------------
# Holidays and business days
# ----------------------------------------------------------------------


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _us_observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def _nymex_holidays(year: int) -> frozenset[date]:
    new_year = date(year, 1, 1)
    days = {
        # CME does not observe a Saturday New Year's on the Friday before.
        new_year + timedelta(days=1) if new_year.weekday() == 6 else new_year,
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _us_observed(date(year, 6, 19)),  # Juneteenth
        _us_observed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _us_observed(date(year, 12, 25)),  # Christmas
    }
    if new_year.weekday() == 5:
        days.discard(new_year)
    return frozenset(days)


@lru_cache(maxsize=None)
def _uk_holidays(year: int) -> frozenset[date]:
    new_year = date(year, 1, 1)
    if new_year.weekday() >= 5:
        new_year += timedelta(days=7 - new_year.weekday())
    christmas, boxing = date(year, 12, 25), date(year, 12, 26)
    if christmas.weekday() == 5:  # Sat: Christmas -> Mon 27, Boxing (Sun) -> Tue 28
        christmas, boxing = date(year, 12, 27), date(year, 12, 28)
    elif christmas.weekday() == 6:  # Sun: Christmas -> Tue 27, Boxing Monday stays
        christmas, boxing = date(year, 12, 27), date(year, 12, 26)
    elif christmas.weekday() == 4:  # Fri: Boxing (Sat) -> Mon 28
        boxing = date(year, 12, 28)
    return frozenset(
        {
            new_year,
            easter(year) - timedelta(days=2),  # Good Friday
            easter(year) + timedelta(days=1),  # Easter Monday
            _nth_weekday(year, 5, 0, 1),  # Early May bank holiday
            _last_weekday(year, 5, 0),  # Spring bank holiday
            _last_weekday(year, 8, 0),  # Summer bank holiday
            christmas,
            boxing,
        }
    )


def is_business_day(d: date, calendar: str) -> bool:
    if d.weekday() >= 5:
        return False
    holidays = _nymex_holidays(d.year) if calendar == "nymex" else _uk_holidays(d.year)
    return d not in holidays


def add_business_days(d: date, n: int, calendar: str) -> date:
    """`d` moved by `n` business days (negative = earlier); `d` itself need not be a business day."""
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    while remaining:
        d += timedelta(days=step)
        if is_business_day(d, calendar):
            remaining -= 1
    return d


def _previous_business_day(d: date, calendar: str) -> date:
    while not is_business_day(d, calendar):
        d -= timedelta(days=1)
    return d


# ----------------------------------------------------------------------
# Expiries and the contract ladder
# ----------------------------------------------------------------------


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def contract_expiry(product: str, month: int, year: int) -> date:
    """Rule-based last trading day of the contract delivering in `month`/`year`."""
    spec = PRODUCTS[product]
    if spec.expiry_rule == "cl":
        prev_year, prev_month = _shift_month(year, month, -1)
        reference = _previous_business_day(date(prev_year, prev_month, 25), spec.calendar)
        return add_business_days(reference, -3, spec.calendar)
    exp_year, exp_month = _shift_month(year, month, -2)
    last_day = date(exp_year + (exp_month == 12), exp_month % 12 + 1, 1) - timedelta(days=1)
    return _previous_business_day(last_day, spec.calendar)


def buffer_day(product: str, month: int, year: int) -> date:
    """The business day before the rule-based expiry: the first day the contract is treated as rolled."""
    return add_business_days(contract_expiry(product, month, year), -1, PRODUCTS[product].calendar)


def api_symbol(product: str, month: int, year: int) -> str:
    """Outright API symbol, e.g. ("BRN", 12, 2026) -> "COZ26"."""
    return f"{PRODUCTS[product].api_code}{MONTH_TO_LETTER[month]}{year % 100:02d}"


def month_token(month: int, year: int) -> str:
    return f"{MONTH_TO_LETTER[month]}{year % 100:02d}"


def parse_contract(symbol: str) -> tuple[str, int, int]:
    """("COZ26") -> ("CO", 12, 2026). Works on the first leg of a spread/fly symbol too."""
    front = symbol.split("-")[0]
    code, token = front[:-3], front[-3:]
    return code, LETTER_TO_MONTH[token[0]], 2000 + int(token[1:])


@dataclass(frozen=True)
class CurveContract:
    position: int  # generic position, 1 = front
    symbol: str  # outright API symbol
    month: int
    year: int
    expiry: date
    buffer_day: date

    @property
    def label(self) -> str:
        return date(self.year, self.month, 1).strftime("%b%y")


def front_month(product: str, as_of: date) -> tuple[int, int]:
    """(month, year) of generic position 1 on `as_of`: first contract whose buffer day is after it."""
    year, month = as_of.year, as_of.month
    for _ in range(36):
        if as_of < buffer_day(product, month, year):
            return month, year
        year, month = _shift_month(year, month, 1)
    raise RuntimeError(f"no live {product} contract found for {as_of}")  # pragma: no cover


def front_symbol_by_expiry(product: str, day: date) -> str:
    """Front contract symbol on `day` by the pure expiry rule: live through its expiry day, no buffer.

    Used to cross-check the generic history (which rolls at the exchange's real expiry)."""
    year, month = day.year, day.month
    for _ in range(36):
        if day <= contract_expiry(product, month, year):
            return api_symbol(product, month, year)
        year, month = _shift_month(year, month, 1)
    raise RuntimeError(f"no {product} contract found for {day}")  # pragma: no cover


def generic_contracts(product: str, as_of: date, n: int = N_MONTHS) -> list[CurveContract]:
    """The n consecutive live contracts on `as_of`, generic position 1..n."""
    month, year = front_month(product, as_of)
    contracts = []
    for position in range(1, n + 1):
        contracts.append(
            CurveContract(position, api_symbol(product, month, year), month, year,
                          contract_expiry(product, month, year), buffer_day(product, month, year))
        )
        year, month = _shift_month(year, month, 1)
    return contracts


def buffer_rolls(product: str, as_of: date) -> list[CurveContract]:
    """Contracts that are on their buffer day today: rolled out one business day early, so the
    rule-based expiry date should be confirmed."""
    calendar = PRODUCTS[product].calendar
    month, year = front_month(product, as_of)
    # The contract just before the front one was dropped today if its buffer day is today.
    prev_year, prev_month = _shift_month(year, month, -1)
    rolled = []
    if buffer_day(product, prev_month, prev_year) == as_of and is_business_day(as_of, calendar):
        rolled.append(
            CurveContract(0, api_symbol(product, prev_month, prev_year), prev_month, prev_year,
                          contract_expiry(product, prev_month, prev_year), as_of)
        )
    return rolled


# ----------------------------------------------------------------------
# Structure ladder
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CurveStructure:
    family: str
    position: int  # generic position of the front leg, 1-based
    label: str  # e.g. "Dec26", "Dec26/Jan27", "Dec26/Jan27/Feb27"
    front_month: int  # delivery month of the front leg (for seasonality)
    legs: tuple[str, ...]  # consecutive outright API symbols
    leg_weights: tuple[int, ...]
    symbol: str  # API symbol to price live ("" for a dfly, which is built from two flies)
    generic_code: str  # history code ("CO1", "CO1-2", "CO1-2-3"; "" for a dfly)
    components: tuple[str, ...]  # live symbols combined to get the value
    component_weights: tuple[int, ...]

    @property
    def display(self) -> str:
        return self.label


def _structure_symbol(contracts: list[CurveContract]) -> str:
    return contracts[0].symbol + "".join(f"-{month_token(c.month, c.year)}" for c in contracts[1:])


def build_structures(product: str, as_of: date, n: int = N_MONTHS) -> dict[str, list[CurveStructure]]:
    """{family: [structures by position]} for the live n-month ladder on `as_of`."""
    api_code = PRODUCTS[product].api_code
    contracts = generic_contracts(product, as_of, n)
    ladder: dict[str, list[CurveStructure]] = {family: [] for family in FAMILIES}

    def make(family: str, start: int) -> CurveStructure:
        legs_count = len(FAMILY_WEIGHTS[family])
        legs = contracts[start:start + legs_count]
        label = "/".join(c.label for c in legs)
        if family == DFLY:
            return CurveStructure(
                family, start + 1, label, legs[0].month, tuple(c.symbol for c in legs),
                FAMILY_WEIGHTS[family], "", "",
                (_structure_symbol(contracts[start:start + 3]), _structure_symbol(contracts[start + 1:start + 4])),
                (1, -1),
            )
        symbol = _structure_symbol(legs)
        generic = api_code + str(start + 1) + "".join(f"-{start + 1 + i}" for i in range(1, legs_count))
        return CurveStructure(
            family, start + 1, label, legs[0].month, tuple(c.symbol for c in legs),
            FAMILY_WEIGHTS[family], symbol, generic, (symbol,), (1,),
        )

    for family in FAMILIES:
        legs_count = len(FAMILY_WEIGHTS[family])
        for start in range(0, n - legs_count + 1):
            ladder[family].append(make(family, start))
    return ladder


def family_length(family: str, n: int = N_MONTHS) -> int:
    return n - len(FAMILY_WEIGHTS[family]) + 1


def live_symbols(product: str, as_of: date, n: int = N_MONTHS) -> list[str]:
    """Every API symbol the live feed must price for this product (outrights, spreads, flies)."""
    ladder = build_structures(product, as_of, n)
    return [s.symbol for family in (OUTRIGHT, SPREAD, FLY) for s in ladder[family]]
