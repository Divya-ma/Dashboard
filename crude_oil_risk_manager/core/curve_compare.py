"""Curve Compare page: live curves (quote VWAP) against the previous settlement, up to 36 months out.

For one product, one curve type (outright / spread / fly / dfly) and the first N contracts, this
module works out which exchange symbols to price, turns raw quotes into prices, and builds the
curve. It does no I/O: quotes, fair values and settlements are handed in.

LIVE PRICE of a symbol = VWAP of its best bid and best ask, weighted by the opposite side's size
(the size-weighted mid):  (bid * ask_size + ask * bid_size) / (bid_size + ask_size).
The side with less size is the one likely to trade next, so the price leans towards it. With no
sizes it is the plain mid. Where there is no usable quote the /fairvalue/ price is used if the
caller has one.

WHERE A POINT'S PRICE COMES FROM, in order (a quiet back month often has no recent quote, but
TAQ still holds its standing bid/ask and /fairvalue/ prices almost the whole ladder):
  1. its own price - the newer of its quote VWAP and its fair value - if at most FRESH_SECONDS old
  2. built from other points (source "derived"), as below
  3. its own price again if at most STALE_LIMIT_SECONDS old (marked "old price")
  4. empty
Every value carries its age so a stale back-month price is visible, not hidden.

    spread k = outright k - outright k+1
    fly k    = quoted fly, else spread k - spread k+1
    dfly k   = M_k - 3 M_k+1 + 3 M_k+2 - M_k+3, built one of three ways (DFLY_METHODS):
        flies    fly k - fly k+1                           (flies fall back to spreads, then outrights)
        spread3  3-month spread - 3 x the next 1-month spread     (M_k - M_k+3) - 3 (M_k+1 - M_k+2)
        spreads  1-month spreads: s k - 2 s k+1 + s k+2

No cut-off month has to be given for where flies stop being quoted.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable

from core.curve_calendar import (
    DFLY, FAMILY_WEIGHTS, FLY, OUTRIGHT, PRODUCTS, SPREAD, CurveContract, generic_contracts, month_token,
)

MAX_MONTHS = 36
FRESH_SECONDS = 86_400.0  # a direct price this young is used as it stands
STALE_LIMIT_SECONDS = 7 * 86_400.0  # an older one is only a last resort, when nothing can be built from newer pieces

SOURCE_QUOTE, SOURCE_FAIR, SOURCE_DERIVED = "quote", "fair value", "derived"

DFLY_FLIES, DFLY_SPREAD3, DFLY_SPREADS = "flies", "spread3", "spreads"
DFLY_METHODS = {
    DFLY_FLIES: "Fly - next fly",
    DFLY_SPREAD3: "3-month spread - 3 x next 1-month spread",
    DFLY_SPREADS: "1-month spreads (1, -2, 1)",
}
DEFAULT_DFLY_METHOD = DFLY_FLIES


def min_months(family: str) -> int:
    """Fewest contracts needed to draw at least one point of this family."""
    return len(FAMILY_WEIGHTS[family])


# ----------------------------------------------------------------------
# Quotes
# ----------------------------------------------------------------------


def vwap(bid: float | None, ask: float | None, bid_size: float | None, ask_size: float | None) -> float | None:
    """Size-weighted mid of the best bid and ask; plain mid without sizes; None without a two-sided quote."""
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(bid) and math.isfinite(ask)):
        return None
    try:
        bid_size, ask_size = max(float(bid_size or 0.0), 0.0), max(float(ask_size or 0.0), 0.0)
    except (TypeError, ValueError):
        bid_size = ask_size = 0.0
    total = bid_size + ask_size
    if total <= 0:
        return (bid + ask) / 2.0
    return (bid * ask_size + ask * bid_size) / total


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    ts: float  # exchange time, unix seconds

    @property
    def price(self) -> float | None:
        return vwap(self.bid, self.ask, self.bid_size, self.ask_size)


# symbol -> (price, unix seconds, source)
PriceBook = dict[str, tuple[float, float, str]]


def build_price_book(
    quotes: dict[str, Quote], fair_values: dict[str, tuple[float, float]] | None = None,
) -> PriceBook:
    """Per symbol, the newer of its quote VWAP and its fair value (the quote wins a tie)."""
    book: PriceBook = {}
    for symbol, (price, ts) in (fair_values or {}).items():
        book[symbol] = (float(price), float(ts), SOURCE_FAIR)
    for symbol, quote in quotes.items():
        price = quote.price
        if price is not None and (symbol not in book or quote.ts >= book[symbol][1]):
            book[symbol] = (price, quote.ts, SOURCE_QUOTE)
    return book


# ----------------------------------------------------------------------
# Symbols
# ----------------------------------------------------------------------


def _chain(contracts: list[CurveContract], start: int, length: int) -> str:
    legs = contracts[start:start + length]
    return legs[0].symbol + "".join(f"-{month_token(c.month, c.year)}" for c in legs[1:])


def _spread3_symbol(contracts: list[CurveContract], start: int) -> str:
    far = contracts[start + 3]
    return f"{contracts[start].symbol}-{month_token(far.month, far.year)}"


def _symbols(contracts: list[CurveContract], family: str, method: str) -> list[str]:
    n = len(contracts)
    if family == OUTRIGHT:
        return [c.symbol for c in contracts]
    if family == SPREAD:
        return [_chain(contracts, i, 2) for i in range(n - 1)]
    if family == FLY:
        return [_chain(contracts, i, 3) for i in range(n - 2)]
    if method == DFLY_SPREAD3:
        return [_spread3_symbol(contracts, i) for i in range(n - 3)]
    return []  # a dfly has no quote of its own


def symbol_tiers(product: str, as_of: date, n: int, family: str, method: str = DEFAULT_DFLY_METHOD) -> list[list[str]]:
    """Symbols worth quoting for this curve in tiers: what the curve uses first, then its fallbacks."""
    contracts = generic_contracts(product, as_of, n)
    order = {
        OUTRIGHT: (OUTRIGHT,),
        SPREAD: (SPREAD, OUTRIGHT),
        FLY: (FLY, SPREAD, OUTRIGHT),
        DFLY: (FLY, SPREAD, OUTRIGHT) if method == DFLY_FLIES else (SPREAD, OUTRIGHT),
    }[family]
    tiers = [_symbols(contracts, fam, method) for fam in order]
    if family == DFLY and method == DFLY_SPREAD3:
        tiers.insert(0, _symbols(contracts, DFLY, DFLY_SPREAD3))
    return [tier for tier in tiers if tier]


def symbols_needed(product: str, as_of: date, n: int, family: str, method: str = DEFAULT_DFLY_METHOD) -> list[str]:
    """Every symbol worth quoting for this curve, most useful first."""
    return list(dict.fromkeys(s for tier in symbol_tiers(product, as_of, n, family, method) for s in tier))


# ----------------------------------------------------------------------
# Curve
# ----------------------------------------------------------------------


@dataclass
class CurvePoint:
    position: int  # 1 = front
    label: str  # "Dec26", "Dec26/Jan27", ...
    axis: str  # compact x-axis text
    symbol: str  # the structure's own symbol ("" for a dfly)
    legs: tuple[str, ...]  # consecutive outright symbols
    value: float | None = None
    source: str | None = None  # quote | fair value | derived | None (empty)
    age_seconds: float | None = None  # age of the oldest price the value rests on
    how: str = ""  # human description of how a derived value was made


def _label(contracts: list[CurveContract], start: int, length: int) -> tuple[str, str]:
    legs = contracts[start:start + length]
    label = "/".join(c.label for c in legs)
    return label, (legs[0].label if length == 1 else f"{legs[0].label}→{legs[-1].label}")


def build_curve(
    product: str, as_of: date, n: int, family: str, book: PriceBook, now: float,
    method: str = DEFAULT_DFLY_METHOD, fresh_age: float = FRESH_SECONDS, stale_age: float = STALE_LIMIT_SECONDS,
) -> list[CurvePoint]:
    """The curve's points in contract order, each with its value, source and (if derived) how."""
    contracts = generic_contracts(product, as_of, n)
    memo: dict[tuple[str, int], tuple[float, str, float, str] | None] = {}

    def pick(symbol: str, derive=None):
        """Own price if fresh, else what `derive()` builds, else an old own price, else None."""
        entry = book.get(symbol)
        age = None if entry is None else now - entry[1]
        if entry is not None and age <= fresh_age:
            return (entry[0], entry[2], entry[1], "")
        built = derive() if derive is not None else None
        if built is not None:
            return built
        if entry is not None and age <= stale_age:
            return (entry[0], entry[2], entry[1], "old price")
        return None

    def _sum(parts, how):
        """Weighted sum of already-built pieces; None if any is missing. The oldest piece sets the age."""
        if any(piece is None for _, piece in parts):
            return None
        total = sum(weight * piece[0] for weight, piece in parts)
        return total, SOURCE_DERIVED, min(piece[2] for _, piece in parts), how

    def outright(i: int):
        key = (OUTRIGHT, i)
        if key not in memo:
            memo[key] = pick(contracts[i].symbol)
        return memo[key]

    def spread(i: int):
        key = (SPREAD, i)
        if key not in memo:
            memo[key] = pick(
                _chain(contracts, i, 2),
                lambda: _sum([(1, outright(i)), (-1, outright(i + 1))], "outright - next outright"),
            )
        return memo[key]

    def fly(i: int):
        key = (FLY, i)
        if key not in memo:
            memo[key] = pick(
                _chain(contracts, i, 3),
                lambda: _sum([(1, spread(i)), (-1, spread(i + 1))], "spread - next spread"),
            )
        return memo[key]

    def spread3(i: int):
        key = ("spread3", i)
        if key not in memo:
            memo[key] = pick(
                _spread3_symbol(contracts, i),
                lambda: _sum([(1, outright(i)), (-1, outright(i + 3))], "outright - outright 3 months on")
                or _sum([(1, spread(i)), (1, spread(i + 1)), (1, spread(i + 2))], "sum of three 1-month spreads"),
            )
        return memo[key]

    def dfly(i: int):
        if method == DFLY_SPREAD3:
            return _sum([(1, spread3(i)), (-3, spread(i + 1))], "3-month spread - 3 x next 1-month spread")
        if method == DFLY_SPREADS:
            return _sum([(1, spread(i)), (-2, spread(i + 1)), (1, spread(i + 2))], "spread - 2 x next + next-next")
        return _sum([(1, fly(i)), (-1, fly(i + 1))], "fly - next fly")

    build = {OUTRIGHT: outright, SPREAD: spread, FLY: fly, DFLY: dfly}[family]
    length = len(FAMILY_WEIGHTS[family])
    points = []
    for i in range(n - length + 1):
        label, axis = _label(contracts, i, length)
        symbol = "" if family == DFLY else _chain(contracts, i, length)  # OUTRIGHT: length 1 = the contract
        point = CurvePoint(
            i + 1, label, axis, symbol, tuple(c.symbol for c in contracts[i:i + length]),
        )
        result = build(i)
        if result is not None:
            value, source, ts, how = result
            point.value, point.source, point.age_seconds, point.how = value, source, max(now - ts, 0.0), how
        points.append(point)
    return points


def settlement_curve(
    product: str, as_of: date, n: int, family: str, settlements: dict[str, float],
) -> list[float | None]:
    """Previous-settlement value of each point, built from outright settlements (None if a leg is missing)."""
    contracts = generic_contracts(product, as_of, n)
    weights = FAMILY_WEIGHTS[family]
    values = []
    for i in range(n - len(weights) + 1):
        legs = [c.symbol for c in contracts[i:i + len(weights)]]
        if any(leg not in settlements for leg in legs):
            values.append(None)
        else:
            values.append(float(sum(w * settlements[leg] for w, leg in zip(weights, legs))))
    return values


# ----------------------------------------------------------------------
# Service (state shared with the page)
# ----------------------------------------------------------------------


@dataclass
class CompareRow:
    point: CurvePoint
    prev: float | None

    @property
    def change(self) -> float | None:
        if self.point.value is None or self.prev is None:
            return None
        return self.point.value - self.prev


@dataclass
class CompareView:
    product: str
    family: str
    method: str
    months: int
    rows: list[CompareRow] = field(default_factory=list)
    quote_error: str | None = None
    settlement_date: date | None = None
    settlement_message: str | None = None
    symbols_requested: int = 0
    symbols_quoted: int = 0
    symbols_priced: int = 0  # symbols with a quote or a fair value
    fair_error: str | None = None
    last_quote_at: float | None = None

    def count(self, source: str | None) -> int:
        return sum(1 for r in self.rows if r.point.source == source)

    def count_older_than(self, seconds: float) -> int:
        return sum(1 for r in self.rows if r.point.age_seconds is not None and r.point.age_seconds > seconds)


class CurveCompare:
    """Ties the quote store, the settlements and the fair-value fallback to the page.

    `view()` is cheap and never calls the API itself: it registers what the page needs with the
    quote store (which polls in the background within the rate limit), then builds the curve from
    whatever the store holds now. Previous settlements are fetched once per UTC day in a thread."""

    def __init__(
        self, quotes, settlements, api, fallback_prices: Callable[[], dict] | None = None,
        clock: Callable[[], float] = time.time, fresh_age: float = FRESH_SECONDS,
        stale_age: float = STALE_LIMIT_SECONDS, background: bool = True, fair_values=None,
    ):
        self.quotes = quotes
        self.settlements = settlements
        self._api = api
        self._fallback = fallback_prices
        self._clock = clock
        self._fresh_age, self._stale_age = fresh_age, stale_age
        self._fair_store = fair_values  # core.quote_store.FairValueStore: /fairvalue/ for the whole ladder
        self._background = background
        self._fetch_lock = threading.Lock()
        self._fetching = False
        self._retry_after = 0.0
        self._settlement_message: str | None = None

    # -- settlements ----------------------------------------------------

    def _all_outrights(self, as_of: date) -> list[str]:
        return sorted({c.symbol for p in PRODUCTS for c in generic_contracts(p, as_of, MAX_MONTHS)})

    def fetch_settlements(self, as_of: date) -> None:
        """Fetch the previous settlement of every outright out to MAX_MONTHS for both products."""
        try:
            symbols = self._all_outrights(as_of)
            report = self.settlements.fetch(self._api, symbols, as_of, as_of)
            self._settlement_message = (
                f"{len(report['missing'])} outright(s) have no settlement" if report["missing"] else None
            )
        except Exception as exc:  # noqa: BLE001 - a failed fetch must not break the page
            self._settlement_message = f"Previous settlements unavailable: {exc}"
            self._retry_after = self._clock() + 300
        finally:
            self._fetching = False

    def _ensure_settlements(self, as_of: date) -> None:
        if self.settlements.fetched_on == as_of.isoformat() or self._fetching or self._clock() < self._retry_after:
            return
        with self._fetch_lock:
            if self._fetching:
                return
            self._fetching = True
        if self._background:
            threading.Thread(target=self.fetch_settlements, args=(as_of,), name="compare-settlements", daemon=True).start()
        else:
            self.fetch_settlements(as_of)

    # -- view -----------------------------------------------------------

    def view(self, product: str, n: int, family: str, method: str = DEFAULT_DFLY_METHOD) -> CompareView:
        now = self._clock()
        as_of = datetime.fromtimestamp(now, tz=timezone.utc).date()
        tiers = symbol_tiers(product, as_of, n, family, method)
        needed = list(dict.fromkeys(s for tier in tiers for s in tier))
        self.quotes.request((product, n, family, method), needed, primary=len(tiers[0]) if tiers else 0)
        self._ensure_settlements(as_of)

        quotes = self.quotes.quotes(needed)
        wanted = set(needed)
        fair: dict[str, tuple[float, float]] = {}
        if self._fair_store is not None:
            self._fair_store.request(needed)
            fair.update(self._fair_store.prices(needed))
        if self._fallback is not None:  # the 10 s live poll covers the first 15 months and is fresher
            try:
                for symbol, value in self._fallback().items():
                    if symbol in wanted and (symbol not in fair or value[1] >= fair[symbol][1]):
                        fair[symbol] = value
            except Exception:  # noqa: BLE001
                pass
        book = build_price_book(quotes, fair)
        points = build_curve(product, as_of, n, family, book, now, method, self._fresh_age, self._stale_age)
        prev = settlement_curve(product, as_of, n, family, self.settlements.prices())
        view = CompareView(product, family, method, n, [CompareRow(p, v) for p, v in zip(points, prev)])
        view.quote_error = self.quotes.last_error
        view.settlement_date = self.settlements.settlement_date()
        view.settlement_message = self._settlement_message
        view.symbols_requested = len(needed)
        view.symbols_quoted = sum(1 for s in needed if s in quotes and quotes[s].price is not None)
        view.symbols_priced = sum(1 for s in needed if s in book)
        view.last_quote_at = max((q.ts for q in quotes.values()), default=None)
        view.fair_error = getattr(self._fair_store, "last_error", None)
        return view

