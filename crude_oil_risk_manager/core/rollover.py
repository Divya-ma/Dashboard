"""Strategy Rollover page: how a generic moved before the front contract expired and after it became the front.

A GENERIC is a position on the curve: generic 2 is the second month, generic 1 the front. When the front contract expires every
contract moves up one place, so the contract that was generic n becomes generic n-1. This page follows ONE fixed contract (or
structure of consecutive contracts: a spread, fly or dfly) through those moves, from `before_days` trading days before the front
expiry E, through E, and on until the next expiry. Several expiries (one line each, the most recent first) are drawn on the same axis
and averaged. The prices are daily settlements (core.oi_data.OiStore).

THE LINE of the expiry of contract c: the structure whose front leg is the contract n-1 months after c (generic n just before E), drawn
against trading days from E (0 on the day c expires, negative before). It runs to the earlier of the structure's own expiry and the
expiry `expiries` months after c, so with n = 3 and 2 expiries a line shows the structure as generic 3, then as generic 2 after E, then as
generic 1 after the next expiry, up to the one after. Expiries of finished contracts are the last day they settled; live ones use the
rule-based date (core.curve_calendar).

ZERO BASE. Every line has the price it had on E (day 0) subtracted, so all lines meet at 0 on the expiry day and the chart shows only
what the structure did around it. A line whose E has not come yet has no zero point and is left out.

AVERAGE. The mean of the finished lines at each day, where at least MIN_LINES lines have a value.

CL-BZ. Its two legs expire on different days, so the box's front is not one contract (core.box): from the BZ expiry to the CL expiry the
front box pairs the CL front with the NEXT BZ month (CL Dec - BZ Jan); from the CL expiry to the BZ expiry both legs are the same month
(CL Jan - BZ Jan); after the BZ expiry they differ again (CL Jan - BZ Feb). The CL-BZ line of a calendar month starts on its first
trading day, so it opens on the different-months front, passes the CL expiry (day 0), the same-month front, and the BZ expiry, and runs
to the CL expiry `expiries` months after the first. It is one spliced series of the front box. The different-month box is built from
stored prices: CL(c) - BZ(b) = [CL(c) - CL(b)] + [CL(b) - BZ(b)], the CL spread plus the same-month box.
"""

from dataclasses import dataclass, field, replace
import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd

from core.box import OUTBOX
from core.curve_calendar import EXTRA_PRODUCTS, FAMILY_WEIGHTS, OUTRIGHT, PRODUCTS, api_symbol, contract_expiry, front_month, product_spec
from core.oi_data import EXPIRED_AFTER_DAYS, FIRST_CONTRACT_YEAR
from core.seasonality import GAP_FILL, _label, pivot_frame, structure_frame, structure_symbols, trading_days_to

logger = logging.getLogger(__name__)

DEFAULT_GENERIC, MAX_GENERIC = 2, 12
DEFAULT_EXPIRIES, MAX_EXPIRIES = 1, 6
DEFAULT_COMPARISONS, MAX_COMPARISONS = 6, 24
DEFAULT_BEFORE, MAX_BEFORE = 20, 120
MIN_LINES = 2  # finished lines needed for an average
CL = "CL"


@dataclass(frozen=True)
class RolloverOptions:
    generic: int = DEFAULT_GENERIC  # the generic followed (1 = front)
    expiries: int = DEFAULT_EXPIRIES  # front expiries the line runs through (CL-BZ: months)
    comparisons: int = DEFAULT_COMPARISONS  # lines drawn
    before_days: int = DEFAULT_BEFORE  # trading days shown before the first expiry (generic mode)
    zero_base: bool = False
    family: str = OUTRIGHT

    def key(self) -> tuple:
        return (self.generic, self.expiries, self.comparisons, self.before_days, self.zero_base, self.family)


@dataclass(frozen=True)
class Mark:
    x: int
    y: float
    text: str


@dataclass(frozen=True)
class Guide:
    """A vertical expiry line shared by every line of the chart."""
    x: int
    text: str


@dataclass
class RolloverLine:
    label: str
    anchor: date  # the expiry at x = 0
    frame: pd.DataFrame  # index: trading days from the anchor; columns price, date, pair (what is priced that day)
    marks: list[Mark] = field(default_factory=list)
    live: bool = False  # the line is still being drawn (a later expiry has not come)
    zero: float | None = None  # what was subtracted on the zero base
    # The expiry days the line passes, as trading days from the anchor, in order. `names` labels each one for the diamond marks (empty:
    # not an expiry), `kinds` for the chart's vertical line, `vlines` says which of them get a vertical line, `pivot` is the anchor (x = 0).
    bounds: list[int] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    vlines: list[int] = field(default_factory=list)
    pivot: int = 0
    dates: list[date] = field(default_factory=list)  # the calendar dates of `bounds`

    @property
    def prices(self) -> pd.Series:
        return self.frame["price"]


@dataclass
class RolloverResult:
    product: str
    options: RolloverOptions
    box: bool
    lines: list[RolloverLine]  # most recent first
    average: pd.Series
    notes: list[str] = field(default_factory=list)
    guides: list[Guide] = field(default_factory=list)  # the vertical expiry lines


def _index(month: int, year: int) -> int:
    return year * 12 + month - 1


def _month_year(index: int) -> tuple[int, int]:
    return index % 12 + 1, index // 12


class Expiries:
    """Expiry date of a product's contract by delivery-month index: the last settlement day of a finished contract, else the rule."""

    def __init__(self, product: str, observed: dict, today: date):
        self._product, self._observed, self._today = product, observed, today

    def of(self, index: int) -> date:
        month, year = _month_year(index)
        rule = contract_expiry(self._product, month, year)
        if rule < self._today - timedelta(days=EXPIRED_AFTER_DAYS):
            seen = self._observed.get(api_symbol(self._product, month, year))
            if seen is not None:
                return pd.Timestamp(seen).date()
        return rule


def _near(prices: pd.Series, x: int) -> float | None:
    """The price on day x, or the latest one within GAP_FILL trading days before it."""
    near = prices[(prices.index <= x) & (prices.index >= x - GAP_FILL)]
    return None if near.empty else float(near.sort_index().iloc[-1])


def _zero_point(prices: pd.Series) -> float | None:
    return _near(prices, 0)


def _usable(line: RolloverLine, zero_base: bool, today: date) -> bool:
    """Only a line that has crossed every expiry line but the last (which has passed or is still to come) is drawn; a zero-based one also
    needs its day 0."""
    if len(line.dates) > 1 and line.dates[-2] > today:
        return False
    return not zero_base or (line.anchor <= today and line.frame.index.max() >= 0 and _zero_point(line.prices) is not None)


def _warp(line: RolloverLine, target: np.ndarray) -> None:
    """Move the line's expiry days to `target` (the same for every line): the days between two expiries are stretched or squeezed."""
    bounds, frame = np.array(line.bounds), line.frame
    x, last = frame.index.to_numpy(), len(bounds) - 1
    segment = np.searchsorted(bounds, x, side="right") - 1
    new = np.where(segment < 0, target[0] + x - bounds[0], target[last] + x - bounds[last]).astype(float)
    inside = (segment >= 0) & (segment < last)
    if inside.any():
        s = segment[inside]
        new[inside] = target[s] + (x[inside] - bounds[s]) * np.diff(target)[s] / np.diff(bounds)[s]
    frame = frame.copy()
    frame.index = np.rint(new).astype(int)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    line.frame = frame.reindex(range(int(frame.index.min()), int(frame.index.max()) + 1)).ffill()
    line.bounds = [int(t) for t in target]


def align(lines: list[RolloverLine]) -> tuple[list[Guide], bool]:
    """Put every line's expiries on the same days. Where the gap between two expiries differs (a holiday), the longest gap is the usual
    one and the others are stretched to it. Returns the vertical lines and whether any line was moved."""
    if not lines or not lines[0].bounds:
        return [], False
    group = [line for line in lines if len(line.bounds) == len(lines[0].bounds) and line.pivot == lines[0].pivot]
    count, pivot = len(lines[0].bounds), lines[0].pivot
    longest = np.max([np.diff(line.bounds) for line in group], axis=0) if count > 1 else np.array([], dtype=int)
    target = np.zeros(count, dtype=int)
    for k in range(pivot + 1, count):
        target[k] = target[k - 1] + longest[k - 1]
    for k in range(pivot - 1, -1, -1):
        target[k] = target[k + 1] - longest[k]
    moved = any(line.bounds != target.tolist() for line in group)
    for line in group:
        _warp(line, target)
    return [Guide(int(target[k]), lines[0].kinds[k]) for k in lines[0].vlines], moved


def _finish(line: RolloverLine, zero_base: bool) -> RolloverLine | None:
    """Zero base: subtract the price on the expiry day (day 0); after each later expiry line the price restarts from that expiry's
    price, so the line is 0 on every vertical line. Then place the diamond marks. None: nothing is left to draw."""
    frame = line.frame
    if zero_base:
        later = np.array(line.bounds[line.pivot + 1:-1], dtype=int)  # the expiries between day 0 and the end
        bases = np.array([np.nan if (b := _near(frame["price"], int(x))) is None else b for x in [line.bounds[line.pivot], *later]])
        if np.isnan(bases[0]):
            return None
        piece = np.searchsorted(later, frame.index.to_numpy(), side="right")
        frame = frame.assign(price=frame["price"].to_numpy() - bases[piece]).dropna(subset=["price"])
        line.zero = float(bases[0])
    if frame.empty:
        return None
    line.frame, line.marks = frame, []
    for k, x in enumerate(line.bounds):
        if line.names[k] and x in frame.index:
            line.marks.append(Mark(x, float(frame["price"].at[x]), line.names[k]))
    return line


# ----------------------------------------------------------------------
# One generic through its expiries
# ----------------------------------------------------------------------


def generic_line(close: pd.DataFrame, product: str, base: int, expiries: Expiries, options: RolloverOptions, today: date) -> RolloverLine | None:
    """The line of the expiry of the contract at delivery-month index `base`, or None when it has no settlements."""
    front = base + options.generic - 1
    month, year = _month_year(front)
    symbols = structure_symbols(product, options.family, month, year)
    built = structure_frame(close, pd.DataFrame(), symbols, FAMILY_WEIGHTS[options.family])
    if built is None:
        return None
    anchor, calendar = expiries.of(base), product_spec(product).calendar
    end = min(expiries.of(base + options.expiries), expiries.of(front))
    dates = [d for d in (expiries.of(base + k) for k in range(options.expiries + 1)) if d <= end]
    x = -trading_days_to(built.index, anchor, calendar)
    keep = (x >= -options.before_days) & (built.index <= pd.Timestamp(end))
    if not keep.any():
        return None
    frame = pd.DataFrame({"price": built["price"].to_numpy()[keep], "date": built.index[keep], "pair": ""}, index=x[keep])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    label = _label(month, year, len(symbols))
    bounds = [int(-v) for v in trading_days_to(pd.DatetimeIndex([pd.Timestamp(d) for d in dates]), anchor, calendar)]
    return RolloverLine(
        label, anchor, frame, live=end > today, bounds=bounds, dates=dates,
        names=["", *(f"{month_label(base + k)} expires" for k in range(1, len(dates)))],
        kinds=["Front expiry", *(f"Expiry {k + 1}" for k in range(1, len(dates)))],
        vlines=list(range(max(len(dates) - 1, 1))),  # the expiries the line passes; the last date is where it ends
    )


def month_label(index: int) -> str:
    month, year = _month_year(index)
    return date(year, month, 1).strftime("%b%y")


# ----------------------------------------------------------------------
# The CL-BZ front box through both legs' expiries
# ----------------------------------------------------------------------


def box_line(cl: pd.DataFrame, box: pd.DataFrame, base: int, cl_expiries: Expiries, bz_expiries: Expiries, options: RolloverOptions, today: date) -> RolloverLine | None:
    """The line of the CL contract at index `base`, whose expiry (day 0) is in the month the line starts."""
    anchor = cl_expiries.of(base)
    start = date(anchor.year, anchor.month, 1)
    end = cl_expiries.of(base + options.expiries)
    days = cl.index.intersection(box.index)
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    xs = dict(zip(days, -trading_days_to(days, anchor, PRODUCTS[OUTBOX].calendar))) if len(days) else {}
    rows = []
    for day in days:
        when = day.date()
        cl_front = next(i for i in range(base, base + 40) if cl_expiries.of(i) >= when)
        bz_front = next(i for i in range(cl_front, cl_front + 40) if bz_expiries.of(i) >= when)
        cl_symbol, bz_symbol = (api_symbol(CL, *_month_year(i)) for i in (cl_front, bz_front))
        box_symbol = api_symbol(OUTBOX, *_month_year(bz_front))
        if bz_front == cl_front:
            value = box.at[day, box_symbol] if box_symbol in box.columns else np.nan
        elif bz_front == cl_front + 1 and {cl_symbol, bz_symbol} <= set(cl.columns) and box_symbol in box.columns:
            value = cl.at[day, cl_symbol] - cl.at[day, bz_symbol] + box.at[day, box_symbol]
        else:
            continue
        if pd.notna(value):
            rows.append((int(xs[day]), float(value), day,
                         f"CL {month_label(cl_front)} − BZ {month_label(bz_front)}"))
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=["x", "price", "date", "pair"]).drop_duplicates("x", keep="last").set_index("x").sort_index()
    calendar = PRODUCTS[OUTBOX].calendar
    events = [(cl_expiries.of(base + k), f"CL {month_label(base + k)} expires", "CL expiry") for k in range(options.expiries + 1)]
    events += [(w, f"BZ {month_label(i)} expires", "BRN expiry") for i in range(base, base + options.expiries + 3)
               if anchor < (w := bz_expiries.of(i)) <= end]
    events.sort(key=lambda event: event[0])
    when = pd.DatetimeIndex([pd.Timestamp(start), *(pd.Timestamp(e[0]) for e in events)])
    bounds = [int(-v) for v in trading_days_to(when, anchor, calendar)]
    return RolloverLine(
        anchor.strftime("%b%y"), anchor, frame, live=end > today, bounds=bounds, dates=[e[0] for e in events], names=["", "", *(e[1] for e in events[1:])],
        kinds=["", *(e[2] for e in events)], vlines=list(range(1, len(bounds))), pivot=1,
    )


# ----------------------------------------------------------------------
# Average and service
# ----------------------------------------------------------------------


def average_of(lines: list[RolloverLine]) -> pd.Series:
    """Mean of the finished lines at each trading day where at least MIN_LINES have a value."""
    finished = [line.prices for line in lines if not line.live]
    if len(finished) < MIN_LINES:
        return pd.Series(dtype=float)
    matrix = pd.concat(finished, axis=1).sort_index()
    return matrix.mean(axis=1).where(matrix.notna().sum(axis=1) >= MIN_LINES).dropna()


class RolloverService:
    """Settlement stores -> the Strategy Rollover chart. Results are cached until the stored settlements change.

    `store` holds CL, Brent, Gasoil and the CL-BZ box; `extra` (optional) the settlements-only products BZ and WBS, which it downloads
    in the background through `api` the first time one of them is shown (`history_days` is a callable returning the depth wanted)."""

    RETRY_SECONDS = 600

    def __init__(
        self, store, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc), extra=None, api=None,
        history_days: Callable[[], int] | None = None, monotonic: Callable[[], float] = time.monotonic, background: bool = True,
    ):
        self._store = store
        self._extra, self._api, self._history_days = extra, api, history_days
        self._clock, self._monotonic, self._background = clock, monotonic, background
        self._wide: dict[str, tuple[str, pd.DataFrame]] = {}
        self._cache: dict[tuple, RolloverResult] = {}
        self._lock = threading.Lock()
        self._retry_after = 0.0

    def _source(self, product: str):
        return self._extra if product in EXTRA_PRODUCTS and self._extra is not None else self._store

    def _close(self, product: str) -> pd.DataFrame:
        source = self._source(product)
        cached = self._wide.get(product)
        if cached is None or cached[0] != source.version:
            self._wide[product] = (source.version, pivot_frame(source.frame(product))[0])
            self._cache.clear()
        return self._wide[product][1]

    def _expiries(self, product: str, today: date) -> Expiries:
        return Expiries(product, self._source(product).contract_expiry_dates(product), today)

    def _ensure_extra(self, today: date) -> None:
        """Start the BZ / WBS download when it is due (never raises)."""
        extra = self._extra
        if extra is None or self._api is None or extra.progress.running or self._monotonic() < self._retry_after:
            return
        days = self._history_days() if self._history_days else 1500
        if extra.updated_on == today.isoformat() and (extra.loaded_history_days or 0) >= days and extra.has_every_product():
            return
        with self._lock:
            if extra.progress.running:
                return
            extra.progress.running, extra.progress.error = True, None
        if self._background:
            threading.Thread(target=self._download, args=(days, today), name="rollover-update", daemon=True).start()
        else:
            self._download(days, today)

    def _download(self, days: int, today: date) -> None:
        progress = self._extra.progress
        try:
            self._extra.update(self._api, days, today)
        except Exception as exc:  # noqa: BLE001 - shown on the page; partial progress is saved
            logger.exception("Rollover settlements update failed")
            progress.error = f"{type(exc).__name__}: {exc}"
            self._retry_after = self._monotonic() + self.RETRY_SECONDS
        finally:
            progress.running = False

    def view(self, product: str, options: RolloverOptions) -> RolloverResult:
        today = self._clock().date()
        if product in EXTRA_PRODUCTS:
            self._ensure_extra(today)
        source = self._source(product)
        key = (source.version, today, product, options.key())
        result = self._cache.get(key)
        if result is None:
            if len(self._cache) > 200:
                self._cache.clear()
            result = self._cache[key] = self._build(product, options, today)
            if not result.lines and source is self._extra:  # a download in progress is shown, not cached
                self._cache.pop(key, None)
        return result

    def _build(self, product: str, options: RolloverOptions, today: date) -> RolloverResult:
        box = product_spec(product).is_box
        source = self._source(product)
        if not source.has_data() or not self._close(product).size:
            where = "it starts when this page is shown" if source is self._extra else "OI · Volume · ATR page"
            progress = source.progress
            message = f"The settlement history is still downloading ({where}); the chart fills in when it arrives."
            if progress.error:
                message = f"The settlement download failed: {progress.error}"
            elif source is self._extra and progress.message:
                message += f" {progress.message}"
            return RolloverResult(product, options, box, [], pd.Series(dtype=float), [message])
        notes = []
        if box:
            lines, guides, moved = self._box_lines(options, today)
        else:
            reach = max(options.generic - 1, 0)  # generic n meets n-1 expiries before it is the front
            if options.expiries > reach:
                notes.append(f"Generic {options.generic} passes at most {reach} expiries before it is the front month, so {reach} are shown.")
            used = replace(options, expiries=min(options.expiries, reach))
            close, expiries = self._close(product), self._expiries(product, today)
            lines, guides, moved = self._lines(lambda base: generic_line(close, product, base, expiries, used, today), product, options, today)
        if len(lines) < options.comparisons:
            notes.append(f"Only {len(lines)} of {options.comparisons} expiries have settlements for this selection (the stored history is limited; "
                         "raise 'OI history' in Settings for more).")
        if options.zero_base:
            notes.append("Zero base: every line has its price on day 0 (the expiry) subtracted, and starts again from 0 on each later expiry "
                         "line. An expiry still to come has no day 0 and is not drawn.")
        if moved:
            notes.append("Gaps between expiries differ from line to line (holidays): each gap is stretched to the longest one so every expiry "
                         "falls on one vertical line.")
        return RolloverResult(product, options, box, lines, average_of(lines), notes, guides)

    def _lines(self, build: Callable[[int], RolloverLine | None], product: str, options: RolloverOptions, today: date):
        """Lines for the expiries from the current front's back, until `comparisons` have data: (lines, vertical lines, any stretched)."""
        base, lines = _index(*front_month(product, today)), []
        floor = _index(1, FIRST_CONTRACT_YEAR)
        while len(lines) < options.comparisons and base >= floor:
            line = build(base)
            if line is not None and _usable(line, options.zero_base, today):
                lines.append(line)
            base -= 1
        guides, moved = align(lines)
        return [line for line in (_finish(line, options.zero_base) for line in lines) if line is not None], guides, moved

    def _box_lines(self, options: RolloverOptions, today: date):
        cl, box = self._close(CL), self._close(OUTBOX)
        if cl.empty or box.empty:
            return [], [], False
        cl_expiries, bz_expiries = self._expiries(CL, today), self._expiries(OUTBOX, today)
        return self._lines(lambda base: box_line(cl, box, base, cl_expiries, bz_expiries, options, today), CL, options, today)
