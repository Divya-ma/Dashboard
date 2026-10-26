"""Seasonality page: where a structure stands today against the same structure in earlier years.

For one product, one curve type (outright / spread / fly / dfly) and the first N contracts from the front, each
structure (say the Apr26 dfly) is compared with the same structure of earlier delivery years (Apr25, Apr24 ...) at the
SAME NUMBER OF TRADING DAYS BEFORE EXPIRY. The prices are daily settlements (core.oi_data.OiStore); a spread, fly or dfly
is the weighted sum of its outrights' settlements on the days every outright settled, so it has a value on every
trading day.

DAYS BEFORE EXPIRY. Trading days (the product's exchange calendar) from the settlement date to the expiry of the
structure's FRONT leg, the first leg to expire: 0 on the expiry day, 1 the day before. A finished contract's expiry is
the last day its front leg settled (exact); a live contract uses the rule-based date (core.curve_calendar).

THE BAND. At each day-before-expiry, over the earlier years that are not excluded: mean +/- z * standard deviation, or
the robust median +/- z * 1.4826 * MAD. The spread never goes below one tick. A band needs at least MIN_YEARS years.

FLAG. The live price (else the last settlement) is `z` spreads away from the centre. A structure is FLAGGED when it has
been outside the band, on the same side, for `run_days` days in a row: the run of settlements ending at the last one,
plus today when the live price is outside on the same side. Outside the band for fewer days is WATCH.

ALSO SEEN. For a flagged or watched structure, every other year is judged against a band made of the remaining years
(so it is not compared with itself): did it leave the band on the same side within SIMILAR_WINDOW trading days of
today's day-before-expiry, and had it come back REVERT_HORIZON trading days later?

REBASE. Optionally every curve is shown as the change since a reference day before expiry. That removes the
oil-price level (Brent at 20 dollars in 2020, 120 in 2022), which otherwise dominates an outright.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from core.curve_calendar import FAMILY_WEIGHTS, PRODUCTS, api_symbol, contract_expiry, generic_contracts, holidays_in
from core.curve_compare import DEFAULT_DFLY_METHOD, MAX_MONTHS, CurvePoint, min_months

MEAN_SD, MEDIAN_MAD = "mean_sd", "median_mad"
BAND_METHODS = {MEAN_SD: "Mean ± SD", MEDIAN_MAD: "Median ± MAD (robust)"}
DEFAULT_YEARS = 7
MAX_YEARS = 12
DEFAULT_DAYS = 250
MAX_DAYS = 2000
DEFAULT_Z = 2.0
DEFAULT_RUN_DAYS = 3
DEFAULT_MONTHS = 15
GAP_FILL = 3  # a missing day is carried forward at most this many trading days when curves are lined up
MIN_YEARS = 2  # earlier years needed for a band
MIN_OTHERS = 3  # earlier years needed for the "also seen" band that leaves one year out
SIMILAR_WINDOW = 10
REVERT_HORIZON = 20
MAD_SCALE = 1.4826
TRADING_DAYS_PER_YEAR = 252

OK, WATCH, FLAGGED, NO_DATA = "OK", "WATCH", "FLAGGED", "NO DATA"
RICH, CHEAP = "rich", "cheap"


@dataclass(frozen=True)
class SeasonOptions:
    years: int = DEFAULT_YEARS  # earlier delivery years compared with the live one
    max_days: int = DEFAULT_DAYS  # trading days before expiry to show
    band: str = MEAN_SD
    z: float = DEFAULT_Z
    run_days: int = DEFAULT_RUN_DAYS
    excluded: tuple[int, ...] = ()  # delivery years left out of the band
    rebase_days: int | None = None  # None: raw price; N: change since N trading days before expiry
    dfly_method: str = DEFAULT_DFLY_METHOD

    def key(self) -> tuple:
        return (self.years, self.max_days, self.band, self.z, self.run_days, tuple(sorted(self.excluded)), self.rebase_days)


# ----------------------------------------------------------------------
# Settlement history of a structure
# ----------------------------------------------------------------------


def pivot_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(close, volume), each wide: dates down, contract symbols across."""
    if frame is None or frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    close = frame.pivot(index="date", columns="contract", values="close").sort_index()
    volume = frame.pivot(index="date", columns="contract", values="volume").sort_index()
    return close, volume


def structure_symbols(product: str, family: str, month: int, year: int) -> list[str]:
    """The consecutive outright symbols of the structure whose front leg delivers in month/year."""
    base = year * 12 + month - 1
    return [api_symbol(product, (base + k) % 12 + 1, (base + k) // 12) for k in range(len(FAMILY_WEIGHTS[family]))]


def trading_days_to(days: pd.DatetimeIndex, expiry: date, calendar: str) -> np.ndarray:
    """Exchange business days from each date to `expiry` (0 on the expiry day, negative after it)."""
    index = pd.DatetimeIndex(days)
    if len(index) == 0:
        return np.array([], dtype=int)
    holidays = np.array(holidays_in(calendar, range(index.min().year, expiry.year + 1)), dtype="datetime64[D]")
    return np.busday_count(index.values.astype("datetime64[D]"), np.datetime64(expiry), holidays=holidays)


def structure_frame(close: pd.DataFrame, volume: pd.DataFrame, symbols: list[str], weights: tuple[int, ...]) -> pd.DataFrame | None:
    """price and volume of the structure on every day all its outrights settled (volume: the thinnest leg's)."""
    if close.empty or any(symbol not in close.columns for symbol in symbols):
        return None
    legs = close[symbols].dropna()
    if legs.empty:
        return None
    price = (legs * np.array(weights, dtype=float)).sum(axis=1)
    thin = volume[symbols].reindex(legs.index).min(axis=1, skipna=False) if not volume.empty else pd.Series(np.nan, index=legs.index)
    return pd.DataFrame({"price": price, "volume": thin})


def aligned(series: pd.Series, top: int) -> pd.Series:
    """The series on every day-before-expiry from 0 to `top`, a short gap carried forward in time."""
    full = series.reindex(range(top, -1, -1)).ffill(limit=GAP_FILL)
    return full.sort_index()


@dataclass
class SeasonCurve:
    year: int  # delivery year of the front leg
    label: str  # "Apr26" (the structure's own label: Apr26/May26/Jun26)
    expiry: date
    frame: pd.DataFrame  # index: trading days before expiry; columns price, volume, date
    live: bool = False
    ref_value: float | None = None  # what was subtracted when the curve is rebased

    @property
    def prices(self) -> pd.Series:
        return self.frame["price"]

    @property
    def last_td(self) -> int:
        return int(self.frame.index.min())


def _label(month: int, year: int, length: int) -> str:
    parts = []
    base = year * 12 + month - 1
    for k in range(length):
        parts.append(date((base + k) // 12, (base + k) % 12 + 1, 1).strftime("%b%y"))
    return "/".join(parts)


def build_curve(
    close: pd.DataFrame, volume: pd.DataFrame, product: str, family: str, month: int, year: int, top: int, today: date,
    live: bool, rebase_days: int | None = None,
) -> SeasonCurve | None:
    """The settlement history of one delivery year's structure against days before expiry (None: no data, or no
    value at the rebase reference)."""
    symbols = structure_symbols(product, family, month, year)
    built = structure_frame(close, volume, symbols, FAMILY_WEIGHTS[family])
    if built is None:
        return None
    if live:
        expiry = contract_expiry(product, month, year)
    else:
        expiry = close[symbols[0]].dropna().index.max().date()
    td = trading_days_to(built.index, expiry, PRODUCTS[product].calendar)
    keep = (td >= 0) & (td <= top)
    if not keep.any():
        return None
    frame = pd.DataFrame(
        {"price": built["price"].to_numpy()[keep], "volume": built["volume"].to_numpy()[keep], "date": built.index[keep]},
        index=td[keep],
    )
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    curve = SeasonCurve(year, _label(month, year, len(symbols)), expiry, frame, live)
    if rebase_days is None:
        return curve
    reference = aligned(frame["price"], max(top, rebase_days)).get(rebase_days)
    if reference is None or not np.isfinite(reference):
        return None
    curve.frame = frame[frame.index <= rebase_days].assign(price=lambda f: f["price"] - reference)
    curve.ref_value = float(reference)
    return curve if not curve.frame.empty else None


# ----------------------------------------------------------------------
# The band and the scores
# ----------------------------------------------------------------------


def band(curves: list[SeasonCurve], top: int, method: str, floor: float) -> pd.DataFrame:
    """centre, spread and number of years at each day before expiry 0..top."""
    grid = range(top + 1)
    if not curves:
        nan = pd.Series(np.nan, index=grid)
        return pd.DataFrame({"center": nan, "spread": nan, "n": pd.Series(0, index=grid)})
    matrix = pd.concat([aligned(c.prices, top) for c in curves], axis=1)
    count = matrix.notna().sum(axis=1)
    if method == MEDIAN_MAD:
        center = matrix.median(axis=1)
        spread = MAD_SCALE * matrix.sub(center, axis=0).abs().median(axis=1)
    else:
        center = matrix.mean(axis=1)
        spread = matrix.std(axis=1, ddof=1)
    enough = count >= MIN_YEARS
    return pd.DataFrame({"center": center.where(enough), "spread": spread.clip(lower=floor).where(enough), "n": count})


def z_scores(curve: SeasonCurve, bands: pd.DataFrame, top: int) -> pd.Series:
    """z of a curve at each of its own days: (price - centre) / spread."""
    price = aligned(curve.prices, top)
    z = (price - bands["center"]) / bands["spread"]
    return z.reindex(curve.frame.index)


def trailing_run(z_chronological: np.ndarray, threshold: float) -> tuple[int, int]:
    """(days, side) of the run of consecutive |z| >= threshold on the same side that ends the series; side +1 rich, -1 cheap."""
    run, side = 0, 0
    for value in z_chronological[::-1]:
        if not np.isfinite(value) or abs(value) < threshold:
            break
        current = 1 if value > 0 else -1
        if side and current != side:
            break
        side = current
        run += 1
    return run, side


@dataclass
class SimilarYear:
    year: int
    peak_z: float
    days_before_expiry: int  # where the deviation peaked
    later_z: float | None  # REVERT_HORIZON trading days later
    reverted: bool | None


@dataclass
class StructureSeason:
    position: int
    product: str
    family: str
    month: int
    year: int
    label: str
    expiry: date
    td_now: int
    top: int
    tick: float
    options: SeasonOptions
    past: list[SeasonCurve]  # every earlier year found, newest first
    used: list[SeasonCurve]  # the ones that make the band
    current: SeasonCurve | None
    bands: pd.DataFrame
    last_price: float | None = None
    last_date: date | None = None
    last_volume: float | None = None
    usual_volume: float | None = None
    run: int = 0  # consecutive settlements outside the band, ending at the last one
    run_side: int = 0
    _similar: dict = field(default_factory=dict)

    @property
    def has_band(self) -> bool:
        return self.td_now <= self.top and bool(np.isfinite(self.bands["center"].get(self.td_now, np.nan)))

    def similar(self, side: int) -> list[SimilarYear]:
        """Earlier years that left their own band on `side` around today's day before expiry (see the module docstring)."""
        if side not in self._similar:
            self._similar[side] = _similar_years(self, side)
        return self._similar[side]


def analyse(
    close: pd.DataFrame, volume: pd.DataFrame, product: str, family: str, month: int, year: int, position: int,
    options: SeasonOptions, today: date,
) -> StructureSeason:
    spec = PRODUCTS[product]
    expiry = contract_expiry(product, month, year)
    td_now = int(trading_days_to(pd.DatetimeIndex([pd.Timestamp(today)]), expiry, spec.calendar)[0])
    top = max(options.max_days, td_now, options.rebase_days or 0)
    past = []
    for back in range(1, options.years + 1):
        curve = build_curve(close, volume, product, family, month, year - back, top, today, False, options.rebase_days)
        if curve is not None:
            past.append(curve)
    used = [c for c in past if c.year not in options.excluded]
    current = build_curve(close, volume, product, family, month, year, top, today, True, options.rebase_days)
    bands = band(used, top, options.band, spec.tick_size)
    season = StructureSeason(
        position, product, family, month, year, _label(month, year, len(FAMILY_WEIGHTS[family])), expiry, td_now, top,
        spec.tick_size, options, past, used, current, bands,
    )
    if current is not None:
        last = current.frame.iloc[0]  # the frame is sorted by days before expiry, so the first row is the latest settlement
        season.last_price, season.last_date, season.last_volume = float(last["price"]), last["date"].date(), last["volume"]
        z = z_scores(current, bands, top).sort_index(ascending=False)
        season.run, season.run_side = trailing_run(z.to_numpy(), options.z)
        td_last = current.last_td
        usual = [aligned(c.frame["volume"], top).get(td_last) for c in used]
        usual = [v for v in usual if v is not None and np.isfinite(v)]
        season.usual_volume = float(np.median(usual)) if usual else None
    return season


def _similar_years(season: StructureSeason, side: int) -> list[SimilarYear]:
    options, found = season.options, []
    if side == 0:
        return found
    for curve in season.used:
        others = [c for c in season.used if c is not curve]
        if len(others) < MIN_OTHERS:
            continue
        bands = band(others, season.top, options.band, season.tick)
        z = z_scores(curve, bands, season.top)
        window = z[(z.index >= season.td_now - SIMILAR_WINDOW) & (z.index <= season.td_now + SIMILAR_WINDOW)].dropna() * side
        if window.empty or window.max() < options.z:
            continue
        peak_td = int(window.idxmax())
        later_td = max(peak_td - REVERT_HORIZON, 0)
        later = aligned(curve.prices, season.top).get(later_td)
        later_z = None
        if later is not None and np.isfinite(later) and np.isfinite(bands["center"].get(later_td, np.nan)):
            later_z = float((later - bands["center"][later_td]) / bands["spread"][later_td])
        reverted = None if later_z is None else bool(later_z * side < 0.5 * window.max())
        found.append(SimilarYear(curve.year, float(window.max() * side), peak_td, later_z, reverted))
    return sorted(found, key=lambda s: -s.year)


# ----------------------------------------------------------------------
# Today's score
# ----------------------------------------------------------------------


@dataclass
class Evaluation:
    status: str
    price: float | None = None  # the price judged: live, else the last settlement
    used_live: bool = False
    center: float | None = None
    spread: float | None = None
    deviation: float | None = None
    ticks: float | None = None
    z: float | None = None
    side: int = 0
    run: int = 0
    rank: int | None = None  # earlier years priced below today
    years: int = 0  # earlier years in the band at this day
    similar: list[SimilarYear] = field(default_factory=list)

    @property
    def direction(self) -> str:
        return "" if self.side == 0 else RICH if self.side > 0 else CHEAP


def evaluate(season: StructureSeason, live: float | None) -> Evaluation:
    """Score the structure. `live` is the live price (None: judge the last settlement)."""
    options = season.options
    if not season.has_band or (live is None and season.last_price is None):
        return Evaluation(NO_DATA)
    if options.rebase_days is not None and season.current is None:  # too far from expiry to share the reference day
        return Evaluation(NO_DATA)
    used_live = live is not None
    if used_live and season.current is not None and season.current.ref_value is not None:
        live = live - season.current.ref_value
    price = live if used_live else season.last_price
    center, spread = float(season.bands["center"][season.td_now]), float(season.bands["spread"][season.td_now])
    if not used_live:  # judged on the last settlement: take the band at that day
        td = season.current.last_td
        center, spread = float(season.bands["center"].get(td, np.nan)), float(season.bands["spread"].get(td, np.nan))
        if not np.isfinite(center):
            return Evaluation(NO_DATA)
    z = (price - center) / spread
    side = 0 if abs(z) < options.z else (1 if z > 0 else -1)
    run = season.run
    if used_live:
        run = (season.run + 1 if side == season.run_side else 1) if side else 0
    status = FLAGGED if side and run >= options.run_days else WATCH if side else OK
    values = [aligned(c.prices, season.top).get(season.td_now) for c in season.used]
    values = [v for v in values if v is not None and np.isfinite(v)]
    return Evaluation(
        status, float(price), used_live, center, spread, float(price - center), float((price - center) / season.tick), float(z),
        side, run, sum(1 for v in values if v < price), len(values), season.similar(side) if side else [],
    )


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------


@dataclass
class SeasonRow:
    season: StructureSeason
    evaluation: Evaluation
    point: CurvePoint | None = None  # the live price and where it came from


@dataclass
class SeasonTable:
    product: str
    family: str
    months: int
    as_of: date
    rows: list[SeasonRow]
    notes: list[str] = field(default_factory=list)


class SeasonalityService:
    """Settlement store + Curve Compare live prices -> the Seasonality table and one structure's detail.

    The seasonal analysis of a structure depends only on the stored settlements and the options, so it is cached until
    the store changes; the live price is applied on top on every call. `compare` is a callable returning the
    CurveCompare service (its quote polling starts while the page is open)."""

    def __init__(self, store, compare: Callable[[], object] | None = None, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self._store = store
        self._compare = compare
        self._clock = clock
        self._wide: dict[str, tuple[str, pd.DataFrame, pd.DataFrame]] = {}
        self._cache: dict[tuple, StructureSeason] = {}

    def _matrices(self, product: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        version = self._store.version
        cached = self._wide.get(product)
        if cached is None or cached[0] != version:
            close, volume = pivot_frame(self._store.frame(product))
            self._wide[product] = (version, close, volume)
            self._cache.clear()
        return self._wide[product][1], self._wide[product][2]

    def _season(self, product: str, family: str, month: int, year: int, position: int, options: SeasonOptions, today: date) -> StructureSeason:
        close, volume = self._matrices(product)
        key = (self._store.version, today, product, family, month, year, options.key())
        season = self._cache.get(key)
        if season is None:
            if len(self._cache) > 1500:
                self._cache.clear()
            season = self._cache[key] = analyse(close, volume, product, family, month, year, position, options, today)
        return season

    def _live(self, product: str, months: int, family: str, options: SeasonOptions) -> dict[int, CurvePoint]:
        service = self._compare() if self._compare is not None else None
        if service is None:
            return {}
        try:
            view = service.view(product, months, family, options.dfly_method)
        except Exception:  # noqa: BLE001 - the table still works on settlements
            return {}
        return {row.point.position: row.point for row in view.rows}

    def notes(self, product: str, options: SeasonOptions) -> list[str]:
        notes = []
        if not self._store.has_data():
            return ["The settlement history is still downloading (OI · Volume · ATR page); the table fills in when it arrives."]
        depth = self._store.loaded_history_days
        if depth is not None and options.max_days > depth * TRADING_DAYS_PER_YEAR // 365:
            notes.append(
                f"The stored settlements reach about {depth * TRADING_DAYS_PER_YEAR // 365} trading days before expiry "
                f"({depth} calendar days). Raise 'OI history' in Settings to see further back."
            )
        return notes

    def table(self, product: str, family: str, months: int, options: SeasonOptions) -> SeasonTable:
        months = max(min_months(family), min(int(months), MAX_MONTHS))
        today = self._clock().date()
        contracts = generic_contracts(product, today, months)
        live = self._live(product, months, family, options)
        rows = []
        for index in range(months - len(FAMILY_WEIGHTS[family]) + 1):
            contract = contracts[index]
            season = self._season(product, family, contract.month, contract.year, index + 1, options, today)
            point = live.get(index + 1)
            rows.append(SeasonRow(season, evaluate(season, None if point is None else point.value), point))
        notes = self.notes(product, options)
        notes += self._regime_note(rows)
        return SeasonTable(product, family, months, today, rows, notes)

    @staticmethod
    def _regime_note(rows: list[SeasonRow]) -> list[str]:
        """When most of the table is outside its band on one side, the market as a whole has moved, not one structure."""
        scored = [r for r in rows if r.evaluation.status != NO_DATA]
        for side, name in ((1, RICH), (-1, CHEAP)):
            out = [r for r in scored if r.evaluation.side == side]
            if len(scored) >= 4 and len(out) >= 0.6 * len(scored):
                return [
                    f"{len(out)} of {len(scored)} structures are outside their band on the {name} side. That is a market-wide "
                    "move (a different regime from the earlier years), not a set of individual mispricings."
                ]
        return []

    def row(self, product: str, family: str, months: int, position: int, options: SeasonOptions) -> SeasonRow | None:
        table = self.table(product, family, months, options)
        return next((r for r in table.rows if r.season.position == position), None)
