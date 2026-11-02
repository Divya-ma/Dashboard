"""Front-month series that do not count the contract switch on the first day after an expiry.

A continuous "front month" price is made of different contracts. On the first trading day after a contract expires, the series
moves to the next contract, so its price jumps by roughly the calendar spread. That jump is a SWITCH, not a market move, and it
repeats every month. For CL|BZ it happens twice a month on different days, because the legs expire on different dates: CL after
its own expiry, Brent after its (about three weeks earlier). Counting those days as movement would make CL|BZ look as if it
"drops" on the first day after every CL expiry.

So every front-month series here carries `is_roll` (the first trading day after an expiry) and a daily `changes` array that is
NaN on those days. Any "move over N days" is the sum of the valid changes, never the jump. A fixed delivery contract (what the
Seasonality curves are) never jumps and needs none of this.

Expiry is exact for a finished contract (its last settlement day) and rule-based (core.curve_calendar) for a live one.
No I/O: the caller passes the stored settlement frame (contract, date, oi, close, volume).
"""

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from core.curve_calendar import contract_expiry, parse_contract, product_spec
from core.oi_data import EXPIRED_AFTER_DAYS

MIN_VALID_SHARE = 0.6  # of a lookback window's days must have a usable change
VALUE_TOLERANCE_DAYS = 5  # how far back a value may be carried to a date that has none


@dataclass(frozen=True)
class RollEvent:
    product: str
    day: date  # the first trading day after the expiry
    from_contract: str
    to_contract: str
    jump: float | None  # what a naive continuous series would have shown that day (front price today - front price yesterday)


@dataclass(frozen=True)
class WindowRead:
    total: float | None  # sum of the valid changes; None when too few days are usable
    std: float | None
    valid: int
    rolls: tuple[pd.Timestamp, ...]  # roll days inside the window (their change is not counted)


@dataclass
class ChangeSeries:
    """Daily changes of a front-month series (or of a difference of two) with the roll days marked."""

    name: str
    index: pd.DatetimeIndex
    changes: np.ndarray  # NaN on roll days and wherever a price is missing
    is_roll: np.ndarray

    def read(self, when: pd.Timestamp, n: int) -> WindowRead | None:
        """The last `n` trading days ending on `when` (or the last day before it, within a few days)."""
        pos = int(self.index.searchsorted(when, side="right")) - 1
        if pos < 0 or (when - self.index[pos]).days > VALUE_TOLERANCE_DAYS:
            return None
        low = max(0, pos - n + 1)
        chunk = self.changes[low:pos + 1]
        valid = chunk[np.isfinite(chunk)]
        rolls = tuple(self.index[low:pos + 1][self.is_roll[low:pos + 1]])
        if len(chunk) < n or len(valid) < math.ceil(MIN_VALID_SHARE * n):
            return WindowRead(None, None, int(len(valid)), rolls)
        return WindowRead(float(valid.sum()), float(valid.std(ddof=1)) if len(valid) >= 3 else None, int(len(valid)), rolls)


def total_between(series: ChangeSeries, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    """Sum of the valid changes on the trading days after `start` up to and including `end` (roll days are blank, so not counted)."""
    low, high = int(series.index.searchsorted(start, side="right")), int(series.index.searchsorted(end, side="right"))
    chunk = series.changes[low:high]
    valid = chunk[np.isfinite(chunk)]
    if len(chunk) == 0 or len(valid) < math.ceil(MIN_VALID_SHARE * len(chunk)):
        return None
    return float(valid.sum())


def difference(first: ChangeSeries, second: ChangeSeries, name: str) -> ChangeSeries:
    """The series first - second (CL front - Brent front). A day is dropped when EITHER leg rolled or lacks a price."""
    index = first.index.union(second.index)
    a = pd.Series(first.changes, index=first.index).reindex(index)
    b = pd.Series(second.changes, index=second.index).reindex(index)
    roll_a = pd.Series(first.is_roll, index=first.index).reindex(index, fill_value=False).astype(bool)
    roll_b = pd.Series(second.is_roll, index=second.index).reindex(index, fill_value=False).astype(bool)
    return ChangeSeries(name, index, (a - b).to_numpy(dtype=float), (roll_a | roll_b).to_numpy())


@dataclass
class LegData:
    """One product's stored settlements as a front-month series, with its roll days."""

    product: str
    close: pd.DataFrame  # dates x contract, contracts in delivery order
    oi: pd.DataFrame
    ordered: list[str]  # contract symbols, earliest delivery first (the column order of `close`)
    front: np.ndarray  # per date: index into `ordered` of the front contract, -1 where it is not known
    expiry: dict[str, date]
    rolls: list[RollEvent] = field(default_factory=list)
    series: ChangeSeries | None = None

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def calendar(self) -> str:
        return product_spec(self.product).calendar

    def event_on(self, day: date) -> RollEvent | None:
        return next((e for e in self.rolls if e.day == day), None)


def _delivery_order(symbol: str) -> tuple[int, int, int] | None:
    try:
        _, month, year = parse_contract(symbol)
    except (KeyError, ValueError):
        return None
    return year * 12 + month, month, year


def build_leg_data(product: str, frame: pd.DataFrame) -> LegData | None:
    """Front-month series of one product from its stored settlement frame; None when there is nothing usable."""
    if frame is None or frame.empty:
        return None
    close = frame.pivot(index="date", columns="contract", values="close").sort_index()
    oi = frame.pivot(index="date", columns="contract", values="oi").sort_index()
    parsed = []  # (delivery order, symbol, month, year)
    for symbol in close.columns:
        found = _delivery_order(symbol)
        if found is not None:
            parsed.append((found[0], symbol, found[1], found[2]))
    parsed.sort()
    if not parsed:
        return None
    ordered = [symbol for _, symbol, _, _ in parsed]
    close, oi = close[ordered], oi.reindex(columns=ordered)

    latest = close.index.max()
    expiry: dict[str, date] = {}
    for _, symbol, month, year in parsed:
        last = close[symbol].dropna().index.max()
        finished = pd.notna(last) and last < latest - pd.Timedelta(days=EXPIRED_AFTER_DAYS + 2)
        expiry[symbol] = last.date() if finished else contract_expiry(product, month, year)

    dates = close.index.values.astype("datetime64[D]")
    expiries = np.maximum.accumulate(np.array([np.datetime64(expiry[s], "D") for s in ordered], dtype="datetime64[D]"))
    front = np.minimum(np.searchsorted(expiries, dates, side="left"), len(ordered) - 1)

    # The store starts at a delivery month: before the contract ahead of it expired, the real front month is not stored.
    first_month, first_year = parsed[0][2], parsed[0][3]
    before = first_year * 12 + first_month - 2
    valid_from = contract_expiry(product, before % 12 + 1, before // 12) + timedelta(days=1)
    valid = (dates >= np.datetime64(valid_from, "D")) & (dates <= expiries[-1])
    front = np.where(valid, front, -1)

    values = close.to_numpy(dtype=float)
    changes = np.full(len(close), np.nan)
    is_roll = np.zeros(len(close), dtype=bool)
    rolls: list[RollEvent] = []
    for i in range(1, len(close)):
        if front[i] < 0 or front[i - 1] < 0:
            continue
        if front[i] != front[i - 1]:
            is_roll[i] = True
            today_price, yesterday_price = values[i, front[i]], values[i - 1, front[i - 1]]
            jump = float(today_price - yesterday_price) if np.isfinite(today_price) and np.isfinite(yesterday_price) else None
            rolls.append(RollEvent(product, close.index[i].date(), ordered[front[i - 1]], ordered[front[i]], jump))
        else:
            changes[i] = values[i, front[i]] - values[i - 1, front[i]]
    leg = LegData(product, close, oi, ordered, front, expiry, rolls)
    leg.series = ChangeSeries(product, close.index, changes, is_roll)
    return leg
