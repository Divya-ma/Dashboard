"""Five-minute candles of the live outrights, spreads and flies of CL and BRN, cached as Parquet, for the Churn page.

Source: GET /ohlc/ (interval 5M). With a start and an end it returns every candle of up to 50 instruments in that
range, one list with a `product` field and a `time` in milliseconds (the same shape as the daily candles of
core.atr_data). To keep each reply small the download goes in narrow date windows and small batches.

D-FLIES. A d-fly (M1 - 3 M2 + 3 M3 - M4) has no quote of its own, so its candles are DERIVED from two stored flies,
fly(k) - fly(k+1), see `derive_dfly_bars`: close to close, so the extremes inside a candle are not seen (a target is
a little harder to hit than on a real instrument). It is not downloaded.

WHICH INSTRUMENTS. For each product the first N live (not yet expired) contracts of each family: outrights
(CLZ26), consecutive spreads (CLZ26-F27) and consecutive flies (CLZ26-F27-G27), front first. N is a Churn page
setting. An instrument that expires drops out of the list and, with it, out of the stored file.

WINDOW. The last `window_days` CALENDAR days (default 60). Each instrument remembers up to which day it has been
fetched; a daily update asks only for the days after that (a few days earlier, for late corrections), then old
rows are cut so the files stay small. Progress is saved after every request, so an interrupted download resumes.

Files in `data_dir`: `<product>.parquet` (symbol, ts, open, high, low, close, volume; ts is UTC) and `meta.json`.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from adapters.base import APIError
from core.atr_data import AtrInstrument, instrument_universe
from core.box_candles import fill_missing_boxes
from core.curve_calendar import PRODUCTS
from core.oi_data import OiProgress

logger = logging.getLogger(__name__)

FAMILIES = ("outright", "spread", "fly")
BATCH = 8  # instruments per request
WINDOW_DAYS_PER_REQUEST = 3  # calendar days per request: ~8 x 3 x 276 five-minute candles
OVERLAP_DAYS = 2
COLUMNS = ["symbol", "ts", "open", "high", "low", "close", "volume"]


def tracked_instruments(product: str, today: date, contracts: int) -> list[AtrInstrument]:
    """The first `contracts` live instruments of each family, by family then expiry."""
    live = [i for i in instrument_universe(product, today) if i.expiry > today]
    out: list[AtrInstrument] = []
    for family in FAMILIES:
        mine = sorted((i for i in live if i.family == family), key=lambda i: i.order)
        out += mine[:contracts]
    return out


@dataclass(frozen=True)
class DerivedInstrument:
    """A d-fly, built from two consecutive flies of the same product."""

    product: str
    family: str
    symbol: str  # "<fly>/<next fly>", never sent to the API
    label: str
    legs: tuple[str, str]  # the two fly symbols
    order: int


def dfly_instruments(product: str, flies: list[AtrInstrument]) -> list[DerivedInstrument]:
    """d-flies from consecutive flies (a fly one month apart from the next), front first."""
    flies = sorted(flies, key=lambda i: i.order)
    out = []
    for a, b in zip(flies, flies[1:]):
        if b.order - a.order != 1:
            continue
        label = f"{a.label}/{b.label.split('/')[-1]}"
        out.append(DerivedInstrument(product, "dfly", f"{a.symbol}/{b.symbol}", label, (a.symbol, b.symbol), a.order))
    return out


def derive_dfly_bars(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Candles of fly(first) - fly(second): close to close on every candle either fly printed, each fly's last close
    carried forward. Open is the previous close, high and low the larger and smaller of the two. Volume is the
    smaller of the two flies' daily volumes (a d-fly cannot trade more than its thinner fly), spread evenly over the day's candles."""
    a, b = first.set_index("ts"), second.set_index("ts")
    index = a.index.union(b.index)
    close = (a["close"].reindex(index).ffill() - b["close"].reindex(index).ffill()).dropna()
    previous = close.shift(1).fillna(close)
    volume_a = a["volume"].fillna(0).groupby(a.index.normalize()).sum()
    volume_b = b["volume"].fillna(0).groupby(b.index.normalize()).sum()
    daily = pd.concat([volume_a, volume_b], axis=1).fillna(0).min(axis=1)
    day = close.index.normalize()
    candles = pd.Series(1.0, index=close.index).groupby(day).transform("sum")
    volume = pd.Series(day.map(daily).fillna(0.0).to_numpy() / candles.to_numpy(), index=close.index)  # the day's volume spread over its candles
    return pd.DataFrame({"ts": close.index, "open": previous.to_numpy(), "high": np.maximum(previous, close).to_numpy(),
                         "low": np.minimum(previous, close).to_numpy(), "close": close.to_numpy(), "volume": volume.to_numpy()}).reset_index(drop=True)


def _clean(candles: list[dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame(candles)
    out = pd.DataFrame({
        "symbol": frame["product"].astype(str),
        "ts": pd.to_datetime(frame["time"], unit="ms"),
        "open": pd.to_numeric(frame["open"], errors="coerce"), "high": pd.to_numeric(frame["high"], errors="coerce"),
        "low": pd.to_numeric(frame["low"], errors="coerce"), "close": pd.to_numeric(frame["close"], errors="coerce"),
        "volume": pd.to_numeric(frame.get("volume"), errors="coerce"),
    })
    return out.dropna(subset=["close"]).drop_duplicates(["symbol", "ts"], keep="last")


class ChurnStore:
    def __init__(self, data_dir: str | Path, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self._dir = Path(data_dir)
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self.progress = OiProgress()
        self._meta: dict = {"instruments": {}, "updated_on": None, "window_days": None, "contracts": None}
        meta_path = self._dir / "meta.json"
        if meta_path.exists():
            try:
                self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning("Could not read %s; starting empty", meta_path)

    # -- reading --------------------------------------------------------

    def _path(self, product: str) -> Path:
        return self._dir / f"{product}.parquet"

    @property
    def updated_on(self) -> str | None:
        return self._meta.get("updated_on")

    @property
    def version(self) -> str:
        return f"{self._meta.get('updated_on')}|{self._meta.get('window_days')}|{self._meta.get('contracts')}|{self._meta.get('saved_at')}"

    @property
    def daily_version(self) -> str:
        """Changes only when a new day's (or a wider) download completes; the ranking is cached on it."""
        return f"{self._meta.get('updated_on')}|{self._meta.get('window_days')}|{self._meta.get('contracts')}"

    def is_current(self, today: date, window_days: int, contracts: int, now: datetime | None = None, max_age_minutes: float | None = None) -> bool:
        """Fetched today for at least this window and number of contracts and, when `max_age_minutes` is given, no longer ago than that."""
        current = (self._meta.get("updated_on") == today.isoformat() and (self._meta.get("window_days") or 0) >= window_days
                   and (self._meta.get("contracts") or 0) >= contracts)
        refreshed = self._meta.get("refreshed_at")
        if current and max_age_minutes is not None and now is not None:
            if not refreshed or (now - datetime.fromisoformat(refreshed)).total_seconds() > max_age_minutes * 60:
                return False
        return current

    def has_data(self) -> bool:
        return any(self._path(p).exists() for p in PRODUCTS)

    def frame(self, product: str) -> pd.DataFrame:
        path = self._path(product)
        with self._lock:
            if not path.exists():
                return pd.DataFrame(columns=COLUMNS)
            mtime = path.stat().st_mtime
            cached = self._cache.get(product)
            if cached is None or cached[0] != mtime:
                self._cache[product] = (mtime, pd.read_parquet(path))
            return self._cache[product][1]

    # -- planning -------------------------------------------------------

    def plan(self, product: str, today: date, window_days: int, contracts: int) -> list[tuple[AtrInstrument, date]]:
        """(instrument, first day to request) for every tracked instrument that is not already fetched up to today."""
        todo = []
        first = today - timedelta(days=window_days)
        for instrument in tracked_instruments(product, today, contracts):
            stored = self._meta["instruments"].get(instrument.symbol)
            if stored and stored["fetched_from"] <= first.isoformat():
                start = date.fromisoformat(stored["fetched_to"]) - timedelta(days=OVERLAP_DAYS)
                if stored["fetched_to"] >= today.isoformat():
                    start = today - timedelta(days=OVERLAP_DAYS)
                todo.append((instrument, max(start, first)))
            else:
                todo.append((instrument, first))
        return todo

    # -- downloading ----------------------------------------------------

    @staticmethod
    def _fetch(api, symbols: list[str], start: date, end: date) -> list[dict]:
        unix = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
        reply = api.ohlc("5M", instruments=symbols, start=unix(start), end=unix(end + timedelta(days=1)))
        if isinstance(reply, dict):  # {"status": "SUCCESS", "data": []} when nothing traded in the range
            if reply.get("status", "SUCCESS") != "SUCCESS" or not isinstance(reply.get("data", []), list):
                raise APIError(f"/ohlc/ returned {str(reply)[:200]}")
            reply = reply.get("data", [])
        return [c for c in reply if isinstance(c, dict) and "product" in c and "time" in c]

    def _write_meta(self) -> None:
        self._meta["saved_at"] = self._clock().isoformat()
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self._meta), encoding="utf-8")
        os.replace(tmp, self._dir / "meta.json")

    def _save(self, product: str, new: pd.DataFrame, chunk: list[tuple[AtrInstrument, date]], through: date, today: date, window_days: int) -> None:
        with self._lock:
            path = self._path(product)
            old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
            combined = pd.concat([old, new], ignore_index=True) if len(old) else new.copy()
            combined = combined.drop_duplicates(["symbol", "ts"], keep="last").sort_values(["symbol", "ts"]).reset_index(drop=True)
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            combined.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            for instrument, start in chunk:
                previous = self._meta["instruments"].get(instrument.symbol)
                from_day = min(start.isoformat(), previous["fetched_from"]) if previous else start.isoformat()
                fetched_to = max(through.isoformat(), previous["fetched_to"]) if previous else through.isoformat()
                self._meta["instruments"][instrument.symbol] = {"fetched_from": from_day, "fetched_to": fetched_to}
            self._write_meta()

    def _prune(self, today: date, window_days: int) -> None:
        """Cut rows older than the window and instruments that are no longer tracked (expired)."""
        cutoff = pd.Timestamp(today - timedelta(days=window_days))
        with self._lock:
            for product in PRODUCTS:
                path = self._path(product)
                if not path.exists():
                    continue
                frame = pd.read_parquet(path)
                kept = frame[frame["ts"] >= cutoff]
                live = {i.symbol for i in tracked_instruments(product, today, 200)}
                kept = kept[kept["symbol"].isin(live)]
                if len(kept) != len(frame):
                    tmp = path.with_suffix(".parquet.tmp")
                    kept.to_parquet(tmp, index=False)
                    os.replace(tmp, path)
            alive = {i.symbol for p in PRODUCTS for i in tracked_instruments(p, today, 200)}
            self._meta["instruments"] = {s: m for s, m in self._meta["instruments"].items() if s in alive}

    def update(self, api, window_days: int, contracts: int, today: date | None = None) -> dict:
        """Download what is missing. Saved request by request; raises the API error that stopped it."""
        today = today or self._clock().date()
        work: list[tuple[str, list[tuple[AtrInstrument, date]], date, date]] = []
        for product in PRODUCTS:
            todo = self.plan(product, today, window_days, contracts)
            by_start: dict[date, list[tuple[AtrInstrument, date]]] = {}
            for item in todo:
                by_start.setdefault(item[1], []).append(item)
            for start, items in sorted(by_start.items()):
                for i in range(0, len(items), BATCH):
                    chunk = items[i:i + BATCH]
                    day = start
                    while day <= today:
                        end = min(day + timedelta(days=WINDOW_DAYS_PER_REQUEST - 1), today)
                        work.append((product, chunk, day, end))
                        day = end + timedelta(days=1)
        self.progress.total, self.progress.done = len(work), 0
        report = {"requests": 0, "rows": 0, "instruments": sum(len(self.plan(p, today, window_days, contracts)) for p in PRODUCTS)}
        for product, chunk, start, end in work:
            first, last = chunk[0][0], chunk[-1][0]
            self.progress.message = f"Downloading 5-minute candles: {product} {first.family}s {first.label} to {last.label}, {start} ({self.progress.done + 1} of {self.progress.total})"
            candles = self._fetch(api, [i.symbol for i, _ in chunk], start, end)
            if PRODUCTS[product].is_box:  # a box /ohlc/ has no candles for is built from its CL and BZ legs
                candles += fill_missing_boxes(lambda symbols: self._fetch(api, symbols, start, end), [i.symbol for i, _ in chunk], candles, daily=False)
            frame = _clean(candles)
            report["requests"] += 1
            report["rows"] += len(frame)
            self._save(product, frame, chunk, end, today, window_days)
            self.progress.done += 1
        self._prune(today, window_days)
        with self._lock:
            self._meta["updated_on"] = today.isoformat()
            self._meta["window_days"] = window_days
            self._meta["contracts"] = contracts
            self._meta["refreshed_at"] = self._clock().isoformat()
            self._write_meta()
        return report
