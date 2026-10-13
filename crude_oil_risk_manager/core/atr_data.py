"""Daily OHLC of every outright, spread and fly of CL and BRN, cached as Parquet, for the ATR views.

Source: GET /ohlc/ (interval 1D), which with a start and an end returns EVERY daily candle of up to 50
instruments in that range, one list with a `product` field. (Giving start, end AND count is refused.)

UNIVERSE. For each product the same contracts as the OI page (core.oi_data.contract_universe: January 2019 to
36 months ahead), as three families: outrights (CLZ26), consecutive spreads (CLZ26-F27) and consecutive flies
(CLZ26-F27-G27). A spread or fly is dated by its FIRST leg (it stops trading when that leg expires). Some
structures have little or no history (the API simply returns nothing for them).

HISTORY. Each instrument downloads from `history_days` CALENDAR days before its expiry (Settings, default
400: a year of life plus the 14-day average's warm-up) to today. An instrument is complete once it has
expired and been fetched after its expiry; live ones are topped up from a few days before their last stored
day. Progress is saved batch by batch, so an interrupted download resumes.

Files in `data_dir`: `<product>.parquet` (symbol, date, open, high, low, close, volume) and `meta.json`.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from adapters.base import APIError
from core.curve_calendar import PRODUCTS, month_token
from core.oi_data import EXPIRED_AFTER_DAYS, OiContract, OiProgress, contract_universe

logger = logging.getLogger(__name__)

FAMILIES = ("outright", "spread", "fly")
FULL_BATCH = 20  # instruments per request for a full history (consecutive expiries, so a narrow date range)
TOP_UP_BATCH = 50  # the API's limit
OVERLAP_DAYS = 5
TAIL_DAYS = 5  # candles after the expiry date are not wanted
COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class AtrInstrument:
    product: str
    family: str
    symbol: str  # API symbol
    first: OiContract  # the front leg
    label: str
    expiry: date  # the front leg's (rule-based) expiry

    @property
    def order(self) -> int:
        return self.first.order


def instrument_universe(product: str, today: date, first_year: int | None = None, months_ahead: int | None = None) -> list[AtrInstrument]:
    """Every outright, consecutive spread and consecutive fly of the product, by family then expiry."""
    contracts = contract_universe(product, today, first_year, months_ahead)
    out: list[AtrInstrument] = []
    for family, length in (("outright", 1), ("spread", 2), ("fly", 3)):
        for i in range(len(contracts) - length + 1):
            legs = contracts[i:i + length]
            symbol = legs[0].symbol + "".join(f"-{month_token(c.month, c.year)}" for c in legs[1:])
            label = "/".join(c.label for c in legs)
            out.append(AtrInstrument(product, family, symbol, legs[0], label, legs[0].expiry))
    return out


def family_of(symbol: str) -> str:
    return ("outright", "spread", "fly")[min(symbol.count("-"), 2)]


def _clean(candles: list[dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame(candles)
    out = pd.DataFrame({
        "symbol": frame["product"].astype(str),
        "date": pd.to_datetime(frame["time"], unit="ms").dt.normalize(),
        "open": pd.to_numeric(frame["open"], errors="coerce"), "high": pd.to_numeric(frame["high"], errors="coerce"),
        "low": pd.to_numeric(frame["low"], errors="coerce"), "close": pd.to_numeric(frame["close"], errors="coerce"),
        "volume": pd.to_numeric(frame.get("volume"), errors="coerce"),
    })
    return out.drop_duplicates(["symbol", "date"], keep="last")


class AtrStore:
    def __init__(self, data_dir: str | Path, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self._dir = Path(data_dir)
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self.progress = OiProgress()
        self._meta: dict = {"instruments": {}, "updated_on": None, "history_days": None}
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
    def loaded_history_days(self) -> int | None:
        return self._meta.get("history_days")

    @property
    def version(self) -> str:
        return f"{self._meta.get('updated_on')}|{self._meta.get('history_days')}|{self._meta.get('saved_at')}"

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

    def has_data(self) -> bool:
        return any(self._path(p).exists() for p in PRODUCTS)

    # -- planning -------------------------------------------------------

    def plan(self, product: str, history_days: int, today: date) -> list[tuple[AtrInstrument, date, bool]]:
        """(instrument, request start, is_full) for everything that still needs data, by family then expiry."""
        todo = []
        for instrument in instrument_universe(product, today):
            need_start = instrument.expiry - timedelta(days=history_days)
            if need_start > today:
                continue
            stored = self._meta["instruments"].get(instrument.symbol)
            expired = instrument.expiry < today - timedelta(days=EXPIRED_AFTER_DAYS)
            if stored and stored["fetched_from"] <= need_start.isoformat():
                if expired and stored["fetched_to"] >= (instrument.expiry + timedelta(days=1)).isoformat():
                    continue
                todo.append((instrument, max(need_start, date.fromisoformat(stored["fetched_to"]) - timedelta(days=OVERLAP_DAYS)), False))
            else:
                todo.append((instrument, need_start, True))
        return todo

    # -- downloading ----------------------------------------------------

    @staticmethod
    def _fetch(api, symbols: list[str], start: date, end: date) -> list[dict]:
        unix = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
        reply = api.ohlc("1D", instruments=symbols, start=unix(start), end=unix(end + timedelta(days=1)))
        if isinstance(reply, dict):  # {"status": "SUCCESS", "data": []} when nothing traded in the range
            if reply.get("status", "SUCCESS") != "SUCCESS" or not isinstance(reply.get("data", []), list):
                raise APIError(f"/ohlc/ returned {str(reply)[:200]}")
            reply = reply.get("data", [])
        return [c for c in reply if isinstance(c, dict) and "product" in c and "time" in c]

    def _save(self, product: str, new: pd.DataFrame, batch: list[tuple[AtrInstrument, date, bool]], today: date) -> None:
        with self._lock:
            path = self._path(product)
            old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
            combined = pd.concat([old, new], ignore_index=True) if len(old) else new.copy()
            combined = combined.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            combined.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            for instrument, start, _ in batch:
                previous = self._meta["instruments"].get(instrument.symbol)
                from_day = min(start.isoformat(), previous["fetched_from"]) if previous else start.isoformat()
                self._meta["instruments"][instrument.symbol] = {"fetched_from": from_day, "fetched_to": today.isoformat()}
            self._write_meta()

    def _write_meta(self) -> None:
        self._meta["saved_at"] = self._clock().isoformat()
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self._meta), encoding="utf-8")
        os.replace(tmp, self._dir / "meta.json")

    def update(self, api, history_days: int, today: date | None = None) -> dict:
        """Download what is missing for every product. Saved batch by batch; raises the API error that stopped it."""
        today = today or self._clock().date()
        work: list[tuple[str, list[tuple[AtrInstrument, date, bool]]]] = []
        for product in PRODUCTS:
            todo = self.plan(product, history_days, today)
            for family in FAMILIES:
                mine = [item for item in todo if item[0].family == family]
                for is_full, size in ((True, FULL_BATCH), (False, TOP_UP_BATCH)):
                    part = [item for item in mine if item[2] == is_full]
                    for i in range(0, len(part), size):
                        work.append((product, part[i:i + size]))
        self.progress.total, self.progress.done = len(work), 0
        report = {"requests": 0, "rows": 0, "instruments": 0}
        for product, batch in work:
            start = min(s for _, s, _ in batch)
            end = min(today, max(i.expiry for i, _, _ in batch) + timedelta(days=TAIL_DAYS))
            first, last = batch[0][0], batch[-1][0]
            self.progress.message = f"Downloading ATR data: {product} {first.family}s {first.label} to {last.label} ({self.progress.done + 1} of {self.progress.total})"
            candles = self._fetch(api, [i.symbol for i, _, _ in batch], start, end)
            frame = _clean(candles)
            if not frame.empty:  # keep each instrument's own window (the request range spans the whole batch)
                limits = {i.symbol: (pd.Timestamp(s), pd.Timestamp(i.expiry + timedelta(days=TAIL_DAYS))) for i, s, _ in batch}
                frame = frame[frame["symbol"].isin(limits)]
                low, high = frame["symbol"].map(lambda sym: limits[sym][0]), frame["symbol"].map(lambda sym: limits[sym][1])
                frame = frame[(frame["date"] >= low) & (frame["date"] <= high)]
            report["requests"] += 1
            report["rows"] += len(frame)
            report["instruments"] += len(batch)
            self._save(product, frame, batch, today)
            self.progress.done += 1
        with self._lock:
            self._meta["updated_on"] = today.isoformat()
            self._meta["history_days"] = history_days
            self._write_meta()
        return report
