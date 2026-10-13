"""Daily OI, close and volume of every outright contract, cached as Parquet.

Source: GET /common/dailymarketdata/ (fields oi, close, volume), which holds the previous
trading day's figures - so "live" OI is yesterday's, and everything here is daily.

UNIVERSE. Every outright of CL and BRN whose delivery month is from January FIRST_CONTRACT_YEAR
(2019) to MONTHS_AHEAD (36) months from today. The monitored ladder is the first N of these (set
on the page); the rest are kept because the roll maths needs EVERY contract alive on a day (OI that
leaves one contract shows up in the others).

HISTORY. For each contract the download starts `history_days` CALENDAR days before its expiry
(Settings, default 1500) and runs to today. A contract is complete once it has expired and been
fetched after its expiry; after that it is never requested again. Live contracts are topped up from
a few days before their last stored day, so the daily update is a single small request.
Changing history_days to a larger value re-downloads the contracts that need the extra days.

EXPIRY. For live contracts it is the rule-based date from core.curve_calendar (an approximation).
For finished contracts the page uses the last day the contract has data, which is exact.

Files in `data_dir`: `<product>.parquet` (contract, date, oi, close, volume) and `meta.json`.
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

from core.curve_calendar import PRODUCTS, api_symbol, contract_expiry

logger = logging.getLogger(__name__)

FIRST_CONTRACT_YEAR = 2019
MONTHS_AHEAD = 36
PAGE_SIZE = 20_000
FULL_CHUNK = 12  # contracts per request when downloading a full history (~15k rows)
INCREMENTAL_CHUNK = 60
OVERLAP_DAYS = 5  # a top-up starts this many days before the last stored day (late corrections)
EXPIRED_AFTER_DAYS = 3
COLUMNS = ["contract", "date", "oi", "close", "volume"]


@dataclass(frozen=True)
class OiContract:
    product: str
    symbol: str  # API symbol, e.g. "CLZ26"
    month: int
    year: int
    expiry: date  # rule-based and approximate; finished contracts use their last data day instead

    @property
    def order(self) -> int:
        """Delivery month as a running number, so a later contract has a larger order."""
        return self.year * 12 + self.month

    @property
    def label(self) -> str:
        return date(self.year, self.month, 1).strftime("%b%y")


def contract_universe(
    product: str, today: date, first_year: int | None = None, months_ahead: int | None = None,
) -> list[OiContract]:
    """Every outright from January `first_year` (default FIRST_CONTRACT_YEAR) to `months_ahead`
    (default MONTHS_AHEAD) months after today's month."""
    contracts = []
    year, month = first_year or FIRST_CONTRACT_YEAR, 1
    last = today.year * 12 + today.month - 1 + (MONTHS_AHEAD if months_ahead is None else months_ahead)
    while year * 12 + month - 1 <= last:
        contracts.append(OiContract(product, api_symbol(product, month, year), month, year, contract_expiry(product, month, year)))
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return contracts


@dataclass
class OiProgress:
    running: bool = False
    done: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    finished_at: str | None = None


def _clean(rows: list[dict]) -> pd.DataFrame:
    """API rows -> tidy frame: one row per (contract, date); a missing OI stays NaN."""
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame(rows)
    out = pd.DataFrame({
        "contract": frame["qhcode"].astype(str),
        "date": pd.to_datetime(frame["datetime"]).dt.tz_localize(None).dt.normalize(),
        "oi": pd.to_numeric(frame.get("oi"), errors="coerce"),
        "close": pd.to_numeric(frame.get("close"), errors="coerce"),
        "volume": pd.to_numeric(frame.get("volume"), errors="coerce"),
    })
    return out.drop_duplicates(["contract", "date"], keep="last")


class OiStore:
    def __init__(self, data_dir: str | Path, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self._dir = Path(data_dir)
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self.progress = OiProgress()
        self._meta: dict = {"contracts": {}, "updated_on": None, "history_days": None}
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
        """Changes whenever stored data changes (used to cache research results)."""
        return f"{self._meta.get('updated_on')}|{self._meta.get('history_days')}|{self._meta.get('saved_at')}"

    def frame(self, product: str) -> pd.DataFrame:
        """All stored rows of one product (contract, date, oi, close, volume), sorted."""
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

    def plan(self, product: str, history_days: int, today: date) -> list[tuple[OiContract, date]]:
        """(contract, request start) for every contract that still needs data, in universe order."""
        todo = []
        for contract in contract_universe(product, today):
            need_start = contract.expiry - timedelta(days=history_days)
            if need_start > today:
                continue
            stored = self._meta["contracts"].get(contract.symbol)
            expired = contract.expiry < today - timedelta(days=EXPIRED_AFTER_DAYS)
            if stored and stored["fetched_from"] <= need_start.isoformat():
                if expired and stored["fetched_to"] >= (contract.expiry + timedelta(days=1)).isoformat():
                    continue  # complete: expired and fetched after its expiry
                start = max(need_start, date.fromisoformat(stored["fetched_to"]) - timedelta(days=OVERLAP_DAYS))
            else:
                start = need_start
            todo.append((contract, start))
        return todo

    # -- downloading ----------------------------------------------------

    @staticmethod
    def _fetch(api, codes: list[str], start: date, end: date) -> list[dict]:
        """Every row for these contracts, paging by the reply's own `count`."""
        rows: list[dict] = []
        offset = 0
        while True:
            payload = api.daily_market_data(
                qhcode=",".join(codes), start=start.isoformat(), end=end.isoformat(),
                fields="oi,close,volume", limit=PAGE_SIZE, offset=offset,
            )
            batch = payload.get("results", []) if isinstance(payload, dict) else payload
            rows.extend(batch)
            offset += len(batch)
            count = payload.get("count", len(rows)) if isinstance(payload, dict) else len(rows)
            if not batch or offset >= count:
                return rows

    def _save(self, product: str, new: pd.DataFrame, chunk: list[tuple[OiContract, date]], today: date) -> None:
        with self._lock:
            path = self._path(product)
            old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
            combined = pd.concat([old, new], ignore_index=True) if len(old) else new.copy()
            combined = combined.drop_duplicates(["contract", "date"], keep="last").sort_values(["contract", "date"]).reset_index(drop=True)
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".parquet.tmp")
            combined.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            for contract, start in chunk:
                previous = self._meta["contracts"].get(contract.symbol)
                from_day = min(start.isoformat(), previous["fetched_from"]) if previous else start.isoformat()
                self._meta["contracts"][contract.symbol] = {"fetched_from": from_day, "fetched_to": today.isoformat()}
            self._write_meta()

    def _write_meta(self) -> None:
        self._meta["saved_at"] = self._clock().isoformat()
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self._meta), encoding="utf-8")
        os.replace(tmp, self._dir / "meta.json")

    def update(self, api, history_days: int, today: date | None = None) -> dict:
        """Download what is missing for every product. Progress is saved chunk by chunk, so an
        interrupted or failed run resumes where it stopped. Raises the API error that stopped it."""
        today = today or self._clock().date()
        work: list[tuple[str, list[tuple[OiContract, date]], bool]] = []
        for product in PRODUCTS:
            todo = self.plan(product, history_days, today)
            full = [item for item in todo if item[0].symbol not in self._meta["contracts"] or self._needs_full(item, history_days)]
            top_up = [item for item in todo if item not in full]
            for i in range(0, len(full), FULL_CHUNK):
                work.append((product, full[i:i + FULL_CHUNK], True))
            for i in range(0, len(top_up), INCREMENTAL_CHUNK):
                work.append((product, top_up[i:i + INCREMENTAL_CHUNK], False))
        self.progress.total, self.progress.done = len(work), 0
        report = {"requests": 0, "rows": 0, "contracts": 0}
        for product, chunk, is_full in work:
            start = min(s for _, s in chunk)
            self.progress.message = (
                f"Downloading {product} {chunk[0][0].label}-{chunk[-1][0].label} "
                f"({self.progress.done + 1} of {self.progress.total})"
            )
            rows = self._fetch(api, [c.symbol for c, _ in chunk], start, today)
            report["requests"] += 1
            report["rows"] += len(rows)
            report["contracts"] += len(chunk)
            self._save(product, _clean(rows), chunk, today)
            self.progress.done += 1
        with self._lock:
            self._meta["updated_on"] = today.isoformat()
            self._meta["history_days"] = history_days
            self._write_meta()
        return report

    def _needs_full(self, item: tuple[OiContract, date], history_days: int) -> bool:
        contract, start = item
        stored = self._meta["contracts"].get(contract.symbol)
        return stored is None or start.isoformat() < stored["fetched_from"]

    def contract_expiry_dates(self, product: str) -> dict[str, pd.Timestamp]:
        """Last day with data per contract: the true expiry of finished contracts."""
        frame = self.frame(product)
        if frame.empty:
            return {}
        return frame.groupby("contract")["date"].max().to_dict()
