"""Generic-contract history for the Curve Kinks page, cached as Parquet.

Source: POST /generic-charts/generic/data/ with resolve=1 (so CO1 is always the front Brent
contract) and gap=0 (no roll adjustment: a generic series must show the curve's real shape).
Only position 1 (CO1) and the N-1 consecutive spreads (CO1-2 .. CO14-15) are downloaded, and every
curve type is DERIVED from them:

    outright k = outright 1 - (spread 1 + ... + spread k-1)
    fly k      = spread k - spread k+1
    dfly k     = fly k - fly k+1

Why not download each type: on real data the far outrights trade on only some days (CO13 to CO15
have 625 to 853 of about 1,330 days) and far flies exist for only about 300 days, so aligning all
positions on common dates left 204 days of outright and 140 of fly history, too little for the PCA
or for seasonality. The spreads are complete (about 1,329 days each), so the derived curves keep
the full history and the download shrinks from 42 series to 15.

Each spread row also carries the actual contract it held that day (the API's `Product`), which
gives the delivery month of every position and so lets seasonality work by DELIVERY MONTH rather
than by generic position. The last bar the API returns is today's unfinished one and is never
stored. The first load pulls `years` of daily bars; later refreshes pull only the gap.
`refresh` costs ceil(15 / GROUPS_PER_CALL) POSTs per product, so it runs once a day.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adapters.base import LETTER_TO_MONTH, APIError
from adapters.qh_api import QHApi
from core.curve_calendar import DFLY, FAMILIES, FAMILY_WEIGHTS, FLY, N_MONTHS, OUTRIGHT, PRODUCTS, SPREAD

logger = logging.getLogger(__name__)

GROUPS_PER_CALL = 14
# The API fails a whole batch once the days requested across its series get large. Measured on
# the real API, with "Length mismatch: Expected axis has 0 elements" as the failure: 3 series at
# 1,330 days (3,990) came back; 6 at 1,330 (7,980) and 14 at 1,330 (18,620) failed. So a call is
# limited by total days as well as by series count; 5,000 leaves a margin below the largest success.
MAX_ROWS_PER_CALL = 5000
TRADING_DAYS_PER_YEAR = 260
# refresh() also retries any series that still fails inside a batch on its own.
_COLUMNS = ["date", "close", "contract"]


def generic_code(product: str, family: str, position: int) -> str:
    """History code for a family/position: ("BRN", "fly", 2) -> "CO2-3-4". Dflies have none."""
    if family == DFLY:
        raise ValueError("a dfly has no generic code; it is derived from two flies")
    api = PRODUCTS[product].api_code
    legs = len(FAMILY_WEIGHTS[family])
    return api + "-".join(str(position + i) for i in range(legs))


def _contract_month(contract: str) -> int:
    """Delivery month of the front leg of "COZ26" / "COZ26-F27" / "COZ26-F27-G27"."""
    return LETTER_TO_MONTH[contract.split("-")[0][-3]]


@dataclass
class HistoryMatrix:
    """Complete (no missing day) history of one family: rows = days, columns = generic positions."""

    values: np.ndarray  # T x L
    dates: np.ndarray  # T, datetime64[D]
    months: np.ndarray  # T x L, delivery month of the front leg held by each position on each day
    last_date: date | None = None

    @property
    def rows(self) -> int:
        return self.values.shape[0]


@dataclass
class RefreshReport:
    product: str
    series_total: int = 0
    series_updated: int = 0
    rows_added: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


class CurveHistoryStore:
    """Parquet-backed store of generic-contract daily history."""

    def __init__(self, data_dir: str | Path, api: QHApi):
        self._dir = Path(data_dir)
        self._api = api
        self._lock = threading.Lock()
        self._series: dict[tuple[str, str], pd.DataFrame | None] = {}
        self._matrices: dict[tuple[str, str, int], HistoryMatrix | None] = {}
        self.version = 0  # bumped by every refresh that stored data, so callers can drop derived caches

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _path(self, product: str, code: str) -> Path:
        return self._dir / "generic" / product / f"{code}.parquet"

    def read_series(self, product: str, code: str) -> pd.DataFrame | None:
        """Stored rows (date, close, contract) for one generic code, or None if never fetched."""
        key = (product, code)
        with self._lock:
            if key not in self._series:
                path = self._path(product, code)
                self._series[key] = pd.read_parquet(path) if path.exists() else None
            return self._series[key]

    def _write(self, product: str, code: str, frame: pd.DataFrame) -> None:
        path = self._path(product, code)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        with self._lock:
            self._series[(product, code)] = frame

    def last_date(self, product: str, code: str) -> date | None:
        frame = self.read_series(product, code)
        return None if frame is None or frame.empty else frame["date"].max().date()

    def codes(self, product: str, n: int = N_MONTHS) -> list[tuple[str, int, str]]:
        """[(family, position, code)] of the series that are downloaded: outright 1 and the n-1 spreads."""
        return [(OUTRIGHT, 1, generic_code(product, OUTRIGHT, 1))] + [
            (SPREAD, k, generic_code(product, SPREAD, k)) for k in range(1, n)
        ]

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(payload: dict, code: str, today: date) -> pd.DataFrame:
        entry = (payload or {}).get(f"{code}_1D") or {}
        rows = entry.get("df") or []
        if not rows:
            return pd.DataFrame(columns=_COLUMNS)
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime([r["Date"] for r in rows], unit="ms").normalize(),
                "close": [float(r["Close"]) for r in rows],
                "contract": [str(r.get("Product", "")) for r in rows],
            }
        )
        frame = frame[frame["date"].dt.date < today]  # today's bar is unfinished
        return frame.dropna(subset=["close"])

    @staticmethod
    def _batches(wanted: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
        """Split (code, days) pairs into calls of at most GROUPS_PER_CALL series and MAX_ROWS_PER_CALL days."""
        batches: list[list[tuple[str, int]]] = []
        days = 0
        for code, count in wanted:
            if not batches or len(batches[-1]) >= GROUPS_PER_CALL or days + count > MAX_ROWS_PER_CALL:
                batches.append([])
                days = 0
            batches[-1].append((code, count))
            days += count
        return batches

    @staticmethod
    def _api_error(payload: dict, code: str) -> str | None:
        entry = (payload or {}).get(f"{code}_1D")
        if entry is None:
            return "missing from the API response"
        return entry.get("error") or None

    def _call(self, product: str, batch: list[tuple[str, int]], end_ms: int, report: RefreshReport) -> dict | None:
        groups = [
            {"product": code, "interval": "1D", "resolve": 1, "gap": 0, "end": end_ms, "count": count}
            for code, count in batch
        ]
        try:
            return self._api.generic_chart_data(groups)
        except (APIError, RuntimeError) as exc:
            for code, _ in batch:
                report.errors[code] = str(exc)
            logger.error("Generic history call failed for %s: %s", product, exc)
            return None

    def _store(
        self, product: str, batch: list[tuple[str, int]], payload: dict, today: date, report: RefreshReport
    ) -> dict[str, str]:
        """Merge what the API returned for each code. Returns {code: reason} for the codes that failed."""
        failed: dict[str, str] = {}
        for code, _count in batch:
            api_error = self._api_error(payload, code)
            frame = self._parse(payload, code, today)
            if api_error and frame.empty:
                failed[code] = api_error
                continue
            try:
                report.rows_added += self._merge(product, code, frame)
                report.series_updated += 1
                report.errors.pop(code, None)
            except Exception as exc:  # noqa: BLE001 - one bad series must not lose the others
                failed[code] = str(exc)
                logger.error("Could not store generic history for %s: %s", code, exc)
        return failed

    def refresh(self, product: str, years: int, today: date | None = None, n: int = N_MONTHS) -> RefreshReport:
        """Fetch missing history (first run) or the gap since the last stored day, for every series."""
        today = today or datetime.now(timezone.utc).date()
        report = RefreshReport(product=product)
        full_count = int(years * TRADING_DAYS_PER_YEAR) + 30
        full, incremental = [], []
        for _family, _k, code in self.codes(product, n):
            last = self.last_date(product, code)
            if last is None:
                full.append(code)
            else:
                incremental.append((code, min(1500, (today - last).days + 7)))
        report.series_total = len(full) + len(incremental)

        wanted = [(code, full_count) for code in full]
        if incremental:
            count = max(c for _, c in incremental)  # one count per call: take the largest gap
            wanted += [(code, count) for code, _ in incremental]
        batches = self._batches(wanted)

        end_ms = int(time.time() * 1000)
        for batch in batches:
            payload = self._call(product, batch, end_ms, report)
            if payload is None:
                continue
            failed = self._store(product, batch, payload, today, report)
            if failed and len(batch) > 1:
                # The API can fail a whole batch for one awkward code, so ask for the failures one by one.
                logger.warning(
                    "%s: %d of %d series failed in a batch (%s); retrying them one at a time",
                    product, len(failed), len(batch), next(iter(failed.values()))[:120],
                )
                retry = [(code, count) for code, count in batch if code in failed]
                failed = {}
                for item in retry:
                    single = self._call(product, [item], end_ms, report)
                    if single is not None:
                        failed.update(self._store(product, [item], single, today, report))
            for code, reason in failed.items():
                report.errors[code] = reason
                logger.error("No generic history stored for %s %s: %s", product, code, reason)
        if report.series_updated:
            with self._lock:
                self._matrices.clear()
                self.version += 1
        return report

    def _merge(self, product: str, code: str, new: pd.DataFrame) -> int:
        old = self.read_series(product, code)
        if new.empty:
            if old is None:
                raise ValueError("the API returned no rows for this code")
            return 0
        combined = new if old is None else pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
        added = len(combined) - (0 if old is None else len(old))
        self._write(product, code, combined[_COLUMNS])
        return added

    # ------------------------------------------------------------------
    # Matrices
    # ------------------------------------------------------------------

    def matrix(self, product: str, family: str, n: int = N_MONTHS) -> HistoryMatrix | None:
        """History of one curve type (every position present on every row), or None if there is no data."""
        with self._lock:
            if (product, family, n) in self._matrices:
                return self._matrices[(product, family, n)]
        built = self._derive(product, n)
        with self._lock:
            for name, matrix in built.items():
                self._matrices[(product, name, n)] = matrix
        return built[family]

    def _derive(self, product: str, n: int) -> dict[str, HistoryMatrix | None]:
        """All four curve types from outright 1 and the n-1 spreads (see the module docstring)."""
        nothing: dict[str, HistoryMatrix | None] = {family: None for family in FAMILIES}
        first = self.read_series(product, generic_code(product, OUTRIGHT, 1))
        spreads = [self.read_series(product, generic_code(product, SPREAD, k)) for k in range(1, n)]
        if first is None or first.empty or any(s is None or s.empty for s in spreads):
            return nothing
        closes = pd.concat(
            [first.set_index("date")["close"]] + [s.set_index("date")["close"] for s in spreads],
            axis=1, join="inner", keys=range(n),
        ).dropna()
        if closes.empty:
            return nothing
        contracts = pd.concat(
            [s.set_index("date")["contract"] for s in spreads], axis=1, join="inner", keys=range(n - 1)
        ).reindex(closes.index)
        front = contracts.apply(lambda col: col.map(_safe_month)).to_numpy(dtype=float)  # T x (n-1)
        back = contracts.iloc[:, -1].map(_safe_back_month).to_numpy(dtype=float)  # delivery month of position n
        usable = ~np.isnan(front).any(axis=1) & ~np.isnan(back)
        if not usable.any():
            return nothing
        values = closes.to_numpy(dtype=float)[usable]
        dates = closes.index.to_numpy(dtype="datetime64[D]")[usable]
        last = closes.index[usable][-1].date()
        front, back = front[usable].astype(int), back[usable].astype(int)

        outright_1, spread = values[:, 0], values[:, 1:]  # T x (n-1)
        outright = np.column_stack([outright_1, outright_1[:, None] - np.cumsum(spread, axis=1)])  # T x n
        fly = spread[:, :-1] - spread[:, 1:]  # T x (n-2)
        dfly = fly[:, :-1] - fly[:, 1:]  # T x (n-3)
        parts = {
            OUTRIGHT: (outright, np.column_stack([front, back])),
            SPREAD: (spread, front),
            FLY: (fly, front[:, : n - 2]),
            DFLY: (dfly, front[:, : n - 3]),
        }
        return {
            family: HistoryMatrix(values=v, dates=dates, months=m, last_date=last) for family, (v, m) in parts.items()
        }


def _safe_back_month(contract: str) -> float:
    """Delivery month of the back leg of "COZ26-F27" (the second leg)."""
    try:
        return float(LETTER_TO_MONTH[contract.split("-")[1][0]])
    except (KeyError, IndexError, ValueError):
        return np.nan


def _safe_month(contract: str) -> float:
    try:
        return float(_contract_month(contract))
    except (KeyError, IndexError, ValueError):
        return np.nan
