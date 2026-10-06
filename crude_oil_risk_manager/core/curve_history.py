"""Generic-contract history for the Curve Kinks page, cached as Parquet.

Source: POST /generic-charts/generic/data/ with resolve=1 (so CO1 is always the front Brent
contract) and gap=0 (no roll adjustment: a generic series must show the curve's real shape).
One series per generic code (CO1..CO15 outrights, CO1-2.. spreads, CO1-2-3.. flies); a dfly
series is derived as fly(k) - fly(k+1). Each row also carries the actual contract that
generic position held on that date (the API's `Product`), which is what lets seasonality be
computed by DELIVERY MONTH rather than by generic position.

The last bar the API returns is today's unfinished one; it is never stored (history is closed
days only). The first load pulls `years` of daily bars; later refreshes pull only the gap.
`refresh` costs ceil(series / GROUPS_PER_CALL) POSTs per product, so it runs once a day.
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
from core.curve_calendar import DFLY, FAMILY_WEIGHTS, FLY, N_MONTHS, OUTRIGHT, PRODUCTS, SPREAD, family_length

logger = logging.getLogger(__name__)

GROUPS_PER_CALL = 14
TRADING_DAYS_PER_YEAR = 260
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
        """[(family, position, code)] for every fetched series (outright, spread, fly)."""
        return [
            (family, k, generic_code(product, family, k))
            for family in (OUTRIGHT, SPREAD, FLY)
            for k in range(1, family_length(family, n) + 1)
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

        batches: list[tuple[list[tuple[str, int]]]] = []
        pairs = [(code, full_count) for code in full]
        for start in range(0, len(pairs), GROUPS_PER_CALL):
            batches.append(pairs[start:start + GROUPS_PER_CALL])
        if incremental:
            count = max(c for _, c in incremental)  # one count per call: take the largest gap
            for start in range(0, len(incremental), GROUPS_PER_CALL):
                batches.append([(code, count) for code, _ in incremental[start:start + GROUPS_PER_CALL]])

        end_ms = int(time.time() * 1000)
        for batch in batches:
            groups = [
                {"product": code, "interval": "1D", "resolve": 1, "gap": 0, "end": end_ms, "count": count}
                for code, count in batch
            ]
            try:
                payload = self._api.generic_chart_data(groups)
            except (APIError, RuntimeError) as exc:
                for code, _ in batch:
                    report.errors[code] = str(exc)
                logger.error("Generic history call failed for %s: %s", product, exc)
                continue
            for code, _ in batch:
                try:
                    report.rows_added += self._merge(product, code, self._parse(payload, code, today))
                    report.series_updated += 1
                except Exception as exc:  # noqa: BLE001 - one bad series must not lose the others
                    report.errors[code] = str(exc)
                    logger.exception("Could not store generic history for %s", code)
        if report.series_updated:
            with self._lock:
                self._matrices.clear()
                self.version += 1
        return report

    def _merge(self, product: str, code: str, new: pd.DataFrame) -> int:
        old = self.read_series(product, code)
        if new.empty:
            if old is None:
                raise ValueError("the API returned no data for this code")
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
        """Aligned history of one family (all positions present on every row), or None if no data."""
        key = (product, family, n)
        with self._lock:
            if key in self._matrices:
                return self._matrices[key]
        built = self._build(product, family, n)
        with self._lock:
            self._matrices[key] = built
        return built

    def _frames(self, product: str, family: str, n: int) -> list[pd.DataFrame] | None:
        length = family_length(family, n)
        if family == DFLY:
            flies = [self.read_series(product, generic_code(product, FLY, k)) for k in range(1, length + 2)]
            if any(f is None or f.empty for f in flies):
                return None
            frames = []
            for k in range(length):
                a = flies[k].set_index("date")
                b = flies[k + 1].set_index("date")
                joined = a[["close", "contract"]].join(b[["close"]], rsuffix="_next", how="inner")
                frames.append(pd.DataFrame({"close": joined["close"] - joined["close_next"], "contract": joined["contract"]}))
            return frames
        frames = []
        for k in range(1, length + 1):
            series = self.read_series(product, generic_code(product, family, k))
            if series is None or series.empty:
                return None
            frames.append(series.set_index("date")[["close", "contract"]])
        return frames

    def _build(self, product: str, family: str, n: int) -> HistoryMatrix | None:
        frames = self._frames(product, family, n)
        if not frames:
            return None
        closes = pd.concat([f["close"] for f in frames], axis=1, join="inner", keys=range(len(frames))).dropna()
        if closes.empty:
            return None
        contracts = pd.concat([f["contract"] for f in frames], axis=1, join="inner", keys=range(len(frames))).loc[closes.index]
        months = contracts.apply(lambda col: col.map(_safe_month)).to_numpy(dtype=float)
        keep = ~np.isnan(months).any(axis=1)
        closes, months = closes[keep], months[keep].astype(int)
        if closes.empty:
            return None
        return HistoryMatrix(
            values=closes.to_numpy(dtype=float),
            dates=closes.index.to_numpy(dtype="datetime64[D]"),
            months=months,
            last_date=closes.index[-1].date(),
        )


def _safe_month(contract: str) -> float:
    try:
        return float(_contract_month(contract))
    except (KeyError, IndexError, ValueError):
        return np.nan
