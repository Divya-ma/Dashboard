"""Small persistent stores for the Curve Kinks page: previous-day settlements and daily curve snapshots.

PrevSettlementStore - the previous trading day's settlement (the `close` field of
    /common/dailymarketdata/) for every live outright, fetched once a day at the opening time
    set in Settings. The API has no spread/fly settlements, so those are built from the outright
    settlements with the family's leg weights (the same way an exchange settles a spread).
SnapshotStore - the last live curve of each day (all four families), kept in one Parquet file
    per product so any earlier day's curve can be overlaid on today's.
"""

import json
import logging
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from adapters.qh_api import QHApi
from core.curve_calendar import CurveStructure

logger = logging.getLogger(__name__)

LOOKBACK_CALENDAR_DAYS = 10  # enough to span a long weekend plus a holiday


class PrevSettlementStore:
    """Previous settlements by outright API symbol, persisted as JSON."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._state: dict = {"fetched_on": None, "data": {}}
        if self._path.exists():
            try:
                self._state = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning("Could not read %s; starting empty", self._path)

    @property
    def fetched_on(self) -> str | None:
        """Local (opening-time-zone) date the settlements were last fetched for, ISO string."""
        return self._state.get("fetched_on")

    def prices(self) -> dict[str, float]:
        return {symbol: row["close"] for symbol, row in self._state.get("data", {}).items()}

    def settlement_date(self) -> date | None:
        """The trading day those settlements are for (most common date among the contracts)."""
        dates = [row["date"] for row in self._state.get("data", {}).values()]
        return date.fromisoformat(max(set(dates), key=dates.count)) if dates else None

    def gd_codes(self) -> dict[str, str]:
        return {s: row.get("gd_code", "") for s, row in self._state.get("data", {}).items()}

    def fetch(self, api: QHApi, symbols: list[str], today: date, fetched_on: date) -> dict:
        """Fetch each symbol's latest settlement before `today` and store it. Returns a report.

        One paginated dailymarketdata call (comma-separated qhcodes). Symbols the API has no
        settlement for are listed in `missing`. Raises APIError / RuntimeError on a failed call,
        leaving the previously stored values untouched.
        """
        start = (today - timedelta(days=LOOKBACK_CALENDAR_DAYS)).isoformat()
        rows = api.get_all(
            "daily_market_data", page_size=1000, max_pages=3,
            qhcode=",".join(symbols), start=start, end=today.isoformat(),
        )
        latest: dict[str, dict] = {}
        for row in rows:
            row_date = datetime.fromisoformat(row["datetime"]).date()
            symbol = row.get("qhcode")
            if row.get("close") is None or row_date >= today or symbol not in symbols:
                continue
            if symbol not in latest or row_date > date.fromisoformat(latest[symbol]["date"]):
                latest[symbol] = {"date": row_date.isoformat(), "close": float(row["close"]), "gd_code": row.get("gd_code", "")}
        with self._lock:
            self._state = {"fetched_on": fetched_on.isoformat(), "data": latest}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._state), encoding="utf-8")
        return {"fetched": len(latest), "missing": [s for s in symbols if s not in latest]}

    def structure_settlement(self, structure: CurveStructure) -> float | None:
        """Settlement of an outright / spread / fly / dfly from its outright legs; None if any leg is missing."""
        prices = self.prices()
        if any(leg not in prices for leg in structure.legs):
            return None
        return float(sum(w * prices[leg] for w, leg in zip(structure.leg_weights, structure.legs)))


class SnapshotStore:
    """One Parquet file per product of daily curve snapshots: (snap_date, family, position, label, value)."""

    _COLUMNS = ["snap_date", "family", "position", "label", "value"]

    def __init__(self, data_dir: str | Path):
        self._dir = Path(data_dir) / "snapshots"
        self._lock = threading.Lock()

    def _path(self, product: str) -> Path:
        return self._dir / f"{product}.parquet"

    def _read(self, product: str) -> pd.DataFrame:
        path = self._path(product)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=self._COLUMNS)

    def save(self, product: str, snap_date: date, rows: list[dict]) -> None:
        """Replace the snapshot for `snap_date` with `rows` (dicts with family, position, label, value)."""
        if not rows:
            return
        new = pd.DataFrame([{**r, "snap_date": snap_date.isoformat()} for r in rows])[self._COLUMNS]
        with self._lock:
            old = self._read(product)
            kept = old[old["snap_date"] != snap_date.isoformat()]
            self._dir.mkdir(parents=True, exist_ok=True)
            pd.concat([kept, new], ignore_index=True).to_parquet(self._path(product), index=False)

    def dates(self, product: str) -> list[str]:
        """Snapshot dates, newest first."""
        with self._lock:
            return sorted(self._read(product)["snap_date"].unique().tolist(), reverse=True)

    def load(self, product: str, snap_date: str, family: str) -> dict[str, float]:
        """{contract label: value} of one family on one snapshot day."""
        with self._lock:
            frame = self._read(product)
        subset = frame[(frame["snap_date"] == snap_date) & (frame["family"] == family)]
        return {row.label: float(row.value) for row in subset.itertuples()}
