"""Excel-sourced correlation data: a manual fallback source for the Correlation tab's
watchlist when the live/historical API can't provide a series for a given contract.

Files are uploaded once via the Correlation tab and saved to disk under `data_dir`
(like the price Parquet cache), so they stay available across restarts without
re-uploading. A sheet must have a `Timestamp` column followed by one or more value
columns holding price LEVELS; this module differences them the same way
DataLoader.load_price_differences does (plain diff, no forward-fill — a missing
print is never fabricated into a value), and UTC-indexes them so they align with
every other series in the app. The resulting series are mixed into the same
watchlist as instruments/structures and fed to the same correlation math in
core.correlation (build_correlation_matrix_from_series, rolling_correlation_from_series, etc).

An Excel column is a generic curve/tenor label (e.g. "CL1", "CL1-2"), NOT an
exchange-quoted contract symbol — it does not automatically stand in for any
specific instrument (which one "CL1" means changes as the front month rolls). The
trader adds it to the watchlist explicitly and is responsible for interpreting it.
"""

import re
from pathlib import Path

import pandas as pd

from core.exceptions import CrudeOilRiskError

ALLOWED_SUFFIX = ".xlsx"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def _safe_filename(name: str) -> str:
    """Strip any path components and disallowed characters; reject non-.xlsx names."""
    name = Path(name).name
    name = _SAFE_NAME_RE.sub("_", name)
    if not name.lower().endswith(ALLOWED_SUFFIX):
        raise ValueError("Only .xlsx files are supported.")
    return name


class ExcelCorrelationStore:
    """Persists uploaded workbooks to disk and serves parsed price-difference series from them.

    Parsed sheets are cached in memory per filename until the file is re-saved or deleted,
    so repeated Compute clicks don't re-parse the workbook each time.
    """

    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, pd.DataFrame]] = {}  # filename -> {sheet: levels df, UTC-indexed}

    def list_files(self) -> list[str]:
        return sorted(p.name for p in self._data_dir.glob(f"*{ALLOWED_SUFFIX}"))

    def save_file(self, filename: str, content: bytes) -> str:
        """Save an uploaded workbook (overwriting any existing file of the same name).

        Returns the sanitized filename actually used. Raises ValueError if the name isn't
        a .xlsx file, or CrudeOilRiskError if the bytes aren't a readable workbook.
        """
        safe = _safe_filename(filename)
        path = self._data_dir / safe
        path.write_bytes(content)
        self._cache.pop(safe, None)
        try:
            self._sheets(safe)  # parse eagerly so a bad upload is reported immediately
        except CrudeOilRiskError:
            path.unlink(missing_ok=True)
            self._cache.pop(safe, None)
            raise
        return safe

    def delete_file(self, filename: str) -> None:
        safe = _safe_filename(filename)
        (self._data_dir / safe).unlink(missing_ok=True)
        self._cache.pop(safe, None)

    def _sheets(self, filename: str) -> dict[str, pd.DataFrame]:
        safe = _safe_filename(filename)
        if safe in self._cache:
            return self._cache[safe]
        path = self._data_dir / safe
        if not path.exists():
            raise CrudeOilRiskError(f"Uploaded file not found: {filename}")
        try:
            # Context manager: closes the underlying file handle, so a rejected/replaced
            # upload can still be deleted or overwritten right after (Windows locks open files).
            with pd.ExcelFile(path) as workbook:
                parsed: dict[str, pd.DataFrame] = {}
                for sheet_name in workbook.sheet_names:
                    df = workbook.parse(sheet_name)
                    if "Timestamp" not in df.columns:
                        continue
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
                    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").set_index("Timestamp")
                    parsed[sheet_name] = df.drop(columns=[c for c in df.columns if c == "Timestamp"])
        except CrudeOilRiskError:
            raise
        except Exception as exc:  # noqa: BLE001 - a corrupt/unsupported file must not crash the tab
            raise CrudeOilRiskError(f"Could not read {filename}: {exc}") from exc
        if not parsed:
            raise CrudeOilRiskError(f"No sheet with a 'Timestamp' column was found in {filename}.")
        self._cache[safe] = parsed
        return parsed

    def sheet_names(self, filename: str) -> list[str]:
        return list(self._sheets(filename).keys())

    def column_names(self, filename: str, sheet: str) -> list[str]:
        sheets = self._sheets(filename)
        if sheet not in sheets:
            raise CrudeOilRiskError(f"Sheet '{sheet}' not found in {filename}.")
        return list(sheets[sheet].columns)

    def load_series(self, filename: str, sheet: str, column: str, min_rows: int = 2) -> pd.Series:
        """Daily price differences for one Excel column, UTC-indexed like every other series
        in the app so it aligns with instrument/structure series in the same watchlist.

        Raises CrudeOilRiskError if the file/sheet/column can't be found or too few rows
        remain after differencing.
        """
        sheets = self._sheets(filename)
        if sheet not in sheets:
            raise CrudeOilRiskError(f"Sheet '{sheet}' not found in {filename}.")
        df = sheets[sheet]
        if column not in df.columns:
            raise CrudeOilRiskError(f"Column '{column}' not found in {filename}/{sheet}.")
        levels = pd.to_numeric(df[column], errors="coerce")
        diffs = levels.diff().dropna()
        if len(diffs) < min_rows:
            raise CrudeOilRiskError(
                f"Insufficient data for {filename}/{sheet}/{column}: {len(diffs)} rows after differencing "
                f"(minimum {min_rows})."
            )
        diffs.name = column
        return diffs
