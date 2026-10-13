"""OI research on finished contracts: peak, build-up and last-month roll-out, per product and delivery month.

Every comparison is between contracts of the SAME delivery month in different years (all the Januaries
together, all the Decembers together): each month has its own size and seasonal cycle, so a statistic
that pooled January with other months would describe none of them. For one product and one delivery
month the research gives:

  Q1  the peak OI of each contract and how many days before expiry it came
  Q3  how long before expiry a contract reaches 10 / 25 / 50 / 75 / 90 / 100% of its peak, and what share of
      the peak is still open N days before expiry
  Q2  the last month, one row per calendar day: the share of the peak still open, how much fell that day,
      how much of the fall is estimated to have rolled into other contracts and how much exited, and which
      following months the rolled OI went to (+1, +2, +3, +4 or later)

plus the statistics by delivery month (a row per month, Jan to Dec) of the peak OI and of the OI N days
before expiry (month_oi_tables).

Only FINISHED contracts are used (expired, with the whole cycle on record); the live ones are what the
monitor compares against these results. "Days before expiry" are calendar days and, for the peak, trading
days too. A finished contract's expiry is the last day it has data.

ROLL-OUT uses the roll attribution of core.oi_roll on every day of a contract's last LAST_MONTH_DAYS: its
fall in OI that day is split into the part assumed to have rolled into other contracts and the part that
exited, and the rolled part is spread over the contracts that gained OI that day. It is an estimate, as the
OI of one contract cannot say who rolled.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from core.oi_data import EXPIRED_AFTER_DAYS, contract_universe
from core.oi_roll import attribute_rolls

THRESHOLDS = (10, 25, 50, 75, 90, 100)  # % of the peak
LAST_MONTH_DAYS = 30
OPEN_AT_DAYS = (60, 45, 30, 21, 14, 10, 7, 5, 3, 1)
OI_AT_DAYS = (90, 60, 45, 30, 21, 14, 7, 3)  # absolute OI this many days before expiry, for the per-month tables
OFFSET_LABELS = {1: "+1 month", 2: "+2 months", 3: "+3 months", 4: "+4 or later"}
MIN_CYCLE_ROWS = 60
MIN_PEAK_OI = 1000.0


@dataclass
class ProductResearch:
    product: str
    contracts: pd.DataFrame = field(default_factory=pd.DataFrame)  # one row per finished contract
    pct_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)  # share of peak OI by days before expiry (rows) x contract
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)  # per contract and day of its last month (see build_daily)
    month_peak: pd.DataFrame = field(default_factory=pd.DataFrame)  # peak OI statistics by delivery month
    month_at: pd.DataFrame = field(default_factory=pd.DataFrame)  # OI N days before expiry, by delivery month
    n_finished: int = 0
    n_skipped: int = 0
    error: str | None = None

    @property
    def empty(self) -> bool:
        return self.contracts.empty


def _cycle(symbol: str, series: pd.Series, dates: pd.DatetimeIndex, expiry: pd.Timestamp, contract) -> tuple[dict, pd.Series] | None:
    """One finished contract's cycle row and its %-of-peak series by days before expiry."""
    oi = series.where(series > 0).dropna()
    if len(oi) < MIN_CYCLE_ROWS or oi.max() < MIN_PEAK_OI:
        return None
    peak = float(oi.max())
    peak_date = oi.idxmax()  # the first day the peak was reached
    dte = (expiry - oi.index).days
    pct = pd.Series(oi.to_numpy() / peak, index=dte)
    row = {
        "contract": symbol, "label": contract.label, "month": contract.month, "year": contract.year,
        "expiry": expiry, "peak_oi": peak, "peak_date": peak_date,
        "peak_dte": int((expiry - peak_date).days),
        "peak_dte_trading": int(len(dates) - 1 - dates.get_loc(peak_date)),
        "oi_starts_dte": int(dte[0]),
    }
    for threshold in THRESHOLDS:
        first = oi.index[np.argmax(oi.to_numpy() >= threshold / 100.0 * peak)]
        row[f"t{threshold}"] = int((expiry - first).days)
    for days in OPEN_AT_DAYS:
        earlier = pct[pct.index >= days]  # the last observation at or before that point in time
        row[f"open_{days}"] = float(earlier.iloc[-1]) if len(earlier) else np.nan
    absolute = pd.Series(oi.to_numpy(), index=dte)
    for days in OI_AT_DAYS:
        earlier = absolute[absolute.index >= days]
        row[f"oi_{days}"] = float(earlier.iloc[-1]) if len(earlier) else np.nan
    full = np.arange(int(dte.max()), int(dte.min()) - 1, -1)
    return row, pct.reindex(full).ffill()


def _roll_records(oi: pd.DataFrame, cycles: pd.DataFrame, orders: dict[str, int]) -> pd.DataFrame:
    """Per finished contract and day of its last month: its fall in OI and where the rolled part went."""
    delta = oi.diff()
    expiries = dict(zip(cycles["contract"], cycles["expiry"]))
    window_days = set()
    for expiry in expiries.values():
        window_days.update(d for d in oi.index if 0 <= (expiry - d).days <= LAST_MONTH_DAYS)
    results = {day: attribute_rolls(delta.loc[day]) for day in sorted(window_days)}
    records = []
    for symbol, expiry in expiries.items():
        if symbol not in oi.columns:
            continue
        for day in oi.index[(oi.index <= expiry) & (oi.index >= expiry - timedelta(days=LAST_MONTH_DAYS))]:
            dte = (expiry - day).days
            change = delta.at[day, symbol]
            if not np.isfinite(change) or dte < 1:
                continue
            record = {"contract": symbol, "dte": dte, "change": float(change), "fall": 0.0, "rolled": 0.0,
                      **{f"to_{k}": 0.0 for k in OFFSET_LABELS}}
            if change < 0:
                result = results[day]
                record["fall"] = float(-change)
                record["rolled"] = float(result.rolled_out.get(symbol, 0.0))
                for receiver, amount in result.flows.loc[symbol].items():
                    offset = orders.get(receiver, 0) - orders[symbol]
                    if offset >= 1:
                        record[f"to_{min(offset, 4)}"] += float(amount)
            records.append(record)
    return pd.DataFrame(records)


def build_daily(oi: pd.DataFrame, contracts: pd.DataFrame, pct_matrix: pd.DataFrame, orders: dict[str, int]) -> pd.DataFrame:
    """One row per finished contract and day of its last month, as shares (%) of that contract's peak OI:
    open_pct (still open), change_pct (that day's change), fall_pct, rolled_pct, exited_pct, and to_1..to_4
    (the part of the rolled OI that went to the following months, +1 ... +4 or later)."""
    records = _roll_records(oi, contracts, orders)
    if records.empty:
        return records
    peaks = dict(zip(contracts["contract"], contracts["peak_oi"]))
    peak = records["contract"].map(peaks).astype(float)
    out = pd.DataFrame({"contract": records["contract"], "dte": records["dte"]})
    out["open_pct"] = [
        float(pct_matrix.at[d, c]) * 100 if d in pct_matrix.index and c in pct_matrix.columns else np.nan
        for c, d in zip(records["contract"], records["dte"])
    ]
    for column in ("change", "fall", "rolled"):
        out[f"{column}_pct"] = records[column] / peak * 100
    out["exited_pct"] = out["fall_pct"] - out["rolled_pct"]
    for k in OFFSET_LABELS:
        out[f"to_{k}"] = records[f"to_{k}"] / peak * 100
    return out


def build_research(frame: pd.DataFrame, product: str, today: date) -> ProductResearch:
    """All the research tables of one product from its stored daily frame."""
    result = ProductResearch(product=product)
    if frame.empty:
        result.error = "No OI data downloaded yet"
        return result
    universe = {c.symbol: c for c in contract_universe(product, today)}
    oi = frame.pivot(index="date", columns="contract", values="oi").sort_index()
    row_dates = {name: pd.DatetimeIndex(sorted(group)) for name, group in frame.groupby("contract")["date"]}
    last_day = frame.groupby("contract")["date"].max()
    finished = [s for s, c in universe.items() if s in last_day.index and c.expiry < today - timedelta(days=EXPIRED_AFTER_DAYS)]
    rows, profiles = [], {}
    for symbol in finished:
        cycle = _cycle(symbol, oi[symbol], row_dates[symbol], last_day[symbol], universe[symbol])
        if cycle is None:
            result.n_skipped += 1
            continue
        rows.append(cycle[0])
        profiles[symbol] = cycle[1]
    if not rows:
        result.error = "No finished contract has enough OI history yet"
        return result
    result.contracts = pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)
    result.n_finished = len(rows)
    stacked = pd.concat(profiles.values(), axis=1)
    stacked.columns = list(profiles)
    result.pct_matrix = stacked.sort_index(ascending=False)
    result.pct_matrix.index.name = "days_before_expiry"
    result.month_peak, result.month_at = month_oi_tables(result.contracts)
    result.daily = build_daily(oi, result.contracts, result.pct_matrix, {s: c.order for s, c in universe.items()})
    return result


# ----------------------------------------------------------------------
# One delivery month across the years
# ----------------------------------------------------------------------


def month_contracts(research: ProductResearch, month: int) -> pd.DataFrame:
    """The finished contracts of one delivery month, oldest first."""
    if research.empty:
        return research.contracts
    return research.contracts[research.contracts["month"] == month].reset_index(drop=True)


def _spread(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        return {"median": np.nan, "sd": np.nan, "min": np.nan, "max": np.nan}
    return {
        "median": float(values.median()), "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "min": float(values.min()), "max": float(values.max()),
    }


def threshold_table(contracts: pd.DataFrame) -> pd.DataFrame:
    """How many days before expiry the contracts of ONE month reached each share of their peak: median, SD, min, max."""
    return pd.DataFrame([{"threshold": f"{t}% of peak", **_spread(contracts[f"t{t}"])} for t in THRESHOLDS])


def open_table(contracts: pd.DataFrame) -> pd.DataFrame:
    """The share (%) of the peak still open N days before expiry, for the contracts of ONE month."""
    return pd.DataFrame([{"days_before_expiry": d, **_spread(contracts[f"open_{d}"] * 100)} for d in OPEN_AT_DAYS])


def last_month_table(research: ProductResearch, month: int) -> pd.DataFrame:
    """The last month of the contracts of ONE delivery month, one row per calendar day before expiry
    (LAST_MONTH_DAYS down to 1; a day with no trading in any of those contracts is left blank).

    Per day, across the years' contracts, as % of each contract's peak OI: the median share still open and
    its lowest and highest; the median change that day; the median fall, how much of it rolled and how much
    exited; then of the OI that rolled, the % that went to the next month, 2 months out, 3 and 4 or later
    (pooled over the contracts that day)."""
    days = list(range(LAST_MONTH_DAYS, 0, -1))
    columns = ["dte", "open_median", "open_min", "open_max", "change_median", "fall_median", "rolled_median", "exited_median",
               *[f"share_{k}" for k in OFFSET_LABELS]]
    if research.empty or research.daily.empty:
        return pd.DataFrame({"dte": days}).reindex(columns=columns)
    symbols = set(month_contracts(research, month)["contract"])
    part = research.daily[research.daily["contract"].isin(symbols)]
    rows = []
    for day in days:
        d = part[part["dte"] == day]
        row = {"dte": day}
        if d.empty:
            rows.append(row)
            continue
        rolled_total = d["rolled_pct"].sum()
        row.update({
            "open_median": float(d["open_pct"].median()), "open_min": float(d["open_pct"].min()), "open_max": float(d["open_pct"].max()),
            "change_median": float(d["change_pct"].median()), "fall_median": float(d["fall_pct"].median()),
            "rolled_median": float(d["rolled_pct"].median()), "exited_median": float(d["exited_pct"].median()),
            **{f"share_{k}": float(d[f"to_{k}"].sum() / rolled_total * 100) if rolled_total > 0 else np.nan for k in OFFSET_LABELS},
        })
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def month_oi_tables(contracts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OI statistics for every delivery month (Jan to Dec) across the finished contracts of that month.

    month_peak: the peak OI (n, median, mean, standard deviation, min, 25th/75th percentile, max, and the
    spread of the days-before-expiry at which the peak came). month_at: the OI N days before expiry
    (OI_AT_DAYS), median and standard deviation of each."""
    peak_rows, at_rows = [], []
    for month in range(1, 13):
        part = contracts[contracts["month"] == month]
        if part.empty:
            continue
        peak = part["peak_oi"]
        peak_rows.append({
            "month": month, "median": float(peak.median()), "mean": float(peak.mean()),
            "sd": float(peak.std(ddof=1)) if len(part) > 1 else np.nan, "min": float(peak.min()),
            "p25": float(peak.quantile(0.25)), "p75": float(peak.quantile(0.75)), "max": float(peak.max()),
            "dte_median": float(part["peak_dte"].median()),
            "dte_sd": float(part["peak_dte"].std(ddof=1)) if len(part) > 1 else np.nan,
        })
        row = {"month": month}
        for days in OI_AT_DAYS:
            values = part[f"oi_{days}"].dropna()
            row[f"median_{days}"] = float(values.median()) if len(values) else np.nan
            row[f"sd_{days}"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        at_rows.append(row)
    return pd.DataFrame(peak_rows), pd.DataFrame(at_rows)
