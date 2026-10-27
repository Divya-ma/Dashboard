"""Live OI monitor: is a contract's OI abnormal for how close it is to expiry?

For each of the next N live contracts of a product (N set on the page, default 15) the latest OI
(the previous trading day's) is compared with what the SAME delivery month's finished contracts had the
SAME number of calendar days before THEIR expiry, over the last `years` years (a January against past
Januaries, never against other months: each month has its own size and seasonal cycle). There is no
"recent contracts of any month" baseline: it mixed months and gave conflicting signals. Measures:
  absolute OI                      used while OI is still building
  % of the contract's peak so far  used once OI has passed its peak and is declining (the final peak
                                   is unknowable live; the peak SO FAR is known, and for the finished
                                   contracts the same figure is computed at the same point)
  share of the product's total OI  shown for context (removes growth of the whole market)

DEVIATION. z = (x - median) / scale, with scale the largest of 1.4826 x MAD of the baseline, its standard
deviation and 10% of its median. (With only five past contracts a MAD can be tiny by chance, which made
ordinary contracts look wildly abnormal; the standard deviation and the 10% floor keep z honest.)
A day is "deviating" when |z| >= z_threshold AND OI >= min_oi AND the implied OI gap is at least
min_change (a contract far from expiry has almost no OI, so a big percentage move there is noise).

SUDDEN or SUSTAINED. The run is the number of consecutive trading days up to the latest one that the
contract has been deviating. A run of 1 to `sudden_days` is SUDDEN (something just happened); a longer
run is SUSTAINED (more likely a lasting, fundamental reason than an anomaly: no alert, but shown).

A baseline with fewer than MIN_BASELINE finished contracts of that month is not trusted: no z, no flag.

ROLL. Yesterday's change in OI of every contract is split into what is estimated to have rolled from one
contract to another, what exited and what is fresh (core.oi_roll), because a fall in one contract with a
rise in the next is a roll, not an anomaly.

FRONT CONTRACT. The contract closest to expiry gets its own check (front_check): in the last LAST_MONTH
calendar days, is it draining the way finished contracts did at the same point? It compares, as shares of
the contract's peak OI, how much is still open, how much OI has dropped since 30 days before expiry, how
much of that is estimated to have rolled into other contracts and how much into the next contract. OI that
stays open (and has not rolled) longer than usual is what a squeeze or delivery interest looks like; OI
that leaves early is the opposite.

Everything is in calendar days before expiry for the cycle position and trading days for the run.
Expiry of a live contract is the rule-based date (approximate); of a finished one, its last data day.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from core.oi_data import EXPIRED_AFTER_DAYS, contract_universe
from core.oi_roll import attribute_rolls, daily_roll_matrix
from core.user_settings import parse_number

logger = logging.getLogger(__name__)

KEY_MONITOR_PARAMS = "oi_monitor_params"
STATUS_NORMAL, STATUS_SUDDEN, STATUS_SUSTAINED = "NORMAL", "SUDDEN", "SUSTAINED"
MEASURE_ABSOLUTE, MEASURE_PERCENT = "OI", "% of peak"
MIN_BASELINE = 3  # finished contracts a baseline needs to be trusted
PEAK_DECLINE = 0.97  # OI below this share of its peak so far counts as declining
LOOKBACK_ROWS = 120  # trading days of a live contract's history examined for its run of deviation
SHARE_FROM = pd.Timestamp("2019-01-01")  # totals only cover every contract from here (the universe starts in 2019)
MAD_TO_SD = 1.4826
SCALE_FLOOR_FRACTION = 0.10  # scale is at least this share of the baseline median
LAST_MONTH = 30  # calendar days before expiry the front-contract check covers
CHART_DAYS = 35
# (key, label) of what the front-contract check compares, each as a share of the contract's peak OI
FRONT_METRICS = (
    ("remaining", "OI still open"),
    ("drop", "OI dropped since 30 days before expiry"),
    ("rolled", "Estimated rolled into other contracts (cumulative)"),
    ("to_next", "...of which into the next contract (cumulative)"),
)


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------


@dataclass
class MonitorParams:
    contracts: int = 15  # the next N live contracts
    years: int = 5  # the same month in the last N years
    z_threshold: float = 2.5
    sudden_days: int = 5  # trading days
    min_oi: float = 5000.0
    min_change: float = 2000.0
    alerts_enabled: bool = True

    def key(self) -> tuple:
        return tuple(asdict(self).values())


PARAM_RULES = {  # name: (label, minimum, maximum, integer)
    "contracts": ("Contracts to watch", 1, 36, True),
    "years": ("Years of the same month", 2, 10, True),
    "z_threshold": ("Deviation threshold (z)", 0.5, 10.0, False),
    "sudden_days": ("Sudden window (days)", 1, 60, True),
    "min_oi": ("Minimum OI", 0, 10_000_000, False),
    "min_change": ("Minimum OI change", 0, 10_000_000, False),
}


def validate_params(raw: dict) -> MonitorParams:
    """Parameters from page inputs; ValueError with a message safe to show for anything unusable."""
    defaults = MonitorParams()
    values = {}
    for name, (label, minimum, maximum, integer) in PARAM_RULES.items():
        value = raw.get(name)
        values[name] = getattr(defaults, name) if value in (None, "") else parse_number(
            value, label, minimum=minimum, maximum=maximum, integer=integer
        )
    values["alerts_enabled"] = bool(raw.get("alerts_enabled", defaults.alerts_enabled))
    return MonitorParams(**values)


def load_params(repository) -> MonitorParams:
    try:
        raw = repository.get_setting(KEY_MONITOR_PARAMS, None)
        return validate_params(raw) if isinstance(raw, dict) else MonitorParams()
    except (ValueError, TypeError):
        return MonitorParams()


def save_params(repository, params: MonitorParams) -> None:
    if load_params(repository) != params or repository.get_setting(KEY_MONITOR_PARAMS, None) is None:
        repository.set_setting(KEY_MONITOR_PARAMS, asdict(params))


# ----------------------------------------------------------------------
# Reading the OI / price move
# ----------------------------------------------------------------------


def oi_price_read(d_oi: float | None, d_price: float | None) -> str:
    """What a move in OI together with a move in price usually means."""
    if d_oi is None or d_price is None or not np.isfinite(d_oi) or not np.isfinite(d_price) or d_oi == 0 or d_price == 0:
        return ""
    if d_oi > 0:
        return "new longs" if d_price > 0 else "new shorts"
    return "short covering" if d_price > 0 else "long liquidation"


# ----------------------------------------------------------------------
# The engine
# ----------------------------------------------------------------------


@dataclass
class MonitorResult:
    product: str
    data_date: pd.Timestamp | None = None
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    roll: pd.DataFrame = field(default_factory=pd.DataFrame)
    roll_summary: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)
    front: dict | None = None  # the front-contract check (MonitorEngine.front_check)
    error: str | None = None

    @property
    def flagged(self) -> pd.DataFrame:
        return self.rows[self.rows["status"] != STATUS_NORMAL] if not self.rows.empty else self.rows


def _baseline(dense: pd.DataFrame, symbols: list[str], floor: float) -> dict[str, pd.Series]:
    """Median, robust scale, spread and count of the baseline contracts at every day before expiry."""
    values = dense[symbols] if symbols else dense.iloc[:, :0]
    median = values.median(axis=1)
    mad = values.sub(median, axis=0).abs().median(axis=1)
    sd = values.std(axis=1, ddof=1)
    scale = np.maximum.reduce([MAD_TO_SD * mad, sd.fillna(0.0), SCALE_FLOOR_FRACTION * median.abs(), floor + 0 * median])
    scale = pd.Series(scale, index=median.index)
    return {
        "median": median, "scale": scale, "mean": values.mean(axis=1), "sd": sd,
        "min": values.min(axis=1), "max": values.max(axis=1), "n": values.notna().sum(axis=1),
    }


class MonitorEngine:
    """Builds the monitor from one product's stored daily frame. Construct once per data version."""

    def __init__(self, frame: pd.DataFrame, product: str, today: date, history_days: int, params: MonitorParams):
        self.product, self.today, self.params, self.history_days = product, today, params, history_days
        self._empty = frame.empty
        if self._empty:
            return
        # A day with no OI at all (ICE publishes a day late) is not a data day for OI: use the latest day that has it
        self.oi = frame.pivot(index="date", columns="contract", values="oi").sort_index().dropna(how="all")
        self.close = frame.pivot(index="date", columns="contract", values="close").sort_index().reindex(self.oi.index)
        self.universe = {c.symbol: c for c in contract_universe(product, today)}
        self.last_date = self.oi.index.max()
        self.total = self.oi.sum(axis=1, min_count=1)
        self.share = self.oi.div(self.total, axis=0)
        self.share[self.share.index < SHARE_FROM] = np.nan
        last_day = frame.groupby("contract")["date"].max()
        self.finished = [
            s for s, c in self.universe.items()
            if s in last_day.index and c.expiry < today - timedelta(days=EXPIRED_AFTER_DAYS)
        ]
        self.expiry_actual = last_day.to_dict()
        self.dense = self._dense_frames()
        self._baseline_cache: dict[tuple, dict[str, pd.Series]] = {}

    # -- finished contracts, by days before expiry -----------------------

    def _dense_frames(self) -> dict[str, pd.DataFrame]:
        """For every finished contract, OI / % of running peak / share at each calendar day before expiry
        (as of the last observation at or before that point; 0 before the contract was listed)."""
        horizon = int(self.history_days)
        index = np.arange(horizon, -1, -1)
        out = {"oi": {}, "pct": {}, "share": {}}
        for symbol in self.finished:
            expiry = self.expiry_actual[symbol]
            oi = self.oi[symbol].loc[:expiry]
            oi = oi.where(oi >= 0).dropna()
            if len(oi) < 20:
                continue
            peak = oi.cummax()
            series = {
                "oi": oi, "pct": (oi / peak).where(peak > 0),
                "share": self.share[symbol].reindex(oi.index),
            }
            dte = (expiry - oi.index).days
            first = int(dte.max())
            for name, values in series.items():
                s = pd.Series(values.to_numpy(), index=dte)
                s = s[~s.index.duplicated(keep="last")].reindex(index).ffill()
                s[s.index > first] = 0.0  # not yet listed: no OI (only for days inside the history window)
                out[name][symbol] = s
        dense = {name: pd.DataFrame(columns).sort_index() for name, columns in out.items()}
        return dense

    def _baseline_for(self, measure: str, symbols: tuple[str, ...]) -> dict[str, pd.Series]:
        key = (measure, symbols)
        if key not in self._baseline_cache:
            floor = {"oi": 1.0, "pct": 0.02, "share": 0.0005}[measure]
            self._baseline_cache[key] = _baseline(self.dense[measure], [s for s in symbols if s in self.dense[measure]], floor)
        return self._baseline_cache[key]

    def same_month_symbols(self, live_symbol: str) -> tuple[str, ...]:
        """The finished contracts of the same delivery month in the last `years` years, newest first."""
        month = self.universe[live_symbol].month
        same = [s for s in self.finished if s in self.dense["oi"] and self.universe[s].month == month]
        same.sort(key=lambda s: self.universe[s].order, reverse=True)
        return tuple(same[: self.params.years])

    def years_table(self, symbol: str) -> dict | None:
        """The same delivery month in every earlier year, at the live contract's CURRENT number of days before expiry.

        {"label", "dte", "data_date", "rows": [{"label", "oi", "pct_of_peak", "live"}]}: the live contract first (its OI now and its
        share of the peak so far), then each finished contract of that month, newest first: its OI when it was `dte` days from
        expiry and that OI as a share of the contract's highest OI over its whole life. None for a contract without OI."""
        if self._empty or symbol not in self.oi.columns or symbol not in self.universe:
            return None
        series = self.oi[symbol].dropna()
        if series.empty:
            return None
        contract = self.universe[symbol]
        dte = max((contract.expiry - series.index[-1].date()).days, 0)
        peak_now = float(series.max())
        rows = [{"label": contract.label, "oi": float(series.iloc[-1]), "pct_of_peak": float(series.iloc[-1]) / peak_now if peak_now > 0 else np.nan, "live": True}]
        past = [s for s in self.finished if s in self.dense["oi"] and self.universe[s].month == contract.month and s != symbol]
        for s in sorted(past, key=lambda s: self.universe[s].order, reverse=True):
            oi_series = self.oi[s].loc[: self.expiry_actual[s]].dropna()
            peak = float(oi_series.max()) if len(oi_series) else np.nan
            value = self.dense["oi"][s].get(dte, np.nan)
            value = float(value) if value == value else np.nan
            rows.append({"label": self.universe[s].label, "oi": value, "pct_of_peak": value / peak if peak and peak > 0 and value == value else np.nan, "live": False})
        return {"label": contract.label, "dte": dte, "data_date": series.index[-1].date(), "rows": rows}

    def live_symbol_of_month(self, month: int) -> str | None:
        """The nearest contract of this delivery month that has not expired (None if the data holds none)."""
        for s, c in sorted(self.universe.items(), key=lambda item: item[1].order):
            if c.month == month and s in self.oi.columns and c.expiry >= self.last_date.date() and self.oi[s].notna().any():
                return s
        return None

    # -- live contracts --------------------------------------------------

    def live_symbols(self) -> list[str]:
        live = [
            s for s, c in self.universe.items()
            if s in self.oi.columns and c.expiry >= self.last_date.date() and self.oi[s].notna().any()
        ]
        live.sort(key=lambda s: self.universe[s].order)
        return live[: self.params.contracts]

    def _z(self, value: pd.Series, dte: np.ndarray, base: dict[str, pd.Series]) -> pd.Series:
        median = base["median"].reindex(dte).to_numpy()
        scale = base["scale"].reindex(dte).to_numpy()
        return pd.Series((value.to_numpy() - median) / scale, index=value.index)

    def _history(self, symbol: str) -> pd.DataFrame:
        """The last LOOKBACK_ROWS trading days of a live contract: OI, peak so far, share, days to expiry."""
        series = self.oi[symbol].dropna()
        peak = series.cummax()
        frame = pd.DataFrame({"oi": series, "peak": peak, "close": self.close[symbol].reindex(series.index)})
        frame["pct"] = (frame["oi"] / frame["peak"]).where(frame["peak"] > 0)
        frame["share"] = self.share[symbol].reindex(series.index)
        frame = frame.tail(LOOKBACK_ROWS)
        expiry = self.universe[symbol].expiry
        frame["dte"] = [max((expiry - d.date()).days, 0) for d in frame.index]
        frame["declining"] = frame["oi"] < PEAK_DECLINE * frame["peak"]
        return frame

    def _deviation(self, history: pd.DataFrame, symbols: tuple) -> dict:
        """Per-day z against the same-month baseline, the deviation flags, and today's run of deviation."""
        p = self.params
        dte = history["dte"].to_numpy()
        abs_base = self._baseline_for("oi", symbols)
        pct_base = self._baseline_for("pct", symbols)
        z_abs = self._z(history["oi"], dte, abs_base)
        z_pct = self._z(history["pct"], dte, pct_base)
        n = abs_base["n"].reindex(dte).to_numpy()
        declining = history["declining"].to_numpy()
        z = pd.Series(np.where(declining, z_pct, z_abs), index=history.index)
        gap_abs = history["oi"].to_numpy() - abs_base["median"].reindex(dte).to_numpy()
        gap_pct = (history["pct"].to_numpy() - pct_base["median"].reindex(dte).to_numpy()) * history["peak"].to_numpy()
        gap = pd.Series(np.where(declining, gap_pct, gap_abs), index=history.index)
        reliable = len(symbols) >= MIN_BASELINE
        flag = (z.abs() >= p.z_threshold) & (history["oi"] >= p.min_oi) & (gap.abs() >= p.min_change) & (n >= MIN_BASELINE) & reliable
        return {"z": z, "gap": gap, "flag": flag.fillna(False), "n": n, "base_abs": abs_base, "base_pct": pct_base, "reliable": reliable}

    @staticmethod
    def _run_length(flag: pd.Series) -> int:
        run = 0
        for value in reversed(flag.to_numpy()):
            if not value:
                break
            run += 1
        return run

    def _row(self, symbol: str) -> dict | None:
        p = self.params
        history = self._history(symbol)
        if history.empty:
            return None
        b = self.same_month_symbols(symbol)
        dev = self._deviation(history, b)
        contract = self.universe[symbol]
        now = history.iloc[-1]
        dte_now = int(now["dte"])
        declining = bool(now["declining"])
        row = {
            "symbol": symbol, "label": contract.label, "dte": dte_now, "data_date": history.index[-1].date(),
            "oi": float(now["oi"]), "peak_so_far": float(now["peak"]), "pct_of_peak": float(now["pct"]) if np.isfinite(now["pct"]) else np.nan,
            "measure": MEASURE_PERCENT if declining else MEASURE_ABSOLUTE, "close": float(now["close"]) if np.isfinite(now["close"]) else np.nan,
        }
        # Moves: last day, and over the sudden window
        def change(rows_back: int):
            if len(history) <= rows_back:
                return np.nan, np.nan
            old = history.iloc[-1 - rows_back]
            return float(now["oi"] - old["oi"]), float(now["close"] - old["close"]) if np.isfinite(old["close"]) and np.isfinite(now["close"]) else np.nan
        row["d1_oi"], d1_price = change(1)
        row["dn_oi"], dn_price = change(p.sudden_days)
        row["dn_pct"] = row["dn_oi"] / (now["oi"] - row["dn_oi"]) * 100 if np.isfinite(row["dn_oi"]) and (now["oi"] - row["dn_oi"]) > 0 else np.nan
        row["read_1d"], row["read_n"] = oi_price_read(row["d1_oi"], d1_price), oi_price_read(row["dn_oi"], dn_price)
        # The same month in past years, at today's days before expiry
        base_abs, base_pct = dev["base_abs"], dev["base_pct"]
        row["b_n"] = int(base_abs["n"].get(dte_now, 0)) if len(b) else 0
        row["b_median"] = float(base_abs["median"].get(dte_now, np.nan))
        row["b_sd"] = float(base_abs["sd"].get(dte_now, np.nan))
        row["b_min"] = float(base_abs["min"].get(dte_now, np.nan))
        row["b_max"] = float(base_abs["max"].get(dte_now, np.nan))
        row["b_pct_median"] = float(base_pct["median"].get(dte_now, np.nan))
        row["b_z"] = float(dev["z"].iloc[-1])
        row["b_gap"] = float(dev["gap"].iloc[-1])
        row["b_flag"] = bool(dev["flag"].iloc[-1])
        row["b_reliable"] = bool(dev["reliable"])
        row["b_symbols"] = ", ".join(self.universe[s].label for s in b)
        row["outside_b_range"] = bool(np.isfinite(row["b_min"]) and row["b_n"] >= MIN_BASELINE and not (row["b_min"] <= row["oi"] <= row["b_max"]))
        # Share of the product's total OI
        share_base = self._baseline_for("share", b)
        share_now = float(now["share"]) if np.isfinite(now["share"]) else np.nan
        row["share"] = share_now
        row["share_median"] = float(share_base["median"].get(dte_now, np.nan))
        row["share_z"] = float((share_now - share_base["median"].get(dte_now, np.nan)) / share_base["scale"].get(dte_now, np.nan)) if np.isfinite(share_now) else np.nan
        # Run of deviation and status
        run = self._run_length(dev["flag"])
        row["run_days"] = run
        row["since"] = history.index[-run].date() if run else None
        row["status"] = STATUS_NORMAL if run == 0 else STATUS_SUDDEN if run <= p.sudden_days else STATUS_SUSTAINED
        row["direction"] = "" if run == 0 else ("above" if row["b_z"] > 0 else "below")
        row["small"] = bool(row["oi"] < p.min_oi)
        return row

    # -- yesterday's roll -----------------------------------------------

    def _roll(self, symbols: list[str]) -> tuple[pd.DataFrame, dict]:
        if len(self.oi) < 2:
            return pd.DataFrame(), {}
        delta = self.oi.iloc[-1] - self.oi.iloc[-2]
        result = attribute_rolls(delta)
        front = symbols[0] if symbols else None  # the contract closest to expiry
        from_front = result.flows.loc[front] if front in result.flows.index else pd.Series(dtype=float)
        rows = []
        for symbol in symbols:
            change = delta.get(symbol, np.nan)
            rows.append({
                "symbol": symbol, "label": self.universe[symbol].label, "change": float(change) if np.isfinite(change) else np.nan,
                "rolled_out": float(result.rolled_out.get(symbol, 0.0)), "exited": float(result.exited.get(symbol, 0.0)),
                # only what is estimated to have come from the front contract (nothing for the front contract itself)
                "rolled_in": np.nan if symbol == front else float(from_front.get(symbol, 0.0)),
                "fresh": float(result.fresh.get(symbol, 0.0)),
            })
        summary = {
            "date": self.oi.index[-1].date(), "decrease": result.total_decrease, "increase": result.total_increase,
            "rolled": result.rolled, "exited": float(result.exited.sum()), "fresh": float(result.fresh.sum()),
        }
        return pd.DataFrame(rows), summary

    # -- results ---------------------------------------------------------

    def result(self) -> MonitorResult:
        if self._empty:
            return MonitorResult(self.product, error="No OI data downloaded yet")
        symbols = self.live_symbols()
        rows = [r for r in (self._row(s) for s in symbols) if r is not None]
        if not rows:
            return MonitorResult(self.product, self.last_date, error="No live contract has OI data")
        roll, roll_summary = self._roll(symbols)
        total_now = float(self.total.iloc[-1])
        totals = {
            "total_oi": total_now,
            "d1": total_now - float(self.total.iloc[-2]) if len(self.total) > 1 else np.nan,
            "d5": total_now - float(self.total.iloc[-6]) if len(self.total) > 5 else np.nan,
        }
        return MonitorResult(self.product, self.last_date, pd.DataFrame(rows), roll, roll_summary, totals, self.front_check())

    # -- the front contract's last month ---------------------------------

    def _roll_matrices(self) -> dict[str, pd.DataFrame]:
        if not hasattr(self, "_rolls"):
            by_order = {c.order: s for s, c in self.universe.items()}
            next_of = {s: by_order[c.order + 1] for s, c in self.universe.items() if c.order + 1 in by_order}
            self._rolls = daily_roll_matrix(self.oi.diff(), next_of)
        return self._rolls

    def _last_month_series(self, symbol: str, expiry, oi: pd.Series, peak: float) -> dict[str, pd.Series] | None:
        """One contract's last month as shares of `peak`, by calendar days before `expiry` (0..LAST_MONTH):
        OI still open, OI dropped since LAST_MONTH days out, and the estimated roll out (cumulative)."""
        if peak <= 0 or oi.empty:
            return None
        dates = oi.index
        expiry_day = expiry.date() if isinstance(expiry, pd.Timestamp) else expiry
        dte = pd.Index([(expiry_day - d.date()).days for d in dates])
        by_dte = pd.Series(oi.to_numpy(), index=dte)
        by_dte = by_dte[~by_dte.index.duplicated(keep="last")].sort_index(ascending=False)
        days = np.arange(LAST_MONTH, -1, -1)
        asof = by_dte.reindex(np.union1d(by_dte.index.to_numpy(), days)[::-1]).ffill().reindex(days)
        if asof.isna().all():
            return None
        start = asof.loc[LAST_MONTH]
        out = {"remaining": asof / peak, "drop": (start - asof) / peak}
        mats = self._roll_matrices()
        window = pd.Series((dte < LAST_MONTH) & (dte >= 0), index=dates)
        for key in ("rolled_out", "to_next"):
            daily = pd.Series(mats[key][symbol].reindex(dates).to_numpy(), index=dte)[window.to_numpy()]
            daily = daily.groupby(level=0).sum().reindex(np.arange(LAST_MONTH - 1, -1, -1), fill_value=0.0)
            cumulative = daily.cumsum() / peak
            cumulative.loc[LAST_MONTH] = 0.0
            out["rolled" if key == "rolled_out" else key] = cumulative.sort_index()
        return {k: v.sort_index() for k, v in out.items()}

    def _last_month_frames(self) -> dict[str, pd.DataFrame]:
        """The last month of every finished contract, per metric: days before expiry x contract."""
        if not hasattr(self, "_lm"):
            columns: dict[str, dict[str, pd.Series]] = {key: {} for key, _ in FRONT_METRICS}
            for symbol in self.dense["oi"].columns:
                oi = self.oi[symbol].loc[: self.expiry_actual[symbol]].dropna()
                oi = oi[oi >= 0]
                series = self._last_month_series(symbol, self.expiry_actual[symbol], oi, float(oi.max()) if len(oi) else 0.0)
                if series is None:
                    continue
                for key, values in series.items():
                    columns[key][symbol] = values
            self._lm = {key: pd.DataFrame(cols).sort_index() for key, cols in columns.items()}
        return self._lm

    def front_check(self) -> dict | None:
        """Is the contract closest to expiry draining like finished contracts did at this point?

        Returns None without live data. Otherwise a dict: the contract, its days to expiry, one entry per
        metric (now, the usual for the same month in past years, z) and a verdict,
        plus the series for the chart."""
        symbols = self.live_symbols()
        if not symbols:
            return None
        front = symbols[0]
        contract = self.universe[front]
        live = self.oi[front].dropna()
        if live.empty:
            return None
        dte_now = max((contract.expiry - live.index[-1].date()).days, 0)
        out = {"symbol": front, "label": contract.label, "dte": dte_now, "data_date": live.index[-1].date(), "metrics": [], "note": None, "level": "", "verdict": ""}
        if dte_now > LAST_MONTH:
            out["note"] = f"{contract.label} is {dte_now} days from expiry: the check covers the last {LAST_MONTH} days."
            return out
        peak = float(live.max())
        series = self._last_month_series(front, contract.expiry, live, peak)
        if series is None:
            out["note"] = f"Not enough OI history for {contract.label}."
            return out
        frames = self._last_month_frames()
        b = self.same_month_symbols(front)
        thr = self.params.z_threshold
        zs = {}
        for key, label in FRONT_METRICS:
            now = float(series[key].get(dte_now, np.nan))
            entry = {"key": key, "label": label, "now": now}
            cols = [s for s in b if s in frames[key].columns]
            values = frames[key].loc[dte_now, cols].dropna() if cols else pd.Series(dtype=float)
            base = _baseline(frames[key], cols, 0.02)
            scale = float(base["scale"].get(dte_now, np.nan)) if cols else np.nan
            entry.update({
                "b_n": int(len(values)), "b_median": float(values.median()) if len(values) else np.nan,
                "b_min": float(values.min()) if len(values) else np.nan, "b_max": float(values.max()) if len(values) else np.nan,
                "b_z": (now - float(values.median())) / scale if len(values) and np.isfinite(scale) and scale > 0 else np.nan,
            })
            out["metrics"].append(entry)
            zs[key] = entry["b_z"] if entry["b_n"] >= MIN_BASELINE and np.isfinite(entry["b_z"]) else np.nan
        out["level"], out["verdict"] = front_verdict(zs.get("remaining"), zs.get("rolled"), thr)
        out["chart"] = self._front_chart(front, contract, live, peak, b, frames)
        return out

    def _front_chart(self, front, contract, live, peak, b, frames) -> dict:
        rem = frames["remaining"]
        data = {"label": contract.label, "dte": rem.index.to_numpy()}
        running = live.cummax()
        recent = live[[(contract.expiry - d.date()).days <= CHART_DAYS for d in live.index]]
        data["live"] = ([(contract.expiry - d.date()).days for d in recent.index], (recent / running.reindex(recent.index)).to_numpy())
        cols = [s for s in b if s in rem.columns]
        values = rem[cols] if cols else rem.iloc[:, :0]
        data["B"] = {
            "median": values.median(axis=1).to_numpy(), "min": values.min(axis=1).to_numpy(), "max": values.max(axis=1).to_numpy(),
            "labels": [self.universe[s].label for s in cols], "lines": {self.universe[s].label: rem[s].to_numpy() for s in cols},
        }
        return data

    def chart(self, symbol: str) -> dict | None:
        """Everything the chart of one live contract needs: its OI by days before expiry and the same month's usual path."""
        if self._empty or symbol not in self.oi.columns:
            return None
        series = self.oi[symbol].dropna()
        if series.empty:
            return None
        expiry = self.universe[symbol].expiry
        dte = np.array([max((expiry - d.date()).days, 0) for d in series.index])
        b = self.same_month_symbols(symbol)
        horizon = int(max(dte.max(), 1))
        out = {"symbol": symbol, "label": self.universe[symbol].label, "dte": dte, "oi": series.to_numpy(), "dates": series.index, "threshold": self.params.z_threshold}
        base = self._baseline_for("oi", b)
        index = base["median"].index[base["median"].index <= horizon]
        median = base["median"].reindex(index)
        band = self.params.z_threshold * base["scale"].reindex(index)
        out["B"] = {
            "dte": index.to_numpy(), "median": median.to_numpy(), "low": np.maximum(median - band, 0).to_numpy(),
            "high": (median + band).to_numpy(), "labels": [self.universe[s].label for s in b],
            "lines": {self.universe[s].label: self.dense["oi"][s].reindex(index).to_numpy() for s in b if s in self.dense["oi"]},
        }
        return out


def front_verdict(z_remaining: float | None, z_rolled: float | None, threshold: float) -> tuple[str, str]:
    """(level, text) from how much OI is still open and how much has rolled, each in z against the usual.

    More open than usual with less rolled than usual is the squeeze-like case; the other combinations are named
    so the reason for a surprising number is visible."""
    def known(z):
        return z is not None and np.isfinite(z)

    if not known(z_remaining):
        return "", "Not enough finished contracts to compare with."
    more_open, less_open = z_remaining >= threshold, z_remaining <= -threshold
    rolled_low = known(z_rolled) and z_rolled <= -threshold
    rolled_high = known(z_rolled) and z_rolled >= threshold
    if more_open and rolled_low:
        return "ALERT", "More OI is still open AND less has rolled than usual at this point: possible squeeze or delivery interest. Check the next contract's build and the spread."
    if more_open:
        return "WATCH", "More OI is still open than usual, although the roll is not unusually low: positions are being held towards expiry."
    if less_open and rolled_high:
        return "INFO", "OI has left earlier than usual and more of it has rolled than usual: the roll is running ahead of schedule."
    if less_open:
        return "INFO", "OI has dropped more than usual, but not through rolling: more positions have been closed out."
    if rolled_high:
        return "INFO", "More has rolled than usual, with the OI still open about normal."
    if rolled_low:
        return "WATCH", "Less has rolled than usual, with the OI still open about normal: the roll is lagging."
    return "OK", "In line with past contracts at this point: the roll is progressing as usual."
