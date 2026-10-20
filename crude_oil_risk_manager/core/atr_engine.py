"""ATR views: how a contract's ATR changes as it nears expiry, how it compares with the same month in past
years, and how the ATR of the later contracts relates to the front one (for hedge ratios).

ATR is the standard true range averaged over `period` days (core.atr_calc). Everything is per product and per
family (outright / spread / fly). As with the OI monitor, comparisons are between the SAME delivery month in
different years (a December with past Decembers), at the same number of calendar days before expiry; there is
no pooling of different months. An instrument's expiry for this purpose is its front leg's: for a finished one
the last day that outright traded, for a live one the rule-based date (approximate).

  live table     today's ATR against the same month's usual at the same days to expiry (z as in the OI monitor)
  cycle chart    one contract's ATR path against the same month's past years
  month tables   ATR N days before expiry by delivery month (median, SD), in points or % of price, and as a
                 multiple of the contract's own ATR when it became the front contract (30 days out)
  hedge table    for the days in history when the front contract was a given delivery month: the ratio of the
                 k-th contract's ATR to the front's ATR (median, spread, and today), plus the minimum-variance
                 hedge ratio (beta of daily price changes on the front's) and the correlation, since the ATR ratio
                 alone ignores how closely the two move together
"""

from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from core.atr_calc import atr_frame
from core.atr_data import instrument_universe
from core.hedge import slope_and_corr
from core.oi_data import EXPIRED_AFTER_DAYS
from core.oi_monitor import MIN_BASELINE, _baseline
from core.user_settings import parse_number

KEY_ATR_PARAMS = "atr_params"
FRONT_DAYS = 30  # a contract "becomes the front" about this many days before expiry
DTE_MARKS = ((360, "12m"), (300, "10m"), (270, "9m"), (240, "8m"), (210, "7m"), (180, "6m"), (150, "5m"), (120, "4m"), (90, "3m"), (60, "2m"), (30, "1m"), (14, "2w"), (7, "1w"))
GAP_TOLERANCE = 3  # days beyond an instrument's last candle its ATR is still carried forward


@dataclass
class AtrParams:
    period: int = 14  # days in the ATR average
    contracts: int = 15  # the next N live contracts
    years: int = 5  # the same month in the last N years
    z_threshold: float = 2.5
    skip_idle: bool = True  # leave out days with no trading

    def key(self) -> tuple:
        return tuple(asdict(self).values())


PARAM_RULES = {  # name: (label, minimum, maximum, integer)
    "period": ("ATR period (days)", 2, 100, True),
    "contracts": ("Contracts to watch", 1, 36, True),
    "years": ("Years of the same month", 2, 10, True),
    "z_threshold": ("Deviation threshold (z)", 0.5, 10.0, False),
}


def validate_params(raw: dict) -> AtrParams:
    defaults = AtrParams()
    values = {}
    for name, (label, minimum, maximum, integer) in PARAM_RULES.items():
        value = raw.get(name)
        values[name] = getattr(defaults, name) if value in (None, "") else parse_number(value, label, minimum=minimum, maximum=maximum, integer=integer)
    values["skip_idle"] = bool(raw.get("skip_idle", defaults.skip_idle))
    return AtrParams(**values)


def load_params(repository) -> AtrParams:
    try:
        raw = repository.get_setting(KEY_ATR_PARAMS, None)
        return validate_params(raw) if isinstance(raw, dict) else AtrParams()
    except (ValueError, TypeError):
        return AtrParams()


def save_params(repository, params: AtrParams) -> None:
    if load_params(repository) != params or repository.get_setting(KEY_ATR_PARAMS, None) is None:
        repository.set_setting(KEY_ATR_PARAMS, asdict(params))


MIN_PAIR_DAYS = 20  # fewer shared days than this and the hedge ratio is not trustworthy


@dataclass
class PairHedge:
    """How to hedge a position in contract A with contract B (see pair_hedge)."""

    a: str
    b: str
    days: int  # shared trading days (daily changes) used for the full-history figures
    window_days: int  # shared days actually inside the lookback window
    atr_a: float
    atr_b: float
    atr_date_a: date
    atr_date_b: date
    atr_ratio: float  # lots of B per lot of A that give both the same ATR (points)
    beta_window: float  # minimum-variance lots of B per lot of A, lookback window
    corr_window: float
    beta_all: float  # same over every shared day
    corr_all: float
    residual_window: float  # daily SD (points of A) left after hedging, window
    unhedged_window: float  # daily SD (points of A) of A alone, window
    first_day: date | None
    last_day: date | None


def pair_hedge(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str, window: int) -> PairHedge | None:
    """Hedge a position in A with B. Frames are indexed by date with columns close and atr (core.atr_calc.atr_frame).

    Daily changes are the changes of the close between the days BOTH traded (a day either skipped is left out for
    both, so the two changes always cover the same span). The minimum-variance hedge is the beta of A's changes on
    B's: hold -beta lots of B per lot of A (long A, short beta of B). Its effectiveness is the correlation squared.
    The ATR ratio is the cruder alternative that only matches the size of the daily moves, not their direction."""
    both = pd.concat([a["close"].rename("a"), b["close"].rename("b")], axis=1, join="inner").dropna()
    atr_a, atr_b = a["atr"].dropna(), b["atr"].dropna()
    if len(both) < 3 or atr_a.empty or atr_b.empty:
        return None
    change = both.diff().dropna()
    tail = change.iloc[-max(int(window), 2):]
    beta_w, corr_w = slope_and_corr(tail["a"].to_numpy(), tail["b"].to_numpy())
    beta_all, corr_all = slope_and_corr(change["a"].to_numpy(), change["b"].to_numpy())
    unhedged = float(tail["a"].std(ddof=1)) if len(tail) > 1 else np.nan
    residual = float((tail["a"] - beta_w * tail["b"]).std(ddof=1)) if len(tail) > 1 and np.isfinite(beta_w) else np.nan
    last_a, last_b = float(atr_a.iloc[-1]), float(atr_b.iloc[-1])
    return PairHedge(
        a_name, b_name, int(len(change)), int(len(tail)), last_a, last_b, atr_a.index[-1].date(), atr_b.index[-1].date(),
        last_a / last_b if last_b > 0 else np.nan, beta_w, corr_w, beta_all, corr_all, residual, unhedged,
        change.index[0].date(), change.index[-1].date(),
    )


class AtrEngine:
    """The ATR views of one product, built from its stored OHLC frame. Construct once per data version and parameters."""

    def __init__(self, frame: pd.DataFrame, product: str, today: date, history_days: int, params: AtrParams):
        self.product, self.today, self.params, self.history_days = product, today, params, int(history_days)
        self._empty = frame.empty
        if self._empty:
            return
        self.universe = {i.symbol: i for i in instrument_universe(product, today)}
        self.atr: dict[str, pd.DataFrame] = {}
        for symbol, part in frame.groupby("symbol"):
            if symbol in self.universe and len(part) >= params.period + 2:
                self.atr[symbol] = atr_frame(
                    part.set_index("date")[["open", "high", "low", "close", "volume"]], params.period, params.skip_idle,
                    percent=self.universe[symbol].family == "outright",
                )
        last = frame.groupby("symbol")["date"].max()
        self.last_date = frame["date"].max()
        self.expiry: dict[str, date] = {}
        for symbol, instrument in self.universe.items():
            front = instrument.first.symbol  # a spread or fly ends with its front leg
            finished = instrument.expiry < today - timedelta(days=EXPIRED_AFTER_DAYS)
            actual = last.get(front)
            self.expiry[symbol] = actual.date() if finished and actual is not None else instrument.expiry
        self._dense_cache: dict[tuple, pd.DataFrame] = {}

    def pair_frame(self, symbol: str) -> pd.DataFrame | None:
        """Close and ATR per active day of one instrument (None when it has too little history)."""
        frame = None if self._empty else self.atr.get(symbol)
        return None if frame is None else frame[["close", "atr"]]

    def all_live_symbols(self) -> list[tuple[str, str, str]]:
        """(family, symbol, label) of every live instrument of the page's 'contracts to watch', by family then expiry."""
        return [(f, s, self.universe[s].label) for f in ("outright", "spread", "fly") if not self._empty for s in self.live_symbols(f)]

    # -- finished and live instruments ------------------------------------

    def finished(self, family: str) -> list[str]:
        return [
            s for s, i in self.universe.items()
            if i.family == family and s in self.atr and i.expiry < self.today - timedelta(days=EXPIRED_AFTER_DAYS)
        ]

    def live_symbols(self, family: str) -> list[str]:
        live = [
            s for s, i in self.universe.items()
            if i.family == family and s in self.atr and self.expiry[s] >= self.last_date.date() and not self.atr[s]["atr"].dropna().empty
        ]
        live.sort(key=lambda s: self.universe[s].order)
        return live[: self.params.contracts]

    def same_month(self, symbol: str) -> tuple[str, ...]:
        """The finished instruments of the same family and delivery month in the last `years` years, newest first."""
        instrument = self.universe[symbol]
        same = [s for s in self.finished(instrument.family) if self.universe[s].first.month == instrument.first.month]
        same.sort(key=lambda s: self.universe[s].order, reverse=True)
        return tuple(same[: self.params.years])

    # -- ATR by calendar days before expiry --------------------------------

    def dense(self, family: str, column: str = "atr") -> pd.DataFrame:
        """For every finished instrument of a family, its ATR (or ATR %) at each calendar day before expiry 0..history
        (as of the last candle at or before that point; blank before it started and after it ended)."""
        key = (family, column)
        if key not in self._dense_cache:
            index = np.arange(self.history_days, -1, -1)
            columns = {}
            for symbol in self.finished(family):
                series = self.atr[symbol][column].dropna()
                if series.empty:
                    continue
                dte = pd.Index([(self.expiry[symbol] - d.date()).days for d in series.index])
                by_dte = pd.Series(series.to_numpy(), index=dte)
                by_dte = by_dte[~by_dte.index.duplicated(keep="last")]
                full = by_dte.reindex(np.union1d(by_dte.index.to_numpy(), index)[::-1]).ffill().reindex(index)
                full[full.index < by_dte.index.min() - GAP_TOLERANCE] = np.nan
                columns[symbol] = full
            self._dense_cache[key] = pd.DataFrame(columns).sort_index() if columns else pd.DataFrame(index=np.arange(0, self.history_days + 1))
        return self._dense_cache[key]

    def _usual(self, family: str, symbols: tuple[str, ...], column: str = "atr") -> dict[str, pd.Series]:
        dense = self.dense(family, column)
        return _baseline(dense, [s for s in symbols if s in dense], 0.001)

    # -- the live table ---------------------------------------------------

    def live_table(self, family: str) -> pd.DataFrame:
        """One row per live instrument: ATR, today's range, and how today's ATR compares with the same month's usual."""
        symbols = self.live_symbols(family)
        rows = []
        front_atr = None
        for symbol in symbols:
            frame = self.atr[symbol]
            now = frame.dropna(subset=["atr"]).iloc[-1]
            date_now = frame.dropna(subset=["atr"]).index[-1]
            dte = max((self.expiry[symbol] - date_now.date()).days, 0)
            usual = self._usual(family, self.same_month(symbol))
            n = int(usual["n"].get(dte, 0)) if len(self.same_month(symbol)) else 0
            median, scale = float(usual["median"].get(dte, np.nan)), float(usual["scale"].get(dte, np.nan))
            z = (float(now["atr"]) - median) / scale if n >= MIN_BASELINE and np.isfinite(median) and scale > 0 else np.nan
            front_atr = front_atr if front_atr is not None else float(now["atr"])
            rows.append({
                "symbol": symbol, "label": self.universe[symbol].label, "dte": dte, "date": date_now.date(),
                "close": float(now["close"]), "atr": float(now["atr"]), "atr_pct": float(now["atr_pct"]) if np.isfinite(now["atr_pct"]) else np.nan,
                "tr": float(now["tr"]), "tr_vs_atr": float(now["tr"] / now["atr"]) if now["atr"] > 0 else np.nan,
                "vs_front": float(now["atr"]) / front_atr if front_atr else np.nan,
                "usual_median": median, "usual_min": float(usual["min"].get(dte, np.nan)), "usual_max": float(usual["max"].get(dte, np.nan)),
                "usual_n": n, "z": z,
                "status": "" if not np.isfinite(z) or abs(z) < self.params.z_threshold else ("HIGH" if z > 0 else "LOW"),
            })
        return pd.DataFrame(rows)

    # -- the chart of one contract -----------------------------------------

    def chart(self, symbol: str, unit: str = "atr") -> dict | None:
        """The ATR path of one live instrument against the same month's past years."""
        if self._empty or symbol not in self.atr:
            return None
        column = "atr_pct" if unit == "pct" and self.universe[symbol].family == "outright" else "atr"
        series = self.atr[symbol][column].dropna()
        if series.empty:
            return None
        family = self.universe[symbol].family
        dte = np.array([max((self.expiry[symbol] - d.date()).days, 0) for d in series.index])
        same = self.same_month(symbol)
        base = self._usual(family, same, column)
        horizon = int(max(dte.max(), 1))
        index = base["median"].index[base["median"].index <= horizon]
        median = base["median"].reindex(index)
        band = self.params.z_threshold * base["scale"].reindex(index)
        dense = self.dense(family, column)
        return {
            "symbol": symbol, "label": self.universe[symbol].label, "unit": "% of price" if column == "atr_pct" else "points", "dte": dte,
            "atr": series.to_numpy(), "dates": series.index, "threshold": self.params.z_threshold,
            "usual": {
                "dte": index.to_numpy(), "median": median.to_numpy(), "low": np.maximum(median - band, 0).to_numpy(), "high": (median + band).to_numpy(),
                "labels": [self.universe[s].label for s in same],
                "lines": {self.universe[s].label: dense[s].reindex(index).to_numpy() for s in same if s in dense},
            },
        }

    # -- tables by delivery month -----------------------------------------

    def month_table(self, family: str, unit: str = "atr", relative: bool = False) -> pd.DataFrame:
        """A row per delivery month (Jan to Dec), a column per DTE mark: the median and SD, across that month's
        finished instruments, of the ATR then (points or % of price) or, if `relative`, of the ATR as a multiple of
        its own ATR FRONT_DAYS before expiry (when it became the front contract)."""
        column = "atr_pct" if unit == "pct" and family == "outright" else "atr"
        dense = self.dense(family, column)
        rows = []
        if dense.empty:
            return pd.DataFrame(columns=["month"])
        for month in range(1, 13):
            symbols = [s for s in dense.columns if self.universe[s].first.month == month]
            if not symbols:
                continue
            row = {"month": month}
            for days, label in DTE_MARKS:
                values = dense.loc[days, symbols] if days in dense.index else pd.Series(dtype=float)
                if relative:
                    values = values / dense.loc[FRONT_DAYS, symbols]
                values = values.replace([np.inf, -np.inf], np.nan).dropna()
                row[f"median_{label}"] = float(values.median()) if len(values) else np.nan
                row[f"sd_{label}"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    # -- ratios to the front contract, for hedges ---------------------------

    def hedge_table(self, front_month: int) -> pd.DataFrame:
        """For the days in history when the FRONT outright was of `front_month`: the k-th contract's ATR as a
        multiple of the front's (median, 25th/75th percentile, min, max, and today), and the minimum-variance
        hedge ratio (units of the front per unit of the k-th contract: the beta of daily price changes on the
        front's) with the correlation. Outrights only."""
        if self._empty:
            return pd.DataFrame()
        outrights = [s for s, i in self.universe.items() if i.family == "outright" and s in self.atr]
        atr = pd.DataFrame({s: self.atr[s]["atr"] for s in outrights}).sort_index()
        close = pd.DataFrame({s: self.atr[s]["close"] for s in outrights}).reindex(atr.index)
        change = close.diff()
        order = {s: self.universe[s].order for s in outrights}
        expiry = {s: pd.Timestamp(self.expiry[s]) for s in outrights}
        by_order = {order[s]: s for s in outrights}
        symbols = list(atr.columns)
        orders = np.array([order[s] for s in symbols])
        expiries = np.array([expiry[s] for s in symbols], dtype="datetime64[ns]")
        have = atr.notna().to_numpy()
        fronts: dict[pd.Timestamp, str] = {}
        for i, day in enumerate(atr.index):
            alive = have[i] & (expiries >= np.datetime64(day))
            if alive.any():
                fronts[day] = symbols[int(np.argmin(np.where(alive, orders, np.iinfo(np.int64).max)))]
        days = [d for d, f in fronts.items() if self.universe[f].first.month == front_month]
        n_positions = self.params.contracts
        rows = []
        # today: the live front and the contracts after it
        live_front = fronts.get(atr.index[-1])
        for k in range(2, n_positions + 1):
            ratios, dx, dy = [], [], []
            for day in days:
                front = fronts[day]
                other = by_order.get(order[front] + k - 1)
                if other is None:
                    continue
                a_front, a_other = atr.at[day, front], atr.at[day, other]
                if np.isfinite(a_front) and np.isfinite(a_other) and a_front > 0:
                    ratios.append(a_other / a_front)
                c_front, c_other = change.at[day, front], change.at[day, other]
                if np.isfinite(c_front) and np.isfinite(c_other):
                    dx.append(c_front)
                    dy.append(c_other)
            now = np.nan
            label = f"+{k - 1}"
            if live_front is not None:
                other = by_order.get(order[live_front] + k - 1)
                if other is not None and np.isfinite(atr.iloc[-1].get(other, np.nan)) and atr.iloc[-1][live_front] > 0:
                    now = float(atr.iloc[-1][other] / atr.iloc[-1][live_front])
                    label = self.universe[other].label
            ratios = np.array(ratios)
            beta, corr = slope_and_corr(dy, dx, min_obs=6)  # y = the k-th contract's changes, x = the front's
            rows.append({
                "position": k, "label": label, "days": int(len(ratios)), "now": now,
                "median": float(np.median(ratios)) if len(ratios) else np.nan,
                "p25": float(np.percentile(ratios, 25)) if len(ratios) else np.nan, "p75": float(np.percentile(ratios, 75)) if len(ratios) else np.nan,
                "min": float(ratios.min()) if len(ratios) else np.nan, "max": float(ratios.max()) if len(ratios) else np.nan,
                "hedge": beta, "corr": corr,
            })
        return pd.DataFrame(rows)
