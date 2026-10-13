"""ATR: the standard true range, averaged over a rolling window. Pure maths, no I/O.

TRUE RANGE of a day = max(high - low, |high - previous close|, |low - previous close|): the whole distance
the price travelled, including a gap from yesterday's close. ATR = the simple average of the last `period`
true ranges (default 14).

IDLE DAYS. A thin contract (a far month, a fly) often does not trade: the candle is a flat line at yesterday's
close with no volume. Counting it as a true range of zero drags the ATR down for no market reason, so with
`skip_idle` such days are left out: they neither enter the average nor become the "previous close" (the next
active day's range then includes the whole move since the last real trade).

ATR as a percentage of price is only meaningful for an outright; a spread or fly can sit near zero or be
negative, so it is left blank for them.
"""

import numpy as np
import pandas as pd


def true_range(frame: pd.DataFrame, skip_idle: bool = True) -> pd.Series:
    """True range per day for one instrument (columns high, low, close, volume; indexed by date)."""
    data = frame.sort_index()
    if skip_idle and "volume" in data:
        data = data[~(data["volume"].fillna(1) <= 0)]
    previous = data["close"].shift(1)
    spread = data["high"] - data["low"]
    gap_up, gap_down = (data["high"] - previous).abs(), (data["low"] - previous).abs()
    tr = pd.concat([spread, gap_up, gap_down], axis=1).max(axis=1, skipna=True)
    tr[previous.isna()] = spread[previous.isna()]  # the first day has no previous close
    return tr.where(data["high"].notna() & data["low"].notna())


def atr_frame(frame: pd.DataFrame, period: int = 14, skip_idle: bool = True, percent: bool = True) -> pd.DataFrame:
    """Per active day: close, true range, ATR (points) and ATR as % of the close (outrights only)."""
    data = frame.sort_index()
    if skip_idle and "volume" in data:
        data = data[~(data["volume"].fillna(1) <= 0)]
    tr = true_range(data, skip_idle=False)  # idle days are already removed
    atr = tr.rolling(period, min_periods=period).mean()
    out = pd.DataFrame({"close": data["close"], "tr": tr, "atr": atr})
    out["atr_pct"] = np.where(percent & (out["close"].abs() > 0), out["atr"] / out["close"].abs() * 100.0, np.nan)
    return out
