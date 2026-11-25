"""Morning scan: where to churn today when nothing is at your usual level.

The Churn table ranks every contract by what history says; this scan answers "what can I do right now in a quiet market".
It keeps the rows whose side MAKES money (score above 0), that oscillate (the live price level was reached again and then
the target, at least `min_osc` times in the chosen days) and whose range over the last week is tight (the low-to-high of
the week is at most `max_range_atr` daily ATRs; a contract with no ATR is kept and marked), then gives each one levels to
churn from, so a contract does not have to be at a level you chose in advance:

  Yesterday's close   the last stored five-minute close before today (a candle close, not the exchange settlement)
  Week mid            half way between the week's low and high
  Week low / high     the edges of the week's range

and says which of them is nearest the live price and how far (signed ticks: positive = the level is above the price). The
best hours are the Churn table's own (the London hours whose entries earned most). Tightest range first, then best score.
The other side of a contract that loses on one side is its own row (a FLIP is simply a row with a high score on the
other side), so nothing is flipped here.
"""

import math

import numpy as np
import pandas as pd

DEFAULT_MIN_OSC = 3
DEFAULT_MAX_RANGE_ATR = 3.0
AT_LEVEL_TICKS = 2.0  # within this many ticks of a level counts as being at it

COLUMNS = ("contract", "direction", "price", "range_atr", "osc", "score", "level", "level_price", "distance", "status", "hours", "verdict")
LEVEL_NAMES = ("Yesterday's close", "Week mid", "Week low", "Week high")


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float, np.floating, np.integer)) and math.isfinite(value)


def levels_of(prior_close, week_low, week_high) -> dict[str, float]:
    """The churn levels that can be built from the stored candles (a missing figure leaves its level out)."""
    out: dict[str, float] = {}
    if _finite(prior_close):
        out["Yesterday's close"] = float(prior_close)
    if _finite(week_low) and _finite(week_high):
        out["Week mid"] = (float(week_low) + float(week_high)) / 2.0
    if _finite(week_low):
        out["Week low"] = float(week_low)
    if _finite(week_high):
        out["Week high"] = float(week_high)
    return out


def nearest_level(levels: dict[str, float], price, tick) -> tuple[str, float, float] | None:
    """(name, level, signed distance in ticks: level minus price) of the level closest to the price; None without a price or levels."""
    if not levels or not _finite(price) or not _finite(tick) or tick <= 0:
        return None
    name = min(levels, key=lambda n: abs(levels[n] - price))
    return name, levels[name], (levels[name] - price) / tick


def build_scan(table: pd.DataFrame, extras: dict[str, dict], min_osc: float = DEFAULT_MIN_OSC, max_range_atr: float = DEFAULT_MAX_RANGE_ATR) -> pd.DataFrame:
    """The scan rows from ChurnService.live_table's rows. `extras` maps a symbol to {"tick": ..., "prior_close": ...}."""
    rows = []
    if table is None or table.empty:
        return pd.DataFrame(columns=COLUMNS)
    for r in table.itertuples():
        if not r.score > 0:
            continue
        osc = r.osc if _finite(r.osc) else np.nan
        if np.isfinite(osc) and osc < min_osc:
            continue  # a live price that did not come back often enough; no price (nan) cannot be judged, so the row stays
        width = r.week_high - r.week_low if _finite(r.week_low) and _finite(r.week_high) else np.nan
        range_atr = width / r.atr if np.isfinite(width) and _finite(r.atr) and r.atr > 0 else np.nan
        if np.isfinite(range_atr) and range_atr > max_range_atr:
            continue
        extra = extras.get(r.symbol, {})
        levels = levels_of(extra.get("prior_close"), r.week_low, r.week_high)
        near = nearest_level(levels, r.price, extra.get("tick"))
        if near is None:
            level, level_price, distance, status = "", np.nan, np.nan, "No live price"
        else:
            level, level_price, distance = near
            status = "AT LEVEL" if abs(distance) <= AT_LEVEL_TICKS else f"{abs(distance):.0f} ticks {'above' if distance > 0 else 'below'}"
        rows.append({
            "contract": f"{r.product} {r.label}", "family": r.family, "direction": r.direction, "price": r.price, "range_atr": range_atr, "osc": osc, "score": r.score,
            "level": level, "level_price": level_price, "distance": distance, "status": status, "hours": getattr(r, "hours", "") or "", "verdict": r.verdict,
        })
    out = pd.DataFrame(rows, columns=[*COLUMNS, "family"])
    if out.empty:
        return out
    out["_tight"] = out["range_atr"].fillna(np.inf)  # an unknown range sorts last
    return out.sort_values(["_tight", "score"], ascending=[True, False]).drop(columns="_tight").reset_index(drop=True)
