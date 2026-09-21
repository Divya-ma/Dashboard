"""ATR-based shock scenarios for the VaR & Scenarios tab.

The shock size of each instrument is its previous trading day's True Range,
TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|), read from the local
daily Parquet. Every instrument uses its own TR; nothing is a uniform shock.
The PnL impact of a shock is shock * the instrument's dollar weight
(core.var.leg_dollar_weights), i.e. sum(shock * ratio * side * lots * multiplier)
over the legs on that instrument.
"""

from datetime import date, datetime, timezone

import pandas as pd

from core.data_loader import DataLoader
from core.exceptions import CrudeOilRiskError

KIND_INSTRUMENT = "instrument"
KIND_PORTFOLIO = "portfolio"


def previous_day_true_range(ohlc: pd.DataFrame | None, today: date | None = None) -> float | None:
    """True Range of the most recent completed daily candle, or None if it cannot be computed.

    A candle dated today (UTC) is still forming and is ignored. Needs the candle before it
    for the previous close.
    """
    if ohlc is None or ohlc.empty:
        return None
    today = today or datetime.now(timezone.utc).date()
    completed = ohlc[ohlc.index.date < today]
    if len(completed) < 2:
        return None
    last, previous = completed.iloc[-1], completed.iloc[-2]
    return float(
        max(
            last["high"] - last["low"],
            abs(last["high"] - previous["close"]),
            abs(last["low"] - previous["close"]),
        )
    )


def load_previous_day_atr(symbols: list[str], data_loader: DataLoader, today: date | None = None) -> dict[str, float | None]:
    """{symbol: previous-day True Range or None}, read fresh from the Parquet on every call."""
    result: dict[str, float | None] = {}
    for symbol in symbols:
        try:
            result[symbol] = previous_day_true_range(data_loader.load_daily_ohlc(symbol), today)
        except (CrudeOilRiskError, OSError, KeyError, ValueError):
            result[symbol] = None
    return result


def build_scenarios(weights: dict[str, float], atr: dict[str, float | None]) -> list[dict]:
    """Scenario rows: +/-1 ATR per instrument (others unchanged), then all instruments together.

    Each row: {"kind", "symbol", "sign", "shock", "pnl_impact", "included", "excluded"}.
    An instrument without an ATR gets shock=None and pnl_impact=None and is left out of
    the portfolio-wide rows (`excluded` lists it).
    """
    rows: list[dict] = []
    usable = {symbol: atr.get(symbol) for symbol in weights if atr.get(symbol) is not None}
    for symbol in sorted(weights):
        shock = atr.get(symbol)
        for sign in (1, -1):
            rows.append(
                {
                    "kind": KIND_INSTRUMENT, "symbol": symbol, "sign": sign, "shock": shock,
                    "pnl_impact": None if shock is None else sign * shock * weights[symbol],
                    "included": [symbol] if shock is not None else [], "excluded": [] if shock is not None else [symbol],
                }
            )
    excluded = sorted(set(weights) - set(usable))
    for sign in (1, -1):
        rows.append(
            {
                "kind": KIND_PORTFOLIO, "symbol": None, "sign": sign, "shock": None,
                "pnl_impact": sign * sum(shock * weights[symbol] for symbol, shock in usable.items()) if usable else None,
                "included": sorted(usable), "excluded": excluded,
            }
        )
    return rows
