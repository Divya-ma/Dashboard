"""Candles of a CL-BZ box built from the candles of its CL and BZ legs, for when /ohlc/ has none for the box code.

A box is the CL structure minus the BZ structure of the same months (core.box), so its candle is the difference of the two
legs' candles: WTCLCOF27 = CLF27 - BZZF27 and CLBZZXF27-G27 = CLF27-G27 - BZZF27-G27 (a flybox from the two flies, and so on).

The difference is taken CLOSE TO CLOSE on every candle either leg printed, each leg's last close carried forward, as
core.churn_data.derive_dfly_bars does for a d-fly: the open is the previous close and the high and low are the larger and smaller
of the two. The extremes inside a candle are therefore not seen (the legs' highs do not happen together), which makes a daily range
a little too small and a target a little harder to reach than on a real instrument. Volume is the smaller of the two legs' (the box
cannot trade more than its thinner leg).

Candles are the /ohlc/ dicts ({"product", "time" in milliseconds, "open", "high", "low", "close", "volume"}), in and out. No I/O: the
caller supplies `fetch`, which asks for the candles of some symbols over the date range it is already downloading.
"""

import pandas as pd

from core.box import BOX_API_CODES, BZ_API, CL_LEG, OUTBOX_API, SPREADBOX

SPREADBOX_API = BOX_API_CODES[SPREADBOX]
MAX_PER_REQUEST = 50  # instruments per /ohlc/ request


def leg_symbols(box_symbol: str) -> tuple[str, str] | None:
    """(CL symbol, BZ symbol) a box API symbol is made of, or None if it is not a same-month box.

    "WTCLCOF27" -> ("CLF27", "BZZF27"); "CLBZZXF27-G27-H27" -> ("CLF27-G27-H27", "BZZF27-G27-H27"). A cross-month outbox
    (WTCLCOF27-G27: CL Jan - BZ Feb) is the difference of two different outrights and is not built here."""
    for code in (OUTBOX_API, SPREADBOX_API):
        if box_symbol.startswith(code):
            tail = box_symbol[len(code):]
            if code == OUTBOX_API and "-" in tail:
                return None
            return (f"{CL_LEG}{tail}", f"{BZ_API}{tail}") if tail else None
    return None


def _frame(candles: list[dict], symbol: str) -> pd.DataFrame:
    rows = [c for c in candles if c.get("product") == symbol]
    if not rows:
        return pd.DataFrame(columns=["time", "close", "volume"])
    frame = pd.DataFrame(rows)
    out = pd.DataFrame({
        "time": pd.to_numeric(frame["time"], errors="coerce"),
        "close": pd.to_numeric(frame["close"], errors="coerce"),
        "volume": pd.to_numeric(frame.get("volume"), errors="coerce"),
    })
    return out.dropna(subset=["time", "close"]).drop_duplicates("time", keep="last").set_index("time").sort_index()


def derive_box_candles(box_symbol: str, cl: pd.DataFrame, bz: pd.DataFrame, daily: bool) -> list[dict]:
    """The box's candles from the two legs' frames (index = time in ms, columns close and volume)."""
    if cl.empty or bz.empty:
        return []
    index = cl.index.union(bz.index)
    if daily:  # one candle a day: only days both legs printed
        index = cl.index.intersection(bz.index)
        if index.empty:
            return []
    close = (cl["close"].reindex(index).ffill() - bz["close"].reindex(index).ffill()).dropna()
    if close.empty:
        return []
    previous = close.shift(1).fillna(close)
    volume = pd.concat([cl["volume"].reindex(close.index).fillna(0), bz["volume"].reindex(close.index).fillna(0)], axis=1).min(axis=1)
    return [
        {"product": box_symbol, "time": int(t), "open": float(o), "high": float(max(o, c)), "low": float(min(o, c)), "close": float(c), "volume": float(v)}
        for t, o, c, v in zip(close.index, previous.to_numpy(), close.to_numpy(), volume.to_numpy())
    ]


def fill_missing_boxes(fetch, symbols: list[str], got: list[dict], daily: bool) -> list[dict]:
    """Candles for the box symbols in `symbols` that `got` has none for, built from their legs.

    `fetch(symbols)` returns the candles of those symbols for the same range as `got` (one request). Nothing is fetched when every
    box came back or none can be built from legs."""
    have = {c.get("product") for c in got}
    missing = {symbol: legs for symbol in symbols if symbol not in have and (legs := leg_symbols(symbol)) is not None}
    if not missing:
        return []
    wanted = sorted({leg for legs in missing.values() for leg in legs})
    leg_candles = [candle for i in range(0, len(wanted), MAX_PER_REQUEST) for candle in fetch(wanted[i:i + MAX_PER_REQUEST])]
    out: list[dict] = []
    for symbol, (cl_symbol, bz_symbol) in missing.items():
        out += derive_box_candles(symbol, _frame(leg_candles, cl_symbol), _frame(leg_candles, bz_symbol), daily)
    return out
