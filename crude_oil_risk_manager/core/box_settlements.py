"""Daily settlements of a CL-BZ outbox built from its CL and BZ legs, for when /common/dailymarketdata/ has none for the box code.

A same-month outbox settles as the CL outright minus the BZ outright of the month (core.box): WTCLCOF27 = CLF27 - BZZF27. Rows are the
API's ({"qhcode", "datetime", "oi", "close", "volume"}); the built rows have no open interest (a box has none of its own) and take the
smaller of the two legs' volumes (a box cannot trade more than its thinner leg). Days where either leg has no settlement are left out.
No I/O: the caller supplies `fetch`, which asks the same date range for some codes.
"""

import pandas as pd

from core.box import outbox_components


def fill_missing_box_rows(fetch, symbols: list[str], got: list[dict]) -> list[dict]:
    """Rows for the outbox symbols in `symbols` that `got` has none for, built from their legs' rows.

    `fetch(codes)` returns the rows of those codes (CL and BZ outrights, API form). Nothing is fetched when every outbox came back or
    none of the missing ones is a same-month outbox."""
    have = {row.get("qhcode") for row in got}
    missing = {symbol: parts for symbol in symbols if symbol not in have and (parts := outbox_components(symbol)) is not None}
    if not missing:
        return []
    rows = fetch(sorted({code for parts in missing.values() for code in parts}))
    by_code: dict[str, pd.DataFrame] = {}
    if rows:
        frame = pd.DataFrame(rows)
        frame = frame.assign(
            close=pd.to_numeric(frame.get("close"), errors="coerce"), volume=pd.to_numeric(frame.get("volume"), errors="coerce"),
            day=pd.to_datetime(frame["datetime"]).dt.tz_localize(None).dt.normalize(),
        ).dropna(subset=["close"])
        by_code = {code: part.drop_duplicates("day", keep="last").set_index("day") for code, part in frame.groupby("qhcode")}
    out: list[dict] = []
    for symbol, (cl_code, bz_code) in missing.items():
        cl, bz = by_code.get(cl_code), by_code.get(bz_code)
        if cl is None or bz is None:
            continue
        for day in cl.index.intersection(bz.index):
            cl_volume, bz_volume = cl.at[day, "volume"], bz.at[day, "volume"]
            volume = float(min(cl_volume, bz_volume)) if pd.notna(cl_volume) and pd.notna(bz_volume) else None
            out.append({"qhcode": symbol, "datetime": day.isoformat(), "oi": None, "close": float(cl.at[day, "close"] - bz.at[day, "close"]), "volume": volume})
    return out
