"""Analysis of the trade journal: headline measures, an equity curve, and results grouped by any journal column.

Input is the archive's rows (core.archive.build_archive_rows), so the analysis covers exactly the trades the Archive's filters
leave. Every figure is PER TRADE and NET of transaction costs (the archive's realized PnL). R is a trade's PnL divided by the
dollars it risked at the entry stop (|entry - stop| x lots x multiplier): a trade without a stop has no R and is left out of the
R figures only.

A group is flagged "low" when it has fewer than MIN_SAMPLE trades: with so few trades a win rate or an expectancy is mostly luck.
Nothing here touches the database.
"""

import math
from collections import OrderedDict
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from core.journal import PRODUCT_LABELS

MIN_SAMPLE = 15
_LONDON = ZoneInfo("Europe/London")
NONE_LABEL = "(none)"

DIMENSIONS = OrderedDict([
    ("type", "Structure type"), ("product", "Product"), ("idea", "Trade idea"), ("curve", "Curve shape at entry"),
    ("price_band", "Outright price range at entry"), ("day_move", "Day move at entry"), ("min_move", "1-minute move at entry"),
    ("stop_atr", "Stop distance in ATRs"), ("rr_band", "Planned risk : reward"), ("conviction", "Conviction"),
    ("exit_reason", "Exit reason"), ("mistake", "Mistake tag"), ("hour", "Entry hour (London)"), ("weekday", "Entry weekday"),
    ("held", "Days held"), ("payup_entry", "Payups in entry"), ("payup_exit", "Payups in exit"),
    ("vol", "Volatility at entry vs normal"), ("volume", "Volume at entry vs normal"), ("spread", "Bid-ask at entry"),
    ("brent_wti", "Brent-WTI at entry"), ("crack", "Gasoil crack at entry"), ("kink", "Entered against the curve (z-score)"),
])
_SNAPSHOT_PRICE = {"CO": "e_px_co", "CL": "e_px_cl", "GO": "e_px_go"}
_SNAPSHOT_CURVE = {"CL": "e_curve_cl", "BRN": "e_curve_brn", "G": "e_curve_go"}


def _num(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _own_product(row: dict) -> str | None:
    snap = (row.get("snapshots") or {}).get("entry") or {}
    product = snap.get("product")
    if product in PRODUCT_LABELS:
        return product
    for p in row.get("products") or []:
        if p in PRODUCT_LABELS:
            return p
    return None


def _local(iso: str | None):
    if not iso:
        return None
    try:
        stamp = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return stamp.astimezone(_LONDON) if stamp.tzinfo else stamp


def _bucket(value, edges: list[float], labels: list[str]):
    if value is None:
        return None
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def trade_table(rows: list[dict]) -> pd.DataFrame:
    """One line per trade with the columns the groupings use (None where the journal has nothing)."""
    records = []
    for row in rows:
        product = _own_product(row)
        label = PRODUCT_LABELS.get(product)
        entry_local = _local(row.get("entry_at"))
        rr = row.get("rr_detail") or {}
        risk_points = _num(rr.get("risk")) if rr else None
        atr = _num(row.get("e_atr"))
        held = row.get("days_held")
        payup_entry, payup_exit = _num(row.get("payup_entry")), _num(row.get("payup_exit"))
        records.append({
            "structure_id": row.get("structure_id"), "name": row.get("name"), "type": row.get("type"),
            "products": row.get("products") or [], "product": "+".join(sorted(row.get("products") or [])) or None,
            "own_product": product, "own_label": label,
            "entry_price_own": _num(row.get(_SNAPSHOT_PRICE.get(label, ""), None)) if label else None,
            "curve": row.get(_SNAPSHOT_CURVE.get(product, "")) if product else None,
            "day_move": row.get("e_move_day"), "min_move": row.get("e_move_min"),
            "pnl": _num(row.get("realized_pnl")) or 0.0, "cost": _num(row.get("transaction_costs")) or 0.0,
            "r": _num(row.get("r_multiple")), "rr": _num(row.get("rr")), "risk_usd": _num(row.get("risk_usd")),
            "stop_atr": _bucket(risk_points / atr, [0.5, 1.0, 2.0], ["under 0.5 ATR", "0.5 to 1 ATR", "1 to 2 ATR", "2 ATR or more"]) if risk_points and atr else None,
            "rr_band": _bucket(_num(row.get("rr")), [1.0, 2.0, 3.0], ["under 1 : 1", "1 : 1 to 1 : 2", "1 : 2 to 1 : 3", "1 : 3 or more"]),
            "conviction": None if row.get("conviction") is None else f"{int(row['conviction'])}",
            "exit_reason": row.get("exit_reason"),
            "ideas": list(row.get("ideas") or []), "mistakes": list(row.get("mistakes") or []),
            "hour": f"{entry_local.hour:02d}:00" if entry_local else None,
            "weekday": entry_local.strftime("%a") if entry_local else None,
            "held": _bucket(_num(held), [1, 2, 4, 8], ["same day", "1 day", "2 to 3 days", "4 to 7 days", "8 days or more"]),
            "payup_entry": None if payup_entry is None else _bucket(payup_entry, [1, 2], ["0", "1", "2 or more"]),
            "payup_exit": None if payup_exit is None else _bucket(payup_exit, [1, 2], ["0", "1", "2 or more"]),
            "payups": (payup_entry or 0.0) + (payup_exit or 0.0) if payup_entry is not None or payup_exit is not None else None,
            "vol": _bucket(_num(row.get("e_vol_ratio")), [0.7, 1.3], ["quiet (under 0.7x)", "normal", "fast (over 1.3x)"]),
            "volume": _bucket(_num(row.get("e_volume_ratio")), [0.7, 1.3], ["thin (under 0.7x)", "normal", "heavy (over 1.3x)"]),
            "spread": _bucket(_num(row.get("e_spread_ticks")), [1.01, 2.01], ["tight (1 tick)", "2 ticks", "wide (over 2 ticks)"]),
            "brent_wti_value": _num(row.get("e_brent_wti")), "crack_value": _num(row.get("e_crack")),
            "kink": _bucket(_num(row.get("kink_edge")), [-2.0, 2.0], ["against the curve (z under -2)", "near fair (z within 2)", "with the curve (z over 2)"]),
            "exit_at": row.get("exit_at"), "days_held": _num(held), "mfe": _num(row.get("mfe")), "mae": _num(row.get("mae")),
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["price_band"] = _price_bands(frame)
    frame["brent_wti"] = _thirds(frame["brent_wti_value"], "Brent-WTI")
    frame["crack"] = _thirds(frame["crack_value"], "crack")
    return frame


def _thirds(values: pd.Series, name: str) -> pd.Series:
    """Low / middle / high third of a market figure across the trades (one group if there are too few to split)."""
    out = pd.Series([None] * len(values), index=values.index, dtype=object)
    known = values.dropna()
    if known.empty:
        return out
    if len(known) < 6:
        out[known.index] = f"{name} (too few trades to split)"
        return out
    low, high = known.quantile([1 / 3, 2 / 3])
    for index, value in known.items():
        out[index] = f"{name} low (under {low:g})" if value < low else f"{name} middle ({low:g} to {high:g})" if value < high else f"{name} high ({high:g} and over)"
    return out


def _price_bands(frame: pd.DataFrame) -> pd.Series:
    """Outright price range at entry: thirds of the trades' own product (a CL price and a gasoil price are not comparable)."""
    out = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    for product, part in frame.groupby("own_label"):
        prices = part["entry_price_own"].dropna()
        if prices.empty:
            continue
        if len(prices) < 6:
            out[prices.index] = f"{product} (too few trades to split)"
            continue
        low, high = prices.quantile([1 / 3, 2 / 3])
        for index, value in prices.items():
            out[index] = (f"{product} low (under {low:g})" if value < low else f"{product} middle ({low:g} to {high:g})" if value < high else f"{product} high ({high:g} and over)")
    return out


# ----------------------------------------------------------------------
# Measures
# ----------------------------------------------------------------------


def _profit_factor(pnl: pd.Series) -> float | None:
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    if losses <= 0:
        return None if gains <= 0 else math.inf
    return float(gains / losses)


def measures(part: pd.DataFrame) -> dict:
    """The figures of one set of trades."""
    pnl = part["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) else None
    avg_loss = float(losses.mean()) if len(losses) else None
    r = part["r"].dropna()
    return {
        "trades": int(len(part)), "win_rate": float(len(wins) / len(part)) if len(part) else None,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "payoff": avg_win / abs(avg_loss) if avg_win is not None and avg_loss else None,
        "expectancy": float(pnl.mean()) if len(part) else None,
        "expectancy_r": float(r.mean()) if len(r) else None, "r_trades": int(len(r)),
        "profit_factor": _profit_factor(pnl), "total": float(pnl.sum()), "costs": float(part["cost"].sum()),
        "avg_payups": float(part["payups"].dropna().mean()) if part["payups"].notna().any() else None,
        "avg_days": float(part["days_held"].dropna().mean()) if part["days_held"].notna().any() else None,
    }


def equity_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """Cumulative PnL in exit order with the drawdown from the running peak (the start, at 0, counts as a peak)."""
    if frame.empty:
        return pd.DataFrame(columns=["exit_at", "name", "pnl", "cum", "drawdown"])
    ordered = frame.assign(_t=pd.to_datetime(frame["exit_at"], utc=True, errors="coerce")).sort_values("_t", kind="stable")
    cum = ordered["pnl"].cumsum()
    peak = cum.cummax().clip(lower=0.0)
    return pd.DataFrame({"exit_at": ordered["_t"].to_numpy(), "name": ordered["name"].to_numpy(), "pnl": ordered["pnl"].to_numpy(),
                         "cum": cum.to_numpy(), "drawdown": (cum - peak).to_numpy()})


def streaks(frame: pd.DataFrame) -> tuple[int, int]:
    """(longest winning run, longest losing run) in exit order; a flat trade ends both."""
    best_win = best_loss = win = loss = 0
    for pnl in equity_curve(frame)["pnl"]:
        win = win + 1 if pnl > 0 else 0
        loss = loss + 1 if pnl < 0 else 0
        best_win, best_loss = max(best_win, win), max(best_loss, loss)
    return best_win, best_loss


def headline(frame: pd.DataFrame) -> dict:
    """The whole book: the measures plus the worst drawdown and the longest streaks."""
    if frame.empty:
        return {"trades": 0, "win_rate": None, "avg_win": None, "avg_loss": None, "payoff": None, "expectancy": None, "expectancy_r": None,
                "r_trades": 0, "profit_factor": None, "total": 0.0, "costs": 0.0, "avg_payups": None, "avg_days": None,
                "max_drawdown": 0.0, "longest_win": 0, "longest_loss": 0}
    curve = equity_curve(frame)
    win, loss = streaks(frame)
    return {**measures(frame), "max_drawdown": float(curve["drawdown"].min()), "longest_win": win, "longest_loss": loss}


# ----------------------------------------------------------------------
# Grouping
# ----------------------------------------------------------------------


def _groups(frame: pd.DataFrame, dimension: str):
    """[(group label, trades)] for a dimension; the idea and mistake tags count a trade in every tag it has."""
    if dimension in ("idea", "mistake"):
        column = "ideas" if dimension == "idea" else "mistakes"
        buckets: dict[str, list[int]] = {}
        for index, tags in frame[column].items():
            for tag in (tags or [NONE_LABEL]):
                buckets.setdefault(tag, []).append(index)
        return [(tag, frame.loc[ids]) for tag, ids in buckets.items()]
    column = {"idea": "ideas"}.get(dimension, dimension)
    if column not in frame:
        return []
    labelled = frame[column].where(frame[column].notna(), NONE_LABEL)
    return [(label, frame[labelled == label]) for label in labelled.drop_duplicates()]


def group_table(frame: pd.DataFrame, dimension: str, min_sample: int = MIN_SAMPLE) -> pd.DataFrame:
    """Results grouped by one dimension (see DIMENSIONS), best expectancy first."""
    columns = ["Group", "Trades", "Win rate", "Avg win", "Avg loss", "Payoff", "Expectancy $", "Expectancy R", "R trades", "Profit factor", "Total $", "Avg payups", "Sample"]
    if frame.empty or dimension not in DIMENSIONS:
        return pd.DataFrame(columns=columns)
    rows = []
    for label, part in _groups(frame, dimension):
        m = measures(part)
        rows.append({
            "Group": label, "Trades": m["trades"], "Win rate": m["win_rate"], "Avg win": m["avg_win"], "Avg loss": m["avg_loss"],
            "Payoff": m["payoff"], "Expectancy $": m["expectancy"], "Expectancy R": m["expectancy_r"], "R trades": m["r_trades"],
            "Profit factor": m["profit_factor"], "Total $": m["total"], "Avg payups": m["avg_payups"],
            "Sample": "ok" if m["trades"] >= min_sample else f"low ({m['trades']} of {min_sample})",
        })
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values("Expectancy $", ascending=False, kind="stable").reset_index(drop=True)
