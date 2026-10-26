"""Trade journal: the facts about each trade that the archive keeps and the analysis will group by.

Per structure, stored as one JSON document (repository table `trade_journal`):

    entries / exits  the same snapshot for EVERY entry and exit trade, by trade id ({trade_id: snapshot}); an Archive row built
                     from one exit (core.archive, FIFO) reads the snapshot of the entry it took lots from and of that exit
    entry / exit   the market when the trade was first entered / last exited (core.journal_snapshot):
                     prices   {"CO": .., "CL": .., "GO": ..}  last traded price of each product's front outright
                     move_day / move_min   Bullish | Bearish | Stalemate for the front outright of the structure's product,
                                           from the day's candle and from the last one-minute candle
                     curve    {"CL": .., "BRN": .., "G": ..}  Contango | Backwardation | Contango - Backwardation | ...
                     curve_family   which curve was read (the structure's leg type: fly, then spread, then outright)
                     atr      average true range of the whole structure (points of the structure, per lot)
    ideas          the trade ideas picked from the list in Settings (any number)
    payups         ticks paid up before the fill: {"entry": {"total": x, "legs": {symbol: x}}, "exit": {...}}; typed by hand
    conviction     1 to 5, how sure you were at entry
    exit_reason    why the trade was closed (EXIT_REASONS)
    mistakes       tags for what went wrong (MISTAKES), set when reviewing the trade
    excursion      the best and worst the open position was worth in $ (excursion): max favourable / adverse excursion

Risk:reward is not stored: it is worked out from the stop and target prices on the entry trades (risk_reward).
The functions here are pure (no I/O) apart from the repository helpers at the bottom.
"""

import math
from collections import OrderedDict

from core.models import Trade, TradeEventType

BULLISH, BEARISH, STALEMATE = "Bullish", "Bearish", "Stalemate"
CONTANGO, BACKWARDATION = "Contango", "Backwardation"
CONTANGO_BACKWARDATION, BACKWARDATION_CONTANGO = "Contango - Backwardation", "Backwardation - Contango"
FLAT = "Flat"

STALEMATE_RATIO = 5.0  # high-low at least this many times open-close is a stalemate
BIG_MOVE_TICKS = 50.0  # ... unless open-close is itself this large (then the direction counts)
CURVE_POINTS = 6  # the front contracts a curve shape is read from
# journal product code -> label the journal shows (CO is Brent's API code, GO gasoil's)
PRODUCT_LABELS = OrderedDict([("BRN", "CO"), ("CL", "CL"), ("G", "GO")])

EXIT_REASONS = ["Target hit", "Stop hit", "Time stop", "Dissolved against other trades", "Discretionary", "Expiry / roll", "Other"]
MISTAKES = ["Chased the price", "Ignored my rules", "Sized too big", "Moved or removed the stop", "Exited too early", "Held too long", "Poor fill", "No clear idea"]
CONVICTIONS = [1, 2, 3, 4, 5]

_ENTRY_EVENTS = (TradeEventType.TRADE, TradeEventType.ADD)
_EXIT_EVENTS = (TradeEventType.PARTIAL_EXIT, TradeEventType.STRUCTURE_PARTIAL_EXIT, TradeEventType.FULL_EXIT, TradeEventType.DISSOLVE)


def _finite(*values) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


# ----------------------------------------------------------------------
# Market movement
# ----------------------------------------------------------------------


def classify_move(open_, high, low, close, tick_size: float, big_ticks: float = BIG_MOVE_TICKS, ratio: float = STALEMATE_RATIO) -> str | None:
    """Bullish / Bearish / Stalemate from one candle (None if the candle is incomplete).

    Price up (close above open) is Bullish, down Bearish. A candle whose high-low is `ratio` times its open-close or more
    went nowhere for its range: Stalemate. But when open-close is itself `big_ticks` ticks or more it is a real move, so the
    direction is taken whatever the range.
    """
    if not _finite(open_, high, low, close) or not tick_size or tick_size <= 0:
        return None
    body = close - open_
    spread = high - low
    if abs(body) / tick_size >= big_ticks:
        return BULLISH if body > 0 else BEARISH
    if body == 0 or spread >= ratio * abs(body):
        return STALEMATE
    return BULLISH if body > 0 else BEARISH


# ----------------------------------------------------------------------
# Curve shape
# ----------------------------------------------------------------------


def curve_shape(values, tol_fraction: float = 0.05, min_tol: float = 1e-9) -> str | None:
    """Shape of a curve from its front values (front first): Contango, Backwardation, or one followed by the other.

    Contango = value rising with maturity, Backwardation = falling. A rise that turns into a fall is "Contango -
    Backwardation" (and the reverse "Backwardation - Contango"). Moves smaller than `tol_fraction` of the curve's range, or
    than `min_tol` (pass the contract's tick size), are noise: they do not count as a turn, and a curve that moves less than
    `min_tol` altogether is Flat. With a zig-zag the bigger turn decides. Flat if nothing moves.
    Needs at least 3 values (None otherwise). The values may be a curve of outrights, spreads or flies: the same rule is
    applied to whichever series is given.
    """
    values = [float(v) for v in values if v is not None and _finite(v)]
    if len(values) < 3:
        return None
    span = max(values) - min(values)
    if span <= min_tol:
        return FLAT
    tol = max(span * tol_fraction, min_tol)
    first, last = values[0], values[-1]
    peak = max(range(len(values)), key=lambda i: values[i])
    trough = min(range(len(values)), key=lambda i: values[i])
    # how far an interior extreme sticks out beyond both ends = the size of a turn
    rise_then_fall = values[peak] - max(first, last) if 0 < peak < len(values) - 1 else 0.0
    fall_then_rise = min(first, last) - values[trough] if 0 < trough < len(values) - 1 else 0.0
    if rise_then_fall > tol and rise_then_fall >= fall_then_rise:
        return CONTANGO_BACKWARDATION
    if fall_then_rise > tol:
        return BACKWARDATION_CONTANGO
    net = last - first
    if abs(net) <= tol:
        return FLAT
    return CONTANGO if net > 0 else BACKWARDATION


def curve_values(outrights: list[float | None], family: str) -> list[float | None]:
    """The first CURVE_POINTS values of the outright, spread (M1-M2, M2-M3 ...) or fly (M1-2M2+M3 ...) curve from outright prices."""
    out: list[float | None] = []
    legs = {"outright": 1, "spread": 2, "fly": 3}[family]
    weights = {"outright": (1,), "spread": (1, -1), "fly": (1, -2, 1)}[family]
    for i in range(CURVE_POINTS):
        window = outrights[i:i + legs]
        if len(window) < legs or any(v is None for v in window):
            out.append(None)
        else:
            out.append(sum(w * v for w, v in zip(weights, window)))
    return out


def leg_family(leg_symbols: list[str], symbol_type) -> str:
    """Which curve to read for a structure: fly if any leg is a fly, else spread if any is a spread, else outright."""
    kinds = {symbol_type(s) for s in leg_symbols}
    if "fly" in kinds:
        return "fly"
    if "spread" in kinds:
        return "spread"
    return "outright"


# ----------------------------------------------------------------------
# Risk : reward
# ----------------------------------------------------------------------


def risk_reward(trades: list[Trade]) -> dict | None:
    """Average entry, stop and target of the entry trades and the risk:reward they give; None without both a stop and a target.

    The stop and target are the structure prices typed with each entry trade; with several they are simply averaged (the
    entry too, weighted by lots). risk = |entry - stop|, reward = |target - entry|, ratio = reward / risk (2.0 reads 1 : 2).
    """
    entries = [t for t in trades if t.event_type in _ENTRY_EVENTS]
    stops = [t.stop_loss_price for t in entries if t.stop_loss_price is not None]
    targets = [t.target_price for t in entries if t.target_price is not None]
    lots = sum(t.lots for t in entries)
    if not entries or not stops or not targets or lots <= 0:
        return None
    entry = sum(t.price * t.lots for t in entries) / lots
    stop, target = sum(stops) / len(stops), sum(targets) / len(targets)
    risk, reward = abs(entry - stop), abs(target - entry)
    return {"entry": entry, "stop": stop, "target": target, "risk": risk, "reward": reward, "ratio": reward / risk if risk > 0 else None}


def format_risk_reward(value: dict | None) -> str:
    """'1 : 2.5' (risk : reward), or an empty dash when there is no stop and target."""
    if not value or value.get("ratio") is None:
        return "—"
    return f"1 : {value['ratio']:.2f}"


# ----------------------------------------------------------------------
# The journal document
# ----------------------------------------------------------------------


def empty_journal() -> dict:
    return {
        "entry": None, "exit": None, "entries": {}, "exits": {}, "ideas": [], "payups": {"entry": {"total": None, "legs": {}}, "exit": {"total": None, "legs": {}}},
        "conviction": None, "exit_reason": None, "mistakes": [], "excursion": None,
    }


def with_defaults(journal: dict | None) -> dict:
    base = empty_journal()
    if journal:
        base.update({k: v for k, v in journal.items() if k in base})
        payups = base["payups"]
        for kind in ("entry", "exit"):
            payups.setdefault(kind, {"total": None, "legs": {}})
            payups[kind].setdefault("total", None)
            payups[kind].setdefault("legs", {})
    return base


def entry_and_exit_times(trades: list[Trade]):
    """(first entry time, last exit time) of a structure's trades; None where there is no such trade."""
    entries = [t.timestamp for t in trades if t.event_type in _ENTRY_EVENTS]
    exits = [t.timestamp for t in trades if t.event_type in _EXIT_EVENTS]
    return (min(entries) if entries else None), (max(exits) if exits else None)


def add_payups(journal: dict, kind: str, total, leg_payups: dict | None = None) -> dict:
    """Add typed payups (ticks) to the entry or exit totals; blank adds nothing. Legs are optional and add per leg symbol."""
    if kind not in ("entry", "exit"):
        raise ValueError(f"kind must be entry or exit, got {kind!r}")
    journal = with_defaults(journal)
    slot = journal["payups"][kind]
    if total not in (None, ""):
        if not _finite(float(total)):
            raise ValueError("payups must be a number")
        slot["total"] = (slot["total"] or 0.0) + float(total)
    for symbol, value in (leg_payups or {}).items():
        if value in (None, ""):
            continue
        slot["legs"][symbol] = slot["legs"].get(symbol, 0.0) + float(value)
    return journal


def set_payups(journal: dict, kind: str, total, leg_payups: dict | None = None) -> dict:
    """Replace the entry or exit payups (the Archive editor): blank clears a figure."""
    journal = with_defaults(journal)
    journal["payups"][kind] = {
        "total": None if total in (None, "") else float(total),
        "legs": {s: float(v) for s, v in (leg_payups or {}).items() if v not in (None, "")},
    }
    return journal


def clean_ideas(ideas, allowed: list[str] | None = None) -> list[str]:
    """The picked trade ideas without blanks or repeats, in the order given; if `allowed` is given, only those."""
    seen, out = set(), []
    for idea in ideas or []:
        text = str(idea).strip()
        if not text or text in seen or (allowed is not None and text not in allowed):
            continue
        seen.add(text)
        out.append(text)
    return out


def parse_idea_list(text: str | None) -> list[str]:
    """The Settings text box (one idea per line, or comma separated) as a clean list."""
    parts = []
    for line in str(text or "").replace(",", "\n").splitlines():
        parts.append(line)
    return clean_ideas(parts)


# ----------------------------------------------------------------------
# Conviction, exit reason, mistakes
# ----------------------------------------------------------------------


def clean_conviction(value) -> int | None:
    """1 to 5, or None for blank. Raises ValueError for anything else."""
    if value in (None, ""):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        raise ValueError("conviction must be a number from 1 to 5") from None
    if number not in CONVICTIONS:
        raise ValueError("conviction must be a number from 1 to 5")
    return number


def clean_exit_reason(value, allowed: list[str] | None = None) -> str | None:
    """The exit reason if it is on the list (EXIT_REASONS unless `allowed` is given); blank is None."""
    allowed = EXIT_REASONS if allowed is None else allowed
    if value in (None, ""):
        return None
    if value not in allowed:
        raise ValueError(f"exit reason must be one of: {', '.join(allowed)}")
    return value


def clean_mistakes(values, allowed: list[str] | None = None) -> list[str]:
    allowed = MISTAKES if allowed is None else allowed
    return [m for m in clean_ideas(values) if m in allowed]


# ----------------------------------------------------------------------
# Excursion: how far the trade went for and against you
# ----------------------------------------------------------------------


def excursion(structure, trades: list[Trade], prices, intraday: bool = False) -> dict | None:
    """Maximum favourable and adverse excursion of a trade, in $ at the size entered, from the structure's price path.

    `prices` is the structure's price (as quoted, summed from its legs) indexed by time. The position is marked at every point
    from the first entry to the last exit at the lots-weighted entry price; the best mark is the MFE and the worst the MAE (the
    position is flat at the start, so MFE >= 0 >= MAE). With DAILY closes (`intraday=False`) a spike inside a day is not in them:
    read them as a floor on how far the trade really went. With an intraday path (`intraday=True`, minute / 5-minute / hourly
    closes) the window runs from the entry time to the exit time. None if there are no entries or no prices in the window.
    """
    from core.trade_entry import structure_pnl  # late: trade_entry imports a lot

    entries = [t for t in trades if t.event_type in _ENTRY_EVENTS]
    first, last = entry_and_exit_times(trades)
    if not entries or first is None or prices is None or len(prices) == 0:
        return None
    lots = sum(t.lots for t in entries)
    entry = sum(t.price * t.lots for t in entries) / lots
    end_time = last or first
    if intraday:
        window = [(i, p) for i, p in prices.items() if first <= _aware(i) <= end_time]
        basis = "intraday closes at the size entered"
    else:
        window = [(d, p) for d, p in _dated(prices) if first.date() <= d <= end_time.date()]
        basis = "daily closes at the size entered"
    if not window:
        return None
    direction = entries[0].direction
    path = [structure_pnl(structure, entry, float(p), lots, direction) for _d, p in window]
    return {
        "mfe": round(max(0.0, max(path)), 2), "mae": round(min(0.0, min(path)), 2), "points": len(window), "days": len({_day(d) for d, _p in window}),
        "lots": lots, "basis": basis,
    }


def _aware(stamp):
    from datetime import timezone

    moment = stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _day(stamp):
    return stamp.date() if hasattr(stamp, "date") and callable(stamp.date) else stamp


def _dated(prices):
    for index, value in prices.items():
        yield (index.date() if hasattr(index, "date") else index), value
