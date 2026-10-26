"""Archive rows, filters and summary statistics for closed structures (read-only).

Nothing here touches the database: callers pass CLOSED structures and their trades.
Realized PnL is the sum of the structure's saved exit trades (the value persisted at
exit, never recomputed), net of every transaction cost booked on the structure
(entry/add and exit) — see core.pnl.calculate_net_realized_pnl. Dates come from the
trades (entry) and the structure's closed_at (exit); prices from core.structure_view.
"""

from datetime import date, datetime

from core.journal import PRODUCT_LABELS, format_risk_reward, risk_reward, with_defaults
from core.models import Structure, StructureStatus, Trade, TradeEventType
from core.pnl import calculate_net_realized_pnl, calculate_structure_transaction_costs
from core.structure_view import structure_entry_price, structure_exit_price

RESULT_WIN, RESULT_LOSS, RESULT_FLAT = "Win", "Loss", "Flat"
PNL_ALL, PNL_WINNERS, PNL_LOSERS = "all", "winners", "losers"

_ENTRY_EVENTS = {TradeEventType.TRADE, TradeEventType.ADD}
_EXIT_EVENTS = {TradeEventType.PARTIAL_EXIT, TradeEventType.STRUCTURE_PARTIAL_EXIT, TradeEventType.FULL_EXIT, TradeEventType.DISSOLVE}


def _leg_side(leg) -> str:
    """Effective side of a leg: the ratio's own sign flipped by a sold position."""
    return "buy" if (leg.ratio > 0) == (leg.direction == "buy") else "sell"


def _legs_detail(structure: Structure, entered_lots: float | None) -> list[dict]:
    """Legs with symbol, side and lots. Lots are the entered lots x |ratio| (legs are zeroed at exit)."""
    details = []
    for leg in structure.legs:
        lots = (entered_lots if entered_lots is not None else leg.lots) * abs(leg.ratio)
        details.append({"symbol": leg.contract.symbol, "side": _leg_side(leg), "ratio": leg.ratio, "lots": lots})
    return details


def _legs_summary(details: list[dict]) -> str:
    """'CLZ25 B1, CLH26 S1', or a dash for a structure with no legs."""
    if not details:
        return "—"
    return ", ".join(f"{d['symbol']} {'B' if d['side'] == 'buy' else 'S'}{d['lots']:g}" for d in details)


def _date_or_none(value: datetime | None) -> date | None:
    return value.date() if value else None


def result_for(pnl: float) -> str:
    return RESULT_WIN if pnl > 0 else RESULT_LOSS if pnl < 0 else RESULT_FLAT


# the journal columns: id prefix "e_" / "x_" for the entry / exit snapshot
CURVE_PRODUCTS = ("CL", "BRN", "G")  # the curves are listed CL, BRN, GO
SNAPSHOT_FIELDS = (
    ("px_co", "prices", "CO"), ("px_cl", "prices", "CL"), ("px_go", "prices", "GO"),
    ("move_day", "move_day", None), ("move_min", "move_min", None),
    ("curve_cl", "curve", "CL"), ("curve_brn", "curve", "BRN"), ("curve_go", "curve", "G"),
    ("atr", "atr", None),
    ("vol_ratio", "vol_ratio", None), ("volume_ratio", "volume_ratio", None), ("spread_ticks", "spread_ticks", None),
    ("brent_wti", "brent_wti", None), ("crack", "crack", None), ("kink_z", "kink_z", None),
)


def journal_columns(journal: dict | None, trades: list[Trade]) -> dict:
    """The journal's part of an archive row: risk:reward, the entry and exit snapshots, trade ideas and payups.

    Every figure the journal does not have is None. Payups are ticks typed by hand (entry and exit, whole structure; the
    per-leg figures are kept in `payup_legs`)."""
    journal = with_defaults(journal)
    rr = risk_reward(trades)
    out = {
        "rr": rr["ratio"] if rr else None, "rr_text": format_risk_reward(rr), "rr_detail": rr,
        "ideas": journal["ideas"], "ideas_text": ", ".join(journal["ideas"]),
        "payup_entry": journal["payups"]["entry"]["total"], "payup_exit": journal["payups"]["exit"]["total"],
        "payup_legs": {"entry": journal["payups"]["entry"]["legs"], "exit": journal["payups"]["exit"]["legs"]},
        "snapshots": {"entry": journal["entry"], "exit": journal["exit"]},
        "conviction": journal["conviction"], "exit_reason": journal["exit_reason"],
        "mistakes": journal["mistakes"], "mistakes_text": ", ".join(journal["mistakes"]),
        "mfe": (journal["excursion"] or {}).get("mfe"), "mae": (journal["excursion"] or {}).get("mae"),  # the intraday figures if there are any
    }
    exc = journal["excursion"] or {}
    daily = exc.get("daily") or (exc if str(exc.get("basis", "")).startswith("daily") else None) or {}
    intraday = exc.get("intraday") or {}
    out.update(mfe_d=daily.get("mfe"), mae_d=daily.get("mae"), mfe_i=intraday.get("mfe"), mae_i=intraday.get("mae"))
    for prefix, kind in (("e_", "entry"), ("x_", "exit")):
        snap = journal[kind] or {}
        for column, key, sub_key in SNAPSHOT_FIELDS:
            value = snap.get(key)
            out[prefix + column] = (value or {}).get(sub_key) if sub_key is not None else value
    return out


def build_archive_rows(
    structures: list[Structure], trades_by_structure: dict[str, list[Trade]], journals: dict[str, dict] | None = None,
) -> list[dict]:
    """One row per closed structure, most recently closed first.

    Missing entry/exit dates (data anomalies) stay None; days_held is then None too.
    A structure without legs still gets a row (legs shown as a dash). `journals` (by structure id) adds the trade-journal
    columns (see journal_columns).
    """
    rows = []
    for structure in structures:
        trades = trades_by_structure.get(structure.structure_id, [])
        entries = sorted((t for t in trades if t.event_type in _ENTRY_EVENTS), key=lambda t: t.timestamp)
        exits = sorted((t for t in trades if t.event_type in _EXIT_EVENTS), key=lambda t: t.timestamp)
        entry_date = _date_or_none(entries[0].timestamp if entries else None)
        exit_date = _date_or_none(structure.closed_at or (exits[-1].timestamp if exits else None))
        entered_lots = sum(t.lots for t in entries) if entries else None
        pnl = calculate_net_realized_pnl(trades)
        transaction_costs = calculate_structure_transaction_costs(trades)
        legs = _legs_detail(structure, entered_lots)
        extra = journal_columns((journals or {}).get(structure.structure_id), trades)
        entry_at = entries[0].timestamp.isoformat() if entries else None
        exit_at = (structure.closed_at or (exits[-1].timestamp if exits else None))
        rr = extra["rr_detail"]
        multiplier = structure.legs[0].contract.multiplier if structure.legs else 1000.0
        risk_usd = rr["risk"] * (entered_lots or 0.0) * multiplier if rr and entered_lots else None
        z = extra.get("e_kink_z")
        long_curve = bool(structure.legs) and ((structure.legs[0].direction == "buy") == (structure.legs[0].ratio > 0))
        extra["kink_edge"] = None if z is None else (-z if long_curve else z)  # a long structure that was cheap (z < 0), or a short one that was rich
        extra["risk_usd"] = risk_usd if risk_usd and risk_usd > 0 else None
        extra["r_multiple"] = pnl / extra["risk_usd"] if extra["risk_usd"] else None
        rows.append(
            {
                "structure_id": structure.structure_id,
                "name": structure.name,
                "type": structure.structure_type.label,
                "legs_summary": _legs_summary(legs),
                "legs": legs,
                "symbols": [leg["symbol"] for leg in legs],
                "entry_date": entry_date.isoformat() if entry_date else None,
                "exit_date": exit_date.isoformat() if exit_date else None,
                "days_held": (exit_date - entry_date).days if entry_date and exit_date else None,
                "entry_price": structure_entry_price(structure) if structure.legs else None,
                "exit_price": structure_exit_price(structure, trades),
                "lots": entered_lots,
                "realized_pnl": pnl,
                "transaction_costs": transaction_costs,
                "result": result_for(pnl),
                "notes": structure.notes or "",
                "products": list(structure.products),
                "entry_at": entry_at,
                "exit_at": exit_at.isoformat() if exit_at else None,
                **extra,
            }
        )
    rows.sort(key=lambda r: r["exit_date"] or "", reverse=True)
    return rows


def filter_archive_rows(
    rows: list[dict],
    start: date | None = None,
    end: date | None = None,
    structure_type: str | None = None,
    symbol_contains: str | None = None,
    pnl_filter: str | None = PNL_ALL,
) -> list[dict]:
    """Apply the archive filters together; each is optional.

    The date range is on the exit date (inclusive; rows with no exit date are dropped when a
    range is set). "Winners" keeps PnL >= 0 and "Losers" PnL <= 0, so a flat trade shows up
    under both and is easy to find.
    """
    needle = (symbol_contains or "").strip().lower()
    kept = []
    for row in rows:
        if start or end:
            exit_date = date.fromisoformat(row["exit_date"]) if row["exit_date"] else None
            if exit_date is None or (start and exit_date < start) or (end and exit_date > end):
                continue
        if structure_type and structure_type != "all" and row["type"].lower().replace(" ", "") != structure_type.lower():
            continue
        if needle and not any(needle in symbol.lower() for symbol in row["symbols"]):
            continue
        if pnl_filter == PNL_WINNERS and row["realized_pnl"] < 0:
            continue
        if pnl_filter == PNL_LOSERS and row["realized_pnl"] > 0:
            continue
        kept.append(row)
    return kept


def summarize(rows: list[dict]) -> dict:
    """Summary stats of a set of rows. Flat trades count in the total but not in the win rate."""
    pnls = [row["realized_pnl"] for row in rows]
    winners = sum(1 for p in pnls if p > 0)
    losers = sum(1 for p in pnls if p < 0)
    decided = winners + losers
    return {
        "total": len(pnls),
        "winners": winners,
        "losers": losers,
        "win_rate": winners / decided * 100.0 if decided else 0.0,
        "total_pnl": sum(pnls),
        "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
        "best": max(pnls) if pnls else 0.0,
        "worst": min(pnls) if pnls else 0.0,
    }


# ----------------------------------------------------------------------
# Deleting archived structures
# ----------------------------------------------------------------------


def delete_archived_structures(repository, structure_ids: list[str], shift_reset_baseline: bool = False) -> dict:
    """Permanently delete closed structures (with their legs, trades and PnL history).

    Home's realized PnL is the sum over the open and the CLOSED structures' trades, so deleting a closed structure takes its
    net realized PnL (after transaction costs) out of it; so do today's realized PnL and the archive. Only structures that
    are CLOSED are deleted; anything else is skipped and named.

    An Account Reset (Settings) shows the totals RELATIVE to a baseline captured at the reset, so a structure that was
    already closed at the reset is in that baseline: deleting it would push the shown figure below zero. With
    `shift_reset_baseline` both baselines move by the same amount, so the shown figures are unchanged by that deletion.

    Returns {"deleted": [names], "skipped": [names], "removed_pnl": net realized PnL taken out}.
    """
    from core.user_settings import KEY_PNL_BASELINE, KEY_PNL_REALIZED_BASELINE

    deleted, skipped, removed = [], [], 0.0
    for structure_id in structure_ids:
        structure = repository.get_structure(structure_id)
        if structure is None:
            skipped.append(structure_id)
            continue
        if structure.status != StructureStatus.CLOSED:
            skipped.append(structure.name)
            continue
        removed += calculate_net_realized_pnl(repository.get_trades_for_structure(structure_id))
        repository.delete_structure(structure_id)
        deleted.append(structure.name)
    if shift_reset_baseline and deleted:
        for key in (KEY_PNL_BASELINE, KEY_PNL_REALIZED_BASELINE):
            current = repository.get_setting(key, 0.0) or 0.0
            if current:
                repository.set_setting(key, current - removed)
    return {"deleted": deleted, "skipped": skipped, "removed_pnl": removed}
