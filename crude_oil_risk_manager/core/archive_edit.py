"""Corrections to an archived trade (a closed structure, or one that has exited some lots): every archive column can be changed from where its value lives.

  Structure   name, structure type, notes                              -> the structure row
  Trades      time, price, lots, direction, cost, realized P&L, stop,  -> the trade rows; they drive entry / exit date, days held,
              target                                                      entry / exit price, lots, costs, P&L, result, risk:reward,
                                                                          risk $ and R
  Legs        symbols and ratios                                       -> the leg rows (legs column)
  Market      the entry / exit snapshot cells and MFE / MAE            -> the trade journal document (core.journal)
  Journal     ideas, payups, conviction, exit reason, mistakes         -> already editable (JournalService)

The archive columns that are worked out (days held, result, risk:reward, risk $, R) have no value of their own: change what
they are worked out from.

PRICES AND P&L. Changing a trade's price, lots or direction replays the structure's trades (core.trade_entry.
recompute_structure_from_trades, the same replay a deleted trade uses): the legs' entry prices and each exit's realized P&L are
recalculated. A cost is never recalculated from a price; it follows the lots (scaled in proportion). A realized P&L or cost typed
in the same save always wins over the recalculated one. With the replay off (or when only times, costs, P&L, stops or targets
change) every figure is stored exactly as typed. Either way the structure
keeps its state: an edit that would leave lots open in a closed trade (or close one that still has lots open) is refused. Every change writes one audit line to the structure's notes.
"""

import math
from datetime import datetime, timezone

from core.exceptions import CrudeOilRiskError
from core.archive import SNAPSHOT_FIELDS
from core.journal import BACKWARDATION, BACKWARDATION_CONTANGO, BEARISH, BULLISH, CONTANGO, CONTANGO_BACKWARDATION, FLAT, STALEMATE
from core.models import StructureStatus, TradeEventType
from core.models.structure import parse_structure_type
from core.structure_builder import StructureBuildError
from core.structure_edit import apply_leg_edits
from core.trade_entry import TradeError, recompute_structure_from_trades

TIME_FORMAT = "%Y-%m-%d %H:%M"
NAME_LIMIT = 100
NOTES_LIMIT = 19_000  # the model allows 20,000 and the audit lines still have to fit
EXIT_EVENTS = (TradeEventType.PARTIAL_EXIT, TradeEventType.STRUCTURE_PARTIAL_EXIT, TradeEventType.FULL_EXIT, TradeEventType.DISSOLVE)
REPLAYED_EVENTS = (TradeEventType.PARTIAL_EXIT, TradeEventType.STRUCTURE_PARTIAL_EXIT, TradeEventType.FULL_EXIT)  # exits whose P&L follows the prices
NUMERIC_SNAPSHOT_COLUMNS = ("px_co", "px_cl", "px_go", "atr", "vol_ratio", "volume_ratio", "spread_ticks", "brent_wti", "crack", "kink_z")
MOVES = (BULLISH, BEARISH, STALEMATE)
CURVES = (CONTANGO, BACKWARDATION, CONTANGO_BACKWARDATION, BACKWARDATION_CONTANGO, FLAT)


class ArchiveEditError(CrudeOilRiskError):
    """Blocking problems with an archive edit; `errors` are safe to show the user."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------


def parse_number(value, label: str, allow_blank: bool = True, minimum: float | None = None, strictly_above: float | None = None):
    """A typed number (commas allowed); None for blank when allowed. ValueError with a readable message otherwise."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_blank:
            return None
        raise ValueError(f"{label} is required.")
    try:
        number = float(str(value).replace(",", "")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a number.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} cannot be below {minimum:g}.")
    if strictly_above is not None and number <= strictly_above:
        raise ValueError(f"{label} must be greater than {strictly_above:g}.")
    return number


def parse_timestamp(value, label: str = "Date and time") -> datetime:
    """'2026-05-04', '2026-05-04 14:30', '2026-05-04 14:30:15' or ISO text (read as UTC unless it carries an offset)."""
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value or "").strip().replace("T", " ").replace("Z", "+00:00")
        if not text:
            raise ValueError(f"{label} is required.")
        moment = None
        for pattern in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                moment = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
        if moment is None:
            try:
                moment = datetime.fromisoformat(text)
            except ValueError:
                raise ValueError(f"{label}: use YYYY-MM-DD or YYYY-MM-DD HH:MM (UTC).") from None
    return moment.astimezone(timezone.utc) if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _norm(text) -> str:
    return " ".join(str(text).replace("-", " ").split()).lower()


def _pick(value, allowed: tuple[str, ...], label: str) -> str | None:
    if value is None or not str(value).strip():
        return None
    for option in allowed:
        if _norm(option) == _norm(value):
            return option
    raise ValueError(f"{label} must be one of: {', '.join(allowed)}.")


# ----------------------------------------------------------------------
# Structure: name, type, notes
# ----------------------------------------------------------------------


def edit_structure(repository, structure_id: str, name, structure_type, notes) -> str:
    structure = repository.get_structure(structure_id)
    if structure is None:
        raise ArchiveEditError(["This trade no longer exists."])
    errors = []
    name = (name or "").strip()
    if not name:
        errors.append("Structure name is required.")
    elif len(name) > NAME_LIMIT:
        errors.append(f"Structure name is limited to {NAME_LIMIT} characters.")
    kind = structure.structure_type
    try:
        kind = parse_structure_type(structure_type) or structure.structure_type
    except ValueError as exc:
        errors.append(str(exc))
    notes = structure.notes if notes is None else str(notes)
    if len(notes) > NOTES_LIMIT:
        errors.append(f"Notes are limited to {NOTES_LIMIT:,} characters.")
    if errors:
        raise ArchiveEditError(errors)
    changes = []
    if name != structure.name:
        changes.append(f"name '{structure.name}' -> '{name}'")
    if kind != structure.structure_type:
        changes.append(f"type {structure.structure_type.label} -> {kind.label}")
    if notes != structure.notes:
        changes.append("notes edited")
    if not changes:
        return "No changes to save."
    repository.update_structure_fields(structure_id, name, kind, notes, audit_note="archive edit: " + "; ".join(changes))
    return "Saved: " + "; ".join(changes) + "."


# ----------------------------------------------------------------------
# Legs
# ----------------------------------------------------------------------


def edit_legs(repository, structure_id: str, symbols: list, ratios: list) -> str:
    """Change leg symbols and ratios (the number of legs stays). New contracts are created like the builder does."""
    structure = repository.get_structure(structure_id)
    if structure is None:
        raise ArchiveEditError(["This trade no longer exists."])
    try:
        result = apply_leg_edits(structure, symbols, ratios, repository.get_contract)
    except StructureBuildError as exc:
        raise ArchiveEditError(exc.errors) from exc
    if not result.audit_note.startswith("edited"):
        return "No changes to save."
    for contract in result.new_contracts:
        repository.save_contract(contract)
    repository.update_structure_legs(structure_id, result.legs, "archive " + result.audit_note)
    products = sorted({leg.contract.product for leg in result.legs})
    if products != sorted(structure.products):
        repository.update_structure_fields(structure_id, products=products)
    return "Saved: " + result.audit_note[len("edited: "):] + "." + "".join(f" {w}" for w in result.warnings)


# ----------------------------------------------------------------------
# Trades
# ----------------------------------------------------------------------


def trade_rows(trades) -> list[dict]:
    """The trades as editable table rows (the row `id` is the trade id)."""
    return [
        {
            "id": t.trade_id, "event": t.event_type.value.replace("_", " "), "direction": t.direction,
            "timestamp": t.timestamp.astimezone(timezone.utc).strftime(TIME_FORMAT),
            "price": t.price, "lots": t.lots, "transaction_cost": t.transaction_cost, "realized_pnl": t.realized_pnl,
            "stop_loss_price": t.stop_loss_price, "target_price": t.target_price,
        }
        for t in sorted(trades, key=lambda t: t.timestamp)
    ]


def _edited_trade(trade, row: dict, number: int, errors: list[str]):
    """The trade with the row's figures applied (None, with the reasons in `errors`, if a figure is bad)."""
    tag = f"Trade {number} ({trade.event_type.value.replace('_', ' ')})"
    update, problems = {}, []

    def take(key, parser, label, **kwargs):
        try:
            update[key] = parser(row.get(key), f"{tag}: {label}", **kwargs)
        except ValueError as exc:
            problems.append(str(exc))

    shown = trade.timestamp.astimezone(timezone.utc).strftime(TIME_FORMAT)
    if str(row.get("timestamp") or "").strip() == shown:
        update["timestamp"] = trade.timestamp  # untouched: the table only shows minutes, so keep the stored seconds
    else:
        take("timestamp", parse_timestamp, "date and time")
    take("price", parse_number, "price", allow_blank=False)
    take("lots", parse_number, "lots", allow_blank=False, strictly_above=0)
    take("transaction_cost", parse_number, "transaction cost", allow_blank=True, minimum=0)
    take("realized_pnl", parse_number, "realized P&L")
    take("stop_loss_price", parse_number, "stop")
    take("target_price", parse_number, "target")
    direction = str(row.get("direction") or "").strip().lower()
    if direction not in ("buy", "sell"):
        problems.append(f"{tag}: direction must be buy or sell.")
    update["direction"] = direction
    if "transaction_cost" in update and update["transaction_cost"] is None:
        update["transaction_cost"] = 0.0
    if problems:
        errors.extend(problems)
        return None
    if trade.event_type not in EXIT_EVENTS and update["realized_pnl"] is not None:
        update["realized_pnl"] = None  # only exits carry a realized P&L
    return trade.model_copy(update=update)


def _describe(old, new, number: int) -> list[str]:
    tag = f"trade {number} ({old.event_type.value})"
    labels = (("timestamp", "time"), ("price", "price"), ("lots", "lots"), ("direction", "direction"), ("transaction_cost", "cost"),
              ("realized_pnl", "P&L"), ("stop_loss_price", "stop"), ("target_price", "target"))
    out = []
    for key, label in labels:
        before, after = getattr(old, key), getattr(new, key)
        if before != after:
            show = (lambda v: v.strftime("%Y-%m-%d %H:%M") if isinstance(v, datetime) else v if isinstance(v, str) or v is None else f"{v:g}")
            out.append(f"{tag} {label} {show(before)} -> {show(after)}")
    return out


def edit_trades(repository, structure_id: str, rows: list[dict], recalculate: bool = True) -> str:
    """Apply the edited trade rows (see the module docstring for the price / P&L rules)."""
    structure = repository.get_structure(structure_id)
    if structure is None:
        raise ArchiveEditError(["This trade no longer exists."])
    stored = sorted(repository.get_trades_for_structure(structure_id), key=lambda t: t.timestamp)
    by_id = {t.trade_id: t for t in stored}
    errors, edited = [], {}
    for number, row in enumerate(sorted(rows or [], key=lambda r: str(r.get("timestamp") or "")), start=1):
        trade = by_id.get(row.get("id"))
        if trade is None:
            errors.append(f"Row {number} is not one of this trade's records; reopen the trade.")
            continue
        new = _edited_trade(trade, row, number, errors)
        if new is not None:
            edited[trade.trade_id] = new
    if errors:
        raise ArchiveEditError(errors)
    if not edited:
        return "No trades to save."
    final = {tid: edited.get(tid, t) for tid, t in by_id.items()}
    numbering = {t.trade_id: i for i, t in enumerate(sorted(by_id.values(), key=lambda t: t.timestamp), start=1)}
    changes = [c for tid, new in edited.items() for c in _describe(by_id[tid], new, numbering[tid])]
    if not changes:
        return "No changes to save."

    if recalculate:  # a cost follows the lots (in proportion) unless a cost was typed for that trade
        for trade_id, new in edited.items():
            old = by_id[trade_id]
            if new.lots != old.lots and new.transaction_cost == old.transaction_cost and old.lots > 0:
                final[trade_id] = new.model_copy(update={"transaction_cost": old.transaction_cost * new.lots / old.lots})
                changes.append(f"trade {numbering[trade_id]} cost scaled to the new lots ({old.transaction_cost:g} -> {final[trade_id].transaction_cost:g})")

    moves_the_maths = any(
        (by_id[tid].price, by_id[tid].lots, by_id[tid].direction) != (new.price, new.lots, new.direction) for tid, new in edited.items()
    )
    replay = None
    if recalculate and moves_the_maths:
        try:
            replay = recompute_structure_from_trades(structure, list(final.values()))
        except TradeError as exc:
            raise ArchiveEditError([f"The trades do not replay: {error}" for error in exc.errors]) from exc
        was_closed = structure.status == StructureStatus.CLOSED
        if (replay.structure.status == StructureStatus.CLOSED) != was_closed:
            raise ArchiveEditError([
                "With these prices and lots the exits no longer close the whole position (some lots would stay open). "
                "Make the exit lots add up to the entry lots, or switch the recalculation off." if was_closed else
                "With these lots the exits would close the whole position, which is still open. Exit it from the Structures page "
                "instead, or keep the exit lots below the entry lots."
            ])
        for trade_id, (pnl, _cost) in replay.exit_trade_updates.items():
            as_typed, original = final[trade_id], by_id[trade_id]
            if as_typed.event_type not in REPLAYED_EVENTS or as_typed.realized_pnl != original.realized_pnl:  # a P&L typed now wins
                continue
            if pnl != as_typed.realized_pnl:
                final[trade_id] = as_typed.model_copy(update={"realized_pnl": pnl})
                changes.append(f"trade {numbering[trade_id]} P&L recalculated from the prices ({original.realized_pnl:,.2f} -> {pnl:,.2f})")

    for trade_id in edited:
        repository.update_trade_fields(final[trade_id])
    for trade_id, trade in final.items():
        if trade_id not in edited and trade != by_id[trade_id]:  # an untouched exit whose P&L followed the new entry price
            repository.update_trade_fields(trade)
    audit = "archive edit: " + "; ".join(changes)
    if replay is not None:
        repository.update_structure_legs(structure_id, replay.structure.legs, audit)
    else:
        repository.update_structure_fields(structure_id, audit_note=audit)
    exits = [t for t in final.values() if t.event_type in EXIT_EVENTS]
    if exits and structure.status == StructureStatus.CLOSED and any(by_id[t.trade_id].timestamp != t.timestamp for t in exits):
        repository.set_structure_lifecycle(structure_id, StructureStatus.CLOSED, max(t.timestamp for t in exits), structure.close_trigger)
    return f"Saved {len(edited)} trade(s): " + "; ".join(changes) + "."


# ----------------------------------------------------------------------
# Market snapshot and excursion (the journal)
# ----------------------------------------------------------------------


def parse_snapshot_cell(column: str, value):
    """One snapshot cell: a number, a movement word or a curve shape; None for blank."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if column in NUMERIC_SNAPSHOT_COLUMNS:
        return parse_number(value, column)
    if column == "atr_basis":
        return str(value).strip()[:40]
    if column in ("move_day", "move_min"):
        return _pick(value, MOVES, "A movement")
    return _pick(value, CURVES, "A curve shape")


def apply_snapshot_values(snapshot: dict | None, values: dict) -> dict | None:
    """The snapshot with the given cells ({column: raw value}) applied; an empty result is None."""
    snap = {key: (dict(val) if isinstance(val, dict) else val) for key, val in (snapshot or {}).items()}
    typed_something = False
    for column, key, sub_key in SNAPSHOT_FIELDS:
        if column not in values:
            continue
        parsed = parse_snapshot_cell(column, values[column])
        typed_something = typed_something or parsed is not None
        if sub_key is None:
            snap[key] = parsed
        else:
            part = dict(snap.get(key) or {})
            part[sub_key] = parsed
            snap[key] = part
    if snapshot is None and not typed_something:
        return None  # nothing typed and nothing there: do not invent an empty snapshot
    snap["edited"] = True
    return snap


def apply_excursion_values(excursion: dict | None, values: dict) -> dict | None:
    """MFE / MAE ($) typed for the intraday and the daily path: {"mfe_i", "mae_i", "mfe_d", "mae_d"}; all blank clears them."""
    old = excursion or {}
    daily = dict(old.get("daily") or (old if str(old.get("basis", "")).startswith("daily") else {}))
    intraday = dict(old.get("intraday") or {})
    for part, mfe_key, mae_key in ((intraday, "mfe_i", "mae_i"), (daily, "mfe_d", "mae_d")):
        mfe = parse_number(values.get(mfe_key), "MFE")
        mae = parse_number(values.get(mae_key), "MAE")
        if mfe is not None and mfe < 0:
            raise ValueError("MFE (best open P&L) cannot be negative.")
        if mae is not None and mae > 0:
            raise ValueError("MAE (worst open P&L) cannot be positive; type it as a negative number.")
        part["mfe"], part["mae"] = mfe, mae
        if mfe is None and mae is None:
            part.clear()
    daily, intraday = daily or None, intraday or None
    if daily is None and intraday is None:
        return None
    best = intraday or daily
    for part in (daily, intraday):
        if part is not None:
            part.setdefault("basis", "typed by hand")
    return {"mfe": best["mfe"], "mae": best["mae"], "basis": best.get("basis", "typed by hand"), "lots": old.get("lots"), "daily": daily, "intraday": intraday}
