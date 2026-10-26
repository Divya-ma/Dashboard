"""Keeps the trade journal up to date: snapshots at entry and exit, payups, trade ideas, and the backfill of old trades.

The market snapshot needs the API (two /ohlc/ requests), so it runs on a background thread after the trade is saved: entering or
exiting a trade never waits for it, and if it fails the journal simply has no snapshot for that trade (the backfill can fill it
later). Payups and trade ideas are plain typed data and are saved at once.
"""

import logging
import threading
from datetime import datetime, timezone

from core.journal import (
    add_payups, clean_conviction, clean_exit_reason, clean_ideas, clean_mistakes, entry_and_exit_times, excursion, parse_idea_list,
    set_payups, with_defaults,
)
from core.models import Structure, StructureStatus, TradeEventType
from core.journal import EXIT_REASONS, MISTAKES
from core.user_settings import KEY_EXIT_REASONS, KEY_MISTAKES, KEY_TRADE_IDEAS

logger = logging.getLogger(__name__)

PER_TRADE_KEY = {"entry": "entries", "exit": "exits"}  # journal["entries"][trade id] / journal["exits"][trade id]
_ENTRY_TYPES = (TradeEventType.TRADE, TradeEventType.ADD)
_EXIT_TYPES = (TradeEventType.STRUCTURE_PARTIAL_EXIT, TradeEventType.FULL_EXIT, TradeEventType.DISSOLVE, TradeEventType.PARTIAL_EXIT)
_OPEN_STATUSES = (StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED)


class JournalService:
    def __init__(
        self, repository, snapshotter, clock=lambda: datetime.now(timezone.utc), background: bool = True, price_path=None,
        intraday_path=None, spare_requests=None,
    ):
        self._price_path = price_path  # price_path(structure) -> the structure's daily price series, for the excursion
        self._intraday_path = intraday_path  # intraday_path(structure, start, end) -> its intraday price series (one API request)
        self._spare_requests = spare_requests  # spare_requests() -> API requests free right now; the intraday excursion only runs with room
        self._repository = repository
        self._snapshotter = snapshotter
        self._clock = clock
        self._background = background
        self._lock = threading.Lock()  # one journal write at a time: the snapshot threads and the forms both read-modify-write
        self.threads: list[threading.Thread] = []

    # -- trade ideas (the list lives in Settings) -------------------------

    def idea_options(self) -> list[str]:
        return clean_ideas(self._repository.get_setting(KEY_TRADE_IDEAS, []))

    def save_idea_options(self, text: str | None) -> list[str]:
        ideas = parse_idea_list(text)
        self._repository.set_setting(KEY_TRADE_IDEAS, ideas)
        return ideas

    def exit_reason_options(self) -> list[str]:
        """The exit reasons on offer: the list saved in Settings, or the built-in one until it has been edited."""
        saved = self._repository.get_setting(KEY_EXIT_REASONS, None)
        return list(EXIT_REASONS) if saved is None else clean_ideas(saved)

    def mistake_options(self) -> list[str]:
        saved = self._repository.get_setting(KEY_MISTAKES, None)
        return list(MISTAKES) if saved is None else clean_ideas(saved)

    def save_exit_reason_options(self, text: str | None) -> list[str]:
        reasons = parse_idea_list(text)
        self._repository.set_setting(KEY_EXIT_REASONS, reasons)
        return reasons

    def save_mistake_options(self, text: str | None) -> list[str]:
        tags = parse_idea_list(text)
        self._repository.set_setting(KEY_MISTAKES, tags)
        return tags

    # -- reading / small edits -------------------------------------------

    def journal(self, structure_id: str) -> dict:
        return with_defaults(self._repository.get_journal(structure_id))

    def _update(self, structure_id: str, change) -> dict:
        with self._lock:
            journal = with_defaults(self._repository.get_journal(structure_id))
            journal = change(journal) or journal
            self._repository.save_journal(structure_id, journal)
            return journal

    def set_ideas(self, structure_id: str, ideas) -> dict:
        allowed = self.idea_options()
        picked = clean_ideas(ideas)
        # keep an idea that was later removed from Settings if the trade already had it; only NEW picks must be on the list
        kept = set(self.journal(structure_id)["ideas"])
        return self._update(structure_id, lambda j: j.update(ideas=[i for i in picked if i in allowed or i in kept]))

    def add_entry_inputs(self, structure_id: str, ideas, payups_total, payups_legs: dict | None, conviction=None) -> dict:
        """What the entry form collects: the picks from the ideas list (added to any already there), the payups and the
        conviction (the first entry's stands unless the form is filled in again)."""
        conviction = clean_conviction(conviction)

        def change(journal):
            journal["ideas"] = clean_ideas([*journal["ideas"], *clean_ideas(ideas, self.idea_options())])
            if conviction is not None:
                journal["conviction"] = conviction
            return add_payups(journal, "entry", payups_total, payups_legs)

        return self._update(structure_id, change)

    def add_exit_payups(self, structure_id: str, payups_total, payups_legs: dict | None, exit_reason=None) -> dict:
        reason = clean_exit_reason(exit_reason, self.exit_reason_options())

        def change(journal):
            if reason is not None:
                journal["exit_reason"] = reason
            return add_payups(journal, "exit", payups_total, payups_legs)

        return self._update(structure_id, change)

    def set_review(self, structure_id: str, conviction, exit_reason, mistakes) -> dict:
        """The Archive editor's review fields (blank clears)."""
        # a reason or tag already on the trade stays valid even if it was later taken off the lists in Settings
        had = self.journal(structure_id)
        conviction = clean_conviction(conviction)
        reason = clean_exit_reason(exit_reason, [*self.exit_reason_options(), *([had["exit_reason"]] if had["exit_reason"] else [])])
        tags = clean_mistakes(mistakes, [*self.mistake_options(), *had["mistakes"]])
        return self._update(structure_id, lambda j: j.update(conviction=conviction, exit_reason=reason, mistakes=tags))

    def set_market(self, structure_id: str, entry_values: dict | None, exit_values: dict | None, excursion_values: dict | None) -> dict:
        """The Archive editor's market table and MFE / MAE: {column: typed value} for the entry and the exit snapshot (blank
        clears a cell), and the four excursion figures. Raises ValueError, writing nothing, if any cell is invalid."""
        from core.archive_edit import apply_excursion_values, apply_snapshot_values  # late: archive_edit imports the trade engine

        with self._lock:
            journal = with_defaults(self._repository.get_journal(structure_id))
            for kind, values in (("entry", entry_values), ("exit", exit_values)):
                if values:
                    journal[kind] = apply_snapshot_values(journal.get(kind), values)
            if excursion_values is not None:
                journal["excursion"] = apply_excursion_values(journal.get("excursion"), excursion_values)
            self._repository.save_journal(structure_id, journal)
            return journal

    INTRADAY_MIN_SPARE = 8  # /ohlc/ requests that must be free (this minute, hour and day) before an intraday excursion is fetched

    def intraday_possible(self) -> bool:
        return self._intraday_path is not None and (self._spare_requests is None or self._spare_requests() >= self.INTRADAY_MIN_SPARE)

    def record_excursion(self, structure: Structure, force_intraday: bool = False) -> None:
        """Work out and keep the trade's MFE / MAE: from daily closes always, and from intraday closes too when the API has
        spare requests (a trade can be opened and closed within a day, which daily closes cannot see). The figures kept at the top
        (`mfe`, `mae`) are the intraday ones when there are any, else the daily ones."""
        trades = self._repository.get_trades_for_structure(structure.structure_id)
        first, last = entry_and_exit_times(trades)
        if first is None:
            return
        previous = self.journal(structure.structure_id).get("excursion") or {}
        daily = previous.get("daily") or (previous if previous.get("basis", "").startswith("daily") else None)
        intraday = previous.get("intraday")
        try:
            if self._price_path is not None:
                daily = excursion(structure, trades, self._price_path(structure)) or daily
        except Exception:  # noqa: BLE001
            logger.exception("Journal: daily excursion failed for %s", structure.name)
        if self.intraday_possible():
            try:
                fresh = excursion(structure, trades, self._intraday_path(structure, first, last or first), intraday=True)
                intraday = fresh or intraday
            except Exception:  # noqa: BLE001 - rate limit, API down: keep the daily one, the backfill tries again later
                logger.warning("Journal: intraday excursion unavailable for %s", structure.name, exc_info=True)
        if daily is None and intraday is None:
            return
        best = intraday or daily
        result = {"mfe": best["mfe"], "mae": best["mae"], "basis": best["basis"], "lots": best.get("lots"), "daily": daily, "intraday": intraday}
        self._update(structure.structure_id, lambda j: j.update(excursion=result))

    def set_payups(self, structure_id: str, kind: str, total, legs: dict | None) -> dict:
        return self._update(structure_id, lambda j: set_payups(j, kind, total, legs))

    # -- snapshots --------------------------------------------------------

    def _take(self, structure: Structure, kind: str, when: datetime, overwrite: bool, trade_id: str | None = None, set_legacy: bool = True) -> None:
        try:
            snap = self._snapshotter.snapshot(structure, when)
        except Exception:  # noqa: BLE001
            logger.exception("Journal: %s snapshot failed for %s", kind, structure.name)
            return

        def change(journal):
            if set_legacy and (journal.get(kind) is None or overwrite):
                journal[kind] = snap
            if trade_id:  # the snapshot of THIS entry / exit, which an Archive row of one exit reads
                journal[PER_TRADE_KEY[kind]][trade_id] = snap
            return journal

        self._update(structure.structure_id, change)
        if kind == "exit":
            self.record_excursion(self._repository.get_structure(structure.structure_id) or structure)

    def record(
        self, structure: Structure, kind: str, when: datetime | None = None, overwrite: bool = False, trade_id: str | None = None,
        set_legacy: bool = True,
    ) -> threading.Thread | None:
        """Take the market snapshot of an "entry" (the first one is kept as THE entry) or an "exit" (the latest one is kept as
        THE exit; `set_legacy=False` leaves them alone, for an add). With `trade_id` it is also kept for that one trade, so every
        entry and every exit has its own."""
        when = when or self._clock()
        if self._snapshotter is None:
            return None
        if not self._background:
            self._take(structure, kind, when, overwrite, trade_id, set_legacy)
            return None
        thread = threading.Thread(target=self._take, args=(structure, kind, when, overwrite, trade_id, set_legacy), name=f"journal-{kind}", daemon=True)
        self.threads = [t for t in self.threads if t.is_alive()] + [thread]
        thread.start()
        return thread

    def _adopt_legacy_snapshots(self, structure_id: str, trades) -> None:
        """The first entry's snapshot (`entry`) and the last exit's (`exit`) were kept before every trade had its own: file them
        under their trade so they need no new API request."""
        entries = sorted((t for t in trades if t.event_type in _ENTRY_TYPES), key=lambda t: t.timestamp)
        exits = sorted((t for t in trades if t.event_type in _EXIT_TYPES), key=lambda t: t.timestamp)

        def change(journal):
            for kind, trade in (("entry", entries[0] if entries else None), ("exit", exits[-1] if exits else None)):
                if trade is not None and journal.get(kind) is not None:
                    journal[PER_TRADE_KEY[kind]].setdefault(trade.trade_id, journal[kind])
            return journal

        self._update(structure_id, change)

    def backfill(self, limit: int | None = None) -> dict:
        """Fill the missing snapshots of every entry and exit of the closed and the partly exited structures, from the trade
        times. Returns counts.

        A trade costs three API requests, so `limit` (structures per run) caps a run; call again for the rest."""
        done = missing = 0
        for structure in self._repository.get_all_structures(status_filter=[StructureStatus.CLOSED, *_OPEN_STATUSES]):
            trades = self._repository.get_trades_for_structure(structure.structure_id)
            if not any(t.event_type in _EXIT_TYPES for t in trades):
                continue  # nothing exited: not in the Archive
            self._adopt_legacy_snapshots(structure.structure_id, trades)
            journal = self.journal(structure.structure_id)
            wanted = [
                (("entry" if t.event_type in _ENTRY_TYPES else "exit"), t)
                for t in sorted(trades, key=lambda t: t.timestamp)
                if t.event_type in (*_ENTRY_TYPES, *_EXIT_TYPES) and t.trade_id not in journal[PER_TRADE_KEY["entry" if t.event_type in _ENTRY_TYPES else "exit"]]
            ]
            first, last = entry_and_exit_times(trades)
            excursion_done = journal.get("excursion") or {}
            if (
                not wanted and structure.status == StructureStatus.CLOSED and last is not None
                and (self._price_path is not None or self._intraday_path is not None)
                and (not excursion_done or (excursion_done.get("intraday") is None and self.intraday_possible()))
            ):
                self.record_excursion(structure)  # daily needs no API; the intraday part is fetched only when requests are spare
            if not wanted:
                continue
            missing += 1
            if limit is not None and done >= limit:
                continue
            for kind, trade in wanted:
                self._take(structure, kind, trade.timestamp, overwrite=False, trade_id=trade.trade_id)
            done += 1
        return {"filled": done, "left": max(missing - done, 0)}
