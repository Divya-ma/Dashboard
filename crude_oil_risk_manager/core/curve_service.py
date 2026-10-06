"""The Curve Kinks engine: turns live prices + history into curves, kinks, alerts and a status page.

`CurveService.on_prices` is called after every live poll (10 s). It builds the 15-month ladder
for each product (outrights, spreads, flies, dflies), values every structure live, runs the
kink maths (core.curve_kinks) for ALL of them whether or not the page is open, raises alerts,
keeps the kink event log, saves daily snapshots and rebuilds the data-quality flags. The page
only reads `state()`.

Background jobs (started from `on_prices`, never blocking it):
  * history refresh - once per UTC day, generic history for every series (core.curve_history)
  * previous settlements - once per day at the opening time set in Settings
Both retry after a failure rather than waiting for the next day.

Alerts: a kink alerts when it first appears at/above the minimum priority, when its priority
rises, or when it reappears after the cooldown. A kink that stays does not repeat. One alert is
also sent on each product's buffer day (a contract rolled out one day before its rule-based
expiry), because the rule-based expiry should then be confirmed.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import numpy as np

from adapters.base import APIError
from adapters.qh_api import QHApi
from core.curve_calendar import (
    FAMILIES,
    FAMILY_LABELS,
    FLY,
    N_MONTHS,
    OUTRIGHT,
    CurveStructure,
    buffer_rolls,
    build_structures,
    front_symbol_by_expiry,
    live_symbols,
)
from core.curve_history import CurveHistoryStore, generic_code
from core.curve_kinks import METHOD_LABELS, METHODS, PositionResult, detect, is_kink, prepare_history
from core.curve_settings import PRIORITY_RANK, CurveParams, load_open_time, load_params
from core.curve_store import PrevSettlementStore, SnapshotStore
from core.models import AlertLevel
from core.user_settings import KEY_API_TOKEN

logger = logging.getLogger(__name__)

KEY_HISTORY_REFRESHED_ON = "curve_history_refreshed_on"
RETRY_AFTER_FAILURE = timedelta(minutes=30)
CONSISTENCY_TOLERANCE = 0.10  # points; live spread vs the difference of the live outrights
STALE_HISTORY_DAYS = 5

PriceMap = dict[str, tuple[float, float]]  # API symbol -> (price, unix timestamp seconds)


@dataclass
class PositionView:
    structure: CurveStructure
    live: float | None
    prev_settle: float | None
    stale: bool
    result: PositionResult | None

    @property
    def change(self) -> float | None:
        return None if self.live is None or self.prev_settle is None else self.live - self.prev_settle


@dataclass
class FamilyView:
    product: str
    family: str
    positions: list[PositionView]
    history_rows: int = 0
    history_last: date | None = None
    methods_available: list[str] = field(default_factory=list)
    suspect: bool = False  # too much of the curve flagged: a data mismatch or whole-curve move, not kinks


@dataclass
class QualityFlag:
    severity: str  # "BUFFER" | "WARNING" | "INFO"
    product: str
    text: str


@dataclass
class CurveState:
    updated_at: datetime | None = None
    as_of: date | None = None
    families: dict[tuple[str, str], FamilyView] = field(default_factory=dict)
    kinks: list[dict] = field(default_factory=list)  # current kinks, strongest first
    quality: list[QualityFlag] = field(default_factory=list)


def structure_value(structure: CurveStructure, prices: PriceMap) -> float | None:
    """Live value of a structure from its components (a dfly = fly - next fly); None if one is missing."""
    total = 0.0
    for symbol, weight in zip(structure.components, structure.component_weights):
        if symbol not in prices:
            return None
        total += weight * prices[symbol][0]
    return total


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


class CurveService:
    def __init__(
        self,
        repository,
        api: QHApi,
        history: CurveHistoryStore,
        settlements: PrevSettlementStore,
        snapshots: SnapshotStore,
        alert_manager,
        products: tuple[str, ...] = ("CL", "BRN"),
        n: int = N_MONTHS,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        run_in_background: bool = True,
    ):
        self._repository = repository
        self._api = api
        self.history = history
        self.settlements = settlements
        self.snapshots = snapshots
        self._alerts = alert_manager
        self.products = products
        self.n = n
        self._clock = clock
        self._background = run_in_background
        self._compute_lock = threading.Lock()  # serialises compute; state() never takes it
        self._state = CurveState()
        self._ladders: dict[tuple[str, date], dict[str, list[CurveStructure]]] = {}
        self._prepared: dict[tuple, object] = {}
        self._active_alert_rank: dict[tuple, int] = {}
        self._last_alert_at: dict[tuple, datetime] = {}
        self._open_events: dict[tuple, int] = {}
        self._buffer_alerted: set[tuple[str, date]] = set()
        self._last_snapshot: dict[str, datetime] = {}
        self._jobs: dict[str, threading.Thread] = {}
        self._retry_after: dict[str, datetime] = {}
        self.job_messages: dict[str, str] = {}
        repository.clear_open_kink_events()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def params(self) -> CurveParams:
        return load_params(self._repository)

    def state(self) -> CurveState:
        return self._state  # replaced atomically by on_prices

    def ladder(self, product: str, today: date) -> dict[str, list[CurveStructure]]:
        key = (product, today)
        if key not in self._ladders:
            self._ladders = {k: v for k, v in self._ladders.items() if k[1] == today}
            self._ladders[key] = build_structures(product, today, self.n)
        return self._ladders[key]

    def symbols_to_poll(self) -> list[str]:
        """Every API symbol the live feed must price for the curves (outrights, spreads, flies)."""
        today = self._clock().date()
        return sorted({s for p in self.products for s in live_symbols(p, today, self.n)})

    def outright_symbols(self) -> list[str]:
        today = self._clock().date()
        return sorted({s.symbol for p in self.products for s in self.ladder(p, today)[OUTRIGHT]})

    def on_prices(self, prices: PriceMap) -> None:
        """Recompute everything from a fresh live poll. Never raises (the poll must not break)."""
        if not prices:
            return
        try:
            with self._compute_lock:
                self._maintain()
                self._state = self._compute(prices)
        except Exception:  # noqa: BLE001
            logger.exception("Curve computation failed")

    # ------------------------------------------------------------------
    # Background jobs
    # ------------------------------------------------------------------

    def _token_configured(self) -> bool:
        return bool(self._repository.get_setting(KEY_API_TOKEN, ""))

    def _start_job(self, name: str, target: Callable[[], None]) -> None:
        if not self._background:
            target()
            return
        thread = threading.Thread(target=target, name=f"curve-{name}", daemon=True)
        self._jobs[name] = thread
        thread.start()

    def _job_running(self, name: str) -> bool:
        thread = self._jobs.get(name)
        return thread is not None and thread.is_alive()

    def _maintain(self) -> None:
        if not self._token_configured():
            return
        now = self._clock()
        params = self.params()

        today = now.date().isoformat()
        if (
            self._repository.get_setting(KEY_HISTORY_REFRESHED_ON) != today
            and not self._job_running("history")
            and now >= self._retry_after.get("history", now)
        ):
            self._start_job("history", lambda: self._refresh_history(params.history_years, now.date()))

        open_time, zone = load_open_time(self._repository)
        local = now.astimezone(zone)
        due = self.settlements.fetched_on is None or (
            local.time() >= open_time and self.settlements.fetched_on != local.date().isoformat()
        )
        if due and not self._job_running("settlement") and now >= self._retry_after.get("settlement", now):
            self._start_job("settlement", lambda: self._refresh_settlements(now.date(), local.date()))

    def _refresh_history(self, years: int, today: date) -> None:
        failed = []
        for product in self.products:
            report = self.history.refresh(product, years, today, self.n)
            if report.errors:
                failed.append(f"{product}: {len(report.errors)} series failed")
        if failed:
            self.job_messages["history"] = "History refresh incomplete (" + "; ".join(failed) + "); will retry"
            self._retry_after["history"] = self._clock() + RETRY_AFTER_FAILURE
        else:
            self.job_messages["history"] = f"History refreshed {self._clock().strftime('%Y-%m-%d %H:%M')} UTC"
            self._repository.set_setting(KEY_HISTORY_REFRESHED_ON, today.isoformat())

    def _refresh_settlements(self, today: date, local_today: date) -> None:
        try:
            report = self.settlements.fetch(self._api, self.outright_symbols(), today, local_today)
            self.job_messages["settlement"] = (
                f"Previous settlements fetched ({report['fetched']} contracts"
                + (f", missing {', '.join(report['missing'][:4])}" if report["missing"] else "") + ")"
            )
        except (APIError, RuntimeError) as exc:
            self.job_messages["settlement"] = f"Previous settlement fetch failed: {exc}"
            self._retry_after["settlement"] = self._clock() + RETRY_AFTER_FAILURE
            logger.error("Previous settlement fetch failed: %s", exc)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _prepared_history(self, product: str, family: str, params: CurveParams):
        history = self.history.matrix(product, family, self.n)
        if history is None:
            return None, None
        key = (product, family, self.history.version, params.lookback_days, params.pca_components, params.min_scale)
        if key not in self._prepared:
            self._prepared = {k: v for k, v in self._prepared.items() if k[2] == self.history.version}
            self._prepared[key] = prepare_history(history.values, history.dates, history.months, params)
        return history, self._prepared[key]

    def _compute(self, prices: PriceMap) -> CurveState:
        now = self._clock()
        today = now.date()
        params = self.params()
        state = CurveState(updated_at=now, as_of=today)
        quality: list[QualityFlag] = []
        timestamps = {symbol: ts for symbol, (_, ts) in prices.items()}

        for product in self.products:
            ladder = self.ladder(product, today)
            for family in FAMILIES:
                structures = ladder[family]
                values = np.array([
                    v if (v := structure_value(s, prices)) is not None else np.nan for s in structures
                ])
                history, prepared = self._prepared_history(product, family, params)
                results = detect(
                    values, prepared, family != OUTRIGHT, params,
                    [s.front_month for s in structures], today,
                )
                positions = []
                for structure, result, value in zip(structures, results, values):
                    stale = any(
                        now.timestamp() - timestamps.get(c, 0) > params.stale_seconds for c in structure.components
                    ) and bool(np.isfinite(value))
                    positions.append(PositionView(
                        structure, None if np.isnan(value) else float(value),
                        self.settlements.structure_settlement(structure), stale, result,
                    ))
                methods = ["fit", "neighbour"] + (["pca"] if prepared and prepared.pca_components is not None else [])
                methods += ["history"] if (prepared is not None and family != OUTRIGHT) else []
                view = FamilyView(
                    product, family, positions, 0 if history is None else history.rows,
                    None if history is None else history.last_date, methods,
                )
                kinked = sum(1 for p in positions if p.result is not None and is_kink(p.result))
                if kinked > max(3, int(params.max_kink_fraction * len(positions))):
                    view.suspect = True
                    quality.append(QualityFlag(
                        "WARNING", product,
                        f"{FAMILY_LABELS[family]} curve: {kinked} of {len(positions)} points flagged. That looks like a "
                        "data mismatch (stale history, wrong generic code) or a whole-curve move, so no kinks or "
                        "alerts are reported for this curve",
                    ))
                state.families[(product, family)] = view
            quality.extend(self._quality_flags(product, today, ladder, prices, params, now, state))

        state.kinks = self._current_kinks(state, params)
        state.quality = quality
        self._handle_alerts(state, params, now)
        self._update_event_log(state, params)
        self._maybe_snapshot(state, params, now)
        return state

    # ------------------------------------------------------------------
    # Kinks, alerts, log, snapshots
    # ------------------------------------------------------------------

    @staticmethod
    def _kink_row(view: FamilyView, position: PositionView) -> dict:
        result, structure = position.result, position.structure
        code = structure.generic_code or (
            f"{generic_code(view.product, FLY, structure.position)} - {generic_code(view.product, FLY, structure.position + 1)}"
        )
        return {
            "product": view.product, "family": view.family, "label": structure.label,
            "position": structure.position, "generic": code, "symbol": structure.symbol,
            "direction": result.direction, "priority": result.priority, "score": result.score,
            "n_flags": result.n_flags, "z": result.z, "flags": result.flags, "value": position.live,
            "implied": result.implied, "residual": result.residual, "seasonal_z": result.seasonal_z,
            "seasonal_normal": result.seasonal_normal, "seasonal_obs": result.seasonal_obs,
            "prev_settle": position.prev_settle, "stale": position.stale,
        }

    def _current_kinks(self, state: CurveState, params: CurveParams) -> list[dict]:
        rows = [
            self._kink_row(view, p)
            for view in state.families.values()
            for p in view.positions
            if not view.suspect and p.result is not None and is_kink(p.result)
        ]
        rows.sort(key=lambda r: (PRIORITY_RANK[r["priority"]], r["score"]), reverse=True)
        return rows

    @staticmethod
    def kink_key(row: dict) -> tuple:
        return (row["product"], row["family"], row["label"])

    @staticmethod
    def alert_text(row: dict) -> tuple[str, str]:
        z = ", ".join(
            f"{METHOD_LABELS[m]} {row['z'][m]:+.1f}" for m in METHODS if row["flags"].get(m) and row["z"].get(m) is not None
        )
        seasonal = ""
        if row["seasonal_normal"]:
            seasonal = f" Seasonally normal for this delivery month at this time of year (seasonal z {row['seasonal_z']:+.1f})."
        elif row["seasonal_z"] is not None:
            seasonal = f" Not explained by seasonality (seasonal z {row['seasonal_z']:+.1f})."
        title = f"Kink {row['priority']}: {row['product']} {FAMILY_LABELS[row['family']].lower()} {row['label']}"
        body = (
            f"{row['generic']} is {row['direction']} vs the curve. {row['n_flags']} method(s): {z}. "
            f"Live {_fmt(row['value'])}, implied by neighbours {_fmt(row['implied'])}, score {row['score']:.1f}.{seasonal}"
        )
        return title, body

    def _handle_alerts(self, state: CurveState, params: CurveParams, now: datetime) -> None:
        if not params.alerts_enabled:
            self._active_alert_rank.clear()
            return
        current: dict[tuple, int] = {}
        to_send: list[dict] = []
        for row in state.kinks:
            if not meets_priority_row(row, params.alert_min_priority):
                continue
            key = self.kink_key(row)
            rank = PRIORITY_RANK[row["priority"]]
            current[key] = rank
            previous = self._active_alert_rank.get(key)
            last = self._last_alert_at.get(key)
            cooled = last is None or now - last >= timedelta(minutes=params.cooldown_minutes)
            if previous is None and not cooled:
                continue
            if previous is not None and rank <= previous:
                continue
            to_send.append(row)
            self._last_alert_at[key] = now
        self._active_alert_rank = current
        self._send_kink_alerts(to_send)

        for product in self.products:
            for rolled in buffer_rolls(product, now.date()):
                key = (product, now.date())
                if key in self._buffer_alerted:
                    continue
                self._buffer_alerted.add(key)
                self._alerts.send_alert(
                    AlertLevel.WARNING,
                    f"Buffer day: {product} {rolled.symbol} rolled out early",
                    f"{rolled.symbol} is treated as expired from today (one business day before the rule-based "
                    f"expiry {rolled.expiry.isoformat()}). Confirm the expiry date; curves now start at the next contract.",
                )

    def _send_kink_alerts(self, rows: list[dict]) -> None:
        """One alert per product per poll. One bad outright also shows up in the spreads, flies and
        dflies built from it, so those travel together: the strongest kink heads the alert and the
        related ones are listed after it."""
        for product in dict.fromkeys(r["product"] for r in rows):
            group = sorted(
                (r for r in rows if r["product"] == product),
                key=lambda r: (PRIORITY_RANK[r["priority"]], r["score"]), reverse=True,
            )
            title, body = self.alert_text(group[0])
            if len(group) > 1:
                title += f" (+{len(group) - 1} related)"
                body += " Also: " + "; ".join(
                    f"{FAMILY_LABELS[r['family']].lower()} {r['label']} {r['direction']} ({r['priority']})"
                    for r in group[1:7]
                ) + ("..." if len(group) > 7 else ".")
            level = AlertLevel.WARNING if group[0]["priority"] == "HIGH" else AlertLevel.INFO
            self._alerts.send_alert(level, title, body)

    def _update_event_log(self, state: CurveState, params: CurveParams) -> None:
        current = {self.kink_key(r): r for r in state.kinks if meets_priority_row(r, params.alert_min_priority)}
        for key, row in current.items():
            detail = {"z": row["z"], "value": row["value"], "implied": row["implied"], "residual": row["residual"],
                      "seasonal_z": row["seasonal_z"], "generic": row["generic"]}
            event_id = self._open_events.get(key)
            if event_id is None:
                self._open_events[key] = self._repository.open_kink_event(
                    row["product"], row["family"], row["position"], row["label"], row["symbol"] or row["generic"],
                    row["direction"], row["score"], row["priority"], row["n_flags"], row["seasonal_normal"], detail,
                )
            else:
                self._repository.update_kink_event(
                    event_id, row["score"], row["priority"], row["n_flags"], row["seasonal_normal"], detail, PRIORITY_RANK
                )
        for key in [k for k in self._open_events if k not in current]:
            self._repository.clear_kink_event(self._open_events.pop(key))

    def _maybe_snapshot(self, state: CurveState, params: CurveParams, now: datetime) -> None:
        for product in self.products:
            last = self._last_snapshot.get(product)
            if last is not None and now - last < timedelta(minutes=params.snapshot_minutes):
                continue
            rows = [
                {"family": family, "position": p.structure.position, "label": p.structure.label, "value": p.live}
                for family in FAMILIES
                for p in state.families[(product, family)].positions
                if p.live is not None
            ]
            if rows:
                self.snapshots.save(product, now.date(), rows)
                self._last_snapshot[product] = now

    # ------------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------------

    def _quality_flags(
        self, product: str, today: date, ladder, prices: PriceMap, params: CurveParams, now: datetime, state: CurveState
    ) -> list[QualityFlag]:
        flags: list[QualityFlag] = []
        needed = [s.symbol for family in FAMILIES[:3] for s in ladder[family]]
        missing = [s for s in needed if s not in prices]
        if missing:
            flags.append(QualityFlag("WARNING", product, f"{len(missing)} of {len(needed)} live prices missing (e.g. {', '.join(missing[:3])}); affected points are skipped"))
        stale = [s for s in needed if s in prices and now.timestamp() - prices[s][1] > params.stale_seconds]
        if stale:
            flags.append(QualityFlag("WARNING", product, f"{len(stale)} live prices older than {params.stale_seconds}s (e.g. {', '.join(stale[:3])})"))

        outrights, spreads = ladder[OUTRIGHT], ladder["spread"]
        off = [
            s.label for s, a, b in zip(spreads, outrights, outrights[1:])
            if s.symbol in prices and a.symbol in prices and b.symbol in prices
            and abs(prices[s.symbol][0] - (prices[a.symbol][0] - prices[b.symbol][0])) > CONSISTENCY_TOLERANCE
        ]
        if off:
            flags.append(QualityFlag("INFO", product, f"Live spread differs from the outright difference by more than {CONSISTENCY_TOLERANCE:.2f} for {', '.join(off[:4])}"))

        for family in FAMILIES:
            history = self.history.matrix(product, family, self.n)
            label = FAMILY_LABELS[family].lower()
            if history is None:
                flags.append(QualityFlag("WARNING", product, f"No history loaded for {label} curves yet: only the Fit and Neighbour methods (no history) run"))
                continue
            if history.rows < params.lookback_days // 2:
                flags.append(QualityFlag("INFO", product, f"Short {label} history: {history.rows} aligned days (lookback {params.lookback_days})"))
            if history.last_date and (today - history.last_date).days > STALE_HISTORY_DAYS:
                flags.append(QualityFlag("WARNING", product, f"{label.capitalize()} history ends {history.last_date.isoformat()}"))

        settled = [s.symbol for s in ladder[OUTRIGHT] if s.symbol not in self.settlements.prices()]
        if self.settlements.fetched_on is None:
            flags.append(QualityFlag("WARNING", product, "Previous settlements not fetched yet"))
        elif settled:
            flags.append(QualityFlag("INFO", product, f"No previous settlement for {len(settled)} contracts (e.g. {', '.join(settled[:3])})"))

        for rolled in buffer_rolls(product, today):
            flags.append(QualityFlag("BUFFER", product, f"Buffer day: {rolled.symbol} rolled out one day before its rule-based expiry {rolled.expiry.isoformat()}; confirm the expiry"))

        series = self.history.read_series(product, generic_code(product, OUTRIGHT, 1))
        if series is not None and not series.empty:
            last_day = series["date"].iloc[-1].date()
            held = series["contract"].iloc[-1]
            expected = front_symbol_by_expiry(product, last_day)
            if held and held != expected:
                flags.append(QualityFlag("WARNING", product, f"Generic position 1 held {held} on {last_day.isoformat()} but the calendar rule gives {expected}: check the expiry rule"))
        return flags


def meets_priority_row(row: dict, minimum: str) -> bool:
    return PRIORITY_RANK[row["priority"]] >= PRIORITY_RANK[minimum]
