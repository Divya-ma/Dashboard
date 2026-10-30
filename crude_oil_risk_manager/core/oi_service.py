"""The OI / Volume / ATR page's engine: keeps the daily OI data fresh, serves the research and the live monitor.

`ensure_fresh()` is called by the page while it is open. It starts a background download when the
data was last updated on an earlier day (or the history length in Settings grew) and none is running;
a failed run is not retried for RETRY_SECONDS. `research()` builds the per-product research tables
from the stored data and caches them until the data changes. `monitor()` builds the live monitor for a
product and parameter set the same way. Nothing here blocks the page.

ALERTS. Once per day, after fresh data arrives (or when the app starts with today's data already stored),
every contract that has just become SUDDEN and is outside the same-month baseline (core.oi_monitor) is sent as one
alert per product, using the parameters last used on the page. Each deviation alerts once, however many
times the check runs: (contract, day the run started) is remembered in the settings table.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable

from core.curve_calendar import OI_PRODUCTS
from core.models import AlertLevel
from core.oi_monitor import STATUS_SUDDEN, MonitorEngine, MonitorParams, MonitorResult, load_params, save_params
from core.oi_research import ProductResearch, build_research
from core.atr_engine import AtrEngine, AtrParams, load_params as load_atr_params, save_params as save_atr_params
from core.oi_settings import load_atr_history_days, load_history_days

logger = logging.getLogger(__name__)

RETRY_SECONDS = 600
KEY_ALERTS_SENT = "oi_alerts_sent"
KEY_ALERTS_CHECKED = "oi_alerts_checked_on"
MAX_ALERT_MEMORY = 600
MAX_ENGINES = 8


class OiService:
    def __init__(
        self, repository, api, store, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        background: bool = True, monotonic: Callable[[], float] = time.monotonic, alert_manager=None, atr_store=None,
    ):
        self._repository = repository
        self._api = api
        self.store = store
        self._clock = clock
        self._background = background
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._research_lock = threading.Lock()  # a long research build must not block start/stop of a download
        self._engine_lock = threading.Lock()  # nor the live monitor, which is the page's main view
        self._alerts = alert_manager
        self.atr_store = atr_store  # daily OHLC for the ATR views; downloaded after the OI data, one shared progress line
        if atr_store is not None:
            atr_store.progress = store.progress
        self._atr_engines: dict[tuple, AtrEngine] = {}
        self._retry_after = 0.0
        self._research: dict[str, ProductResearch] = {}
        self._research_key: tuple | None = None
        self._engines: dict[tuple, tuple[MonitorEngine, MonitorResult]] = {}

    def history_days(self) -> int:
        return load_history_days(self._repository)

    def atr_history_days(self) -> int:
        return load_atr_history_days(self._repository)

    # -- downloading ----------------------------------------------------

    def needs_update(self) -> bool:
        today = self._clock().date().isoformat()
        loaded = self.store.loaded_history_days
        if self.store.updated_on != today or loaded is None or loaded < self.history_days() or not self.store.has_every_product():
            return True
        atr = self.atr_store
        return atr is not None and (
            atr.updated_on != today or atr.loaded_history_days is None or atr.loaded_history_days < self.atr_history_days() or not atr.has_every_product()
        )

    def _run(self) -> None:
        progress = self.store.progress
        try:
            report = self.store.update(self._api, self.history_days(), self._clock().date())
            progress.message = f"Updated: {report['contracts']} contracts, {report['rows']:,} rows, {report['requests']} requests"
            progress.error = None
            self.check_alerts()
            if self.atr_store is not None:
                atr = self.atr_store.update(self._api, self.atr_history_days(), self._clock().date())
                progress.message += f"; ATR data: {atr['instruments']} instruments, {atr['rows']:,} rows, {atr['requests']} requests"
        except Exception as exc:  # noqa: BLE001 - shown on the page; partial progress is already saved
            logger.exception("OI update failed")
            progress.error = f"{type(exc).__name__}: {exc}"
            self._retry_after = self._monotonic() + RETRY_SECONDS
        finally:
            progress.running = False
            progress.finished_at = self._clock().isoformat()

    def refresh_now(self) -> bool:
        """Start an update now unless one is running. Returns whether one was started."""
        with self._lock:
            if self.store.progress.running:
                return False
            self.store.progress.running = True
            self.store.progress.error = None
        if self._background:
            threading.Thread(target=self._run, name="oi-update", daemon=True).start()
        else:
            self._run()
        return True

    def ensure_fresh(self) -> None:
        """Start the daily update if it is due (never raises)."""
        if self._monotonic() < self._retry_after or self.store.progress.running:
            return
        try:
            if self.needs_update():
                self.refresh_now()
            elif self._repository.get_setting(KEY_ALERTS_CHECKED, None) != self._clock().date().isoformat():
                self.check_alerts()  # today's data was already stored (an app restart): still check it once
        except Exception:  # noqa: BLE001
            logger.exception("OI freshness check failed")

    # -- reading --------------------------------------------------------

    def status(self) -> dict:
        progress = self.store.progress
        return {
            "running": progress.running, "done": progress.done, "total": progress.total,
            "message": progress.message, "error": progress.error, "updated_on": self.store.updated_on,
            "history_days": self.store.loaded_history_days, "has_data": self.store.has_data(),
            "version": self.store.version + ("" if self.atr_store is None else "|" + self.atr_store.version),
        }

    def research_for(self, product: str) -> ProductResearch:
        """The research of ONE product, built on first use and cached until the stored data changes."""
        today = self._clock().date()
        key = (self.store.version, today)
        with self._research_lock:
            if key != self._research_key:
                self._research, self._research_key = {}, key
            if product not in self._research:
                self._research[product] = build_research(self.store.frame(product), product, today)
            return self._research[product]

    def research(self) -> dict[str, ProductResearch]:
        """Research of every product (each built once; see research_for)."""
        return {p: self.research_for(p) for p in OI_PRODUCTS}

    # -- live monitor ---------------------------------------------------

    def monitor_params(self) -> MonitorParams:
        return load_params(self._repository)

    def save_monitor_params(self, params: MonitorParams) -> None:
        """Remember the parameters last used on the page; the daily alert check uses them."""
        save_params(self._repository, params)

    def monitor(self, product: str, params: MonitorParams) -> tuple[MonitorEngine, MonitorResult]:
        """The monitor of one product for these parameters (cached until the stored data changes)."""
        today = self._clock().date()
        key = (self.store.version, today, product, self.history_days(), params.key())
        with self._engine_lock:
            if key not in self._engines:
                if len(self._engines) >= MAX_ENGINES:
                    self._engines.pop(next(iter(self._engines)))
                engine = MonitorEngine(self.store.frame(product), product, today, self.history_days(), params)
                self._engines[key] = (engine, engine.result())
            return self._engines[key]

    def check_alerts(self) -> int:
        """Alert on every contract that has just become SUDDEN outside the same-month baseline. Returns how many were sent."""
        if self._alerts is None:
            return 0
        today = self._clock().date().isoformat()
        params = self.monitor_params()
        if not params.alerts_enabled:
            self._repository.set_setting(KEY_ALERTS_CHECKED, today)
            return 0
        remembered = list(self._repository.get_setting(KEY_ALERTS_SENT, []) or [])
        sent = 0
        for product in OI_PRODUCTS:
            _, result = self.monitor(product, params)
            if result.rows.empty:
                continue
            fresh = result.rows[(result.rows["status"] == STATUS_SUDDEN) & result.rows["b_flag"]]  # outside the same-month baseline
            new = [row for row in fresh.itertuples() if f"{row.symbol}|{row.since}" not in remembered]
            if not new:
                continue
            lines = [
                f"{r.label}: OI {r.oi:,.0f} ({r.direction} its usual {r.b_median:,.0f}, z {r.b_z:+.1f}), "
                f"{r.run_days} day(s) so far, {r.dte} days to expiry"
                for r in new
            ]
            title = f"OI anomaly {product}: " + ", ".join(r.label for r in new)
            try:
                self._alerts.send_alert(AlertLevel.WARNING, title, "\n".join(lines), max_body_length=1900)
                remembered += [f"{r.symbol}|{r.since}" for r in new]
                sent += len(new)
            except Exception:  # noqa: BLE001 - never let alerting break the update
                logger.exception("OI alert failed")
        self._repository.set_setting(KEY_ALERTS_SENT, remembered[-MAX_ALERT_MEMORY:])
        self._repository.set_setting(KEY_ALERTS_CHECKED, today)
        return sent

    # -- ATR ------------------------------------------------------------

    def atr_params(self) -> AtrParams:
        return load_atr_params(self._repository)

    def save_atr_params(self, params: AtrParams) -> None:
        save_atr_params(self._repository, params)

    def atr_engine(self, product: str, params: AtrParams) -> AtrEngine:
        """The ATR views of one product for these parameters (cached until the stored data changes)."""
        if self.atr_store is None:
            raise RuntimeError("ATR data is not configured")
        today = self._clock().date()
        key = (self.atr_store.version, today, product, self.atr_history_days(), params.key())
        with self._engine_lock:
            if key not in self._atr_engines:
                if len(self._atr_engines) >= MAX_ENGINES:
                    self._atr_engines.pop(next(iter(self._atr_engines)))
                self._atr_engines[key] = AtrEngine(self.atr_store.frame(product), product, today, self.atr_history_days(), params)
            return self._atr_engines[key]
