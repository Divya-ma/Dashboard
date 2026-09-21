"""Simple dependency container shared between ui/app.py and the callback modules.

ui/app.py is the ONLY place adapters and the repository are instantiated; it
populates the module-level `container` at startup. Callback modules import
`container` from here, so they never need to import ui.app (which would be
circular, and would re-run startup when app.py is launched as a script).
"""

import threading
from dataclasses import dataclass, field

from adapters.base import HistoricalDataAdapter, LiveDataAdapter
from core.alerts import AlertManager
from db.repository import Repository

NO_UPDATE_LABEL = "Last: --:--:--"


@dataclass
class LivePriceCache:
    """Server-side copy of the latest live poll, so page loads don't hit the API again."""

    prices: dict = field(default_factory=dict)
    fetched_at: float | None = None  # time.monotonic() of the last successful poll
    updated_label: str = NO_UPDATE_LABEL
    is_stale: bool = False


@dataclass
class Container:
    repository: Repository | None = None
    live_adapter: LiveDataAdapter | None = None
    historical_adapter: HistoricalDataAdapter | None = None
    alert_manager: AlertManager | None = None
    live_cache: LivePriceCache = field(default_factory=LivePriceCache)
    pnl_stop_alert_active: bool = False
    sync_thread: threading.Thread | None = None  # latest morning-sync thread (startup or manual)


container = Container()
