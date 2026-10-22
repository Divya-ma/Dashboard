"""Dash application entry point.

Runs as `python ui/app.py`, `python -m ui.app`, or under gunicorn via
`ui.app:server`. This is the only module that instantiates the repository and
the data adapters; callbacks reach them through ui.container.
"""

import logging
import secrets
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dash  # noqa: E402
import dash_ag_grid  # noqa: E402,F401  - Dash refuses a component library first imported inside a callback (the Structures page is loaded lazily)
import dash_auth  # noqa: E402
import dash_bootstrap_components as dbc  # noqa: E402
from dash import dcc, html  # noqa: E402

from adapters.historical.vendor import VendorHistoricalAdapter  # noqa: E402
from adapters.base import SymbolTranslator  # noqa: E402
from adapters.live.fairvalue import FairValueLiveAdapter  # noqa: E402
from adapters.qh_api import make_api  # noqa: E402
from config.settings import settings  # noqa: E402
from core.alerts import AlertManager  # noqa: E402
from core.curve_compare import CurveCompare  # noqa: E402
from core.curve_history import CurveHistoryStore  # noqa: E402
from core.curve_service import CurveService  # noqa: E402
from core.curve_store import PrevSettlementStore, SnapshotStore  # noqa: E402
from core.excel_correlation import ExcelCorrelationStore  # noqa: E402
from core.atr_data import AtrStore  # noqa: E402
from core.churn_data import ChurnStore  # noqa: E402
from core.churn_service import ChurnService  # noqa: E402
from core.oi_data import OiStore  # noqa: E402
from core.oi_service import OiService  # noqa: E402
from core.journal_service import JournalService  # noqa: E402
from core.journal_snapshot import MarketSnapshotter, intraday_price_path, structure_price_path  # noqa: E402
from core.quote_store import FairValueStore, QuoteStore  # noqa: E402
from core.settlement_prices import SettlementFirstLoader, SettlementPrices  # noqa: E402
from core.structure_builder import active_composite_symbols  # noqa: E402
from core.user_settings import KEY_API_TOKEN, KEY_REQUIRE_STOP_TARGET, KEY_STALENESS  # noqa: E402
from db.repository import Repository  # noqa: E402
from ui.callbacks.correlation_callbacks import register_correlation_callbacks  # noqa: E402
from ui.callbacks.corr_matrix_callbacks import register_corr_matrix_callbacks  # noqa: E402
from ui.callbacks.exposure_callbacks import register_exposure_callbacks  # noqa: E402
from ui.callbacks.home_callbacks import register_home_callbacks  # noqa: E402
from ui.callbacks.settings_callbacks import register_settings_callbacks  # noqa: E402
from ui.callbacks.shell_callbacks import register_callbacks  # noqa: E402
from ui.callbacks.structure_detail_callbacks import register_structure_detail_callbacks  # noqa: E402
from ui.callbacks.var_callbacks import register_var_callbacks  # noqa: E402
from ui.callbacks.curve_callbacks import register_curve_callbacks  # noqa: E402
from ui.callbacks.curve_compare_callbacks import register_curve_compare_callbacks  # noqa: E402
from ui.callbacks.oi_callbacks import register_oi_callbacks  # noqa: E402
from ui.callbacks.churn_callbacks import register_churn_callbacks  # noqa: E402
from ui.callbacks.archive_callbacks import register_archive_callbacks  # noqa: E402
from ui.callbacks.structure_builder_callbacks import register_structure_builder_callbacks  # noqa: E402
from ui.callbacks.structures_callbacks import register_structures_callbacks  # noqa: E402
from ui.callbacks.dissolve_callbacks import register_dissolve_callbacks  # noqa: E402
from ui.container import container  # noqa: E402
from ui.layouts.shell import (  # noqa: E402
    build_alert_container,
    build_global_css,
    build_price_alert_container,
    build_shell,
)
from ui.layouts.structure_builder import builder_save_toast, new_structure_modal  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

LIVE_POLL_INTERVAL_MS = 10 * 1000  # /fairvalue/ has its own independent 7 req/min budget
PNL_REFRESH_INTERVAL_MS = 5 * 1000
COUNTDOWN_INTERVAL_MS = 1 * 1000


def _resolve_path(path_str: str) -> str:
    """Relative paths in settings are resolved against the project root, not the cwd."""
    path = Path(path_str)
    return str(path if path.is_absolute() else _PROJECT_ROOT / path)


# ----------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------

_startup_time = datetime.now(timezone.utc)

repository = Repository(_resolve_path(settings.DB_PATH))
repository.initialize_db()


def _is_token_configured() -> bool:
    return bool(repository.get_setting(KEY_API_TOKEN, ""))


TOKEN_CONFIGURED = _is_token_configured()

# ONE shared QH API client: every endpoint's rate-limit budget (minute/hour/day) is enforced
# in one place, whichever adapter or callback makes the call.
qh_api = make_api(lambda: repository.get_setting(KEY_API_TOKEN, ""))

live_adapter = FairValueLiveAdapter(
    repository, repository.get_setting(KEY_STALENESS, settings.LIVE_STALENESS_THRESHOLD_SECONDS), api=qh_api
)
historical_adapter = VendorHistoricalAdapter(repository, _resolve_path(settings.HISTORICAL_DATA_DIR), api=qh_api)
alert_manager = AlertManager(
    repository,
    fallback_webhook_url=settings.TEAMS_WEBHOOK_URL if settings.TEAMS_ALERTS_ENABLED else "",
)

container.repository = repository
container.live_adapter = live_adapter
container.historical_adapter = historical_adapter
container.alert_manager = alert_manager
# Settlements of CL and BRN are the app's price history; stored candles are the fallback (see SettlementFirstLoader).
_oi_store = OiStore(_resolve_path(settings.OI_DATA_DIR))
container.settlements = SettlementPrices(_oi_store)
container.data_loader = SettlementFirstLoader(_resolve_path(settings.HISTORICAL_DATA_DIR), historical_adapter, container.settlements)
container.excel_store = ExcelCorrelationStore(_resolve_path(settings.CORRELATION_UPLOADS_DIR))
container.qh_api = qh_api

_curve_dir = _resolve_path(settings.CURVE_DATA_DIR)
_prev_settlements = PrevSettlementStore(Path(_curve_dir) / "prev_settlement.json")  # shared by Curve Kinks and Curve Compare
container.curve_service = CurveService(
    repository, qh_api, CurveHistoryStore(_curve_dir, qh_api), _prev_settlements, SnapshotStore(_curve_dir), alert_manager,
)
# The curve ladder (outrights, spreads, flies of CL and BRN) is priced in the same /fairvalue/
# request as the position prices, so it costs no extra call.
live_adapter.set_curve_symbols_provider(container.curve_service.symbols_to_poll)
# Curve Compare polls /taq/ quotes only while its page is open; /fairvalue/ prices from the poll above
# fill in any symbol that has no quote.
container.curve_compare = CurveCompare(
    QuoteStore(qh_api), _prev_settlements, qh_api,
    fallback_prices=live_adapter.curve_prices, fair_values=FairValueStore(qh_api),
)
# OI / Volume / ATR page: daily OI of every outright; downloads only while that page is open.
container.oi_service = OiService(
    repository, qh_api, _oi_store,
    alert_manager=alert_manager, atr_store=AtrStore(_resolve_path(settings.ATR_DATA_DIR)),
)


def _atr_frame(product: str):
    service = container.oi_service
    return None if service is None or service.atr_store is None else service.atr_store.frame(product)


def _atr_period() -> int:
    service = container.oi_service
    return service.atr_params().period if service is not None else 14


def _kink_z(structure):
    """The structure's live Curve Kinks score for the journal (None if it is not a structure the Curve Kinks page prices)."""
    service = container.curve_service
    if service is None:
        return None
    try:
        symbols = tuple(SymbolTranslator.internal_to_api(leg.contract.symbol) for leg in structure.legs)
        return service.kink_z(symbols, tuple(leg.ratio for leg in structure.legs))
    except ValueError:
        return None


# Trade journal: the market at each entry and exit (two /ohlc/ requests, on a background thread), trade ideas and payups.
if repository.get_setting(KEY_REQUIRE_STOP_TARGET, None) is None:  # first run with this feature: a new trade needs a stop and a target (Settings can turn it off)
    repository.set_setting(KEY_REQUIRE_STOP_TARGET, True)
container.journal = JournalService(
    repository, MarketSnapshotter(qh_api, _atr_frame, _atr_period, curve_z=_kink_z), price_path=lambda structure: structure_price_path(structure, container.data_loader),
    intraday_path=lambda structure, start, end: intraday_price_path(structure, qh_api, start, end), spare_requests=lambda: qh_api.spare_requests("ohlc"),
)


def _atr_lookup(product: str) -> dict[str, float]:
    """Latest daily ATR (points) of every instrument of a product: the Churn page's holding rule reads it."""
    service = container.oi_service
    if service is None or service.atr_store is None:
        return {}
    engine = service.atr_engine(product, service.atr_params())
    if engine._empty:
        return {}
    return {symbol: float(frame["atr"].dropna().iloc[-1]) for symbol, frame in engine.atr.items() if not frame["atr"].dropna().empty}


# Churn page: five-minute candles of the live ladder; downloads only while that page is open.
container.churn_service = ChurnService(repository, qh_api, ChurnStore(_resolve_path(settings.CHURN_DATA_DIR)), atr_provider=_atr_lookup, live_prices=live_adapter.curve_prices)


def _run_morning_sync() -> None:
    try:
        extra_symbols = active_composite_symbols(repository)
        historical_adapter.run_morning_sync(repository, extra_symbols=extra_symbols)
    except Exception:  # noqa: BLE001 - a background sync failure must never take the app down
        logger.exception("Morning sync crashed")


if TOKEN_CONFIGURED:
    container.sync_thread = threading.Thread(target=_run_morning_sync, name="morning-sync", daemon=True)
    container.sync_thread.start()
else:
    logger.warning("No API token configured; skipping morning sync until a token is entered")

logger.info(
    "Startup at %s complete; TOKEN_CONFIGURED=%s", _startup_time.isoformat(), TOKEN_CONFIGURED
)

# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    assets_folder=str(_PROJECT_ROOT / "ui" / "assets"),  # AG Grid cell renderers
    suppress_callback_exceptions=True,
    title=settings.APP_TITLE,
)
server = app.server  # for gunicorn: ui.app:server
server.secret_key = settings.AUTH_SECRET_KEY or secrets.token_hex(32)

if settings.AUTH_USERNAME and settings.AUTH_PASSWORD:
    dash_auth.BasicAuth(app, {settings.AUTH_USERNAME: settings.AUTH_PASSWORD})
    logger.info("HTTP Basic Auth enabled for user %s", settings.AUTH_USERNAME)
else:
    logger.warning(
        "AUTH_USERNAME/AUTH_PASSWORD not set — the dashboard has NO login. "
        "Set them in .env before this is reachable from anywhere but localhost."
    )

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>__GLOBAL_CSS__</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
""".replace("__GLOBAL_CSS__", build_global_css())


def serve_layout() -> html.Div:
    """Built on every page load so the setup banner reflects the current token state."""
    token_configured = _is_token_configured()
    return html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="store-token-configured", data=token_configured),
            dcc.Store(id="store-live-prices", data={}),
            dcc.Store(id="store-portfolio-pnl", data={}),
            dcc.Store(id="store-selected-structure-id", data=None),
            dcc.Store(id="store-pending-delete", data=None),
            dcc.Store(id="store-builder-template", data=None),
            dcc.Store(id="store-builder-legs", data=[]),
            dcc.Store(id="store-builder-step", data=1),
            dcc.Store(id="store-builder-correlation", data=[]),
            dcc.Interval(id="interval-live-poll", interval=LIVE_POLL_INTERVAL_MS, n_intervals=0),
            dcc.Interval(id="interval-pnl-refresh", interval=PNL_REFRESH_INTERVAL_MS, n_intervals=0),
            dcc.Interval(id="interval-countdown", interval=COUNTDOWN_INTERVAL_MS, n_intervals=0),
            build_alert_container(),
            build_price_alert_container(),
            build_shell(token_configured),
            new_structure_modal(),
            builder_save_toast(),
        ]
    )


app.layout = serve_layout
register_callbacks(app)
register_home_callbacks(app)
register_settings_callbacks(app)
register_structures_callbacks(app)
register_structure_builder_callbacks(app)
register_structure_detail_callbacks(app)
register_dissolve_callbacks(app)
register_correlation_callbacks(app)
register_corr_matrix_callbacks(app)
register_exposure_callbacks(app)
register_var_callbacks(app)
register_curve_callbacks(app)
register_curve_compare_callbacks(app)
register_oi_callbacks(app)
register_churn_callbacks(app)
register_archive_callbacks(app)


if __name__ == "__main__":
    app.run(host=settings.APP_HOST, port=settings.APP_PORT, debug=settings.APP_DEBUG)
