"""Dash application entry point.

Runs as `python ui/app.py`, `python -m ui.app`, or under gunicorn via
`ui.app:server`. This is the only module that instantiates the repository and
the data adapters; callbacks reach them through ui.container.
"""

import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dash  # noqa: E402
import dash_bootstrap_components as dbc  # noqa: E402
from dash import dcc, html  # noqa: E402

from adapters.historical.vendor import VendorHistoricalAdapter  # noqa: E402
from adapters.live.vendor import VendorLiveAdapter  # noqa: E402
from config.settings import settings  # noqa: E402
from core.alerts import AlertManager  # noqa: E402
from db.repository import Repository  # noqa: E402
from ui.callbacks.shell_callbacks import register_callbacks  # noqa: E402
from ui.container import container  # noqa: E402
from ui.layouts.shell import build_alert_container, build_global_css, build_shell  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TOKEN_SETTING_KEY = "api_access_token"
LIVE_POLL_INTERVAL_MS = 60 * 1000
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
    return bool(repository.get_setting(TOKEN_SETTING_KEY, ""))


TOKEN_CONFIGURED = _is_token_configured()

live_adapter = VendorLiveAdapter(repository, settings.LIVE_STALENESS_THRESHOLD_SECONDS)
# One of the API's 7 calls/minute is reserved for the live price poll.
historical_adapter = VendorHistoricalAdapter(
    repository,
    _resolve_path(settings.HISTORICAL_DATA_DIR),
    calls_per_minute=max(1, settings.HISTORICAL_API_CALLS_PER_MINUTE - 1),
)
alert_manager = AlertManager(
    repository,
    fallback_webhook_url=settings.TEAMS_WEBHOOK_URL if settings.TEAMS_ALERTS_ENABLED else "",
)

container.repository = repository
container.live_adapter = live_adapter
container.historical_adapter = historical_adapter
container.alert_manager = alert_manager


def _run_morning_sync() -> None:
    try:
        historical_adapter.run_morning_sync(repository)
    except Exception:  # noqa: BLE001 - a background sync failure must never take the app down
        logger.exception("Morning sync crashed")


if TOKEN_CONFIGURED:
    threading.Thread(target=_run_morning_sync, name="morning-sync", daemon=True).start()
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
    suppress_callback_exceptions=True,
    title=settings.APP_TITLE,
)
server = app.server  # for gunicorn: ui.app:server

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
            dcc.Interval(id="interval-live-poll", interval=LIVE_POLL_INTERVAL_MS, n_intervals=0),
            dcc.Interval(id="interval-pnl-refresh", interval=PNL_REFRESH_INTERVAL_MS, n_intervals=0),
            dcc.Interval(id="interval-countdown", interval=COUNTDOWN_INTERVAL_MS, n_intervals=0),
            build_alert_container(),
            build_shell(token_configured),
        ]
    )


app.layout = serve_layout
register_callbacks(app)


if __name__ == "__main__":
    app.run(host=settings.APP_HOST, port=settings.APP_PORT, debug=settings.APP_DEBUG)
