"""Settings tab callbacks: load/save settings, test the API and Teams, run the morning sync.

Values are validated by core.user_settings before anything is stored. The API
token and Teams webhook are secrets: they are never logged, and error messages
shown to the user never contain them.
"""

import logging
import threading

from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from adapters.base import APIError
from config.settings import settings
from core.user_settings import (
    KEY_API_TOKEN,
    KEY_CORRELATION_WINDOW,
    KEY_MARGIN_LIMIT,
    KEY_PNL_STOP,
    KEY_ROLL_WARNING_DAYS,
    KEY_STALENESS,
    KEY_STRUCTURE_MAX_LOSS,
    KEY_TEAMS_ENABLED,
    KEY_TEAMS_WEBHOOK,
    KEY_VAR_CONFIDENCE,
    validate_defaults,
    validate_thresholds,
    validate_token,
    validate_webhook_url,
)
from ui.container import container

logger = logging.getLogger(__name__)

CONNECTION_TEST_SYMBOL = "CLZ26"

STATUS_CONFIGURED = "✅ Token configured"
STATUS_NOT_CONFIGURED = "❌ Not configured"
STATUS_TESTING = "🔄 Testing..."

SAVE_THRESHOLDS_LABEL = "Save Thresholds"
SAVE_DEFAULTS_LABEL = "Save Defaults"
SAVE_TEAMS_LABEL = "Save Teams Settings"

_THRESHOLD_FIELDS = [
    "settings-portfolio-pnl-stop",
    "settings-structure-max-loss",
    "settings-staleness-threshold",
    "settings-margin-limit",
    "settings-roll-warning-days",
]
_DEFAULT_FIELDS = ["settings-var-confidence", "settings-correlation-window"]
_TEAMS_FIELDS = ["settings-teams-webhook", "settings-teams-enabled"]


def _token_status(token: str) -> str:
    return STATUS_CONFIGURED if token else STATUS_NOT_CONFIGURED


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------


def load_settings(pathname):
    """Fill every settings field with its saved value (only when the Settings tab is open)."""
    if pathname != "/settings":
        raise PreventUpdate

    repo = container.repository
    token = repo.get_setting(KEY_API_TOKEN, "")
    webhook = repo.get_setting(KEY_TEAMS_WEBHOOK, "")

    # Teams sends whenever a webhook is saved unless explicitly disabled, so mirror that here.
    teams_enabled = repo.get_setting(KEY_TEAMS_ENABLED, None)
    if teams_enabled is None:
        teams_enabled = bool(webhook) or settings.TEAMS_ALERTS_ENABLED

    return (
        token,
        repo.get_setting(KEY_PNL_STOP, settings.ALERT_PORTFOLIO_PNL_STOP),
        repo.get_setting(KEY_STRUCTURE_MAX_LOSS, settings.ALERT_STRUCTURE_MAX_LOSS),
        repo.get_setting(KEY_STALENESS, settings.LIVE_STALENESS_THRESHOLD_SECONDS),
        repo.get_setting(KEY_MARGIN_LIMIT, settings.DEFAULT_MARGIN_LIMIT),
        repo.get_setting(KEY_ROLL_WARNING_DAYS, settings.ROLL_WARNING_DAYS_BEFORE_EXPIRY),
        repo.get_setting(KEY_VAR_CONFIDENCE, settings.DEFAULT_VAR_CONFIDENCE),
        repo.get_setting(KEY_CORRELATION_WINDOW, settings.DEFAULT_CORRELATION_WINDOW),
        webhook,
        bool(teams_enabled),
        _token_status(token),
    )


# ----------------------------------------------------------------------
# API token
# ----------------------------------------------------------------------


def toggle_token_visibility(show):
    """Show the token as text or hide it as a password field."""
    return "text" if show else "password"


def save_token(n_clicks, token_value):
    """Validate and store the Bearer token, clear the setup banner and force a fresh price poll."""
    try:
        token = validate_token(token_value)
    except ValueError as exc:
        return f"❌ {exc}", no_update, no_update

    container.repository.set_setting(KEY_API_TOKEN, token)
    container.live_cache.fetched_at = None  # next poll must use the new token, not the cache
    logger.info("API token saved")
    return "✅ Token saved successfully", True, {"display": "none"}


def test_connection(n_clicks, token_value):
    """Try one real API call with the typed (or, if blank, the saved) token. Nothing is stored."""
    try:
        token = validate_token(token_value) if token_value and token_value.strip() else None
    except ValueError as exc:
        return f"❌ {exc}"
    token = token or container.repository.get_setting(KEY_API_TOKEN, "")
    if not token:
        return "❌ Enter a token first"

    try:
        candles = container.live_adapter.check_connection(token, CONNECTION_TEST_SYMBOL)
    except (APIError, RuntimeError, ValueError) as exc:
        return f"❌ Connection failed: {exc}"
    if candles == 0:
        return f"✅ Connection successful (token accepted; no data returned for {CONNECTION_TEST_SYMBOL})"
    return "✅ Connection successful"


# ----------------------------------------------------------------------
# Thresholds / defaults
# ----------------------------------------------------------------------


def save_thresholds(n_clicks, pnl_stop, structure_max_loss, staleness, margin_limit, roll_days):
    """Validate and store the risk thresholds; the staleness threshold takes effect immediately."""
    try:
        values = validate_thresholds(pnl_stop, structure_max_loss, staleness, margin_limit, roll_days)
    except ValueError as exc:
        return "❌ Not saved", f"❌ {exc}"

    for key, value in values.items():
        container.repository.set_setting(key, value)

    adapter = container.live_adapter
    if hasattr(adapter, "set_staleness_threshold"):
        adapter.set_staleness_threshold(values[KEY_STALENESS])
    container.live_cache.fetched_at = None  # re-evaluate staleness on the next poll
    return "✅ Saved", ""


def save_defaults(n_clicks, var_confidence, correlation_window):
    """Validate and store the analysis defaults."""
    try:
        values = validate_defaults(var_confidence, correlation_window)
    except ValueError as exc:
        return "❌ Not saved", f"❌ {exc}"

    for key, value in values.items():
        container.repository.set_setting(key, value)
    return "✅ Saved", ""


def _reset_label(label: str):
    def reset(*_values):
        """Put the save button's label back once a field is edited after saving."""
        return label

    return reset


# ----------------------------------------------------------------------
# Teams
# ----------------------------------------------------------------------


def save_teams(n_clicks, webhook_url, enabled):
    """Validate and store the Teams webhook and enabled flag."""
    try:
        webhook = validate_webhook_url(webhook_url)
    except ValueError as exc:
        return "❌ Not saved", f"❌ {exc}"

    container.repository.set_setting(KEY_TEAMS_WEBHOOK, webhook)
    container.repository.set_setting(KEY_TEAMS_ENABLED, bool(enabled))
    container.alert_manager.clear_cache()

    if enabled and not webhook:
        return "✅ Saved", "✅ Saved — no webhook URL set, so no Teams alerts will be sent"
    return "✅ Saved", "✅ Teams settings saved"


def test_teams(n_clicks, webhook_url):
    """Send a test message to Teams using the typed webhook (or, if blank, the saved one)."""
    try:
        webhook = validate_webhook_url(webhook_url) or None
    except ValueError as exc:
        return f"❌ Failed: {exc}"

    manager = container.alert_manager
    if manager.send_test_alert(webhook_url=webhook):
        return "✅ Test message sent"
    return f"❌ Failed: {manager.last_error or 'unknown error'}"


# ----------------------------------------------------------------------
# Data management
# ----------------------------------------------------------------------


def _run_sync() -> None:
    try:
        container.historical_adapter.run_morning_sync(container.repository)
    except Exception:  # noqa: BLE001 - a background sync failure must never take the app down
        logger.exception("Manual morning sync crashed")


def run_sync_now(n_clicks):
    """Start the morning sync on a background thread (one at a time)."""
    if not container.repository.get_setting(KEY_API_TOKEN, ""):
        return "❌ Configure your API token before running a sync"

    running = container.sync_thread
    if running is not None and running.is_alive():
        return "🔄 A sync is already running..."

    thread = threading.Thread(target=_run_sync, name="morning-sync-manual", daemon=True)
    container.sync_thread = thread
    thread.start()
    return "🔄 Sync started in background..."


def _format_size(total_bytes: int) -> str:
    if total_bytes < 1024 * 1024:
        return f"{total_bytes / 1024:.1f} KB"
    return f"{total_bytes / (1024 * 1024):.1f} MB"


def refresh_data_status(n_intervals):
    """Morning sync status and local cache summary; polled while the Settings tab is open."""
    adapter = container.historical_adapter
    thread = container.sync_thread

    if thread is not None and thread.is_alive():
        sync_text = "🔄 Sync running..."
    else:
        summary = adapter.get_sync_summary()
        if summary["last_sync_utc"] is None:
            sync_text = "No sync has run yet"
        else:
            errors = f", {summary['symbols_error']} with errors" if summary["symbols_error"] else ""
            sync_text = (
                f"Last sync: {summary['last_sync_utc'].strftime('%Y-%m-%d %H:%M UTC')} · "
                f"{summary['symbols_tracked']} symbols ({summary['symbols_ok']} ok{errors})"
            )

    data = adapter.get_local_data_summary()
    data_text = f"{data['symbol_count']} cached symbols · {_format_size(data['total_bytes'])}"
    return sync_text, data_text


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_settings_callbacks(app) -> None:
    """Attach the Settings tab callbacks to the Dash app."""
    app.callback(
        Output("settings-api-token", "value"),
        Output("settings-portfolio-pnl-stop", "value"),
        Output("settings-structure-max-loss", "value"),
        Output("settings-staleness-threshold", "value"),
        Output("settings-margin-limit", "value"),
        Output("settings-roll-warning-days", "value"),
        Output("settings-var-confidence", "value"),
        Output("settings-correlation-window", "value"),
        Output("settings-teams-webhook", "value"),
        Output("settings-teams-enabled", "value"),
        Output("settings-token-status", "children"),
        Input("url", "pathname"),
    )(load_settings)

    app.callback(
        Output("settings-api-token", "type"),
        Input("settings-token-show", "value"),
    )(toggle_token_visibility)

    app.callback(
        Output("settings-token-status", "children", allow_duplicate=True),
        Output("store-token-configured", "data", allow_duplicate=True),
        Output("banner-setup-required", "style", allow_duplicate=True),
        Input("settings-save-token", "n_clicks"),
        State("settings-api-token", "value"),
        prevent_initial_call=True,
    )(save_token)

    app.callback(
        Output("settings-token-status", "children", allow_duplicate=True),
        Input("settings-test-connection", "n_clicks"),
        State("settings-api-token", "value"),
        prevent_initial_call=True,
        running=[
            (Output("settings-test-connection", "disabled"), True, False),
            (Output("settings-token-status", "children"), STATUS_TESTING, ""),
        ],
    )(test_connection)

    app.callback(
        Output("settings-save-thresholds", "children"),
        Output("settings-thresholds-feedback", "children"),
        Input("settings-save-thresholds", "n_clicks"),
        *[State(field, "value") for field in _THRESHOLD_FIELDS],
        prevent_initial_call=True,
    )(save_thresholds)

    app.callback(
        Output("settings-save-defaults", "children"),
        Output("settings-defaults-feedback", "children"),
        Input("settings-save-defaults", "n_clicks"),
        *[State(field, "value") for field in _DEFAULT_FIELDS],
        prevent_initial_call=True,
    )(save_defaults)

    app.callback(
        Output("settings-save-teams", "children"),
        Output("settings-teams-status", "children", allow_duplicate=True),
        Input("settings-save-teams", "n_clicks"),
        *[State(field, "value") for field in _TEAMS_FIELDS],
        prevent_initial_call=True,
    )(save_teams)

    app.callback(
        Output("settings-teams-status", "children"),
        Input("settings-test-teams", "n_clicks"),
        State("settings-teams-webhook", "value"),
        prevent_initial_call=True,
        running=[(Output("settings-test-teams", "disabled"), True, False)],
    )(test_teams)

    # Editing a field after saving puts the save button's label back.
    for button_id, fields, label in [
        ("settings-save-thresholds", _THRESHOLD_FIELDS, SAVE_THRESHOLDS_LABEL),
        ("settings-save-defaults", _DEFAULT_FIELDS, SAVE_DEFAULTS_LABEL),
        ("settings-save-teams", _TEAMS_FIELDS, SAVE_TEAMS_LABEL),
    ]:
        app.callback(
            Output(button_id, "children", allow_duplicate=True),
            *[Input(field, "value") for field in fields],
            prevent_initial_call=True,
        )(_reset_label(label))

    app.callback(
        Output("settings-sync-status", "children", allow_duplicate=True),
        Input("settings-run-sync", "n_clicks"),
        prevent_initial_call=True,
    )(run_sync_now)

    app.callback(
        Output("settings-sync-status", "children"),
        Output("settings-data-summary", "children"),
        Input("settings-data-refresh", "n_intervals"),
    )(refresh_data_status)
