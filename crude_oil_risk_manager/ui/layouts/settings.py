"""Settings tab layout: API token, thresholds, analysis defaults, Teams alerts and data management.

Field values here are only initial defaults; ui/callbacks/settings_callbacks.py
loads the saved values every time the tab is opened.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from config.settings import settings
from core.user_settings import KEY_API_TOKEN
from ui.container import container
from ui.layouts.shell import COLORS

_LABEL_STYLE = {"color": COLORS["TEXT_PRIMARY"], "marginBottom": "4px"}
_STATUS_STYLE = {"color": COLORS["TEXT_PRIMARY"], "marginTop": "10px", "minHeight": "24px"}
_HINT_STYLE = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "12px"}


def _field(label: str, component, hint: str | None = None) -> html.Div:
    children = [dbc.Label(label, html_for=getattr(component, "id", None), style=_LABEL_STYLE), component]
    if hint:
        children.append(html.Div(hint, style=_HINT_STYLE))
    return html.Div(children, className="mb-3")


def _number_input(component_id: str, value, **kwargs) -> dbc.Input:
    return dbc.Input(id=component_id, type="number", value=value, debounce=False, **kwargs)


def _api_section() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            _field(
                "Bearer Token",
                dbc.Input(id="settings-api-token", type="password", placeholder="Enter your Bearer token"),
            ),
            dbc.Switch(id="settings-token-show", label="Show token", value=False, className="mb-3"),
            dbc.Button("Save Token", id="settings-save-token", color="primary", className="me-2"),
            dbc.Button("Test Connection", id="settings-test-connection", color="secondary"),
            html.Div("❌ Not configured", id="settings-token-status", style=_STATUS_STYLE),
        ],
        title="🔑 API Configuration",
        item_id="api",
    )


def _thresholds_section() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            _field(
                "Portfolio PnL Stop ($)",
                _number_input("settings-portfolio-pnl-stop", settings.ALERT_PORTFOLIO_PNL_STOP),
                "Negative number. An alert fires when total PnL falls below it.",
            ),
            _field(
                "Per-Structure Max Loss ($)",
                _number_input("settings-structure-max-loss", settings.ALERT_STRUCTURE_MAX_LOSS),
                "Negative number.",
            ),
            _field(
                "Data Staleness Threshold (seconds)",
                _number_input("settings-staleness-threshold", settings.LIVE_STALENESS_THRESHOLD_SECONDS),
                "A live price older than this is flagged stale.",
            ),
            _field(
                "Portfolio Margin Limit ($)",
                _number_input("settings-margin-limit", settings.DEFAULT_MARGIN_LIMIT),
            ),
            _field(
                "Roll Warning (days before expiry)",
                _number_input("settings-roll-warning-days", settings.ROLL_WARNING_DAYS_BEFORE_EXPIRY),
            ),
            dbc.Button("Save Thresholds", id="settings-save-thresholds", color="primary"),
            html.Div("", id="settings-thresholds-feedback", style=_STATUS_STYLE),
        ],
        title="⚠️ Risk & Alert Thresholds",
        item_id="thresholds",
    )


def _defaults_section() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            _field(
                "VaR Confidence Level",
                dbc.RadioItems(
                    id="settings-var-confidence",
                    options=[{"label": "95%", "value": 0.95}, {"label": "99%", "value": 0.99}],
                    value=0.95,
                    inline=True,
                ),
            ),
            _field(
                "Default Correlation Window (days)",
                _number_input("settings-correlation-window", settings.DEFAULT_CORRELATION_WINDOW),
                "Between 20 and 500.",
            ),
            dbc.Button("Save Defaults", id="settings-save-defaults", color="primary"),
            html.Div("", id="settings-defaults-feedback", style=_STATUS_STYLE),
        ],
        title="📊 Analysis Defaults",
        item_id="defaults",
    )


def _teams_section() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            _field(
                "Teams Incoming Webhook URL",
                dbc.Input(id="settings-teams-webhook", type="password", placeholder="https://..."),
            ),
            dbc.Switch(id="settings-teams-enabled", label="Enable Teams Alerts", value=False, className="mb-3"),
            dbc.Button("Test Teams Alert", id="settings-test-teams", color="secondary", className="me-2"),
            dbc.Button("Save Teams Settings", id="settings-save-teams", color="primary"),
            html.Div("", id="settings-teams-status", style=_STATUS_STYLE),
        ],
        title="📣 Microsoft Teams Alerts",
        item_id="teams",
    )


def _data_section() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            html.Div("Morning Sync Status", style=_LABEL_STYLE),
            html.Div("", id="settings-sync-status", className="mb-3", style={"color": COLORS["TEXT_PRIMARY"]}),
            dbc.Button("Run Morning Sync Now", id="settings-run-sync", color="secondary", className="mb-4"),
            html.Div("Local Data Summary", style=_LABEL_STYLE),
            html.Div("", id="settings-data-summary", style={"color": COLORS["TEXT_PRIMARY"]}),
            dcc.Interval(id="settings-data-refresh", interval=3000, n_intervals=0),
        ],
        title="💾 Data Management",
        item_id="data",
    )


def settings_layout() -> html.Div:
    """Settings tab: collapsible sections, with API Configuration open until a token is set."""
    token_configured = bool(container.repository.get_setting(KEY_API_TOKEN, "")) if container.repository else False
    return html.Div(
        [
            html.H4("⚙️ Settings", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "20px"}),
            dbc.Accordion(
                [_api_section(), _thresholds_section(), _defaults_section(), _teams_section(), _data_section()],
                id="settings-accordion",
                always_open=True,
                active_item=["thresholds"] if token_configured else ["api"],
            ),
        ],
        style={"maxWidth": "760px"},
    )
