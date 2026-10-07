"""Settings tab layout: API token, thresholds, analysis defaults, Teams alerts and data management.

Field values here are only initial defaults; ui/callbacks/settings_callbacks.py
loads the saved values every time the tab is opened.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from config.settings import settings
from core.curve_settings import DEFAULT_OPEN_TIME, DEFAULT_OPEN_TZ, TIMEZONE_CHOICES
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
                "Correlation Window (days)",
                _number_input("settings-correlation-window", settings.DEFAULT_CORRELATION_WINDOW),
                "Between 20 and 500. A single point-in-time correlation: the Structure Builder's "
                "correlation check and the Correlation tab's Heatmap lookback.",
            ),
            _field(
                "Rolling Correlation Window (days)",
                _number_input("settings-rolling-correlation-window", settings.DEFAULT_ROLLING_CORRELATION_WINDOW),
                "Between 20 and 500. A different calculation: the trailing window for the "
                "Correlation tab's Time Series, Year Overlay and Summary Table views.",
            ),
            dbc.Button("Save Defaults", id="settings-save-defaults", color="primary"),
            html.Div("", id="settings-defaults-feedback", style=_STATUS_STYLE),
        ],
        title="📊 Analysis Defaults",
        item_id="defaults",
    )


# (field, label, hint, step) for the trade-plan settings, in the order shown
CURVE_TRADE_FIELDS = [
    ("risk_per_trade", "Risk per trade ($)", "Most you will lose at the stop on the whole entry; sets the number of lots", 100),
    ("risk_per_lot", "Risk per lot ($)", "Most you will lose per lot at the stop; caps how far the stop can be", 50),
    ("point_value", "Point value ($ per point per lot)", "1,000 for CL and Brent (1,000 barrels)", 1),
    ("max_lots", "Max lots", "Never suggest more than this, however tight the stop", 1),
    ("vol_stop_mult", "Volatility stop (x daily move)", "Stop distance is this many daily moves, but never beyond the per-lot cap", 0.1),
    ("vol_window", "Volatility window (days)", "Days of daily moves behind that volatility", 5),
    ("reversion_horizon", "Reversion horizon (days)", "How long a kink is given to close when estimating the target", 1),
    ("min_reversion_fraction", "Minimum share of the gap in the target", "The target takes at least this share of the gap to fair value (0.05 to 1)", 0.05),
    ("min_reward_risk", "Minimum reward:risk for an alert", "0 = off. Otherwise no alert unless the plan reaches this", 0.1),
    ("hedge_count", "Hedge alternatives shown", "Ranked best first; at least one is always shown", 1),
    ("hedge_min_corr", "Minimum hedge correlation", "A hedge should reach this |correlation| to be listed (0 to 0.99)", 0.05),
    ("hedge_lookback", "Hedge lookback (days)", "Days of daily changes for correlation and the VaR ratio", 10),
]
CURVE_HEDGE_FAMILIES = [("outright", "Outright"), ("spread", "Spread"), ("fly", "Fly"), ("dfly", "Dfly")]


def _curve_trade_fields() -> list:
    """Risk appetite, sizing and hedge rules for the Curve Kinks trade plan."""
    fields = [
        html.Hr(style={"borderColor": COLORS["BORDER_COLOR"]}),
        html.Div("Trade plan: risk, sizing and hedges", style={**_LABEL_STYLE, "fontWeight": "bold", "marginBottom": "10px"}),
    ]
    fields += [
        _field(label, dbc.Input(id=f"settings-curve-{name}", type="number", step=step), hint)
        for name, label, hint, step in CURVE_TRADE_FIELDS
    ]
    fields.append(dbc.Switch(
        id="settings-curve-hedge_exclude_overlap", label="Hedges must not share a contract with the kinked structure",
        value=True, className="mb-3",
    ))
    fields.append(html.Div("Which curve types may hedge each kind of kink (same product only)", style=_LABEL_STYLE))
    for family, label in CURVE_HEDGE_FAMILIES:
        fields.append(dbc.Row([
            dbc.Col(html.Div(f"{label} kink", style={"color": COLORS["TEXT_PRIMARY"], "paddingTop": "4px"}), md=4),
            dbc.Col(dbc.Checklist(
                id=f"settings-curve-hedge-{family}", options=[{"label": n, "value": f} for f, n in CURVE_HEDGE_FAMILIES],
                value=[], inline=True,
            ), md=8),
        ], className="mb-1"))
    fields += [
        dbc.Button("Save Trade Plan Settings", id="settings-save-curve-trade", color="primary", className="mt-3"),
        html.Div("", id="settings-curve-trade-feedback", style=_STATUS_STYLE),
    ]
    return fields


def _curve_section() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            _field(
                "Market opening time (HH:MM)",
                dbc.Input(id="settings-curve-open-time", type="text", value=DEFAULT_OPEN_TIME, placeholder="07:00"),
                "Once a day, from this time, the previous trading day's settlement prices are fetched "
                "for the Curve Kinks page (the dashed line on each curve).",
            ),
            _field(
                "Time zone",
                dcc.Dropdown(
                    id="settings-curve-open-tz",
                    options=[{"label": tz, "value": tz} for tz in TIMEZONE_CHOICES],
                    value=DEFAULT_OPEN_TZ,
                    clearable=False,
                ),
            ),
            dbc.Button("Save Curve Settings", id="settings-save-curve", color="primary"),
            html.Div("", id="settings-curve-feedback", style=_STATUS_STYLE),
            *_curve_trade_fields(),
        ],
        title="🎯 Curve Kinks",
        item_id="curve",
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


def _account_reset_section() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        [
            html.Div(
                "Zero out the cumulative Total PnL shown on Home (and used by the Portfolio PnL Stop "
                "alert) without touching any saved structures, trades or transaction costs — use this "
                "to start tracking performance from a clean slate.",
                style={**_HINT_STYLE, "marginBottom": "12px"},
            ),
            html.Div("", id="settings-pnl-baseline-status", style={**_STATUS_STYLE, "marginTop": 0, "marginBottom": "12px"}),
            dbc.Button("Reset Total PnL to Zero", id="settings-reset-pnl-btn", color="warning", outline=True, className="me-2"),
            dbc.Button("Clear Reset (Show True Total)", id="settings-clear-pnl-reset-btn", color="secondary", outline=True),
            dbc.Collapse(
                html.Div(
                    [
                        html.Div(id="settings-reset-pnl-confirm-text", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "10px"}),
                        dbc.Button("Confirm: Reset to Zero", id="settings-confirm-reset-pnl-btn", color="danger", size="sm", className="me-2"),
                        dbc.Button("Cancel", id="settings-cancel-reset-pnl-btn", color="secondary", outline=True, size="sm"),
                    ],
                    style={"marginTop": "12px", "padding": "12px", "border": f"1px solid {COLORS['ACCENT_YELLOW']}", "borderRadius": "6px"},
                ),
                id="settings-reset-pnl-collapse",
                is_open=False,
            ),
        ],
        title="🔄 Account Reset",
        item_id="account-reset",
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
                [_api_section(), _thresholds_section(), _defaults_section(), _curve_section(), _teams_section(), _account_reset_section(), _data_section()],
                id="settings-accordion",
                always_open=True,
                active_item=["thresholds"] if token_configured else ["api"],
            ),
        ],
        style={"maxWidth": "760px"},
    )
