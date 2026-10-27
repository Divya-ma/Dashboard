"""OI / Volume / ATR page callbacks.

The status callback also keeps the OI data fresh (starting the daily download in the background when
it is due). The research and the live monitor are rebuilt only when the stored data, the product or a
parameter changes; neither blocks the page.

The status callback runs every few seconds but must write `oi-version` ONLY when the data version has
changed: Dash re-runs a callback whenever its input is written, even with an identical value, which
rebuilt every table and chart each tick (flicker, and any zoom lost). The chart keeps its zoom between
rebuilds through a stable `uirevision`.
"""

import logging
from dataclasses import asdict
from datetime import datetime, timezone

from dash import Input, Output, State, callback_context, html, no_update
from dash.exceptions import MissingCallbackContextException

from core.curve_calendar import PRODUCTS, product_of_symbol as product_of
from core.oi_monitor import STATUS_SUDDEN, MonitorParams, validate_params
from ui.container import container
from ui.layouts.oi_monitor_view import PARAM_INPUTS, build_monitor_chart, monitor_body, years_body
from ui.layouts.oi_tab import default_month, research_body, status_text
from ui.layouts.shell import COLORS

logger = logging.getLogger(__name__)

_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_ERROR = {"color": COLORS["ACCENT_RED"]}
_PARAM_FIELDS = [field for _, _, field, _, _ in PARAM_INPUTS]


def _triggered_id():
    try:
        return callback_context.triggered_id
    except MissingCallbackContextException:  # called directly, not by Dash
        return None


def update_status(n_intervals, n_clicks, current_version=None):
    """Status line; starts the download when it is due or the button is pressed."""
    service = container.oi_service
    if service is None:
        return html.Div("The OI service is not running.", style=_MUTED), no_update
    if _triggered_id() == "oi-update-btn":
        service.refresh_now()
    else:
        service.ensure_fresh()
    status = service.status()
    version = status["version"] if status["has_data"] else "none"
    if status["running"] or version == current_version:
        version = no_update  # nothing new to render: leave the tables and charts alone
    return status_text(status), version


def _years_for_month(service, product, month):
    """The nearest live contract of this delivery month against its earlier years; None if there is none or it cannot be built."""
    try:
        engine, _ = service.monitor(product, service.monitor_params())
        symbol = None if engine._empty else engine.live_symbol_of_month(month)
        return engine.years_table(symbol) if symbol else None
    except Exception:  # noqa: BLE001 - the rest of the research must still show
        logger.exception("OI years table failed")
        return None


def render_years(symbol, params_dict):
    """The earlier years of the chosen live contract's delivery month, at today's days before expiry (Live monitor)."""
    service = container.oi_service
    product = product_of(symbol) if symbol else None
    if service is None or product is None or not params_dict:
        return years_body(None)
    try:
        engine, _ = service.monitor(product, validate_params(params_dict))
        return years_body(engine.years_table(symbol))
    except Exception:  # noqa: BLE001
        logger.exception("OI years table failed")
        return html.Div("Could not build the earlier-years table; see the log.", style=_ERROR)


def render_research(version, product, active_tab="research", month=None):
    """Research tables and charts of the chosen product and delivery month, built only when the Research tab is open.

    The research of both products is a few seconds of CPU; building it while the live monitor (the main
    tab) is showing would slow the monitor down for nothing."""
    service = container.oi_service
    if active_tab != "research":
        return no_update
    if service is None or version is None:
        return html.Div()
    try:
        product = product or "CL"
        month = int(month or default_month())
        return research_body(service.research_for(product), product, month, _years_for_month(service, product, month))
    except Exception:  # noqa: BLE001 - never leave the tab blank without a reason
        logger.exception("OI research failed")
        return html.Div("Could not build the research tables; see the log.", style=_ERROR)


def render_monitor(version, product, *values):
    """The live monitor of the chosen product. Returns (body, contract options, contract value, params)."""
    service = container.oi_service
    if service is None or version is None:
        return html.Div(), [], None, no_update
    raw = dict(zip(_PARAM_FIELDS, values[: len(_PARAM_FIELDS)]))
    raw["alerts_enabled"] = bool(values[len(_PARAM_FIELDS)]) if len(values) > len(_PARAM_FIELDS) else True
    current_contract = values[len(_PARAM_FIELDS) + 1] if len(values) > len(_PARAM_FIELDS) + 1 else None
    try:
        params = validate_params(raw)
    except ValueError as exc:
        return html.Div(f"⚠️ {exc}", style=_ERROR), no_update, no_update, no_update
    try:
        service.save_monitor_params(params)  # the daily alert check uses what was last used here
        _, result = service.monitor(product or "CL", params)
        body = monitor_body(result, params, datetime.now(timezone.utc).date())
    except Exception:  # noqa: BLE001
        logger.exception("OI monitor failed")
        return html.Div("Could not build the monitor; see the log.", style=_ERROR), no_update, no_update, no_update
    options, value = [], None
    if not result.rows.empty:
        options = [
            {"label": f"{r.label}{'  ⚠ ' + r.status if r.status != 'NORMAL' else ''}", "value": r.symbol}
            for r in result.rows.itertuples()
        ]
        symbols = [o["value"] for o in options]
        flagged = result.rows[result.rows["status"] == STATUS_SUDDEN]
        value = current_contract if current_contract in symbols else (flagged.iloc[0]["symbol"] if not flagged.empty else symbols[0])
    return body, options, value, asdict(params)


def render_chart(symbol, params_dict):
    """The chart of one live contract; its zoom survives a re-render (uirevision is the contract)."""
    service = container.oi_service
    product = product_of(symbol)
    if service is None or product is None or not params_dict:
        return build_monitor_chart(None)
    try:
        engine, _ = service.monitor(product, validate_params(params_dict))
        return build_monitor_chart(engine.chart(symbol))
    except Exception:  # noqa: BLE001
        logger.exception("OI chart failed")
        return build_monitor_chart(None)


def register_oi_callbacks(app) -> None:
    from ui.callbacks.atr_callbacks import register_atr_callbacks

    register_atr_callbacks(app)
    app.callback(
        Output("oi-status", "children"),
        Output("oi-version", "data"),
        Input("oi-refresh", "n_intervals"),
        Input("oi-update-btn", "n_clicks"),
        State("oi-version", "data"),
    )(update_status)

    app.callback(
        Output("oi-research-wrap", "children"),
        Input("oi-version", "data"),
        Input("oi-product", "value"),
        Input("oi-tabs", "active_tab"),
        Input("oi-month", "value"),
    )(render_research)

    app.callback(
        Output("oi-monitor-wrap", "children"),
        Output("oi-contract", "options"),
        Output("oi-contract", "value"),
        Output("oi-params", "data"),
        Input("oi-version", "data"),
        Input("oi-product", "value"),
        *(Input(component_id, "value") for component_id, *_ in PARAM_INPUTS),
        Input("oi-p-alerts", "value"),
        State("oi-contract", "value"),
    )(render_monitor)

    app.callback(
        Output("oi-chart", "figure"),
        Input("oi-contract", "value"),
        Input("oi-params", "data"),
    )(render_chart)

    app.callback(
        Output("oi-years-wrap", "children"),
        Input("oi-contract", "value"),
        Input("oi-params", "data"),
    )(render_years)
