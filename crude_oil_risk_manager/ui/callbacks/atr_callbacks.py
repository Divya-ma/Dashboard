"""ATR tab callbacks: the tables are built only when the ATR tab is open, from the cached engine of
core.atr_engine; the chart of one contract keeps its zoom through a stable uirevision."""

import logging
from dataclasses import asdict

from dash import Input, Output, State, html, no_update

from core.atr_engine import validate_params
from ui.container import container
from ui.layouts.atr_view import PARAM_INPUTS, atr_body, build_atr_chart
from ui.layouts.shell import COLORS

logger = logging.getLogger(__name__)

_ERROR = {"color": COLORS["ACCENT_RED"]}
_FIELDS = [field for _, _, field, _, _ in PARAM_INPUTS]


def product_of(symbol: str | None) -> str | None:
    """CL or BRN from a contract symbol (BRN contracts are CO...)."""
    if not symbol:
        return None
    return "BRN" if symbol.startswith("CO") else "CL"


def render_atr(version, product, family, unit, month, *values):
    """Returns (body, contract options, contract value, params) for the chosen product, curve and unit."""
    service = container.oi_service
    *param_values, skip_idle, active_tab, current = values
    if active_tab != "atr" or service is None or service.atr_store is None or version is None:
        return no_update, no_update, no_update, no_update
    raw = dict(zip(_FIELDS, param_values))
    raw["skip_idle"] = bool(skip_idle)
    try:
        params = validate_params(raw)
    except ValueError as exc:
        return html.Div(f"⚠️ {exc}", style=_ERROR), no_update, no_update, no_update
    product, family = product or "CL", family or "outright"
    try:
        service.save_atr_params(params)
        engine = service.atr_engine(product, params)
        body = atr_body(engine, family, unit or "atr", int(month or 1), params)
        live = engine.live_table(family) if not engine._empty else None
    except Exception:  # noqa: BLE001
        logger.exception("ATR tab failed")
        return html.Div("Could not build the ATR tables; see the log.", style=_ERROR), no_update, no_update, no_update
    options, value = [], None
    if live is not None and not live.empty:
        options = [{"label": f"{r.label}{'  ⚠ ' + r.status if r.status else ''}", "value": r.symbol} for r in live.itertuples()]
        symbols = [o["value"] for o in options]
        flagged = live[live["status"] != ""]
        value = current if current in symbols else (flagged.iloc[0]["symbol"] if not flagged.empty else symbols[0])
    return body, options, value, asdict(params)


def render_atr_chart(symbol, params_dict, unit):
    service = container.oi_service
    product = product_of(symbol)
    if service is None or service.atr_store is None or product is None or not params_dict:
        return build_atr_chart(None)
    try:
        engine = service.atr_engine(product, validate_params(params_dict))
        return build_atr_chart(engine.chart(symbol, unit or "atr"))
    except Exception:  # noqa: BLE001
        logger.exception("ATR chart failed")
        return build_atr_chart(None)


def register_atr_callbacks(app) -> None:
    app.callback(
        Output("atr-wrap", "children"),
        Output("atr-contract", "options"),
        Output("atr-contract", "value"),
        Output("atr-params", "data"),
        Input("oi-version", "data"),
        Input("oi-product", "value"),
        Input("atr-family", "value"),
        Input("atr-unit", "value"),
        Input("atr-month", "value"),
        *(Input(component_id, "value") for component_id, *_ in PARAM_INPUTS),
        Input("atr-p-idle", "value"),
        Input("oi-tabs", "active_tab"),
        State("atr-contract", "value"),
    )(render_atr)

    app.callback(
        Output("atr-chart", "figure"),
        Input("atr-contract", "value"),
        Input("atr-params", "data"),
        Input("atr-unit", "value"),
    )(render_atr_chart)
