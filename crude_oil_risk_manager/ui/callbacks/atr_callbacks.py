"""ATR tab callbacks: the tables are built only when the ATR tab is open, from the cached engine of
core.atr_engine; the chart of one contract keeps its zoom through a stable uirevision."""

import logging
from dataclasses import asdict

from dash import Input, Output, State, html, no_update

from core.atr_engine import pair_hedge, validate_params
from core.contract_specs import spec_for_product
from core.curve_calendar import PRODUCTS, product_of_symbol as product_of
from ui.container import container
from ui.layouts.atr_view import PARAM_INPUTS, atr_body, build_atr_chart, hedge_calculator_result
from ui.layouts.shell import COLORS

logger = logging.getLogger(__name__)

_ERROR = {"color": COLORS["ACCENT_RED"]}
_FIELDS = [field for _, _, field, _, _ in PARAM_INPUTS]


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


def hedge_options(version, active_tab, params_dict, current_a, current_b):
    """The live instruments of every product for both pickers (a pick is kept while it is still listed)."""
    service = container.oi_service
    if active_tab != "atr" or service is None or service.atr_store is None or version is None or not params_dict:
        return no_update, no_update, no_update, no_update
    try:
        params = validate_params(params_dict)
        options = []
        for product in PRODUCTS:
            engine = service.atr_engine(product, params)
            options += [{"label": f"{product} {family}: {label}", "value": symbol} for family, symbol, label in engine.all_live_symbols()]
    except Exception:  # noqa: BLE001
        logger.exception("Hedge calculator options failed")
        return no_update, no_update, no_update, no_update
    values = [o["value"] for o in options]
    a = current_a if current_a in values else (values[0] if values else None)
    b = current_b if current_b in values and current_b != a else next((v for v in values if v != a), None)
    return options, options, a, b


def render_hedge(a, b, lots, window, params_dict):
    """The hedge of contract A by contract B (either may be CL, BRN or G, of any family)."""
    service = container.oi_service
    muted = {"color": COLORS["TEXT_SECONDARY"]}
    if service is None or service.atr_store is None or not a or not b or not params_dict:
        return html.Div("Pick two contracts.", style=muted)
    if a == b:
        return html.Div("Pick two different contracts.", style=muted)
    try:
        lots = float(lots) if lots not in (None, "") else 1.0
        window = int(window) if window not in (None, "") else 60
        if lots <= 0 or not 2 <= window <= 5000:
            raise ValueError
    except (TypeError, ValueError):
        return html.Div("⚠️ Lots must be above 0 and the lookback between 2 and 5,000 days.", style=_ERROR)
    try:
        params = validate_params(params_dict)
        frames, names = {}, {}
        for symbol in (a, b):
            product = product_of(symbol)
            engine = service.atr_engine(product, params)
            frames[symbol] = engine.pair_frame(symbol)
            names[symbol] = f"{product} {engine.universe[symbol].label}" if symbol in getattr(engine, "universe", {}) else symbol
        if frames[a] is None or frames[b] is None:
            return html.Div("One of the contracts has too little history for an ATR yet.", style=muted)
        size_ratio = spec_for_product(product_of(a)).multiplier / spec_for_product(product_of(b)).multiplier
        return hedge_calculator_result(pair_hedge(frames[a], frames[b], names[a], names[b], window, size_ratio), lots, window)
    except Exception:  # noqa: BLE001
        logger.exception("Hedge calculator failed")
        return html.Div("Could not work out the hedge; see the log.", style=_ERROR)


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

    app.callback(
        Output("atr-hedge-a", "options"),
        Output("atr-hedge-b", "options"),
        Output("atr-hedge-a", "value"),
        Output("atr-hedge-b", "value"),
        Input("oi-version", "data"),
        Input("oi-tabs", "active_tab"),
        Input("atr-params", "data"),
        State("atr-hedge-a", "value"),
        State("atr-hedge-b", "value"),
    )(hedge_options)

    app.callback(
        Output("atr-hedge-result", "children"),
        Input("atr-hedge-a", "value"),
        Input("atr-hedge-b", "value"),
        Input("atr-hedge-lots", "value"),
        Input("atr-hedge-window", "value"),
        Input("atr-params", "data"),
    )(render_hedge)
