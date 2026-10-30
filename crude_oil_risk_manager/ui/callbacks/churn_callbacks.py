"""Churn page callbacks.

The status callback keeps the five-minute candles fresh (it starts the daily download in the background when due)
and writes `churn-version` ONLY when the stored data changed, so the table and charts are not rebuilt on every
tick. The ranking is simulated once per data version and parameter set (core.churn_service caches it), so
filtering and switching contracts afterwards is instant.
"""

import logging
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd
from dash import Input, Output, State, callback_context, html, no_update
from dash.exceptions import MissingCallbackContextException

from core.churn_costs import DEFAULT_COMBO_REBATES, DEFAULT_COSTS, validate_combo, validate_costs
from core.churn_service import validate_params
from ui.container import container
from ui.layouts.churn_tab import (
    COST_FIELDS, PARAM_INPUTS, breakeven_table, build_session_figure, build_worse_fill_figures, table_data,
)
from ui.layouts.curve_tab import empty_figure
from ui.layouts.shell import COLORS

logger = logging.getLogger(__name__)

_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_ERROR = {"color": COLORS["ACCENT_RED"]}
_FIELDS = [field for _, _, field, _, _ in PARAM_INPUTS]
_COST_IDS = [(product, field) for product in DEFAULT_COSTS for field, _, _ in COST_FIELDS]


def _triggered_id():
    try:
        return callback_context.triggered_id
    except MissingCallbackContextException:  # called directly, not by Dash
        return None


def status_text(status: dict) -> html.Div:
    parts = []
    if status["running"]:
        total = status["total"] or 1
        parts.append(f"⏳ {status['message']} ({status['done']}/{status['total']} requests, {status['done'] / total:.0%})")
    elif status["updated_on"]:
        parts.append(f"Candles updated {status['updated_on']}")
        if status["message"]:
            parts.append(status["message"])
    else:
        parts.append("No candles downloaded yet. They download automatically, or press the button.")
    children = [html.Span(" · ".join(parts))]
    if status["error"]:
        children.append(html.Div(f"⚠️ {status['error']}", style=_ERROR))
    return html.Div(children)


def update_status(n_intervals, n_clicks, current_version=None):
    """Status line; starts the download when it is due or the button is pressed."""
    service = container.churn_service
    if service is None:
        return html.Div("The churn service is not running.", style=_MUTED), no_update
    if _triggered_id() == "churn-update-btn":
        service.refresh_now()
    else:
        service.ensure_fresh()
    status = service.status()
    version = status["version"] if status["has_data"] else "none"
    if status["running"] or version == current_version:
        version = no_update
    return status_text(status), version


def render_churn(version, product, family, direction, _costs_version, *param_values):
    """(note, contract options, contract value, params, session chart). Builds the simulation if it is not built yet."""
    service = container.churn_service
    if service is None or version is None:
        return html.Div(), [], None, no_update, empty_figure("No data yet")
    try:
        params = validate_params(dict(zip(_FIELDS, param_values)))
    except ValueError as exc:
        return html.Div(f"⚠️ {exc}", style=_ERROR), no_update, no_update, no_update, no_update
    try:
        service.save_params(params)
        results = service.results(params)
        rows = service.scored(params)
    except Exception:  # noqa: BLE001
        logger.exception("Churn ranking failed")
        return html.Div("Could not build the ranking; see the log.", style=_ERROR), no_update, no_update, no_update, no_update
    if rows is None or rows.empty:
        return (html.Div("No instrument has enough five-minute candles yet. The download starts when this page opens; the first one takes a few minutes.", style=_MUTED),
                [], None, asdict(params), empty_figure("No scored rows"))
    shown = rows
    for column, value in (("product", product), ("family", family), ("direction", direction)):
        if value and value != "all":
            shown = shown[shown[column] == value]
    options = [{"label": f"#{n}  {r.product} {r.label} {r.direction} (score {r.score:.0f})", "value": f"{r.symbol}|{r.direction}"} for n, r in enumerate(shown.itertuples(), 1)]
    note = html.Div(
        f"Five-minute candles to {results.as_of}, a {params.window_days}-day window; contracts from #{params.start_from} onwards. Trades enter by crossing the spread, aim for a target above the fill, stop out at {params.stop_mult:g} targets and are closed at {params.window_end:g}:00 London "
        f"(trading hours {params.window_start:g}:00 to {params.window_end:g}:00). Scores use the last 40% of the days; the target was picked on the first 60%. Figures are per lot after transaction cost. "
        "Spread width is an assumption you set above until it is measured from quotes; d-flies are priced from their legs with no spread. "
        "The oscillation count and the live price refresh every 10 seconds. Hover over a heading for what it means.")
    return note, options, (options[0]["value"] if options else None), asdict(params), build_session_figure(service.sessions(rows, results.details))


def render_detail(selected, params_dict, version):
    """Net $/day and hit probability against the target for one contract, for each size of worse fill."""
    service = container.churn_service
    blank = (empty_figure("Pick a contract"), empty_figure("Pick a contract"))
    if service is None or not selected or not params_dict:
        return blank
    try:
        symbol, direction = selected.split("|")
        params = validate_params(params_dict)
        grid = service.detail_grid(symbol, direction, params)
        entry = service.results(params).details.get((symbol, direction))
        if grid is None or entry is None:
            return blank
        label = f"{entry['spec'].product} {entry['spec'].label} {direction}"
        return build_worse_fill_figures(grid, label, entry["best"])
    except Exception:  # noqa: BLE001
        logger.exception("Churn detail failed")
        return blank


def render_table(_n, params_dict, _version, product, family, direction):
    """The table's rows: the ranking with the live price and oscillations. Reads the built simulation and the latest curve poll; never builds the simulation or calls the API."""
    service = container.churn_service
    if service is None or not params_dict:
        return no_update, no_update
    try:
        table = service.live_table(validate_params(params_dict), product, family, direction)
    except Exception:  # noqa: BLE001
        logger.exception("Churn table failed")
        return no_update, no_update
    if table is None:
        return no_update, no_update
    if table.empty:
        return [], []
    return table_data(table)


def save_costs_callback(n_clicks, *values):
    """Validate and save the TC / rebate table; bumps `churn-costs-version` so the ranking is rebuilt."""
    if not n_clicks:
        return no_update, no_update
    service = container.churn_service
    if service is None:
        return html.Span("The churn service is not running.", style=_ERROR), no_update
    raw: dict = {}
    for (product, field), value in zip(_COST_IDS, values[: len(_COST_IDS)]):
        raw.setdefault(product, {})[field] = value
    combo_raw = dict(zip(DEFAULT_COMBO_REBATES, values[len(_COST_IDS):-1]))
    try:
        costs, combo = validate_costs(raw), validate_combo(combo_raw)
    except ValueError as exc:
        return html.Span(f"⚠️ {exc}", style=_ERROR), no_update
    service.save_costs(costs, combo, bool(values[-1]))
    return html.Span("Saved.", style={"color": COLORS["ACCENT_GREEN"]}), n_clicks


def render_breakeven(_costs_version):
    service = container.churn_service
    if service is None:
        return html.Div()
    return breakeven_table(service.effective_costs(), service.combo_rebates())


def register_churn_callbacks(app) -> None:
    app.callback(
        Output("churn-status", "children"),
        Output("churn-version", "data"),
        Input("churn-refresh", "n_intervals"),
        Input("churn-update-btn", "n_clicks"),
        State("churn-version", "data"),
    )(update_status)

    app.callback(
        Output("churn-note", "children"),
        Output("churn-contract", "options"),
        Output("churn-contract", "value"),
        Output("churn-params", "data"),
        Output("churn-session-chart", "figure"),
        Input("churn-version", "data"),
        Input("churn-product", "value"),
        Input("churn-family", "value"),
        Input("churn-direction", "value"),
        Input("churn-costs-version", "data"),
        *(Input(component_id, "value") for component_id, *_ in PARAM_INPUTS),
    )(render_churn)

    app.callback(
        Output("churn-table", "data"),
        Output("churn-table", "tooltip_data"),
        Input("churn-live-refresh", "n_intervals"),
        Input("churn-params", "data"),
        Input("churn-version", "data"),
        Input("churn-product", "value"),
        Input("churn-family", "value"),
        Input("churn-direction", "value"),
    )(render_table)

    app.callback(
        Output("churn-net-chart", "figure"),
        Output("churn-hit-chart", "figure"),
        Input("churn-contract", "value"),
        Input("churn-params", "data"),
        Input("churn-version", "data"),
    )(render_detail)

    app.callback(
        Output("churn-costs-msg", "children"),
        Output("churn-costs-version", "data"),
        Input("churn-costs-save", "n_clicks"),
        *(State(f"churn-c-{product}-{field}", "value") for product, field in _COST_IDS),
        *(State(f"churn-combo-{product}", "value") for product in DEFAULT_COMBO_REBATES),
        State("churn-use-rebates", "value"),
    )(save_costs_callback)

    app.callback(
        Output("churn-breakeven", "children"),
        Input("churn-costs-version", "data"),
    )(render_breakeven)
