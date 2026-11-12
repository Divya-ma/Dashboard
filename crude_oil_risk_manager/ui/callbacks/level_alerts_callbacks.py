"""Price Alerts page callbacks: leg rows, create, list (live), delete. The rules live in core/level_alerts.py."""

import logging

from dash import ALL, Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from core.level_alerts import LevelAlertError, build_level_alert
from core.structure_builder import EVENT_ADD, EVENT_REMOVE, EVENT_TEMPLATE, next_leg_rows
from core.structure_utils import STRUCTURE_TEMPLATES
from core.structure_view import prices_from_store
from ui.container import container
from ui.layouts.level_alerts_tab import DEFAULT_TEMPLATE, alerts_table, leg_rows
from ui.layouts.shell import COLORS

logger = logging.getLogger(__name__)


def _message(text: str, ok: bool, warnings: list[str] | None = None) -> list:
    color = COLORS["ACCENT_GREEN"] if ok else COLORS["ACCENT_RED"]
    return [html.Div(("✓ " if ok else "⛔ ") + text, style={"color": color}),
            *[html.Div(f"⚠️ {w}", style={"color": COLORS["ACCENT_YELLOW"]}) for w in warnings or []]]


def update_leg_rows(template, add_clicks, remove_clicks, symbols, ratios):
    """The leg rows after a template pick, '+ Add leg' or a remove click; what was typed in the other legs is kept."""
    trigger = callback_context.triggered_id
    template = template if template in STRUCTURE_TEMPLATES else DEFAULT_TEMPLATE
    current = [{"symbol": s, "ratio": r} for s, r in zip(symbols or [], ratios or [])]
    if trigger is None or trigger == "la-template":
        event, index = EVENT_TEMPLATE, None
    elif trigger == "la-add-leg":
        if not add_clicks:
            raise PreventUpdate
        event, index = EVENT_ADD, None
    elif isinstance(trigger, dict) and trigger.get("type") == "la-remove":
        if not callback_context.triggered[0]["value"]:
            raise PreventUpdate  # the button was just created, not clicked
        event, index = EVENT_REMOVE, trigger["index"]
    else:
        raise PreventUpdate
    rows = next_leg_rows(template, event, current, index)
    return leg_rows(rows, removable=STRUCTURE_TEMPLATES[template]["legs"] is None)


def create_alert(n_clicks, template, symbols, ratios, level, trigger, name, live_prices, counter):
    if not n_clicks:
        raise PreventUpdate
    repository = container.repository
    try:
        alert, new_contracts, warnings = build_level_alert(
            name, template, symbols, ratios, level, trigger, prices_from_store(live_prices), repository.get_contract)
        for contract in new_contracts:  # saved so the live poll prices them from now on
            repository.save_contract(contract)
        repository.save_level_alert(alert)
    except LevelAlertError as exc:
        return _message(" ".join(exc.errors), False), no_update, no_update, no_update
    except Exception:  # noqa: BLE001 - a readable message instead of a silent callback error
        logger.exception("Could not create the price alert")
        return _message("Could not save the alert; see the server log.", False), no_update, no_update, no_update
    when = "rises to or above" if alert.direction == "above" else "falls to or below"
    done = f"Alert '{alert.name}' created: it fires once when the price {when} {alert.level:g}."
    return _message(done, True, warnings), (counter or 0) + 1, None, None  # the level and name are cleared for the next alert


def render_table(live_prices, _refresh):
    try:
        alerts = container.repository.get_level_alerts()
    except Exception:  # noqa: BLE001
        logger.exception("Could not read the price alerts")
        return html.Div("The price alerts could not be read; see the server log.", style={"color": COLORS["ACCENT_RED"]})
    return alerts_table(alerts, prices_from_store(live_prices))


def delete_alert(n_clicks_list, counter):
    trigger = callback_context.triggered_id
    if not isinstance(trigger, dict) or trigger.get("type") != "la-delete":
        raise PreventUpdate
    if not callback_context.triggered[0]["value"]:
        raise PreventUpdate  # the button was just created, not clicked
    container.repository.delete_level_alert(trigger["index"])
    return (counter or 0) + 1


def register_level_alerts_callbacks(app) -> None:
    app.callback(
        Output("la-leg-rows", "children"),
        Input("la-template", "value"),
        Input("la-add-leg", "n_clicks"),
        Input({"type": "la-remove", "index": ALL}, "n_clicks"),
        State({"type": "la-symbol", "index": ALL}, "value"),
        State({"type": "la-ratio", "index": ALL}, "value"),
    )(update_leg_rows)

    app.callback(
        Output("la-status", "children"),
        Output("la-refresh", "data"),
        Output("la-level", "value"),
        Output("la-name", "value"),
        Input("la-create", "n_clicks"),
        State("la-template", "value"),
        State({"type": "la-symbol", "index": ALL}, "value"),
        State({"type": "la-ratio", "index": ALL}, "value"),
        State("la-level", "value"),
        State("la-trigger", "value"),
        State("la-name", "value"),
        State("store-live-prices", "data"),
        State("la-refresh", "data"),
        prevent_initial_call=True,
    )(create_alert)

    app.callback(
        Output("la-table", "children"),
        Input("store-live-prices", "data"),
        Input("la-refresh", "data"),
    )(render_table)

    app.callback(
        Output("la-refresh", "data", allow_duplicate=True),
        Input({"type": "la-delete", "index": ALL}, "n_clicks"),
        State("la-refresh", "data"),
        prevent_initial_call=True,
    )(delete_alert)
