"""Price Alerts page callbacks: leg rows, create, list (live), delete. The rules live in core/level_alerts.py."""

import logging

from dash import ALL, Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from core.level_alerts import LevelAlertError, build_level_alert
from core.structure_builder import EVENT_ADD, EVENT_REMOVE, EVENT_TEMPLATE, next_leg_rows
from core.structure_utils import STRUCTURE_TEMPLATES
from core.structure_view import prices_from_store
from ui.container import container
from ui.layouts.level_alerts_tab import CREATE_LABEL, DEFAULT_TEMPLATE, EDIT_TITLE, NEW_TITLE, SAVE_LABEL, alerts_table, leg_rows
from ui.layouts.shell import COLORS

logger = logging.getLogger(__name__)


def _message(text: str, ok: bool, warnings: list[str] | None = None) -> list:
    color = COLORS["ACCENT_GREEN"] if ok else COLORS["ACCENT_RED"]
    return [html.Div(("✓ " if ok else "⛔ ") + text, style={"color": color}),
            *[html.Div(f"⚠️ {w}", style={"color": COLORS["ACCENT_YELLOW"]}) for w in warnings or []]]


def template_for(legs: list[dict]) -> str:
    """The template whose ratios are exactly these legs' (a spread, a fly...), else Custom."""
    ratios = [leg["ratio"] for leg in legs]
    return next((key for key, spec in STRUCTURE_TEMPLATES.items() if key != "custom" and spec["ratios"] == ratios), "custom")


def update_leg_rows(template, add_clicks, remove_clicks, symbols, ratios, editing=None):
    """The leg rows after a template pick, '+ Add leg', a remove click or 'Edit' on an alert; what was typed in the other legs is kept."""
    template = template if template in STRUCTURE_TEMPLATES else DEFAULT_TEMPLATE
    if editing and any(str(t.get("prop_id", "")).startswith("la-edit.") for t in callback_context.triggered):
        return leg_rows([{"symbol": leg["symbol"], "ratio": leg["ratio"]} for leg in editing["legs"]], removable=STRUCTURE_TEMPLATES[template]["legs"] is None)
    trigger = callback_context.triggered_id
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


def create_alert(n_clicks, template, symbols, ratios, level, trigger, name, live_prices, counter, editing=None):
    if not n_clicks or editing:  # while an alert is being edited the same button saves it (save_edit)
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


_NEW_FORM = (NEW_TITLE, CREATE_LABEL, {"display": "none"})
_EDIT_FORM = (EDIT_TITLE, SAVE_LABEL, {"display": "inline-block"})


def start_edit(n_clicks_list):
    """'Edit' on a row: the alert goes into the form. (edit state, template, level, direction, name, title, button label, cancel style, message)"""
    trigger = callback_context.triggered_id
    if not isinstance(trigger, dict) or trigger.get("type") != "la-edit":
        raise PreventUpdate
    if not callback_context.triggered[0]["value"]:
        raise PreventUpdate  # the button was just created, not clicked
    alert = next((a for a in container.repository.get_level_alerts() if a.alert_id == trigger["index"]), None)
    if alert is None:
        raise PreventUpdate
    template = template_for(alert.legs)
    note = ("Saving arms it again: it fires the next time the price reaches the level." if alert.status == "triggered"
            else "Change the legs, level or name, then save.")
    return ({"id": alert.alert_id, "template": template, "legs": alert.legs}, template, alert.level, alert.direction, alert.name, *_EDIT_FORM,
            _message(f"Editing '{alert.name}'. {note}", True))


def cancel_edit(n_clicks):
    """Leave edit mode: the form is a new-alert form again (the legs typed stay). (edit state, level, name, title, label, cancel style, message)"""
    if not n_clicks:
        raise PreventUpdate
    return None, None, None, *_NEW_FORM, ""


def save_edit(n_clicks, template, symbols, ratios, level, trigger, name, live_prices, counter, editing):
    """The Save button while an alert is being edited: the same checks as a new alert, the same id.
    (status, refresh, level, name, edit state, title, label, cancel style)"""
    if not n_clicks or not editing:
        raise PreventUpdate
    repository = container.repository
    try:
        alert, new_contracts, warnings = build_level_alert(
            name, template, symbols, ratios, level, trigger, prices_from_store(live_prices), repository.get_contract)
        alert.alert_id = editing["id"]
        for contract in new_contracts:
            repository.save_contract(contract)
        if not repository.update_level_alert(alert):
            return (_message("That alert was deleted meanwhile, so nothing was saved.", False), (counter or 0) + 1, None, None, None, *_NEW_FORM)
    except LevelAlertError as exc:
        return (_message(" ".join(exc.errors), False), *[no_update] * 7)
    except Exception:  # noqa: BLE001
        logger.exception("Could not save the price alert")
        return (_message("Could not save the changes; see the server log.", False), *[no_update] * 7)
    when = "rises to or above" if alert.direction == "above" else "falls to or below"
    return (_message(f"Alert '{alert.name}' saved and armed: it fires once when the price {when} {alert.level:g}.", True, warnings),
            (counter or 0) + 1, None, None, None, *_NEW_FORM)


def _button_click(button_type: str):
    """The alert id if `button_type` was just clicked (not merely drawn); else raise PreventUpdate."""
    trigger = callback_context.triggered_id
    if not isinstance(trigger, dict) or trigger.get("type") != button_type:
        raise PreventUpdate
    if not callback_context.triggered[0]["value"]:
        raise PreventUpdate  # the button was just created, not clicked
    return trigger["index"]


def pause_alert(n_clicks_list, counter):
    """'Stop' on a row: the alert is skipped by the price check and dropped from the live feed until started again."""
    container.repository.pause_level_alert(_button_click("la-pause"))
    return (counter or 0) + 1


def resume_alert(n_clicks_list, counter):
    """'Start' on a row: the alert is live again."""
    container.repository.resume_level_alert(_button_click("la-resume"))
    return (counter or 0) + 1


def delete_alert(n_clicks_list, counter):
    container.repository.delete_level_alert(_button_click("la-delete"))
    return (counter or 0) + 1


def register_level_alerts_callbacks(app) -> None:
    app.callback(
        Output("la-leg-rows", "children"),
        Input("la-template", "value"),
        Input("la-add-leg", "n_clicks"),
        Input({"type": "la-remove", "index": ALL}, "n_clicks"),
        State({"type": "la-symbol", "index": ALL}, "value"),
        State({"type": "la-ratio", "index": ALL}, "value"),
        Input("la-edit", "data"),
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
        State("la-edit", "data"),
        prevent_initial_call=True,
    )(create_alert)

    app.callback(
        Output("la-edit", "data"),
        Output("la-template", "value"),
        Output("la-level", "value", allow_duplicate=True),
        Output("la-trigger", "value"),
        Output("la-name", "value", allow_duplicate=True),
        Output("la-form-title", "children"),
        Output("la-create", "children"),
        Output("la-cancel", "style"),
        Output("la-status", "children", allow_duplicate=True),
        Input({"type": "la-edit", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )(start_edit)

    app.callback(
        Output("la-edit", "data", allow_duplicate=True),
        Output("la-level", "value", allow_duplicate=True),
        Output("la-name", "value", allow_duplicate=True),
        Output("la-form-title", "children", allow_duplicate=True),
        Output("la-create", "children", allow_duplicate=True),
        Output("la-cancel", "style", allow_duplicate=True),
        Output("la-status", "children", allow_duplicate=True),
        Input("la-cancel", "n_clicks"),
        prevent_initial_call=True,
    )(cancel_edit)

    app.callback(
        Output("la-status", "children", allow_duplicate=True),
        Output("la-refresh", "data", allow_duplicate=True),
        Output("la-level", "value", allow_duplicate=True),
        Output("la-name", "value", allow_duplicate=True),
        Output("la-edit", "data", allow_duplicate=True),
        Output("la-form-title", "children", allow_duplicate=True),
        Output("la-create", "children", allow_duplicate=True),
        Output("la-cancel", "style", allow_duplicate=True),
        Input("la-create", "n_clicks"),
        State("la-template", "value"),
        State({"type": "la-symbol", "index": ALL}, "value"),
        State({"type": "la-ratio", "index": ALL}, "value"),
        State("la-level", "value"),
        State("la-trigger", "value"),
        State("la-name", "value"),
        State("store-live-prices", "data"),
        State("la-refresh", "data"),
        State("la-edit", "data"),
        prevent_initial_call=True,
    )(save_edit)

    app.callback(
        Output("la-table", "children"),
        Input("store-live-prices", "data"),
        Input("la-refresh", "data"),
    )(render_table)

    app.callback(
        Output("la-refresh", "data", allow_duplicate=True),
        Input({"type": "la-pause", "index": ALL}, "n_clicks"),
        State("la-refresh", "data"),
        prevent_initial_call=True,
    )(pause_alert)

    app.callback(
        Output("la-refresh", "data", allow_duplicate=True),
        Input({"type": "la-resume", "index": ALL}, "n_clicks"),
        State("la-refresh", "data"),
        prevent_initial_call=True,
    )(resume_alert)

    app.callback(
        Output("la-refresh", "data", allow_duplicate=True),
        Input({"type": "la-delete", "index": ALL}, "n_clicks"),
        State("la-refresh", "data"),
        prevent_initial_call=True,
    )(delete_alert)
