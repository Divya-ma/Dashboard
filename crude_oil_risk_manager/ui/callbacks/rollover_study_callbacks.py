"""Trigger study callbacks (Strategy Rollover page). They read core.rollover_triggers.RolloverStudyService.

The study uses the page's own selection (product, curve, generic, expiries, days before), so it follows the controls; nothing is stored
about a clicked row. A saved rule keeps the selection and the window, not a position on any ladder.
"""

import logging

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

from core.rollover import DEFAULT_COMPARISONS
from core.rollover_triggers import MAX_LINES, MAX_LOOKBACK, MAX_X, MIN_LOOKBACK, MIN_X
from core.trigger_stats import NO_EVIDENCE, POSSIBLE, STRONG
from ui.callbacks.rollover_callbacks import _whole, parse_options
from ui.container import container
from ui.layouts.curve_tab import empty_figure
from ui.layouts.rollover_study import (
    LINE_COLUMNS, PLACEHOLDER, board_rows, forward_view, line_columns, line_rows, notes_block, roll_block, rule_block, scan_view,
    scatter_figure, title_text, values_view, verdict_block,
)
from ui.layouts.rollover_tab import ROLLOVER_PRODUCTS

logger = logging.getLogger(__name__)

PAGE_CONTROLS = [
    Input("rollover-product", "value"), Input("rollover-family", "value"), Input("rollover-generic", "value"),
    Input("rollover-expiries", "value"), Input("rollover-before", "value"),
]
STUDY_CONTROLS = [
    Input("rollover-study-from", "value"), Input("rollover-study-to", "value"), Input("rollover-study-lookback", "value"),
    Input("rollover-study-lines", "value"),
]


def _selection(product, family, generic, expiries, before):
    if product not in ROLLOVER_PRODUCTS:
        raise ValueError(PLACEHOLDER)
    return product, parse_options(family, generic, expiries, DEFAULT_COMPARISONS, before, False)


def _window(x_from, x_to, lookback, lines) -> tuple[int, int, int, int]:
    return (
        _whole(x_from, MIN_X, MAX_X, "From"), _whole(x_to, MIN_X, MAX_X, "To"),
        _whole(lookback, MIN_LOOKBACK, MAX_LOOKBACK, "Lookback"), _whole(lines, 1, MAX_LINES, "Expiries studied"),
    )


def _study(product, family, generic, expiries, before, x_from, x_to, lookback, lines):
    service = container.rollover_study
    if service is None:
        raise ValueError("The trigger study is not running")
    product, options = _selection(product, family, generic, expiries, before)
    return service.study(product, options, *_window(x_from, x_to, lookback, lines))


def _blank(message: str):
    return message, html.Div(), LINE_COLUMNS, [], [], [], None, [], [], html.Div(), html.Div(), notes_block(None)


def _best_trigger(study) -> str | None:
    for verdicts in ((STRONG, POSSIBLE), (NO_EVIDENCE,)):
        found = next((s for s in study.screens if s.result.verdict in verdicts), None)
        if found is not None:
            return found.spec.key
    return study.screens[0].spec.key if study.screens else None


def render_study(product, family, generic, expiries, before, x_from, x_to, lookback, lines, chosen):
    """Title, verdict, expiry table, trigger scoreboard, trigger picker, readings table, rule, switch days and notes."""
    try:
        study = _study(product, family, generic, expiries, before, x_from, x_to, lookback, lines)
    except ValueError as exc:
        return _blank(str(exc))
    except Exception:  # noqa: BLE001 - a bad study must not blank the page
        logger.exception("Rollover trigger study failed")
        return _blank("Could not build the study; see the log")
    keys = [s.spec.key for s in study.screens]
    options = [{"label": s.spec.label, "value": s.spec.key} for s in study.screens]
    value_columns, value_rows = values_view(study)
    return (
        title_text(study), verdict_block(study), line_columns(study), line_rows(study), board_rows(study), options,
        chosen if chosen in keys else _best_trigger(study), value_columns, value_rows, rule_block(study), roll_block(study), notes_block(study),
    )


def render_scatter(trigger, product, family, generic, expiries, before, x_from, x_to, lookback, lines):
    try:
        study = _study(product, family, generic, expiries, before, x_from, x_to, lookback, lines)
    except ValueError as exc:
        return empty_figure(str(exc))
    except Exception:  # noqa: BLE001
        logger.exception("Rollover trigger scatter failed")
        return empty_figure("Could not build this chart; see the log")
    return scatter_figure(study, trigger)


def save_rule(n_clicks, product, family, generic, expiries, before, x_from, x_to, lookback, lines):
    try:
        study = _study(product, family, generic, expiries, before, x_from, x_to, lookback, lines)
        rule = container.rollover_study.save_rule(study)
    except ValueError as exc:
        return str(exc)
    except Exception:  # noqa: BLE001
        logger.exception("Saving the rule failed")
        return "Could not save the rule; see the log"
    return f"Saved: tracking {rule['label']} from now on ({rule['trigger_label']})"


def render_forward(n_intervals, saved, deleted):
    service = container.rollover_study
    if service is None:
        return [], []
    try:
        return forward_view(service.forward_test(), service.rules())
    except Exception:  # noqa: BLE001
        logger.exception("Rollover forward test failed")
        return [], []


def delete_rule(n_clicks, rule_id):
    service = container.rollover_study
    if service is None or not rule_id:
        return None, "Choose a rule first"
    return None, "Deleted" if service.delete_rule(rule_id) else "That rule is already gone"


def run_scan(n_clicks, product, family, generic, expiries, before, lookback, lines):
    service = container.rollover_study
    if service is None:
        return [], "The trigger study is not running"
    try:
        product, options = _selection(product, family, generic, expiries, before)
        result = service.scan(product, options, _whole(lookback, MIN_LOOKBACK, MAX_LOOKBACK, "Lookback"), _whole(lines, 1, MAX_LINES, "Expiries studied"))
    except ValueError as exc:
        return [], str(exc)
    except Exception:  # noqa: BLE001
        logger.exception("Rollover window scan failed")
        return [], "Could not scan; see the log"
    return scan_view(result)


def apply_scan_row(active_cell, rows):
    """A clicked scan row sets the window."""
    row_id = (active_cell or {}).get("row_id")
    match = next((r for r in rows or [] if r.get("id") == row_id), None)
    if match is None:
        raise PreventUpdate
    return match["x_from"], match["x_to"]


def register_rollover_study_callbacks(app) -> None:
    states = [State(i.component_id, i.component_property) for i in [*PAGE_CONTROLS, *STUDY_CONTROLS]]
    app.callback(
        Output("rollover-study-title", "children"), Output("rollover-study-verdict", "children"),
        Output("rollover-study-lines-table", "columns"), Output("rollover-study-lines-table", "data"), Output("rollover-study-board", "data"),
        Output("rollover-study-trigger", "options"), Output("rollover-study-trigger", "value"),
        Output("rollover-study-values", "columns"), Output("rollover-study-values", "data"),
        Output("rollover-study-rule", "children"), Output("rollover-study-rolls", "children"), Output("rollover-study-notes", "children"),
        *PAGE_CONTROLS, *STUDY_CONTROLS, State("rollover-study-trigger", "value"),
    )(render_study)
    app.callback(Output("rollover-study-scatter", "figure"), Input("rollover-study-trigger", "value"), *PAGE_CONTROLS, *STUDY_CONTROLS)(render_scatter)
    app.callback(
        Output("rollover-study-rule-feedback", "children"), Input("rollover-study-save-rule", "n_clicks"), *states, prevent_initial_call=True,
    )(save_rule)
    app.callback(
        Output("rollover-forward-table", "data"), Output("rollover-forward-pick", "options"),
        Input("rollover-refresh", "n_intervals"), Input("rollover-study-rule-feedback", "children"), Input("rollover-forward-feedback", "children"),
    )(render_forward)
    app.callback(
        Output("rollover-forward-pick", "value"), Output("rollover-forward-feedback", "children"),
        Input("rollover-forward-delete", "n_clicks"), State("rollover-forward-pick", "value"), prevent_initial_call=True,
    )(delete_rule)
    app.callback(
        Output("rollover-scan-table", "data"), Output("rollover-scan-note", "children"), Input("rollover-scan-run", "n_clicks"),
        State("rollover-product", "value"), State("rollover-family", "value"), State("rollover-generic", "value"),
        State("rollover-expiries", "value"), State("rollover-before", "value"), State("rollover-study-lookback", "value"),
        State("rollover-study-lines", "value"), prevent_initial_call=True,
    )(run_scan)
    app.callback(
        Output("rollover-study-from", "value"), Output("rollover-study-to", "value"), Input("rollover-scan-table", "active_cell"),
        State("rollover-scan-table", "data"), prevent_initial_call=True,
    )(apply_scan_row)
