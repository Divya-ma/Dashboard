"""Trade Idea Analyzer callbacks: leg toggle, open-structure loading and Analyze.

Validation and maths live in core.trade_idea; these functions gather the form
values, call it and render the result.
"""

from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from core.structure_view import statuses_for_filter
from core.trade_idea import DEFAULT_LEGS, STRUCTURE_LEG_COUNTS, analyze_trade_idea, parse_idea_inputs
from ui.container import container
from ui.layouts.idea_tab import MAX_LEGS, build_messages, build_output

_PATH = "/trade-analyzer"
_SHOWN: dict = {}
_HIDDEN = {"display": "none"}


def toggle_leg_inputs(structure_type):
    """Show 1, 2 or 3 leg rows and reset their direction/lots to the usual pattern for the type."""
    count = STRUCTURE_LEG_COUNTS.get(structure_type, 1)
    defaults = DEFAULT_LEGS.get(structure_type, DEFAULT_LEGS["outright"])
    styles = [_SHOWN if number <= count else _HIDDEN for number in range(1, MAX_LEGS + 1)]
    directions = [defaults[i][0] if i < count else "buy" for i in range(MAX_LEGS)]
    lots = [defaults[i][1] if i < count else 1 for i in range(MAX_LEGS)]
    return (*styles, *directions, *lots)


def load_open_structures(pathname):
    """On tab open: remember which structures are open, for the correlation section."""
    if pathname != _PATH:
        raise PreventUpdate
    structures = container.repository.get_all_structures(status_filter=statuses_for_filter("open"))
    return [{"structure_id": s.structure_id, "name": s.name} for s in structures]


def analyze(
    n_clicks, structure_type, symbol_1, direction_1, lots_1, symbol_2, direction_2, lots_2,
    symbol_3, direction_3, lots_3, entry, stop, target, lookback, open_structures,
):
    """Validate the idea, analyze it against local history and render sections A-D."""
    if not n_clicks:
        raise PreventUpdate
    idea, errors, warnings = parse_idea_inputs(
        structure_type,
        [symbol_1, symbol_2, symbol_3],
        [direction_1, direction_2, direction_3],
        [lots_1, lots_2, lots_3],
        entry, stop, target,
    )
    if idea is None:
        return build_messages(errors, warnings)

    repository = container.repository
    structures = [repository.get_structure(item["structure_id"]) for item in open_structures or []]
    structures = [s for s in structures if s is not None]
    lookback_days = int(lookback)
    analysis = analyze_trade_idea(idea, lookback_days, structures, container.data_loader)
    return build_output(analysis, warnings, lookback_days, has_open_structures=bool(structures))


def register_idea_callbacks(app) -> None:
    """Attach the Trade Idea Analyzer callbacks to the Dash app."""
    app.callback(
        *(Output(f"idea-leg-row-{n}", "style") for n in range(1, MAX_LEGS + 1)),
        *(Output(f"idea-leg-{n}-direction", "value") for n in range(1, MAX_LEGS + 1)),
        *(Output(f"idea-leg-{n}-lots", "value") for n in range(1, MAX_LEGS + 1)),
        Input("idea-structure-type", "value"),
        prevent_initial_call=True,
    )(toggle_leg_inputs)

    app.callback(
        Output("idea-open-structures", "data"),
        Input("url", "pathname"),
    )(load_open_structures)

    app.callback(
        Output("idea-output", "children"),
        Input("idea-analyze-btn", "n_clicks"),
        State("idea-structure-type", "value"),
        State("idea-leg-1-symbol", "value"),
        State("idea-leg-1-direction", "value"),
        State("idea-leg-1-lots", "value"),
        State("idea-leg-2-symbol", "value"),
        State("idea-leg-2-direction", "value"),
        State("idea-leg-2-lots", "value"),
        State("idea-leg-3-symbol", "value"),
        State("idea-leg-3-direction", "value"),
        State("idea-leg-3-lots", "value"),
        State("idea-entry", "value"),
        State("idea-stop", "value"),
        State("idea-target", "value"),
        State("idea-lookback", "value"),
        State("idea-open-structures", "data"),
        prevent_initial_call=True,
    )(analyze)
