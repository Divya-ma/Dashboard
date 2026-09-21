"""VaR & Scenarios callbacks.

Two independent callbacks: the VaR section (tab open, lookback change) and the
scenario table (tab open, Refresh Scenarios). The maths lives in core.var and
core.scenarios; these functions load the open structures, call it and render.
"""

from dash import Input, Output
from dash.exceptions import PreventUpdate

from core.scenarios import build_scenarios, load_previous_day_atr
from core.structure_view import statuses_for_filter
from core.var import leg_dollar_weights, portfolio_pnl_var
from ui.container import container
from ui.layouts.var_tab import (
    NOTHING_TO_ANALYZE,
    build_histogram,
    build_scenario_note,
    build_scenario_table_rows,
    build_var_cards,
    build_var_warnings,
)

_PATH = "/var-scenario"


def _open_structures() -> list:
    return container.repository.get_all_structures(status_filter=statuses_for_filter("open"))


def update_var_section(pathname, lookback):
    """Cards, warnings and histogram: on tab open and whenever the lookback changes."""
    if pathname != _PATH:
        raise PreventUpdate
    lookback_days = int(lookback)
    result = portfolio_pnl_var(_open_structures(), lookback_days, container.data_loader)
    card_95, card_99, card_lookback, card_observations = build_var_cards(result, lookback_days)
    return (
        card_95, card_99, card_lookback, card_observations,
        build_histogram(result, lookback_days),
        build_var_warnings(result),
    )


def update_scenarios(pathname, n_clicks):
    """Scenario table: on tab open and on Refresh Scenarios (which re-reads the previous day's TR)."""
    if pathname != _PATH:
        raise PreventUpdate
    weights = leg_dollar_weights(_open_structures())
    if not weights:
        return [], NOTHING_TO_ANALYZE, ""
    atr = load_previous_day_atr(sorted(weights), container.data_loader)
    rows = build_scenarios(weights, atr)
    return build_scenario_table_rows(rows), "", build_scenario_note(rows)


def register_var_callbacks(app) -> None:
    """Attach the VaR & Scenarios callbacks to the Dash app."""
    app.callback(
        Output("var-card-95", "children"),
        Output("var-card-99", "children"),
        Output("var-card-lookback", "children"),
        Output("var-card-observations", "children"),
        Output("var-histogram", "figure"),
        Output("var-warnings", "children"),
        Input("url", "pathname"),
        Input("var-lookback", "value"),
    )(update_var_section)

    app.callback(
        Output("scenario-table", "data"),
        Output("scenario-message", "children"),
        Output("scenario-note", "children"),
        Input("url", "pathname"),
        Input("scenario-refresh-btn", "n_clicks"),
    )(update_scenarios)
