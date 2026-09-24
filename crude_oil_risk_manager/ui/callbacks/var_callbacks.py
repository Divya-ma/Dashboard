"""VaR & Scenarios callbacks.

Two independent callbacks: the VaR section (tab open, lookback change) and the
scenario table (tab open, Refresh Scenarios). The maths lives in core.var and
core.scenarios; these functions load the open structures, call it and render.
"""

from dash import Input, Output
from dash.exceptions import PreventUpdate

from core.scenarios import build_scenarios, load_previous_day_atr
from core.structure_view import statuses_for_filter
from core.user_settings import KEY_CORRELATION_WINDOW
from core.var import (
    DISTRIBUTION_NORMAL,
    MonteCarloVarResult,
    leg_dollar_weights,
    monte_carlo_var,
    portfolio_pnl_var,
)
from ui.container import container
from ui.layouts.var_tab import (
    METHOD_MONTE_CARLO,
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


def toggle_mc_controls(method):
    """Show the Monte Carlo distribution/simulation controls only when that method is selected."""
    is_mc = method == METHOD_MONTE_CARLO
    return {"display": "block" if is_mc else "none", "marginBottom": "12px"}


def toggle_distribution_params(distribution):
    """Show Mean/Std for Normal, Low/High for Uniform."""
    is_normal = distribution == DISTRIBUTION_NORMAL
    return (
        {"display": "inline-block" if is_normal else "none"},
        {"display": "none" if is_normal else "inline-block"},
    )


def update_var_section(pathname, lookback, method, distribution, mean, std, low, high, n_simulations):
    """Cards, warnings and histogram: on tab open and whenever any VaR control changes."""
    if pathname != _PATH:
        raise PreventUpdate
    lookback_days = int(lookback)
    structures = _open_structures()

    if method == METHOD_MONTE_CARLO:
        weights = leg_dollar_weights(structures)
        n_sims = int(n_simulations or 10_000)
        dist_params = (
            {"mean": float(mean or 0.0), "std": float(std or 0.0)}
            if distribution == DISTRIBUTION_NORMAL
            else {"low": float(low or 0.0), "high": float(high or 0.0)}
        )
        window = int(container.repository.get_setting(KEY_CORRELATION_WINDOW, lookback_days))
        try:
            result = monte_carlo_var(
                weights, container.data_loader, distribution, dist_params,
                correlation_window=window, n_simulations=n_sims,
            )
        except ValueError as exc:
            result = MonteCarloVarResult(error=str(exc))
        period_label = f"{n_sims:,} sims"
        title = f"Simulated Portfolio PnL Distribution (Monte Carlo, {distribution.title()})"
    else:
        result = portfolio_pnl_var(structures, lookback_days, container.data_loader)
        period_label = f"{lookback_days}d"
        title = f"Portfolio PnL Distribution ({lookback_days}d Historical)"

    card_95, card_99, card_lookback, card_observations = build_var_cards(result, period_label)
    return (
        card_95, card_99, card_lookback, card_observations,
        build_histogram(result, title),
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
        Input("var-method", "value"),
        Input("var-mc-distribution", "value"),
        Input("var-mc-mean", "value"),
        Input("var-mc-std", "value"),
        Input("var-mc-low", "value"),
        Input("var-mc-high", "value"),
        Input("var-mc-simulations", "value"),
    )(update_var_section)

    app.callback(
        Output("var-mc-controls", "style"),
        Input("var-method", "value"),
    )(toggle_mc_controls)

    app.callback(
        Output("var-mc-normal-params", "style"),
        Output("var-mc-uniform-params", "style"),
        Input("var-mc-distribution", "value"),
    )(toggle_distribution_params)

    app.callback(
        Output("scenario-table", "data"),
        Output("scenario-message", "children"),
        Output("scenario-note", "children"),
        Input("url", "pathname"),
        Input("scenario-refresh-btn", "n_clicks"),
    )(update_scenarios)
