"""Structures tab callbacks: active/closed grids, closed-section toggle and row actions.

Row building lives in core.structure_view; PnL comes from `store-portfolio-pnl`
(written by shell_callbacks.refresh_portfolio_pnl) and is never recalculated here.
"""

import logging

from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate

from core.models import StructureStatus
from core.structure_view import (
    build_active_rows,
    build_closed_rows,
    prices_from_store,
    statuses_for_filter,
)
from ui.container import container
from ui.layouts.shell import COLORS

logger = logging.getLogger(__name__)

ACTION_VIEW = "view"
ACTION_ENTER_TRADE = "enter_trade"
ACTION_EXIT = "exit"
_ACTIONS = {ACTION_VIEW, ACTION_ENTER_TRADE, ACTION_EXIT}


def update_active_structures(
    portfolio_pnl, status_filter, product_filter, sort_by, live_prices
):
    """Rows for the active grid: filtered, sorted, PnL taken from the portfolio PnL store."""
    structures = container.repository.get_all_structures(status_filter=statuses_for_filter(status_filter))
    per_structure = (portfolio_pnl or {}).get("per_structure", {})
    return build_active_rows(structures, per_structure, prices_from_store(live_prices), product_filter, sort_by)


def update_closed_structures(pathname, n_intervals, is_open):
    """Rows for the closed grid and the toggle button label with the closed count."""
    repository = container.repository
    structures = repository.get_all_structures(status_filter=[StructureStatus.CLOSED])
    trades = {s.structure_id: repository.get_trades_for_structure(s.structure_id) for s in structures}
    verb = "Hide" if is_open else "Show"
    return build_closed_rows(structures, trades), f"📁 {verb} Closed Structures ({len(structures)})"


def toggle_closed(n_clicks, is_open):
    return not is_open


def handle_structure_action(cell_data):
    """Row button clicks. 'view' opens the detail modal; trade entry / exit modals arrive in later phases."""
    payload = (cell_data or {}).get("value") or {}
    action = payload.get("action")
    structure_id = payload.get("structure_id") or (cell_data or {}).get("rowId")
    if action not in _ACTIONS or not structure_id:
        raise PreventUpdate

    if action == ACTION_VIEW:
        return True, structure_id
    logger.info("Structure action %r requested for %s (no modal built yet)", action, structure_id)
    return no_update, structure_id


def close_detail_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


def render_structure_detail(structure_id):
    """Placeholder detail body (name, type, status, legs); the full view arrives in Phase 5.3."""
    if not structure_id:
        return ""
    structure = container.repository.get_structure(structure_id)
    if structure is None:
        return html.Div("Structure not found.", style={"color": COLORS["ACCENT_RED"]})

    legs = [
        html.Li(f"{leg.ratio:+d} × {leg.contract.display_name}  ·  {leg.lots:g} lots")
        for leg in structure.legs
    ]
    return html.Div(
        [
            html.H4(structure.name, style={"color": COLORS["TEXT_PRIMARY"]}),
            html.Div(
                f"{structure.structure_type.value.capitalize()} · {structure.status.value.upper()}",
                style={"color": COLORS["TEXT_SECONDARY"], "marginBottom": "12px"},
            ),
            html.Ul(legs),
            html.Div("Full detail view coming in Phase 5.3.", style={"color": COLORS["TEXT_SECONDARY"]}),
        ]
    )


def register_structures_callbacks(app) -> None:
    """Attach the Structures tab callbacks to the Dash app."""
    # Triggered by the portfolio PnL store (not the raw interval) so rows never lag a refresh behind it.
    app.callback(
        Output("structures-active-grid", "rowData"),
        Input("store-portfolio-pnl", "data"),
        Input("filter-structure-status", "value"),
        Input("filter-structure-product", "value"),
        Input("filter-structure-sort", "value"),
        State("store-live-prices", "data"),
    )(update_active_structures)

    app.callback(
        Output("structures-closed-grid", "rowData"),
        Output("btn-toggle-closed", "children"),
        Input("url", "pathname"),
        Input("interval-pnl-refresh", "n_intervals"),
        Input("structures-closed-collapse", "is_open"),
    )(update_closed_structures)

    app.callback(
        Output("structures-closed-collapse", "is_open"),
        Input("btn-toggle-closed", "n_clicks"),
        State("structures-closed-collapse", "is_open"),
        prevent_initial_call=True,
    )(toggle_closed)

    app.callback(
        Output("modal-structure-detail", "is_open"),
        Output("store-selected-structure-id", "data"),
        Input("structures-active-grid", "cellRendererData"),
        prevent_initial_call=True,
    )(handle_structure_action)

    app.callback(
        Output("modal-structure-detail", "is_open", allow_duplicate=True),
        Input("btn-close-structure-detail", "n_clicks"),
        prevent_initial_call=True,
    )(close_detail_modal)

    app.callback(
        Output("modal-structure-detail-body", "children"),
        Input("store-selected-structure-id", "data"),
    )(render_structure_detail)
