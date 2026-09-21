"""Structures tab callbacks: active/closed grids, closed-section toggle and row actions.

Row building lives in core.structure_view; PnL comes from `store-portfolio-pnl`
(written by shell_callbacks.refresh_portfolio_pnl) and is never recalculated here.
"""

from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate

from core.models import StructureStatus
from core.structure_utils import clone_structure_as_shell
from core.structure_view import (
    build_active_rows,
    build_closed_rows,
    prices_from_store,
    statuses_for_filter,
)
from ui.container import container

ACTION_VIEW = "view"
ACTION_ENTER_TRADE = "enter_trade"
ACTION_EXIT = "exit"
_ACTIONS = {ACTION_VIEW, ACTION_ENTER_TRADE, ACTION_EXIT}


def toast_update(message, ok: bool = True, header: str | None = None) -> tuple:
    """(is_open, children, icon, header) for `detail-toast`, the page's feedback toast."""
    if isinstance(message, list):
        message = [html.Div(m) for m in message]
    return True, message, "success" if ok else "danger", header or ("Done" if ok else "Could not complete")


def reuse_closed_structure(structure_id):
    """Save a fresh SHELL copy of a closed structure and return it; None if it is not closed."""
    repository = container.repository
    source = repository.get_structure(structure_id) if structure_id else None
    if source is None or source.status != StructureStatus.CLOSED:
        return None
    shell = clone_structure_as_shell(source)
    repository.save_structure(shell)
    return shell


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
    """Row button clicks. View, Enter Trade and Exit all open the detail modal, which holds the trade forms."""
    payload = (cell_data or {}).get("value") or {}
    action = payload.get("action")
    structure_id = payload.get("structure_id") or (cell_data or {}).get("rowId")
    if action not in _ACTIONS or not structure_id:
        raise PreventUpdate

    return True, structure_id


def handle_closed_structure_action(
    cell_data, status_filter, product_filter, sort_by, portfolio_pnl, live_prices
):
    """'Reuse as Shell' on a closed row: save a fresh shell copy and refresh the active grid."""
    payload = (cell_data or {}).get("value") or {}
    structure_id = payload.get("structure_id") or (cell_data or {}).get("rowId")
    if payload.get("action") != "reuse" or not structure_id:
        raise PreventUpdate
    shell = reuse_closed_structure(structure_id)
    if shell is None:
        return (no_update, *toast_update("Only a closed structure can be reused.", ok=False))
    rows = update_active_structures(portfolio_pnl, status_filter, product_filter, sort_by, live_prices)
    return (rows, *toast_update(f"Structure '{shell.name}' ready as new shell", header="Structure reused"))


def close_detail_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


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
        Output("structures-active-grid", "rowData", allow_duplicate=True),
        Output("detail-toast", "is_open", allow_duplicate=True),
        Output("detail-toast", "children", allow_duplicate=True),
        Output("detail-toast", "icon", allow_duplicate=True),
        Output("detail-toast", "header", allow_duplicate=True),
        Input("structures-closed-grid", "cellRendererData"),
        State("filter-structure-status", "value"),
        State("filter-structure-product", "value"),
        State("filter-structure-sort", "value"),
        State("store-portfolio-pnl", "data"),
        State("store-live-prices", "data"),
        prevent_initial_call=True,
    )(handle_closed_structure_action)

