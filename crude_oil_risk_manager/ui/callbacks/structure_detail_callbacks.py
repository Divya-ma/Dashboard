"""Structure detail callbacks: rendering, trade entry, full exit and edit mode.

Calculations live in core.trade_entry / core.structure_edit (which use core.pnl);
these functions load data, call them, persist the result and render feedback.
All database access happens here, never in the layout.

Some outputs differ from the naive wiring on purpose: content is rendered into
`modal-structure-detail-body` (not the whole modal, which would delete its header
and Close button), and the confirmation modals have static footers with only their
body replaced, so their buttons always exist.
"""

import logging

from dash import ALL, MATCH, Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from core.models import StructureStatus
from core.pnl import calculate_structure_transaction_costs
from core.structure_builder import StructureBuildError
from core.structure_edit import apply_leg_edits
from core.structure_view import prices_from_store, structure_entry_price, structure_live_price
from core.trade_entry import (
    TradeError,
    enter_trade,
    exit_structure,
    partial_exit_legs,
    recompute_structure_from_trades,
    structure_open_lots,
    structure_pnl,
)
from ui.callbacks.structures_callbacks import reuse_closed_structure, toast_update as _toast, update_active_structures
from ui.container import container
from ui.layouts.shell import COLORS
from ui.layouts.structure_detail import (
    format_price,
    format_pnl,
    pnl_span,
    structure_detail_layout,
)

logger = logging.getLogger(__name__)

_GRID_STATES = (
    State("filter-structure-status", "value"),
    State("filter-structure-product", "value"),
    State("filter-structure-sort", "value"),
    State("store-portfolio-pnl", "data"),
    State("store-live-prices", "data"),
)
_TOAST_OUTPUTS = (
    Output("detail-toast", "is_open", allow_duplicate=True),
    Output("detail-toast", "children", allow_duplicate=True),
    Output("detail-toast", "icon", allow_duplicate=True),
    Output("detail-toast", "header", allow_duplicate=True),
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _active_alert_trade_ids(structure, trades, live_prices) -> frozenset[str]:
    """Trade ids with a stop-loss/target level currently hit and still repeating (see
    core.alerts.AlertManager.active_alert_kinds) — these get a "Stop Alerts" button."""
    alert_manager = container.alert_manager
    if alert_manager is None:
        return frozenset()
    live = structure_live_price(structure, prices_from_store(live_prices))
    return frozenset(
        trade.trade_id
        for trade in trades
        if (trade.stop_loss_price is not None or trade.target_price is not None)
        and alert_manager.active_alert_kinds(trade, live)
    )


def _render_body(structure_id, live_prices, edit_mode: bool = False):
    """Load everything the detail layout needs and build it."""
    repository = container.repository
    structure = repository.get_structure(structure_id) if structure_id else None
    if structure is None:
        return html.Div("Structure not found.", style={"color": COLORS["ACCENT_RED"]})
    trades = repository.get_trades_for_structure(structure_id)
    active_alert_trade_ids = _active_alert_trade_ids(structure, trades, live_prices)
    return structure_detail_layout(
        structure, live_prices, trades, repository.get_latest_pnl(structure_id), edit_mode, active_alert_trade_ids
    )


def _grid_rows(status_filter, product_filter, sort_by, portfolio_pnl, live_prices):
    return update_active_structures(portfolio_pnl, status_filter, product_filter, sort_by, live_prices)


def _load(structure_id):
    structure = container.repository.get_structure(structure_id) if structure_id else None
    if structure is None:
        raise PreventUpdate
    return structure


# ----------------------------------------------------------------------
# Rendering / small helpers
# ----------------------------------------------------------------------


def render_structure_detail(structure_id, live_prices):
    if not structure_id:
        return ""
    return _render_body(structure_id, live_prices)


def fill_live_price(n_clicks, structure_id, live_prices):
    """Structure live price (composite of the leg prices) into an entry/exit price box."""
    if not n_clicks:
        raise PreventUpdate
    live = structure_live_price(_load(structure_id), prices_from_store(live_prices))
    if live is None:
        raise PreventUpdate
    return round(live, 4)


def fill_exit_all_lots(n_clicks, structure_id):
    if not n_clicks:
        raise PreventUpdate
    return structure_open_lots(_load(structure_id))


def update_live_labels(live_prices, structure_id):
    if not structure_id:
        raise PreventUpdate
    live = structure_live_price(_load(structure_id), prices_from_store(live_prices))
    label = f"Live: {format_price(live)}"
    return label, label, f"Current live price: {format_price(live)}"


def toggle_add_trade(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open


def toggle_leg_prices(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open


def fill_live_leg_price(n_clicks, structure_id, live_prices):
    """One leg's live price into its price box (Use Live button, MATCH-paired by leg index)."""
    if not n_clicks:
        raise PreventUpdate
    index = callback_context.triggered_id["index"]
    structure = _load(structure_id)
    if index >= len(structure.legs):
        raise PreventUpdate
    price = prices_from_store(live_prices).get(structure.legs[index].contract.symbol)
    if price is None:
        raise PreventUpdate
    return round(price, 4)


def toggle_leg_exit(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open


def fill_live_exit_leg_price(n_clicks, structure_id, live_prices):
    """One leg's live price into its exit price box (Use Live button, MATCH-paired by leg_id)."""
    if not n_clicks:
        raise PreventUpdate
    leg_id = callback_context.triggered_id["index"]
    structure = _load(structure_id)
    leg = next((leg for leg in structure.legs if leg.leg_id == leg_id), None)
    if leg is None:
        raise PreventUpdate
    price = prices_from_store(live_prices).get(leg.contract.symbol)
    if price is None:
        raise PreventUpdate
    return round(price, 4)


# ----------------------------------------------------------------------
# Previews
# ----------------------------------------------------------------------


def preview_entry_pnl(entry_price, lots, direction, structure_id, live_prices):
    """PnL the position would show right now if entered at `entry_price` on `direction`."""
    if not structure_id or entry_price is None or lots is None:
        return ""
    structure = container.repository.get_structure(structure_id)
    if structure is None:
        return ""
    live = structure_live_price(structure, prices_from_store(live_prices))
    if live is None:
        return "Live price unavailable; PnL preview needs a live price."
    try:
        pnl = structure_pnl(structure, entry_price, live, lots, direction)
    except ValueError:
        return ""
    return [f"At {entry_price:g}, current PnL would be: ", pnl_span(pnl)]


def preview_exit_pnl(exit_price, lots, structure_id):
    if not structure_id or exit_price is None or lots is None:
        return ""
    structure = container.repository.get_structure(structure_id)
    entry = structure_entry_price(structure) if structure else None
    if entry is None:
        return ""
    try:
        pnl = structure_pnl(structure, entry, exit_price, lots)
    except ValueError:
        return ""
    return [f"Realized PnL at {exit_price:g}: ", pnl_span(pnl)]


# ----------------------------------------------------------------------
# Trade entry
# ----------------------------------------------------------------------


def confirm_trade_entry(
    n_clicks, price, lots, direction, notes, stop_loss_price, target_price, leg_prices, structure_id, live_prices,
    status_filter, product_filter, sort_by, portfolio_pnl, grid_live_prices,
):
    """Enter a trade at one structure price; the system derives the per-leg entries unless
    `leg_prices` (every slot filled in) gives real per-leg fills instead."""
    if not n_clicks:
        raise PreventUpdate
    repository = container.repository
    structure = _load(structure_id)
    leg_prices = [p if p not in (None, "") else None for p in (leg_prices or [])] or None
    try:
        result = enter_trade(
            structure, price, lots, direction, notes, prices_from_store(live_prices),
            stop_loss_price=stop_loss_price, target_price=target_price, leg_prices=leg_prices,
        )
        # Legs first (with the audit line), then status, then the trade record.
        repository.update_structure_legs(structure_id, result.legs, result.audit_note)
        if structure.status != result.status:
            repository.update_structure_status(structure_id, result.status)
        repository.save_trade(result.trade)
    except TradeError as exc:
        return (no_update, no_update, *_toast(exc.errors, ok=False, header="Trade not entered"))
    except Exception:  # noqa: BLE001
        logger.exception("Trade entry failed for %s", structure_id)
        return (no_update, no_update, *_toast("Could not save the trade; see the server log.", ok=False))

    rows = _grid_rows(status_filter, product_filter, sort_by, portfolio_pnl, grid_live_prices)
    message = f"Trade entered: {lots:g} lots at {price:g}"
    return (_render_body(structure_id, live_prices), rows, *_toast(message, header="Trade entered"))


# ----------------------------------------------------------------------
# Full exit
# ----------------------------------------------------------------------


def open_exit_confirmation(n_clicks, exit_price, lots, structure_id):
    """Always shown before an exit runs; also validates and previews the realized PnL."""
    if not n_clicks:
        raise PreventUpdate
    structure = _load(structure_id)
    try:
        result = exit_structure(structure, exit_price, lots, None)
    except TradeError as exc:
        body = [html.Div(f"⛔ {message}", style={"color": COLORS["ACCENT_RED"]}) for message in exc.errors]
        return True, body, True
    prior_tc = calculate_structure_transaction_costs(container.repository.get_trades_for_structure(structure_id))
    total_tc = prior_tc + result.trade.transaction_cost
    net_pnl = result.realized_pnl - total_tc
    body = [
        html.P(f"Are you sure you want to close this structure at {exit_price:g}?"),
        html.P(["Realized PnL (before transaction costs): ", pnl_span(result.realized_pnl, fontWeight="bold")]),
    ]
    if total_tc:
        body.append(html.P(f"Total transaction cost (entry + exit): -${total_tc:,.0f}", style={"color": COLORS["ACCENT_YELLOW"]}))
        body.append(html.P(["Net realized PnL: ", pnl_span(net_pnl, fontWeight="bold")]))
    body.append(html.P("This cannot be undone.", style={"color": COLORS["ACCENT_YELLOW"]}))
    return True, body, False


def cancel_exit(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


def execute_full_exit(
    n_clicks, exit_price, lots, notes, structure_id,
    status_filter, product_filter, sort_by, portfolio_pnl, live_prices,
):
    """Close the structure: exit trade, zeroed legs, status CLOSED (manual). No partial exits."""
    if not n_clicks:
        raise PreventUpdate
    repository = container.repository
    structure = _load(structure_id)
    prior_tc = calculate_structure_transaction_costs(repository.get_trades_for_structure(structure_id))
    try:
        result = exit_structure(structure, exit_price, lots, notes)
        repository.update_structure_legs(structure_id, result.legs, result.audit_note)
        repository.save_trade(result.trade)
        repository.update_structure_status(structure_id, StructureStatus.CLOSED, close_trigger="manual")
    except TradeError as exc:
        return (no_update, no_update, *_toast(exc.errors, ok=False, header="Exit not executed"), False)
    except Exception:  # noqa: BLE001
        logger.exception("Exit failed for %s", structure_id)
        return (no_update, no_update, *_toast("Could not save the exit; see the server log.", ok=False), False)

    rows = _grid_rows(status_filter, product_filter, sort_by, portfolio_pnl, live_prices)
    net_pnl = result.realized_pnl - prior_tc - result.trade.transaction_cost
    message = f"{structure.name} closed at {exit_price:g}. Realized PnL (net of TC): {format_pnl(net_pnl)}"
    # The portfolio PnL stop is re-checked on the next 5s PnL refresh (shell_callbacks.check_alerts).
    return (False, rows, *_toast(message, header="Structure closed"), False)


# ----------------------------------------------------------------------
# Per-leg partial exit
# ----------------------------------------------------------------------


def _leg_exits_from_inputs(selected, lots_values, price_values, ids) -> dict:
    """{leg_id: (price, lots)} for every checked row, from the leg-exit form's
    checklist/lots/price State lists — all positionally aligned via `ids` (the matched
    component ids, read alongside them from the same pattern-matched row order)."""
    leg_exits = {}
    for checked, lots, price, id_dict in zip(selected, lots_values, price_values, ids):
        if checked:
            leg_exits[id_dict["index"]] = (price, lots)
    return leg_exits


def open_leg_exit_confirmation(n_clicks, selected, lots_values, price_values, ids, structure_id):
    """Always shown before a per-leg exit runs; also validates and previews realized PnL per leg."""
    if not n_clicks:
        raise PreventUpdate
    structure = _load(structure_id)
    leg_exits = _leg_exits_from_inputs(selected, lots_values, price_values, ids)
    try:
        result = partial_exit_legs(structure, leg_exits, None)
    except TradeError as exc:
        body = [html.Div(f"⛔ {message}", style={"color": COLORS["ACCENT_RED"]}) for message in exc.errors]
        return True, body, True

    by_id = {leg.leg_id: leg for leg in structure.legs}
    body = [html.P("Are you sure you want to exit these legs?")]
    for trade in result.trades:
        symbol = by_id[trade.leg_id].contract.symbol
        body.append(html.P([f"{symbol}: {trade.lots:g} lots at {trade.price:g} — realized ", pnl_span(trade.realized_pnl, fontWeight="bold")]))
    total_tc = sum(trade.transaction_cost for trade in result.trades)
    if total_tc:
        body.append(html.P(f"Total transaction cost: -${total_tc:,.0f}", style={"color": COLORS["ACCENT_YELLOW"]}))
    body.append(html.P(["Total realized PnL (before transaction costs): ", pnl_span(result.realized_pnl, fontWeight="bold")]))
    body.append(html.P("This cannot be undone.", style={"color": COLORS["ACCENT_YELLOW"]}))
    return True, body, False


def cancel_leg_exit(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


def execute_leg_exit(
    n_clicks, selected, lots_values, price_values, ids, notes, structure_id,
    status_filter, product_filter, sort_by, portfolio_pnl, live_prices,
):
    """Persist a per-leg exit: one Trade row per leg acted on; status becomes CLOSED only
    once every leg is at zero, PARTIALLY_CLOSED if some but not all are."""
    if not n_clicks:
        raise PreventUpdate
    repository = container.repository
    structure = _load(structure_id)
    leg_exits = _leg_exits_from_inputs(selected, lots_values, price_values, ids)
    try:
        result = partial_exit_legs(structure, leg_exits, notes)
        repository.update_structure_legs(structure_id, result.legs, result.audit_note)
        for trade in result.trades:
            repository.save_trade(trade)
        if structure.status != result.status:
            close_trigger = "manual" if result.status == StructureStatus.CLOSED else None
            repository.update_structure_status(structure_id, result.status, close_trigger=close_trigger)
    except TradeError as exc:
        return (no_update, no_update, *_toast(exc.errors, ok=False, header="Exit not executed"), False)
    except Exception:  # noqa: BLE001
        logger.exception("Leg exit failed for %s", structure_id)
        return (no_update, no_update, *_toast("Could not save the exit; see the server log.", ok=False), False)

    rows = _grid_rows(status_filter, product_filter, sort_by, portfolio_pnl, live_prices)
    legs_desc = ", ".join(f"{t.lots:g} lots at {t.price:g}" for t in result.trades)
    message = f"{structure.name}: exited {legs_desc}. Realized PnL (gross): {format_pnl(result.realized_pnl)}"
    return (_render_body(structure_id, live_prices), rows, *_toast(message, header="Legs exited"), False)


# ----------------------------------------------------------------------
# Reuse a closed structure
# ----------------------------------------------------------------------


def toggle_reuse_panel(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open


def reuse_structure(n_clicks, structure_id, new_name):
    """Save a fresh shell copy of the closed structure (optionally renamed) and close the modal."""
    if not n_clicks:
        raise PreventUpdate
    _load(structure_id)
    shell = reuse_closed_structure(structure_id)
    if shell is None:
        return (no_update, *_toast("Only a closed structure can be reused.", ok=False))
    if (new_name or "").strip():
        shell = shell.model_copy(update={"name": new_name.strip()[:100]})
        container.repository.save_structure(shell)
    return (False, *_toast(f"Structure '{shell.name}' ready as new shell", header="Structure reused"))


# ----------------------------------------------------------------------
# Rename structure
# ----------------------------------------------------------------------


def toggle_rename_panel(n_clicks, is_open):
    if not n_clicks:
        raise PreventUpdate
    return not is_open


def cancel_rename(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


def save_rename(n_clicks, new_name, structure_id, live_prices, status_filter, product_filter, sort_by, portfolio_pnl):
    """Rename a structure; legs, trades and PnL history are untouched."""
    if not n_clicks:
        raise PreventUpdate
    structure = _load(structure_id)
    name = (new_name or "").strip()
    if not name:
        return no_update, no_update, no_update, *_toast("Structure name cannot be empty.", ok=False, header="Not renamed")
    if len(name) > 100:
        return no_update, no_update, no_update, *_toast("Structure name is limited to 100 characters.", ok=False, header="Not renamed")
    if name == structure.name:
        return False, no_update, no_update, *_toast("Name unchanged.", header="No change")

    container.repository.update_structure_name(structure_id, name)
    rows = _grid_rows(status_filter, product_filter, sort_by, portfolio_pnl, live_prices)
    body = _render_body(structure_id, live_prices)
    return False, body, rows, *_toast(f"Renamed to '{name}'.", header="Structure renamed")


# ----------------------------------------------------------------------
# Edit mode
# ----------------------------------------------------------------------


def initiate_edit(n_clicks, structure_id, live_prices):
    """No trades: edit straight away. Trades exist: ask first (warn + confirm, never block)."""
    if not n_clicks:
        raise PreventUpdate
    _load(structure_id)
    if container.repository.get_trades_for_structure(structure_id):
        return True, no_update
    return no_update, _render_body(structure_id, live_prices, edit_mode=True)


def proceed_with_edit(n_clicks, structure_id, live_prices):
    if not n_clicks:
        raise PreventUpdate
    structure = _load(structure_id)
    container.repository.update_structure_legs(
        structure_id, structure.legs, "edit mode opened after confirmation (structure has trades)"
    )
    return _render_body(structure_id, live_prices, edit_mode=True), False


def cancel_edit_confirm(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


def cancel_edit(n_clicks, structure_id, live_prices):
    if not n_clicks:
        raise PreventUpdate
    return _render_body(structure_id, live_prices)


def save_edit(n_clicks, symbols, ratios, entry_prices, structure_id, live_prices):
    """Save inline leg edits (symbol, ratio, and a traded leg's entry price correction)
    with an audit line describing what changed."""
    if not n_clicks:
        raise PreventUpdate
    repository = container.repository
    structure = _load(structure_id)
    try:
        result = apply_leg_edits(structure, symbols, ratios, repository.get_contract, entry_prices)
        for contract in result.new_contracts:
            repository.save_contract(contract)
        repository.update_structure_legs(structure_id, result.legs, result.audit_note)
    except StructureBuildError as exc:
        return (no_update, *_toast(exc.errors, ok=False, header="Edit not saved"))
    except Exception:  # noqa: BLE001
        logger.exception("Saving edit failed for %s", structure_id)
        return (no_update, *_toast("Could not save the edit; see the server log.", ok=False))
    return (_render_body(structure_id, live_prices), *_toast(result.warnings or "Changes saved and logged.", header="Structure updated"))


# ----------------------------------------------------------------------
# Delete structure / delete trade
# ----------------------------------------------------------------------


def open_delete_structure_confirm(n_clicks, structure_id):
    """Ask for confirmation before permanently deleting a whole structure."""
    if not n_clicks:
        raise PreventUpdate
    structure = _load(structure_id)
    trade_count = len(container.repository.get_trades_for_structure(structure_id))
    body = [
        html.P(f"Permanently delete '{structure.name}'?"),
        html.P(
            f"This removes the structure, all {trade_count} trade(s) and its PnL history. "
            "This cannot be undone — use this for a structure created with the wrong parameters, "
            "not to close out a real position (use Exit Structure for that).",
            style={"color": COLORS["ACCENT_YELLOW"]},
        ),
    ]
    return True, body, {"kind": "structure", "id": structure_id}


def open_delete_trade_confirm(delete_clicks, structure_id):
    """Ask for confirmation before deleting one trade, previewing what will be recomputed."""
    trigger = callback_context.triggered_id
    if not isinstance(trigger, dict) or trigger.get("type") != "trade-delete-btn":
        raise PreventUpdate
    if not callback_context.triggered[0]["value"]:
        raise PreventUpdate  # the button was just created (id assigned), not actually clicked
    trade_id = trigger["index"]
    trades = {t.trade_id: t for t in container.repository.get_trades_for_structure(structure_id)}
    trade = trades.get(trade_id)
    if trade is None:
        raise PreventUpdate
    body = [
        html.P(
            f"Delete the {trade.event_type.value.replace('_', ' ')} of {trade.lots:g} lots "
            f"at {trade.price:g} ({trade.timestamp:%Y-%m-%d %H:%M})?"
        ),
        html.P(
            "Lots, entry price and PnL will be recomputed as if this trade had never been "
            "entered; any later trades on this structure are replayed on top of that. This cannot be undone.",
            style={"color": COLORS["ACCENT_YELLOW"]},
        ),
    ]
    return True, body, {"kind": "trade", "id": trade_id}


def cancel_delete(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False, None


def stop_trade_alert(n_clicks_list, structure_id, live_prices):
    """Stop the repeating stop-loss/target alert for one trade (button click, no confirmation
    needed since it only silences alerts and resumes automatically next time the level is
    crossed again)."""
    trigger = callback_context.triggered_id
    if not isinstance(trigger, dict) or trigger.get("type") != "trade-stop-alert-btn":
        raise PreventUpdate
    if not callback_context.triggered[0]["value"]:
        raise PreventUpdate  # the button was just created (id assigned), not actually clicked
    if container.alert_manager is None:
        raise PreventUpdate
    container.alert_manager.stop_repeating_alert(trigger["index"])
    body = _render_body(structure_id, live_prices)
    return (body, *_toast("Repeating alerts stopped for this trade.", header="Alerts stopped"))


def _delete_trade_and_recompute(structure_id: str, trade_id: str) -> None:
    """Remove one trade and replay every remaining trade so lots, entry price, status and
    any surviving exit's realized PnL land exactly where they'd be without it."""
    repository = container.repository
    structure = repository.get_structure(structure_id)
    if structure is None:
        return
    remaining = [t for t in repository.get_trades_for_structure(structure_id) if t.trade_id != trade_id]
    result = recompute_structure_from_trades(structure, remaining)
    repository.delete_trade(trade_id)
    repository.update_structure_legs(structure_id, result.structure.legs, result.audit_note)
    repository.set_structure_lifecycle(
        structure_id, result.structure.status, result.structure.closed_at, result.structure.close_trigger
    )
    for exit_trade_id, (realized_pnl, transaction_cost) in result.exit_trade_updates.items():
        repository.update_trade_realized_pnl(exit_trade_id, realized_pnl, transaction_cost)


def execute_delete(
    n_clicks, pending, structure_id, live_prices,
    status_filter, product_filter, sort_by, portfolio_pnl,
):
    """Dispatch to structure or trade deletion depending on what was confirmed."""
    if not n_clicks or not pending:
        raise PreventUpdate
    kind, target_id = pending.get("kind"), pending.get("id")
    rows_args = (status_filter, product_filter, sort_by, portfolio_pnl, live_prices)

    try:
        if kind == "structure":
            structure = container.repository.get_structure(target_id)
            name = structure.name if structure else "Structure"
            container.repository.delete_structure(target_id)
            rows = _grid_rows(*rows_args)
            return (False, False, no_update, rows, None, *_toast(f"'{name}' deleted.", header="Structure deleted"))
        if kind == "trade":
            _delete_trade_and_recompute(structure_id, target_id)
            rows = _grid_rows(*rows_args)
            body = _render_body(structure_id, live_prices)
            return (False, no_update, body, rows, None, *_toast("Trade deleted; PnL recomputed.", header="Trade deleted"))
        raise PreventUpdate
    except TradeError as exc:
        return (False, no_update, no_update, no_update, None, *_toast(exc.errors, ok=False, header="Delete failed"))
    except Exception:  # noqa: BLE001
        logger.exception("Delete failed for %s %s", kind, target_id)
        return (False, no_update, no_update, no_update, None, *_toast("Could not delete; see the server log.", ok=False))


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------

def _body_dup() -> Output:
    return Output("modal-structure-detail-body", "children", allow_duplicate=True)


def register_structure_detail_callbacks(app) -> None:
    """Attach the structure detail callbacks to the Dash app."""
    app.callback(
        Output("modal-structure-detail-body", "children"),
        Input("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
        prevent_initial_call=True,
    )(render_structure_detail)

    for button, target in (("btn-use-live-price", "trade-entry-price"), ("btn-use-live-exit-price", "trade-exit-price")):
        app.callback(
            Output(target, "value"),
            Input(button, "n_clicks"),
            State("store-selected-structure-id", "data"),
            State("store-live-prices", "data"),
            prevent_initial_call=True,
        )(fill_live_price)

    app.callback(
        Output("trade-exit-lots", "value"),
        Input("btn-exit-all-lots", "n_clicks"),
        State("store-selected-structure-id", "data"),
        prevent_initial_call=True,
    )(fill_exit_all_lots)

    app.callback(
        Output("trade-live-price-label", "children"),
        Output("exit-live-price-label", "children"),
        Output("trade-alert-live-reference", "children"),
        Input("store-live-prices", "data"),
        State("store-selected-structure-id", "data"),
        prevent_initial_call=True,
    )(update_live_labels)

    app.callback(
        Output("trade-entry-collapse", "is_open"),
        Input("btn-toggle-add-trade", "n_clicks"),
        State("trade-entry-collapse", "is_open"),
        prevent_initial_call=True,
    )(toggle_add_trade)

    app.callback(
        Output("leg-prices-collapse", "is_open"),
        Input("btn-toggle-leg-prices", "n_clicks"),
        State("leg-prices-collapse", "is_open"),
        prevent_initial_call=True,
    )(toggle_leg_prices)

    app.callback(
        Output({"type": "trade-leg-price", "index": MATCH}, "value"),
        Input({"type": "btn-use-live-leg-price", "index": MATCH}, "n_clicks"),
        State("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
        prevent_initial_call=True,
    )(fill_live_leg_price)

    app.callback(
        Output("leg-exit-collapse", "is_open"),
        Input("btn-toggle-leg-exit", "n_clicks"),
        State("leg-exit-collapse", "is_open"),
        prevent_initial_call=True,
    )(toggle_leg_exit)

    app.callback(
        Output({"type": "exit-leg-price", "index": MATCH}, "value"),
        Input({"type": "btn-use-live-exit-leg-price", "index": MATCH}, "n_clicks"),
        State("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
        prevent_initial_call=True,
    )(fill_live_exit_leg_price)

    app.callback(
        Output("trade-pnl-preview", "children"),
        Input("trade-entry-price", "value"),
        Input("trade-entry-lots", "value"),
        Input("trade-direction", "value"),
        State("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
    )(preview_entry_pnl)

    app.callback(
        Output("exit-pnl-preview", "children"),
        Input("trade-exit-price", "value"),
        Input("trade-exit-lots", "value"),
        State("store-selected-structure-id", "data"),
    )(preview_exit_pnl)

    app.callback(
        _body_dup(),
        Output("structures-active-grid", "rowData", allow_duplicate=True),
        *_TOAST_OUTPUTS,
        Input("btn-confirm-trade", "n_clicks"),
        State("trade-entry-price", "value"),
        State("trade-entry-lots", "value"),
        State("trade-direction", "value"),
        State("trade-entry-notes", "value"),
        State("trade-stop-loss-price", "value"),
        State("trade-target-price", "value"),
        State({"type": "trade-leg-price", "index": ALL}, "value"),
        State("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
        *_GRID_STATES[:4],
        State("store-live-prices", "data"),
        prevent_initial_call=True,
    )(confirm_trade_entry)

    app.callback(
        Output("modal-confirm-exit", "is_open"),
        Output("modal-confirm-exit-body", "children"),
        Output("btn-confirm-exit-final", "disabled"),
        Input("btn-confirm-exit", "n_clicks"),
        State("trade-exit-price", "value"),
        State("trade-exit-lots", "value"),
        State("store-selected-structure-id", "data"),
        prevent_initial_call=True,
    )(open_exit_confirmation)

    app.callback(
        Output("modal-confirm-exit", "is_open", allow_duplicate=True),
        Input("btn-cancel-exit", "n_clicks"),
        prevent_initial_call=True,
    )(cancel_exit)

    app.callback(
        Output("modal-structure-detail", "is_open", allow_duplicate=True),
        Output("structures-active-grid", "rowData", allow_duplicate=True),
        *_TOAST_OUTPUTS,
        Output("modal-confirm-exit", "is_open", allow_duplicate=True),
        Input("btn-confirm-exit-final", "n_clicks"),
        State("trade-exit-price", "value"),
        State("trade-exit-lots", "value"),
        State("trade-exit-notes", "value"),
        State("store-selected-structure-id", "data"),
        *_GRID_STATES,
        prevent_initial_call=True,
    )(execute_full_exit)

    app.callback(
        Output("modal-confirm-leg-exit", "is_open"),
        Output("modal-confirm-leg-exit-body", "children"),
        Output("btn-confirm-leg-exit-final", "disabled"),
        Input("btn-open-leg-exit", "n_clicks"),
        State({"type": "exit-leg-selected", "index": ALL}, "value"),
        State({"type": "exit-leg-lots", "index": ALL}, "value"),
        State({"type": "exit-leg-price", "index": ALL}, "value"),
        State({"type": "exit-leg-selected", "index": ALL}, "id"),
        State("store-selected-structure-id", "data"),
        prevent_initial_call=True,
    )(open_leg_exit_confirmation)

    app.callback(
        Output("modal-confirm-leg-exit", "is_open", allow_duplicate=True),
        Input("btn-cancel-leg-exit", "n_clicks"),
        prevent_initial_call=True,
    )(cancel_leg_exit)

    app.callback(
        _body_dup(),
        Output("structures-active-grid", "rowData", allow_duplicate=True),
        *_TOAST_OUTPUTS,
        Output("modal-confirm-leg-exit", "is_open", allow_duplicate=True),
        Input("btn-confirm-leg-exit-final", "n_clicks"),
        State({"type": "exit-leg-selected", "index": ALL}, "value"),
        State({"type": "exit-leg-lots", "index": ALL}, "value"),
        State({"type": "exit-leg-price", "index": ALL}, "value"),
        State({"type": "exit-leg-selected", "index": ALL}, "id"),
        State("leg-exit-notes", "value"),
        State("store-selected-structure-id", "data"),
        *_GRID_STATES,
        prevent_initial_call=True,
    )(execute_leg_exit)

    app.callback(
        Output("reuse-collapse", "is_open"),
        Input("btn-reuse-structure", "n_clicks"),
        State("reuse-collapse", "is_open"),
        prevent_initial_call=True,
    )(toggle_reuse_panel)

    app.callback(
        Output("modal-structure-detail", "is_open", allow_duplicate=True),
        *_TOAST_OUTPUTS,
        Input("btn-confirm-reuse", "n_clicks"),
        State("store-selected-structure-id", "data"),
        State("reuse-structure-name", "value"),
        prevent_initial_call=True,
    )(reuse_structure)

    app.callback(
        Output("rename-collapse", "is_open", allow_duplicate=True),
        Input("btn-rename-structure", "n_clicks"),
        State("rename-collapse", "is_open"),
        prevent_initial_call=True,
    )(toggle_rename_panel)

    app.callback(
        Output("rename-collapse", "is_open", allow_duplicate=True),
        Input("btn-cancel-rename", "n_clicks"),
        prevent_initial_call=True,
    )(cancel_rename)

    app.callback(
        Output("rename-collapse", "is_open", allow_duplicate=True),
        _body_dup(),
        Output("structures-active-grid", "rowData", allow_duplicate=True),
        *_TOAST_OUTPUTS,
        Input("btn-confirm-rename", "n_clicks"),
        State("rename-structure-input", "value"),
        State("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
        State("filter-structure-status", "value"),
        State("filter-structure-product", "value"),
        State("filter-structure-sort", "value"),
        State("store-portfolio-pnl", "data"),
        prevent_initial_call=True,
    )(save_rename)

    app.callback(
        Output("modal-confirm-edit", "is_open"),
        _body_dup(),
        Input("btn-edit-structure", "n_clicks"),
        State("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
        prevent_initial_call=True,
    )(initiate_edit)

    app.callback(
        _body_dup(),
        Output("modal-confirm-edit", "is_open", allow_duplicate=True),
        Input("btn-confirm-edit-proceed", "n_clicks"),
        State("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
        prevent_initial_call=True,
    )(proceed_with_edit)

    app.callback(
        Output("modal-confirm-edit", "is_open", allow_duplicate=True),
        Input("btn-cancel-edit-confirm", "n_clicks"),
        prevent_initial_call=True,
    )(cancel_edit_confirm)

    app.callback(
        _body_dup(),
        Input("btn-cancel-edit", "n_clicks"),
        State("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
        prevent_initial_call=True,
    )(cancel_edit)

    app.callback(
        _body_dup(),
        *_TOAST_OUTPUTS,
        Input("btn-save-edit", "n_clicks"),
        State({"type": "edit-leg-symbol", "index": ALL}, "value"),
        State({"type": "edit-leg-ratio", "index": ALL}, "value"),
        State("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
        prevent_initial_call=True,
    )(save_edit)

    app.callback(
        Output("modal-confirm-delete", "is_open", allow_duplicate=True),
        Output("modal-confirm-delete-body", "children", allow_duplicate=True),
        Output("store-pending-delete", "data", allow_duplicate=True),
        Input("btn-delete-structure", "n_clicks"),
        State("store-selected-structure-id", "data"),
        prevent_initial_call=True,
    )(open_delete_structure_confirm)

    app.callback(
        Output("modal-confirm-delete", "is_open", allow_duplicate=True),
        Output("modal-confirm-delete-body", "children", allow_duplicate=True),
        Output("store-pending-delete", "data", allow_duplicate=True),
        Input({"type": "trade-delete-btn", "index": ALL}, "n_clicks"),
        State("store-selected-structure-id", "data"),
        prevent_initial_call=True,
    )(open_delete_trade_confirm)

    app.callback(
        Output("modal-confirm-delete", "is_open", allow_duplicate=True),
        Output("store-pending-delete", "data", allow_duplicate=True),
        Input("btn-cancel-delete", "n_clicks"),
        prevent_initial_call=True,
    )(cancel_delete)

    app.callback(
        _body_dup(),
        *_TOAST_OUTPUTS,
        Input({"type": "trade-stop-alert-btn", "index": ALL}, "n_clicks"),
        State("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
        prevent_initial_call=True,
    )(stop_trade_alert)

    app.callback(
        Output("modal-confirm-delete", "is_open", allow_duplicate=True),
        Output("modal-structure-detail", "is_open", allow_duplicate=True),
        _body_dup(),
        Output("structures-active-grid", "rowData", allow_duplicate=True),
        Output("store-pending-delete", "data", allow_duplicate=True),
        *_TOAST_OUTPUTS,
        Input("btn-confirm-delete-final", "n_clicks"),
        State("store-pending-delete", "data"),
        State("store-selected-structure-id", "data"),
        State("store-live-prices", "data"),
        State("filter-structure-status", "value"),
        State("filter-structure-product", "value"),
        State("filter-structure-sort", "value"),
        State("store-portfolio-pnl", "data"),
        prevent_initial_call=True,
    )(execute_delete)
