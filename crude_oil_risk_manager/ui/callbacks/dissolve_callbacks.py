"""Dissolve callbacks: choose the structures, give leg prices or a typed gross P&L, preview, confirm.

The maths is core.dissolve; this file loads the structures, calls it and saves the result.
"""

import logging

from dash import ALL, Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from core.dissolve import CLOSE_TRIGGER, DissolveError, dissolve_structures, leg_symbols
from core.models import StructureStatus
from core.structure_view import prices_from_store, structure_entry_price
from ui.callbacks.structures_callbacks import toast_update as _toast, update_active_structures
from ui.container import container
from ui.layouts.dissolve import MODE_TYPED
from ui.layouts.shell import COLORS

logger = logging.getLogger(__name__)

_OPEN = (StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED)
_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_SMALL_LABEL = {"fontSize": "12px", "color": COLORS["TEXT_SECONDARY"]}
_SHOWN = {"marginTop": "8px"}
_HIDDEN = {"display": "none"}

_GRID_STATES = (
    State("filter-structure-status", "value"),
    State("filter-structure-product", "value"),
    State("filter-structure-sort", "value"),
    State("store-portfolio-pnl", "data"),
    State("store-live-prices", "data"),
)
_FORM_STATES = (
    State("dissolve-select", "value"),
    State({"type": "dissolve-lots", "index": ALL}, "value"),
    State({"type": "dissolve-lots", "index": ALL}, "id"),
    State({"type": "dissolve-price", "index": ALL}, "value"),
    State({"type": "dissolve-price", "index": ALL}, "id"),
    State("dissolve-mode", "value"),
    State("dissolve-gross", "value"),
    State("dissolve-notes", "value"),
)


def _open_structures() -> list:
    return container.repository.get_all_structures(status_filter=list(_OPEN))


def _label(structure) -> str:
    traded = [leg for leg in structure.legs if leg.is_traded]
    side = "BUY" if traded and all(leg.direction == "buy" for leg in traded) else "SELL" if traded and all(leg.direction == "sell" for leg in traded) else "MIXED"
    entry = structure_entry_price(structure)
    return f"{structure.name} — {side} {structure.legs[0].lots:g} lots" + (f" @ {entry:g}" if entry is not None else "")


def open_or_close(open_clicks, cancel_clicks):
    """Open the modal with the current open structures to choose from, or close it."""
    from dash import callback_context

    trigger = callback_context.triggered_id
    if trigger == "btn-dissolve" and open_clicks:
        options = [{"label": _label(s), "value": s.structure_id} for s in _open_structures()]
        return True, options, []
    if trigger == "btn-dissolve-cancel" and cancel_clicks:
        return False, no_update, no_update
    raise PreventUpdate


def build_inputs(selected, live_prices):
    """A lots box per ticked structure (all open lots to start) and a price box per leg contract (the live price to start)."""
    if not selected:
        return [], []
    repository = container.repository
    structures = [s for s in (repository.get_structure(sid) for sid in selected) if s is not None]
    prices = prices_from_store(live_prices)
    lots_rows = [
        dbc.Row([
            dbc.Col(html.Div(s.name, style=_SMALL_LABEL), md=7, className="d-flex align-items-center"),
            dbc.Col(dbc.Input(id={"type": "dissolve-lots", "index": s.structure_id}, type="number", min=0, step="any",
                              value=s.legs[0].lots, size="sm"), md=5),
        ], className="g-1 mb-1")
        for s in structures
    ]
    lots_wrap = [html.Div("Or set each one on its own (of its open lots)", style=_SMALL_LABEL), *lots_rows]
    price_rows = [
        dbc.Row([
            dbc.Col(html.Div(symbol, style=_SMALL_LABEL), md=4, className="d-flex align-items-center"),
            dbc.Col(dbc.Input(id={"type": "dissolve-price", "index": symbol}, type="number", step="any",
                              value=round(prices[symbol], 4) if symbol in prices else None, placeholder="price", size="sm"), md=4),
        ], className="g-1 mb-1")
        for symbol in leg_symbols(structures)
    ]
    price_wrap = [html.Div("Price each leg contract was traded at when the group was cut (one price per contract)", style=_SMALL_LABEL), *price_rows]
    return lots_wrap, price_wrap


def set_all_lots(lots_all, ids):
    """One number of lots for every ticked structure (capped at what each has open); blank puts every box back to all its lots."""
    if not ids:
        raise PreventUpdate
    repository = container.repository
    values = []
    for item in ids:
        structure = repository.get_structure(item["index"])
        open_lots = structure.legs[0].lots if structure is not None else None
        values.append(open_lots if lots_all in (None, "") or open_lots is None else min(float(lots_all), open_lots))
    return values


def toggle_mode(mode):
    legs = mode != MODE_TYPED
    return _SHOWN if legs else _HIDDEN, _HIDDEN if legs else _SHOWN


def _compute(selected, lots_values, lots_ids, price_values, price_ids, mode, gross, notes):
    repository = container.repository
    structures = [s for s in (repository.get_structure(sid) for sid in selected or []) if s is not None]
    lots = {i["index"]: v for v, i in zip(lots_values or [], lots_ids or []) if v not in (None, "")}
    prices = {i["index"]: v for v, i in zip(price_values or [], price_ids or [])} if mode != MODE_TYPED else None  # leg-price mode only
    trades = {s.structure_id: repository.get_trades_for_structure(s.structure_id) for s in structures}
    return dissolve_structures(
        structures, lots=lots, leg_prices=prices, typed_gross=gross if mode == MODE_TYPED else None,
        notes=notes, trades_by_structure=trades,
    )


def _money(value: float) -> html.Span:
    color = COLORS["ACCENT_GREEN"] if value >= 0 else COLORS["ACCENT_RED"]
    return html.Span(f"{'+' if value >= 0 else '-'}${abs(value):,.0f}", style={"color": color, "fontWeight": "bold"})


def preview(selected, lots_values, lots_ids, price_values, price_ids, mode, gross, notes):
    if not selected:
        return html.Div("Tick the structures that were cut against each other.", style=_MUTED), True
    try:
        result = _compute(selected, lots_values, lots_ids, price_values, price_ids, mode, gross, notes)
    except DissolveError as exc:
        return [html.Div(f"⛔ {message}", style={"color": COLORS["ACCENT_RED"]}) for message in exc.errors], True
    rows = [html.Div([f"{item.structure.name}: {item.lots:g} lots, share of gross ", _money(item.realized_pnl),
                      f" (entry costs ${item.entry_costs:,.0f})"], style=_MUTED) for item in result.items]
    body = [
        *rows,
        html.Div([f"Gross P&L ({'from the leg prices' if result.method == 'legs' else 'typed'}): ", _money(result.gross_pnl)], style={"marginTop": "8px"}),
        html.Div(["Entry costs of all of them (already booked, no exit cost): ", html.Span(f"-${result.entry_costs:,.0f}", style={"color": COLORS["ACCENT_YELLOW"]})]),
        html.Div(["Net P&L: ", _money(result.net_pnl)], style={"fontSize": "16px"}),
        *[html.Div(f"⚠️ {w}", style={"color": COLORS["ACCENT_YELLOW"], "marginTop": "6px"}) for w in result.warnings],
        html.Div("This cannot be undone.", style={"color": COLORS["ACCENT_YELLOW"], "marginTop": "6px"}),
    ]
    return body, False


def execute(n_clicks, selected, lots_values, lots_ids, price_values, price_ids, mode, gross, notes,
            status_filter, product_filter, sort_by, portfolio_pnl, live_prices):
    if not n_clicks:
        raise PreventUpdate
    repository = container.repository
    try:
        result = _compute(selected, lots_values, lots_ids, price_values, price_ids, mode, gross, notes)
        names = ", ".join(item.structure.name for item in result.items)
        for item in result.items:
            sid = item.structure.structure_id
            repository.update_structure_legs(
                sid, item.legs, f"dissolved {item.lots:g} lots with {names}; share of gross {item.realized_pnl:+,.0f} (no exit cost)")
            repository.save_trade(item.trade)
            if item.status == StructureStatus.CLOSED:
                repository.update_structure_status(sid, StructureStatus.CLOSED, close_trigger=CLOSE_TRIGGER)
            if container.journal is not None:
                container.journal.record(repository.get_structure(sid), "exit", overwrite=True, trade_id=item.trade.trade_id)
    except DissolveError as exc:
        return (no_update, no_update, *_toast(exc.errors, ok=False, header="Not dissolved"))
    except Exception:  # noqa: BLE001
        logger.exception("Dissolve failed")
        return (no_update, no_update, *_toast("Could not save the dissolve; see the server log.", ok=False))
    rows = update_active_structures(portfolio_pnl, status_filter, product_filter, sort_by, live_prices)
    sign = "+" if result.net_pnl >= 0 else "-"
    message = f"Dissolved {len(result.items)} structures. Gross {result.gross_pnl:+,.0f}, entry costs -{result.entry_costs:,.0f}, net {sign}${abs(result.net_pnl):,.0f}."
    return (False, rows, *_toast(message, header="Structures dissolved"))


def register_dissolve_callbacks(app) -> None:
    app.callback(
        Output("modal-dissolve", "is_open"),
        Output("dissolve-select", "options"),
        Output("dissolve-select", "value"),
        Input("btn-dissolve", "n_clicks"),
        Input("btn-dissolve-cancel", "n_clicks"),
        prevent_initial_call=True,
    )(open_or_close)

    app.callback(
        Output("dissolve-lots-wrap", "children"),
        Output("dissolve-prices-wrap", "children"),
        Input("dissolve-select", "value"),
        State("store-live-prices", "data"),
    )(build_inputs)

    app.callback(
        Output({"type": "dissolve-lots", "index": ALL}, "value"),
        Input("dissolve-lots-all", "value"),
        State({"type": "dissolve-lots", "index": ALL}, "id"),
        prevent_initial_call=True,
    )(set_all_lots)

    app.callback(
        Output("dissolve-prices-wrap", "style"),
        Output("dissolve-gross-wrap", "style"),
        Input("dissolve-mode", "value"),
    )(toggle_mode)

    app.callback(
        Output("dissolve-preview", "children"),
        Output("btn-dissolve-confirm", "disabled"),
        Input("dissolve-select", "value"),
        Input({"type": "dissolve-lots", "index": ALL}, "value"),
        Input({"type": "dissolve-lots", "index": ALL}, "id"),
        Input({"type": "dissolve-price", "index": ALL}, "value"),
        Input({"type": "dissolve-price", "index": ALL}, "id"),
        Input("dissolve-mode", "value"),
        Input("dissolve-gross", "value"),
        Input("dissolve-notes", "value"),
    )(preview)

    app.callback(
        Output("modal-dissolve", "is_open", allow_duplicate=True),
        Output("structures-active-grid", "rowData", allow_duplicate=True),
        Output("detail-toast", "is_open", allow_duplicate=True),
        Output("detail-toast", "children", allow_duplicate=True),
        Output("detail-toast", "icon", allow_duplicate=True),
        Output("detail-toast", "header", allow_duplicate=True),
        Input("btn-dissolve-confirm", "n_clicks"),
        *_FORM_STATES,
        *_GRID_STATES,
        prevent_initial_call=True,
    )(execute)
