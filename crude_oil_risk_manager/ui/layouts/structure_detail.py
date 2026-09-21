"""Structure detail view: header, PnL summary, legs, net outright equivalent, trades, entry and exit forms.

`structure_detail_layout` is a pure function of the data it is given (no database
or adapter access); ui/callbacks/structure_detail_callbacks.py loads the data and
renders it into the detail modal body. Every form component is always present
(hidden when it does not apply) so the callbacks that target it never miss.
"""

from datetime import datetime

import dash_bootstrap_components as dbc
from dash import html

from core.models import PnLRecord, Structure, StructureStatus, Trade, TradeEventType
from core.pnl import calculate_structure_realized_pnl, calculate_structure_unrealized_pnl
from core.structure_utils import net_outright_equivalent
from core.structure_view import prices_from_store, structure_entry_price, structure_live_price
from ui.layouts.shell import COLORS

_HIDDEN = {"display": "none"}
_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_TEXT = {"color": COLORS["TEXT_PRIMARY"]}

EMPTY = "—"
EXIT_CONFIRM_TITLE = "Confirm Full Exit"

_EVENT_COLORS = {
    TradeEventType.TRADE: COLORS["ACCENT_GREEN"],
    TradeEventType.ADD: COLORS["ACCENT_GREEN"],
    TradeEventType.FULL_EXIT: COLORS["ACCENT_RED"],
    TradeEventType.PARTIAL_EXIT: COLORS["ACCENT_YELLOW"],
    TradeEventType.ROLL: COLORS["ACCENT_YELLOW"],
}


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------


def format_pnl(value: float | None) -> str:
    if value is None:
        return EMPTY
    return f"+${value:,.0f}" if value >= 0 else f"-${abs(value):,.0f}"


def pnl_color(value: float | None) -> str:
    if value is None or value == 0:
        return COLORS["TEXT_PRIMARY"]
    return COLORS["ACCENT_GREEN"] if value > 0 else COLORS["ACCENT_RED"]


def pnl_span(value: float | None, **style) -> html.Span:
    return html.Span(format_pnl(value), style={"color": pnl_color(value), **style})


def format_price(value: float | None) -> str:
    return EMPTY if value is None else f"{value:,.2f}"


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------


def _status_badge(status: StructureStatus) -> dbc.Badge:
    label = "OPEN" if status in (StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED) else status.value.upper()
    background = {"OPEN": COLORS["ACCENT_GREEN"], "SHELL": COLORS["TEXT_SECONDARY"], "CLOSED": COLORS["BORDER_COLOR"]}[label]
    text_color = COLORS["DARK_BG"] if label == "OPEN" else COLORS["TEXT_PRIMARY"]
    return dbc.Badge(label, style={"backgroundColor": background, "color": text_color, "fontSize": "14px"}, className="ms-2")


def _days_held(structure: Structure) -> str:
    if structure.status == StructureStatus.SHELL:
        return f"Days held: {EMPTY}"
    end = structure.closed_at or datetime.now(structure.created_at.tzinfo)
    return f"Days held: {(end - structure.created_at).days}"


def _header(structure: Structure, has_trades: bool) -> html.Div:
    edit_label = "🔒 Edit" if has_trades else "✏️ Edit"
    edit = [
        dbc.Button(
            edit_label, id="btn-edit-structure", color="secondary", outline=True, size="sm",
            style={"float": "right"}, disabled=structure.status == StructureStatus.CLOSED,
        )
    ]
    if has_trades:
        edit.append(
            dbc.Tooltip(
                "Structure has trades — changes will be confirmed and logged",
                target="btn-edit-structure",
            )
        )
    products = [
        dbc.Badge(p, pill=True, className="me-1", style={"backgroundColor": COLORS["ACCENT_BLUE"], "color": COLORS["TEXT_PRIMARY"]})
        for p in sorted({leg.contract.product for leg in structure.legs})
    ]
    return html.Div(
        [
            *edit,
            html.Div(
                [html.Span(structure.name, style={"fontSize": "26px", "fontWeight": "bold", **_TEXT}), _status_badge(structure.status)]
            ),
            html.Div([*products, html.Span(_days_held(structure), style={**_MUTED, "marginLeft": "8px"})], style={"marginTop": "6px"}),
        ],
        style={"marginBottom": "20px", "overflow": "hidden"},
    )


def _reuse_panel() -> html.Div:
    """Closed structures only: reuse the legs as a fresh shell (optionally under a new name)."""
    return html.Div(
        [
            dbc.Button("♻️ Reuse as Shell", id="btn-reuse-structure", color="success", outline=True, size="sm"),
            dbc.Collapse(
                html.Div(
                    [
                        dbc.Input(id="reuse-structure-name", placeholder="New name (optional)", maxLength=100, className="mb-2"),
                        dbc.Button("Confirm Reuse", id="btn-confirm-reuse", color="success"),
                    ],
                    style={"maxWidth": "420px", "marginTop": "10px"},
                ),
                id="reuse-collapse",
                is_open=False,
            ),
        ],
        style={"marginBottom": "20px"},
    )


def _metric_card(label: str, body, big: bool = False) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(label, style={**_MUTED, "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                    html.Div(body, style={"fontSize": "28px" if big else "20px", "fontWeight": "bold", **_TEXT}),
                ]
            ),
            style={"backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}", "height": "100%"},
        ),
        md=3,
        className="mb-3",
    )


def _price_move(entry: float | None, live: float | None, total: float | None) -> list:
    if entry is None or live is None:
        return [EMPTY]
    arrow = "▲" if live > entry else "▼" if live < entry else "▬"
    return [f"{format_price(entry)} → {format_price(live)} ", html.Span(arrow, style={"color": pnl_color(total)})]


def _pnl_summary(unrealized, realized, entry, live, note: str | None) -> html.Div:
    total = None if unrealized is None else unrealized + realized
    cards = dbc.Row(
        [
            _metric_card("Unrealized PnL", pnl_span(unrealized)),
            _metric_card("Realized PnL", pnl_span(realized)),
            _metric_card("Total PnL", pnl_span(total), big=True),
            _metric_card("Entry Price / Live Price", _price_move(entry, live, total)),
        ]
    )
    return html.Div([cards, html.Div(note, style={**_MUTED, "marginBottom": "12px"})] if note else [cards])


def _table(headers: list[str], rows: list[list], footer: list | None = None) -> dbc.Table:
    head = html.Thead(html.Tr([html.Th(h) for h in headers]))
    body = html.Tbody([html.Tr([html.Td(c) for c in row]) for row in rows] + ([html.Tr(footer)] if footer else []))
    return dbc.Table([head, body], striped=True, color="dark", hover=False, size="sm", className="align-middle")


def _legs_table(structure: Structure, prices: dict[str, float], breakdown: dict[str, float], total, edit_mode: bool):
    rows = []
    for number, leg in enumerate(structure.legs, start=1):
        if edit_mode:
            symbol = dbc.Input(id={"type": "edit-leg-symbol", "index": number - 1}, value=leg.contract.symbol, debounce=True, size="sm")
            ratio = dbc.Input(id={"type": "edit-leg-ratio", "index": number - 1}, type="number", step=1, value=leg.ratio, size="sm")
        else:
            symbol, ratio = leg.contract.symbol, f"{leg.ratio:+d}"
        entry = leg.average_entry_price if leg.average_entry_price is not None else leg.entry_price
        rows.append(
            [
                str(number), symbol, ratio, f"{leg.lots:g}", format_price(entry),
                format_price(prices.get(leg.contract.symbol)),
                pnl_span(breakdown.get(leg.leg_id)) if leg.leg_id in breakdown else EMPTY,
                "Long" if (leg.ratio > 0) == (leg.direction == "buy") else "Short",
            ]
        )
    footer = [html.Td("Total Structure PnL:", colSpan=6, style={"textAlign": "right", "fontWeight": "bold"}), html.Td(pnl_span(total, fontWeight="bold"), colSpan=2)]
    parts = [
        html.H6("Legs", style=_TEXT),
        _table(["Leg #", "Symbol", "Ratio", "Lots", "Entry Price", "Live Price", "Leg PnL", "Direction"], rows, footer),
    ]
    if edit_mode:
        parts.append(
            html.Div(
                [
                    dbc.Button("💾 Save Changes", id="btn-save-edit", color="success", size="sm", className="me-2"),
                    dbc.Button("Cancel Edit", id="btn-cancel-edit", color="secondary", outline=True, size="sm"),
                    html.Span("  Only symbols and ratios can be edited; lots and entry prices are kept.", style=_MUTED),
                ],
                className="mb-3",
            )
        )
    return html.Div(parts)


def _net_outrights(structure: Structure) -> html.Details:
    lots = structure.legs[0].lots
    per = f"{lots:g} lots" if lots > 0 else "1 lot"
    net, ignored = net_outright_equivalent([{"symbol": l.contract.symbol, "ratio": l.ratio} for l in structure.legs], lots or 1.0)
    rows = [
        [symbol, "Long" if qty > 0 else "Short" if qty < 0 else "Flat", html.Span(f"{qty:+g}", style={"color": pnl_color(qty)})]
        for symbol, qty in sorted(net.items())
    ]
    body = [
        html.Div(
            "This is a mathematical decomposition based on leg ratios. The structure is exchange-quoted.",
            style={**_MUTED, "margin": "8px 0"},
        ),
        _table(["Contract", "Direction", f"Net Lots ({per})"], rows) if rows else html.Div("No decomposable legs.", style=_MUTED),
    ]
    if ignored:
        body.append(html.Div(f"Ignored: {', '.join(ignored)}", style=_MUTED))
    return html.Details(
        [html.Summary("📊 Net Outright Equivalent (informational)", style={"cursor": "pointer", **_TEXT}), *body],
        style={"marginBottom": "20px"},
    )


def _trade_history(trades: list[Trade]) -> html.Div:
    rows = []
    for trade in sorted(trades, key=lambda t: t.timestamp, reverse=True):
        color = _EVENT_COLORS[trade.event_type]
        rows.append(
            [
                trade.timestamp.strftime("%Y-%m-%d %H:%M"),
                dbc.Badge(trade.event_type.value.replace("_", " ").upper(), style={"backgroundColor": "transparent", "border": f"1px solid {color}", "color": color}),
                format_price(trade.price),
                f"{trade.lots:g}",
                pnl_span(trade.realized_pnl) if trade.realized_pnl is not None else EMPTY,
                trade.notes or EMPTY,
            ]
        )
    return html.Div(
        [html.H6("Trade History", style=_TEXT), _table(["Date/Time", "Event", "Price", "Lots", "Realized PnL", "Notes"], rows)],
        style={"marginBottom": "20px"},
    )


def _live_label(live: float | None, component_id: str) -> html.Span:
    return html.Span(f"Live: {format_price(live)}", id=component_id, style={**_MUTED, "marginLeft": "12px"})


def _field(label: str, *controls) -> html.Div:
    return html.Div([html.Label(label, style={**_MUTED, "display": "block", "marginBottom": "4px"}), *controls], className="mb-3")


def _entry_form(structure: Structure, live: float | None) -> html.Div:
    is_open = structure.status != StructureStatus.SHELL
    title = "Add to Position" if is_open else "Enter Trade"
    form = html.Div(
        [
            html.H5(title, style=_TEXT),
            _field(
                "Structure Price",
                dbc.Input(id="trade-entry-price", type="number", placeholder="Exchange-quoted price", style={"display": "inline-block", "width": "220px"}),
                dbc.Button("Use Live Price", id="btn-use-live-price", size="sm", color="secondary", className="ms-2"),
                _live_label(live, "trade-live-price-label"),
            ),
            _field("Lots", dbc.Input(id="trade-entry-lots", type="number", placeholder="Number of lots", min=0.01, step=0.01, style={"width": "220px"})),
            html.H6("🎯 Price Alerts (Optional)", style=_TEXT),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Input(id="trade-stop-loss-price", type="number", placeholder="Stop loss price", step=0.01),
                            html.Div("Alert triggered when live price crosses this level", style=_MUTED),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Input(id="trade-target-price", type="number", placeholder="Target price", step=0.01),
                            html.Div("Alert triggered when live price reaches this level", style=_MUTED),
                        ],
                        md=4,
                    ),
                ],
                className="mb-1",
            ),
            html.Div(f"Current live price: {format_price(live)}", id="trade-alert-live-reference", style={**_MUTED, "marginBottom": "16px"}),
            _field(
                "Direction (for the trade log; the leg ratios already encode direction)",
                dbc.RadioItems(id="trade-direction", options=[{"label": "Buy", "value": "buy"}, {"label": "Sell", "value": "sell"}], value="buy", inline=True),
            ),
            html.Div(id="trade-pnl-preview", style={**_MUTED, "marginBottom": "12px"}),
            _field("Notes", dbc.Textarea(id="trade-entry-notes", rows=2, placeholder="Optional notes", maxLength=200)),
            dbc.Button("✅ Confirm Trade Entry", id="btn-confirm-trade", color="success", size="lg"),
        ]
    )
    toggle = (
        dbc.Button("+ Add Trade", id="btn-toggle-add-trade", color="secondary", outline=True, size="sm", className="mb-2")
        if structure.status in (StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED)
        else None
    )
    visible = structure.status != StructureStatus.CLOSED
    return html.Div(
        [toggle, dbc.Collapse(form, id="trade-entry-collapse", is_open=structure.status == StructureStatus.SHELL)],
        id="trade-entry-form",
        style={"marginBottom": "20px"} if visible else _HIDDEN,
    )


def _exit_form(structure: Structure, live: float | None) -> html.Div:
    is_open = structure.status in (StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED)
    return html.Div(
        [
            html.Hr(style={"borderColor": COLORS["BORDER_COLOR"]}),
            html.H5("Exit Structure", style={"color": COLORS["ACCENT_YELLOW"]}),
            _field(
                "Exit Price",
                dbc.Input(id="trade-exit-price", type="number", placeholder="Exchange-quoted exit price", style={"display": "inline-block", "width": "220px"}),
                dbc.Button("Use Live Price", id="btn-use-live-exit-price", size="sm", color="secondary", className="ms-2"),
                _live_label(live, "exit-live-price-label"),
            ),
            _field(
                "Lots to Exit",
                dbc.Input(id="trade-exit-lots", type="number", placeholder="Lots to exit", style={"display": "inline-block", "width": "220px"}),
                dbc.Button("Exit All", id="btn-exit-all-lots", size="sm", color="warning", className="ms-2"),
            ),
            html.Div(id="exit-pnl-preview", style={**_MUTED, "marginBottom": "12px"}),
            _field("Notes", dbc.Textarea(id="trade-exit-notes", rows=2, maxLength=200)),
            dbc.Button("🚪 Confirm Full Exit", id="btn-confirm-exit", color="danger", size="lg"),
        ],
        id="trade-exit-form",
        style={} if is_open else _HIDDEN,
    )


# ----------------------------------------------------------------------
# Public layout
# ----------------------------------------------------------------------


def structure_detail_layout(
    structure: Structure,
    live_prices: dict | None,
    trades: list[Trade],
    pnl_record: PnLRecord | None,
    edit_mode: bool = False,
) -> html.Div:
    """Content of the structure detail modal. Pure: everything it shows comes from the arguments."""
    prices = prices_from_store(live_prices)
    is_open = structure.status in (StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED)
    unrealized, breakdown, missing = calculate_structure_unrealized_pnl(structure.legs, prices)
    realized = calculate_structure_realized_pnl(trades)
    entry, live = structure_entry_price(structure), structure_live_price(structure, prices)

    summary_note = None
    if is_open and missing:
        # Some live prices are missing: fall back to the last saved snapshot instead of a partial number.
        if pnl_record is not None:
            unrealized, realized = pnl_record.unrealized_pnl, pnl_record.realized_pnl
            breakdown = {}
            summary_note = f"Live prices missing for {', '.join(missing)}; showing the PnL saved at {pnl_record.timestamp:%H:%M:%S} UTC."
        else:
            unrealized, summary_note = None, f"Live prices missing for {', '.join(missing)}; PnL unavailable."
    total = None if unrealized is None else unrealized + realized

    sections = [_header(structure, bool(trades))]
    if structure.status == StructureStatus.CLOSED:
        sections.append(_reuse_panel())
    if is_open:
        sections.append(_pnl_summary(unrealized, realized, entry, live, summary_note))
    elif structure.status == StructureStatus.CLOSED:
        total = realized
    sections.append(_legs_table(structure, prices, breakdown, total, edit_mode))
    sections.append(_net_outrights(structure))
    if trades:
        sections.append(_trade_history(trades))
    sections.append(_entry_form(structure, live))
    sections.append(_exit_form(structure, live))
    return html.Div(sections)


# ----------------------------------------------------------------------
# Confirmation modals and toast (part of the Structures page layout)
# ----------------------------------------------------------------------


def confirm_exit_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(EXIT_CONFIRM_TITLE), close_button=False),
            dbc.ModalBody(id="modal-confirm-exit-body"),
            dbc.ModalFooter(
                [
                    dbc.Button("Cancel", id="btn-cancel-exit", color="secondary", outline=True),
                    dbc.Button("Confirm Exit", id="btn-confirm-exit-final", color="danger"),
                ]
            ),
        ],
        id="modal-confirm-exit",
        centered=True,
        backdrop="static",
        is_open=False,
    )


def confirm_edit_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Edit Structure"), close_button=False),
            dbc.ModalBody(
                [
                    html.Div("⚠️ This structure has active trades.", style={"fontWeight": "bold", "color": COLORS["ACCENT_YELLOW"]}),
                    html.Div("Changes will be logged in the audit trail. Are you sure you want to edit?"),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button("Cancel", id="btn-cancel-edit-confirm", color="secondary", outline=True),
                    dbc.Button("Proceed with Edit", id="btn-confirm-edit-proceed", color="warning"),
                ]
            ),
        ],
        id="modal-confirm-edit",
        centered=True,
        backdrop="static",
        is_open=False,
    )


def detail_toast() -> dbc.Toast:
    """Feedback for trade entry, exit and edits (own toast: the alert container is rewritten every 5s)."""
    return dbc.Toast(
        "", id="detail-toast", header="", icon="success", is_open=False, dismissable=True, duration=6000,
        style={"position": "fixed", "bottom": "20px", "right": "20px", "width": "380px", "zIndex": 2200},
    )
