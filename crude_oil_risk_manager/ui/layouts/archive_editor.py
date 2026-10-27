"""Archive: the editor for every column of a closed trade (the 'Edit this trade' panel of the detail card).

Four tabs, each saved on its own: Structure (name, type, notes), Trades (times, prices, lots, costs, P&L, stops and targets: they
drive the entry / exit date, prices, lots, costs, P&L, result, risk:reward and R columns), Legs (symbols, ratios) and Market
(the entry / exit snapshot cells and MFE / MAE). Ideas, payups, conviction, exit reason and mistakes are edited in the
journal editor above it. The layout is static; ui/callbacks/archive_callbacks.py fills and saves it, core/archive_edit.py does the work.
"""

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from core.archive import SNAPSHOT_FIELDS
from core.models.structure import CHOOSABLE_STRUCTURE_TYPES, StructureType
from ui.layouts.shell import COLORS

SNAPSHOT_LABELS = {
    "px_co": "CO price", "px_cl": "CL price", "px_go": "GO price", "move_day": "Day move", "move_min": "1-min move",
    "curve_cl": "CL curve", "curve_brn": "BRN curve", "curve_go": "GO curve", "atr": "ATR (x100)", "atr_basis": "ATR basis",
    "vol_ratio": "Volatility vs normal", "volume_ratio": "Volume vs normal", "spread_ticks": "Bid-ask (ticks)",
    "brent_wti": "Brent-WTI", "crack": "Gasoil crack", "kink_z": "Curve z-score",
}
TYPE_CHOICES = [{"label": kind.label, "value": kind.value} for kind in (*CHOOSABLE_STRUCTURE_TYPES, StructureType.CUSTOM)]

_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_LABEL = {**_MUTED, "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "6px"}
_CARD = {"backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}"}

TRADE_COLUMNS = [
    {"id": "event", "name": "Event", "editable": False},
    {"id": "direction", "name": "Buy / sell", "editable": True, "presentation": "dropdown"},
    {"id": "timestamp", "name": "Date and time (UTC)", "editable": True},
    {"id": "price", "name": "Price", "type": "numeric", "editable": True},
    {"id": "lots", "name": "Lots", "type": "numeric", "editable": True},
    {"id": "transaction_cost", "name": "Transaction cost ($)", "type": "numeric", "editable": True},
    {"id": "realized_pnl", "name": "Realized P&L, gross ($)", "type": "numeric", "editable": True},
    {"id": "stop_loss_price", "name": "Stop", "type": "numeric", "editable": True},
    {"id": "target_price", "name": "Target", "type": "numeric", "editable": True},
]
LEG_COLUMNS = [
    {"id": "leg", "name": "Leg", "editable": False},
    {"id": "symbol", "name": "Symbol", "editable": True},
    {"id": "ratio", "name": "Ratio", "type": "numeric", "editable": True},
]
MARKET_COLUMNS = [
    {"id": "field", "name": "Market", "editable": False},
    {"id": "entry", "name": "At entry", "editable": True},
    {"id": "exit", "name": "At exit", "editable": True},
]
EXCURSION_INPUTS = [
    ("archive-edit-mfe-i", "MFE intraday ($, best open P&L)"), ("archive-edit-mae-i", "MAE intraday ($, worst, negative)"),
    ("archive-edit-mfe-d", "MFE daily ($)"), ("archive-edit-mae-d", "MAE daily ($)"),
]


def _grid(component_id: str, columns: list[dict], **kwargs) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=component_id, columns=columns, data=[], editable=False, style_table={"overflowX": "auto"},
        style_header={"backgroundColor": COLORS["SIDEBAR_BG"], "color": COLORS["TEXT_SECONDARY"], "border": f"1px solid {COLORS['BORDER_COLOR']}", "fontWeight": "bold"},
        style_cell={"backgroundColor": COLORS["CARD_BG"], "color": COLORS["TEXT_PRIMARY"], "border": f"1px solid {COLORS['BORDER_COLOR']}", "padding": "6px 10px", "textAlign": "left"},
        style_data_conditional=[{"if": {"column_editable": True}, "backgroundColor": COLORS["SIDEBAR_BG"]}],
        **kwargs,
    )


def _status(component_id: str) -> html.Div:
    return html.Div(id=component_id, style={"fontSize": "13px", "minHeight": "20px", "marginTop": "8px", "whiteSpace": "pre-wrap"})


def _structure_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            dbc.Col([html.Div("Structure name", style=_LABEL), dbc.Input(id="archive-edit-name", maxLength=100)], md=6),
            dbc.Col([html.Div("Structure type", style=_LABEL), dcc.Dropdown(id="archive-edit-type", options=TYPE_CHOICES, clearable=False)], md=3),
        ], className="mb-3"),
        html.Div("Notes (the audit lines of past edits are part of the notes)", style=_LABEL),
        dbc.Textarea(id="archive-edit-notes", rows=5, className="mb-3"),
        dbc.Button("Save structure", id="archive-edit-structure-save", color="primary", size="sm"),
        _status("archive-edit-structure-status"),
    ], className="pt-3")


def _trades_tab() -> html.Div:
    return html.Div([
        html.Div(
            "One row per recorded trade. Click a cell to change it: the entry and exit dates, prices, lots, transaction cost, realized "
            "P&L, result, risk:reward and R columns are all worked out from these rows. Times are UTC (YYYY-MM-DD HH:MM).",
            style={**_MUTED, "marginBottom": "8px"},
        ),
        _grid("archive-edit-trades", TRADE_COLUMNS, dropdown={"direction": {"options": [{"label": "buy", "value": "buy"}, {"label": "sell", "value": "sell"}]}}),
        dbc.Checkbox(
            id="archive-edit-recalc", value=True, className="mt-3",
            label="Recalculate the leg entry prices and each exit's P&L and cost when a price, lots or buy/sell is changed (a P&L or cost typed here is kept as typed)",
        ),
        dbc.Button("Save trades", id="archive-edit-trades-save", color="primary", size="sm", className="mt-2"),
        _status("archive-edit-trades-status"),
    ], className="pt-3")


def _legs_tab() -> html.Div:
    return html.Div([
        html.Div("Change a leg's contract or ratio. The number of legs stays; the entry prices follow the Trades tab.", style={**_MUTED, "marginBottom": "8px"}),
        _grid("archive-edit-legs", LEG_COLUMNS),
        dbc.Button("Save legs", id="archive-edit-legs-save", color="primary", size="sm", className="mt-3"),
        _status("archive-edit-legs-status"),
    ], className="pt-3")


def _market_tab() -> html.Div:
    return html.Div([
        html.Div(
            "The market at entry and at exit. Prices and ratios are numbers; moves are Bullish, Bearish or Stalemate; curves are "
            "Contango, Backwardation, Contango - Backwardation, Backwardation - Contango or Flat. Leave a cell empty for none.",
            style={**_MUTED, "marginBottom": "8px"},
        ),
        _grid("archive-edit-market", MARKET_COLUMNS),
        html.Div("Best and worst open P&L (a typed value replaces the computed one)", style={**_LABEL, "marginTop": "14px"}),
        dbc.Row([
            dbc.Col([html.Div(label, style=_MUTED), dbc.Input(id=component_id, type="number", step="any", size="sm")], md=3)
            for component_id, label in EXCURSION_INPUTS
        ]),
        dbc.Button("Save market data", id="archive-edit-market-save", color="primary", size="sm", className="mt-3"),
        _status("archive-edit-market-status"),
    ], className="pt-3")


def trade_editor() -> html.Div:
    """The collapsed 'Edit this trade' panel (all components exist from the start so every callback input is always present)."""
    return html.Div([
        dbc.Button("✏️ Edit this trade (every column)", id="archive-edit-toggle", color="warning", outline=True, size="sm", className="mt-2 mb-2"),
        dbc.Collapse(
            dbc.Card(dbc.CardBody(dbc.Tabs([
                dbc.Tab(_structure_tab(), label="Structure", tab_id="edit-structure"),
                dbc.Tab(_trades_tab(), label="Trades", tab_id="edit-trades"),
                dbc.Tab(_legs_tab(), label="Legs", tab_id="edit-legs"),
                dbc.Tab(_market_tab(), label="Market data", tab_id="edit-market"),
            ], id="archive-edit-tabs", active_tab="edit-structure")), style=_CARD),
            id="archive-edit-collapse", is_open=False,
        ),
    ], id="archive-trade-editor")


MARKET_FIELDS = [(column, SNAPSHOT_LABELS[column]) for column, _key, _sub in SNAPSHOT_FIELDS]
