"""Archive tab: filters, summary stats and the table of closed structures (read-only).

The layout is static; ui/callbacks/archive_callbacks.py fills it. All colours come
from COLORS. The only write is deleting closed structures the user ticked and confirmed.
"""

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
from dash.dash_table.Format import Format, Group, Scheme, Symbol

from core.archive import RESULT_LOSS, RESULT_WIN, SNAPSHOT_FIELDS, display_atr
from core.journal import CONVICTIONS, EXIT_REASONS, MISTAKES
from core.models.structure import CHOOSABLE_STRUCTURE_TYPES
from ui.layouts.archive_analysis import analysis_tab
from ui.layouts.archive_editor import SNAPSHOT_LABELS, trade_editor
from ui.layouts.shell import COLORS

EMPTY = "—"
NO_ARCHIVE_TEXT = "No closed structures in archive"
NO_MATCH_TEXT = "No structures match the current filters"
PAGE_SIZE = 20

TYPE_OPTIONS = [
    {"label": "All", "value": "all"},
    {"label": "Outright", "value": "outright"},
    {"label": "Spread", "value": "spread"},
    {"label": "Fly", "value": "fly"},
    *({"label": kind.label, "value": kind.value} for kind in CHOOSABLE_STRUCTURE_TYPES if kind.value not in ("outright", "spread", "fly")),
]
PNL_OPTIONS = [
    {"label": "All", "value": "all"},
    {"label": "Winners only", "value": "winners"},
    {"label": "Losers only", "value": "losers"},
]

CARD_STYLE = {"backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}"}
_HEADING = {"color": COLORS["TEXT_PRIMARY"], "marginBottom": "12px"}
_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_LABEL = {**_MUTED, "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "6px"}
_VALUE = {"fontSize": "24px", "fontWeight": "bold", "color": COLORS["TEXT_PRIMARY"]}

# (id suffix, label, accent colour)
STAT_CARDS = [
    ("total", "Total Trades", COLORS["ACCENT_BLUE"]),
    ("winners", "Winners", COLORS["ACCENT_GREEN"]),
    ("losers", "Losers", COLORS["ACCENT_RED"]),
    ("win-rate", "Win Rate", COLORS["ACCENT_YELLOW"]),
    ("total-pnl", "Total PnL ($)", COLORS["ACCENT_GREEN"]),
    ("avg-pnl", "Avg PnL ($)", COLORS["ACCENT_BLUE"]),
    ("best", "Best Trade ($)", COLORS["ACCENT_GREEN"]),
    ("worst", "Worst Trade ($)", COLORS["ACCENT_RED"]),
]


# ----------------------------------------------------------------------
# Formatting builders
# ----------------------------------------------------------------------


def money(value: float) -> str:
    """'$1,234.50' / '-$1,234.50' with commas and two decimals."""
    return f"-${abs(value):,.2f}" if value < 0 else f"${value:,.2f}"


def _colored(value: float | None, good: bool | None = None) -> dict:
    """Value style: green when good is True, red when False, neutral when None."""
    color = COLORS["TEXT_PRIMARY"] if good is None else COLORS["ACCENT_GREEN"] if good else COLORS["ACCENT_RED"]
    return {**_VALUE, "color": color}


def build_stats(summary: dict) -> dict[str, tuple[str, dict]]:
    """{card suffix: (text, style)}. Total PnL is green/red by sign, Win Rate by the 50% line."""
    total = summary["total"]
    pnl = summary["total_pnl"]
    win_rate_good = None if summary["winners"] + summary["losers"] == 0 else summary["win_rate"] >= 50
    return {
        "total": (str(total), _colored(None)),
        "winners": (str(summary["winners"]), _colored(None)),
        "losers": (str(summary["losers"]), _colored(None)),
        "win-rate": (f"{summary['win_rate']:.1f}%", _colored(None, win_rate_good)),
        "total-pnl": (money(pnl), _colored(None, None if pnl == 0 else pnl > 0)),
        "avg-pnl": (money(summary["avg_pnl"]), _colored(None)),
        "best": (money(summary["best"]), _colored(None)),
        "worst": (money(summary["worst"]), _colored(None)),
    }


def _or_dash(value, fmt=None):
    if value is None:
        return EMPTY
    return fmt(value) if fmt else value


def _journal_cells(row: dict) -> dict:
    """The trade-journal cells of a table row (a dash where the journal has nothing)."""
    cells = {
        "rr": row.get("rr_text", EMPTY),
        "ideas": row.get("ideas_text") or EMPTY,
        "payup_entry": _or_dash(row.get("payup_entry"), lambda v: f"{v:g}"),
        "payup_exit": _or_dash(row.get("payup_exit"), lambda v: f"{v:g}"),
        "conviction": _or_dash(row.get("conviction")),
        "exit_reason": row.get("exit_reason") or EMPTY,
        "mistakes": row.get("mistakes_text") or EMPTY,
        "risk_usd": _or_dash(row.get("risk_usd"), lambda v: round(v)),
        "r_multiple": _or_dash(row.get("r_multiple"), lambda v: round(v, 2)),
        "mfe_i": _or_dash(row.get("mfe_i"), lambda v: round(v)),
        "mae_i": _or_dash(row.get("mae_i"), lambda v: round(v)),
        "mfe_d": _or_dash(row.get("mfe_d"), lambda v: round(v)),
        "mae_d": _or_dash(row.get("mae_d"), lambda v: round(v)),
    }
    for prefix in ("e_", "x_"):
        for column, _key, _sub in SNAPSHOT_FIELDS:
            value = row.get(prefix + column)
            value = display_atr(value) if column == "atr" else value
            cells[prefix + column] = EMPTY if value is None else round(value, 4) if isinstance(value, float) else value
    return cells


def build_table_rows(rows: list[dict]) -> list[dict]:
    """Scalar DataTable rows (the row `id` is what a click reports back)."""
    return [
        {
            "id": row["id"],  # one row per exit: a structure exited in two goes in two rows
            "name": row["name"],
            "type": row["type"],
            "legs": row["legs_summary"],
            "entry_date": _or_dash(row["entry_date"]),
            "exit_date": _or_dash(row["exit_date"]),
            "days_held": _or_dash(row["days_held"]),
            "entry_price": _or_dash(row["entry_price"], lambda v: round(v, 4)),
            "exit_price": _or_dash(row["exit_price"], lambda v: round(v, 4)),
            "lots": _or_dash(row["lots"], lambda v: f"{v:g}"),
            "open_lots": f"{row['open_lots']:g}" if row.get("open_lots") else EMPTY,
            "transaction_costs": row["transaction_costs"],
            "realized_pnl": row["realized_pnl"],
            "result": row["result"],
            **_journal_cells(row),
        }
        for row in rows
    ]


def journal_detail(row: dict) -> list:
    """The journal part of the detail panel: risk:reward, the market at entry and at exit, ideas and per-leg payups."""
    parts = [html.Div("Trade journal", style=_LABEL)]
    rr = row.get("rr_detail")
    if rr:
        parts.append(html.Div(
            f"Risk : Reward {row['rr_text']} — average entry {rr['entry']:,.2f}, average stop {rr['stop']:,.2f}, average target {rr['target']:,.2f}",
            style={**_MUTED, "marginBottom": "6px"},
        ))
    else:
        parts.append(html.Div("Risk : Reward — needs a stop and a target typed with the entry.", style={**_MUTED, "marginBottom": "6px"}))
    snaps = row.get("snapshots") or {}
    if snaps.get("entry") or snaps.get("exit"):
        head = html.Thead(html.Tr([html.Th("Market"), html.Th("At entry"), html.Th("At exit")]))
        body = html.Tbody([
            html.Tr([html.Td(_SNAPSHOT_LABELS[column]), html.Td(_cell(_shown(column, row.get("e_" + column)))), html.Td(_cell(_shown(column, row.get("x_" + column))))])
            for column, _key, _sub in SNAPSHOT_FIELDS
        ] + [html.Tr([html.Td("Taken at"), html.Td(_when(snaps.get("entry"))), html.Td(_when(snaps.get("exit")))])])
        parts.append(dbc.Table([head, body], color="dark", striped=True, size="sm"))
    else:
        parts.append(html.Div("No market snapshot yet: use 'Fill market data for old trades' above the table.", style={**_MUTED, "marginBottom": "6px"}))
    if row.get("mfe_i") is not None or row.get("mfe_d") is not None:
        def fmt(a, b):
            return "—" if a is None else f"best {a:+,.0f}, worst {b:+,.0f}"

        parts.append(html.Div(
            f"Open PnL at the size entered — intraday candles: {fmt(row.get('mfe_i'), row.get('mae_i'))}; daily closes: {fmt(row.get('mfe_d'), row.get('mae_d'))}",
            style={**_MUTED, "marginBottom": "6px"},
        ))
    legs = row.get("payup_legs") or {}
    leg_text = "; ".join(f"{kind}: " + ", ".join(f"{s} {v:g}" for s, v in legs.get(kind, {}).items()) for kind in ("entry", "exit") if legs.get(kind))
    if leg_text:
        parts.append(html.Div(f"Payups per leg (ticks) — {leg_text}", style={**_MUTED, "marginBottom": "6px"}))
    return parts


def _shown(column: str, value):
    return display_atr(value) if column == "atr" else value


def _cell(value) -> str:
    return EMPTY if value is None else f"{value:,.2f}" if isinstance(value, float) else str(value)


def _when(snapshot) -> str:
    return (snapshot or {}).get("at", EMPTY)[:16].replace("T", " ") if snapshot else EMPTY


def _exit_note(row: dict) -> list:
    """Which exit of the structure this row is, and how its entry was matched (first in, first out)."""
    if row.get("exits_of_structure", 1) <= 1 and not row.get("open_lots"):
        return []
    text = f"Exit {row['exit_number']} of {row['exits_of_structure']} of this structure: {row['lots']:g} lots, matched to the earliest entry lots (first in, first out)."
    if row.get("open_lots"):
        text += f" {row['open_lots']:g} lots are still open."
    return [html.Div(text, style={**_MUTED, "marginBottom": "8px"})]


def build_detail(row: dict | None) -> html.Div | None:
    """Expanded view of one closed structure: legs, prices, PnL, days held and (read-only) notes."""
    if row is None:
        return None
    leg_rows = [
        html.Tr([html.Td(leg["symbol"]), html.Td("Buy" if leg["side"] == "buy" else "Sell"), html.Td(f"{leg['lots']:g}")])
        for leg in row["legs"]
    ] or [html.Tr(html.Td(EMPTY, colSpan=3))]
    legs_table = dbc.Table(
        [html.Thead(html.Tr([html.Th("Symbol"), html.Th("Direction"), html.Th("Lots")])), html.Tbody(leg_rows)],
        color="dark", striped=True, size="sm",
    )
    pnl = row["realized_pnl"]
    pnl_color = COLORS["ACCENT_GREEN"] if pnl > 0 else COLORS["ACCENT_RED"] if pnl < 0 else COLORS["TEXT_PRIMARY"]

    def fact(label, value, style=None):
        return dbc.Col(
            [html.Div(label, style=_LABEL), html.Div(value, style={"fontWeight": "bold", "color": COLORS["TEXT_PRIMARY"], **(style or {})})],
            md=3, className="mb-3",
        )

    facts = dbc.Row(
        [
            fact("Entry Price", _or_dash(row["entry_price"], lambda v: f"{v:,.2f}")),
            fact("Exit Price", _or_dash(row["exit_price"], lambda v: f"{v:,.2f}")),
            fact("Transaction Cost", money(row["transaction_costs"])),
            fact("Realized PnL (net of TC)", money(pnl), {"color": pnl_color}),
            fact("Days Held", _or_dash(row["days_held"])),
        ]
    )
    parts = [
        html.H5(f"{row['name']} — {row['type']}", style=_HEADING),
        *_exit_note(row),
        html.Div("Legs", style=_LABEL),
        legs_table,
        facts,
        *journal_detail(row),
    ]
    if row["notes"].strip():
        parts += [
            html.Div("Notes (read-only)", style=_LABEL),
            html.Pre(row["notes"].strip(), style={"whiteSpace": "pre-wrap", "color": COLORS["TEXT_PRIMARY"], "backgroundColor": COLORS["SIDEBAR_BG"], "padding": "10px", "borderRadius": "4px"}),
        ]
    return html.Div(parts)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _filter_bar() -> dbc.Card:
    def field(label, component, **columns):
        return dbc.Col([html.Div(label, style=_LABEL), component], className="mb-2", **columns)

    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    field("Closed Between", dcc.DatePickerRange(id="archive-dates", clearable=True, display_format="YYYY-MM-DD"), md=3),
                    field("Structure type", dbc.Select(id="archive-type", options=TYPE_OPTIONS, value="all"), md=2),
                    field("Symbol contains", dbc.Input(id="archive-symbol", placeholder="e.g. CLZ"), md=3),
                    field(
                        "PnL",
                        dbc.RadioItems(
                            id="archive-pnl", options=PNL_OPTIONS, value="all", className="btn-group flex-nowrap",
                            inputClassName="btn-check", labelClassName="btn btn-outline-secondary btn-sm", labelCheckedClassName="active",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Button("Apply Filters", id="archive-apply-btn", color="secondary", size="sm", className="me-2"),
                            dbc.Button("Clear Filters", id="archive-clear-btn", color="secondary", outline=True, size="sm"),
                        ],
                        md=12, className="mb-2",
                    ),
                ],
                className="g-3",
            )
        ),
        style=CARD_STYLE,
        className="mb-3",
    )


def _stats_row() -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([html.Div(label, style=_LABEL), html.Div(EMPTY, id=f"archive-stat-{suffix}", style=_VALUE)]),
                    style={**CARD_STYLE, "borderTop": f"3px solid {accent}", "height": "100%"},
                ),
                md=6, xl=3, className="mb-3",
            )
            for suffix, label, accent in STAT_CARDS
        ]
    )


_SNAPSHOT_LABELS = SNAPSHOT_LABELS
# entry columns first, then the same for the exit
JOURNAL_COLUMNS = [
    {"id": prefix + column, "name": f"{'Entry' if prefix == 'e_' else 'Exit'} {_SNAPSHOT_LABELS[column]}", "type": "any"}
    for prefix in ("e_", "x_")
    for column, _key, _sub in SNAPSHOT_FIELDS
]

TABLE_COLUMNS = [
    {"id": "name", "name": "Structure Name"},
    {"id": "type", "name": "Type"},
    {"id": "legs", "name": "Legs"},
    {"id": "entry_date", "name": "Entry Date"},
    {"id": "exit_date", "name": "Exit Date"},
    {"id": "days_held", "name": "Days Held", "type": "any"},
    {"id": "entry_price", "name": "Entry Price (pts)", "type": "any"},
    {"id": "exit_price", "name": "Exit Price (pts)", "type": "any"},
    {"id": "lots", "name": "Lots", "type": "any"},
    {"id": "open_lots", "name": "Lots Still Open", "type": "any"},
    {
        "id": "transaction_costs", "name": "Transaction Cost ($)", "type": "numeric",
        "format": Format(precision=2, scheme=Scheme.fixed, group=Group.yes, symbol=Symbol.yes, symbol_prefix="$"),
    },
    {
        "id": "realized_pnl", "name": "Realized PnL ($, net of TC)", "type": "numeric",
        "format": Format(precision=2, scheme=Scheme.fixed, group=Group.yes, symbol=Symbol.yes, symbol_prefix="$"),
    },
    {"id": "result", "name": "Result"},
    {"id": "rr", "name": "Risk : Reward"},
    {"id": "ideas", "name": "Trade Ideas"},
    {"id": "payup_entry", "name": "Payups in Entry", "type": "any"},
    {"id": "payup_exit", "name": "Payups in Exit", "type": "any"},
    {"id": "conviction", "name": "Conviction", "type": "any"},
    {"id": "exit_reason", "name": "Exit Reason"},
    {"id": "mistakes", "name": "Mistakes"},
    {"id": "risk_usd", "name": "Risk at Stop ($)", "type": "any"},
    {"id": "r_multiple", "name": "Result in R", "type": "any"},
    {"id": "mfe_i", "name": "MFE Intraday ($)", "type": "any"},
    {"id": "mae_i", "name": "MAE Intraday ($)", "type": "any"},
    {"id": "mfe_d", "name": "MFE Daily ($)", "type": "any"},
    {"id": "mae_d", "name": "MAE Daily ($)", "type": "any"},
    *JOURNAL_COLUMNS,
]


def _table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="archive-table",
        columns=TABLE_COLUMNS,
        data=[],
        row_selectable="multi",
        selected_rows=[],
        page_size=PAGE_SIZE,
        page_action="native",
        sort_action="native",
        sort_by=[{"column_id": "exit_date", "direction": "desc"}],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": COLORS["SIDEBAR_BG"], "color": COLORS["TEXT_SECONDARY"],
            "border": f"1px solid {COLORS['BORDER_COLOR']}", "fontWeight": "bold",
        },
        style_cell={
            "backgroundColor": COLORS["CARD_BG"], "color": COLORS["TEXT_PRIMARY"],
            "border": f"1px solid {COLORS['BORDER_COLOR']}", "padding": "8px 12px", "textAlign": "right",
        },
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "left"} for c in ("name", "type", "legs", "entry_date", "exit_date", "result")
        ],
        style_data_conditional=[
            {"if": {"state": "active"}, "backgroundColor": COLORS["ACCENT_BLUE"], "border": f"1px solid {COLORS['ACCENT_GREEN']}"},
            {"if": {"column_id": "realized_pnl", "filter_query": "{realized_pnl} > 0"}, "color": COLORS["ACCENT_GREEN"], "fontWeight": "bold"},
            {"if": {"column_id": "realized_pnl", "filter_query": "{realized_pnl} < 0"}, "color": COLORS["ACCENT_RED"], "fontWeight": "bold"},
            {"if": {"column_id": "result", "filter_query": f'{{result}} = "{RESULT_WIN}"'}, "color": COLORS["ACCENT_GREEN"]},
            {"if": {"column_id": "result", "filter_query": f'{{result}} = "{RESULT_LOSS}"'}, "color": COLORS["ACCENT_RED"]},
        ],
    )


def _journal_editor() -> html.Div:
    """Trade ideas and payups of the selected trade (filled by a callback; saved with one button)."""
    return html.Div(
        [
            html.Div("Edit the journal of this trade", style={**_LABEL, "marginTop": "10px"}),
            dbc.Row(
                [
                    dbc.Col([html.Div("Trade ideas", style=_MUTED), dcc.Dropdown(id="archive-ideas", options=[], value=[], multi=True, placeholder="Pick any number (the list is in Settings)")], md=6),
                    dbc.Col([html.Div("Payups in entry (ticks)", style=_MUTED), dbc.Input(id="archive-payup-entry", type="number", step="any", size="sm")], md=2),
                    dbc.Col([html.Div("Payups in exit (ticks)", style=_MUTED), dbc.Input(id="archive-payup-exit", type="number", step="any", size="sm")], md=2),
                    dbc.Col(dbc.Button("Save", id="archive-journal-save", color="primary", size="sm"), md=2, className="d-flex align-items-end"),
                ],
                className="g-2 mb-1",
            ),
            dbc.Row(
                [
                    dbc.Col([html.Div("Conviction at entry (1 to 5)", style=_MUTED), dcc.Dropdown(id="archive-conviction", options=[{"label": str(c), "value": c} for c in CONVICTIONS], value=None, placeholder="—")], md=2),
                    dbc.Col([html.Div("Why it was closed", style=_MUTED), dcc.Dropdown(id="archive-exit-reason", options=[{"label": r, "value": r} for r in EXIT_REASONS], value=None, placeholder="—")], md=4),
                    dbc.Col([html.Div("Mistakes (what went wrong)", style=_MUTED), dcc.Dropdown(id="archive-mistakes", options=[{"label": m, "value": m} for m in MISTAKES], value=[], multi=True, placeholder="None")], md=6),
                ],
                className="g-2 mb-1",
            ),
            html.Div(id="archive-journal-status", style={"fontSize": "13px", "minHeight": "20px"}),
        ],
        id="archive-journal-editor",
    )


def _delete_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Delete from the archive")),
            dbc.ModalBody(
                [
                    html.Div(id="archive-delete-body"),
                    html.Div(
                        dbc.Checkbox(
                            id="archive-delete-baseline", value=False,
                            label="These were closed before the Account Reset: keep the Home figure unchanged (move the reset baseline with them)",
                        ),
                        id="archive-delete-baseline-wrap", style={"display": "none", "marginTop": "10px"},
                    ),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button("Cancel", id="archive-delete-cancel", color="secondary", outline=True),
                    dbc.Button("Delete permanently", id="archive-delete-confirm", color="danger"),
                ]
            ),
        ],
        id="archive-delete-modal",
        is_open=False,
    )


def archive_layout() -> html.Div:
    """Filter bar, summary stats row and the archive table with an expandable detail panel."""
    return html.Div(
        [
            dcc.Store(id="archive-rows", data=[]),
            dcc.Store(id="archive-selected", data=None),
            dcc.Store(id="archive-delete-ids", data=[]),
            dcc.Store(id="archive-deleted", data=0),
            _delete_modal(),
            html.H3("📁 Archive", style={**_HEADING, "marginBottom": "16px"}),
            _filter_bar(),
            _stats_row(),
            dbc.Tabs(
                [
                    dbc.Tab(_trades_tab(), label="Trades", tab_id="trades"),
                    dbc.Tab(analysis_tab(), label="Analysis", tab_id="analysis"),
                ],
                id="archive-tabs", active_tab="trades",
            ),
        ]
    )


def _trades_tab() -> html.Div:
    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(NO_ARCHIVE_TEXT, id="archive-message", style={**_MUTED, "marginBottom": "8px", "fontSize": "15px"}),
                        html.Div(
                            [
                                dbc.Button("⬇ Export to Excel", id="archive-export-btn", color="success", outline=True, size="sm", className="me-2",
                                           title="Every trade the table shows now (after the filters), with its legs and the summary"),
                                dcc.Download(id="archive-export-download"),
                                dbc.Button("🗑 Delete selected", id="archive-delete-btn", color="danger", outline=True, size="sm"),
                                dbc.Button("📊 Fill market data for old trades", id="archive-backfill-btn", color="info", outline=True, size="sm", className="ms-2",
                                           title="Market prices, movement, curve shape and ATR at the entry and exit of trades that have none (a few per click)"),
                                html.Span("Tick the boxes of the trades to delete (test trades, say).", style={**_MUTED, "marginLeft": "12px"}),
                                html.Span(id="archive-delete-status", style={"marginLeft": "12px", "fontSize": "13px"}),
                                dcc.Loading(html.Span(id="archive-backfill-status", style={"marginLeft": "12px", "fontSize": "13px"}), type="dot"),
                            ],
                            style={"marginBottom": "8px"},
                        ),
                        _table(),
                    ]
                ),
                style=CARD_STYLE,
                className="mb-3",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(id="archive-detail"),
                        _journal_editor(),
                        trade_editor(),
                        # Always in the layout: a callback Input that is missing from the page never fires.
                        dbc.Button("Close Detail", id="archive-detail-close", color="secondary", outline=True, size="sm"),
                    ]
                ),
                style={**CARD_STYLE, "display": "none"},
                id="archive-detail-card",
                className="mb-3",
            ),
        ]
    )
