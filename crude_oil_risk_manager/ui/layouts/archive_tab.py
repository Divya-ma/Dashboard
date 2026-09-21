"""Archive tab: filters, summary stats and the table of closed structures (read-only).

The layout is static; ui/callbacks/archive_callbacks.py fills it. All colours come
from COLORS. Nothing on this tab writes to the database.
"""

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
from dash.dash_table.Format import Format, Group, Scheme, Symbol

from core.archive import RESULT_LOSS, RESULT_WIN
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


def build_table_rows(rows: list[dict]) -> list[dict]:
    """Scalar DataTable rows (the row `id` is what a click reports back)."""
    return [
        {
            "id": row["structure_id"],
            "name": row["name"],
            "type": row["type"],
            "legs": row["legs_summary"],
            "entry_date": _or_dash(row["entry_date"]),
            "exit_date": _or_dash(row["exit_date"]),
            "days_held": _or_dash(row["days_held"]),
            "entry_price": _or_dash(row["entry_price"], lambda v: round(v, 4)),
            "exit_price": _or_dash(row["exit_price"], lambda v: round(v, 4)),
            "lots": _or_dash(row["lots"], lambda v: f"{v:g}"),
            "realized_pnl": row["realized_pnl"],
            "result": row["result"],
        }
        for row in rows
    ]


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
            fact("Realized PnL", money(pnl), {"color": pnl_color}),
            fact("Days Held", _or_dash(row["days_held"])),
        ]
    )
    parts = [
        html.H5(f"{row['name']} — {row['type']}", style=_HEADING),
        html.Div("Legs", style=_LABEL),
        legs_table,
        facts,
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
    {
        "id": "realized_pnl", "name": "Realized PnL ($)", "type": "numeric",
        "format": Format(precision=2, scheme=Scheme.fixed, group=Group.yes, symbol=Symbol.yes, symbol_prefix="$"),
    },
    {"id": "result", "name": "Result"},
]


def _table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="archive-table",
        columns=TABLE_COLUMNS,
        data=[],
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


def archive_layout() -> html.Div:
    """Filter bar, summary stats row and the archive table with an expandable detail panel."""
    return html.Div(
        [
            dcc.Store(id="archive-rows", data=[]),
            dcc.Store(id="archive-selected", data=None),
            html.H3("📁 Archive", style={**_HEADING, "marginBottom": "16px"}),
            _filter_bar(),
            _stats_row(),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(NO_ARCHIVE_TEXT, id="archive-message", style={**_MUTED, "marginBottom": "8px", "fontSize": "15px"}),
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
