"""Home tab layout: portfolio metric cards, winner/loser, stale-data warning and live structures table.

This is the skeleton only; ui/callbacks/home_callbacks.py fills it from the
`store-portfolio-pnl` store.
"""

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import html

from ui.layouts.shell import COLORS

MAX_TABLE_ROWS = 20

_LABEL_STYLE = {
    "fontSize": "12px",
    "color": COLORS["TEXT_SECONDARY"],
    "textTransform": "uppercase",
    "letterSpacing": "0.05em",
    "marginBottom": "6px",
}
_VALUE_STYLE = {"fontSize": "26px", "fontWeight": "bold", "color": COLORS["TEXT_PRIMARY"], "lineHeight": "1.2"}


def _rgba(hex_color: str, alpha: float) -> str:
    """'#rrggbb' -> 'rgba(r, g, b, alpha)' so tints are derived from the COLORS palette."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _card(label: str, body_id: str, accent: str, value_style: dict | None = None) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(label, style=_LABEL_STYLE),
                html.Div("—", id=body_id, style={**_VALUE_STYLE, **(value_style or {})}),
            ]
        ),
        style={
            "backgroundColor": COLORS["CARD_BG"],
            "border": f"1px solid {COLORS['BORDER_COLOR']}",
            "borderTop": f"3px solid {accent}",
            "height": "100%",
        },
    )


# AG Grid 33+ themes via its Theming API (the old "ag-theme-alpine-dark" CSS class no longer applies).
GRID_THEME = {
    "function": (
        "themeQuartz.withParams({"
        f"backgroundColor: '{COLORS['CARD_BG']}', "
        f"foregroundColor: '{COLORS['TEXT_PRIMARY']}', "
        f"headerBackgroundColor: '{COLORS['SIDEBAR_BG']}', "
        f"headerTextColor: '{COLORS['TEXT_SECONDARY']}', "
        f"borderColor: '{COLORS['BORDER_COLOR']}', "
        "browserColorScheme: 'dark'})"
    )
}

_PNL_FORMATTER = {
    "function": (
        "params.value == null ? '' : "
        "(params.value >= 0 ? '+$' : '-$') + d3.format(',.0f')(Math.abs(params.value))"
    )
}
_PNL_CELL_STYLE = {
    "styleConditions": [
        {"condition": "params.value >= 0", "style": {"color": COLORS["ACCENT_GREEN"]}},
        {"condition": "params.value < 0", "style": {"color": COLORS["ACCENT_RED"]}},
    ]
}


def _pnl_column(field: str, header: str) -> dict:
    return {
        "field": field,
        "headerName": header,
        "type": "rightAligned",
        "valueFormatter": _PNL_FORMATTER,
        "cellStyle": _PNL_CELL_STYLE,
    }


COLUMN_DEFS = [
    {"field": "name", "headerName": "Name", "flex": 2, "minWidth": 180},
    {"field": "products", "headerName": "Product(s)"},
    _pnl_column("unrealized", "Unrealized PnL"),
    _pnl_column("realized", "Realized PnL"),
    _pnl_column("total", "Total PnL"),
    {"field": "status", "headerName": "Status"},
    {"field": "days_held", "headerName": "Days Held", "type": "rightAligned"},
]

ROW_STYLE_CONDITIONS = [
    {
        "condition": "params.data.total > 0",
        "style": {"backgroundColor": _rgba(COLORS["ACCENT_GREEN"], 0.08)},
    },
    {
        "condition": "params.data.total < 0",
        "style": {"backgroundColor": _rgba(COLORS["ACCENT_RED"], 0.08)},
    },
]


def home_layout() -> html.Div:
    """Skeleton of the Home tab with the component ids the callbacks fill in."""
    metric_cards = dbc.Row(
        [
            dbc.Col(_card("Total PnL", "home-card-total-pnl", COLORS["ACCENT_GREEN"]), md=4, xl=2, className="mb-3"),
            dbc.Col(_card("Today's PnL", "home-card-today-pnl", COLORS["ACCENT_GREEN"]), md=4, xl=2, className="mb-3"),
            dbc.Col(_card("Open Structures", "home-card-open-structures", COLORS["ACCENT_BLUE"]), md=4, xl=2, className="mb-3"),
            dbc.Col(_card("Open Legs", "home-card-open-legs", COLORS["ACCENT_BLUE"]), md=4, xl=2, className="mb-3"),
            dbc.Col(_card("Margin Used", "home-card-margin-used", COLORS["ACCENT_YELLOW"]), md=4, xl=2, className="mb-3"),
            dbc.Col(
                _card("Net Lots", "home-card-net-lots", COLORS["ACCENT_BLUE"], {"fontSize": "16px"}),
                md=4, xl=2, className="mb-3",
            ),
        ]
    )

    winner_loser_stale = dbc.Row(
        [
            dbc.Col(
                _card("Largest Winner", "home-winner-card", COLORS["ACCENT_GREEN"], {"fontSize": "18px"}),
                md=4, className="mb-3",
            ),
            dbc.Col(
                _card("Largest Loser", "home-loser-card", COLORS["ACCENT_RED"], {"fontSize": "18px"}),
                md=4, className="mb-3",
            ),
            dbc.Col(html.Div(id="home-stale-banner", style={"display": "none"}), md=4, className="mb-3"),
        ]
    )

    table = html.Div(
        [
            html.Div("Live Structures", style={**_LABEL_STYLE, "fontSize": "14px", "marginBottom": "10px"}),
            dag.AgGrid(
                id="home-structures-table",
                columnDefs=COLUMN_DEFS,
                rowData=[],
                getRowId="params.data.structure_id",
                defaultColDef={"sortable": True, "resizable": True},
                columnSize="responsiveSizeToFit",
                getRowStyle={"styleConditions": ROW_STYLE_CONDITIONS},
                dashGridOptions={"domLayout": "autoHeight", "suppressCellFocus": True, "animateRows": False, "theme": GRID_THEME},
                style={"width": "100%"},
            ),
        ]
    )

    return html.Div([metric_cards, winner_loser_stale, table])
