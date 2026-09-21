"""Exposure Map tab: net exposure bar chart, exposure table and concentration warnings.

The layout is static; ui/callbacks/exposure_callbacks.py fills it from the open
structures. All colours come from COLORS.
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from ui.layouts.correlation_tab import empty_figure
from ui.layouts.shell import COLORS

DEFAULT_THRESHOLD = 10
NO_EXPOSURE_TEXT = "No open structures — no exposure to display"
FLAT_TEXT = "Net exposure is flat — no bars to display"
NO_WARNINGS_TEXT = "No concentration warnings at current threshold"

_CARD_STYLE = {"backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}"}
_HEADING = {"color": COLORS["TEXT_PRIMARY"], "marginBottom": "12px"}
_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}


# ----------------------------------------------------------------------
# Builders (pure: rows in, components out)
# ----------------------------------------------------------------------


def _lots(value: float) -> str:
    return f"{value:g}"


def build_bar_figure(rows: list[dict]) -> go.Figure:
    """Horizontal net-lots bars, chronological top to bottom; zero-net months are left out."""
    if not rows:
        return empty_figure(NO_EXPOSURE_TEXT)
    bars = [row for row in rows if row["net_lots"] != 0]
    if not bars:
        return empty_figure(FLAT_TEXT)

    figure = go.Figure(
        go.Bar(
            x=[row["net_lots"] for row in bars],
            y=[row["label"] for row in bars],
            orientation="h",
            marker={"color": [COLORS["ACCENT_GREEN"] if row["net_lots"] > 0 else COLORS["ACCENT_RED"] for row in bars]},
            text=[f"{row['net_lots']:+g}" for row in bars],
            textposition="outside",
            customdata=[", ".join(row["structures"]) for row in bars],
            hovertemplate="%{y}: %{x:+g} lots<br>%{customdata}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_width=2, line_color=COLORS["TEXT_PRIMARY"])
    figure.update_layout(
        title={"text": "Net Exposure by Contract Month", "font": {"size": 18}},
        paper_bgcolor=COLORS["CARD_BG"],
        plot_bgcolor=COLORS["CARD_BG"],
        font={"color": COLORS["TEXT_PRIMARY"]},
        margin={"l": 90, "r": 50, "t": 60, "b": 50},
        xaxis={"title": "Net lots (long +, short −)", "gridcolor": COLORS["BORDER_COLOR"], "zeroline": False},
        yaxis={"categoryorder": "array", "categoryarray": [row["label"] for row in bars], "autorange": "reversed"},
        showlegend=False,
    )
    return figure


def build_table_rows(rows: list[dict]) -> list[dict]:
    """DataTable rows: every month, including flat ones."""
    return [
        {
            "symbol": row["symbol"],
            "net_lots": row["net_lots"],
            "gross_long": row["gross_long"],
            "gross_short": row["gross_short"],
            "structures": ", ".join(row["structures"]),
        }
        for row in rows
    ]


TABLE_COLUMNS = [
    {"id": "symbol", "name": "Contract"},
    {"id": "net_lots", "name": "Net Lots", "type": "numeric"},
    {"id": "gross_long", "name": "Gross Long", "type": "numeric"},
    {"id": "gross_short", "name": "Gross Short", "type": "numeric"},
    {"id": "structures", "name": "Contributing Structures"},
]


def build_alerts(flagged: list[dict]) -> list:
    """One amber alert per flagged month, or a single green all-clear."""
    if not flagged:
        return [dbc.Alert(NO_WARNINGS_TEXT, color="success", className="mb-2")]
    return [
        dbc.Alert(
            f"{item['label']}: net {item['side']} {_lots(item['lots'])} lots (structures: {', '.join(item['structures'])})",
            color="warning",
            className="mb-2",
        )
        for item in flagged
    ]


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="exposure-table",
        columns=TABLE_COLUMNS,
        data=[],
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": COLORS["SIDEBAR_BG"],
            "color": COLORS["TEXT_SECONDARY"],
            "border": f"1px solid {COLORS['BORDER_COLOR']}",
            "fontWeight": "bold",
        },
        style_cell={
            "backgroundColor": COLORS["CARD_BG"],
            "color": COLORS["TEXT_PRIMARY"],
            "border": f"1px solid {COLORS['BORDER_COLOR']}",
            "padding": "8px 12px",
            "textAlign": "right",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_cell_conditional=[
            {"if": {"column_id": "symbol"}, "textAlign": "left"},
            {"if": {"column_id": "structures"}, "textAlign": "left"},
        ],
        style_data_conditional=[
            {"if": {"column_id": "net_lots", "filter_query": "{net_lots} > 0"}, "color": COLORS["ACCENT_GREEN"], "fontWeight": "bold"},
            {"if": {"column_id": "net_lots", "filter_query": "{net_lots} < 0"}, "color": COLORS["ACCENT_RED"], "fontWeight": "bold"},
            {"if": {"column_id": "net_lots", "filter_query": "{net_lots} = 0"}, "color": COLORS["TEXT_PRIMARY"]},
        ],
    )


def exposure_layout() -> html.Div:
    """Bar chart, table and concentration warnings stacked vertically."""
    return html.Div(
        [
            dcc.Store(id="exposure-data", data=[]),
            html.H3("📈 Exposure Map", style={**_HEADING, "marginBottom": "16px"}),
            dbc.Card(
                dbc.CardBody(
                    dcc.Graph(
                        id="exposure-bar-chart",
                        figure=empty_figure(NO_EXPOSURE_TEXT),
                        config={"displayModeBar": False},
                        style={"height": "420px"},
                    )
                ),
                style=_CARD_STYLE,
                className="mb-3",
            ),
            dbc.Card(
                dbc.CardBody([html.H5("Exposure Table", style=_HEADING), _table()]),
                style=_CARD_STYLE,
                className="mb-3",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Concentration Warnings", style=_HEADING),
                        html.Div(
                            [
                                html.Label("Flag positions exceeding (lots):", style={**_MUTED, "marginRight": "12px"}),
                                dbc.Input(
                                    id="exposure-threshold", type="number", value=DEFAULT_THRESHOLD, min=0, step=1,
                                    style={"width": "120px", "display": "inline-block"},
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.Div(id="exposure-alerts", children=build_alerts([])),
                    ]
                ),
                style=_CARD_STYLE,
                className="mb-3",
            ),
        ]
    )
