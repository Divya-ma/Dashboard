"""VaR & Scenarios tab: historical VaR with a PnL histogram, and ATR-based shock scenarios.

The layout is static; ui/callbacks/var_callbacks.py fills it. All colours come from COLORS.
"""

import math

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dash_table.Format import Format, Group, Scheme, Sign, Symbol

from core.scenarios import KIND_PORTFOLIO
from ui.layouts.correlation_tab import empty_figure
from ui.layouts.shell import COLORS

LOOKBACK_OPTIONS = [{"label": f"{days}d", "value": str(days)} for days in (30, 60, 90, 252)]
DEFAULT_LOOKBACK = "60"
NOTHING_TO_ANALYZE = "No open structures — nothing to analyze"
ATR_UNAVAILABLE = "ATR unavailable"
SHOCK_NOTE = "Shock size = previous day's True Range per instrument"
SHOCK_LABEL = "1-Day ATR Shock (prev day)"

_CARD_STYLE = {"backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}"}
_HEADING = {"color": COLORS["TEXT_PRIMARY"], "marginBottom": "12px"}
_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_LABEL = {**_MUTED, "textTransform": "uppercase", "letterSpacing": "0.05em"}
_PLACEHOLDER = "—"


def _money(value: float) -> str:
    return f"${value:,.0f}"


# ----------------------------------------------------------------------
# VaR section builders
# ----------------------------------------------------------------------


def build_var_cards(result, lookback_days: int) -> tuple[str, str, str, str]:
    """Texts for the four cards: VaR 95%, VaR 99%, lookback, observations."""
    if result is None or result.error:
        return _PLACEHOLDER, _PLACEHOLDER, f"{lookback_days}d", _PLACEHOLDER
    observations = str(result.observations)
    if result.observations < result.requested:
        observations += f" (of {result.requested} requested)"
    return _money(result.var[0.95]), _money(result.var[0.99]), f"{lookback_days}d", observations


def build_var_warnings(result) -> list:
    """Amber banner: skipped symbols and the insufficient-data warning."""
    if result is None:
        return []
    return [dbc.Alert(f"⚠️ {message}", color="warning", className="mb-2") for message in result.warnings]


def build_histogram(result, lookback_days: int) -> go.Figure:
    """Portfolio daily PnL distribution: red loss bins, green profit bins, dashed VaR lines."""
    if result is None or result.error:
        return empty_figure(result.error if result else NOTHING_TO_ANALYZE, is_error=bool(result and result.error != NOTHING_TO_ANALYZE))
    pnl = result.pnl.to_numpy(dtype=float)
    low, high = float(pnl.min()), float(pnl.max())
    span = high - low
    width = span / 20 if span > 0 else max(abs(high), 1.0) / 10
    # Bin edges are multiples of the width, so zero is always an edge: no bar straddles profit and loss.
    edges = np.arange(math.floor(low / width) * width, high + width * 1.0001, width)
    counts, edges = np.histogram(pnl, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2

    figure = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=width * 0.95,
            marker={"color": [COLORS["ACCENT_RED"] if c < 0 else COLORS["ACCENT_GREEN"] for c in centers]},
            hovertemplate="PnL %{x:$,.0f}: %{y} day(s)<extra></extra>",
        )
    )
    for confidence, color, position in ((0.95, COLORS["ACCENT_YELLOW"], "top right"), (0.99, COLORS["TEXT_PRIMARY"], "top left")):
        figure.add_vline(
            x=result.cutoff[confidence],
            line_dash="dash",
            line_color=color,
            annotation_text=f"VaR {confidence:.0%}: {_money(result.var[confidence])}",
            annotation_position=position,
            annotation_font_color=color,
        )
    figure.update_layout(
        title={"text": f"Portfolio PnL Distribution ({lookback_days}d Historical)", "font": {"size": 18}},
        paper_bgcolor=COLORS["CARD_BG"],
        plot_bgcolor=COLORS["CARD_BG"],
        font={"color": COLORS["TEXT_PRIMARY"]},
        margin={"l": 60, "r": 30, "t": 70, "b": 50},
        xaxis={"title": "PnL ($)", "tickprefix": "$", "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis={"title": "Frequency", "gridcolor": COLORS["BORDER_COLOR"]},
        showlegend=False,
        bargap=0,
    )
    return figure


# ----------------------------------------------------------------------
# Scenarios section builders
# ----------------------------------------------------------------------


def build_scenario_table_rows(rows: list[dict]) -> list[dict]:
    """DataTable rows: Scenario | Shock Size (pts) | PnL Impact ($) | Direction."""
    table = []
    for row in rows:
        arrow = "▲ Up" if row["sign"] > 0 else "▼ Down"
        label = "+ATR" if row["sign"] > 0 else "−ATR"
        if row["kind"] == KIND_PORTFOLIO:
            scenario, shock = f"All Instruments {label}", "each instrument's own ATR"
        else:
            scenario = f"{row['symbol']} {label}"
            shock = ATR_UNAVAILABLE if row["shock"] is None else f"{row['shock']:.2f}"
        table.append({"scenario": scenario, "shock": shock, "pnl_impact": row["pnl_impact"], "direction": arrow})
    return table


def build_scenario_note(rows: list[dict]) -> str:
    """Which instruments the portfolio-wide rows had to leave out (no ATR)."""
    excluded = next((r["excluded"] for r in rows if r["kind"] == KIND_PORTFOLIO), [])
    return f"Excluded from the portfolio-wide scenarios (ATR unavailable): {', '.join(excluded)}." if excluded else ""


SCENARIO_COLUMNS = [
    {"id": "scenario", "name": "Scenario"},
    {"id": "shock", "name": "Shock Size (pts)"},
    {
        "id": "pnl_impact", "name": "PnL Impact ($)", "type": "numeric",
        "format": Format(precision=0, scheme=Scheme.fixed, group=Group.yes, sign=Sign.positive, symbol=Symbol.yes, symbol_prefix="$"),
    },
    {"id": "direction", "name": "Direction"},
]


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _metric_card(label: str, value_id: str, accent: str) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(label, style=_LABEL),
                    html.Div(_PLACEHOLDER, id=value_id, style={"fontSize": "26px", "fontWeight": "bold", "color": COLORS["TEXT_PRIMARY"]}),
                ]
            ),
            style={**_CARD_STYLE, "borderTop": f"3px solid {accent}", "height": "100%"},
        ),
        md=6, xl=3, className="mb-3",
    )


def _scenario_table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="scenario-table",
        columns=SCENARIO_COLUMNS,
        data=[],
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
            {"if": {"column_id": "scenario"}, "textAlign": "left"},
            {"if": {"column_id": "direction"}, "textAlign": "left"},
        ],
        style_data_conditional=[
            {"if": {"column_id": "pnl_impact", "filter_query": "{pnl_impact} > 0"}, "color": COLORS["ACCENT_GREEN"], "fontWeight": "bold"},
            {"if": {"column_id": "pnl_impact", "filter_query": "{pnl_impact} < 0"}, "color": COLORS["ACCENT_RED"], "fontWeight": "bold"},
            {"if": {"column_id": "shock", "filter_query": f'{{shock}} = "{ATR_UNAVAILABLE}"'}, "color": COLORS["ACCENT_YELLOW"]},
            {"if": {"filter_query": '{scenario} contains "All Instruments"'}, "borderTop": f"2px solid {COLORS['BORDER_COLOR']}"},
        ],
    )


def var_scenario_layout() -> html.Div:
    """VaR section on top, scenario analysis below."""
    var_section = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H5("Historical Simulation VaR (1-day)", style={**_HEADING, "display": "inline-block", "marginRight": "24px"}),
                        html.Span("Lookback", style={**_MUTED, "marginRight": "8px"}),
                        dbc.Select(id="var-lookback", options=LOOKBACK_OPTIONS, value=DEFAULT_LOOKBACK, style={"width": "110px", "display": "inline-block"}),
                    ],
                    className="mb-3",
                ),
                html.Div(id="var-warnings"),
                dbc.Row(
                    [
                        _metric_card("VaR 95% (1-day)", "var-card-95", COLORS["ACCENT_YELLOW"]),
                        _metric_card("VaR 99% (1-day)", "var-card-99", COLORS["ACCENT_RED"]),
                        _metric_card("Lookback", "var-card-lookback", COLORS["ACCENT_BLUE"]),
                        _metric_card("Observations", "var-card-observations", COLORS["ACCENT_BLUE"]),
                    ]
                ),
                dcc.Graph(id="var-histogram", figure=empty_figure(NOTHING_TO_ANALYZE), config={"displayModeBar": False}, style={"height": "420px"}),
            ]
        ),
        style=_CARD_STYLE,
        className="mb-3",
    )
    scenario_section = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H5("Scenario Analysis — ATR-Based Shocks", style={**_HEADING, "display": "inline-block", "marginRight": "24px"}),
                        dbc.Button("Refresh Scenarios", id="scenario-refresh-btn", color="secondary", outline=True, size="sm"),
                    ]
                ),
                html.Div(f"{SHOCK_NOTE}  ·  {SHOCK_LABEL}", style={**_MUTED, "marginBottom": "12px"}),
                html.Div(NOTHING_TO_ANALYZE, id="scenario-message", style={**_MUTED, "marginBottom": "8px"}),
                _scenario_table(),
                html.Div(id="scenario-note", style={"color": COLORS["ACCENT_YELLOW"], "fontSize": "13px", "marginTop": "10px"}),
            ]
        ),
        style=_CARD_STYLE,
        className="mb-3",
    )
    return html.Div([html.H3("⚠️ VaR & Scenarios", style={**_HEADING, "marginBottom": "16px"}), var_section, scenario_section])
