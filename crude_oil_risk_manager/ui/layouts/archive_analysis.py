"""The Archive's Analysis tab: headline measures, equity curve and drawdown, and results grouped by any journal column.

Built from core.journal_analysis on the trades the Archive's filters leave. Colours from COLORS.
"""

import math

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dash_table.Format import Format, Group, Scheme, Symbol
from plotly.subplots import make_subplots

from core.journal_analysis import DIMENSIONS, MIN_SAMPLE
from ui.layouts.shell import COLORS

EMPTY = "—"
_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_LABEL = {**_MUTED, "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "6px"}
_CARD = {"backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}"}
_MONEY = Format(precision=0, scheme=Scheme.fixed, group=Group.yes, symbol=Symbol.yes, symbol_prefix="$")


def _money(value) -> str:
    return EMPTY if value is None else f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def _pct(value) -> str:
    return EMPTY if value is None else f"{value * 100:.0f}%"


def _ratio(value) -> str:
    return EMPTY if value is None else "∞" if math.isinf(value) else f"{value:.2f}"


def build_cards(h: dict) -> list:
    """The headline measures as small cards: (label, value, good?)."""
    def tone(value):
        return None if value is None or value == 0 else value > 0

    items = [
        ("Trades", str(h["trades"]), None, f"{h['r_trades']} with a stop (R)"),
        ("Win rate", _pct(h["win_rate"]), None if h["win_rate"] is None else h["win_rate"] >= 0.5, ""),
        ("Expectancy per trade", _money(h["expectancy"]), tone(h["expectancy"]), f"R: {_ratio(h['expectancy_r'])}" if h["expectancy_r"] is not None else "R needs a stop on the trades"),
        ("Profit factor", _ratio(h["profit_factor"]), None if h["profit_factor"] is None else h["profit_factor"] >= 1.0, "gross wins ÷ gross losses"),
        ("Payoff (avg win ÷ avg loss)", _ratio(h["payoff"]), None if h["payoff"] is None else h["payoff"] >= 1.0, f"avg win {_money(h['avg_win'])}, avg loss {_money(h['avg_loss'])}"),
        ("Total PnL (net)", _money(h["total"]), tone(h["total"]), f"costs paid {_money(h['costs'])}"),
        ("Worst drawdown", _money(h["max_drawdown"]) if h["max_drawdown"] else EMPTY, None if not h["max_drawdown"] else False, f"longest losing run {h['longest_loss']}, winning {h['longest_win']}"),
        ("Avg payups", EMPTY if h["avg_payups"] is None else f"{h['avg_payups']:.1f} ticks", None, "entry + exit per trade"),
    ]
    cards = []
    for label, value, good, note in items:
        color = COLORS["TEXT_PRIMARY"] if good is None else COLORS["ACCENT_GREEN"] if good else COLORS["ACCENT_RED"]
        cards.append(dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(label, style=_LABEL), html.Div(value, style={"fontSize": "22px", "fontWeight": "bold", "color": color}), html.Div(note, style=_MUTED),
        ]), style={**_CARD, "height": "100%"}), md=6, xl=3, className="mb-3"))
    return cards


def build_equity_figure(curve) -> go.Figure:
    """Cumulative net PnL by trade in exit order, with the drawdown below."""
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    if len(curve):
        x = list(range(1, len(curve) + 1))
        names = list(curve["name"])
        figure.add_trace(go.Scatter(x=x, y=list(curve["cum"]), mode="lines+markers", name="Cumulative PnL", text=names,
                                    line={"color": COLORS["ACCENT_BLUE"], "width": 2}, hovertemplate="%{text}<br>Trade %{x}: %{y:$,.0f}<extra></extra>"), row=1, col=1)
        figure.add_trace(go.Scatter(x=x, y=list(curve["drawdown"]), mode="lines", fill="tozeroy", name="Drawdown", text=names,
                                    line={"color": COLORS["ACCENT_RED"], "width": 1}, hovertemplate="%{text}<br>%{y:$,.0f}<extra></extra>"), row=2, col=1)
    figure.update_layout(
        height=420, margin={"l": 60, "r": 20, "t": 30, "b": 40}, showlegend=False, paper_bgcolor=COLORS["CARD_BG"], plot_bgcolor=COLORS["CARD_BG"],
        font={"color": COLORS["TEXT_PRIMARY"]}, title={"text": "Equity curve (net of costs, trades in exit order)", "font": {"size": 15}},
    )
    figure.update_yaxes(gridcolor=COLORS["BORDER_COLOR"], tickprefix="$", tickformat=",")
    figure.update_xaxes(gridcolor=COLORS["BORDER_COLOR"], title_text="Trade number", row=2, col=1)
    if not len(curve):
        figure.add_annotation(text="No trades to analyse", showarrow=False, font={"size": 16, "color": COLORS["TEXT_SECONDARY"]})
    return figure


def build_bar_figure(table) -> go.Figure:
    """Expectancy per trade of each group (faded when the group has too few trades to trust)."""
    figure = go.Figure()
    if len(table):
        low = [str(s).startswith("low") for s in table["Sample"]]
        values = [0.0 if v is None or v != v else v for v in table["Expectancy $"]]
        colors = [COLORS["TEXT_SECONDARY"] if is_low else COLORS["ACCENT_GREEN"] if v >= 0 else COLORS["ACCENT_RED"] for v, is_low in zip(values, low)]
        figure.add_trace(go.Bar(x=list(table["Group"]), y=values, marker_color=colors, text=[f"n={n}" for n in table["Trades"]], textposition="outside",
                                hovertemplate="%{x}<br>%{y:$,.0f} per trade<extra></extra>"))
    figure.update_layout(
        height=320, margin={"l": 60, "r": 20, "t": 40, "b": 80}, paper_bgcolor=COLORS["CARD_BG"], plot_bgcolor=COLORS["CARD_BG"],
        font={"color": COLORS["TEXT_PRIMARY"]}, title={"text": f"Expectancy per trade by group (grey = under {MIN_SAMPLE} trades)", "font": {"size": 15}},
    )
    figure.update_yaxes(gridcolor=COLORS["BORDER_COLOR"], tickprefix="$", tickformat=",")
    figure.update_xaxes(tickangle=-25, automargin=True)
    return figure


_IDS = {
    "Group": "group", "Trades": "trades", "Win rate": "win_rate", "Avg win": "avg_win", "Avg loss": "avg_loss", "Payoff": "payoff",
    "Expectancy $": "exp_usd", "Expectancy R": "exp_r", "R trades": "r_trades", "Profit factor": "profit_factor", "Total $": "total",
    "Avg payups": "avg_payups", "Sample": "sample",
}
_DECIMALS = {"Payoff": 2, "Profit factor": 2, "Expectancy R": 2, "Avg payups": 1, "Win rate": 3}


def table_records(table) -> list[dict]:
    """The group table as DataTable rows: ids without spaces, numbers rounded, the profit factor's infinity as text."""
    records = []
    for row in table.to_dict("records"):
        record = {}
        for key, value in row.items():
            if isinstance(value, float):
                value = None if value != value else "∞" if math.isinf(value) else round(value, _DECIMALS.get(key, 0))
            record[_IDS[key]] = EMPTY if value is None else value
        records.append(record)
    return records


GROUP_COLUMNS = [
    {"id": "group", "name": "Group"}, {"id": "trades", "name": "Trades", "type": "numeric"},
    {"id": "win_rate", "name": "Win rate", "type": "numeric", "format": Format(precision=0, scheme=Scheme.percentage)},
    {"id": "avg_win", "name": "Avg win ($)", "type": "numeric", "format": _MONEY}, {"id": "avg_loss", "name": "Avg loss ($)", "type": "numeric", "format": _MONEY},
    {"id": "payoff", "name": "Payoff"}, {"id": "exp_usd", "name": "Expectancy ($)", "type": "numeric", "format": _MONEY},
    {"id": "exp_r", "name": "Expectancy (R)"}, {"id": "r_trades", "name": "Trades with R", "type": "numeric"},
    {"id": "profit_factor", "name": "Profit factor"}, {"id": "total", "name": "Total ($)", "type": "numeric", "format": _MONEY},
    {"id": "avg_payups", "name": "Avg payups"}, {"id": "sample", "name": "Sample"},
]


def _round_records(table, ids: dict, decimals: dict) -> list[dict]:
    records = []
    for row in table.to_dict("records"):
        record = {}
        for key, value in row.items():
            if isinstance(value, float):
                value = None if value != value else "∞" if math.isinf(value) else round(value, decimals.get(key, 0))
            record[ids[key]] = EMPTY if value is None else value
        records.append(record)
    return records


_TWO_IDS = {"Group A": "a", "Group B": "b", "Trades": "trades", "Win rate": "win_rate", "Expectancy $": "exp_usd", "Expectancy R": "exp_r",
            "Profit factor": "profit_factor", "Total $": "total", "Sample": "sample"}
_SEP_IDS = {"Column": "column", "Trades used": "trades", "Groups": "groups", "Best group": "best", "Worst group": "worst", "Win rate spread": "spread",
            "Cramer's V": "v", "p-value": "p", "Reading": "reading"}


def two_way_records(table) -> list[dict]:
    return _round_records(table, _TWO_IDS, {"Win rate": 3, "Expectancy R": 2, "Profit factor": 2})


def separation_records(table) -> list[dict]:
    return _round_records(table, _SEP_IDS, {"Win rate spread": 3, "Cramer's V": 2, "p-value": 3})


TWO_WAY_COLUMNS = [
    {"id": "a", "name": "First column's group"}, {"id": "b", "name": "Second column's group"}, {"id": "trades", "name": "Trades", "type": "numeric"},
    {"id": "win_rate", "name": "Win rate", "type": "numeric", "format": Format(precision=0, scheme=Scheme.percentage)},
    {"id": "exp_usd", "name": "Expectancy ($)", "type": "numeric", "format": _MONEY}, {"id": "exp_r", "name": "Expectancy (R)"},
    {"id": "profit_factor", "name": "Profit factor"}, {"id": "total", "name": "Total ($)", "type": "numeric", "format": _MONEY}, {"id": "sample", "name": "Sample"},
]
SEPARATION_COLUMNS = [
    {"id": "column", "name": "Column"}, {"id": "trades", "name": "Trades used", "type": "numeric"}, {"id": "groups", "name": "Groups", "type": "numeric"},
    {"id": "best", "name": "Wins most in"}, {"id": "worst", "name": "Wins least in"},
    {"id": "spread", "name": "Win rate spread", "type": "numeric", "format": Format(precision=0, scheme=Scheme.percentage)},
    {"id": "v", "name": "Strength (Cramer's V)"}, {"id": "p", "name": "p-value"}, {"id": "reading", "name": "Reading"},
]


def _table_style(left_columns: tuple) -> dict:
    return dict(
        sort_action="native", page_size=20, style_table={"overflowX": "auto"},
        style_header={"backgroundColor": COLORS["SIDEBAR_BG"], "color": COLORS["TEXT_SECONDARY"], "border": f"1px solid {COLORS['BORDER_COLOR']}", "fontWeight": "bold"},
        style_cell={"backgroundColor": COLORS["CARD_BG"], "color": COLORS["TEXT_PRIMARY"], "border": f"1px solid {COLORS['BORDER_COLOR']}", "padding": "8px 12px", "textAlign": "right"},
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"} for c in left_columns],
    )


def analysis_tab() -> html.Div:
    return html.Div(
        [
            html.Div(
                f"Every figure is per trade and net of transaction costs, for the trades the filters above leave. R is the PnL divided by the dollars "
                f"risked at the stop typed with the entry. A group with under {MIN_SAMPLE} trades is marked 'low': with so few trades it is mostly luck.",
                style={**_MUTED, "margin": "8px 0 12px"},
            ),
            dbc.Row(id="analysis-cards"),
            dbc.Card(dbc.CardBody(dcc.Graph(id="analysis-equity", config={"displayModeBar": False})), style=_CARD, className="mb-3"),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("What separates winners from losers", style=_LABEL),
                        html.Div(id="analysis-sep-note", style={**_MUTED, "marginBottom": "8px"}),
                        dash_table.DataTable(
                            id="analysis-sep-table", columns=SEPARATION_COLUMNS, data=[],
                            style_data_conditional=[
                                {"if": {"filter_query": '{reading} = "stands out"'}, "color": COLORS["ACCENT_GREEN"], "fontWeight": "bold"},
                                {"if": {"filter_query": '{reading} = "no sign"'}, "color": COLORS["TEXT_SECONDARY"]},
                            ],
                            **_table_style(("column", "best", "worst", "reading")),
                        ),
                    ]
                ),
                style=_CARD, className="mb-3",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div("Two columns together", style=_LABEL),
                        dbc.Row([
                            dbc.Col([html.Div("First column", style=_MUTED), dbc.Select(id="analysis-two-a", options=[{"label": v, "value": k} for k, v in DIMENSIONS.items()], value="curve")], md=4),
                            dbc.Col([html.Div("Second column", style=_MUTED), dbc.Select(id="analysis-two-b", options=[{"label": v, "value": k} for k, v in DIMENSIONS.items()], value="type")], md=4),
                        ], className="mb-3"),
                        dash_table.DataTable(
                            id="analysis-two-table", columns=TWO_WAY_COLUMNS, data=[],
                            style_data_conditional=[
                                {"if": {"filter_query": '{sample} contains "low"'}, "color": COLORS["TEXT_SECONDARY"], "fontStyle": "italic"},
                                {"if": {"column_id": "exp_usd", "filter_query": "{exp_usd} > 0"}, "color": COLORS["ACCENT_GREEN"]},
                                {"if": {"column_id": "exp_usd", "filter_query": "{exp_usd} < 0"}, "color": COLORS["ACCENT_RED"]},
                            ],
                            **_table_style(("a", "b", "sample")),
                        ),
                    ]
                ),
                style=_CARD, className="mb-3",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row([
                            dbc.Col([html.Div("Group the trades by", style=_LABEL), dbc.Select(id="analysis-dimension", options=[{"label": v, "value": k} for k, v in DIMENSIONS.items()], value="type")], md=4),
                        ], className="mb-3"),
                        dcc.Graph(id="analysis-bars", config={"displayModeBar": False}),
                        dash_table.DataTable(
                            id="analysis-table", columns=GROUP_COLUMNS, data=[], sort_action="native", page_size=25,
                            style_table={"overflowX": "auto"},
                            style_header={"backgroundColor": COLORS["SIDEBAR_BG"], "color": COLORS["TEXT_SECONDARY"], "border": f"1px solid {COLORS['BORDER_COLOR']}", "fontWeight": "bold"},
                            style_cell={"backgroundColor": COLORS["CARD_BG"], "color": COLORS["TEXT_PRIMARY"], "border": f"1px solid {COLORS['BORDER_COLOR']}", "padding": "8px 12px", "textAlign": "right"},
                            style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"} for c in ("group", "sample")],
                            style_data_conditional=[
                                {"if": {"filter_query": '{sample} contains "low"'}, "color": COLORS["TEXT_SECONDARY"], "fontStyle": "italic"},
                                {"if": {"column_id": "exp_usd", "filter_query": "{exp_usd} > 0"}, "color": COLORS["ACCENT_GREEN"]},
                                {"if": {"column_id": "exp_usd", "filter_query": "{exp_usd} < 0"}, "color": COLORS["ACCENT_RED"]},
                            ],
                        ),
                    ]
                ),
                style=_CARD, className="mb-3",
            ),
        ]
    )
