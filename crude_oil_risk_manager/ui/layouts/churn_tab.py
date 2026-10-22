"""Churn page (route /churn): which contracts offer the best intraday small-target churning, from five-minute candles.

For every live outright, spread, fly and d-fly of CL, BRN and gasoil and both sides the page simulates the trade you describe
(cross the spread to enter, a target that covers the transaction cost with a few dollars over, flat by the end of the
trading day) and ranks them by the churn score of core.churn_service. ONE table holds the ranking and the live part: the
live price, the number of oscillations of that price level in the chosen range, and the score; the GO / WAIT verdict of the
live price is the colour and the tooltip of the Score. Below it: what survives a worse entry fill, the best hour of the day
and the cost table (TC, rebates and break-even targets). Colours from COLORS.
"""

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dash_table.Format import Format, Group, Scheme

from core.churn_costs import DEFAULT_COMBO_REBATES, DEFAULT_COSTS, STRUCTURE_LOTS, breakeven_ticks, round_trip_cost
from core.churn_service import ChurnParams
from ui.container import container
from ui.layouts.curve_tab import _CARD, _LABEL, _MUTED, _layout, _table, empty_figure
from ui.layouts.shell import COLORS

REFRESH_MS = 3000
FAMILIES = ("outright", "spread", "fly", "dfly")
FAMILY_NAMES = {"outright": "Outright", "spread": "Spread", "fly": "Fly", "dfly": "D-fly"}
LIVE_REFRESH_MS = 10_000
K_COLORS = {0: "#3cb371", 1: "#f5a623", 2: "#e67e22", 3: "#d9534f"}

# (id, label, ChurnParams field, step, hint)
PARAM_INPUTS = [
    ("churn-p-contracts", "Contracts per curve", "contracts", 1, "The first N live outrights, spreads and flies of each product are downloaded and simulated (the front ones too)"),
    ("churn-p-startfrom", "Start from contract #", "start_from", 1, "Show and score only the contracts from this one onwards (front = 1). Spreads, flies and d-flies count by their first month"),
    ("churn-p-window", "Window (calendar days)", "window_days", 5, "How many days of five-minute candles to simulate over (a longer window means a longer first download)"),
    ("churn-p-osc", "Oscillation range (days)", "osc_days", 1, "The oscillations of the live price level are counted over this many trading days"),
    ("churn-p-margin", "Margin over cost ($)", "margin", 0.5, "A target must earn at least the round-trip cost plus this, per lot"),
    ("churn-p-maxtarget", "Largest target (ticks)", "max_target", 1, "Targets from the smallest one that covers the cost up to this are tried"),
    ("churn-p-sp-out", "Outright spread (ticks)", "spread_outright", 0.5, "Quoted bid-ask spread assumed when crossing: half on entry, half on a stop-out or the closing exit"),
    ("churn-p-sp-spr", "Spread spread (ticks)", "spread_spread", 0.5, "Same, for spreads"),
    ("churn-p-sp-fly", "Fly spread (ticks)", "spread_fly", 0.5, "Same, for flies"),
    ("churn-p-sp-dfly", "D-fly spread (ticks)", "spread_dfly", 0.5, "Same, for d-flies. They are built from the legs, so there is none: 0"),
    ("churn-p-wstart", "Trading hours start (London)", "window_start", 0.5, "Trades are only taken inside your trading hours, in London time (summer time is handled)"),
    ("churn-p-wend", "Trading hours end (London)", "window_end", 0.5, "A trade still open at this hour is closed by crossing the spread"),
    ("churn-p-stop", "Stop (x target)", "stop_mult", 0.5, "A trade is stopped out when the price moves this many targets against the fill; the simulation then moves on to a new price"),
    ("churn-p-patience", "Wait for the level (min)", "patience_min", 15, "After a win, wait this long for the price to come back to the level and buy again there; otherwise move on to a new price"),
    ("churn-p-minvol", "Min volume / day (lots)", "min_volume", 10, "Rows trading less than this are shown but not scored"),
]
COST_FIELDS = [("tick_size", "Tick size", 0.01), ("tick_value", "Tick value ($)", 1), ("tc", "TC per side per lot ($)", 0.01), ("rebate", "Rebate per side per lot ($)", 0.01)]

_TWO = Format(precision=2, scheme=Scheme.fixed)
_ONE = Format(precision=1, scheme=Scheme.fixed)
_INT = Format(group=Group.yes, precision=0, scheme=Scheme.fixed)
_PCT = Format(precision=0, scheme=Scheme.percentage)
_THREE = Format(precision=3, scheme=Scheme.fixed)


def _col(name, column_id, fmt=None):
    column = {"name": name, "id": column_id}
    if fmt is not None:
        column.update({"type": "numeric", "format": fmt})
    return column


def _num(value, digits=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), digits)


def _field(label: str, component, md: int, hint: str | None = None) -> dbc.Col:
    return dbc.Col([html.Div(label, style=_LABEL, title=hint), component], md=md, className="mb-2")


# ----------------------------------------------------------------------
# Controls and cost panel
# ----------------------------------------------------------------------


def churn_controls(params: ChurnParams) -> html.Div:
    inputs = [
        _field(name, dbc.Input(id=component_id, type="number", value=getattr(params, field), step=step, debounce=True, size="sm"), 2, hint)
        for component_id, name, field, step, hint in PARAM_INPUTS
    ]
    filters = dbc.Row([
        _field("Product", dcc.Dropdown(id="churn-product", options=[{"label": "All", "value": "all"}, {"label": "CL", "value": "CL"}, {"label": "BRN", "value": "BRN"}, {"label": "Gasoil", "value": "G"}], value="all", clearable=False), 2),
        _field("Curve", dcc.Dropdown(id="churn-family", options=[{"label": "All", "value": "all"}, *({"label": FAMILY_NAMES[f], "value": f} for f in FAMILIES)], value="all", clearable=False), 2),
        _field("Direction", dcc.Dropdown(id="churn-direction", options=[{"label": "Both", "value": "all"}, {"label": "Long", "value": "long"}, {"label": "Short", "value": "short"}], value="all", clearable=False), 2),
    ], className="g-2")
    return html.Div([filters, dbc.Row(inputs, className="g-2")], style={**_CARD, "padding": "12px 14px", "borderRadius": "6px", "marginBottom": "10px"})


def cost_panel(costs, combo, use_rebates: bool = False) -> html.Details:
    """Editable TC / rebate table plus the break-even targets it implies."""
    head = html.Tr([html.Th("Product", style=_LABEL), *(html.Th(label, style=_LABEL) for _, label, _ in COST_FIELDS)])
    body = [
        html.Tr([html.Td(product, style={"color": COLORS["TEXT_PRIMARY"], "paddingRight": "12px"}),
                 *(html.Td(dbc.Input(id=f"churn-c-{product}-{field}", type="number", value=getattr(costs[product], field), step=step, size="sm", debounce=True, style={"width": "110px"}))
                   for field, _, step in COST_FIELDS)])
        for product in DEFAULT_COSTS
    ]
    combo_inputs = html.Div([
        html.Span("Quoted CL/BZ structures on NYMEX pay a rebate per side per lot of: ", style=_MUTED),
        *(html.Span([f"{product} ", dbc.Input(id=f"churn-combo-{product}", type="number", value=combo[product], step=0.01, size="sm", debounce=True, style={"width": "90px", "display": "inline-block", "marginRight": "12px"})])
          for product in DEFAULT_COMBO_REBATES),
    ], style={"marginTop": "8px"})
    return html.Details([
        html.Summary("Costs: TC, rebates and break-even targets", style={"color": COLORS["TEXT_PRIMARY"], "cursor": "pointer", "fontSize": "15px", "margin": "6px 0"}),
        html.Div([
            html.Table([html.Thead(head), html.Tbody(body)], style={"borderSpacing": "8px", "borderCollapse": "separate"}),
            combo_inputs,
            dbc.Switch(id="churn-use-rebates", label="Subtract rebates from the TC (off: every trade pays the full TC, so a win is real profit)", value=use_rebates, className="mt-2"),
            html.Div([dbc.Button("Save costs", id="churn-costs-save", size="sm", color="secondary", outline=True, className="me-3"), html.Span(id="churn-costs-msg", style=_MUTED)], style={"marginTop": "8px"}),
            html.Div(id="churn-breakeven", style={"marginTop": "10px"}),
        ], style={**_CARD, "padding": "12px 14px", "borderRadius": "6px"}),
    ], style={"marginBottom": "12px"})


def breakeven_table(costs) -> html.Div:
    """Round-trip cost and break-even ticks of each product's outright, spread and fly (cost on the net lots of the structure)."""
    rows = []
    for product in ("CL", "BRN", "BZ", "WBS", "G"):
        cells = [html.Td(product, style={"color": COLORS["TEXT_PRIMARY"], "paddingRight": "14px"})]
        for family in ("outright", "spread", "fly", "dfly"):
            cost = round_trip_cost(product, family, costs)
            cells.append(html.Td(f"${cost:.2f} = {breakeven_ticks(product, family, costs):.2f} tick", style={"color": COLORS["TEXT_SECONDARY"], "paddingRight": "14px", "fontSize": "13px"}))
        rows.append(html.Tr(cells))
    head = html.Tr([html.Th("Round trip, per lot", style=_LABEL), *(html.Th(f"{f} ({STRUCTURE_LOTS[f]} lots)", style=_LABEL) for f in ("outright", "spread", "fly", "dfly"))])
    return html.Table([html.Thead(head), html.Tbody(rows)])


# ----------------------------------------------------------------------
# The table (ranking and live part together)
# ----------------------------------------------------------------------

COLUMN_TIPS = {
    "contract": "Hover over a contract for its target, the entry fill now, the target price and the net per day.",
    "price": "The live price from the curve poll (a d-fly is its first fly minus the next one). Blank if the poll has no price for it.",
    "volume_day": "Average volume per day (lots) in the window. Rows below the minimum are not scored.",
    "atr": "Daily ATR in points, for information.",
    "cost": "Round-trip transaction cost per lot of the structure: TC on its net lots, both sides (rebates only if switched on).",
    "net_hit": "What one lot earns when the target is reached: the target in dollars minus the round-trip cost.",
    "osc": "How many times, in the chosen range of days, the market came back to the entry level of this trade (the live price, plus half the spread for a long) and then moved the target in the trade's favour within one trading day.",
    "score": "0-100: geometric mean of edge (net $/day rank among the rows shown), hit probability, worse-price survival, positive days and 1 - blow-up rate, all measured on the last 40% of the days (the target is picked on the first 60%). 0 = not scored: loses money or too thin, hover for the numbers. Green = GO now, amber = WAIT, grey = closed: hover for why.",
    "side": "long = buy and sell the target above; short = sell and buy back the target below.",
    "worse": "Share of the net $/day that survives an entry fill 1 and 2 ticks worse than planned (average), 0-100%.",
}

COLUMNS = [
    _col("Contract", "contract"), _col("Live Price", "price", _THREE), _col("Volume/day", "volume_day", _INT), _col("ATR", "atr", _TWO), _col("Cost", "cost_rt", _TWO),
    _col("Per lot Net Profit", "net_hit", _TWO), _col("Number of oscillation", "osc", _INT), _col("Score", "score", _ONE), _col("Side", "direction"),
    _col("Worse Price Score", "worse_score", _PCT), {"name": "Verdict", "id": "verdict"},
]
TIP_KEYS = {"contract": "contract", "price": "price", "volume_day": "volume_day", "atr": "atr", "cost_rt": "cost", "net_hit": "net_hit", "osc": "osc", "score": "score", "direction": "side", "worse_score": "worse"}


def table_data(table) -> tuple[list[dict], list[dict]]:
    """(rows, tooltips) of the merged table from core.churn_service.ChurnService.live_table."""
    rows, tips = [], []
    for r in table.itertuples():
        rows.append({
            "contract": f"{r.product} {FAMILY_NAMES[r.family]} {r.label}", "price": _num(r.price, 3), "volume_day": _num(r.volume_day, 0), "atr": _num(r.atr, 2), "cost_rt": _num(r.cost_rt),
            "net_hit": _num(r.net_hit), "osc": None if not np.isfinite(r.osc) else int(r.osc), "score": _num(r.score, 1), "direction": r.direction, "worse_score": _num(r.worse_score, 4),
            "verdict": r.verdict,
        })
        levels = (f"Target {int(r.target)} ticks: enter about {r.entry:.3f}, exit {r.target_price:.3f}. " if np.isfinite(r.entry) else f"Target {int(r.target)} ticks. ")
        tips.append({
            "contract": {"value": levels + f"Net per day in the window: ${r.net_day:.2f} per lot.", "type": "text"},
            "score": {"value": f"{r.verdict}: {r.reason}", "type": "text"},
        })
    return rows, tips


def table_styles() -> list[dict]:
    styles = [{"if": {"column_id": "score", "filter_query": f'{{verdict}} = "{name}"'}, "color": color, "fontWeight": "bold"} for name, color in (("GO", "#3cb371"), ("WAIT", "#f5a623"))]
    return [
        *styles,
        {"if": {"column_id": "score", "filter_query": "{score} = 0"}, "color": COLORS["TEXT_SECONDARY"]},
        {"if": {"column_id": "osc", "filter_query": "{osc} = 0"}, "color": COLORS["TEXT_SECONDARY"]},
        {"if": {"column_id": "net_hit", "filter_query": "{net_hit} < 0"}, "color": COLORS["ACCENT_RED"]},
    ]


def churn_table():
    """The one table of the page. It stays in the layout and only its data is replaced, so the sorting you chose survives a refresh."""
    table = _table("churn-table", COLUMNS, [], table_styles(), page_size=40)
    table.hidden_columns = ["verdict"]
    table.tooltip_header = {column: {"value": COLUMN_TIPS[tip], "type": "text"} for column, tip in TIP_KEYS.items()}
    table.tooltip_delay = 0
    table.tooltip_duration = None
    return table


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------


def build_worse_fill_figures(grid, label: str, best: int) -> tuple[go.Figure, go.Figure]:
    """Net $/day and hit probability against the target, one line per worse-fill size."""
    if grid is None or grid.empty:
        return empty_figure("Pick a contract"), empty_figure("Pick a contract")
    net, hit = go.Figure(), go.Figure()
    for k, part in grid.groupby("worse"):
        name = "planned fill" if k == 0 else f"{k} tick{'s' if k > 1 else ''} worse"
        style = {"color": K_COLORS[int(k)], "width": 3 if k == 0 else 2}
        net.add_trace(go.Scatter(x=part["target"], y=part["net_day"], mode="lines+markers", name=name, line=style, hovertemplate="target %{x}: $%{y:.2f} per day<extra>" + name + "</extra>"))
        hit.add_trace(go.Scatter(x=part["target"], y=part["hit_prob"], mode="lines+markers", name=name, line=style, hovertemplate="target %{x}: %{y:.0%}<extra>" + name + "</extra>"))
    for figure, title, axis in ((net, f"{label}: net $ per day against the target", "net $ per day per lot"), (hit, f"{label}: chance the target is reached", "hit probability")):
        figure.add_vline(x=best, line={"color": COLORS["ACCENT_YELLOW"], "dash": "dash", "width": 1}, annotation_text="best", annotation_position="top")
        figure.update_layout(**_layout(title={"text": title, "font": {"size": 14}}, xaxis={"title": "Target (ticks)", "dtick": 1, "gridcolor": COLORS["BORDER_COLOR"]},
                                       yaxis={"title": axis, "gridcolor": COLORS["BORDER_COLOR"], **({"tickformat": ".0%", "range": [0, 1.02]} if "hit" in axis else {})},
                                       margin={"l": 70, "r": 30, "t": 60, "b": 50}))
    net.add_hline(y=0, line={"color": COLORS["TEXT_SECONDARY"], "width": 1})
    return net, hit


def build_session_figure(sessions) -> go.Figure:
    """Net $ per trade and hit probability by London entry hour, pooled over the best-scoring rows."""
    if sessions is None or sessions.empty:
        return empty_figure("No scored rows yet")
    figure = go.Figure()
    figure.add_trace(go.Bar(x=sessions["hour"], y=sessions["net_trade"], name="net $ per trade", marker_color="#4da3ff", yaxis="y",
                            hovertemplate="%{x}:00 London: $%{y:.2f} per trade<extra></extra>"))
    figure.add_trace(go.Scatter(x=sessions["hour"], y=sessions["hit_prob"], name="hit probability", mode="lines+markers", line={"color": COLORS["ACCENT_GREEN"], "width": 2}, yaxis="y2",
                                hovertemplate="%{x}:00 London: %{y:.0%} reach the target<extra></extra>"))
    figure.update_layout(**_layout(
        title={"text": "Which hour churns best (entry hour, London time; the best-scoring contracts pooled)", "font": {"size": 14}},
        xaxis={"title": "Entry hour (London)", "dtick": 1, "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis={"title": "net $ per trade", "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis2={"title": "hit probability", "overlaying": "y", "side": "right", "tickformat": ".0%", "range": [0, 1.02], "showgrid": False},
        margin={"l": 70, "r": 70, "t": 60, "b": 50},
    ))
    return figure


def churn_layout() -> html.Div:
    service = container.churn_service
    params = service.params() if service is not None else ChurnParams()
    costs, combo = service.costs() if service is not None else (DEFAULT_COSTS, DEFAULT_COMBO_REBATES)
    use_rebates = service.use_rebates() if service is not None else False
    return html.Div([
        dcc.Interval(id="churn-refresh", interval=REFRESH_MS, n_intervals=0),
        dcc.Store(id="churn-version"),
        dcc.Store(id="churn-params"),
        dcc.Store(id="churn-costs-version", data=0),
        html.H3("♻️ Churn", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "4px"}),
        html.Div("Which contracts give the best chance of many small, cost-covering wins within a trading day. Simulated from five-minute candles of every live CL, BRN and gasoil outright, spread, fly and d-fly; the live price and the oscillation count are in the same table.", style={**_MUTED, "marginBottom": "10px"}),
        dbc.Row([
            dbc.Col(html.Div(id="churn-status", style={**_MUTED, "minHeight": "20px"}), md=10, className="pt-1"),
            dbc.Col(dbc.Button("Download / update data", id="churn-update-btn", size="sm", color="secondary", outline=True), md=2, className="text-end"),
        ], className="mb-2 align-items-center"),
        cost_panel(costs, combo, use_rebates),
        churn_controls(params),
        dcc.Interval(id="churn-live-refresh", interval=LIVE_REFRESH_MS, n_intervals=0),
        dcc.Loading(html.Div(id="churn-note", style={**_MUTED, "margin": "6px 0 8px", "minHeight": "20px"}), type="dot"),
        churn_table(),
        html.H6("One contract: does it still churn with a worse fill?", style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "22px"}),
        html.Div("Pick a row's contract and direction. Each line is the same trade with the entry fill that many ticks worse than planned.", style={**_MUTED, "marginBottom": "8px"}),
        dbc.Row([dbc.Col([html.Div("Contract and direction", style=_LABEL), dcc.Dropdown(id="churn-contract", options=[], clearable=False)], md=4)], className="mb-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="churn-net-chart", figure=empty_figure("Pick a contract"), config={"displayModeBar": False}, style={"height": "380px"}), lg=6),
            dbc.Col(dcc.Graph(id="churn-hit-chart", figure=empty_figure("Pick a contract"), config={"displayModeBar": False}, style={"height": "380px"}), lg=6),
        ]),
        html.H6("Best session", style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "22px"}),
        dcc.Graph(id="churn-session-chart", figure=empty_figure("Loading"), config={"displayModeBar": False}, style={"height": "380px"}),
    ], style={"paddingTop": "4px"})
