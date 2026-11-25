"""Trade Analyzer page (route /trade-analyzer): the chance a trade reaches its target before its stop, and where to put them.

Left to right: the form (structure, side, lots, risk per structure and per trade, days to give the trade, optional stop and
target of your own), then the result: headline numbers, the stop / target table, the spread of replayed trades, the earlier
years used, today's price context and the effect on the open portfolio. The layout is static; the builders below turn a
core.trade_analysis.Analysis into components and ui/callbacks/trade_analyzer_callbacks.py calls them.

The old Curve Kinks page this replaced is kept, unused, in ui/layouts/curve_tab.py.
"""

import math

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from core.structure_utils import STRUCTURE_TEMPLATES
from core.trade_analysis import CONTEXT_NAMES, DEFAULT_HORIZON, DEFAULT_WINDOW, MAX_HORIZON, Analysis, PortfolioEffect, load_risk_per_trade
from ui.container import container
from ui.layouts.curve_tab import _CARD, _LABEL, _MUTED, _layout, empty_figure
from ui.layouts.shell import COLORS

SIDE_OPTIONS = [{"label": "Buy", "value": 1}, {"label": "Sell", "value": -1}]
PLACEHOLDER = "Enter a structure and press Analyze."
GOOD, BAD = COLORS["ACCENT_GREEN"], COLORS["ACCENT_RED"]
FAN_COLOR = "#4da3ff"

GRID_COLUMNS = [
    {"name": "Stop (pts)", "id": "stop"}, {"name": "Target (pts)", "id": "target"}, {"name": "Reward:risk", "id": "rr"},
    {"name": "Chance of winning", "id": "p"}, {"name": "Range (90%)", "id": "range"}, {"name": "Still open at the end", "id": "unres"},
    {"name": "Risk $", "id": "risk"}, {"name": "Reward $", "id": "reward"}, {"name": "Expected $", "id": "exp"},
    {"name": "Expected $ (low end)", "id": "exp_low"}, {"name": "Max lots", "id": "max_lots"},
]
YEAR_COLUMNS = [
    {"name": "Year", "id": "year"}, {"name": "Structure", "id": "structure"}, {"name": "Days replayed", "id": "days"},
    {"name": "Won", "id": "won"}, {"name": "Lost", "id": "lost"}, {"name": "Still open", "id": "unresolved"}, {"name": "Weight", "id": "weight"},
]
CONTEXT_COLUMNS = [
    {"name": "Measure", "id": "name"}, {"name": "Today", "id": "now"}, {"name": "Where today sits among the replayed days", "id": "rank"},
]


def _pct(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value * 100:.{digits}f}%"


def _money(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{'-' if value < 0 else ''}${abs(value):,.0f}"


def _points(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.3g}"


def _card(title: str, value: str, note: str = "", color: str | None = None) -> dbc.Col:
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.Div(title, style=_LABEL),
            html.Div(value, style={"fontSize": "26px", "fontWeight": "bold", "color": color or COLORS["TEXT_PRIMARY"]}),
            html.Div(note, style=_MUTED),
        ]), style=_CARD), md=3, className="mb-3",
    )


def grid_rows(analysis: Analysis) -> list[dict]:
    spec, dollars = analysis.spec, analysis.dollars_per_point
    return [
        {
            "stop": _points(g.stop), "target": _points(g.target), "rr": f"{g.reward_risk:.1f}", "p": _pct(g.p_win),
            "range": f"{_pct(g.p_low)} to {_pct(g.p_high)}", "unres": _pct(g.unresolved_share),
            "risk": _money(g.stop * dollars * spec.lots), "reward": _money(g.target * dollars * spec.lots),
            "exp": _money(g.expectancy), "exp_low": _money(g.expectancy_low), "max_lots": g.max_lots,
        }
        for g in analysis.grid
    ]


def year_rows(analysis: Analysis) -> list[dict]:
    return [{**y, "weight": _pct(y["weight"]), "year": str(y["year"])} for y in analysis.years]


def context_rows(analysis: Analysis) -> list[dict]:
    return [
        {"name": CONTEXT_NAMES.get(k, k), "now": f"{v:,.2f}", "rank": f"above {analysis.context_rank[k]:.0f}% of the days" if k in analysis.context_rank else ""}
        for k, v in analysis.context_now.items()
    ]


def highlight(analysis: Analysis) -> list[dict]:
    """The row of the headline stop / target pair."""
    rows = grid_rows(analysis)
    head = analysis.headline
    for i, g in enumerate(analysis.grid):
        if head is not None and (g.stop, g.target) == (head.stop, head.target):
            return [{"if": {"row_index": i}, "backgroundColor": "#1f3a5f", "fontWeight": "bold"}] if i < len(rows) else []
    return []


def verdict(analysis: Analysis) -> str:
    """One plain-English paragraph for the headline pair."""
    spec, head = analysis.spec, analysis.headline
    side = "buying" if spec.side > 0 else "selling"
    text = (
        f"{side.capitalize()} {spec.lots:g} lots of {spec.symbol} with a stop {_points(head.stop)} points away and a target "
        f"{_points(head.target)} points away: in {analysis.n_paths} comparable days of {analysis.n_years} earlier years the target was "
        f"met before the stop {_pct(head.p_win)} of the time that either was reached (90% range {_pct(head.p_low)} to {_pct(head.p_high)}). "
        f"Closing what was still open at the end, the average result was {_money(head.expectancy)} per trade."
    )
    if analysis.random_walk_p is not None and math.isfinite(analysis.random_walk_p):
        text += f" A random walk with the same volatility would win {_pct(analysis.random_walk_p)} of the time."
    return text


def summary_cards(analysis: Analysis) -> dbc.Row:
    spec, head = analysis.spec, analysis.headline
    dollars = analysis.dollars_per_point
    risk, reward = head.stop * dollars * spec.lots, head.target * dollars * spec.lots
    edge = head.expectancy_low > 0
    entry = "n/a" if analysis.entry is None else f"{analysis.entry:.3f} ({analysis.entry_source})"
    return dbc.Row([
        _card("Chance target is met first", _pct(head.p_win), f"90% range {_pct(head.p_low)} to {_pct(head.p_high)}", GOOD if edge else None),
        _card("Expected result per trade", _money(head.expectancy), f"low end {_money(head.expectancy_low)}", GOOD if edge else BAD if head.expectancy < 0 else None),
        _card("Risk / reward", f"{_money(risk)} / {_money(reward)}", f"{head.reward_risk:.1f} to 1; at most {head.max_lots} lots at this stop"),
        _card("Entry", entry, f"{analysis.days_to_expiry} trading days to the front leg's expiry"),
        _card("Regime-weighted", _pct(analysis.weighted.p_win) if analysis.weighted else "n/a", f"about {analysis.effective_n:.0f} effective days; unweighted {_pct(analysis.unweighted.p_win) if analysis.unweighted else 'n/a'}"),
        _card("Random-walk benchmark", _pct(analysis.random_walk_p), "same volatility, no memory"),
        _card("Still open at the end", _pct(head.unresolved_share), f"of trades reached neither level in {analysis.horizon} days"),
        _card("Widest stop allowed", f"{_points(analysis.max_stop)} pts", f"{_money(spec.risk_trade)} over {spec.lots:g} lots"),
    ])


def build_fan_figure(analysis: Analysis) -> go.Figure:
    """Where replayed trades stood day by day (10th, 50th, 90th percentile of profit) with the headline stop and target."""
    head = analysis.headline
    days = list(range(1, analysis.horizon + 1))
    figure = go.Figure()
    for key, name, dash in (("p90", "90th percentile", "dot"), ("p50", "Median", "solid"), ("p10", "10th percentile", "dot")):
        figure.add_trace(go.Scatter(x=days, y=analysis.fan[key], mode="lines", name=name, line={"color": FAN_COLOR, "dash": dash, "width": 2}, connectgaps=False))
    figure.add_hline(y=head.target, line_color=GOOD, line_dash="dash", annotation_text=f"target +{_points(head.target)}")
    figure.add_hline(y=-head.stop, line_color=BAD, line_dash="dash", annotation_text=f"stop -{_points(head.stop)}")
    figure.update_layout(**_layout(
        title={"text": "Replayed trades: profit in points by day held", "font": {"size": 14}},
        xaxis={"title": "Trading days held", "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis={"title": "Profit (points)", "gridcolor": COLORS["BORDER_COLOR"], "zerolinecolor": COLORS["TEXT_SECONDARY"]},
        margin={"l": 70, "r": 30, "t": 50, "b": 50},
    ))
    return figure


def portfolio_panel(effect: PortfolioEffect | None) -> html.Div:
    if effect is None:
        return html.Div(PLACEHOLDER, style=_MUTED)
    if effect.message and not effect.var_after:
        return html.Div(effect.message, style=_MUTED)
    rows = []
    for confidence in sorted(effect.var_before):
        before, after = effect.var_before[confidence], effect.var_after.get(confidence)
        if after is None:
            continue
        change = after - before
        rows.append(html.Div([
            html.Span(f"1-day VaR {confidence:.0%}: ", style=_MUTED),
            html.Span(f"{_money(before)} to {_money(after)} ", style={"fontWeight": "bold"}),
            html.Span(f"({'+' if change >= 0 else '-'}{_money(abs(change))}, {'adds to' if change > 0 else 'reduces'} the risk you hold)",
                      style={"color": BAD if change > 0 else GOOD}),
        ], style={"marginBottom": "4px"}))
    corr = effect.correlation
    if corr is None:
        corr_line = "Correlation with the portfolio: not enough shared history."
    else:
        word = "moves with" if corr > 0.3 else "moves against" if corr < -0.3 else "is mostly unrelated to"
        corr_line = f"Daily P&L correlation with your open positions: {corr:+.2f} (this trade {word} the book), over {effect.observations} days."
    return html.Div([*rows, html.Div(corr_line, style={"marginTop": "6px"}),
                     html.Div(f"Against {effect.open_structures} open structure(s), historical simulation.", style=_MUTED)])


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _field(label: str, component, md: int) -> dbc.Col:
    return dbc.Col([html.Div(label, style=_LABEL), component], md=md, className="mb-2")


def _number_input(component_id: str, value=None, **kwargs) -> dbc.Input:
    return dbc.Input(id=component_id, type="number", value=value, **kwargs)


DEFAULT_TEMPLATE = "spread"


def leg_rows(rows: list[dict], removable: bool) -> list:
    """One row per leg: symbol and ratio (what was typed is kept by the callback), and a remove button for a Custom structure."""
    out = []
    for index, row in enumerate(rows):
        out.append(dbc.Row([
            dbc.Col(html.Div(f"Leg {index + 1}", style={"color": COLORS["TEXT_PRIMARY"], "paddingTop": "8px"}), width=2, md=1),
            dbc.Col(dbc.Input(id={"type": "ta-symbol", "index": index}, value=row.get("symbol"), debounce=True,
                              placeholder="e.g. BRNJ27 or BRNJ27-K27"), width=7, md=5),
            dbc.Col(dbc.Input(id={"type": "ta-ratio", "index": index}, type="number", step=1, value=row.get("ratio"), placeholder="Ratio"), width=3, md=2),
            dbc.Col(dbc.Button("✕", id={"type": "ta-remove", "index": index}, color="danger", outline=True, size="sm") if removable else None, width=12, md=1),
        ], className="mb-2"))
    return out


def trade_analyzer_layout() -> html.Div:
    repository = container.repository
    risk_trade = load_risk_per_trade(repository) if repository is not None else 5000.0
    form = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([html.Div("Structure type", style=_LABEL), dcc.Dropdown(
                id="ta-template", options=[{"label": f"{spec['icon']} {spec['label']}", "value": key} for key, spec in STRUCTURE_TEMPLATES.items()],
                value=DEFAULT_TEMPLATE, clearable=False, className="builder-dropdown")], md=3),
        ], className="mb-3"),
        html.Div("Legs", style=_LABEL),
        html.Div(id="ta-leg-rows"),
        dbc.Button("+ Add leg", id="ta-add-leg", color="secondary", outline=True, size="sm", className="mb-2"),
        html.Div("A leg can itself be a spread or a fly (e.g. BRNJ27-K27), as when building a structure. Buy means buying the legs as entered.",
                 style={**_MUTED, "marginBottom": "14px"}),
        dbc.Row([
            _field("Side", dcc.Dropdown(id="ta-side", options=SIDE_OPTIONS, value=1, clearable=False), 2),
            _field("Lots", _number_input("ta-lots", 5, min=1, step=1), 2),
            _field("Entry price (blank = last settlement)", _number_input("ta-entry", None, step="any"), 2),
            _field("Risk per trade ($)", _number_input("ta-risk-trade", risk_trade, min=1, step="any"), 2),
        ]),
        dbc.Row([
            _field(f"Days to give the trade (1-{MAX_HORIZON})", _number_input("ta-horizon", DEFAULT_HORIZON, min=1, max=MAX_HORIZON, step=1), 3),
            _field("Comparable days: +/- trading days to expiry", _number_input("ta-window", DEFAULT_WINDOW, min=0, max=30, step=1), 3),
            _field("Your stop (points, optional)", _number_input("ta-stop", None, min=0, step="any"), 2),
            _field("Your target (points, optional)", _number_input("ta-target", None, min=0, step="any"), 2),
            dbc.Col([
                html.Div(" ", style=_LABEL),
                dbc.Button("Analyze", id="ta-run", color="primary", className="me-2"),
                dbc.Button("Save risk per trade", id="ta-save", color="secondary", outline=True),
            ], md=2, className="mb-2"),
        ]),
        html.Div(id="ta-status", style={**_MUTED, "minHeight": "20px"}),
    ]), style=_CARD, className="mb-3")
    return html.Div([
        html.H3("🎯 Trade Analyzer", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "4px"}),
        html.Div(
            "Enter a structure and a trade. The same structure of earlier years is replayed from days at the same distance from "
            "expiry, to find how often the target is met before the stop, and which stop and target pairs have worked best "
            "within your risk per trade. Daily settlements only.",
            style={**_MUTED, "marginBottom": "12px"},
        ),
        form,
        dcc.Loading(html.Div([
            html.Div(id="ta-verdict", style={"color": COLORS["TEXT_PRIMARY"], "fontSize": "15px", "marginBottom": "12px"}),
            html.Div(id="ta-cards"),
            dbc.Card(dbc.CardBody([html.H5("Stops and targets", style={"color": COLORS["TEXT_PRIMARY"]}),
                                   html.Div("Best expected result first. Chance of winning counts only trades that reached the stop or the target; "
                                            "'still open' trades are closed at their last price in the expected result.", style=_MUTED),
                                   html.Div(id="ta-grid")]), style=_CARD, className="mb-3"),
            dbc.Card(dbc.CardBody(dcc.Graph(id="ta-fan", figure=empty_figure(PLACEHOLDER), config={"displayModeBar": False}, style={"height": "380px"})),
                     style=_CARD, className="mb-3"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Price context today", style={"color": COLORS["TEXT_PRIMARY"]}),
                                               html.Div("Replayed days with a context like today's count for more (the regime-weighted figure).", style=_MUTED),
                                               html.Div(id="ta-context")]), style=_CARD, className="mb-3"), md=5),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Effect on your portfolio", style={"color": COLORS["TEXT_PRIMARY"]}),
                                               html.Div(id="ta-portfolio", children=portfolio_panel(None))]), style=_CARD, className="mb-3"), md=7),
            ]),
            dbc.Card(dbc.CardBody([html.H5("Earlier years replayed", style={"color": COLORS["TEXT_PRIMARY"]}), html.Div(id="ta-years")]),
                     style=_CARD, className="mb-3"),
            html.Div(id="ta-notes", style=_MUTED),
        ])),
    ], style={"padding": "24px"})

