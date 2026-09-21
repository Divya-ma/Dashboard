"""Trade Idea Analyzer tab: idea input panel (left) and analysis output (right).

The layout is static; ui/callbacks/idea_callbacks.py validates the inputs and fills
the output column. All colours come from COLORS.
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from core.trade_idea import (
    DOLLARS_PER_POINT_PER_LOT,
    IdeaAnalysis,
    correlation_label,
)
from ui.layouts.shell import COLORS

LOOKBACK_OPTIONS = [{"label": f"{days}d", "value": str(days)} for days in (10, 20, 30, 60, 90)]
DEFAULT_LOOKBACK = "30"
PLACEHOLDER_TEXT = "Define a trade idea and click Analyze"
MAX_LEGS = 3

_CARD_STYLE = {"backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}"}
_HEADING = {"color": COLORS["TEXT_PRIMARY"], "marginBottom": "12px"}
_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_LABEL = {**_MUTED, "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "6px"}
_HIDDEN = {"display": "none"}

_CORRELATION_COLORS = {
    "high": COLORS["ACCENT_RED"],
    "moderate": COLORS["ACCENT_YELLOW"],
    "low": COLORS["ACCENT_GREEN"],
    "n/a": COLORS["TEXT_SECONDARY"],
}
_CORRELATION_CAPTIONS = {"high": "High", "moderate": "Moderate", "low": "Low", "n/a": "n/a"}


# ----------------------------------------------------------------------
# Output builders
# ----------------------------------------------------------------------


def _pts(value: float) -> str:
    return f"{value:,.2f}"


def _metric(label: str, value: str, accent: str = COLORS["ACCENT_BLUE"], columns: dict | None = None) -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(label, style=_LABEL),
                    html.Div(value, style={"fontSize": "22px", "fontWeight": "bold", "color": COLORS["TEXT_PRIMARY"]}),
                ]
            ),
            style={**_CARD_STYLE, "borderTop": f"3px solid {accent}", "height": "100%"},
        ),
        **(columns or {"md": 6, "xl": 4}),
        className="mb-3",
    )


def _section(title: str, body) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([html.H5(title, style=_HEADING), *([body] if not isinstance(body, list) else body)]),
        style=_CARD_STYLE,
        className="mb-3",
    )


def _table(headers: list[str], rows: list[list]) -> html.Table:
    cell = {"padding": "6px 12px", "borderBottom": f"1px solid {COLORS['BORDER_COLOR']}"}
    head = html.Thead(html.Tr([html.Th(h, style={**cell, **_MUTED, "textAlign": "left"}) for h in headers]))
    body = html.Tbody([html.Tr([html.Td(c, style=cell) for c in row]) for row in rows])
    return html.Table([head, body], style={"width": "100%", "color": COLORS["TEXT_PRIMARY"]})


def build_risk_reward(analysis: IdeaAnalysis) -> dbc.Row:
    return dbc.Row(
        [
            _metric("Risk (pts)", _pts(analysis.risk), COLORS["ACCENT_RED"]),
            _metric("Reward (pts)", _pts(analysis.reward), COLORS["ACCENT_GREEN"]),
            _metric("R:R Ratio", f"{analysis.rr_ratio:.1f} : 1", COLORS["ACCENT_YELLOW"]),
            _metric("Lots", f"{analysis.idea.total_lots}"),
            _metric("Est. $ Risk", f"${analysis.dollar_risk:,.0f}", COLORS["ACCENT_RED"]),
            _metric("Est. $ Reward", f"${analysis.dollar_reward:,.0f}", COLORS["ACCENT_GREEN"]),
        ]
    )


def build_historical_context(analysis: IdeaAnalysis) -> list:
    stats = analysis.stats
    cards = dbc.Row(
        [
            _metric("Mean daily move (pts)", f"{stats['mean']:+.3f}"),
            _metric("Std dev of daily moves (pts)", _pts(stats["std"])),
            _metric("Max single-day gain (pts)", f"{stats['max_gain']:+.2f}", COLORS["ACCENT_GREEN"]),
            _metric("Max single-day loss (pts)", f"{stats['max_loss']:+.2f}", COLORS["ACCENT_RED"]),
            _metric("Days beyond risk distance", f"{stats['pct_beyond_risk']:.1f}%", COLORS["ACCENT_RED"]),
            _metric("Days beyond reward distance", f"{stats['pct_beyond_reward']:.1f}%", COLORS["ACCENT_GREEN"]),
        ]
    )
    note = f"{analysis.observations} daily observations, per 1 structure unit."
    if analysis.observations < analysis.requested:
        note += f" Only {analysis.observations} common days available (requested {analysis.requested}d)."
    return [cards, html.Div(note, style=_MUTED)]


def build_correlation_section(analysis: IdeaAnalysis, has_open_structures: bool) -> list:
    """Structure Name | Correlation, red above 0.7, amber 0.4-0.7, green below 0.4."""
    if not has_open_structures:
        return [html.Div("No open positions to correlate against", style=_MUTED)]
    parts = []
    if analysis.correlations:
        rows = []
        for item in analysis.correlations:
            label = correlation_label(item["correlation"])
            color = _CORRELATION_COLORS[label]
            shown = "n/a" if item["correlation"] is None else f"{item['correlation']:+.2f}"
            rows.append(
                [item["name"], html.Span(shown, style={"color": color, "fontWeight": "bold"}),
                 html.Span(_CORRELATION_CAPTIONS[label], style={"color": color})]
            )
        parts.append(_table(["Structure Name", "Correlation", ""], rows))
    if analysis.skipped_structures:
        parts.append(
            html.Div(
                "Skipped (missing data): " + "; ".join(f"{name} — {why}" for name, why in analysis.skipped_structures.items()),
                style={"color": COLORS["ACCENT_YELLOW"], "fontSize": "13px", "marginTop": "10px"},
            )
        )
    return parts or [html.Div("No open structures had enough data to correlate against.", style=_MUTED)]


def build_pnl_chart(analysis: IdeaAnalysis, lookback_days: int) -> go.Figure:
    """Cumulative hypothetical PnL per structure unit, with dashed lines at 0, +reward and -risk."""
    series = analysis.unit_series
    figure = go.Figure(
        go.Scatter(
            x=series.index, y=series.cumsum(), mode="lines",
            line={"color": COLORS["ACCENT_GREEN"], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f} pts<extra></extra>",
        )
    )
    for level, color, text in (
        (0, COLORS["TEXT_SECONDARY"], "0"),
        (analysis.reward, COLORS["ACCENT_GREEN"], f"Target +{analysis.reward:.2f}"),
        (-analysis.risk, COLORS["ACCENT_RED"], f"Stop −{analysis.risk:.2f}"),
    ):
        figure.add_hline(y=level, line_dash="dash", line_color=color, annotation_text=text,
                         annotation_position="top left", annotation_font_color=color)
    figure.update_layout(
        title={"text": f"Hypothetical Historical PnL ({lookback_days}d)", "font": {"size": 18}},
        paper_bgcolor=COLORS["CARD_BG"], plot_bgcolor=COLORS["CARD_BG"], font={"color": COLORS["TEXT_PRIMARY"]},
        margin={"l": 60, "r": 30, "t": 60, "b": 50}, showlegend=False,
        xaxis={"title": "Date", "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis={"title": "Cumulative PnL (pts)", "gridcolor": COLORS["BORDER_COLOR"], "zeroline": False},
    )
    return figure


def build_messages(errors: list[str], warnings: list[str]) -> list:
    """Blocking errors in red, non-blocking warnings in amber."""
    return [dbc.Alert(f"⛔ {message}", color="danger", className="mb-2") for message in errors] + [
        dbc.Alert(f"⚠️ {message}", color="warning", className="mb-2") for message in warnings
    ]


def build_output(analysis: IdeaAnalysis, warnings: list[str], lookback_days: int, has_open_structures: bool) -> html.Div:
    """The right column after Analyze: warnings, then sections A-D (or the errors)."""
    if analysis.errors:
        return html.Div(build_messages(analysis.errors, warnings))
    return html.Div(
        [
            *build_messages([], warnings),
            _section("A. Risk / Reward Summary", [
                build_risk_reward(analysis),
                html.Div(f"Est. $ uses ${DOLLARS_PER_POINT_PER_LOT:g} per point per lot × total lots.", style=_MUTED),
            ]),
            _section("B. Historical Context", build_historical_context(analysis)),
            _section("C. Correlation with Portfolio", build_correlation_section(analysis, has_open_structures)),
            _section("D. Price Chart", dcc.Graph(figure=build_pnl_chart(analysis, lookback_days), config={"displayModeBar": False}, style={"height": "380px"})),
        ]
    )


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _leg_row(number: int) -> html.Div:
    return html.Div(
        [
            html.Div(f"Leg {number}", style=_LABEL),
            dbc.Input(id=f"idea-leg-{number}-symbol", placeholder="e.g. CLZ26 or CLZ26-F27", className="mb-2"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.RadioItems(
                            id=f"idea-leg-{number}-direction",
                            options=[{"label": "Buy", "value": "buy"}, {"label": "Sell", "value": "sell"}],
                            value="buy",
                            className="btn-group",
                            inputClassName="btn-check",
                            labelClassName="btn btn-outline-secondary btn-sm",
                            labelCheckedClassName="active",
                        ),
                        width=6,
                    ),
                    dbc.Col(dbc.Input(id=f"idea-leg-{number}-lots", type="number", min=1, step=1, value=1, placeholder="Lots"), width=6),
                ]
            ),
        ],
        id=f"idea-leg-row-{number}",
        style={"marginBottom": "16px", **({} if number == 1 else _HIDDEN)},
    )


def _number_field(label: str, component_id: str, placeholder: str) -> dbc.Col:
    return dbc.Col(
        [html.Div(label, style=_LABEL), dbc.Input(id=component_id, type="number", step=0.01, placeholder=placeholder)],
        md=4, className="mb-3",
    )


def _input_panel() -> dbc.Card:
    structure_type = dbc.RadioItems(
        id="idea-structure-type",
        options=[{"label": "Outright", "value": "outright"}, {"label": "Spread", "value": "spread"}, {"label": "Fly", "value": "fly"}],
        value="outright",
        className="btn-group mb-3",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-secondary",
        labelCheckedClassName="active",
    )
    body = [
        html.H5("Define Trade Idea", style=_HEADING),
        html.Div("Structure type", style=_LABEL),
        structure_type,
        *[_leg_row(number) for number in range(1, MAX_LEGS + 1)],
        html.Hr(style={"borderColor": COLORS["BORDER_COLOR"]}),
        dbc.Row(
            [
                _number_field("Entry Price (pts)", "idea-entry", "Structure price"),
                _number_field("Stop (pts)", "idea-stop", "Stop level"),
                _number_field("Target (pts)", "idea-target", "Target level"),
            ]
        ),
        html.Div("Stop and target are price levels, not distances.", style={**_MUTED, "marginBottom": "12px"}),
        html.Div("Lookback for analysis", style=_LABEL),
        dbc.Select(id="idea-lookback", options=LOOKBACK_OPTIONS, value=DEFAULT_LOOKBACK, className="mb-3"),
        dbc.Button(
            "Analyze", id="idea-analyze-btn", className="w-100",
            style={"backgroundColor": COLORS["ACCENT_GREEN"], "borderColor": COLORS["ACCENT_GREEN"], "color": COLORS["DARK_BG"], "fontWeight": "bold"},
        ),
    ]
    return dbc.Card(dbc.CardBody(body), style=_CARD_STYLE)


def trade_analyzer_layout() -> html.Div:
    """Idea input on the left, analysis output on the right."""
    return html.Div(
        [
            dcc.Store(id="idea-open-structures", data=[]),
            html.H3("🎯 Trade Idea Analyzer", style={**_HEADING, "marginBottom": "16px"}),
            dbc.Row(
                [
                    dbc.Col(_input_panel(), lg=4, className="mb-3"),
                    dbc.Col(
                        html.Div(
                            html.Div(PLACEHOLDER_TEXT, style={**_MUTED, "fontSize": "16px", "textAlign": "center", "padding": "120px 0"}),
                            id="idea-output",
                            style={"minHeight": "300px"},
                        ),
                        lg=8,
                        className="mb-3",
                    ),
                ]
            ),
        ]
    )
