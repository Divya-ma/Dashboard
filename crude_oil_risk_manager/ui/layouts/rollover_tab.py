"""Strategy Rollover page (route /strategy-rollover): how a generic moved before the front expiry and after it became the front.

Pick a product, the generic (2 = the second month), how many expiries to run through and how many earlier expiries to compare; one
line per expiry is drawn against trading days from the front expiry, with their average. 'Zero base' subtracts each line's price on
the expiry day so every line meets at 0 there. CL-BZ is different (core.rollover): its line is the front box through the CL and BZ
expiries of a calendar month. All colours come from COLORS.
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from core.curve_calendar import EXTRA_PRODUCTS, FAMILIES, FAMILY_LABELS, OUTRIGHT, PRODUCTS, product_spec
from core.rollover import (
    DEFAULT_BEFORE, DEFAULT_COMPARISONS, DEFAULT_EXPIRIES, DEFAULT_GENERIC, MAX_BEFORE, MAX_COMPARISONS, MAX_EXPIRIES, MAX_GENERIC,
    RolloverResult,
)
from ui.layouts.curve_tab import _CARD, _LABEL, _MUTED, _dropdown, _layout, empty_figure
from ui.layouts.shell import COLORS

ROLLOVER_PRODUCTS = {**PRODUCTS, **EXTRA_PRODUCTS}
REFRESH_MS = 60_000
PLACEHOLDER = "Choose a product to see how its generics rolled"
AVERAGE_COLOR = "#ffd60a"
LINE_COLORS = ["#f4a261", "#2a9d8f", "#e76f51", "#4da3ff", "#c77dff", "#b388eb", "#90be6d", "#f28482", "#84a59d", "#f6bd60", "#9d8189", "#7ec8e3"]
LIVE_COLOR = "#ffffff"  # the contract still being drawn stands out from the finished ones


def build_status(result: RolloverResult | None, error: str | None = None) -> html.Div:
    if error:
        return html.Div(error, style={"color": COLORS["ACCENT_RED"]})
    if result is None:
        return html.Div("Waiting for data...")
    done = sum(1 for line in result.lines if not line.live)
    parts = [f"{len(result.lines)} expiries drawn", f"{done} finished (in the average)"]
    children = [html.Span(" · ".join(parts))]
    for note in result.notes:
        children.append(html.Div(f"⚠️ {note}", style={"color": COLORS["ACCENT_YELLOW"], "marginTop": "4px"}))
    return html.Div(children)


def build_rollover_figure(result: RolloverResult | None) -> go.Figure:
    if result is None:
        return empty_figure(PLACEHOLDER)
    if not result.lines:
        return empty_figure("No settlement history for this selection yet")
    options, figure = result.options, go.Figure()
    shades = iter(LINE_COLORS * (len(result.lines) // len(LINE_COLORS) + 1))
    lives = 0
    for line in result.lines:
        colour = LIVE_COLOR if line.live else next(shades)
        dash = "dash" if line.live and lives else "solid"  # a second live line is told apart by its dashes
        lives += line.live
        frame = line.frame
        figure.add_trace(go.Scatter(
            x=frame.index, y=frame["price"], mode="lines", name=f"{line.label}{' (live)' if line.live else ''}",
            line={"color": colour, "width": 3.4 if line.live else 1.5, "dash": dash}, opacity=1.0 if line.live else 0.85,
            customdata=list(zip(frame["date"].dt.strftime("%Y-%m-%d"), frame["pair"])),
            hovertemplate=f"<b>{line.label}</b> %{{customdata[0]}}<br>%{{customdata[1]}}<br>day %{{x}}: %{{y:.3f}}<extra></extra>",
        ))
        if line.marks:
            figure.add_trace(go.Scatter(
                x=[m.x for m in line.marks], y=[m.y for m in line.marks], mode="markers", showlegend=False,
                marker={"symbol": "diamond", "size": 9, "color": colour, "line": {"width": 1, "color": COLORS["TEXT_PRIMARY"]}},
                text=[f"{line.label}: {m.text}" for m in line.marks], hovertemplate="%{text}<extra></extra>",
            ))
    if len(result.average):
        figure.add_trace(go.Scatter(
            x=result.average.index, y=result.average, mode="lines", name="Average (finished)",
            line={"color": AVERAGE_COLOR, "width": 2.4, "dash": "dot"}, hovertemplate="average %{y:.3f}<extra></extra>",
        ))
    for guide in result.guides:
        figure.add_vline(x=guide.x, line={"color": COLORS["TEXT_SECONDARY"], "width": 1, "dash": "dot" if guide.x else "solid"})
        figure.add_annotation(x=guide.x, y=0, yref="paper", text=guide.text, showarrow=False, yanchor="bottom", font={"size": 11})
    level = "Change since the expiry day" if options.zero_base else "Settlement"
    what = "front box" if result.box else f"generic {options.generic} {FAMILY_LABELS[options.family].lower()}"
    figure.update_layout(**_layout(
        title={"text": f"{product_spec(result.product).name}: {what} around expiry", "font": {"size": 14}},
        xaxis={"title": "Trading days from the front expiry", "gridcolor": COLORS["BORDER_COLOR"], "zeroline": False},
        yaxis={"title": level, "gridcolor": COLORS["BORDER_COLOR"], "zerolinecolor": COLORS["TEXT_SECONDARY"]},
        legend={"orientation": "h", "yanchor": "top", "y": -0.15},
        margin={"l": 70, "r": 30, "t": 50, "b": 110},
        hovermode="closest",
        uirevision=f"{result.product}|{options.key()}",
    ))
    return figure


def rollover_layout() -> html.Div:
    product_options = [{"label": p.name, "value": c} for c, p in ROLLOVER_PRODUCTS.items()]

    def number(component_id, value, low, high):
        return dcc.Input(id=component_id, type="number", min=low, max=high, step=1, value=value, debounce=True, className="form-control")

    def field(label, control, width):
        return dbc.Col([html.Div(label, style=_LABEL), control], md=width)

    return html.Div([
        dcc.Interval(id="rollover-refresh", interval=REFRESH_MS, n_intervals=0),
        html.H3("🔁 Strategy Rollover", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "4px"}),
        html.Div(id="rollover-status", style={**_MUTED, "marginBottom": "12px", "minHeight": "20px"}),
        dbc.Row([
            field("Product", _dropdown("rollover-product", product_options, "BRN"), 2),
            field("Curve", _dropdown("rollover-family", [{"label": FAMILY_LABELS[f], "value": f} for f in FAMILIES], OUTRIGHT), 2),
            field(f"Generic (1-{MAX_GENERIC})", number("rollover-generic", DEFAULT_GENERIC, 1, MAX_GENERIC), 2),
            field(f"Expiries (1-{MAX_EXPIRIES})", number("rollover-expiries", DEFAULT_EXPIRIES, 1, MAX_EXPIRIES), 2),
            field(f"Comparisons (1-{MAX_COMPARISONS})", number("rollover-comparisons", DEFAULT_COMPARISONS, 1, MAX_COMPARISONS), 2),
            field(f"Days before (1-{MAX_BEFORE})", number("rollover-before", DEFAULT_BEFORE, 1, MAX_BEFORE), 2),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Switch(id="rollover-zero", label="Zero base (subtract the price on the expiry day)", value=False), md=5),
        ], className="mb-3"),
        dbc.Card(dbc.CardBody(
            dcc.Graph(id="rollover-graph", figure=empty_figure(PLACEHOLDER), config={"displayModeBar": False}, style={"height": "600px"}),
        ), style=_CARD, className="mb-3"),
        html.Div(
            "A line follows one contract (or spread, fly or dfly) from the chosen number of trading days before the front expiry, through "
            "the expiry (day 0), when it moves up one generic, to the expiry that many months later; a generic can pass at most generic − 1 "
            "expiries. Each expiry has a vertical line and diamonds mark it on every line; where the gap between two expiries differs, it is "
            "stretched to the longest. With zero base each line is 0 on every vertical line. The white line is the contract still live, "
            "the yellow dotted line the average of the finished lines. CL-BZ ignores Curve and Generic: its front box is CL month − BZ month, "
            "which changes at each leg's expiry, so a line starts on the first trading day of a month, passes the CL expiry (day 0) and "
            "the BRN expiry, and ends at the CL expiry 'Expiries' months later. Settlement prices only.",
            style={**_MUTED, "marginTop": "10px"},
        ),
    ])
