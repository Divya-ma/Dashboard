"""Curve Compare page (route /curve-compare): live curve against the previous settlement.

Pick a product, how many months out (1-36), the curve type (outright / spread / fly / dfly) and,
for a dfly, how its price is built. The chart shows the live curve (VWAP of best bid / ask) and the
previous settlement; points built from other quotes are drawn as hollow diamonds, and a curve stops
where nothing can be priced. All colours come from COLORS.
"""

from datetime import datetime, timezone

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from core.curve_calendar import FAMILIES, FAMILY_LABELS, OUTRIGHT, PRODUCTS
from core.curve_compare import (
    DEFAULT_DFLY_METHOD, DFLY_METHODS, MAX_MONTHS, QUOTE_LAG_WARNING_SECONDS, SOURCE_DERIVED, SOURCE_FAIR, SOURCE_QUOTE,
    CompareView,
)
from ui.layouts.curve_tab import _CARD, _LABEL, _MUTED, _dropdown, _fmt, _layout, _table, empty_figure
from ui.layouts.shell import COLORS

REFRESH_MS = 10_000
OLD_SECONDS = 3600
DEFAULT_MONTHS = 32
PLACEHOLDER = "Waiting for quotes..."

LIVE_COLOR, PREV_COLOR = "#4da3ff", "#9aa0a6"
SOURCE_MARKERS = {
    SOURCE_QUOTE: ("Live (quote VWAP)", {"symbol": "circle", "size": 8, "color": LIVE_COLOR}),
    SOURCE_FAIR: ("Live (fair value)", {"symbol": "square", "size": 8, "color": COLORS["ACCENT_YELLOW"]}),
    SOURCE_DERIVED: (
        "Live (derived)",
        {"symbol": "diamond-open", "size": 10, "color": LIVE_COLOR, "line": {"width": 2, "color": LIVE_COLOR}},
    ),
}


def _age(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f}m" if seconds < 5400 else f"{seconds / 3600:.1f}h"


def _hover(row) -> str:
    point = row.point
    lines = [f"<b>{point.label}</b>"]
    lines.append(f"Live {_fmt(point.value)}" + (f" ({point.source})" if point.source else ""))
    if point.how:
        lines.append(f"= {point.how}")
    if point.age_seconds is not None:
        lines.append(f"price age {_age(point.age_seconds)}")
    if row.prev is not None:
        lines.append(f"Prev settle {_fmt(row.prev)}")
    if row.change is not None:
        lines.append(f"Change {_fmt(row.change, signed=True)}")
    return "<br>".join(lines)


def build_compare_figure(view: CompareView | None) -> go.Figure:
    """Live line (marker per source) over the previous-settlement line."""
    if view is None or not view.rows:
        return empty_figure(PLACEHOLDER)
    rows = view.rows
    if not any(r.point.value is not None or r.prev is not None for r in rows):
        return empty_figure("No prices yet for this curve")
    x = [r.point.axis for r in rows]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=x, y=[r.prev for r in rows], mode="lines+markers", name="Prev settlement",
        line={"color": PREV_COLOR, "width": 2, "dash": "dash"}, marker={"size": 5, "color": PREV_COLOR},
        connectgaps=False, hovertext=[_hover(r) for r in rows], hoverinfo="text",
    ))
    figure.add_trace(go.Scatter(  # the connecting line, no markers: markers are drawn per source below
        x=x, y=[r.point.value for r in rows], mode="lines", name="Live", showlegend=False,
        line={"color": LIVE_COLOR, "width": 2}, connectgaps=False, hoverinfo="skip",
    ))
    for source, (name, marker) in SOURCE_MARKERS.items():
        ys = [r.point.value if r.point.source == source else None for r in rows]
        if any(y is not None for y in ys):
            # faded marker = the price is more than an hour old (a quiet back month)
            opacity = [1.0 if (r.point.age_seconds or 0) <= OLD_SECONDS else 0.4 for r in rows]
            figure.add_trace(go.Scatter(
                x=x, y=ys, mode="markers", name=name, marker={**marker, "opacity": opacity},
                hovertext=[_hover(r) for r in rows], hoverinfo="text",
            ))
    title = f"{PRODUCTS[view.product].name} {FAMILY_LABELS[view.family]}, first {view.months} contracts"
    figure.update_layout(**_layout(
        title={"text": title, "font": {"size": 14}},
        xaxis={"tickangle": -45, "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis={"title": "Price", "gridcolor": COLORS["BORDER_COLOR"], "zerolinecolor": COLORS["BORDER_COLOR"]},
        hovermode="closest",
        uirevision=f"{view.product}|{view.family}|{view.months}|{view.method}",  # keep the zoom across refreshes
    ))
    return figure


def build_change_figure(view: CompareView | None) -> go.Figure:
    """Bar per contract: live minus previous settlement."""
    if view is None or not any(r.change is not None for r in view.rows):
        return empty_figure("")
    changes = [r.change for r in view.rows]
    colours = [COLORS["ACCENT_GREEN"] if (c or 0) >= 0 else COLORS["ACCENT_RED"] for c in changes]
    figure = go.Figure(go.Bar(
        x=[r.point.axis for r in view.rows], y=changes, marker={"color": colours},
        hovertext=[_hover(r) for r in view.rows], hoverinfo="text",
    ))
    figure.update_layout(**_layout(
        title={"text": "Live minus previous settlement", "font": {"size": 13}},
        margin={"l": 70, "r": 30, "t": 40, "b": 80},
        xaxis={"tickangle": -45}, yaxis={"gridcolor": COLORS["BORDER_COLOR"], "zerolinecolor": COLORS["TEXT_SECONDARY"]},
        uirevision=f"{view.product}|{view.family}|{view.months}|{view.method}",
    ))
    return figure


def build_status(view: CompareView | None) -> html.Div:
    if view is None:
        return html.Div(PLACEHOLDER)
    parts = []
    if view.last_quote_at is not None:
        parts.append(f"Latest quote {datetime.fromtimestamp(view.last_quote_at, tz=timezone.utc):%H:%M:%S} UTC")
    parts.append(f"{view.symbols_priced} of {view.symbols_requested} symbols priced ({view.symbols_quoted} with a quote)")
    parts.append(f"points: {view.count(SOURCE_QUOTE)} quote, {view.count(SOURCE_FAIR)} fair value, "
                 f"{view.count(SOURCE_DERIVED)} derived, {view.count(None)} empty")
    if view.symbols_wide:
        parts.append(f"{view.symbols_wide} quote(s) ignored as too wide to trust (fair value used instead)")
    old = view.count_older_than(OLD_SECONDS)
    if old:
        parts.append(f"{old} point(s) rest on prices over an hour old (faded)")
    if view.settlement_date is not None:
        parts.append(f"Prev settlement of {view.settlement_date.isoformat()}")
    children = [html.Span(" · ".join(parts))]
    lag = view.quote_lag_seconds
    if lag is not None and lag > QUOTE_LAG_WARNING_SECONDS:
        children.append(html.Div(
            f"⚠️ The newest TAQ quote is {_age(lag)} old (TAQ can run behind). Prices that rest on older quotes are not "
            "mixed with newer fair values; those points use a fair-value basis or stay empty until fresher quotes arrive.",
            style={"color": COLORS["ACCENT_YELLOW"], "marginTop": "4px"},
        ))
    elif lag is None and view.symbols_requested:
        children.append(html.Div(
            "⚠️ No TAQ quote yet for these contracts (TAQ can run behind); the curve uses fair values for now.",
            style={"color": COLORS["ACCENT_YELLOW"], "marginTop": "4px"},
        ))
    for message, colour in (
        (view.quote_error, "ACCENT_RED"), (view.fair_error, "ACCENT_RED"), (view.settlement_message, "ACCENT_YELLOW"),
    ):
        if message:
            children.append(html.Div(f"⚠️ {message}", style={"color": COLORS[colour], "marginTop": "4px"}))
    return html.Div(children)


def table_rows(view: CompareView | None) -> list[dict]:
    if view is None:
        return []
    return [
        {
            "position": r.point.position, "contract": r.point.label,
            "live": _fmt(r.point.value, 3), "source": r.point.source or "", "age": _age(r.point.age_seconds),
            "prev": _fmt(r.prev, 3) if r.prev is not None else "", "change": _fmt(r.change, 3, signed=True) if r.change is not None else "",
            "how": r.point.how,
        }
        for r in view.rows
    ]


TABLE_COLUMNS = [
    {"name": "#", "id": "position"}, {"name": "Contract", "id": "contract"}, {"name": "Live", "id": "live"},
    {"name": "Source", "id": "source"}, {"name": "Price age", "id": "age"}, {"name": "Prev settle", "id": "prev"},
    {"name": "Change", "id": "change"}, {"name": "Built as", "id": "how"},
]


def build_table(view: CompareView | None) -> html.Div:
    return html.Div(_table("compare-table", TABLE_COLUMNS, table_rows(view), page_size=20))


def curve_compare_layout() -> html.Div:
    product_options = [{"label": p.name, "value": c} for c, p in PRODUCTS.items()]
    method_options = [{"label": label, "value": key} for key, label in DFLY_METHODS.items()]
    return html.Div([
        dcc.Interval(id="compare-refresh", interval=REFRESH_MS, n_intervals=0),
        html.H3("📉 Curve Compare", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "4px"}),
        html.Div(id="compare-status", style={**_MUTED, "marginBottom": "12px", "minHeight": "20px"}),
        dbc.Row([
            dbc.Col([html.Div("Product", style=_LABEL), _dropdown("compare-product", product_options, "CL")], md=2),
            dbc.Col([html.Div(f"Months (1-{MAX_MONTHS})", style=_LABEL),
                     dcc.Input(id="compare-months", type="number", min=1, max=MAX_MONTHS, step=1, value=DEFAULT_MONTHS,
                               debounce=True, className="form-control")], md=2),
            dbc.Col([html.Div("Curve", style=_LABEL),
                     _dropdown("compare-family", [{"label": FAMILY_LABELS[f], "value": f} for f in FAMILIES], OUTRIGHT)], md=3),
            dbc.Col([html.Div("D-fly built from", style=_LABEL),
                     _dropdown("compare-method", method_options, DEFAULT_DFLY_METHOD)], md=5),
        ], className="mb-3"),
        dbc.Card(dbc.CardBody([
            dcc.Graph(id="compare-graph", figure=empty_figure(PLACEHOLDER), config={"displayModeBar": False}, style={"height": "460px"}),
            dcc.Graph(id="compare-change", figure=empty_figure(""), config={"displayModeBar": False}, style={"height": "230px"}),
        ]), style=_CARD, className="mb-3"),
        html.Div(id="compare-table-wrap"),
        html.Div(
            "Live price = VWAP of the best bid and ask, weighted by the opposite side's size; where a contract has "
            "no quote its fair value is used. Hollow diamonds are built from other prices. Faded markers rest on a "
            "price over an hour old (quiet back months update rarely, see the Price age column); gaps mean nothing "
            "could be priced.",
            style={**_MUTED, "marginTop": "10px"},
        ),
    ])
