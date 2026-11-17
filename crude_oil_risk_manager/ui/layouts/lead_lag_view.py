"""Lead-Lag section of the Correlation tab (core.lead_lag / core.lead_lag_data).

Pick a product group and a 4-contract window (or "All", one contract per product); the matrix shows, for every
pair, which one's 5-minute moves line up with the other's a few bars later, and by how much. Click a cell for
the full cross-correlation-by-lag chart and a plain-English read. The layout is static; ui/callbacks/lead_lag_callbacks.py
fills it. Colours from COLORS.
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from core.lead_lag import NO_SIGN, STANDS_OUT, WORTH_A_LOOK
from core.lead_lag_data import DEFAULT_LOOKBACK_DAYS, GROUP_OPTIONS, MAX_LOOKBACK_DAYS
from ui.layouts.curve_tab import _LABEL, _MUTED, _layout, empty_figure
from ui.layouts.shell import COLORS

PLACEHOLDER_TEXT = "Pick a group and click Compute"
DEFAULT_MONTH = 1
CCF_COLOR = "#4da3ff"
READING_COLORS = {STANDS_OUT: COLORS["ACCENT_GREEN"], WORTH_A_LOOK: COLORS["ACCENT_YELLOW"], NO_SIGN: COLORS["TEXT_SECONDARY"]}


def _field(label: str, component, md: int, hint: str | None = None) -> dbc.Col:
    return dbc.Col([html.Div(label, style=_LABEL, title=hint), component], md=md, className="mb-2")


def lead_lag_section() -> html.Div:
    controls = dbc.Row(
        [
            _field("Group", dcc.Dropdown(id="ll-group", options=GROUP_OPTIONS, value=GROUP_OPTIONS[1]["value"], clearable=False), 3),
            _field("Starting month (1 = front)", dbc.Input(id="ll-month", type="number", value=DEFAULT_MONTH, min=1, max=20, step=1, debounce=True, size="sm"), 3,
                   "A single product shows this contract and the next 3; \"All\" shows this one contract of each product"),
            _field("Lookback (days)", dbc.Input(id="ll-lookback", type="number", value=DEFAULT_LOOKBACK_DAYS, min=1, max=MAX_LOOKBACK_DAYS, step=1, debounce=True, size="sm"), 3,
                   "Calendar days of 5-minute bars pulled (cached locally after the first time)"),
            dbc.Col(dbc.Button("Compute", id="ll-compute-btn", className="mb-2",
                                style={"backgroundColor": COLORS["ACCENT_GREEN"], "borderColor": COLORS["ACCENT_GREEN"], "color": COLORS["DARK_BG"], "fontWeight": "bold"}),
                    md=3, className="d-flex align-items-end"),
        ],
        className="g-2",
    )
    return html.Div([
        dcc.Store(id="ll-params"),
        html.H5("Lead-Lag (intraday)", style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "24px", "marginBottom": "4px"}),
        html.Div(
            "5-minute bar price differences, prewhitened (each series' own momentum removed first) before cross-correlating, so a "
            "cell's lag reflects one contract actually leading the other rather than either one's own autocorrelation. Positive lag "
            "on a cell means the row leads the column; negative means the column leads the row. Colour is the strength (|ρ|) at that "
            "best lag. Click a cell for the full picture.",
            style={**_MUTED, "marginBottom": "10px"},
        ),
        controls,
        dcc.Loading(html.Div(id="ll-status", style={**_MUTED, "marginBottom": "6px"}), type="dot"),
        dcc.Graph(id="ll-matrix", figure=empty_figure(PLACEHOLDER_TEXT), config={"displayModeBar": False}),
        html.Div(id="ll-detail", style={"marginTop": "14px"}),
        dcc.Graph(id="ll-ccf-chart", figure=empty_figure("Click a cell of the matrix"), config={"displayModeBar": False}, style={"height": "340px"}),
    ])


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def build_matrix_figure(lag_matrix, strength_matrix, bar_minutes: int) -> go.Figure:
    """Heatmap: colour = strength (|rho| at the best lag, 0..1); cell text = that lag in minutes, signed from the
    row's point of view (row leads column when positive)."""
    if lag_matrix is None or lag_matrix.empty:
        return empty_figure(PLACEHOLDER_TEXT)
    labels = list(lag_matrix.index)
    lag_values = lag_matrix.to_numpy(dtype=float)
    strength_values = strength_matrix.to_numpy(dtype=float)
    z = [[None if v != v else float(v) for v in row] for row in strength_values]
    text = []
    for i, row in enumerate(lag_values):
        line = []
        for j, lag in enumerate(row):
            if i == j:
                line.append("—")
            elif strength_values[i, j] != strength_values[i, j]:
                line.append("n/a")
            else:
                minutes = int(lag) * bar_minutes
                line.append(f"{minutes:+d}m" if minutes else "0m")
        text.append(line)
    figure = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, text=text, texttemplate="%{text}", textfont={"color": COLORS["DARK_BG"], "size": 13},
        zmin=0, zmax=1, colorscale=[[0.0, COLORS["TEXT_PRIMARY"]], [1.0, COLORS["ACCENT_GREEN"]]],
        colorbar={"title": {"text": "|ρ|"}}, xgap=1, ygap=1,
        hovertemplate="row %{y}, column %{x}: lag %{text}, strength %{z:.2f}<extra></extra>",
    ))
    figure.update_layout(**_layout(
        title={"text": f"Lead-lag, {bar_minutes}-min bars (row leads column by the time shown)", "font": {"size": 14}},
        xaxis={"tickangle": -35, "automargin": True}, yaxis={"autorange": "reversed", "automargin": True},
    ))
    return figure


def build_ccf_figure(ccf, best, label_a: str, label_b: str, bar_minutes: int) -> go.Figure:
    """Stem chart of the cross-correlation by lag, the Bonferroni significance band, and the picked lag marked."""
    if ccf is None or ccf.empty:
        return empty_figure("Click a cell of the matrix")
    lags = list(ccf.index)
    values = [None if v != v else v for v in ccf["correlation"]]
    figure = go.Figure()
    for lag, value in zip(lags, values):
        if value is None:
            continue
        figure.add_trace(go.Scatter(x=[lag, lag], y=[0, value], mode="lines", line={"color": COLORS["BORDER_COLOR"], "width": 1}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(
        x=lags, y=values, mode="markers", marker={"size": 8, "color": CCF_COLOR}, name="CCF",
        customdata=[int(n) for n in ccf["n"]], hovertemplate=f"lag %{{x}} bar(s) ({bar_minutes}-min): %{{y:.3f}}, n=%{{customdata}}<extra></extra>",
    ))
    if best is not None:
        figure.add_trace(go.Scatter(
            x=[best["lag"]], y=[best["correlation"]], mode="markers", marker={"size": 14, "symbol": "star", "color": READING_COLORS[best["reading"]]},
            name=f"best ({best['reading']})", hovertemplate=f"best lag %{{x}}: %{{y:.3f}}<extra></extra>",
        ))
    figure.add_hline(y=0, line={"color": COLORS["TEXT_SECONDARY"], "width": 1})
    figure.update_layout(**_layout(
        title={"text": f"{label_a} vs {label_b}: cross-correlation by lag", "font": {"size": 14}},
        xaxis={"title": f"lag (bars, {bar_minutes} min each; positive = {label_a} leads)", "gridcolor": COLORS["BORDER_COLOR"], "dtick": 1},
        yaxis={"title": "correlation (prewhitened)", "range": [-1.05, 1.05], "gridcolor": COLORS["BORDER_COLOR"]},
        showlegend=False, margin={"l": 70, "r": 30, "t": 50, "b": 50},
    ))
    return figure
