"""Structure correlation matrix section at the top of the Correlation tab (core.corr_matrix).

Pick a target and a base curve (CL or BRN: outright, spread, fly, d-fly); the matrix shows the correlation of
the first N structures of one against the first N of the other, from settlement data. Click a cell for how
that correlation changed; pick two structures for their hedge ratio and how it changed. The layout is static;
ui/callbacks/corr_matrix_callbacks.py fills it. Colours from COLORS.
"""

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from core.corr_matrix import structure_options
from ui.layouts.curve_tab import _CARD, _LABEL, _MUTED, _layout, empty_figure
from ui.layouts.shell import COLORS

DEFAULT_KEY = "BRN|fly"
DEFAULT_N = 15
DEFAULT_CHART_DAYS = 120
DEFAULT_HEDGE_DAYS = 60
MODE_OPTIONS = [{"label": "Current market (no date range)", "value": "current"}, {"label": "Historical date range", "value": "historical"}]
COLORSCALE = [[0.0, "#d9534f"], [0.5, "#2b2f36"], [1.0, "#3cb371"]]


def _section(title: str, hint: str, body) -> html.Div:
    return html.Div([
        html.H6(title, style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "24px", "marginBottom": "2px"}),
        html.Div(hint, style={**_MUTED, "marginBottom": "8px"}),
        body,
    ])


def _field(label: str, component, md: int, hint: str | None = None) -> dbc.Col:
    return dbc.Col([html.Div(label, style=_LABEL, title=hint), component], md=md, className="mb-2")


def corr_matrix_section(rolling: int, lookback: int) -> html.Div:
    options = structure_options()
    controls = html.Div([
        dbc.Row([
            _field("Target (rows)", dcc.Dropdown(id="cm-target", options=options, value=DEFAULT_KEY, clearable=False), 3),
            _field("Base (columns)", dcc.Dropdown(id="cm-base", options=options, value=DEFAULT_KEY, clearable=False), 3),
            _field("Front structures", dbc.Input(id="cm-n", type="number", value=DEFAULT_N, min=2, max=30, step=1, debounce=True, size="sm"), 2,
                   "How many of the front structures of each curve (the matrix is N by N)"),
            _field("Rolling window (days)", dbc.Input(id="cm-rolling", type="number", value=rolling, min=2, step=1, debounce=True, size="sm"), 2,
                   "Each day's correlation is measured over this many days of settlement changes"),
            _field("Lookback (days)", dbc.Input(id="cm-lookback", type="number", value=lookback, min=1, step=1, debounce=True, size="sm"), 2,
                   "The matrix shows the average of the daily correlations over this many days (the date range instead, in historical mode)"),
        ], className="g-2"),
        dbc.Row([
            _field("Period", dcc.RadioItems(id="cm-mode", options=MODE_OPTIONS, value="current", inline=True, inputStyle={"marginRight": "6px", "marginLeft": "14px"},
                                            labelStyle={"color": COLORS["TEXT_PRIMARY"], "fontSize": "13px"}), 6),
            dbc.Col(html.Div([
                html.Div("Date range (replaces the lookback)", style=_LABEL),
                dcc.DatePickerRange(id="cm-dates", display_format="DD MMM YYYY", start_date=None, end_date=None, minimum_nights=2, clearable=True),
            ], id="cm-dates-wrap", style={"display": "none"}), md=6, className="mb-2"),
        ], className="g-2"),
    ], style={**_CARD, "padding": "12px 14px", "borderRadius": "6px", "marginBottom": "10px"})
    return html.Div([
        dcc.Store(id="cm-params"),
        html.H5("Structure correlation matrix", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "4px"}),
        html.Div("Settlement data. Each cell: the lookback average of the daily rolling-window correlation of the two structures' daily changes. Click a cell to see how it changed.",
                 style={**_MUTED, "marginBottom": "10px"}),
        controls,
        dcc.Loading(html.Div(id="cm-status", style={**_MUTED, "marginBottom": "6px"}), type="dot"),
        dcc.Graph(id="cm-matrix", figure=empty_figure("Loading the matrix"), config={"displayModeBar": False}),
        _section("How the clicked correlation changed", "The daily rolling-window correlation of the two structures you clicked in the matrix.",
                 html.Div([
                     dbc.Row([_field("Days shown", dbc.Input(id="cm-chart-days", type="number", value=DEFAULT_CHART_DAYS, min=5, step=10, debounce=True, size="sm"), 2)], className="g-2"),
                     dcc.Graph(id="cm-chart", figure=empty_figure("Click a cell of the matrix"), config={"displayModeBar": False}, style={"height": "380px"}),
                 ])),
        _section("Hedge ratio of two structures", "Pick one structure of the target list and one of the base list: lots of the base per lot of the target (minimum variance: the beta of the target's daily changes on the base's).",
                 html.Div([
                     dbc.Row([
                         _field("Target structure (held)", dcc.Dropdown(id="cm-hedge-target", options=[], clearable=False), 3),
                         _field("Base structure (hedge)", dcc.Dropdown(id="cm-hedge-base", options=[], clearable=False), 3),
                         _field("Days shown", dbc.Input(id="cm-hedge-days", type="number", value=DEFAULT_HEDGE_DAYS, min=5, step=10, debounce=True, size="sm"), 2),
                         _field("Beta window (days)", dbc.Input(id="cm-beta-window", type="number", value=None, min=5, step=5, debounce=True, size="sm", placeholder="auto"), 2,
                                "Days each beta is measured over. Empty: the span the rest of the app uses (rolling + lookback - 1, at least 30)"),
                     ], className="g-2"),
                     dcc.Loading(html.Div(id="cm-hedge-text"), type="dot"),
                     dcc.Graph(id="cm-hedge-chart", figure=empty_figure("Pick two structures"), config={"displayModeBar": False}, style={"height": "380px"}),
                 ])),
    ], style={"paddingBottom": "10px"})


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def build_matrix_figure(values, days, target_name: str, base_name: str) -> go.Figure:
    """Heatmap of the matrix: target rows (front at the top) against base columns."""
    if values is None or values.empty:
        return empty_figure("No data")
    z = values.to_numpy(dtype=float)
    figure = go.Figure(go.Heatmap(
        z=z, x=list(values.columns), y=list(values.index), zmin=-1, zmax=1, colorscale=COLORSCALE, xgap=1, ygap=1,
        text=np.where(np.isfinite(z), np.char.mod("%.2f", z), ""), texttemplate="%{text}", textfont={"size": 11},
        customdata=days.to_numpy(), colorbar={"title": "corr", "thickness": 12},
        hovertemplate="Target %{y}<br>Base %{x}<br>correlation %{z:.2f} (%{customdata} days averaged)<extra></extra>",
    ))
    size = max(len(values.index), 1)
    figure.update_layout(**_layout(
        title={"text": f"Target {target_name} (rows) against base {base_name} (columns)", "font": {"size": 14}},
        xaxis={"title": f"Base: {base_name}", "side": "top", "tickangle": -40}, yaxis={"title": f"Target: {target_name}", "autorange": "reversed"},
        height=int(min(max(46 * size + 190, 380), 1100)), margin={"l": 140, "r": 30, "t": 150, "b": 30},
    ))
    return figure


def build_history_figure(series, title: str, average: float | None, note: str) -> go.Figure:
    if series is None or len(series) == 0:
        return empty_figure(note or "No data")
    figure = go.Figure(go.Scatter(x=series.index, y=series.to_numpy(), mode="lines", name="daily correlation", line={"color": COLORS["ACCENT_GREEN"], "width": 2},
                                  hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra></extra>"))
    if average is not None and np.isfinite(average):
        figure.add_hline(y=average, line={"color": COLORS["ACCENT_YELLOW"], "dash": "dash", "width": 1}, annotation_text=f"matrix value {average:.2f}", annotation_position="top left")
    figure.update_layout(**_layout(title={"text": title, "font": {"size": 14}}, yaxis={"title": "correlation", "range": [-1.05, 1.05], "gridcolor": COLORS["BORDER_COLOR"]},
                                   xaxis={"gridcolor": COLORS["BORDER_COLOR"]}, hovermode="x unified", showlegend=False, margin={"l": 70, "r": 30, "t": 50, "b": 50}))
    return figure


def build_hedge_figure(frame, title: str, note: str) -> go.Figure:
    if frame is None or len(frame) == 0:
        return empty_figure(note or "No data")
    figure = go.Figure(go.Scatter(x=frame.index, y=frame["beta"].to_numpy(), mode="lines", name="hedge ratio", line={"color": "#4da3ff", "width": 2},
                                  customdata=frame["corr"].to_numpy(), hovertemplate="%{x|%d %b %Y}: %{y:.2f} (correlation %{customdata:.2f})<extra></extra>"))
    figure.update_layout(**_layout(title={"text": title, "font": {"size": 14}}, yaxis={"title": "lots of base per lot of target", "gridcolor": COLORS["BORDER_COLOR"], "zeroline": True},
                                   xaxis={"gridcolor": COLORS["BORDER_COLOR"]}, hovermode="x unified", showlegend=False, margin={"l": 70, "r": 30, "t": 50, "b": 50}))
    return figure
