"""Correlation tab: watchlist builder (left) and four views on the right —
Heatmap, Time Series, Year Overlay and Summary Table.

The layout is static; ui/callbacks/correlation_callbacks.py fills the structure
dropdown, the watchlist and every view. The watchlist lives in a session-storage
store so it survives switching tabs and is shared by all four views. Every view is
computed on price DIFFERENCES (never price levels — see core/correlation.py) and
reads local Parquet only; it never triggers an API backfill (the explicit
"Backfill Missing Data" button does that instead, asynchronously). All colours come
from COLORS.
"""

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from config.settings import settings
from core.user_settings import KEY_CORRELATION_WINDOW, KEY_ROLLING_CORRELATION_WINDOW
from ui.container import container
from ui.layouts.shell import COLORS

LOOKBACK_OPTIONS = [{"label": f"{days}d", "value": str(days)} for days in (10, 20, 30, 60, 90)]
PLACEHOLDER_TEXT = "Add instruments or structures and click Compute"
MIN_ITEMS_TEXT = "Add at least 2 items to compute"

TS_WINDOW_PRESETS = [10, 20, 30, 60, 90]
DEFAULT_TS_WINDOWS = [20, 60]


def default_correlation_window() -> int:
    """Point-in-time correlation window (Heatmap lookback), from Settings > Analysis Defaults."""
    if container.repository is None:
        return settings.DEFAULT_CORRELATION_WINDOW
    return int(container.repository.get_setting(KEY_CORRELATION_WINDOW, settings.DEFAULT_CORRELATION_WINDOW))


def default_rolling_window() -> int:
    """Rolling correlation window (Time Series / Year Overlay / Summary Table), from Settings.

    A different calculation from default_correlation_window (core.correlation.
    calculate_rolling_correlation vs calculate_correlation), so it has its own setting.
    """
    if container.repository is None:
        return settings.DEFAULT_ROLLING_CORRELATION_WINDOW
    return int(
        container.repository.get_setting(KEY_ROLLING_CORRELATION_WINDOW, settings.DEFAULT_ROLLING_CORRELATION_WINDOW)
    )

MONTH_TICKVALS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_TICKTEXT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_HIDDEN = {"display": "none"}
_LABEL = {**_MUTED, "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "6px"}
_SUMMARY_COLUMNS = [
    {"id": "base", "name": "Base"},
    {"id": "target", "name": "Target"},
    {"id": "mean", "name": "Mean", "type": "numeric"},
    {"id": "std", "name": "Std", "type": "numeric"},
    {"id": "min", "name": "Min", "type": "numeric"},
    {"id": "max", "name": "Max", "type": "numeric"},
    {"id": "last", "name": "Last", "type": "numeric"},
    {"id": "beta", "name": "Beta (target/base)", "type": "numeric"},
    {"id": "r_squared", "name": "R²", "type": "numeric"},
    {"id": "n_obs", "name": "N Obs", "type": "numeric"},
]


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def _base_layout(**overrides) -> dict:
    layout = {
        "paper_bgcolor": COLORS["CARD_BG"],
        "plot_bgcolor": COLORS["CARD_BG"],
        "font": {"color": COLORS["TEXT_PRIMARY"]},
        "margin": {"l": 90, "r": 30, "t": 60, "b": 90},
    }
    layout.update(overrides)
    return layout


def empty_figure(message: str, is_error: bool = False) -> go.Figure:
    """Blank chart carrying a centred message (the placeholder or an inline error)."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 16, "color": COLORS["ACCENT_RED"] if is_error else COLORS["TEXT_SECONDARY"]},
    )
    figure.update_layout(**_base_layout(xaxis={"visible": False}, yaxis={"visible": False}))
    return figure


def build_heatmap_figure(matrix, lookback_days: int) -> go.Figure:
    """Correlation heatmap: red (-1) -> white (0) -> green (+1), value printed in each cell."""
    labels = list(matrix.index)
    values = matrix.to_numpy(dtype=float)
    z = [[None if v != v else float(v) for v in row] for row in values]
    text = [["n/a" if v != v else f"{v:.2f}" for v in row] for row in values]
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"color": COLORS["DARK_BG"], "size": 13},
            zmin=-1,
            zmax=1,
            colorscale=[[0.0, COLORS["ACCENT_RED"]], [0.5, COLORS["TEXT_PRIMARY"]], [1.0, COLORS["ACCENT_GREEN"]]],
            colorbar={"title": {"text": "ρ"}, "tickvals": [-1, -0.5, 0, 0.5, 1]},
            xgap=1,
            ygap=1,
            hovertemplate="%{y} vs %{x}: %{text}<extra></extra>",
        )
    )
    figure.update_layout(
        **_base_layout(
            title={"text": f"Correlation Matrix — {lookback_days}d", "font": {"size": 18}},
            xaxis={"tickangle": -35, "automargin": True},
            yaxis={"autorange": "reversed", "automargin": True},
        )
    )
    return figure


def build_time_series_figure(
    series_by_window: dict, base_label: str, target_label: str,
    highlight_flips: bool = True, show_avg: bool = False, only_avg: bool = False,
) -> go.Figure:
    """Trailing correlation over full history, one line per window, with optional sign-flip
    markers and an average-across-windows overlay."""
    figure = go.Figure()
    if not only_avg:
        first_window = next(iter(series_by_window), None)
        for window, series in series_by_window.items():
            figure.add_trace(
                go.Scatter(
                    x=series.index, y=series.to_numpy(dtype=float), mode="lines", name=f"{window}d",
                    connectgaps=False,
                )
            )
            if highlight_flips and len(series) > 1:
                values = series.to_numpy(dtype=float)
                sign = np.sign(values)
                flip_positions = (sign[1:] != sign[:-1]).nonzero()[0] + 1
                if len(flip_positions):
                    flips = series.index[flip_positions]
                    figure.add_trace(
                        go.Scatter(
                            x=flips, y=series.loc[flips].to_numpy(dtype=float), mode="markers",
                            marker={"color": COLORS["TEXT_PRIMARY"], "size": 7, "symbol": "x"},
                            name=f"{window}d sign flip", showlegend=(window == first_window),
                        )
                    )
    if show_avg and len(series_by_window) > 1:
        avg = pd.concat(series_by_window.values(), axis=1).mean(axis=1, skipna=True)
        figure.add_trace(
            go.Scatter(
                x=avg.index, y=avg.to_numpy(dtype=float), mode="lines", name="Average",
                line={"color": COLORS["ACCENT_YELLOW"], "width": 3, "dash": "dash"}, connectgaps=False,
            )
        )
    figure.add_hline(y=0, line_dash="dot", line_color=COLORS["TEXT_SECONDARY"])
    figure.update_layout(
        **_base_layout(
            title={"text": f"{base_label} vs {target_label} — trailing correlation", "font": {"size": 18}},
            xaxis={"title": "Date"},
            yaxis={"title": "Trailing correlation", "range": [-1, 1], "zeroline": False},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
    )
    return figure


def build_year_overlay_figure(
    frame, base_label: str, target_label: str, window: int, show_avg: bool = False, only_avg: bool = False,
) -> go.Figure:
    """The same pair's rolling correlation plotted by day-of-year, one line per calendar year."""
    figure = go.Figure()
    if not only_avg:
        for year, group in frame.groupby("year"):
            group = group.sort_values("doy")
            figure.add_trace(
                go.Scatter(x=group["doy"], y=group["value"], mode="lines", name=str(year), connectgaps=False)
            )
    if show_avg and frame["year"].nunique() > 1:
        avg_by_doy = frame.groupby("doy")["value"].mean().sort_index()
        figure.add_trace(
            go.Scatter(
                x=avg_by_doy.index, y=avg_by_doy.to_numpy(dtype=float), mode="lines", name="Average",
                line={"color": COLORS["ACCENT_YELLOW"], "width": 3, "dash": "dash"},
            )
        )
    figure.add_hline(y=0, line_dash="dot", line_color=COLORS["TEXT_SECONDARY"])
    figure.update_layout(
        **_base_layout(
            title={"text": f"{base_label} vs {target_label} — year-over-year ({window}d)", "font": {"size": 18}},
            xaxis={"title": "Month"},
            yaxis={"title": "Trailing correlation", "range": [-1, 1], "zeroline": False},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )
    )
    figure.update_xaxes(tickmode="array", tickvals=MONTH_TICKVALS, ticktext=MONTH_TICKTEXT, range=[1, 366])
    return figure


def build_summary_table(rows: list[dict]) -> dash_table.DataTable:
    """Mean/std/min/max/last rolling correlation and beta for every watchlist pair at one window."""
    data = [
        {
            **row,
            **{k: round(row[k], 3) for k in ("mean", "std", "min", "max", "last")},
            "beta": round(row["beta"], 3) if row.get("beta") is not None else "—",
            "r_squared": round(row["r_squared"], 3) if row.get("r_squared") is not None else "—",
        }
        for row in rows
    ]
    return dash_table.DataTable(
        id="corr-summary-table",
        columns=_SUMMARY_COLUMNS,
        data=data,
        sort_action="native",
        page_size=25,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": COLORS["SIDEBAR_BG"], "color": COLORS["TEXT_SECONDARY"],
            "border": f"1px solid {COLORS['BORDER_COLOR']}", "fontWeight": "bold",
        },
        style_cell={
            "backgroundColor": COLORS["CARD_BG"], "color": COLORS["TEXT_PRIMARY"],
            "border": f"1px solid {COLORS['BORDER_COLOR']}", "padding": "8px 12px", "textAlign": "right",
        },
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"} for c in ("base", "target")],
        style_data_conditional=[
            {"if": {"column_id": "mean", "filter_query": "{mean} >= 0.7"}, "color": COLORS["ACCENT_GREEN"], "fontWeight": "bold"},
            {"if": {"column_id": "mean", "filter_query": "{mean} <= -0.7"}, "color": COLORS["ACCENT_RED"], "fontWeight": "bold"},
        ],
    )


# ----------------------------------------------------------------------
# Watchlist rendering
# ----------------------------------------------------------------------


_TYPE_LABELS = {"structure": "STRUCTURE", "excel": "EXCEL"}


def render_watchlist(items: list[dict], warnings: dict[str, str | None] | None = None) -> html.Div:
    """One row per item: type, label, an optional warning and the remove (×) button."""
    if not items:
        return html.Div("No items yet. Add at least 2 to compute.", style=_MUTED)
    warnings = warnings or {}
    rows = []
    for index, item in enumerate(items):
        warning = warnings.get(f"{item['type']}:{item['key']}")
        body = [
            html.Span(
                _TYPE_LABELS.get(item["type"], "INSTRUMENT"),
                style={**_MUTED, "fontSize": "10px", "marginRight": "8px", "letterSpacing": "0.05em"},
            ),
            html.Span(item["label"], style={"color": COLORS["TEXT_PRIMARY"], "fontWeight": "bold"}),
        ]
        if warning:
            body.append(html.Div(f"⚠️ {warning} — skipped", style={"color": COLORS["ACCENT_YELLOW"], "fontSize": "12px"}))
        rows.append(
            html.Div(
                [
                    html.Div(body, style={"flex": "1", "minWidth": 0}),
                    dbc.Button(
                        "×", id={"type": "corr-remove", "index": index}, color="danger", outline=True, size="sm",
                        title="Remove",
                    ),
                ],
                style={
                    "display": "flex", "alignItems": "center", "gap": "8px", "padding": "8px 10px",
                    "backgroundColor": COLORS["SIDEBAR_BG"], "borderRadius": "4px", "marginBottom": "6px",
                    "borderLeft": f"3px solid {COLORS['ACCENT_YELLOW'] if warning else COLORS['ACCENT_GREEN']}",
                },
            )
        )
    return html.Div(rows)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _left_panel() -> dbc.Card:
    mode = dbc.RadioItems(
        id="corr-mode",
        options=[
            {"label": "Instrument", "value": "instrument"},
            {"label": "Structure", "value": "structure"},
            {"label": "Excel", "value": "excel"},
        ],
        value="instrument",
        className="btn-group mb-3",
        inputClassName="btn-check",
        labelClassName="btn btn-outline-secondary",
        labelCheckedClassName="active",
    )
    instrument_group = html.Div(
        [
            html.Div("Exchange symbol", style=_LABEL),
            dbc.Input(id="corr-instrument-input", placeholder="e.g. CLZ25, CLZ25-H26, CLZ25-F26-G26"),
        ],
        id="corr-instrument-group",
    )
    structure_group = html.Div(
        [
            html.Div("Open structure", style=_LABEL),
            dcc.Dropdown(id="corr-structure-select", options=[], placeholder="Select an open structure...", clearable=True),
        ],
        id="corr-structure-group",
        style=_HIDDEN,
    )
    excel_group = html.Div(
        [
            html.Div(
                "Manual fallback source: use this when the API can't provide a symbol. "
                "Excel columns are generic curve labels (e.g. CL1), not exchange contract symbols — "
                "you decide what each one stands in for.",
                style={**_MUTED, "marginBottom": "8px"},
            ),
            html.Div("Uploaded file", style=_LABEL),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(id="corr-excel-file-select", options=[], placeholder="Choose a file...", clearable=True),
                        width=8,
                    ),
                    dbc.Col(
                        dcc.Upload(
                            id="corr-excel-upload",
                            children=dbc.Button("+ Upload", color="secondary", outline=True, size="sm", style={"width": "100%"}),
                            accept=".xlsx",
                            multiple=False,
                        ),
                        width=4,
                    ),
                ],
                className="g-1 mb-1",
            ),
            html.Div(id="corr-excel-upload-status", style={**_MUTED, "minHeight": "18px"}),
            html.Div("Sheet", style=_LABEL),
            dcc.Dropdown(id="corr-excel-sheet-select", options=[], placeholder="Select a sheet...", clearable=False, className="mb-2"),
            html.Div("Column(s)", style=_LABEL),
            dcc.Dropdown(id="corr-excel-columns", options=[], value=[], multi=True, placeholder="Select column(s)..."),
            dbc.Button(
                "🗑 Delete file", id="corr-excel-delete-btn", color="danger", outline=True, size="sm", className="mt-2",
            ),
        ],
        id="corr-excel-group",
        style=_HIDDEN,
    )
    body = [
        html.Div("Build watchlist", style=_LABEL),
        mode,
        instrument_group,
        structure_group,
        excel_group,
        dbc.Button("Add", id="corr-add-btn", color="secondary", className="mt-2 mb-1"),
        html.Div(id="corr-add-message", style={"color": COLORS["ACCENT_YELLOW"], "fontSize": "13px", "minHeight": "20px"}),
        html.Hr(style={"borderColor": COLORS["BORDER_COLOR"]}),
        html.Div("Watchlist (minimum 2)", style=_LABEL),
        html.Div(id="corr-watchlist-list", style={"maxHeight": "280px", "overflowY": "auto"}),
        dbc.Button(
            "🔄 Backfill Missing Data", id="corr-backfill-btn", color="secondary", outline=True,
            size="sm", className="mt-2", style=_HIDDEN,
        ),
        html.Div(id="corr-backfill-status", style={**_MUTED, "minHeight": "18px"}),
    ]
    return dbc.Card(
        dbc.CardBody(body),
        style={"backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}"},
    )


def _compute_button(component_id: str) -> dbc.Button:
    return dbc.Button(
        "Compute", id=component_id, className="mb-3",
        style={"backgroundColor": COLORS["ACCENT_GREEN"], "borderColor": COLORS["ACCENT_GREEN"], "color": COLORS["DARK_BG"], "fontWeight": "bold"},
    )


def _base_target_row(base_id: str, target_id: str) -> dbc.Row:
    """Base/Target column pickers, populated from the current watchlist labels."""
    return dbc.Row(
        [
            dbc.Col([html.Div("Base", style=_LABEL), dcc.Dropdown(id=base_id, options=[], clearable=False)], md=6, className="mb-2"),
            dbc.Col([html.Div("Target", style=_LABEL), dcc.Dropdown(id=target_id, options=[], clearable=False)], md=6, className="mb-2"),
        ]
    )


def _heatmap_tab() -> html.Div:
    default_lookback = default_correlation_window()
    options = LOOKBACK_OPTIONS
    if not any(int(opt["value"]) == default_lookback for opt in options):
        options = sorted([*options, {"label": f"{default_lookback}d", "value": str(default_lookback)}], key=lambda o: int(o["value"]))
    controls = dbc.Row(
        [
            dbc.Col([html.Div("Lookback period", style=_LABEL), dbc.Select(id="corr-lookback", options=options, value=str(default_lookback))], md=4),
            dbc.Col(
                [
                    html.Div("As-of date (optional)", style=_LABEL),
                    dcc.DatePickerSingle(id="corr-asof-date", placeholder="Latest available", clearable=True),
                ],
                md=4,
            ),
            dbc.Col(_compute_button("corr-compute-btn"), md=4, className="d-flex align-items-end"),
        ],
        className="mb-2",
    )
    return html.Div(
        [
            controls,
            dcc.Graph(id="corr-heatmap", figure=empty_figure(PLACEHOLDER_TEXT), config={"displayModeBar": False}, style={"height": "560px"}),
            html.Div(id="corr-status", style={**_MUTED, "marginTop": "10px", "minHeight": "20px"}),
        ]
    )


def _time_series_tab() -> html.Div:
    window_checks = dbc.Checklist(
        id="corr-ts-windows", options=[{"label": f"{w}d", "value": w} for w in TS_WINDOW_PRESETS],
        value=DEFAULT_TS_WINDOWS, inline=True, className="mb-1",
    )
    controls = [
        _base_target_row("corr-ts-base", "corr-ts-target"),
        html.Div("Windows to overlay", style=_LABEL),
        window_checks,
        dbc.Input(id="corr-ts-custom-windows", placeholder="Custom window(s), comma/space separated", size="sm", className="mb-2"),
        dbc.Checklist(
            id="corr-ts-options",
            options=[{"label": "Highlight sign-change points", "value": "flip"}, {"label": "Show average across windows", "value": "avg"}],
            value=["flip"], inline=True, className="mb-2",
        ),
        _compute_button("corr-ts-compute-btn"),
    ]
    return html.Div(
        [
            *controls,
            dcc.Graph(id="corr-ts-graph", figure=empty_figure(PLACEHOLDER_TEXT), config={"displayModeBar": False}, style={"height": "500px"}),
            html.Div(id="corr-ts-status", style={**_MUTED, "marginTop": "10px", "minHeight": "20px"}),
        ]
    )


def _year_overlay_tab() -> html.Div:
    controls = [
        _base_target_row("corr-yr-base", "corr-yr-target"),
        dbc.Row(
            [
                dbc.Col([html.Div("Window (days)", style=_LABEL), dbc.Input(id="corr-yr-window", type="number", min=2, step=1, value=default_rolling_window())], md=4),
                dbc.Col(
                    dbc.Checklist(
                        id="corr-yr-options", options=[{"label": "Show average across years", "value": "avg"}],
                        value=[], inline=True, className="mt-4",
                    ),
                    md=4,
                ),
                dbc.Col(_compute_button("corr-yr-compute-btn"), md=4, className="d-flex align-items-end"),
            ]
        ),
    ]
    return html.Div(
        [
            *controls,
            dcc.Graph(id="corr-yr-graph", figure=empty_figure(PLACEHOLDER_TEXT), config={"displayModeBar": False}, style={"height": "500px"}),
            html.Div(id="corr-yr-status", style={**_MUTED, "marginTop": "10px", "minHeight": "20px"}),
        ]
    )


def _summary_tab() -> html.Div:
    controls = dbc.Row(
        [
            dbc.Col([html.Div("Window (days)", style=_LABEL), dbc.Input(id="corr-sum-window", type="number", min=2, step=1, value=default_rolling_window())], md=4),
            dbc.Col(_compute_button("corr-sum-compute-btn"), md=4, className="d-flex align-items-end"),
        ],
        className="mb-2",
    )
    return html.Div(
        [
            controls,
            html.Div(id="corr-sum-table-wrap"),
            html.Div(id="corr-sum-status", style={**_MUTED, "marginTop": "10px", "minHeight": "20px"}),
        ]
    )


def correlation_layout() -> html.Div:
    """The Correlation tab: watchlist builder on the left, Heatmap/Time Series/Year
    Overlay/Summary Table views (sharing that watchlist) on the right."""
    return html.Div(
        [
            dcc.Store(id="corr-watchlist", data=[], storage_type="session"),
            dcc.Store(id="corr-missing-symbols", data=[]),
            html.H3("🔗 Correlation", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "16px"}),
            dbc.Row(
                [
                    dbc.Col(_left_panel(), lg=4, className="mb-3"),
                    dbc.Col(
                        dbc.Tabs(
                            [
                                dbc.Tab(_heatmap_tab(), label="Heatmap", tab_id="heatmap"),
                                dbc.Tab(_time_series_tab(), label="Time Series", tab_id="time-series"),
                                dbc.Tab(_year_overlay_tab(), label="Year Overlay", tab_id="year-overlay"),
                                dbc.Tab(_summary_tab(), label="Summary Table", tab_id="summary"),
                            ],
                            id="corr-view-tabs",
                            active_tab="heatmap",
                        ),
                        lg=8,
                        className="mb-3",
                    ),
                ]
            ),
        ]
    )
