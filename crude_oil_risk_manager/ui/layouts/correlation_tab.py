"""Correlation Heatmap tab: watchlist builder (left) and heatmap (right).

The layout is static; ui/callbacks/correlation_callbacks.py fills the structure
dropdown, the watchlist and the heatmap. The watchlist lives in a session-storage
store so it survives switching tabs. All colours come from COLORS.
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from ui.layouts.shell import COLORS

LOOKBACK_OPTIONS = [{"label": f"{days}d", "value": str(days)} for days in (10, 20, 30, 60, 90)]
DEFAULT_LOOKBACK = "30"
PLACEHOLDER_TEXT = "Add instruments or structures and click Compute"
MIN_ITEMS_TEXT = "Add at least 2 items to compute"

_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_HIDDEN = {"display": "none"}
_LABEL = {**_MUTED, "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "6px"}


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


# ----------------------------------------------------------------------
# Watchlist rendering
# ----------------------------------------------------------------------


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
                "STRUCTURE" if item["type"] == "structure" else "INSTRUMENT",
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
        options=[{"label": "Instrument", "value": "instrument"}, {"label": "Structure", "value": "structure"}],
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
    body = [
        html.Div("Build watchlist", style=_LABEL),
        mode,
        instrument_group,
        structure_group,
        dbc.Button("Add", id="corr-add-btn", color="secondary", className="mt-2 mb-1"),
        html.Div(id="corr-add-message", style={"color": COLORS["ACCENT_YELLOW"], "fontSize": "13px", "minHeight": "20px"}),
        html.Hr(style={"borderColor": COLORS["BORDER_COLOR"]}),
        html.Div("Watchlist (minimum 2)", style=_LABEL),
        html.Div(id="corr-watchlist-list", style={"maxHeight": "280px", "overflowY": "auto"}),
        html.Hr(style={"borderColor": COLORS["BORDER_COLOR"]}),
        html.Div("Lookback period", style=_LABEL),
        dbc.Select(id="corr-lookback", options=LOOKBACK_OPTIONS, value=DEFAULT_LOOKBACK, className="mb-3"),
        dbc.Button(
            "Compute", id="corr-compute-btn", className="w-100",
            style={"backgroundColor": COLORS["ACCENT_GREEN"], "borderColor": COLORS["ACCENT_GREEN"], "color": COLORS["DARK_BG"], "fontWeight": "bold"},
        ),
        html.Div(id="corr-status", style={**_MUTED, "marginTop": "10px", "minHeight": "40px"}),
    ]
    return dbc.Card(
        dbc.CardBody(body),
        style={"backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}"},
    )


def correlation_layout() -> html.Div:
    """The Correlation tab: watchlist builder on the left, heatmap on the right."""
    return html.Div(
        [
            dcc.Store(id="corr-watchlist", data=[], storage_type="session"),
            html.H3("🔗 Correlation", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "16px"}),
            dbc.Row(
                [
                    dbc.Col(_left_panel(), lg=4, className="mb-3"),
                    dbc.Col(
                        dcc.Graph(
                            id="corr-heatmap",
                            figure=empty_figure(PLACEHOLDER_TEXT),
                            config={"displayModeBar": False},
                            style={"height": "620px"},
                        ),
                        lg=8,
                        className="mb-3",
                    ),
                ]
            ),
        ]
    )
