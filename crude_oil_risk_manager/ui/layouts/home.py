"""Home tab layout: portfolio metric cards, winner/loser, stale-data warning and the live structures as packets.

This is the skeleton only; ui/callbacks/home_callbacks.py fills it from the
`store-portfolio-pnl` store.
"""

import dash_bootstrap_components as dbc
from dash import html

from ui.layouts.shell import COLORS

_LABEL_STYLE = {
    "fontSize": "12px",
    "color": COLORS["TEXT_SECONDARY"],
    "textTransform": "uppercase",
    "letterSpacing": "0.05em",
    "marginBottom": "6px",
}
_VALUE_STYLE = {"fontSize": "26px", "fontWeight": "bold", "color": COLORS["TEXT_PRIMARY"], "lineHeight": "1.2"}


def _rgba(hex_color: str, alpha: float) -> str:
    """'#rrggbb' -> 'rgba(r, g, b, alpha)' so tints are derived from the COLORS palette."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _card(label: str, body_id: str, accent: str, value_style: dict | None = None) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(label, style=_LABEL_STYLE),
                html.Div("—", id=body_id, style={**_VALUE_STYLE, **(value_style or {})}),
            ]
        ),
        style={
            "backgroundColor": COLORS["CARD_BG"],
            "border": f"1px solid {COLORS['BORDER_COLOR']}",
            "borderTop": f"3px solid {accent}",
            "height": "100%",
        },
    )


# AG Grid 33+ themes via its Theming API (the old "ag-theme-alpine-dark" CSS class no longer applies).
GRID_THEME = {
    "function": (
        "themeQuartz.withParams({"
        f"backgroundColor: '{COLORS['CARD_BG']}', "
        f"foregroundColor: '{COLORS['TEXT_PRIMARY']}', "
        f"headerBackgroundColor: '{COLORS['SIDEBAR_BG']}', "
        f"headerTextColor: '{COLORS['TEXT_SECONDARY']}', "
        f"borderColor: '{COLORS['BORDER_COLOR']}', "
        "browserColorScheme: 'dark'})"
    )
}

REALIZED_VALUE_STYLE = {"fontSize": "2rem", "fontWeight": "bold", "lineHeight": "1.2", "color": COLORS["TEXT_PRIMARY"]}


def _realized_metric(label: str, value_id: str, note_id: str | None = None) -> dbc.Col:
    children = [html.Div(label, style=_LABEL_STYLE), html.Div("—", id=value_id, style=REALIZED_VALUE_STYLE)]
    if note_id:
        children.append(html.Div("", id=note_id, style={"fontSize": "11px", "color": COLORS["ACCENT_YELLOW"], "marginTop": "2px"}))
    return dbc.Col(children, md=6)


def _realized_bar() -> html.Div:
    """Persistent realized-PnL bar above the cards; it only changes when a position is exited."""
    return html.Div(
        [
            dbc.Row(
                [
                    _realized_metric("TOTAL REALIZED PnL (ALL TIME)", "home-total-realized-pnl", "home-pnl-reset-note"),
                    _realized_metric("TODAY'S REALIZED PnL", "home-today-realized-pnl"),
                ]
            ),
            html.Hr(style={"borderColor": COLORS["BORDER_COLOR"], "margin": "16px 0 0 0"}),
        ],
        id="home-realized-pnl-bar",
        style={
            "backgroundColor": COLORS["SIDEBAR_BG"],
            "border": f"1px solid {COLORS['BORDER_COLOR']}",
            "borderRadius": "6px",
            "padding": "16px 20px 0 20px",
            "marginBottom": "20px",
        },
    )


def home_layout() -> html.Div:
    """Skeleton of the Home tab with the component ids the callbacks fill in."""
    metric_cards = dbc.Row(
        [
            dbc.Col(_card("Unrealized PnL", "home-card-unrealized-pnl", COLORS["ACCENT_GREEN"]), md=4, xl=True, className="mb-3"),
            dbc.Col(_card("Open Structures", "home-card-open-structures", COLORS["ACCENT_BLUE"]), md=4, xl=True, className="mb-3"),
            dbc.Col(_card("Open Legs", "home-card-open-legs", COLORS["ACCENT_BLUE"]), md=4, xl=True, className="mb-3"),
            dbc.Col(
                _card("Net Lots by Product", "home-card-net-lots", COLORS["ACCENT_BLUE"], {"fontSize": "16px"}),
                md=4, xl=True, className="mb-3",
            ),
            dbc.Col(_card("Margin Used", "home-card-margin-used", COLORS["ACCENT_YELLOW"]), md=4, xl=True, className="mb-3"),
        ]
    )

    winner_loser_stale = dbc.Row(
        [
            dbc.Col(
                _card("Largest Winner", "home-winner-card", COLORS["ACCENT_GREEN"], {"fontSize": "18px"}),
                md=4, className="mb-3",
            ),
            dbc.Col(
                _card("Largest Loser", "home-loser-card", COLORS["ACCENT_RED"], {"fontSize": "18px"}),
                md=4, className="mb-3",
            ),
            dbc.Col(html.Div(id="home-stale-banner", style={"display": "none"}), md=4, className="mb-3"),
        ]
    )

    table = html.Div(
        [
            html.Div("Live Structures", style={**_LABEL_STYLE, "fontSize": "14px", "marginBottom": "10px"}),
            html.Div(id="home-structures-table", children=[html.Div("No open structures.", style={"color": COLORS["TEXT_SECONDARY"]})]),
        ]
    )

    return html.Div([_realized_bar(), metric_cards, winner_loser_stale, table])
