"""Price Alerts page: build a structure from its legs, type a price level, and get an alert when its price gets there.

Left, the form (template -> one row per leg -> level -> when to fire); below it, every alert made so far with its live price and
how far it is from the level, and a delete button on each. The layout is static except the leg rows and the table, which
ui/callbacks/level_alerts_callbacks.py fills; core/level_alerts.py holds the rules.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from core.level_alerts import ABOVE, AUTO, BELOW, DIRECTION_LABELS, TRIGGERED, level_price
from core.structure_utils import STRUCTURE_TEMPLATES
from ui.layouts.shell import COLORS

EMPTY = "—"
DEFAULT_TEMPLATE = "spread"

_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_LABEL = {**_MUTED, "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "6px"}
_CARD = {"backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}"}
_CELL = {"padding": "8px 10px", "borderBottom": f"1px solid {COLORS['BORDER_COLOR']}", "fontSize": "14px", "verticalAlign": "middle"}

TRIGGER_OPTIONS = [
    {"label": "Auto: from where the price is now", "value": AUTO},
    {"label": "Price rises to or above the level", "value": ABOVE},
    {"label": "Price falls to or below the level", "value": BELOW},
]


def leg_rows(rows: list[dict], removable: bool) -> list:
    """One row per leg: symbol and ratio (typed values are kept by the callback), and a remove button for a Custom structure."""
    out = []
    for index, row in enumerate(rows):
        out.append(dbc.Row([
            dbc.Col(html.Div(f"Leg {index + 1}", style={"color": COLORS["TEXT_PRIMARY"], "paddingTop": "8px"}), width=2, md=1),
            dbc.Col(dbc.Input(id={"type": "la-symbol", "index": index}, value=row.get("symbol"), debounce=True,
                              placeholder="e.g. BRNJ27 or BRNJ27-K27"), width=7, md=5),
            dbc.Col(dbc.Input(id={"type": "la-ratio", "index": index}, type="number", step=1, value=row.get("ratio"), placeholder="Ratio"), width=3, md=2),
            dbc.Col(dbc.Button("✕", id={"type": "la-remove", "index": index}, color="danger", outline=True, size="sm") if removable else None, width=12, md=1),
        ], className="mb-2"))
    return out


def _form() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H5("New price alert", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col([html.Div("Structure type", style=_LABEL), dcc.Dropdown(
                id="la-template", options=[{"label": f"{spec['icon']} {spec['label']}", "value": key} for key, spec in STRUCTURE_TEMPLATES.items()],
                value=DEFAULT_TEMPLATE, clearable=False, className="builder-dropdown")], md=3),
        ], className="mb-3"),
        html.Div("Legs", style=_LABEL),
        html.Div(id="la-leg-rows"),
        dbc.Button("+ Add leg", id="la-add-leg", color="secondary", outline=True, size="sm", className="mb-3"),
        html.Div("A leg can itself be a spread or a fly (e.g. BRNJ27-K27). The structure's price is worked out from the legs' live prices like on the Structures page.", style={**_MUTED, "marginBottom": "14px"}),
        dbc.Row([
            dbc.Col([html.Div("Price level", style=_LABEL), dbc.Input(id="la-level", type="number", step="any", placeholder="e.g. 0.45")], md=2),
            dbc.Col([html.Div("Fire when", style=_LABEL), dcc.Dropdown(id="la-trigger", options=TRIGGER_OPTIONS, value=AUTO, clearable=False, className="builder-dropdown")], md=4),
            dbc.Col([html.Div("Name (optional)", style=_LABEL), dbc.Input(id="la-name", maxLength=100, placeholder="blank = the structure's symbol")], md=4),
            dbc.Col([html.Div(" ", style=_LABEL), dbc.Button("🔔 Create alert", id="la-create", color="primary")], md=2),
        ], className="g-2"),
        html.Div(id="la-status", style={"fontSize": "13px", "minHeight": "20px", "marginTop": "10px", "whiteSpace": "pre-wrap"}),
    ]), style=_CARD, className="mb-4")


def _heading(text: str) -> html.Div:
    return html.Div(text, style={**_LABEL, "marginBottom": "0"})


def alerts_table(alerts: list, prices: dict[str, float]) -> html.Div:
    """Every alert: active ones first (closest to firing first), then the triggered ones, newest first."""
    if not alerts:
        return html.Div("No price alerts yet. Make one above.", style={**_MUTED, "padding": "12px 0"})

    def away(alert, price):
        return None if price is None else abs(alert.level - price)

    live = {alert.alert_id: level_price(alert.legs, prices) for alert in alerts}
    active = sorted((a for a in alerts if a.status != TRIGGERED), key=lambda a: (away(a, live[a.alert_id]) is None, away(a, live[a.alert_id]) or 0.0))
    done = sorted((a for a in alerts if a.status == TRIGGERED), key=lambda a: a.triggered_at or a.created_at, reverse=True)
    rows = []
    for alert in [*active, *done]:
        price, fired = live[alert.alert_id], alert.status == TRIGGERED
        legs = " / ".join(f"{leg['ratio']:+g} {leg['symbol']}" for leg in alert.legs)
        status = (html.Span(f"✓ Triggered {alert.triggered_at:%Y-%m-%d %H:%M} UTC at {alert.triggered_price:g}", style={"color": COLORS["ACCENT_YELLOW"]})
                  if fired else html.Span("● Active", style={"color": COLORS["ACCENT_GREEN"]}))
        rows.append(html.Tr([
            html.Td(alert.name, style={**_CELL, "fontWeight": "bold"}),
            html.Td(legs, style={**_CELL, **_MUTED, "fontFamily": "monospace"}),
            html.Td(f"{alert.level:g}", style={**_CELL, "fontWeight": "bold"}),
            html.Td(DIRECTION_LABELS[alert.direction], style=_CELL),
            html.Td(EMPTY if price is None else f"{price:g}", style=_CELL),
            html.Td(EMPTY if fired or price is None else f"{abs(alert.level - price):g} to go", style=_CELL),
            html.Td(status, style=_CELL),
            html.Td(f"{alert.created_at:%Y-%m-%d %H:%M}", style={**_CELL, **_MUTED}),
            html.Td(dbc.Button("🗑 Delete", id={"type": "la-delete", "index": alert.alert_id}, color="danger", outline=True, size="sm"), style=_CELL),
        ], style={"opacity": "0.75"} if fired else {}))
    head = html.Thead(html.Tr([html.Th(h, style={**_CELL, **_MUTED, "textAlign": "left"}) for h in
                               ("Name", "Legs", "Level", "Fires when price", "Live price", "Distance", "Status", "Created (UTC)", "")]))
    return html.Div(html.Table([head, html.Tbody(rows)], style={"width": "100%", "color": COLORS["TEXT_PRIMARY"]}), style={"overflowX": "auto"})


def level_alerts_layout() -> html.Div:
    return html.Div([
        html.H3("🔔 Price Alerts", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "4px"}),
        html.Div("Make a structure, type a price level, and get an alert (in the app and in Teams) when its live price gets there. "
                 "Each alert fires once and stays in the list as triggered until you delete it.", style={**_MUTED, "marginBottom": "16px"}),
        _form(),
        dbc.Card(dbc.CardBody([_heading("Your alerts"), html.Div(id="la-table")]), style=_CARD),
        dcc.Store(id="la-refresh", data=0),
    ], style={"padding": "24px"})
