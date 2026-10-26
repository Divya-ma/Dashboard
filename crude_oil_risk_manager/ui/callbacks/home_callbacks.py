"""Home tab callbacks: metric cards, structures table and row-click navigation.

`store-portfolio-pnl` has exactly one writer, shell_callbacks.refresh_portfolio_pnl,
which runs on every tab (so alerts keep working away from Home). These callbacks
only render that store, and they refresh whenever it updates.
"""

import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from config.settings import settings
from core.user_settings import KEY_MARGIN_LIMIT
from ui.container import container
from ui.layouts.home import REALIZED_VALUE_STYLE
from ui.layouts.shell import COLORS

_STATUS_LABELS = {"open": "Open", "partially_closed": "Partially Closed"}
_SMALL_TEXT = {"fontSize": "12px", "color": COLORS["TEXT_SECONDARY"], "fontWeight": "normal", "marginTop": "4px"}
_HIDDEN = {"display": "none"}


def format_pnl(value: float) -> str:
    """'+$1,234' for gains (and zero), '-$1,234' for losses."""
    if value >= 0:
        return f"+${value:,.0f}"
    return f"-${abs(value):,.0f}"


def pnl_color(value: float) -> str:
    """Green for gains (and zero), red for losses."""
    return COLORS["ACCENT_GREEN"] if value >= 0 else COLORS["ACCENT_RED"]


def _pnl_span(value: float) -> html.Span:
    return html.Span(format_pnl(value), style={"color": pnl_color(value)})


def _unrealized_card(gross: float, transaction_costs: float) -> list:
    """Gross unrealized P&L (headline), plus a 'net of TC' line for costs already booked
    against the open legs (entry/add costs — a future exit cost isn't in either figure)."""
    net = gross - transaction_costs
    children = [_pnl_span(gross)]
    if transaction_costs:
        children.append(
            html.Div(
                ["Net of TC: ", html.Span(format_pnl(net), style={"color": pnl_color(net)})],
                style=_SMALL_TEXT,
            )
        )
    return children


def _structure_summary(entry: dict | None, per_structure: dict):
    """Name and PnL of a largest winner/loser, or a dash if there is none."""
    if not entry:
        return "—"
    name = per_structure.get(entry["structure_id"], {}).get("name", entry["structure_id"])
    return [html.Div(name, style={"color": COLORS["TEXT_PRIMARY"]}), _pnl_span(entry["pnl"])]


def _net_lots(net_lots_by_product: dict) -> str:
    """Compact 'CL: +5 | BRN: -3' breakdown."""
    if not net_lots_by_product:
        return "—"
    return " | ".join(f"{product}: {lots:+,g}" for product, lots in sorted(net_lots_by_product.items()))


def _realized_value(value: float | None) -> tuple[str, dict]:
    """Text and style for a realized-PnL bar value: green if positive, red if negative."""
    style = {**REALIZED_VALUE_STYLE}
    if value is None:
        return "—", style
    style["color"] = COLORS["TEXT_PRIMARY"] if value == 0 else pnl_color(value)
    return format_pnl(value), style


def _reset_note(baseline: float | None) -> str:
    """Small note under Total Realized PnL when Settings > Account Reset has a baseline active."""
    if not baseline:
        return ""
    return f"🔄 Reset active — shown relative to a ${baseline:,.0f} baseline (Settings > Account Reset)"


def _margin_card() -> list:
    """Margin usage is not tracked (contracts carry no margin data); only the configured limit is shown."""
    limit = container.repository.get_setting(KEY_MARGIN_LIMIT, settings.DEFAULT_MARGIN_LIMIT)
    return ["—", html.Div(f"Limit: ${limit:,.0f} · usage not tracked yet", style=_SMALL_TEXT)]


def _stale_banner(portfolio_pnl: dict) -> tuple[dict, list | str]:
    stale = portfolio_pnl.get("stale_symbols_in_use", [])
    missing = portfolio_pnl.get("missing_price_symbols", [])
    if not stale and not missing:
        return _HIDDEN, ""

    lines = []
    if stale:
        lines.append(html.Div(f"⚠️ Stale data detected for: {', '.join(stale)}"))
    if missing:
        lines.append(html.Div(f"⚠️ No live price for: {', '.join(missing)} (excluded from PnL)"))
    style = {
        "display": "block",
        "backgroundColor": COLORS["CARD_BG"],
        "border": f"1px solid {COLORS['ACCENT_YELLOW']}",
        "borderLeft": f"5px solid {COLORS['ACCENT_YELLOW']}",
        "borderRadius": "6px",
        "color": COLORS["ACCENT_YELLOW"],
        "padding": "16px",
        "height": "100%",
    }
    return style, lines


def update_home_metrics(portfolio_pnl):
    """Render the realized-PnL bar and the metric cards from the portfolio PnL store.

    The store is built by shell_callbacks.refresh_portfolio_pnl, which already loads
    open and closed structures, so the all-time and today's realized PnL arrive with it.
    """
    if not portfolio_pnl:
        empty_text, empty_style = _realized_value(None)
        return (empty_text, empty_style, "", empty_text, empty_style, "—", "—", "—", "—", "—", "—", "—", _HIDDEN, "")

    per_structure = portfolio_pnl.get("per_structure", {})
    banner_style, banner_children = _stale_banner(portfolio_pnl)
    total_text, total_style = _realized_value(portfolio_pnl.get("total_realized_all_time"))
    today_text, today_style = _realized_value(portfolio_pnl.get("todays_realized_pnl"))
    reset_note = _reset_note(portfolio_pnl.get("pnl_realized_baseline"))

    return (
        total_text,
        total_style,
        reset_note,
        today_text,
        today_style,
        _unrealized_card(portfolio_pnl["total_unrealized"], portfolio_pnl.get("total_open_costs", portfolio_pnl.get("total_transaction_costs", 0.0))),
        str(portfolio_pnl["open_structure_count"]),
        str(portfolio_pnl["open_leg_count"]),
        _net_lots(portfolio_pnl.get("net_lots_by_product", {})),
        _margin_card(),
        _structure_summary(portfolio_pnl.get("largest_winner"), per_structure),
        _structure_summary(portfolio_pnl.get("largest_loser"), per_structure),
        banner_style,
        banner_children,
    )


def update_stop_risk_card(portfolio_pnl):
    """The whole book's stop risk, root of r·C·rᵀ (core.portfolio_risk), with the sum of the stops and what to be aware of."""
    if not portfolio_pnl:
        return "—"
    info = portfolio_pnl.get("stop_risk")
    if info is None:
        return ["—", html.Div("Could not be calculated; see the log.", style=_SMALL_TEXT)]
    warning = {**_SMALL_TEXT, "color": COLORS["ACCENT_YELLOW"]}
    if info["value"] is None:
        text = "No open structures." if not info["open_structures"] else "Set a stop on a structure to see this."
        notes = [html.Div(text, style=_SMALL_TEXT)]
        if info["without_stop"]:
            notes.append(html.Div(f"No stop on: {', '.join(info['without_stop'])}", style=warning))
        return ["—", *notes]
    lines = [html.Span(f"${info['value']:,.0f}")]
    detail = f"{info['counted']} structure(s) · sum of the stops ${info['sum_of_stops']:,.0f}"
    if info.get("diversification") is not None and info["counted"] > 1:
        detail += f" · correlation takes off {info['diversification']:.0%}"
    lines.append(html.Div(detail, style=_SMALL_TEXT))
    if info["counted"] > 1:
        lines.append(html.Div(f"{info['rolling']}d rolling correlation, averaged over {info['lookback']}d", style=_SMALL_TEXT))
    if info["without_stop"]:
        lines.append(html.Div(f"No stop, left out: {', '.join(info['without_stop'])}", style=warning))
    if info["assumed_pairs"]:
        lines.append(html.Div(f"{info['assumed_pairs']} pair(s) assumed fully correlated (not enough price history)", style=warning))
    if info.get("error"):
        lines.append(html.Div(info["error"], style=warning))
    return lines


_CELL = {"padding": "4px 8px", "textAlign": "right", "whiteSpace": "nowrap"}
_HEAD = {**_CELL, "color": COLORS["TEXT_SECONDARY"], "fontSize": "11px", "textTransform": "uppercase", "fontWeight": "normal"}
_LEG_COLUMNS = ["Contract", "Side", "Lots", "Entry", "Live", "Risk", "Unrealized"]


def _price(value: float | None) -> str:
    return "—" if value is None else f"{value:,.2f}"


def _risk(value: float | None) -> str:
    return "—" if value is None else f"${value:,.0f}"


def _side_color(side: str) -> str:
    return {"BUY": COLORS["ACCENT_GREEN"], "SELL": COLORS["ACCENT_RED"]}.get(side, COLORS["TEXT_SECONDARY"])


def _stat(label: str, value) -> html.Div:
    return html.Div(
        [html.Div(label, style={"fontSize": "10px", "textTransform": "uppercase", "color": COLORS["TEXT_SECONDARY"]}),
         html.Div(value, style={"fontSize": "14px", "color": COLORS["TEXT_PRIMARY"]})],
        style={"marginRight": "18px"},
    )


def _leg_row(leg: dict) -> html.Tr:
    return html.Tr([
        html.Td(leg["symbol"], style={**_CELL, "textAlign": "left"}),
        html.Td(leg["side"], style={**_CELL, "color": _side_color(leg["side"])}),
        html.Td(f"{leg['lots']:,g}", style=_CELL),
        html.Td(_price(leg["entry"]), style=_CELL),
        html.Td(_price(leg["live"]), style=_CELL),
        html.Td(_risk(leg["risk"]), style=_CELL),
        html.Td("—" if leg["unrealized"] is None else _pnl_span(leg["unrealized"]), style=_CELL),
    ], style={"borderTop": f"1px solid {COLORS['BORDER_COLOR']}"})


def _packet_card(packet: dict) -> dbc.Card:
    title = dcc.Link(
        packet["name"], href=f"/structures/{packet['structure_id']}",
        style={"color": COLORS["TEXT_PRIMARY"], "fontWeight": "bold", "fontSize": "16px", "textDecoration": "none"},
    )
    stale = html.Span("  ⚠ stale price", style={"color": COLORS["ACCENT_YELLOW"], "fontSize": "11px"}) if packet.get("is_stale") else ""
    stats = html.Div([
        _stat("Side", html.Span(packet["side"], style={"color": _side_color(packet["side"]), "fontWeight": "bold"})),
        _stat("Lots", f"{packet['lots']:,g}"), _stat("Entry", _price(packet["entry"])), _stat("Live", _price(packet["live"])),
        _stat("Target", _price(packet.get("target"))), _stat("Stop", _price(packet.get("stop"))),
        _stat("Risk (1d VaR 95%)", _risk(packet["risk"])), _stat("Unrealized", _pnl_span(packet["unrealized"])),
    ], style={"display": "flex", "flexWrap": "wrap", "margin": "8px 0 10px"})
    legs = html.Table([
        html.Thead(html.Tr([html.Th(c, style={**_HEAD, "textAlign": "left" if c == "Contract" else "right"}) for c in _LEG_COLUMNS])),
        html.Tbody([_leg_row(leg) for leg in packet["legs"]]),
    ], style={"width": "100%", "fontSize": "12px"})
    return dbc.Card(dbc.CardBody([html.Div([title, stale]), stats, legs]), style={
        "backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}",
        "borderLeft": f"4px solid {_side_color(packet['side'])}", "height": "100%",
    })


def update_home_table(portfolio_pnl):
    """Live structures: one packet (card) per open structure with its legs; click the name to open the structure."""
    packets = (portfolio_pnl or {}).get("packets") or []
    if not packets:
        return [html.Div("No open structures.", style={"color": COLORS["TEXT_SECONDARY"]})]
    return dbc.Row([dbc.Col(_packet_card(p), xl=4, lg=6, className="mb-3") for p in packets])


def register_home_callbacks(app) -> None:
    """Attach the Home tab callbacks to the Dash app."""
    app.callback(
        Output("home-total-realized-pnl", "children"),
        Output("home-total-realized-pnl", "style"),
        Output("home-pnl-reset-note", "children"),
        Output("home-today-realized-pnl", "children"),
        Output("home-today-realized-pnl", "style"),
        Output("home-card-unrealized-pnl", "children"),
        Output("home-card-open-structures", "children"),
        Output("home-card-open-legs", "children"),
        Output("home-card-net-lots", "children"),
        Output("home-card-margin-used", "children"),
        Output("home-winner-card", "children"),
        Output("home-loser-card", "children"),
        Output("home-stale-banner", "style"),
        Output("home-stale-banner", "children"),
        Input("store-portfolio-pnl", "data"),
    )(update_home_metrics)

    app.callback(
        Output("home-structures-table", "children"),
        Input("store-portfolio-pnl", "data"),
    )(update_home_table)

    app.callback(
        Output("home-card-stop-risk", "children"),
        Input("store-portfolio-pnl", "data"),
    )(update_stop_risk_card)
