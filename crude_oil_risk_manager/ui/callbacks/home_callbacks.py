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
        _unrealized_card(portfolio_pnl["total_unrealized"], portfolio_pnl.get("total_transaction_costs", 0.0)),
        str(portfolio_pnl["open_structure_count"]),
        str(portfolio_pnl["open_leg_count"]),
        _net_lots(portfolio_pnl.get("net_lots_by_product", {})),
        _margin_card(),
        _structure_summary(portfolio_pnl.get("largest_winner"), per_structure),
        _structure_summary(portfolio_pnl.get("largest_loser"), per_structure),
        banner_style,
        banner_children,
    )


_CELL = {"padding": "6px 10px", "textAlign": "right", "whiteSpace": "nowrap"}
_HEAD = {**_CELL, "color": COLORS["TEXT_SECONDARY"], "fontSize": "11px", "textTransform": "uppercase", "fontWeight": "normal"}
_COLUMNS = ["Structure", "Side", "Lots", "Entry", "Live", "Risk (1d VaR 95%)", "Unrealized PnL"]


def _price(value: float | None) -> str:
    return "—" if value is None else f"{value:,.2f}"


def _risk(value: float | None) -> str:
    return "—" if value is None else f"${value:,.0f}"


def _structure_row(row: dict) -> html.Tr:
    side_color = {"BUY": COLORS["ACCENT_GREEN"], "SELL": COLORS["ACCENT_RED"]}.get(row["side"], COLORS["TEXT_SECONDARY"])
    name = dcc.Link(row["name"], href=f"/structures/{row['structure_id']}", style={"color": COLORS["TEXT_PRIMARY"]})
    if row.get("is_stale"):
        name = html.Span([name, html.Span(" ⚠ stale", style={"color": COLORS["ACCENT_YELLOW"], "fontSize": "11px"})])
    return html.Tr([
        html.Td(name, style={**_CELL, "textAlign": "left"}),
        html.Td(row["side"], style={**_CELL, "color": side_color, "fontWeight": "bold"}),
        html.Td(f"{row['lots']:,g}", style=_CELL),
        html.Td(_price(row["entry"]), style=_CELL),
        html.Td(_price(row["live"]), style=_CELL),
        html.Td(_risk(row["risk"]), style=_CELL),
        html.Td(_pnl_span(row["unrealized"]), style=_CELL),
    ], style={"borderTop": f"1px solid {COLORS['BORDER_COLOR']}"})


def _packet_card(packet: dict) -> dbc.Card:
    header = html.Div([
        html.Span(packet["name"], style={"fontWeight": "bold", "fontSize": "15px"}),
        html.Span(f"  {len(packet['rows'])} structure{'s' if len(packet['rows']) != 1 else ''}", style=_SMALL_TEXT),
        html.Span([
            html.Span(f"Packet risk {_risk(packet['risk'])}", style={"color": COLORS["TEXT_SECONDARY"], "marginRight": "16px"}),
            _pnl_span(packet["unrealized"]),
        ], style={"float": "right"}),
    ], style={"marginBottom": "8px"})
    table = html.Table([
        html.Thead(html.Tr([html.Th(c, style={**_HEAD, "textAlign": "left" if c == "Structure" else "right"}) for c in _COLUMNS])),
        html.Tbody([_structure_row(r) for r in packet["rows"]]),
    ], style={"width": "100%", "fontSize": "13px"})
    return dbc.Card(dbc.CardBody([header, table]), className="mb-3", style={
        "backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}"})


def update_home_table(portfolio_pnl):
    """Live structures as packets (product + shape), each a table of its structures; click a name to open it."""
    packets = (portfolio_pnl or {}).get("packets") or []
    if not packets:
        return [html.Div("No open structures.", style={"color": COLORS["TEXT_SECONDARY"]})]
    return [_packet_card(p) for p in packets]


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
