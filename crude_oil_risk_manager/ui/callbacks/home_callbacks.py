"""Home tab callbacks: metric cards, structures table and row-click navigation.

`store-portfolio-pnl` has exactly one writer, shell_callbacks.refresh_portfolio_pnl,
which runs on every tab (so alerts keep working away from Home). These callbacks
only render that store, and they refresh whenever it updates.
"""

from dash import Input, Output, html
from dash.exceptions import PreventUpdate

from config.settings import settings
from core.user_settings import KEY_MARGIN_LIMIT
from ui.container import container
from ui.layouts.home import MAX_TABLE_ROWS
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


def _structure_summary(entry: dict | None, per_structure: dict):
    """Name and PnL of a largest winner/loser, or a dash if there is none."""
    if not entry:
        return "—"
    name = per_structure.get(entry["structure_id"], {}).get("name", entry["structure_id"])
    return [html.Div(name, style={"color": COLORS["TEXT_PRIMARY"]}), _pnl_span(entry["pnl"])]


def _net_lots(net_lots_by_product: dict) -> list | str:
    if not net_lots_by_product:
        return "—"
    return [
        html.Div(f"{product}  {lots:+,g}", style={"color": COLORS["TEXT_PRIMARY"]})
        for product, lots in sorted(net_lots_by_product.items())
    ]


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
    """Render all metric cards from the portfolio PnL store."""
    if not portfolio_pnl:
        return ("—", "—", "—", "—", "—", "—", "—", "—", _HIDDEN, "")

    per_structure = portfolio_pnl.get("per_structure", {})
    todays_pnl = portfolio_pnl.get("todays_pnl")
    banner_style, banner_children = _stale_banner(portfolio_pnl)

    return (
        _pnl_span(portfolio_pnl["total_pnl"]),
        "—" if todays_pnl is None else _pnl_span(todays_pnl),
        str(portfolio_pnl["open_structure_count"]),
        str(portfolio_pnl["open_leg_count"]),
        _margin_card(),
        _net_lots(portfolio_pnl.get("net_lots_by_product", {})),
        _structure_summary(portfolio_pnl.get("largest_winner"), per_structure),
        _structure_summary(portfolio_pnl.get("largest_loser"), per_structure),
        banner_style,
        banner_children,
    )


def update_home_table(portfolio_pnl):
    """Row data for the live structures table: open structures, best PnL first, max 20 rows."""
    if not portfolio_pnl:
        return []

    rows = [
        {
            "structure_id": structure_id,
            "name": info.get("name", structure_id),
            "products": ", ".join(info.get("products", [])),
            "unrealized": info["unrealized"],
            "realized": info["realized"],
            "total": info["total"],
            "status": _STATUS_LABELS.get(info.get("status"), info.get("status", "")),
            "days_held": info.get("days_held"),
        }
        for structure_id, info in portfolio_pnl.get("per_structure", {}).items()
    ]
    rows.sort(key=lambda row: row["total"], reverse=True)
    return rows[:MAX_TABLE_ROWS]


def navigate_to_structure(cell_clicked):
    """Go to /structures/{structure_id} when a table row is clicked."""
    structure_id = (cell_clicked or {}).get("rowId")
    if not structure_id:
        raise PreventUpdate
    return f"/structures/{structure_id}"


def register_home_callbacks(app) -> None:
    """Attach the Home tab callbacks to the Dash app."""
    app.callback(
        Output("home-card-total-pnl", "children"),
        Output("home-card-today-pnl", "children"),
        Output("home-card-open-structures", "children"),
        Output("home-card-open-legs", "children"),
        Output("home-card-margin-used", "children"),
        Output("home-card-net-lots", "children"),
        Output("home-winner-card", "children"),
        Output("home-loser-card", "children"),
        Output("home-stale-banner", "style"),
        Output("home-stale-banner", "children"),
        Input("store-portfolio-pnl", "data"),
    )(update_home_metrics)

    app.callback(
        Output("home-structures-table", "rowData"),
        Input("store-portfolio-pnl", "data"),
    )(update_home_table)

    app.callback(
        Output("url", "pathname"),
        Input("home-structures-table", "cellClicked"),
        prevent_initial_call=True,
    )(navigate_to_structure)
