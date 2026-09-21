"""Shell callbacks: page routing, live price polling, refresh countdown and alert toasts.

Callbacks stay thin: calculations come from core/, data access from the
Repository, and adapters are reached through ui.container (populated by
ui/app.py). The callbacks are plain module-level functions; register_callbacks()
attaches them to the app, so this module never imports ui.app.
"""

import importlib
import logging
import time
from datetime import datetime, timezone

import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate

from adapters.base import APIError, AuthenticationError, SymbolTranslator
from config.settings import settings
from core.models import AlertLevel, StructureStatus, TradeEventType
from core.pnl import (
    build_pnl_record,
    calculate_portfolio_pnl,
    calculate_todays_pnl,
    calculate_todays_realized_pnl,
)
from core.user_settings import KEY_API_TOKEN, KEY_PNL_STOP
from ui.container import NO_UPDATE_LABEL, container
from ui.layouts.placeholder import (
    archive_layout,
    correlation_layout,
    exposure_layout,
    structures_layout,
    trade_analyzer_layout,
    under_construction,
    var_scenario_layout,
)
from ui.layouts.shell import COLORS, stale_indicator

logger = logging.getLogger(__name__)

# A poll newer than this is served from the server-side cache, so page loads and
# extra browser tabs cannot burn the API's 7-requests/minute budget.
LIVE_CACHE_TTL_SECONDS = 50
POLL_PERIOD_SECONDS = 60

_OPEN_STATUSES = [StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED]

# path -> (module in ui/layouts, layout function name)
_ROUTES = {
    "/": ("home", "home_layout"),
    "/structures": ("structures", "structures_layout"),
    "/correlation": ("correlation_tab", "correlation_layout"),
    "/exposure": ("exposure_tab", "exposure_layout"),
    "/var-scenario": ("var_tab", "var_scenario_layout"),
    "/trade-analyzer": ("idea_tab", "trade_analyzer_layout"),
    "/archive": ("archive_tab", "archive_layout"),
    "/settings": ("settings", "settings_layout"),
}

_PLACEHOLDER_LAYOUTS = {
    "structures_layout": structures_layout,
    "correlation_layout": correlation_layout,
    "exposure_layout": exposure_layout,
    "var_scenario_layout": var_scenario_layout,
    "trade_analyzer_layout": trade_analyzer_layout,
    "archive_layout": archive_layout,
}


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------


def _resolve_layout(module_name: str, function_name: str):
    """Return the real layout function if its module exists, else a placeholder."""
    module_path = f"ui.layouts.{module_name}"
    try:
        module = importlib.import_module(module_path)
        return getattr(module, function_name)
    except ImportError as exc:
        if getattr(exc, "name", None) != module_path:
            logger.exception("Layout module %s failed to import", module_path)
    except AttributeError:
        logger.warning("Layout module %s has no %s", module_path, function_name)
    return _PLACEHOLDER_LAYOUTS.get(function_name, under_construction)


def render_page(pathname, token_configured):
    """Route the URL to a page layout and show/hide the setup-required banner."""
    banner_style = {"display": "none"} if token_configured else {"display": "block"}

    path = pathname or "/"
    if len(path) > 1:
        path = path.rstrip("/")

    if path.startswith("/structures/"):
        path = "/structures"  # /structures/<structure_id> is handled by the structures page

    route = _ROUTES.get(path)
    if route is None:
        return (
            html.Div(
                f"404 — page not found: {path}",
                style={"color": COLORS["TEXT_PRIMARY"], "padding": "40px", "fontSize": "20px"},
            ),
            banner_style,
        )
    return _resolve_layout(*route)(), banner_style


# ----------------------------------------------------------------------
# Live prices
# ----------------------------------------------------------------------


def _is_translatable(symbol: str) -> bool:
    try:
        SymbolTranslator.internal_to_api(symbol)
        return True
    except ValueError:
        logger.warning("Skipping contract %r: symbol not recognized by the symbol translator", symbol)
        return False


def _price_map(live_prices: dict) -> dict[str, float]:
    return {symbol: data["price"] for symbol, data in (live_prices or {}).items()}


def _stale_symbols(live_prices: dict) -> list[str]:
    return [symbol for symbol, data in (live_prices or {}).items() if data.get("is_stale")]


def _load_open_positions():
    """Open/partially-closed structures and their trades, from the repository."""
    structures = container.repository.get_all_structures(status_filter=_OPEN_STATUSES)
    trades = {s.structure_id: container.repository.get_trades_for_structure(s.structure_id) for s in structures}
    return structures, trades


def _load_closed_positions():
    """Closed structures and their trades: their realized PnL stays in the all-time total."""
    closed = container.repository.get_all_structures(status_filter=[StructureStatus.CLOSED])
    trades = {s.structure_id: container.repository.get_trades_for_structure(s.structure_id) for s in closed}
    return closed, trades


def _save_pnl_records(live_prices: dict) -> None:
    """Persist a PnLRecord per open structure; structures with missing prices are skipped."""
    try:
        structures, trades = _load_open_positions()
        prices = _price_map(live_prices)
        stale = _stale_symbols(live_prices)
        summary = calculate_portfolio_pnl(structures, trades, prices, stale)
        for structure in structures:
            info = summary["per_structure"].get(structure.structure_id)
            if info is None or info["missing_prices"]:
                logger.warning(
                    "Not saving PnL record for %s: missing prices %s",
                    structure.name, info["missing_prices"] if info else "n/a",
                )
                continue
            record = build_pnl_record(
                structure.structure_id, structure.legs, trades[structure.structure_id], prices, stale
            )
            container.repository.save_pnl_record(record)
    except Exception:  # noqa: BLE001 - a PnL history failure must not break the price poll
        logger.exception("Failed to save PnL records")


def _last_known_stale():
    """Last cached prices with every entry flagged stale (used when a poll fails)."""
    cache = container.live_cache
    stale_prices = {symbol: {**data, "is_stale": True} for symbol, data in cache.prices.items()}
    return stale_prices, cache.updated_label, stale_indicator(True)


def fetch_live_prices(n_intervals):
    """Poll live prices for all saved contracts and update the sidebar status."""
    repository = container.repository
    cache = container.live_cache

    token = repository.get_setting(KEY_API_TOKEN, "")
    if not token:
        return {}, NO_UPDATE_LABEL, stale_indicator(True), False

    if cache.fetched_at is not None and time.monotonic() - cache.fetched_at < LIVE_CACHE_TTL_SECONDS:
        return cache.prices, cache.updated_label, stale_indicator(cache.is_stale), True

    symbols = sorted({c.symbol for c in repository.get_all_contracts() if _is_translatable(c.symbol)})
    if not symbols:
        return {}, NO_UPDATE_LABEL, stale_indicator(False), True

    try:
        live = container.live_adapter.get_live_prices(symbols)
    except AuthenticationError:
        logger.error("Live price poll rejected: invalid or expired access token")
        return (*_last_known_stale(), False)
    except (APIError, RuntimeError, ValueError) as exc:
        logger.error("Live price poll failed: %s", exc)
        return (*_last_known_stale(), True)

    prices = {
        symbol: {
            "price": p.price,
            "timestamp_iso": p.timestamp.isoformat(),
            "is_stale": p.is_stale,
            "open": p.raw_open,
            "high": p.raw_high,
            "low": p.raw_low,
            "volume": p.raw_volume,
        }
        for symbol, p in live.items()
    }
    missing = [s for s in symbols if s not in prices]
    for symbol in missing:
        if symbol in cache.prices:
            prices[symbol] = {**cache.prices[symbol], "is_stale": True}

    is_stale = bool(missing) or any(d["is_stale"] for d in prices.values())
    label = f"Last: {datetime.now().strftime('%H:%M:%S')}"
    cache.prices = prices
    cache.fetched_at = time.monotonic()
    cache.updated_label = label
    cache.is_stale = is_stale

    _save_pnl_records(prices)
    return prices, label, stale_indicator(is_stale), True


def update_countdown(countdown_intervals, poll_intervals):
    """Seconds remaining until the next live price poll."""
    seconds_since = (countdown_intervals or 0) % POLL_PERIOD_SECONDS
    remaining = POLL_PERIOD_SECONDS - seconds_since
    return f"Next refresh in: {remaining}s"


# ----------------------------------------------------------------------
# Portfolio PnL and alerts
# ----------------------------------------------------------------------


def _add_structure_details(summary: dict, structures, stale_symbols: list[str]) -> None:
    """Add display details (name, products, status, days held) and the stale/missing symbols in use."""
    for structure in structures:
        summary["per_structure"][structure.structure_id].update(
            name=structure.name,
            products=sorted({leg.contract.product for leg in structure.legs}),
            status=structure.status.value,
            days_held=structure.days_held,
        )
    in_use = {leg.contract.symbol for s in structures for leg in s.legs if leg.is_traded}
    summary["stale_symbols_in_use"] = sorted(in_use & set(stale_symbols))
    summary["missing_price_symbols"] = sorted(
        {sym for info in summary["per_structure"].values() for sym in info["missing_prices"]}
    )


def _todays_pnl(structures, trades, prices, stale, summary) -> float | None:
    """Portfolio PnL change since each structure's first snapshot today (UTC).

    Structures with a missing price or no snapshot yet today are skipped. Returns
    None if the calculation fails, so the UI shows "—" instead of a wrong number.
    """
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    total = 0.0
    try:
        for structure in structures:
            sid = structure.structure_id
            if summary["per_structure"][sid]["missing_prices"]:
                continue
            baseline = container.repository.get_first_pnl_since(sid, start_of_day)
            if baseline is None:
                continue
            current = build_pnl_record(sid, structure.legs, trades[sid], prices, stale)
            total += calculate_todays_pnl([baseline, current])
    except Exception:  # noqa: BLE001
        logger.exception("Could not calculate today's PnL")
        return None
    return total


def refresh_portfolio_pnl(n_intervals, live_prices):
    """Recalculate portfolio PnL from cached prices (no API call) into a JSON-safe dict."""
    try:
        structures, trades = _load_open_positions()
        closed, closed_trades = _load_closed_positions()
    except Exception:  # noqa: BLE001
        logger.exception("Could not load positions for PnL refresh")
        return no_update

    prices = _price_map(live_prices)
    stale = _stale_symbols(live_prices)
    summary = calculate_portfolio_pnl(structures, trades, prices, stale, closed, closed_trades)
    all_trades = [t for group in (*trades.values(), *closed_trades.values()) for t in group]
    summary["todays_realized_pnl"] = calculate_todays_realized_pnl(all_trades)
    _add_structure_details(summary, structures, stale)
    summary["todays_pnl"] = _todays_pnl(structures, trades, prices, stale, summary)
    summary["calculated_at"] = summary["calculated_at"].isoformat()
    summary["has_missing_prices"] = bool(summary["missing_price_symbols"])
    return summary


def _toast_for_alert(alert) -> dbc.Toast:
    icons = {AlertLevel.INFO: "primary", AlertLevel.WARNING: "warning", AlertLevel.CRITICAL: "danger"}
    return dbc.Toast(
        alert.body,
        id=f"toast-{alert.alert_id}",
        header=alert.title,
        icon=icons[alert.level],
        is_open=True,
        dismissable=True,
        style={"width": "350px", "marginBottom": "10px"},
    )


def check_alerts(n_intervals, portfolio_pnl):
    """Raise a toast (once per breach) when portfolio PnL falls below the stop threshold."""
    if not portfolio_pnl:
        return []

    total_pnl = portfolio_pnl.get("total_pnl")
    if total_pnl is None or portfolio_pnl.get("has_missing_prices"):
        # PnL built from partial prices could raise a false alarm; leave state untouched.
        return no_update

    threshold = float(container.repository.get_setting(KEY_PNL_STOP, settings.ALERT_PORTFOLIO_PNL_STOP))
    if total_pnl >= threshold:
        container.pnl_stop_alert_active = False
        return []
    if container.pnl_stop_alert_active:
        return no_update

    container.pnl_stop_alert_active = True
    alert = container.alert_manager.send_alert(
        AlertLevel.CRITICAL,
        "Portfolio P&L stop breached",
        f"Portfolio P&L ${total_pnl:,.0f} is below the stop threshold of ${threshold:,.0f}.",
    )
    return [_toast_for_alert(alert)]


# ----------------------------------------------------------------------
# Stop-loss / target price alerts
# ----------------------------------------------------------------------

_ENTRY_EVENTS = (TradeEventType.TRADE, TradeEventType.ADD)


def _price_alert_toast(alert) -> dbc.Toast:
    """Stop-loss toasts stay until dismissed (duration 0); target toasts close after 8 seconds."""
    critical = alert.level == AlertLevel.CRITICAL
    return dbc.Toast(
        alert.body,
        id=f"toast-{alert.alert_id}",
        header=alert.title,
        icon="danger" if critical else "success",
        duration=0 if critical else 8000,
        is_open=True,
        dismissable=True,
        style={"width": "380px", "marginBottom": "10px", "backgroundColor": COLORS["CARD_BG"]},
    )


def _still_open(toast) -> bool:
    """False for a toast the user already dismissed (its is_open was set to False client-side)."""
    props = toast.get("props", {}) if isinstance(toast, dict) else {}
    return props.get("is_open", True)


def check_price_alerts(live_prices, existing_toasts):
    """After each live price update, raise stop-loss / target alerts as persistent toasts.

    Uses its own toast container: `alert-toast-container` is rewritten every 5s by
    check_alerts, which would wipe a stop-loss toast that must stay until dismissed.
    """
    manager = container.alert_manager
    if not live_prices or manager is None:
        raise PreventUpdate
    structures, trades = _load_open_positions()
    open_trades = [
        t
        for group in trades.values()
        for t in group
        if t.event_type in _ENTRY_EVENTS and (t.stop_loss_price is not None or t.target_price is not None)
    ]
    if not open_trades:
        raise PreventUpdate
    alerts = manager.check_price_alerts(
        _price_map(live_prices), open_trades, {s.structure_id: s for s in structures}
    )
    if not alerts:
        raise PreventUpdate
    kept = [toast for toast in (existing_toasts or []) if _still_open(toast)]
    return kept + [_price_alert_toast(alert) for alert in alerts]


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_callbacks(app) -> None:
    """Attach all shell callbacks to the Dash app."""
    app.callback(
        Output("page-content", "children"),
        Output("banner-setup-required", "style"),
        Input("url", "pathname"),
        State("store-token-configured", "data"),
    )(render_page)

    app.callback(
        Output("store-live-prices", "data"),
        Output("sidebar-last-updated", "children"),
        Output("sidebar-stale-indicator", "children"),
        Output("store-token-configured", "data"),
        Input("interval-live-poll", "n_intervals"),
    )(fetch_live_prices)

    app.callback(
        Output("sidebar-refresh-countdown", "children"),
        Input("interval-countdown", "n_intervals"),
        State("interval-live-poll", "n_intervals"),
    )(update_countdown)

    app.callback(
        Output("store-portfolio-pnl", "data"),
        Input("interval-pnl-refresh", "n_intervals"),
        State("store-live-prices", "data"),
    )(refresh_portfolio_pnl)

    app.callback(
        Output("alert-toast-container", "children"),
        Input("interval-pnl-refresh", "n_intervals"),
        State("store-portfolio-pnl", "data"),
    )(check_alerts)

    app.callback(
        Output("price-alert-toast-container", "children"),
        Input("store-live-prices", "data"),
        State("price-alert-toast-container", "children"),
        prevent_initial_call=True,
    )(check_price_alerts)
