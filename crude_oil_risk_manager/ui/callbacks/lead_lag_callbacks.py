"""Callbacks of the Correlation tab's Lead-Lag section (core.lead_lag / core.lead_lag_data).

The Compute button validates the inputs, fetches/caches the group's 5-minute bars (adapters.historical.vendor
.VendorHistoricalAdapter -- an API call only when the local cache does not already cover the window) and builds
the matrix; the click chart reads the matrix's own cached detail by the exact same key, so clicking never
re-fetches. Explicit "Compute" (never auto-wired to the controls) keeps this feature off the shared /ohlc/ rate
budget until someone actually asks for it.
"""

import logging
from datetime import date, datetime, timezone

from dash import Input, Output, State, html, no_update

from core.exceptions import CrudeOilRiskError
from core.lead_lag import describe_lead_lag, lead_lag_matrix
from core.lead_lag_data import ALL_GROUPS, BAR_MINUTES, DEFAULT_LOOKBACK_DAYS, MAX_LOOKBACK_DAYS, group_symbols, load_bar_diffs
from ui.container import container
from ui.layouts.lead_lag_view import PLACEHOLDER_TEXT, READING_COLORS, build_ccf_figure, build_matrix_figure, empty_figure
from ui.layouts.shell import COLORS

logger = logging.getLogger(__name__)

_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_ERROR = {"color": COLORS["ACCENT_RED"], "fontSize": "13px"}
_CACHE: dict[tuple, dict] = {}
_CACHE_LIMIT = 12


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _whole(value, name: str, minimum: int, default: int, maximum: int) -> int:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None
    if number != int(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be a whole number from {minimum} to {maximum}.")
    return int(number)


def validate_inputs(group, month, lookback) -> dict:
    if not group:
        raise ValueError("Pick a group first.")
    return {
        "group": group, "month": _whole(month, "Starting month", 1, 1, 24),
        "lookback": _whole(lookback, "Lookback", 1, DEFAULT_LOOKBACK_DAYS, MAX_LOOKBACK_DAYS),
        "as_of": _today().isoformat(),
    }


def _compute(params: dict) -> dict:
    """The matrix for these params, cached by (group, month, lookback, as_of) -- the click chart hits this same
    cache right after a Compute, so it never re-fetches. Raises RuntimeError (no token) or CrudeOilRiskError."""
    key = (params["group"], params["month"], params["lookback"], params["as_of"])
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
        symbols = group_symbols(params["group"], params["month"], _today())
        diffs = load_bar_diffs(symbols, container.historical_adapter, params["lookback"])
        lag_matrix, strength_matrix, detail = lead_lag_matrix(diffs)
        _CACHE[key] = {"lag_matrix": lag_matrix, "strength_matrix": strength_matrix, "detail": detail, "n_symbols": len(symbols), "n_usable": len(diffs)}
    return _CACHE[key]


def render_matrix(n_clicks, group, month, lookback):
    """(matrix figure, status, params for the click chart to key off)."""
    if container.historical_adapter is None:
        return empty_figure(PLACEHOLDER_TEXT), html.Div("Historical data is not available.", style=_ERROR), no_update
    try:
        params = validate_inputs(group, month, lookback)
        result = _compute(params)
    except ValueError as exc:
        return no_update, html.Div(f"⚠️ {exc}", style=_ERROR), no_update
    except RuntimeError as exc:  # no API token configured
        return empty_figure(PLACEHOLDER_TEXT), html.Div(str(exc), style=_ERROR), no_update
    except CrudeOilRiskError as exc:
        return empty_figure(PLACEHOLDER_TEXT), html.Div(f"⚠️ {exc}", style=_ERROR), no_update
    except Exception:  # noqa: BLE001
        logger.exception("Lead-lag matrix failed")
        return empty_figure(PLACEHOLDER_TEXT), html.Div("Could not build the matrix; see the log.", style=_ERROR), no_update
    missing = result["n_symbols"] - result["n_usable"]
    window = "4 consecutive months" if params["group"] != ALL_GROUPS else f"1 contract from each of {result['n_symbols']} products"
    status = html.Div([
        html.Div(f"{window}, last {params['lookback']} day(s) of 5-minute bars, as of {params['as_of']}."),
        html.Div(f"{missing} of {result['n_symbols']} contract(s) had no usable bars and are left out." if missing else ""),
    ])
    return build_matrix_figure(result["lag_matrix"], result["strength_matrix"], BAR_MINUTES), status, params


def render_click_detail(click, params):
    """(explanation text, CCF-by-lag figure) for the cell clicked in the matrix."""
    empty_text = html.Div("Click a cell of the matrix.", style=_MUTED)
    if not params:
        return empty_text, empty_figure("Click a cell of the matrix")
    point = (click or {}).get("points", [None])[0]
    if point is None:
        return empty_text, empty_figure("Click a cell of the matrix")
    row_label, col_label = point["y"], point["x"]
    if row_label == col_label:
        return html.Div("A contract against itself has no lead-lag.", style=_MUTED), empty_figure("Pick two different contracts")
    try:
        result = _compute(params)
    except (ValueError, RuntimeError, CrudeOilRiskError) as exc:
        return html.Div(f"⚠️ {exc}", style=_ERROR), empty_figure("")
    detail = result["detail"]
    entry = detail.get((row_label, col_label))
    flipped = entry is None
    if flipped:
        entry = detail.get((col_label, row_label))
    if entry is None:
        return html.Div("Not enough shared bars between these two for a lead-lag read.", style=_MUTED), empty_figure("Not enough shared data")
    a, b = (col_label, row_label) if flipped else (row_label, col_label)
    sentence = describe_lead_lag(a, b, entry, BAR_MINUTES)
    text = html.Div(sentence, style={"color": READING_COLORS.get(entry["reading"], COLORS["TEXT_PRIMARY"]), "fontSize": "15px", "margin": "4px 0"})
    return text, build_ccf_figure(entry["ccf"], entry, a, b, BAR_MINUTES)


def register_lead_lag_callbacks(app) -> None:
    app.callback(
        Output("ll-matrix", "figure"),
        Output("ll-status", "children"),
        Output("ll-params", "data"),
        Input("ll-compute-btn", "n_clicks"),
        State("ll-group", "value"),
        State("ll-month", "value"),
        State("ll-lookback", "value"),
        prevent_initial_call=True,
    )(render_matrix)

    app.callback(
        Output("ll-detail", "children"),
        Output("ll-ccf-chart", "figure"),
        Input("ll-matrix", "clickData"),
        Input("ll-params", "data"),
    )(render_click_detail)
