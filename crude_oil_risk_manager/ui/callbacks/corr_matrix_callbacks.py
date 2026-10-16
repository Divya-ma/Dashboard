"""Callbacks of the structure correlation matrix at the top of the Correlation tab (core.corr_matrix).

The matrix callback validates the inputs and publishes them in `cm-params`; the click chart and the hedge
curve read that store, so all three always describe the same target, base, window and period. The data is
the OI page's settlement store (no API call is made here, except that the daily OI/settlement update is
started if it is due, exactly as the OI page does).
"""

import logging
from datetime import date, datetime, timezone

import numpy as np
from dash import Input, Output, State, html, no_update

from core.corr_matrix import (
    as_of_for, build_structure_set, correlation_matrix, daily_history, default_beta_window, key_label, lookback_value,
    pair_daily, parse_key, rolling_beta, validate_range,
)
from ui.container import container
from ui.layouts.corr_matrix_view import build_history_figure, build_hedge_figure, build_matrix_figure
from ui.layouts.shell import COLORS

logger = logging.getLogger(__name__)

_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_ERROR = {"color": COLORS["ACCENT_RED"], "fontSize": "13px"}
_CACHE: dict[tuple, object] = {}
_CACHE_LIMIT = 24


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _whole(value, name: str, minimum: int, default: int | None = None, maximum: int | None = None) -> int:
    if value in (None, ""):
        if default is None:
            raise ValueError(f"{name} is required.")
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None
    if number != int(number) or number < minimum or (maximum is not None and number > maximum):
        raise ValueError(f"{name} must be a whole number from {minimum}" + (f" to {maximum}." if maximum else " up."))
    return int(number)


def _day(text) -> date | None:
    return date.fromisoformat(str(text)[:10]) if text else None


def validate_inputs(target, base, n, rolling, lookback, mode, start, end) -> dict:
    parse_key(target)
    parse_key(base)
    mode = mode or "current"
    start_day, end_day = _day(start), _day(end)
    validate_range(mode, start_day, end_day)
    return {
        "target": target, "base": base, "n": _whole(n, "Front structures", 2, 15, 30), "rolling": _whole(rolling, "Rolling window", 2),
        "lookback": _whole(lookback, "Lookback", 1), "mode": mode,
        "start": start_day.isoformat() if mode == "historical" else None, "end": end_day.isoformat() if mode == "historical" else None,
    }


def _structure_set(product_key: str, params: dict):
    """The first N structures of one curve as of the period's end, cached until the stored data changes."""
    service = container.oi_service
    product, family = parse_key(product_key)
    as_of = as_of_for(params["mode"], _day(params["end"]), _today())
    key = (service.store.version, product, family, params["n"], as_of)
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = build_structure_set(service.store.frame(product), product, family, params["n"], as_of, _today())
    return _CACHE[key]


def _sets(params: dict):
    return _structure_set(params["target"], params), _structure_set(params["base"], params)


def _ready() -> str | None:
    service = container.oi_service
    if service is None:
        return "The settlement store is not running."
    service.ensure_fresh()
    if not service.store.has_data():
        return "No settlement data downloaded yet. It is downloading now (the OI page shows the progress); come back in a few minutes."
    return None


def _period_text(params: dict, target, base) -> str:
    if params["mode"] == "historical":
        return (f"Historical range {params['start']} to {params['end']}: the {params['n']} front structures on {params['end']}; "
                f"every daily correlation inside the range is averaged.")
    return f"Current market: the {params['n']} front structures today; the last {params['lookback']} daily correlations are averaged."


def render_matrix(target, base, n, rolling, lookback, mode, start, end, current_ht, current_hb):
    """(matrix figure, status line, params, hedge dropdown options and values)."""
    problem = _ready()
    if problem:
        return build_matrix_figure(None, None, "", ""), html.Div(problem, style=_MUTED), no_update, no_update, no_update, no_update, no_update
    try:
        params = validate_inputs(target, base, n, rolling, lookback, mode, start, end)
    except (ValueError, TypeError) as exc:
        return no_update, html.Div(f"⚠️ {exc}", style=_ERROR), no_update, no_update, no_update, no_update, no_update
    try:
        target_set, base_set = _sets(params)
        values, days = correlation_matrix(target_set, base_set, params["rolling"], params["lookback"], _day(params["start"]))
    except ValueError as exc:
        return build_matrix_figure(None, None, "", ""), html.Div(f"⚠️ {exc}", style=_ERROR), no_update, [], [], None, None
    except Exception:  # noqa: BLE001
        logger.exception("Correlation matrix failed")
        return no_update, html.Div("Could not build the matrix; see the log.", style=_ERROR), no_update, no_update, no_update, no_update, no_update
    needed = params["rolling"] + params["lookback"] - 1
    empty = int(values.isna().to_numpy().sum())
    status = html.Div([
        html.Div(f"{_period_text(params, target_set, base_set)} Settlement data up to {target_set.levels.index[-1].date()}."),
        html.Div(f"Rolling window {params['rolling']} days"
                 + (f"; a pair needs {needed} shared days. " if params["mode"] == "current" else ". ")
                 + (f"{empty} cell(s) are blank: not enough shared history (a contract that has not started trading, or a window with no variance)." if empty else "")),
    ])
    hedge_target = [{"label": label, "value": label} for label in target_set.labels]
    hedge_base = [{"label": label, "value": label} for label in base_set.labels]
    ht = current_ht if current_ht in target_set.labels else target_set.labels[0]
    hb = current_hb if current_hb in base_set.labels else base_set.labels[min(1, len(base_set.labels) - 1)]
    return build_matrix_figure(values, days, key_label(params["target"]), key_label(params["base"])), status, params, hedge_target, hedge_base, ht, hb


def render_click_chart(click, params, chart_days):
    """The daily correlation of the clicked cell over the last N days."""
    if not params:
        return build_history_figure(None, "", None, "Click a cell of the matrix")
    try:
        point = (click or {}).get("points", [None])[0]
        if point is None:
            return build_history_figure(None, "", None, "Click a cell of the matrix")
        target_set, base_set = _sets(params)
        t, b = point["y"], point["x"]
        if t not in target_set.labels or b not in base_set.labels:
            return build_history_figure(None, "", None, "Click a cell of the matrix")
        days = _whole(chart_days, "Days shown", 5, 120)
        if t == b and params["target"] == params["base"]:
            return build_history_figure(None, "", None, "A structure against itself is always 1.00")
        full = pair_daily(target_set.changes[t], base_set.changes[b], params["rolling"])
        average, _ = lookback_value(full, params["lookback"], _day(params["start"]))
        series = daily_history(target_set, base_set, t, b, params["rolling"], days)
        title = f"{key_label(params['target'])} {t} against {key_label(params['base'])} {b}: {params['rolling']}-day correlation, last {len(series)} days"
        return build_history_figure(series, title, average, "Not enough shared history for this pair")
    except (ValueError, TypeError) as exc:
        return build_history_figure(None, "", None, str(exc))
    except Exception:  # noqa: BLE001
        logger.exception("Correlation chart failed")
        return build_history_figure(None, "", None, "Could not build the chart; see the log.")


def render_hedge(target_label, base_label, days, beta_window_days, params):
    """(text, figure): the current hedge ratio of two structures and its path over the last N days."""
    muted = html.Div("Pick two structures.", style=_MUTED)
    if not params or not target_label or not base_label:
        return muted, build_hedge_figure(None, "", "Pick two structures")
    try:
        days = _whole(days, "Days shown", 5, 60)
        window = _whole(beta_window_days, "Beta window", 5, default_beta_window(params["rolling"], params["lookback"]))
        target_set, base_set = _sets(params)
        if target_label not in target_set.labels or base_label not in base_set.labels:
            return muted, build_hedge_figure(None, "", "Pick two structures")
        if target_label == base_label and params["target"] == params["base"]:
            return html.Div("Pick two different structures.", style=_MUTED), build_hedge_figure(None, "", "Pick two different structures")
        frame = rolling_beta(target_set.changes[target_label], base_set.changes[base_label], window)
    except (ValueError, TypeError) as exc:
        return html.Div(f"⚠️ {exc}", style=_ERROR), build_hedge_figure(None, "", "")
    except Exception:  # noqa: BLE001
        logger.exception("Hedge ratio failed")
        return html.Div("Could not work out the hedge; see the log.", style=_ERROR), build_hedge_figure(None, "", "")
    if frame.empty:
        return html.Div(f"Not enough shared days for a {window}-day hedge ratio.", style=_MUTED), build_hedge_figure(None, "", "Not enough shared history")
    shown = frame.iloc[-days:]
    last = shown.iloc[-1]
    target_name, base_name = f"{key_label(params['target'])} {target_label}", f"{key_label(params['base'])} {base_label}"
    side = "the opposite side" if last["beta"] > 0 else "the same side"
    corr = last["corr"]
    text = html.Div([
        html.Div(f"Per 1 lot of {target_name}, trade {abs(last['beta']):.2f} lot(s) of {base_name} on {side} "
                 f"(as of {shown.index[-1].date()}, measured over the last {window} shared days).",
                 style={"color": COLORS["TEXT_PRIMARY"], "fontSize": "15px", "margin": "4px 0"}),
        html.Div(f"Over the {len(shown)} days shown the ratio ranged {shown['beta'].min():.2f} to {shown['beta'].max():.2f} (median {shown['beta'].median():.2f}). "
                 + (f"Correlation of the daily changes now {corr:.2f}: the hedge removes about {corr ** 2 * 100:.0f}% of the variance. " if np.isfinite(corr) else "")
                 + "Settlement changes; one lot of a spread, fly or d-fly is one lot of each leg.", style=_MUTED),
    ])
    return text, build_hedge_figure(shown, f"Hedge ratio: {base_name} per lot of {target_name} ({window}-day beta)", "")


def show_dates(mode):
    return {"display": "block"} if mode == "historical" else {"display": "none"}


def register_corr_matrix_callbacks(app) -> None:
    app.callback(Output("cm-dates-wrap", "style"), Input("cm-mode", "value"))(show_dates)

    app.callback(
        Output("cm-matrix", "figure"),
        Output("cm-status", "children"),
        Output("cm-params", "data"),
        Output("cm-hedge-target", "options"),
        Output("cm-hedge-base", "options"),
        Output("cm-hedge-target", "value"),
        Output("cm-hedge-base", "value"),
        Input("cm-target", "value"),
        Input("cm-base", "value"),
        Input("cm-n", "value"),
        Input("cm-rolling", "value"),
        Input("cm-lookback", "value"),
        Input("cm-mode", "value"),
        Input("cm-dates", "start_date"),
        Input("cm-dates", "end_date"),
        State("cm-hedge-target", "value"),
        State("cm-hedge-base", "value"),
    )(render_matrix)

    app.callback(
        Output("cm-chart", "figure"),
        Input("cm-matrix", "clickData"),
        Input("cm-params", "data"),
        Input("cm-chart-days", "value"),
    )(render_click_chart)

    app.callback(
        Output("cm-hedge-text", "children"),
        Output("cm-hedge-chart", "figure"),
        Input("cm-hedge-target", "value"),
        Input("cm-hedge-base", "value"),
        Input("cm-hedge-days", "value"),
        Input("cm-beta-window", "value"),
        Input("cm-params", "data"),
    )(render_hedge)
