"""Curve Kinks page callbacks: render the selected curve, snapshot list, log, backtest, thresholds.

The maths and the alerting run in core.curve_service on every live poll whether or not this
page is open; these callbacks only read its latest state (no API calls, no recomputation).
"""

from datetime import timezone

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

from core.curve_backtest import run_backtest
from core.curve_calendar import PRODUCTS
from core.curve_settings import save_params, validate_params
from ui.container import container
from ui.layouts.curve_tab import (
    PARAM_FIELDS,
    PLACEHOLDER,
    build_backtest_table,
    build_change_table,
    build_curve_figure,
    build_kinks_table,
    build_plan_cards,
    build_log_table,
    build_quality,
    build_strength_figure,
    empty_figure,
)
from ui.layouts.shell import COLORS

_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}


def _status_line(service, state) -> html.Div:
    """Update time, background-job status, counts and any buffer-day banner."""
    if state.updated_at is None:
        return html.Div(PLACEHOLDER)
    parts = [f"Updated {state.updated_at.astimezone(timezone.utc).strftime('%H:%M:%S')} UTC"]
    parts += [msg for msg in service.job_messages.values()]
    parts.append(f"{len(state.kinks)} kink(s) now")
    children = [html.Span(" · ".join(parts))]
    for flag in state.quality:
        if flag.severity == "BUFFER":
            children.append(html.Div(f"⚠️ {flag.text}", style={"color": COLORS["ACCENT_RED"], "fontWeight": "bold", "marginTop": "4px"}))
    return html.Div(children)


def render_curve(n_intervals, product, family, snapshot_date):
    """Redraw the chosen curve and refresh the tables from the engine's latest state."""
    service = container.curve_service
    if service is None:
        message = html.Div("The curve engine is not running.", style=_MUTED)
        return empty_figure("Curve engine not running"), empty_figure(""), message, message, message, message, message
    state = service.state()
    params = service.params()
    view = state.families.get((product, family))
    snapshot = service.snapshots.load(product, snapshot_date, family) if snapshot_date else None
    return (
        build_curve_figure(view, snapshot, snapshot_date),
        build_strength_figure(view, params),
        _status_line(service, state),
        build_kinks_table(state.kinks),
        build_change_table(view),
        build_quality(state.quality),
        build_plan_cards(state.kinks),
    )


def snapshot_options(product):
    """Snapshot dates available for the product, newest first."""
    service = container.curve_service
    if service is None:
        return []
    return [{"label": d, "value": d} for d in service.snapshots.dates(product)]


def render_log(n_intervals):
    if container.repository is None:
        raise PreventUpdate
    return build_log_table(container.repository.get_kink_events(200))


def run_backtest_callback(n_clicks, product, family, days):
    if not n_clicks:
        raise PreventUpdate
    service = container.curve_service
    if service is None:
        return html.Div("The curve engine is not running.", style=_MUTED)
    try:
        test_days = int(days or 250)
    except (TypeError, ValueError):
        return html.Div("Days to test must be a number.", style={"color": COLORS["ACCENT_RED"]})
    history = service.history.matrix(product, family, service.n)
    return build_backtest_table(run_backtest(product, family, history, service.params(), test_days=max(30, test_days)))


def save_thresholds(n_clicks, *values):
    """Validate and store the threshold form."""
    if not n_clicks:
        raise PreventUpdate
    names = [name for name, *_ in PARAM_FIELDS]
    raw = dict(zip(names, values[: len(names)]))
    raw["alert_min_priority"], raw["alerts_enabled"] = values[len(names)], values[len(names) + 1]
    selected = values[len(names) + 2:]  # one list of curve types per product, in PRODUCTS order
    raw["alert_structures"] = [f"{code}:{family}" for code, families in zip(PRODUCTS, selected) for family in families or []]
    try:
        params = validate_params(raw)
    except ValueError as exc:
        return html.Span(f"❌ Not saved: {exc}", style={"color": COLORS["ACCENT_RED"]})
    save_params(container.repository, params)
    return html.Span("✅ Saved; used from the next compute.", style={"color": COLORS["ACCENT_GREEN"]})


def register_curve_callbacks(app) -> None:
    app.callback(
        Output("curve-graph", "figure"),
        Output("curve-strength", "figure"),
        Output("curve-status", "children"),
        Output("curve-kinks-wrap", "children"),
        Output("curve-change-wrap", "children"),
        Output("curve-quality-wrap", "children"),
        Output("curve-plans-wrap", "children"),
        Input("curve-refresh", "n_intervals"),
        Input("curve-product", "value"),
        Input("curve-family", "value"),
        Input("curve-snapshot", "value"),
    )(render_curve)

    app.callback(Output("curve-snapshot", "options"), Input("curve-product", "value"))(snapshot_options)

    app.callback(Output("curve-log-wrap", "children"), Input("curve-refresh", "n_intervals"))(render_log)

    app.callback(
        Output("curve-bt-output", "children"),
        Input("curve-bt-run", "n_clicks"),
        State("curve-bt-product", "value"),
        State("curve-bt-family", "value"),
        State("curve-bt-days", "value"),
        prevent_initial_call=True,
    )(run_backtest_callback)

    app.callback(
        Output("curve-params-message", "children"),
        Input("curve-params-save", "n_clicks"),
        *(State(f"curve-param-{name}", "value") for name, *_ in PARAM_FIELDS),
        State("curve-param-alert_min_priority", "value"),
        State("curve-param-alerts_enabled", "value"),
        *(State(f"curve-alert-{code}", "value") for code in PRODUCTS),
        prevent_initial_call=True,
    )(save_thresholds)
