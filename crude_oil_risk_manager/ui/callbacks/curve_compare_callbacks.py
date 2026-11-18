"""Curve Compare page callbacks. They only read core.curve_compare.CurveCompare, which polls quotes
in the background; opening or refreshing the page costs no API call itself."""

import logging

from dash import Input, Output, html

from core.curve_calendar import DFLY
from core.curve_compare import DEFAULT_DFLY_METHOD, DFLY_METHODS, MAX_MONTHS, min_months
from ui.container import container
from ui.layouts.curve_compare_tab import (
    build_change_figure, build_compare_figure, build_status, build_table, empty_figure,
)
from ui.layouts.shell import COLORS

logger = logging.getLogger(__name__)


def _notice(text: str):
    figure = empty_figure(text)
    return figure, empty_figure(""), html.Div(text, style={"color": COLORS["TEXT_SECONDARY"]}), html.Div()


def parse_months(value) -> int | None:
    """The months field as an int within 1..MAX_MONTHS, or None if it is empty or not a whole number."""
    try:
        months = int(value)
    except (TypeError, ValueError):
        return None
    return months if 1 <= months <= MAX_MONTHS and months == float(value) else None


def render_compare(n_intervals, product, months, family, method, snapshot_date=None):
    service = container.curve_compare
    if service is None:
        return _notice("The curve compare service is not running")
    n = parse_months(months)
    if n is None:
        return _notice(f"Enter a whole number of months from 1 to {MAX_MONTHS}")
    if n < min_months(family):
        return _notice(f"A {family} needs at least {min_months(family)} contracts")
    try:
        view = service.view(product, n, family, method if method in DFLY_METHODS else DEFAULT_DFLY_METHOD)
    except Exception:  # noqa: BLE001 - a bad render must not leave the page blank
        logger.exception("Curve compare failed")
        return _notice("Could not build this curve; see the log")
    snapshots = getattr(container.curve_compare, "snapshots", None)
    snapshot = snapshots.load(product, snapshot_date, family) if snapshots is not None and snapshot_date else None
    return build_compare_figure(view, snapshot, snapshot_date), build_change_figure(view), build_status(view), build_table(view)


def snapshot_options(product):
    """The days a curve snapshot was saved for the product, newest first."""
    store = getattr(container.curve_compare, "snapshots", None)
    return [] if store is None else [{"label": d, "value": d} for d in store.dates(product)]


def method_disabled(family):
    """The d-fly method only matters for a d-fly."""
    return family != DFLY


def register_curve_compare_callbacks(app) -> None:
    app.callback(
        Output("compare-graph", "figure"),
        Output("compare-change", "figure"),
        Output("compare-status", "children"),
        Output("compare-table-wrap", "children"),
        Input("compare-refresh", "n_intervals"),
        Input("compare-product", "value"),
        Input("compare-months", "value"),
        Input("compare-family", "value"),
        Input("compare-method", "value"),
        Input("compare-snapshot", "value"),
    )(render_compare)

    app.callback(Output("compare-snapshot", "options"), Output("compare-snapshot", "value"), Input("compare-product", "value"))(
        lambda product: (snapshot_options(product), None)
    )

    app.callback(Output("compare-method", "disabled"), Input("compare-family", "value"))(method_disabled)
