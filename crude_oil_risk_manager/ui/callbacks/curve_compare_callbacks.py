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


def render_compare(n_intervals, product, months, family, method):
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
    return build_compare_figure(view), build_change_figure(view), build_status(view), build_table(view)


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
    )(render_compare)

    app.callback(Output("compare-method", "disabled"), Input("compare-family", "value"))(method_disabled)
