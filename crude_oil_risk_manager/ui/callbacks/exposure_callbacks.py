"""Exposure Map callbacks: load open structures on tab open, re-evaluate warnings on threshold change.

The aggregation lives in core.exposure_map; these functions load the structures,
call it and render the result.
"""

import math

from dash import Input, Output, html
from dash.exceptions import PreventUpdate

from core.exposure_map import build_exposure_map, concentration_warnings
from core.structure_view import statuses_for_filter
from ui.container import container
from ui.layouts.exposure_tab import (
    DEFAULT_THRESHOLD,
    build_alerts,
    build_bar_figure,
    build_table_rows,
)
from ui.layouts.shell import COLORS


def load_exposure(pathname):
    """On tab open: query the open structures, aggregate, and render the chart and table."""
    if pathname != "/exposure":
        raise PreventUpdate
    structures = container.repository.get_all_structures(status_filter=statuses_for_filter("open"))
    rows = build_exposure_map(structures)
    return build_bar_figure(rows), build_table_rows(rows), rows


def update_concentration_alerts(threshold, rows):
    """Re-render only the warning panel: on a threshold change, or when new exposure data arrives."""
    if isinstance(threshold, bool) or threshold is None or not math.isfinite(threshold) or threshold < 0:
        return html.Div("Enter a threshold of 0 or more.", style={"color": COLORS["ACCENT_YELLOW"]})
    return build_alerts(concentration_warnings(rows or [], threshold))


def register_exposure_callbacks(app) -> None:
    """Attach the Exposure Map callbacks to the Dash app."""
    app.callback(
        Output("exposure-bar-chart", "figure"),
        Output("exposure-table", "data"),
        Output("exposure-data", "data"),
        Input("url", "pathname"),
    )(load_exposure)

    app.callback(
        Output("exposure-alerts", "children"),
        Input("exposure-threshold", "value"),
        Input("exposure-data", "data"),
    )(update_concentration_alerts)
