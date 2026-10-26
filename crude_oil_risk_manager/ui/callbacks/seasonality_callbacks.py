"""Seasonality page callbacks. They read core.seasonality.SeasonalityService, which caches the seasonal analysis until the
stored settlements change and takes live prices from Curve Compare (whose quote polling runs while this page is open)."""

import logging

from dash import Input, Output, ctx

from core.curve_calendar import DFLY
from core.curve_compare import DEFAULT_DFLY_METHOD, DFLY_METHODS, MAX_MONTHS, min_months
from core.seasonality import BAND_METHODS, MAX_DAYS, MAX_YEARS, MEAN_SD, SeasonOptions
from ui.container import container
from ui.layouts.curve_tab import empty_figure
from ui.layouts.seasonality_tab import LEVEL_CHANGE, PLACEHOLDER, build_season_figure, build_status, table_rows

logger = logging.getLogger(__name__)

INPUTS = [
    Input("season-product", "value"), Input("season-family", "value"), Input("season-months", "value"),
    Input("season-years", "value"), Input("season-days", "value"), Input("season-method", "value"),
    Input("season-band", "value"), Input("season-z", "value"), Input("season-run", "value"),
    Input("season-excluded", "value"), Input("season-level", "value"), Input("season-reference", "value"),
]


def _whole(value, low: int, high: int, name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: enter a whole number from {low} to {high}") from None
    if number != int(number) or not low <= number <= high:
        raise ValueError(f"{name}: enter a whole number from {low} to {high}")
    return int(number)


def parse_months(value, minimum: int = 1) -> int:
    return _whole(value, minimum, MAX_MONTHS, "Months")


def parse_options(years, days, method, band, z, run, excluded, level, reference) -> SeasonOptions:
    """The page's controls as SeasonOptions; ValueError with a readable message for a bad entry."""
    try:
        z_value = float(z)
    except (TypeError, ValueError):
        raise ValueError("Flag beyond: enter a number of standard deviations") from None
    if not 0.5 <= z_value <= 10:
        raise ValueError("Flag beyond: enter a number from 0.5 to 10")
    rebase = _whole(reference, 5, MAX_DAYS, "Reference day") if level == LEVEL_CHANGE else None
    return SeasonOptions(
        years=_whole(years, 1, MAX_YEARS, "Earlier years"),
        max_days=_whole(days, 10, MAX_DAYS, "Days before expiry"),
        band=band if band in BAND_METHODS else MEAN_SD,
        z=z_value,
        run_days=_whole(run, 1, 60, "Days in a row"),
        excluded=tuple(sorted(int(y) for y in (excluded or []))),
        rebase_days=rebase,
        dfly_method=method if method in DFLY_METHODS else DEFAULT_DFLY_METHOD,
    )


def _inputs(product, family, months, years, days, method, band, z, run, excluded, level, reference):
    return family, parse_months(months, min_months(family)), parse_options(years, days, method, band, z, run, excluded, level, reference)


def render_table(n_intervals, *controls):
    service = container.seasonality
    if service is None:
        return [], build_status(None, "The seasonality service is not running")
    product = controls[0]
    try:
        family, months, options = _inputs(*controls)
    except ValueError as exc:
        return [], build_status(None, str(exc))
    try:
        table = service.table(product, family, months, options)
    except Exception:  # noqa: BLE001 - a bad render must not leave the page blank
        logger.exception("Seasonality table failed")
        return [], build_status(None, "Could not build the table; see the log")
    return table_rows(table), build_status(table)


def selected_position(trigger, active_cell):
    """The row the user clicked (its position), or None when a control changed and the old row may no longer exist."""
    if trigger == "season-table" and active_cell:
        return active_cell.get("row_id")
    return None


def select_row(active_cell, *controls):
    return selected_position(ctx.triggered_id, active_cell)


def render_chart(n_intervals, selected, show_band, *controls):
    service = container.seasonality
    if service is None or selected is None:
        return empty_figure(PLACEHOLDER)
    product = controls[0]
    try:
        family, months, options = _inputs(*controls)
        row = service.row(product, family, months, int(selected), options)
    except ValueError as exc:
        return empty_figure(str(exc))
    except Exception:  # noqa: BLE001
        logger.exception("Seasonality chart failed")
        return empty_figure("Could not build this chart; see the log")
    return build_season_figure(row, bool(show_band))


def method_disabled(family):
    """The d-fly construction only matters for a d-fly."""
    return family != DFLY


def reference_disabled(level):
    return level != LEVEL_CHANGE


def register_seasonality_callbacks(app) -> None:
    app.callback(
        Output("season-table", "data"), Output("season-status", "children"), Input("season-refresh", "n_intervals"), *INPUTS,
    )(render_table)
    app.callback(
        Output("season-selected", "data"), Input("season-table", "active_cell"), *INPUTS,
    )(select_row)
    app.callback(
        Output("season-graph", "figure"), Input("season-refresh", "n_intervals"), Input("season-selected", "data"),
        Input("season-show-band", "value"), *INPUTS,
    )(render_chart)
    app.callback(Output("season-method", "disabled"), Input("season-family", "value"))(method_disabled)
    app.callback(Output("season-reference", "disabled"), Input("season-level", "value"))(reference_disabled)
