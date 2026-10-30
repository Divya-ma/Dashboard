"""Strategy Rollover page callbacks. They read core.rollover.RolloverService, which caches the result until the stored settlements change."""

import logging

from dash import Input, Output

from core.curve_calendar import FAMILIES, OUTRIGHT, product_spec
from core.rollover import MAX_BEFORE, MAX_COMPARISONS, MAX_EXPIRIES, MAX_GENERIC, RolloverOptions
from ui.container import container
from ui.layouts.curve_tab import empty_figure
from ui.layouts.rollover_tab import ROLLOVER_PRODUCTS, build_rollover_figure, build_status

logger = logging.getLogger(__name__)


def _whole(value, low: int, high: int, name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: enter a whole number from {low} to {high}") from None
    if number != int(number) or not low <= number <= high:
        raise ValueError(f"{name}: enter a whole number from {low} to {high}")
    return int(number)


def parse_options(family, generic, expiries, comparisons, before, zero) -> RolloverOptions:
    """The page's controls as RolloverOptions; ValueError with a readable message for a bad entry."""
    return RolloverOptions(
        generic=_whole(generic, 1, MAX_GENERIC, "Generic"),
        expiries=_whole(expiries, 1, MAX_EXPIRIES, "Expiries"),
        comparisons=_whole(comparisons, 1, MAX_COMPARISONS, "Comparisons"),
        before_days=_whole(before, 1, MAX_BEFORE, "Days before"),
        zero_base=bool(zero),
        family=family if family in FAMILIES else OUTRIGHT,
    )


def render(n_intervals, product, family, generic, expiries, comparisons, before, zero):
    service = container.rollover
    if service is None:
        return empty_figure("The rollover service is not running"), build_status(None, "The rollover service is not running")
    if product not in ROLLOVER_PRODUCTS:
        return empty_figure("Choose a product"), build_status(None)
    try:
        result = service.view(product, parse_options(family, generic, expiries, comparisons, before, zero))
    except ValueError as exc:
        return empty_figure(str(exc)), build_status(None, str(exc))
    except Exception:  # noqa: BLE001 - a bad render must not leave the page blank
        logger.exception("Strategy rollover failed")
        return empty_figure("Could not build this chart; see the log"), build_status(None, "Could not build this chart; see the log")
    return build_rollover_figure(result), build_status(result)


def controls_disabled(product):
    """A CL-BZ line is its front box: no curve type or generic to choose."""
    return product in ROLLOVER_PRODUCTS and product_spec(product).is_box


def register_rollover_callbacks(app) -> None:
    app.callback(
        Output("rollover-graph", "figure"), Output("rollover-status", "children"), Input("rollover-refresh", "n_intervals"),
        Input("rollover-product", "value"), Input("rollover-family", "value"), Input("rollover-generic", "value"),
        Input("rollover-expiries", "value"), Input("rollover-comparisons", "value"), Input("rollover-before", "value"),
        Input("rollover-zero", "value"),
    )(render)
    app.callback(
        Output("rollover-family", "disabled"), Output("rollover-generic", "disabled"), Output("rollover-before", "disabled"),
        Input("rollover-product", "value"),
    )(lambda product: (controls_disabled(product),) * 3)
