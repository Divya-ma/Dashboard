"""Correlation Heatmap callbacks: mode toggle, watchlist editing and the compute action.

The maths lives in core.correlation (series building, alignment, Pearson matrix);
these functions gather inputs, call it and render the result. The data loader and
repository come from ui.container.
"""

from datetime import datetime, timezone

from dash import ALL, Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from core.correlation import (
    compute_watchlist_correlation,
    normalize_instrument_symbol,
    watchlist_item_warning,
)
from core.structure_view import statuses_for_filter
from ui.container import container
from ui.layouts.correlation_tab import (
    build_heatmap_figure,
    empty_figure,
    render_watchlist,
)
from ui.layouts.shell import COLORS

_SHOWN: dict = {}
_HIDDEN = {"display": "none"}


def _open_structures() -> dict:
    """Open structures by id (status OPEN; PARTIALLY_CLOSED is legacy and treated as open)."""
    structures = container.repository.get_all_structures(status_filter=statuses_for_filter("open"))
    return {s.structure_id: s for s in structures}


def _legs_summary(structure) -> str:
    return " / ".join(f"{leg.ratio:+d} {leg.contract.symbol}" for leg in structure.legs)


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------


def toggle_mode(mode):
    """Show the symbol box for Instrument mode, the structure dropdown for Structure mode."""
    is_structure = mode == "structure"
    return (_HIDDEN if is_structure else _SHOWN), (_SHOWN if is_structure else _HIDDEN)


def populate_structure_options(pathname):
    """Open structures as dropdown options (name + legs summary), refreshed whenever the tab opens."""
    if pathname != "/correlation":
        raise PreventUpdate
    return [
        {"label": f"{s.name} — {_legs_summary(s)}", "value": sid} for sid, s in _open_structures().items()
    ]


def add_item(n_clicks, mode, instrument, structure_id, watchlist):
    """Append an instrument or an open structure. Duplicates are ignored silently."""
    if not n_clicks:
        raise PreventUpdate
    items = list(watchlist or [])

    if mode == "structure":
        if not structure_id:
            return no_update, "Select a structure first.", no_update
        structure = _open_structures().get(structure_id)
        if structure is None:
            return no_update, "That structure is no longer open.", no_update
        new = {"type": "structure", "key": structure_id, "label": structure.name}
        taken = {item["label"] for item in items}
        if new["label"] in taken:  # two structures may share a name; keep matrix labels unique
            new["label"] = f"{structure.name} ({structure_id[:4]})"
        cleared = no_update
    else:
        symbol = normalize_instrument_symbol(instrument)
        if not symbol:
            return no_update, "Enter an exchange symbol first.", no_update
        new = {"type": "instrument", "key": symbol, "label": symbol}
        cleared = ""

    if any(item["type"] == new["type"] and item["key"] == new["key"] for item in items):
        return no_update, "", cleared
    return [*items, new], "", cleared


def remove_item(remove_clicks, watchlist):
    """Remove the clicked row. Freshly rendered buttons (no click yet) are ignored."""
    trigger = callback_context.triggered_id
    if not isinstance(trigger, dict) or not callback_context.triggered[0]["value"]:
        raise PreventUpdate
    index = trigger["index"]
    items = list(watchlist or [])
    if not 0 <= index < len(items):
        raise PreventUpdate
    del items[index]
    return items


def render_watchlist_rows(watchlist):
    """Watchlist rows, each with a warning if its data is missing (checked against the local Parquet)."""
    items = watchlist or []
    loader = container.data_loader
    structures = _open_structures() if any(i["type"] == "structure" for i in items) else {}
    warnings = {
        f"{item['type']}:{item['key']}": watchlist_item_warning(item, structures, loader) for item in items
    } if loader is not None else {}
    return render_watchlist(items, warnings)


# ----------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------


def _status(text: str, is_error: bool = False) -> html.Div:
    return html.Div(text, style={"color": COLORS["ACCENT_RED"] if is_error else COLORS["TEXT_SECONDARY"]})


def compute_heatmap(n_clicks, watchlist, lookback):
    """Build the watchlist series, correlate them and render the heatmap and status line."""
    if not n_clicks:
        raise PreventUpdate
    lookback_days = int(lookback)
    items = watchlist or []
    structures = _open_structures() if any(i["type"] == "structure" for i in items) else {}
    result = compute_watchlist_correlation(items, structures, lookback_days, container.data_loader)

    skipped = f" Skipped: {', '.join(result.skipped)}." if result.skipped else ""
    if result.error:
        return empty_figure(result.error, is_error=True), _status(f"⚠️ {result.error}.{skipped}", is_error=True)

    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    text = f"Last computed {now} UTC · {result.observations} days."
    if result.observations < lookback_days:
        text += f" Only {result.observations} common days available (requested {lookback_days}d)."
    return build_heatmap_figure(result.matrix, lookback_days), _status(text + skipped)


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_correlation_callbacks(app) -> None:
    """Attach the Correlation tab callbacks to the Dash app."""
    app.callback(
        Output("corr-instrument-group", "style"),
        Output("corr-structure-group", "style"),
        Input("corr-mode", "value"),
        prevent_initial_call=True,
    )(toggle_mode)

    app.callback(
        Output("corr-structure-select", "options"),
        Input("url", "pathname"),
    )(populate_structure_options)

    app.callback(
        Output("corr-watchlist", "data"),
        Output("corr-add-message", "children"),
        Output("corr-instrument-input", "value"),
        Input("corr-add-btn", "n_clicks"),
        State("corr-mode", "value"),
        State("corr-instrument-input", "value"),
        State("corr-structure-select", "value"),
        State("corr-watchlist", "data"),
        prevent_initial_call=True,
    )(add_item)

    app.callback(
        Output("corr-watchlist", "data", allow_duplicate=True),
        Input({"type": "corr-remove", "index": ALL}, "n_clicks"),
        State("corr-watchlist", "data"),
        prevent_initial_call=True,
    )(remove_item)

    app.callback(
        Output("corr-watchlist-list", "children"),
        Input("corr-watchlist", "data"),
    )(render_watchlist_rows)

    app.callback(
        Output("corr-heatmap", "figure"),
        Output("corr-status", "children"),
        Input("corr-compute-btn", "n_clicks"),
        State("corr-watchlist", "data"),
        State("corr-lookback", "value"),
        prevent_initial_call=True,
    )(compute_heatmap)
