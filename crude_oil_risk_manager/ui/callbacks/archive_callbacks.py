"""Archive tab callbacks: load, filter, summarize and show the detail of closed structures.

Strictly read-only: the repository is only queried, never written. The row building,
filtering and statistics live in core.archive.
"""

from datetime import date

from dash import Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate

from core.archive import build_archive_rows, filter_archive_rows, summarize
from core.models import StructureStatus
from ui.container import container
from ui.layouts.archive_tab import (
    CARD_STYLE,
    NO_ARCHIVE_TEXT,
    NO_MATCH_TEXT,
    STAT_CARDS,
    build_detail,
    build_stats,
    build_table_rows,
)


_PATH = "/archive"
_STYLED_STATS = ("total-pnl", "win-rate")  # the cards whose colour follows the value
_DEFAULTS = {"start": None, "end": None, "type": "all", "symbol": "", "pnl": "all"}


def _parse_date(value) -> date | None:
    """DatePickerRange values are ISO strings (possibly with a time part); anything else is no filter."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _load_rows() -> list[dict]:
    repository = container.repository
    closed = repository.get_all_structures(status_filter=[StructureStatus.CLOSED])
    trades = {s.structure_id: repository.get_trades_for_structure(s.structure_id) for s in closed}
    return build_archive_rows(closed, trades)


def refresh_archive(pathname, apply_clicks, clear_clicks, start, end, structure_type, symbol, pnl_filter):
    """Tab open, Apply Filters and Clear Filters: re-query, filter, and refresh the table and stats.

    Clear Filters also resets the controls to their defaults and shows the unfiltered set.
    """
    trigger = callback_context.triggered_id
    if trigger == "url" and pathname != _PATH:
        raise PreventUpdate
    if trigger == "archive-clear-btn":
        start, end, structure_type, symbol, pnl_filter = (_DEFAULTS[k] for k in ("start", "end", "type", "symbol", "pnl"))
        resets = (start, end, structure_type, symbol, pnl_filter)
    else:
        resets = (no_update,) * 5
    if trigger == "url":  # opening the tab always starts unfiltered
        start = end = symbol = None
        structure_type, pnl_filter = "all", "all"

    everything = _load_rows()
    rows = filter_archive_rows(everything, _parse_date(start), _parse_date(end), structure_type, symbol, pnl_filter)
    message = "" if rows else (NO_ARCHIVE_TEXT if not everything else NO_MATCH_TEXT)
    stats = build_stats(summarize(rows))
    texts = [stats[suffix][0] for suffix, _, _ in STAT_CARDS]
    styles = [stats[suffix][1] for suffix in _STYLED_STATS]
    return (build_table_rows(rows), message, rows, *texts, *styles, *resets)


def select_row(active_cell, close_clicks):
    """A clicked row selects its structure; Close Detail clears the selection (and the active cell)."""
    if callback_context.triggered_id == "archive-detail-close":
        if not close_clicks:
            raise PreventUpdate
        return None, None
    structure_id = (active_cell or {}).get("row_id")
    if not structure_id:
        raise PreventUpdate
    return structure_id, no_update


def render_detail(selected_id, rows):
    """Show the detail panel for the selected row if it is still in the current (filtered) set."""
    row = next((r for r in rows or [] if r["structure_id"] == selected_id), None)
    if row is None:
        return None, {**CARD_STYLE, "display": "none"}
    return build_detail(row), {**CARD_STYLE, "display": "block"}


def register_archive_callbacks(app) -> None:
    """Attach the Archive tab callbacks to the Dash app."""
    app.callback(
        Output("archive-table", "data"),
        Output("archive-message", "children"),
        Output("archive-rows", "data"),
        *(Output(f"archive-stat-{suffix}", "children") for suffix, _, _ in STAT_CARDS),
        *(Output(f"archive-stat-{suffix}", "style") for suffix in _STYLED_STATS),
        Output("archive-dates", "start_date"),
        Output("archive-dates", "end_date"),
        Output("archive-type", "value"),
        Output("archive-symbol", "value"),
        Output("archive-pnl", "value"),
        Input("url", "pathname"),
        Input("archive-apply-btn", "n_clicks"),
        Input("archive-clear-btn", "n_clicks"),
        State("archive-dates", "start_date"),
        State("archive-dates", "end_date"),
        State("archive-type", "value"),
        State("archive-symbol", "value"),
        State("archive-pnl", "value"),
    )(refresh_archive)

    app.callback(
        Output("archive-selected", "data"),
        Output("archive-table", "active_cell"),
        Input("archive-table", "active_cell"),
        Input("archive-detail-close", "n_clicks"),
        prevent_initial_call=True,
    )(select_row)

    app.callback(
        Output("archive-detail", "children"),
        Output("archive-detail-card", "style"),
        Input("archive-selected", "data"),
        Input("archive-rows", "data"),
    )(render_detail)
