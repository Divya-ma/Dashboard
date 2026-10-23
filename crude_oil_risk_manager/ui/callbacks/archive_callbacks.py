"""Archive tab callbacks: load, filter, summarize and show the detail of closed structures.

Read-only apart from one action: deleting the closed structures the user ticked and confirmed (core.archive.delete_archived_structures).
The row building, filtering and statistics live in core.archive.
"""

import logging
from datetime import date

from dash import Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate

from core.archive_export import archive_to_excel
from core.journal_analysis import equity_curve, group_table, headline, separation, trade_table, two_way
from ui.layouts.archive_analysis import (
    build_bar_figure, build_cards, build_equity_figure, separation_records, table_records, two_way_records,
)
from core.archive import build_archive_rows, delete_archived_structures, filter_archive_rows, summarize
from core.models import StructureStatus
from core.pnl import calculate_net_realized_pnl
from core.user_settings import KEY_PNL_BASELINE, KEY_PNL_REALIZED_BASELINE
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


logger = logging.getLogger(__name__)

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
    return build_archive_rows(closed, trades, repository.get_journals())


def refresh_archive(pathname, apply_clicks, clear_clicks, deleted, start, end, structure_type, symbol, pnl_filter):
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


def render_analysis(rows, dimension):
    """The Analysis tab for the trades the filters leave: headline cards, equity curve, and the results grouped by one column."""
    frame = trade_table(rows or [])
    table = group_table(frame, dimension or "type")
    return build_cards(headline(frame)), build_equity_figure(equity_curve(frame)), build_bar_figure(table), table_records(table)


def render_two_way(rows, dim_a, dim_b):
    """Results for every pair of groups of two journal columns, for the trades the filters leave."""
    return two_way_records(two_way(trade_table(rows or []), dim_a or "curve", dim_b or "type"))


def render_separation(rows):
    """Which columns line up with winning (and whether that could be chance), for the trades the filters leave."""
    table, note = separation(trade_table(rows or []))
    return note, separation_records(table)


def export_to_excel(n_clicks, rows):
    """Download the trades the table shows (after its filters) as an Excel workbook."""
    if not n_clicks:
        raise PreventUpdate
    if not rows:
        raise PreventUpdate
    return dcc.send_bytes(archive_to_excel(rows), f"archive_{date.today().isoformat()}.xlsx")


BACKFILL_PER_CLICK = 3  # structures per click: each takes six /ohlc/ requests (three for the entry, three for the exit); 30 a minute are allowed


def backfill_market_data(n_clicks):
    """Fill the market columns of a few old trades from their own dates (call again for the rest)."""
    if not n_clicks:
        raise PreventUpdate
    if container.journal is None:
        return "The journal is not available."
    report = container.journal.backfill(limit=BACKFILL_PER_CLICK)
    if report["filled"] == 0 and report["left"] == 0:
        return "Every closed trade already has its market data."
    return f"Filled {report['filled']} trade(s); {report['left']} left. Click again for the rest, then reopen the tab to see them."


def fill_journal_editor(selected_id):
    """The selected trade's ideas, the ideas on offer (plus any the trade has that were later taken off the list) and payups."""
    if not selected_id or container.journal is None:
        return [], [], None, None, None, None, [], [], []
    journal = container.journal.journal(selected_id)

    def options(listed, kept):
        return [{"label": i, "value": i} for i in dict.fromkeys([*listed, *kept])]  # a pick taken off the list later still shows

    return (options(container.journal.idea_options(), journal["ideas"]), journal["ideas"], journal["payups"]["entry"]["total"], journal["payups"]["exit"]["total"],
            journal["conviction"], journal["exit_reason"], journal["mistakes"],
            options(container.journal.exit_reason_options(), [journal["exit_reason"]] if journal["exit_reason"] else []),
            options(container.journal.mistake_options(), journal["mistakes"]))


def save_journal_edits(n_clicks, selected_id, ideas, payup_entry, payup_exit, conviction=None, exit_reason=None, mistakes=None):
    if not n_clicks:
        raise PreventUpdate
    if not selected_id or container.journal is None:
        return "Select a trade first."
    try:
        container.journal.set_ideas(selected_id, ideas)
        journal = container.journal.journal(selected_id)
        container.journal.set_payups(selected_id, "entry", payup_entry, journal["payups"]["entry"]["legs"])
        container.journal.set_payups(selected_id, "exit", payup_exit, journal["payups"]["exit"]["legs"])
        container.journal.set_review(selected_id, conviction, exit_reason, mistakes)
    except (ValueError, TypeError) as exc:
        return f"Not saved: {exc}"
    return "Saved. Reopen the tab (or click Apply Filters) to see it in the table."


def _ticked_ids(virtual_data, selected_rows) -> list[str]:
    """Structure ids of the ticked rows (indices into the rows as displayed: sorted, filtered, on any page)."""
    rows = virtual_data or []
    return [rows[i]["id"] for i in selected_rows or [] if 0 <= i < len(rows)]


def open_delete(n_clicks, virtual_data, selected_rows):
    """Ask before deleting: how many, and how much realized PnL leaves Home's totals with them."""
    if not n_clicks:
        raise PreventUpdate
    ids = _ticked_ids(virtual_data, selected_rows)
    if not ids:
        return True, [html.Div("Tick the trades to delete first.", style={"color": "#f0b429"})], [], {"display": "none"}, True
    repository = container.repository
    lines, total = [], 0.0
    for sid in ids:
        structure = repository.get_structure(sid)
        if structure is None:
            continue
        pnl = calculate_net_realized_pnl(repository.get_trades_for_structure(sid))
        total += pnl
        lines.append(html.Li(f"{structure.name}: {'+' if pnl >= 0 else '-'}${abs(pnl):,.0f}"))
    reset_active = bool(repository.get_setting(KEY_PNL_BASELINE, 0.0) or repository.get_setting(KEY_PNL_REALIZED_BASELINE, 0.0))
    body = [
        html.P(f"Delete {len(lines)} closed trade(s) for good, with their trades and history?"),
        html.Ul(lines, style={"maxHeight": "180px", "overflowY": "auto"}),
        html.P(f"Home's realized PnL (all time and today's) will drop by {'+' if total >= 0 else '-'}${abs(total):,.0f}: their PnL is no longer counted.",
               style={"fontWeight": "bold"}),
        html.P("This cannot be undone.", style={"color": "#f0b429"}),
    ]
    return True, body, ids, ({"display": "block", "marginTop": "10px"} if reset_active else {"display": "none"}), False


def cancel_delete(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


def confirm_delete(n_clicks, ids, shift_baseline, counter):
    if not n_clicks or not ids:
        raise PreventUpdate
    try:
        report = delete_archived_structures(container.repository, ids, bool(shift_baseline))
    except Exception:  # noqa: BLE001
        logger.exception("Archive delete failed")
        return False, "Could not delete; see the server log.", no_update, no_update, no_update
    message = f"Deleted {len(report['deleted'])} trade(s); realized PnL {'+' if report['removed_pnl'] >= 0 else '-'}${abs(report['removed_pnl']):,.0f} no longer counted."
    if report["skipped"]:
        message += f" Not deleted (not closed): {', '.join(report['skipped'])}."
    return False, message, [], None, (counter or 0) + 1


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
        Input("archive-deleted", "data"),
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

    app.callback(
        Output("archive-delete-modal", "is_open"),
        Output("archive-delete-body", "children"),
        Output("archive-delete-ids", "data"),
        Output("archive-delete-baseline-wrap", "style"),
        Output("archive-delete-confirm", "disabled"),
        Input("archive-delete-btn", "n_clicks"),
        State("archive-table", "derived_virtual_data"),
        State("archive-table", "derived_virtual_selected_rows"),
        prevent_initial_call=True,
    )(open_delete)

    app.callback(
        Output("archive-delete-modal", "is_open", allow_duplicate=True),
        Input("archive-delete-cancel", "n_clicks"),
        prevent_initial_call=True,
    )(cancel_delete)

    app.callback(
        Output("archive-delete-modal", "is_open", allow_duplicate=True),
        Output("archive-delete-status", "children"),
        Output("archive-table", "selected_rows"),
        Output("archive-selected", "data", allow_duplicate=True),
        Output("archive-deleted", "data"),
        Input("archive-delete-confirm", "n_clicks"),
        State("archive-delete-ids", "data"),
        State("archive-delete-baseline", "value"),
        State("archive-deleted", "data"),
        prevent_initial_call=True,
    )(confirm_delete)

    app.callback(
        Output("archive-backfill-status", "children"),
        Input("archive-backfill-btn", "n_clicks"),
        prevent_initial_call=True,
    )(backfill_market_data)

    app.callback(
        Output("archive-ideas", "options"),
        Output("archive-ideas", "value"),
        Output("archive-payup-entry", "value"),
        Output("archive-payup-exit", "value"),
        Output("archive-conviction", "value"),
        Output("archive-exit-reason", "value"),
        Output("archive-mistakes", "value"),
        Output("archive-exit-reason", "options"),
        Output("archive-mistakes", "options"),
        Input("archive-selected", "data"),
    )(fill_journal_editor)

    app.callback(
        Output("archive-journal-status", "children"),
        Input("archive-journal-save", "n_clicks"),
        State("archive-selected", "data"),
        State("archive-ideas", "value"),
        State("archive-payup-entry", "value"),
        State("archive-payup-exit", "value"),
        State("archive-conviction", "value"),
        State("archive-exit-reason", "value"),
        State("archive-mistakes", "value"),
        prevent_initial_call=True,
    )(save_journal_edits)

    app.callback(
        Output("analysis-cards", "children"),
        Output("analysis-equity", "figure"),
        Output("analysis-bars", "figure"),
        Output("analysis-table", "data"),
        Input("archive-rows", "data"),
        Input("analysis-dimension", "value"),
    )(render_analysis)

    app.callback(
        Output("archive-export-download", "data"),
        Input("archive-export-btn", "n_clicks"),
        State("archive-rows", "data"),
        prevent_initial_call=True,
    )(export_to_excel)

    app.callback(
        Output("analysis-two-table", "data"),
        Input("archive-rows", "data"),
        Input("analysis-two-a", "value"),
        Input("analysis-two-b", "value"),
    )(render_two_way)

    app.callback(
        Output("analysis-sep-note", "children"),
        Output("analysis-sep-table", "data"),
        Input("archive-rows", "data"),
    )(render_separation)
