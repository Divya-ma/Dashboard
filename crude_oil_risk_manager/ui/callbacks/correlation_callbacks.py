"""Correlation tab callbacks: mode toggle, watchlist editing and the four compute actions
(Heatmap, Time Series, Year Overlay, Summary Table).

The maths lives in core.correlation (series building, alignment, Pearson matrix, rolling
correlation, pair summary); these functions gather inputs, call it and render the result.
The data loader and repository come from ui.container.
"""

import base64
from datetime import date, datetime, timezone

from dash import ALL, Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from core.correlation import (
    build_watchlist_series,
    compute_watchlist_correlation,
    excel_item_key,
    normalize_instrument_symbol,
    rolling_correlation_from_series,
    watchlist_item_missing_symbols,
    watchlist_item_warning,
    watchlist_pair_summary,
    year_overlay_frame,
)
from core.exceptions import CrudeOilRiskError
from core.structure_builder import backfill_symbols_async
from core.structure_view import statuses_for_filter
from ui.container import container
from ui.layouts.correlation_tab import (
    DEFAULT_TS_WINDOWS,
    MIN_ITEMS_TEXT,
    build_heatmap_figure,
    build_summary_table,
    build_time_series_figure,
    build_year_overlay_figure,
    default_rolling_window,
    empty_figure,
    render_watchlist,
)
from ui.layouts.shell import COLORS

_SHOWN: dict = {}
_HIDDEN = {"display": "none"}
_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}


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
    """Show the matching input group for the selected watchlist source mode."""
    return (
        _SHOWN if mode == "instrument" else _HIDDEN,
        _SHOWN if mode == "structure" else _HIDDEN,
        _SHOWN if mode == "excel" else _HIDDEN,
    )


def populate_structure_options(pathname):
    """Open structures as dropdown options (name + legs summary), refreshed whenever the tab opens."""
    if pathname != "/correlation":
        raise PreventUpdate
    return [
        {"label": f"{s.name} — {_legs_summary(s)}", "value": sid} for sid, s in _open_structures().items()
    ]


def add_item(n_clicks, mode, instrument, structure_id, excel_file, excel_sheet, excel_columns, watchlist):
    """Append an instrument, an open structure, or one or more Excel columns at once.

    Duplicates are ignored silently (instrument/structure) or skipped from the batch (excel).
    """
    if not n_clicks:
        raise PreventUpdate
    items = list(watchlist or [])

    if mode == "structure":
        if not structure_id:
            return no_update, "Select a structure first.", no_update, no_update
        structure = _open_structures().get(structure_id)
        if structure is None:
            return no_update, "That structure is no longer open.", no_update, no_update
        new = {"type": "structure", "key": structure_id, "label": structure.name}
        taken = {item["label"] for item in items}
        if new["label"] in taken:  # two structures may share a name; keep matrix labels unique
            new["label"] = f"{structure.name} ({structure_id[:4]})"
        if any(item["type"] == new["type"] and item["key"] == new["key"] for item in items):
            return no_update, "", no_update, no_update
        return [*items, new], "", no_update, no_update

    if mode == "excel":
        if not excel_file or not excel_sheet:
            return no_update, "Choose a file and sheet first.", no_update, no_update
        columns = excel_columns or []
        if not columns:
            return no_update, "Select at least one column first.", no_update, no_update
        existing_keys = {item["key"] for item in items if item["type"] == "excel"}
        existing_labels = {item["label"] for item in items}
        added = 0
        for column in columns:
            key = excel_item_key(excel_file, excel_sheet, column)
            if key in existing_keys:
                continue
            label = f"{column} ({excel_file})"
            if label in existing_labels:
                label = f"{column} ({excel_file}/{excel_sheet})"
            items.append({"type": "excel", "key": key, "label": label})
            existing_keys.add(key)
            existing_labels.add(label)
            added += 1
        message = f"Added {added} column(s)." if added else "Already in the watchlist."
        return items, message, no_update, []

    symbol = normalize_instrument_symbol(instrument)
    if not symbol:
        return no_update, "Enter an exchange symbol first.", no_update, no_update
    new = {"type": "instrument", "key": symbol, "label": symbol}
    if any(item["type"] == new["type"] and item["key"] == new["key"] for item in items):
        return no_update, "", "", no_update
    return [*items, new], "", "", no_update


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
    """Watchlist rows (each with a warning if its data is missing), plus the missing-symbol
    set for the "Backfill Missing Data" button/store."""
    items = watchlist or []
    loader = container.data_loader
    structures = _open_structures() if any(i["type"] == "structure" for i in items) else {}
    if loader is None:
        return render_watchlist(items, {}), _HIDDEN, []

    excel_store = container.excel_store
    warnings = {
        f"{item['type']}:{item['key']}": watchlist_item_warning(item, structures, loader, excel_store)
        for item in items
    }
    missing = sorted({sym for item in items for sym in watchlist_item_missing_symbols(item, structures, loader)})
    return render_watchlist(items, warnings), (_SHOWN if missing else _HIDDEN), missing


def populate_base_target_options(watchlist):
    """Base/Target dropdown options for Time Series and Year Overlay: the watchlist's own labels."""
    labels = [item["label"] for item in (watchlist or [])]
    options = [{"label": label, "value": label} for label in labels]
    return options, options, options, options


def backfill_missing(n_clicks, missing_symbols):
    """Kick off an async backfill for every watchlist symbol currently flagged as missing.

    Never blocks the UI: runs on a background thread (the same helper the Structure
    Builder uses), so the trader can click Compute again once it finishes.
    """
    if not n_clicks:
        raise PreventUpdate
    if not missing_symbols:
        return "Nothing to backfill."
    backfill_symbols_async(container.historical_adapter, missing_symbols)
    return f"🔄 Backfilling {', '.join(missing_symbols)} in the background — try Compute again in a minute."


# ----------------------------------------------------------------------
# Excel fallback source: file upload / selection
# ----------------------------------------------------------------------


def _excel_file_options() -> list[dict]:
    store = container.excel_store
    return [{"label": name, "value": name} for name in store.list_files()] if store else []


def populate_excel_file_options(pathname):
    """Previously-uploaded files, refreshed whenever the Correlation tab opens."""
    if pathname != "/correlation":
        raise PreventUpdate
    return _excel_file_options()


def upload_excel_file(contents, filename):
    """Save an uploaded workbook to disk (persists across restarts) and select it."""
    if not contents or not filename:
        raise PreventUpdate
    store = container.excel_store
    if store is None:
        return no_update, "Excel storage is not available."
    try:
        _header, encoded = contents.split(",", 1)
        data = base64.b64decode(encoded)
        saved_name = store.save_file(filename, data)
    except (ValueError, CrudeOilRiskError) as exc:
        return no_update, f"Could not save {filename}: {exc}"
    return saved_name, f"Uploaded {saved_name}."


def delete_excel_file(n_clicks, selected_file):
    """Remove an uploaded file from disk and clear the dependent dropdowns."""
    if not n_clicks or not selected_file:
        raise PreventUpdate
    store = container.excel_store
    if store is None:
        raise PreventUpdate
    store.delete_file(selected_file)
    return None, f"Deleted {selected_file}."


def populate_excel_sheet_options(selected_file):
    """Sheet dropdown options for the chosen uploaded file."""
    if not selected_file:
        return [], None
    store = container.excel_store
    if store is None:
        raise PreventUpdate
    try:
        sheets = store.sheet_names(selected_file)
    except CrudeOilRiskError:
        sheets = []
    return [{"label": s, "value": s} for s in sheets], (sheets[0] if len(sheets) == 1 else None)


def populate_excel_column_options(selected_file, selected_sheet):
    """Column multi-select options for the chosen file/sheet."""
    if not selected_file or not selected_sheet:
        return [], []
    store = container.excel_store
    if store is None:
        raise PreventUpdate
    try:
        columns = store.column_names(selected_file, selected_sheet)
    except CrudeOilRiskError:
        columns = []
    return [{"label": c, "value": c} for c in columns], []


# ----------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------


def _status(text: str, is_error: bool = False) -> html.Div:
    return html.Div(text, style={"color": COLORS["ACCENT_RED"] if is_error else COLORS["TEXT_SECONDARY"]})


def compute_heatmap(n_clicks, watchlist, lookback, as_of_str):
    """Build the watchlist series, correlate them and render the heatmap and status line."""
    if not n_clicks:
        raise PreventUpdate
    lookback_days = int(lookback)
    as_of = date.fromisoformat(as_of_str) if as_of_str else None
    items = watchlist or []
    structures = _open_structures() if any(i["type"] == "structure" for i in items) else {}
    result = compute_watchlist_correlation(
        items, structures, lookback_days, container.data_loader, as_of, container.excel_store
    )

    skipped = f" Skipped: {', '.join(result.skipped)}." if result.skipped else ""
    if result.error:
        return empty_figure(result.error, is_error=True), _status(f"⚠️ {result.error}.{skipped}", is_error=True)

    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    as_of_desc = f" (as of {as_of})" if as_of else ""
    text = f"Last computed {now} UTC{as_of_desc} · {result.observations} days."
    if result.observations < lookback_days:
        text += f" Only {result.observations} common days available (requested {lookback_days}d)."
    return build_heatmap_figure(result.matrix, lookback_days), _status(text + skipped)


# ----------------------------------------------------------------------
# Time Series / Year Overlay / Summary Table
# ----------------------------------------------------------------------


def _items_for_labels(watchlist, labels: set[str]) -> list[dict]:
    return [item for item in (watchlist or []) if item["label"] in labels]


def _series_for_pair(watchlist, base: str, target: str) -> tuple[dict, dict]:
    """{label: series} for just the base/target watchlist items, plus {label: skip reason}."""
    items = _items_for_labels(watchlist, {base, target})
    structures = _open_structures() if any(i["type"] == "structure" for i in items) else {}
    return build_watchlist_series(items, structures, container.data_loader, container.excel_store)


def _parse_windows(checked: list[int] | None, custom_text: str | None, defaults: list[int]) -> list[int]:
    windows = set(checked or [])
    for token in (custom_text or "").replace(",", " ").split():
        try:
            window = int(token)
        except ValueError:
            continue
        if window > 1:
            windows.add(window)
    return sorted(windows) if windows else sorted(defaults)


def _no_pair_selected():
    return empty_figure("Pick a Base and a Target column first.", is_error=True), _status(
        "Pick a Base and a Target column first.", is_error=True
    )


def compute_time_series(n_clicks, watchlist, base, target, windows_checked, custom_windows, options):
    """Trailing correlation over full history for one pair, any window(s) overlaid."""
    if not n_clicks:
        raise PreventUpdate
    if not base or not target:
        return _no_pair_selected()

    windows = _parse_windows(windows_checked, custom_windows, DEFAULT_TS_WINDOWS)
    series, skipped = _series_for_pair(watchlist, base, target)
    if base not in series or target not in series:
        reason = "; ".join(f"{label}: {msg}" for label, msg in skipped.items()) or "no local data"
        return empty_figure(reason, is_error=True), _status(f"⚠️ {reason}", is_error=True)

    series_by_window, errors = {}, []
    for window in windows:
        try:
            series_by_window[window] = rolling_correlation_from_series(series[base], series[target], window)
        except (CrudeOilRiskError, ValueError) as exc:
            errors.append(f"{window}d: {exc}")
    if not series_by_window:
        return empty_figure("No rolling correlation available for these windows.", is_error=True), _status(
            "⚠️ " + "; ".join(errors), is_error=True
        )

    options = options or []
    figure = build_time_series_figure(
        series_by_window, base, target, highlight_flips="flip" in options, show_avg="avg" in options
    )
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    text = f"Last computed {now} UTC · windows: {', '.join(f'{w}d' for w in series_by_window)}."
    if errors:
        text += " Skipped: " + "; ".join(errors)
    return figure, _status(text)


def compute_year_overlay(n_clicks, watchlist, base, target, window, options):
    """The same pair's rolling correlation compared year over year by day-of-year."""
    if not n_clicks:
        raise PreventUpdate
    if not base or not target:
        return _no_pair_selected()

    window = int(window or default_rolling_window())
    series, skipped = _series_for_pair(watchlist, base, target)
    if base not in series or target not in series:
        reason = "; ".join(f"{label}: {msg}" for label, msg in skipped.items()) or "no local data"
        return empty_figure(reason, is_error=True), _status(f"⚠️ {reason}", is_error=True)

    try:
        rolling = rolling_correlation_from_series(series[base], series[target], window)
    except (CrudeOilRiskError, ValueError) as exc:
        return empty_figure(str(exc), is_error=True), _status(f"⚠️ {exc}", is_error=True)

    frame = year_overlay_frame(rolling)
    figure = build_year_overlay_figure(frame, base, target, window, show_avg="avg" in (options or []))
    years = sorted(frame["year"].unique().tolist())
    return figure, _status(f"{len(years)} year(s) of data: {', '.join(str(y) for y in years)}.")


def compute_summary(n_clicks, watchlist, window):
    """Mean/std/min/max/last rolling correlation across every valid watchlist pair."""
    if not n_clicks:
        raise PreventUpdate
    items = watchlist or []
    if len(items) < 2:
        return html.Div(), _status(MIN_ITEMS_TEXT, is_error=True)

    window = int(window or default_rolling_window())
    structures = _open_structures() if any(i["type"] == "structure" for i in items) else {}
    series, skipped = build_watchlist_series(items, structures, container.data_loader, container.excel_store)
    if len(series) < 2:
        return html.Div(), _status("Need at least 2 valid series", is_error=True)

    rows = watchlist_pair_summary(series, window)
    if not rows:
        return html.Div(), _status("No pair had enough shared history for this window.", is_error=True)

    skip_text = f" Skipped: {', '.join(skipped)}." if skipped else ""
    return build_summary_table(rows), _status(f"{len(rows)} pair(s) at {window}d.{skip_text}")


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------


def register_correlation_callbacks(app) -> None:
    """Attach the Correlation tab callbacks to the Dash app."""
    app.callback(
        Output("corr-instrument-group", "style"),
        Output("corr-structure-group", "style"),
        Output("corr-excel-group", "style"),
        Input("corr-mode", "value"),
        prevent_initial_call=True,
    )(toggle_mode)

    app.callback(
        Output("corr-structure-select", "options"),
        Input("url", "pathname"),
    )(populate_structure_options)

    app.callback(
        Output("corr-excel-file-select", "options"),
        Input("url", "pathname"),
    )(populate_excel_file_options)

    app.callback(
        Output("corr-excel-file-select", "value", allow_duplicate=True),
        Output("corr-excel-upload-status", "children"),
        Input("corr-excel-upload", "contents"),
        State("corr-excel-upload", "filename"),
        prevent_initial_call=True,
    )(upload_excel_file)

    app.callback(
        Output("corr-excel-file-select", "value", allow_duplicate=True),
        Output("corr-excel-upload-status", "children", allow_duplicate=True),
        Input("corr-excel-delete-btn", "n_clicks"),
        State("corr-excel-file-select", "value"),
        prevent_initial_call=True,
    )(delete_excel_file)

    app.callback(
        Output("corr-excel-sheet-select", "options"),
        Output("corr-excel-sheet-select", "value"),
        Input("corr-excel-file-select", "value"),
    )(populate_excel_sheet_options)

    app.callback(
        Output("corr-excel-columns", "options"),
        Output("corr-excel-columns", "value", allow_duplicate=True),
        Input("corr-excel-file-select", "value"),
        Input("corr-excel-sheet-select", "value"),
        prevent_initial_call=True,
    )(populate_excel_column_options)

    app.callback(
        Output("corr-watchlist", "data"),
        Output("corr-add-message", "children"),
        Output("corr-instrument-input", "value"),
        Output("corr-excel-columns", "value", allow_duplicate=True),
        Input("corr-add-btn", "n_clicks"),
        State("corr-mode", "value"),
        State("corr-instrument-input", "value"),
        State("corr-structure-select", "value"),
        State("corr-excel-file-select", "value"),
        State("corr-excel-sheet-select", "value"),
        State("corr-excel-columns", "value"),
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
        Output("corr-backfill-btn", "style"),
        Output("corr-missing-symbols", "data"),
        Input("corr-watchlist", "data"),
    )(render_watchlist_rows)

    app.callback(
        Output("corr-backfill-status", "children"),
        Input("corr-backfill-btn", "n_clicks"),
        State("corr-missing-symbols", "data"),
        prevent_initial_call=True,
    )(backfill_missing)

    app.callback(
        Output("corr-heatmap", "figure"),
        Output("corr-status", "children"),
        Input("corr-compute-btn", "n_clicks"),
        State("corr-watchlist", "data"),
        State("corr-lookback", "value"),
        State("corr-asof-date", "date"),
        prevent_initial_call=True,
    )(compute_heatmap)

    app.callback(
        Output("corr-ts-base", "options"),
        Output("corr-ts-target", "options"),
        Output("corr-yr-base", "options"),
        Output("corr-yr-target", "options"),
        Input("corr-watchlist", "data"),
    )(populate_base_target_options)

    app.callback(
        Output("corr-ts-graph", "figure"),
        Output("corr-ts-status", "children"),
        Input("corr-ts-compute-btn", "n_clicks"),
        State("corr-watchlist", "data"),
        State("corr-ts-base", "value"),
        State("corr-ts-target", "value"),
        State("corr-ts-windows", "value"),
        State("corr-ts-custom-windows", "value"),
        State("corr-ts-options", "value"),
        prevent_initial_call=True,
    )(compute_time_series)

    app.callback(
        Output("corr-yr-graph", "figure"),
        Output("corr-yr-status", "children"),
        Input("corr-yr-compute-btn", "n_clicks"),
        State("corr-watchlist", "data"),
        State("corr-yr-base", "value"),
        State("corr-yr-target", "value"),
        State("corr-yr-window", "value"),
        State("corr-yr-options", "value"),
        prevent_initial_call=True,
    )(compute_year_overlay)

    app.callback(
        Output("corr-sum-table-wrap", "children"),
        Output("corr-sum-status", "children"),
        Input("corr-sum-compute-btn", "n_clicks"),
        State("corr-watchlist", "data"),
        State("corr-sum-window", "value"),
        prevent_initial_call=True,
    )(compute_summary)
