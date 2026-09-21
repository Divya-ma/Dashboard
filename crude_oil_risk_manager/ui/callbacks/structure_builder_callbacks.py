"""New Structure builder callbacks: modal state, step navigation, leg rows, previews and save.

Logic lives in core.structure_utils / core.structure_builder; these functions
gather component values, call it and render the result. Adapters and the
repository are reached through ui.container.

Compared with a naive one-callback-per-widget wiring, a few outputs are owned by
a single callback on purpose (step indicator, step visibility, leg rows), and
the Save click is handled only by `save_structure`, so a failed validation can
keep the modal open.
"""

import logging

from dash import ALL, MATCH, Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from config.settings import settings
from core.models import StructureStatus
from core.structure_builder import (
    EVENT_ADD,
    EVENT_REMOVE,
    EVENT_TEMPLATE,
    StructureBuildError,
    backfill_symbols_async,
    build_shell_structure,
    check_portfolio_correlation,
    correlation_candidates,
    next_leg_rows,
    save_shell_structure,
)
from core.structure_utils import (
    STRUCTURE_TEMPLATES,
    net_outright_equivalent,
    normalize_symbol,
    split_validation_messages,
    validate_structure_legs,
)
from core.user_settings import KEY_CORRELATION_WINDOW
from ui.callbacks.structures_callbacks import update_active_structures
from ui.container import container
from ui.layouts.shell import COLORS
from ui.layouts.structure_builder import (
    CORRELATION_HINT,
    HIDDEN,
    SHOWN,
    STEP_COUNT,
    build_leg_row,
    data_table,
    render_correlation_table,
    render_exposure_table,
    render_messages,
    render_step_indicator,
)

logger = logging.getLogger(__name__)

_ACTIVE_STATUSES = [StructureStatus.SHELL, StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED]
_CARD_PREFIX = "template-card-"
_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}


def _legs(symbols, ratios) -> list[dict]:
    return [{"symbol": normalize_symbol(s), "ratio": r} for s, r in zip(symbols or [], ratios or [])]


# ----------------------------------------------------------------------
# Modal / template / steps
# ----------------------------------------------------------------------


def toggle_builder_modal(open_clicks, cancel_clicks, is_open):
    """Open (with all state reset) from '+ New Structure'; close from Cancel. Save closes via save_structure."""
    trigger = callback_context.triggered_id
    if trigger == "btn-new-structure" and open_clicks:
        return True, 1, None, [], None, None, None, None, None, None, html.Div(CORRELATION_HINT, style=_MUTED)
    if trigger == "btn-builder-cancel" and cancel_clicks:
        return (False,) + (no_update,) * 10
    raise PreventUpdate


def select_template(*clicks):
    trigger = callback_context.triggered_id
    if not isinstance(trigger, str) or not trigger.startswith(_CARD_PREFIX):
        raise PreventUpdate
    template = trigger[len(_CARD_PREFIX):]
    if template not in STRUCTURE_TEMPLATES or not any(clicks):
        raise PreventUpdate
    return template


def highlight_template(template):
    return tuple("template-card selected" if template == key else "template-card" for key in STRUCTURE_TEMPLATES)


def navigate_steps(next_clicks, back_clicks, current_step, template):
    trigger = callback_context.triggered_id
    step = current_step or 1
    if trigger == "btn-builder-next" and next_clicks:
        if step == 1 and template not in STRUCTURE_TEMPLATES:
            raise PreventUpdate
        return min(step + 1, STEP_COUNT)
    if trigger == "btn-builder-back" and back_clicks:
        return max(step - 1, 1)
    raise PreventUpdate


def render_step_visibility(step, template):
    """Show only the current step; Back/Next/Save follow the step (Next needs a template on step 1)."""
    step = step or 1
    styles = [SHOWN if number == step else HIDDEN for number in range(1, STEP_COUNT + 1)]
    back = HIDDEN if step == 1 else SHOWN
    next_style = HIDDEN if step == STEP_COUNT else SHOWN
    next_disabled = step == 1 and template not in STRUCTURE_TEMPLATES
    save = SHOWN if step == STEP_COUNT else HIDDEN
    return (*styles, back, next_style, next_disabled, save)


def update_step_indicator(step, template):
    return render_step_indicator(step or 1, template)


# ----------------------------------------------------------------------
# Leg rows
# ----------------------------------------------------------------------


def _contract_options() -> list[str]:
    return sorted(c.symbol for c in container.repository.get_all_contracts())


def render_leg_rows(template, add_clicks, remove_clicks, symbols, ratios, current_legs):
    """Rebuild the leg rows after a template pick, '+ Add Leg' or a remove click, keeping typed values."""
    trigger = callback_context.triggered_id
    remove_index = None
    if trigger == "store-builder-template":
        event = EVENT_TEMPLATE
    elif trigger == "btn-add-leg" and add_clicks:
        event = EVENT_ADD
    elif isinstance(trigger, dict) and trigger.get("type") == "leg-remove" and (callback_context.triggered[0]["value"]):
        event, remove_index = EVENT_REMOVE, trigger["index"]
    else:
        raise PreventUpdate

    current = _legs(symbols, ratios) if symbols else (current_legs or [])
    rows = next_leg_rows(template, event, current, remove_index)
    custom = template == "custom"
    options = _contract_options() if rows else []
    children = [
        build_leg_row(i, row["symbol"], row["ratio"], options, removable=custom and len(rows) > 1)
        for i, row in enumerate(rows)
    ]
    return children, rows, (SHOWN if custom else HIDDEN)


def fill_symbol_from_saved(selected):
    """Picking a saved contract copies it into the symbol box (and clears the picker)."""
    if not selected:
        raise PreventUpdate
    return selected, None


# ----------------------------------------------------------------------
# Previews
# ----------------------------------------------------------------------


def update_exposure_preview(symbols, ratios, legs_store):
    net, ignored = net_outright_equivalent(_legs(symbols, ratios))
    return render_exposure_table(net, ignored)


def check_correlation(n_clicks, symbols, ratios, template):
    """Correlation of the candidate structure vs active structures; runs only on the button."""
    if not n_clicks:
        raise PreventUpdate
    repository = container.repository
    candidates = correlation_candidates(template, [normalize_symbol(s) for s in symbols or []], ratios or [])
    if not candidates:
        return html.Div("Enter at least one valid leg symbol first.", style=_MUTED), []

    window = int(repository.get_setting(KEY_CORRELATION_WINDOW, settings.DEFAULT_CORRELATION_WINDOW))
    portfolio = repository.get_all_structures(status_filter=_ACTIVE_STATUSES)
    rows = check_portfolio_correlation(candidates, portfolio, window, container.data_loader)
    return render_correlation_table(rows), rows


# ----------------------------------------------------------------------
# Validation / review
# ----------------------------------------------------------------------


def update_validation_messages(step, legs_store, symbols, ratios):
    """Live leg warnings on the Details and Review steps (blocking errors are red)."""
    if (step or 1) < 3:
        return []
    errors, warnings = split_validation_messages(validate_structure_legs(_legs(symbols, ratios)))
    return render_messages(errors, warnings)


def render_review(step, template, symbols, ratios, name, multiplier, tick_size, tick_value, notes, correlation_rows):
    if step != STEP_COUNT:
        raise PreventUpdate
    legs = _legs(symbols, ratios)
    net, ignored = net_outright_equivalent(legs)
    label = STRUCTURE_TEMPLATES.get(template or "", {}).get("label", "—")
    text = {"color": COLORS["TEXT_PRIMARY"]}
    sections = [
        html.H4(name or "(no name)", style=text),
        html.Div(f"{label} · Multiplier {multiplier or '—'} · Tick size {tick_size or '—'} · Tick value {tick_value or '—'}",
                 style={"color": COLORS["TEXT_SECONDARY"], "marginBottom": "12px"}),
        data_table(["Leg", "Symbol", "Ratio"], [[str(i), l["symbol"] or "—", f"{l['ratio']:+g}" if l["ratio"] else "—"]
                                                  for i, l in enumerate(legs, start=1)]),
        html.H6("Net Outright Equivalent (per 1 lot)", style={**text, "marginTop": "16px"}),
        render_exposure_table(net, ignored),
    ]
    flagged = [r for r in correlation_rows or [] if r["classification"] == "highly_correlated"]
    if flagged:
        sections.append(html.H6("Correlation warnings", style={**text, "marginTop": "16px"}))
        sections.append(render_correlation_table(flagged))
    if notes:
        sections.append(html.Div(f"Notes: {notes}", style={"color": COLORS["TEXT_SECONDARY"], "marginTop": "12px"}))
    return html.Div(sections)


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------


def save_structure(
    n_clicks, name, template, symbols, ratios, multiplier, tick_size, tick_value, notes,
    status_filter, product_filter, sort_by, portfolio_pnl, live_prices,
):
    """Build and save a SHELL structure; on errors keep the modal open and show them."""
    if not n_clicks:
        raise PreventUpdate
    repository = container.repository
    try:
        built = build_shell_structure(
            name, template, symbols, ratios, multiplier, tick_size, tick_value, notes, repository.get_contract
        )
        save_shell_structure(repository, built)
    except StructureBuildError as exc:
        return no_update, no_update, render_messages(exc.errors, exc.warnings), no_update, no_update
    except Exception:  # noqa: BLE001 - surface a readable message instead of a silent callback error
        logger.exception("Saving structure failed")
        return no_update, no_update, render_messages(["Could not save the structure; see the server log."], []), no_update, no_update

    backfill_symbols_async(container.historical_adapter, [c.symbol for c in built.new_contracts])
    rows = update_active_structures(portfolio_pnl, status_filter, product_filter, sort_by, live_prices)
    message = f"'{built.structure.name}' saved as a shell structure."
    if built.new_contracts:
        message += f" Backfilling history for {', '.join(c.symbol for c in built.new_contracts)}."
    return False, rows, [], True, message


# ----------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------

_LEG_SYMBOLS = {"type": "leg-symbol", "index": ALL}
_LEG_RATIOS = {"type": "leg-ratio", "index": ALL}


def register_structure_builder_callbacks(app) -> None:
    """Attach the New Structure builder callbacks to the Dash app."""
    # Registered first: these own the outputs the later callbacks write with allow_duplicate.
    app.callback(
        Output("modal-new-structure", "is_open"),
        Output("store-builder-step", "data"),
        Output("store-builder-template", "data"),
        Output("store-builder-legs", "data"),
        Output("builder-name", "value"),
        Output("builder-multiplier", "value"),
        Output("builder-tick-size", "value"),
        Output("builder-tick-value", "value"),
        Output("builder-notes", "value"),
        Output("store-builder-correlation", "data"),
        Output("builder-correlation-panel", "children"),
        Input("btn-new-structure", "n_clicks"),
        Input("btn-builder-cancel", "n_clicks"),
        State("modal-new-structure", "is_open"),
        prevent_initial_call=True,
    )(toggle_builder_modal)

    app.callback(
        Output("store-builder-template", "data", allow_duplicate=True),
        *(Input(f"{_CARD_PREFIX}{key}", "n_clicks") for key in STRUCTURE_TEMPLATES),
        prevent_initial_call=True,
    )(select_template)

    app.callback(
        *(Output(f"{_CARD_PREFIX}{key}", "className") for key in STRUCTURE_TEMPLATES),
        Input("store-builder-template", "data"),
    )(highlight_template)

    app.callback(
        Output("store-builder-step", "data", allow_duplicate=True),
        Input("btn-builder-next", "n_clicks"),
        Input("btn-builder-back", "n_clicks"),
        State("store-builder-step", "data"),
        State("store-builder-template", "data"),
        prevent_initial_call=True,
    )(navigate_steps)

    app.callback(
        Output("builder-step-1", "style"),
        Output("builder-step-2", "style"),
        Output("builder-step-3", "style"),
        Output("builder-step-4", "style"),
        Output("btn-builder-back", "style"),
        Output("btn-builder-next", "style"),
        Output("btn-builder-next", "disabled"),
        Output("btn-save-structure", "style"),
        Input("store-builder-step", "data"),
        Input("store-builder-template", "data"),
    )(render_step_visibility)

    app.callback(
        Output("builder-step-indicator", "children"),
        Input("store-builder-step", "data"),
        Input("store-builder-template", "data"),
    )(update_step_indicator)

    app.callback(
        Output("builder-leg-rows", "children"),
        Output("store-builder-legs", "data", allow_duplicate=True),
        Output("btn-add-leg", "style"),
        Input("store-builder-template", "data"),
        Input("btn-add-leg", "n_clicks"),
        Input({"type": "leg-remove", "index": ALL}, "n_clicks"),
        State(_LEG_SYMBOLS, "value"),
        State(_LEG_RATIOS, "value"),
        State("store-builder-legs", "data"),
        prevent_initial_call=True,
    )(render_leg_rows)

    app.callback(
        Output({"type": "leg-symbol", "index": MATCH}, "value"),
        Output({"type": "leg-autocomplete", "index": MATCH}, "value"),
        Input({"type": "leg-autocomplete", "index": MATCH}, "value"),
        prevent_initial_call=True,
    )(fill_symbol_from_saved)

    app.callback(
        Output("builder-exposure-preview", "children"),
        Input(_LEG_SYMBOLS, "value"),
        Input(_LEG_RATIOS, "value"),
        Input("store-builder-legs", "data"),
        prevent_initial_call=True,
    )(update_exposure_preview)

    app.callback(
        Output("builder-correlation-panel", "children", allow_duplicate=True),
        Output("store-builder-correlation", "data", allow_duplicate=True),
        Input("btn-refresh-correlation", "n_clicks"),
        State(_LEG_SYMBOLS, "value"),
        State(_LEG_RATIOS, "value"),
        State("store-builder-template", "data"),
        prevent_initial_call=True,
    )(check_correlation)

    app.callback(
        Output("builder-validation-messages", "children"),
        Input("store-builder-step", "data"),
        Input("store-builder-legs", "data"),
        State(_LEG_SYMBOLS, "value"),
        State(_LEG_RATIOS, "value"),
        prevent_initial_call=True,
    )(update_validation_messages)

    app.callback(
        Output("builder-review-body", "children"),
        Input("store-builder-step", "data"),
        State("store-builder-template", "data"),
        State(_LEG_SYMBOLS, "value"),
        State(_LEG_RATIOS, "value"),
        State("builder-name", "value"),
        State("builder-multiplier", "value"),
        State("builder-tick-size", "value"),
        State("builder-tick-value", "value"),
        State("builder-notes", "value"),
        State("store-builder-correlation", "data"),
        prevent_initial_call=True,
    )(render_review)

    app.callback(
        Output("modal-new-structure", "is_open", allow_duplicate=True),
        Output("structures-active-grid", "rowData", allow_duplicate=True),
        Output("builder-validation-messages", "children", allow_duplicate=True),
        Output("builder-save-toast", "is_open"),
        Output("builder-save-toast", "children"),
        Input("btn-save-structure", "n_clicks"),
        State("builder-name", "value"),
        State("store-builder-template", "data"),
        State(_LEG_SYMBOLS, "value"),
        State(_LEG_RATIOS, "value"),
        State("builder-multiplier", "value"),
        State("builder-tick-size", "value"),
        State("builder-tick-value", "value"),
        State("builder-notes", "value"),
        State("filter-structure-status", "value"),
        State("filter-structure-product", "value"),
        State("filter-structure-sort", "value"),
        State("store-portfolio-pnl", "data"),
        State("store-live-prices", "data"),
        prevent_initial_call=True,
    )(save_structure)
