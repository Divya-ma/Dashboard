"""Trade Analyzer callbacks: validate the form, run core.trade_analysis on Analyze, fill the result panels.

Nothing runs until Analyze is pressed (no polling, no API call: only the stored daily settlements are read)."""

import logging
import math
from datetime import datetime, timezone

from dash import ALL, Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate

from core.positions import filter_open_structures
from core.structure_builder import EVENT_ADD, EVENT_REMOVE, EVENT_TEMPLATE, next_leg_rows
from core.structure_utils import STRUCTURE_TEMPLATES, decompose_to_outrights, normalize_symbol
from core.structure_view import statuses_for_filter
from core.trade_analysis import (
    MAX_HORIZON, TradeAnalysisError, TradeSpec, analyse, outright_weights, portfolio_effect, save_risk_per_trade,
)
from ui.container import container
from ui.layouts.curve_tab import _table, empty_figure
from ui.layouts.shell import COLORS
from ui.layouts.trade_analyzer_tab import (
    CONTEXT_COLUMNS, DEFAULT_TEMPLATE, GRID_COLUMNS, PLACEHOLDER, YEAR_COLUMNS, build_fan_figure, context_rows, grid_rows, highlight, leg_rows,
    portfolio_panel, summary_cards, verdict, year_rows,
)

logger = logging.getLogger(__name__)

_ERROR = {"color": COLORS["ACCENT_RED"], "fontSize": "13px"}
_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}


def _today():
    return datetime.now(timezone.utc).date()


def _number(value, name: str, required: bool = True, minimum: float = 0.0, whole: bool = False):
    if value in (None, ""):
        if required:
            raise ValueError(f"{name} is required.")
        return None
    try:
        number = float(str(value).replace(",", ""))
    except ValueError:
        raise ValueError(f"{name} must be a number.") from None
    if not math.isfinite(number) or number <= minimum:
        raise ValueError(f"{name} must be above {minimum:g}.")
    if whole and number != int(number):
        raise ValueError(f"{name} must be a whole number.")
    return int(number) if whole else number


def _legs(symbols, ratios, template) -> tuple:
    """((symbol, ratio), ...) from the leg rows, checked like the structure builder checks a new structure."""
    rows = list(zip(symbols or [], ratios or []))
    spec = STRUCTURE_TEMPLATES.get(template or "")
    if spec is None:
        raise ValueError("Pick a structure type.")
    if spec["legs"] is not None and len(rows) != spec["legs"]:
        raise ValueError(f"A {spec['label'].lower()} has {spec['legs']} leg(s).")
    legs = []
    for number, (symbol, ratio) in enumerate(rows, start=1):
        symbol = normalize_symbol(symbol)
        if not symbol:
            raise ValueError(f"Type the symbol of leg {number}.")
        try:
            decompose_to_outrights(symbol, 1.0)
        except ValueError as exc:
            raise ValueError(f"Leg {number}: {exc}") from None
        try:
            number_ratio = float(ratio)
        except (TypeError, ValueError):
            raise ValueError(f"Leg {number} needs a ratio, a whole number such as 1 or -2.") from None
        if number_ratio == 0 or number_ratio != int(number_ratio):
            raise ValueError(f"Leg {number} needs a whole, non-zero ratio.")
        legs.append((symbol, int(number_ratio)))
    if not legs:
        raise ValueError("Enter at least one leg.")
    return tuple(legs)


def build_spec(template, symbols, ratios, side, lots, entry, risk_trade, horizon, window, stop, target) -> TradeSpec:
    """The TradeSpec for the form values; raises ValueError with a message safe to show."""
    legs = _legs(symbols, ratios, template)
    if side not in (1, -1):
        raise ValueError("Choose buy or sell.")
    window_value = 7 if window in (None, "") else int(_number(window, "Comparable days", minimum=-1, whole=True))
    stop_value, target_value = _number(stop, "Your stop", False), _number(target, "Your target", False)
    if (stop_value is None) != (target_value is None):
        raise ValueError("Give both your stop and your target, or neither.")
    horizon_value = _number(horizon, "Days to give the trade", whole=True)
    if horizon_value > MAX_HORIZON:
        raise ValueError(f"Days to give the trade is at most {MAX_HORIZON}.")
    return TradeSpec(
        legs=legs, side=int(side), lots=_number(lots, "Lots"), risk_trade=_number(risk_trade, "Risk per trade"),
        horizon=horizon_value, window=window_value, entry=_number(entry, "Entry price", False, minimum=-1e12),
        stop=stop_value, target=target_value,
    )


def update_leg_rows(template, add_clicks, remove_clicks, symbols, ratios):
    """The leg rows after a template pick, '+ Add leg' or a remove click; what was typed in the other legs is kept."""
    template = template if template in STRUCTURE_TEMPLATES else DEFAULT_TEMPLATE
    trigger = callback_context.triggered_id
    current = [{"symbol": sym, "ratio": ratio} for sym, ratio in zip(symbols or [], ratios or [])]
    if trigger is None or trigger == "ta-template":
        event, index = EVENT_TEMPLATE, None
    elif trigger == "ta-add-leg":
        if not add_clicks:
            raise PreventUpdate
        event, index = EVENT_ADD, None
    elif isinstance(trigger, dict) and trigger.get("type") == "ta-remove":
        if not callback_context.triggered[0]["value"]:
            raise PreventUpdate  # the button was just created, not clicked
        event, index = EVENT_REMOVE, trigger["index"]
    else:
        raise PreventUpdate
    rows = next_leg_rows(template, event, current, index)
    return leg_rows(rows, removable=STRUCTURE_TEMPLATES[template]["legs"] is None)


def _empty():
    blank = html.Div()
    return (blank, blank, blank, empty_figure(PLACEHOLDER), blank, portfolio_panel(None), blank)


def run_analysis(n_clicks, template, symbols, ratios, side, lots, entry, risk_trade, horizon, window, stop, target):
    """(status, verdict, cards, grid, fan, context, portfolio, years + notes) for the form."""
    if not n_clicks:
        raise PreventUpdate
    settlements = container.settlements
    if settlements is None:
        return (html.Div("Settlement history is not available.", style=_ERROR), *_empty())
    try:
        spec = build_spec(template, symbols, ratios, side, lots, entry, risk_trade, horizon, window, stop, target)
        analysis = analyse(settlements, spec, _today())
    except ValueError as exc:
        return (html.Div(f"⚠️ {exc}", style=_ERROR), *_empty())
    except TradeAnalysisError as exc:
        return (html.Div(f"⚠️ {exc}", style=_ERROR), *_empty())
    except Exception:  # noqa: BLE001 - a readable message instead of a silent callback error
        logger.exception("Trade analysis failed")
        return (html.Div("Could not analyse this trade; see the server log.", style=_ERROR), *_empty())
    effect = _portfolio(spec)
    status = html.Div(f"{spec.symbol}: {analysis.n_paths} comparable days from {analysis.n_years} earlier years, "
                      f"{spec.horizon} days allowed. Showing the pair with the best expected result that fits your risk.")
    grid = _table("ta-grid-table", GRID_COLUMNS, grid_rows(analysis), style_data_conditional=highlight(analysis), page_size=15)
    years = _table("ta-years-table", YEAR_COLUMNS, year_rows(analysis), page_size=15)
    context = _table("ta-context-table", CONTEXT_COLUMNS, context_rows(analysis), page_size=10)
    notes = html.Div([html.Div(f"⚠️ {n}", style={"color": COLORS["ACCENT_YELLOW"], "marginBottom": "4px"}) for n in analysis.notes])
    return (
        status, html.Div(verdict(analysis)), summary_cards(analysis), grid, build_fan_figure(analysis), context,
        portfolio_panel(effect), html.Div([years, notes]),
    )


def _portfolio(spec: TradeSpec):
    repository = container.repository
    if repository is None or container.data_loader is None:
        return None
    try:
        structures = filter_open_structures(repository.get_all_structures(status_filter=statuses_for_filter("open")))
        return portfolio_effect(spec, structures, container.data_loader, repository.get_contract)
    except Exception:  # noqa: BLE001
        logger.exception("Portfolio effect failed")
        return None


def save_limits(n_clicks, risk_trade):
    if not n_clicks:
        raise PreventUpdate
    try:
        trade = _number(risk_trade, "Risk per trade")
    except ValueError as exc:
        return html.Div(f"⚠️ {exc}", style=_ERROR)
    save_risk_per_trade(container.repository, trade)
    return html.Div(f"Saved: ${trade:,.0f} per trade. It is used next time you open this page.", style={"color": COLORS["ACCENT_GREEN"], "fontSize": "13px"})


def register_trade_analyzer_callbacks(app) -> None:
    app.callback(
        Output("ta-status", "children", allow_duplicate=True),
        Output("ta-verdict", "children"),
        Output("ta-cards", "children"),
        Output("ta-grid", "children"),
        Output("ta-fan", "figure"),
        Output("ta-context", "children"),
        Output("ta-portfolio", "children"),
        Output("ta-years", "children"),
        Input("ta-run", "n_clicks"),
        State("ta-template", "value"),
        State({"type": "ta-symbol", "index": ALL}, "value"),
        State({"type": "ta-ratio", "index": ALL}, "value"),
        State("ta-side", "value"),
        State("ta-lots", "value"),
        State("ta-entry", "value"),
        State("ta-risk-trade", "value"),
        State("ta-horizon", "value"),
        State("ta-window", "value"),
        State("ta-stop", "value"),
        State("ta-target", "value"),
        prevent_initial_call=True,
    )(run_analysis)

    app.callback(
        Output("ta-status", "children"),
        Input("ta-save", "n_clicks"),
        State("ta-risk-trade", "value"),
        prevent_initial_call=True,
    )(save_limits)

    app.callback(
        Output("ta-leg-rows", "children"),
        Input("ta-template", "value"),
        Input("ta-add-leg", "n_clicks"),
        Input({"type": "ta-remove", "index": ALL}, "n_clicks"),
        State({"type": "ta-symbol", "index": ALL}, "value"),
        State({"type": "ta-ratio", "index": ALL}, "value"),
    )(update_leg_rows)
