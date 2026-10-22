"""ATR tab of the OI / Volume / ATR page: how ATR changes as a contract nears expiry, how it compares with the
same month in past years, and the ratio of the later contracts' ATR to the front's (core.atr_engine).

ATR is the standard true range (max of high - low and the gaps from yesterday's close) averaged over the period
set here (default 14 days). Comparisons are between the same delivery month in different years only. Colours
from COLORS.
"""

from datetime import datetime, timezone

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dash_table.Format import Format, Group, Scheme

from core.atr_engine import DTE_MARKS, FRONT_DAYS, MIN_PAIR_DAYS, AtrEngine, AtrParams
from ui.container import container
from ui.layouts.curve_tab import _CARD, _LABEL, _MUTED, _layout, _table, empty_figure
from ui.layouts.shell import COLORS

MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
FAMILY_OPTIONS = [{"label": "Outright", "value": "outright"}, {"label": "Spread", "value": "spread"}, {"label": "Fly", "value": "fly"}]
UNIT_OPTIONS = [{"label": "Points", "value": "atr"}, {"label": "% of price (outrights)", "value": "pct"}]
STATUS_COLORS = {"HIGH": COLORS["ACCENT_RED"], "LOW": "#5dade2"}

# (id, label, AtrParams field, step, hint)
PARAM_INPUTS = [
    ("atr-p-period", "ATR period (days)", "period", 1, "Days in the average of the true range (standard: 14)"),
    ("atr-p-contracts", "Contracts to watch", "contracts", 1, "The next N live contracts"),
    ("atr-p-years", "Same month, last N years", "years", 1, "Compare with the same delivery month in the last N years at the same days before expiry"),
    ("atr-p-z", "Deviation threshold (z)", "z_threshold", 0.1, "How many standard deviations from the usual counts as unusually high or low"),
]

_INT = Format(group=Group.yes, precision=0, scheme=Scheme.fixed)
_TWO = Format(precision=2, scheme=Scheme.fixed)
_THREE = Format(precision=3, scheme=Scheme.fixed)
_ONE = Format(precision=1, scheme=Scheme.fixed)


def _col(name, column_id, fmt=None):
    column = {"name": name, "id": column_id}
    if fmt is not None:
        column.update({"type": "numeric", "format": fmt})
    return column


def _number(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), digits)


def atr_controls(params: AtrParams, default_month: int) -> html.Div:
    inputs = [
        dbc.Col([html.Div(name, style=_LABEL, title=hint), dbc.Input(id=component_id, type="number", value=getattr(params, field), step=step, debounce=True, size="sm")],
                md=2, className="mb-2")
        for component_id, name, field, step, hint in PARAM_INPUTS
    ]
    return html.Div([
        dbc.Row([
            dbc.Col([html.Div("Curve", style=_LABEL), dcc.Dropdown(id="atr-family", options=FAMILY_OPTIONS, value="outright", clearable=False)], md=2, className="mb-2"),
            dbc.Col([html.Div("Show", style=_LABEL), dcc.Dropdown(id="atr-unit", options=UNIT_OPTIONS, value="atr", clearable=False)], md=3, className="mb-2"),
            dbc.Col([html.Div("Delivery month", style=_LABEL, title="For the hedge-ratio table: the days in history when the front contract was this month"),
                     dcc.Dropdown(id="atr-month", options=[{"label": MONTH_NAMES[m], "value": m} for m in range(1, 13)], value=default_month, clearable=False)], md=2, className="mb-2"),
            *inputs,
        ], className="g-2"),
        dbc.Switch(id="atr-p-idle", label="Ignore days with no trading (a thin contract's flat candle would drag its ATR down)", value=params.skip_idle, className="mt-1"),
    ], style={**_CARD, "padding": "12px 14px", "borderRadius": "6px", "marginBottom": "12px"})


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

COLUMN_TIPS = {
    "atr": "Standard ATR: the average true range over the period set above, in price points.",
    "atr_pct": "The ATR as a % of the contract's price (outrights only).",
    "tr": "Today's true range: the larger of the day's high-low and the gaps from yesterday's close.",
    "tr_vs_atr": "Today's true range divided by the ATR: above 1 means a bigger-than-average day.",
    "vs_front": "This contract's ATR divided by the front contract's ATR now.",
    "usual": "The ATR of the same delivery month's finished contracts, in past years, at the same days before expiry (median).",
    "z": "Distance of today's ATR from that usual, in standard deviations.",
}


def live_columns(family: str, params: AtrParams) -> list[dict]:
    columns = [_col("Contract", "label"), _col("Days to expiry", "dte", _INT), _col("Last", "close", _TWO), _col("ATR", "atr", _THREE)]
    if family == "outright":
        columns.append(_col("ATR % of price", "atr_pct", _TWO))
    columns += [
        _col("Today's range", "tr", _THREE), _col("Today ÷ ATR", "tr_vs_atr", _TWO), _col("ATR ÷ front ATR", "vs_front", _TWO),
        _col(f"Usual (same month, {params.years}y)", "usual", _THREE), _col("Past range", "range"), _col("z", "z", _ONE), _col("Status", "status"),
    ]
    return columns


def live_rows(table, family: str) -> list[dict]:
    rows = []
    for r in table.itertuples():
        low, high = r.usual_min, r.usual_max
        rows.append({
            "label": r.label, "dte": r.dte, "close": _number(r.close, 2), "atr": _number(r.atr), "atr_pct": _number(r.atr_pct, 2),
            "tr": _number(r.tr), "tr_vs_atr": _number(r.tr_vs_atr, 2), "vs_front": _number(r.vs_front, 2), "usual": _number(r.usual_median),
            "range": f"{low:.3f} to {high:.3f}" if np.isfinite(low) and np.isfinite(high) and r.usual_n else "", "z": _number(r.z, 1), "status": r.status,
        })
    return rows


def live_styles(params: AtrParams) -> list[dict]:
    z = params.z_threshold
    styles = [{"if": {"column_id": "z", "filter_query": f"{{z}} >= {z} || {{z}} <= {-z}"}, "color": COLORS["ACCENT_YELLOW"], "fontWeight": "bold"}]
    for status, color in STATUS_COLORS.items():
        styles.append({"if": {"column_id": "status", "filter_query": f'{{status}} = "{status}"'}, "color": color, "fontWeight": "bold"})
    return styles


def month_columns() -> list[dict]:
    return [{"name": "Delivery month", "id": "month"}, *({"name": f"{label} out ({days}d)", "id": f"m{label}"} for days, label in DTE_MARKS)]


def month_rows(table, relative: bool = False, digits: int = 2) -> list[dict]:
    rows = []
    for _, r in table.iterrows():
        row = {"month": MONTH_NAMES[int(r["month"])]}
        for _, label in DTE_MARKS:
            median, sd = r[f"median_{label}"], r[f"sd_{label}"]
            if not np.isfinite(median):
                row[f"m{label}"] = ""
            elif relative or not np.isfinite(sd):
                row[f"m{label}"] = f"{median:.2f}x" if relative else f"{median:.{digits}f}"
            else:
                row[f"m{label}"] = f"{median:.{digits}f} ({sd:.{digits}f})"
        rows.append(row)
    return rows


HEDGE_COLUMNS = [
    _col("Contract (now)", "label"), _col("ATR ÷ front ATR: now", "now", _TWO), _col("median", "median", _TWO), _col("middle half", "band"),
    _col("lowest to highest", "range"), _col("Hedge ratio (units of front per unit)", "hedge", _TWO), _col("Correlation of daily changes", "corr", _TWO),
]


def hedge_rows(table) -> list[dict]:
    rows = []
    for r in table.itertuples():
        rows.append({
            "label": r.label, "now": _number(r.now, 2), "median": _number(r.median, 2),
            "band": f"{r.p25:.2f} to {r.p75:.2f}" if np.isfinite(r.p25) else "", "range": f"{r.min:.2f} to {r.max:.2f}" if np.isfinite(r.min) else "",
            "hedge": _number(r.hedge, 2), "corr": _number(r.corr, 2),
        })
    return rows


def _section(title: str, hint: str, body) -> html.Div:
    return html.Div([
        html.H6(title, style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "24px", "marginBottom": "2px"}),
        html.Div(hint, style={**_MUTED, "marginBottom": "8px"}),
        body,
    ])


def atr_body(engine: AtrEngine | None, family: str, unit: str, month: int, params: AtrParams) -> html.Div:
    """The live table and the month and hedge tables for one product, curve and unit."""
    if engine is None or engine._empty:
        return html.Div("No ATR data downloaded yet. It downloads automatically, or press the button at the top.", style=_MUTED)
    live = engine.live_table(family)
    if live.empty:
        return html.Div(f"No live {family} has enough history for a {params.period}-day ATR yet.", style=_MUTED)
    table = _table("atr-live-table", live_columns(family, params), live_rows(live, family), live_styles(params), page_size=20)
    table.tooltip_header = {key: {"value": text, "type": "text"} for key, text in COLUMN_TIPS.items()}
    table.tooltip_delay = 0
    table.tooltip_duration = None
    in_pct = unit == "pct" and family == "outright"
    by_month = engine.month_table(family, "pct" if in_pct else "atr")
    relative = engine.month_table(family, "atr", relative=True)
    pieces = [
        html.Div(
            f"True range averaged over {params.period} days (a day with no trading is ignored: {'yes' if params.skip_idle else 'no'}). Data as of {live['date'].iloc[0]}. "
            f"Each contract's ATR is compared with the same delivery month's finished contracts in the last {params.years} years at the same days before expiry "
            "(never other months). Hover over a column heading for what it means.",
            style={**_MUTED, "margin": "6px 0 10px"},
        ),
        table,
        _section(
            f"ATR by delivery month, N months before expiry ({'% of price' if in_pct else 'points'})",
            "Each row is one delivery month over its finished contracts: the median (SD) of the ATR a given number of months before expiry. "
            "Read across a row to see how a contract's ATR changes as it comes in towards expiry.",
            _table("atr-month-table", month_columns(), month_rows(by_month, digits=2 if in_pct else 3), page_size=12),
        ),
        _section(
            f"ATR as a multiple of the contract's own ATR when it became the front ({FRONT_DAYS} days out)",
            "For example 0.70x at 9 months means the ATR was 70% of what it was when the contract became the front. Median across the years of that month.",
            _table("atr-front-multiple-table", month_columns(), month_rows(relative, relative=True), page_size=12),
        ),
    ]
    if family == "outright":
        hedge = engine.hedge_table(month)
        pieces.append(_section(
            f"Later contracts against the front, for the days when the front was {MONTH_NAMES[month]}",
            "The ATR of the k-th contract divided by the front contract's, on every day in history when the front was a "
            f"{MONTH_NAMES[month]} contract (median, middle half, lowest to highest, and now). The hedge ratio is the minimum-variance one: how many units of the front hedge one unit of "
            "that contract (the beta of its daily price changes on the front's); the ATR ratio ignores how closely they move together, the correlation shows it.",
            _table("atr-hedge-table", HEDGE_COLUMNS, hedge_rows(hedge), page_size=20),
        ))
    else:
        pieces.append(html.Div("The hedge-ratio table covers outrights; pick Outright to see it.", style={**_MUTED, "marginTop": "16px"}))
    return html.Div(pieces)


# ----------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------


def build_atr_chart(data: dict | None) -> go.Figure:
    """A contract's ATR by days before expiry against the same month's past years (median and the band that counts as unusual)."""
    if data is None:
        return empty_figure("Pick a contract")
    figure = go.Figure()
    usual = data["usual"]
    horizon = float(np.max(data["dte"])) * 1.04 + 5
    keep = usual["dte"] <= horizon
    x, low, high, median = usual["dte"][keep], usual["low"][keep], usual["high"][keep], usual["median"][keep]
    color = "rgba(77,163,255,{})"
    figure.add_trace(go.Scatter(x=x, y=high, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=x, y=low, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=color.format(0.13), name=f"band (usual ± {data['threshold']:g} z)", hoverinfo="skip"))
    figure.add_trace(go.Scatter(
        x=x, y=median, mode="lines", name=f"Same month ({', '.join(usual['labels'])}): usual", line={"color": color.format(0.95), "width": 2},
        hovertemplate="%{x} days before expiry: usual %{y:.3f}<extra></extra>",
    ))
    for label, y in usual["lines"].items():
        figure.add_trace(go.Scatter(
            x=usual["dte"][keep], y=np.asarray(y)[keep], mode="lines", name=label, line={"color": "rgba(150,160,170,0.55)", "width": 1},
            hovertemplate=label + ": %{y:.3f}<extra></extra>", visible="legendonly",
        ))
    figure.add_trace(go.Scatter(
        x=data["dte"], y=data["atr"], mode="lines", name=f"{data['label']} (live)", line={"color": COLORS["ACCENT_GREEN"], "width": 3},
        customdata=[str(d.date()) for d in data["dates"]], hovertemplate="%{customdata}: %{y:.3f} (%{x} days before expiry)<extra></extra>",
    ))
    figure.update_layout(**_layout(
        title={"text": f"{data['label']}: ATR ({data['unit']}) by days before expiry against its usual path", "font": {"size": 14}},
        xaxis={"title": "Calendar days before expiry (expiry on the right)", "range": [horizon, 0], "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis={"title": f"ATR ({data['unit']})", "gridcolor": COLORS["BORDER_COLOR"], "rangemode": "tozero"}, hovermode="x unified", uirevision=data["symbol"],
    ))
    return figure


def _stat(title: str, value: str, note: str = "", color: str | None = None) -> dbc.Col:
    return dbc.Col(html.Div([
        html.Div(title, style=_LABEL),
        html.Div(value, style={"color": color or COLORS["TEXT_PRIMARY"], "fontSize": "22px", "fontWeight": "bold"}),
        html.Div(note, style={**_MUTED, "marginTop": "2px"}),
    ], style={**_CARD, "padding": "12px 14px", "borderRadius": "6px", "height": "100%"}), md=3, className="mb-2")


def _side(beta: float) -> str:
    return "the opposite side" if beta > 0 else "the same side"


def hedge_calculator_result(result, lots: float, window: int) -> html.Div:
    """The answer for one pair: how many lots of B hedge `lots` of A (result is core.atr_engine.PairHedge or None)."""
    if result is None:
        return html.Div("These two contracts have no overlapping trading days with enough history yet.", style=_MUTED)
    a, b = result.a, result.b
    pieces = []
    if result.window_days < MIN_PAIR_DAYS:
        pieces.append(dbc.Alert(f"Only {result.window_days} shared trading days in the window: treat the ratios as unreliable.", color="warning", className="py-2"))
    cards = []
    if np.isfinite(result.beta_window):
        need = abs(result.beta_window) * lots
        sentence = (
            f"To hedge {lots:g} lot(s) of {a}, trade about {need:.2f} lot(s) of {b} on {_side(result.beta_window)} "
            f"({abs(result.beta_window):.2f} lot of {b} per lot of {a}), the minimum-variance hedge over the last {result.window_days} shared days."
        )
        removed = (1 - result.residual_window / result.unhedged_window) * 100 if result.unhedged_window and np.isfinite(result.residual_window) else np.nan
        corr_ok = np.isfinite(result.corr_window)
        cards = [
            _stat("Hedge ratio (recommended)", f"{abs(result.beta_window):.2f}", f"lots of {b} per lot of {a}, on {_side(result.beta_window)}", COLORS["ACCENT_GREEN"]),
            _stat("Correlation of daily changes", f"{result.corr_window:.2f}" if corr_ok else "n/a", f"the hedge removes about {result.corr_window ** 2 * 100:.0f}% of the variance" if corr_ok else ""),
            _stat("Daily risk: unhedged, then hedged", f"{result.unhedged_window:.3f} → {result.residual_window:.3f}", f"points of {a}, 1 standard deviation" + (f" (down {removed:.0f}%)" if np.isfinite(removed) else "")),
            _stat(f"Same over all {result.days} shared days", f"{abs(result.beta_all):.2f}" if np.isfinite(result.beta_all) else "n/a", f"correlation {result.corr_all:.2f}" if np.isfinite(result.corr_all) else ""),
        ]
    else:
        sentence = f"{b} did not move on the shared days, so there is no hedge ratio."
    pieces.append(html.Div(sentence, style={"color": COLORS["TEXT_PRIMARY"], "fontSize": "15px", "margin": "4px 0 10px"}))
    if cards:
        pieces.append(dbc.Row(cards, className="g-2"))
    ratio = result.atr_ratio
    pieces.append(html.Div(
        (f"ATR basis (matches the size of the daily moves only, not their direction): ATR of {a} {result.atr_a:.3f} (as of {result.atr_date_a}) ÷ ATR of {b} {result.atr_b:.3f} "
         f"(as of {result.atr_date_b}) = {ratio:.2f}, so {ratio * lots:.2f} lot(s) of {b} per {lots:g} of {a}"
         + (" (not scaled for contract size)." if result.size_ratio != 1 else ".")) if np.isfinite(ratio) else "ATR basis not available.",
        style={**_MUTED, "marginTop": "6px"},
    ))
    pieces.append(html.Div(
        f"Window: the last {window} shared trading days (data {result.first_day} to {result.last_day}). Changes are close to close on days both contracts traded. "
        "Lots are exchange lots (one lot of a spread or fly is one lot of each leg). CL and Brent are 1,000 barrels a lot and gasoil 100 tonnes, so across products the lot ratio is scaled by contract size"
        + (f" (here {result.size_ratio:g} x)" if result.size_ratio != 1 else "") + ". "
        "A hedge with a different contract is never perfect: the hedged daily risk above is what is left.",
        style={**_MUTED, "marginTop": "4px"},
    ))
    return html.Div(pieces)


def hedge_calculator() -> html.Div:
    """Pick any two contracts (outright, spread or fly, CL, BRN or gasoil) and get the hedge ratio of one against the other."""
    return html.Div([
        html.H6("Hedge calculator: any two contracts", style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "26px", "marginBottom": "2px"}),
        html.Div("Pick the contract you hold and the contract you would hedge it with. Live contracts within 'Contracts to watch' are listed, CL, BRN and gasoil.", style={**_MUTED, "marginBottom": "8px"}),
        dbc.Row([
            dbc.Col([html.Div("Position held (A)", style=_LABEL), dcc.Dropdown(id="atr-hedge-a", options=[], clearable=False)], md=3, className="mb-2"),
            dbc.Col([html.Div("Hedge with (B)", style=_LABEL), dcc.Dropdown(id="atr-hedge-b", options=[], clearable=False)], md=3, className="mb-2"),
            dbc.Col([html.Div("Lots of A", style=_LABEL), dbc.Input(id="atr-hedge-lots", type="number", value=1, min=0, step=1, debounce=True, size="sm")], md=2, className="mb-2"),
            dbc.Col([html.Div("Lookback (shared days)", style=_LABEL, title="The hedge ratio is measured on the last N trading days both contracts traded"),
                     dbc.Input(id="atr-hedge-window", type="number", value=60, min=2, max=5000, step=5, debounce=True, size="sm")], md=2, className="mb-2"),
        ], className="g-2"),
        dcc.Loading(html.Div(id="atr-hedge-result"), type="dot"),
    ])


def atr_tab_content() -> html.Div:
    """The whole ATR tab (the tables are filled by a callback when the tab is open)."""
    service = container.oi_service
    params = service.atr_params() if service is not None and service.atr_store is not None else AtrParams()
    return html.Div([
        dcc.Store(id="atr-params"),
        atr_controls(params, datetime.now(timezone.utc).month % 12 + 1),
        dcc.Loading(html.Div(id="atr-wrap"), type="dot"),
        html.H6("One contract against its usual path", style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "22px"}),
        dbc.Row([dbc.Col([html.Div("Contract", style=_LABEL), dcc.Dropdown(id="atr-contract", options=[], clearable=False)], md=3)], className="mb-2"),
        dcc.Graph(id="atr-chart", figure=empty_figure("Pick a contract"), config={"displayModeBar": False}, style={"height": "460px"}),
        hedge_calculator(),
    ], style={"paddingTop": "14px"})
