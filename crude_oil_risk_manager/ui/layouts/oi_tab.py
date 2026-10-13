"""OI / Volume / ATR page (route /oi-volume-atr). One product at a time, chosen in the dropdown at the top.

Live monitor (the main tab): the OI of the next N contracts against the usual for how close each is to
expiry, with sudden/sustained deviations, the front contract's last-month check, the roll estimate and a
chart per contract (core.oi_monitor).

Research tab: what the OI of a finished contract looks like over its life, from the daily OI of every
outright since 2019 (core.oi_research). Every comparison is between contracts of the SAME delivery month
in different years, so the tab has a month selector (and the two "by delivery month" tables show every
month side by side). No statistic mixes one month with another.

ATR tab: see ui.layouts.atr_view. All colours come from COLORS.
"""

from datetime import datetime, timezone

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from core.curve_calendar import PRODUCTS
from core.oi_monitor import MonitorParams
from core.oi_research import OFFSET_LABELS, OI_AT_DAYS, ProductResearch, last_month_table, month_contracts, open_table, threshold_table
from ui.container import container
from ui.layouts.atr_view import atr_tab_content
from ui.layouts.curve_tab import _LABEL, _MUTED, _layout, _table, empty_figure
from ui.layouts.oi_monitor_view import monitor_controls
from ui.layouts.shell import COLORS

REFRESH_MS = 3000
PRODUCT_COLORS = {"CL": "#4da3ff", "BRN": "#f5a623"}
MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
YEAR_COLORS = ["#4da3ff", "#f5a623", "#2ecc71", "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6", "#f1c40f", "#34495e"]


def _num(value, digits=1):
    return None if value is None or (isinstance(value, float) and not np.isfinite(value)) else round(float(value), digits)


def default_month() -> int:
    """The delivery month most likely to be the front contract: next month."""
    return datetime.now(timezone.utc).month % 12 + 1


# ----------------------------------------------------------------------
# Figures (one delivery month)
# ----------------------------------------------------------------------


def build_profile_figure(research: ProductResearch, month: int) -> go.Figure:
    """OI as a share of the contract's peak by days before expiry: each contract of the month, and their median."""
    contracts = month_contracts(research, month)
    if contracts.empty:
        return empty_figure("No finished contracts of this month yet")
    matrix = research.pct_matrix[list(contracts["contract"])] * 100
    figure = go.Figure()
    for i, (symbol, label) in enumerate(zip(contracts["contract"], contracts["label"])):
        series = matrix[symbol].dropna()
        figure.add_trace(go.Scatter(
            x=series.index, y=series.to_numpy(), mode="lines", name=label,
            line={"color": YEAR_COLORS[i % len(YEAR_COLORS)], "width": 1.2}, opacity=0.75,
            hovertemplate=label + ": %{y:.0f}% of peak at %{x} days<extra></extra>",
        ))
    median = matrix.median(axis=1)
    figure.add_trace(go.Scatter(
        x=median.index, y=median.to_numpy(), mode="lines", name="median", line={"color": COLORS["TEXT_PRIMARY"], "width": 3},
        hovertemplate="median: %{y:.0f}% of peak at %{x} days<extra></extra>",
    ))
    figure.update_layout(**_layout(
        title={"text": f"{MONTH_NAMES[month]} contracts: OI as % of the contract's peak, by days before expiry", "font": {"size": 14}},
        xaxis={"title": "Calendar days before expiry (expiry on the right)", "range": [450, 0], "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis={"title": "% of peak OI", "range": [0, 105], "gridcolor": COLORS["BORDER_COLOR"]}, hovermode="x unified",
        uirevision=f"profile-{month}",
    ))
    return figure


def build_peak_timing_figure(research: ProductResearch, month: int) -> go.Figure:
    """Days before expiry at which each contract of the month hit its peak."""
    contracts = month_contracts(research, month)
    if contracts.empty:
        return empty_figure("")
    figure = go.Figure(go.Bar(
        x=contracts["label"], y=contracts["peak_dte"], marker={"color": PRODUCT_COLORS.get(research.product, "#4da3ff")},
        text=contracts["peak_dte"], textposition="outside", hovertemplate="%{x}: peak %{y} days before expiry<extra></extra>",
    ))
    figure.add_hline(y=float(contracts["peak_dte"].median()), line={"color": COLORS["TEXT_SECONDARY"], "dash": "dash"}, annotation_text="median")
    figure.update_layout(**_layout(
        title={"text": f"{MONTH_NAMES[month]} contracts: days before expiry at the peak OI", "font": {"size": 14}},
        yaxis={"title": "Days before expiry", "gridcolor": COLORS["BORDER_COLOR"]}, showlegend=False, margin={"l": 70, "r": 30, "t": 50, "b": 40},
    ))
    return figure


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

MONTH_PEAK_COLUMNS = [
    {"name": "Delivery month", "id": "month"},
    {"name": "Peak OI: median", "id": "median"}, {"name": "mean", "id": "mean"}, {"name": "SD", "id": "sd"},
    {"name": "min", "id": "min"}, {"name": "25th pct", "id": "p25"}, {"name": "75th pct", "id": "p75"}, {"name": "max", "id": "max"},
    {"name": "Days before expiry at peak: median", "id": "dte_median"}, {"name": "SD", "id": "dte_sd"},
]


def month_peak_rows(research: ProductResearch) -> list[dict]:
    """Peak OI statistics for each delivery month Jan to Dec, across that month's finished contracts."""
    rows = []
    for _, r in research.month_peak.iterrows():
        rows.append({
            "month": MONTH_NAMES[int(r["month"])],
            **{key: _num(r[key], 0) for key in ("median", "mean", "sd", "min", "p25", "p75", "max")},
            "dte_median": _num(r["dte_median"], 0), "dte_sd": _num(r["dte_sd"], 1),
        })
    return rows


def month_at_columns() -> list[dict]:
    columns = [{"name": "Delivery month", "id": "month"}]
    for days in OI_AT_DAYS:
        columns.append({"name": f"{days}d before expiry", "id": f"d{days}"})
    return columns


def month_at_rows(research: ProductResearch) -> list[dict]:
    """OI N days before expiry for each delivery month, as 'median (SD)'."""
    rows = []
    for _, r in research.month_at.iterrows():
        row = {"month": MONTH_NAMES[int(r["month"])]}
        for days in OI_AT_DAYS:
            median, sd = r[f"median_{days}"], r[f"sd_{days}"]
            row[f"d{days}"] = "" if not np.isfinite(median) else f"{median:,.0f} ({sd:,.0f})" if np.isfinite(sd) else f"{median:,.0f}"
        rows.append(row)
    return rows


CONTRACT_COLUMNS = [
    {"name": "Contract", "id": "contract"}, {"name": "Expiry (last data day)", "id": "expiry"}, {"name": "Peak OI", "id": "peak"},
    {"name": "Peak date", "id": "peak_date"}, {"name": "Days before expiry at peak", "id": "dte"}, {"name": "Trading days before expiry at peak", "id": "trading"},
    {"name": "Days to reach 50%", "id": "t50"}, {"name": "Days to reach 90%", "id": "t90"},
]


def contract_rows(contracts) -> list[dict]:
    rows = []
    for _, c in contracts.iterrows():
        rows.append({
            "contract": c["label"], "expiry": c["expiry"].date().isoformat(), "peak": _num(c["peak_oi"], 0),
            "peak_date": c["peak_date"].date().isoformat(), "dte": int(c["peak_dte"]), "trading": int(c["peak_dte_trading"]),
            "t50": int(c["t50"]), "t90": int(c["t90"]),
        })
    return rows


THRESHOLD_COLUMNS = [
    {"name": "Reaches", "id": "threshold"}, {"name": "Days before expiry: median", "id": "median"}, {"name": "SD", "id": "sd"},
    {"name": "min", "id": "min"}, {"name": "max", "id": "max"},
]


def threshold_rows(contracts) -> list[dict]:
    return [{"threshold": r["threshold"], **{k: _num(r[k], 0) for k in ("median", "sd", "min", "max")}} for _, r in threshold_table(contracts).iterrows()]


OPEN_COLUMNS = [
    {"name": "Days before expiry", "id": "days"}, {"name": "% of peak still open: median", "id": "median"}, {"name": "SD", "id": "sd"},
    {"name": "min", "id": "min"}, {"name": "max", "id": "max"},
]


def open_rows(contracts) -> list[dict]:
    return [{"days": int(r["days_before_expiry"]), **{k: _num(r[k], 1) for k in ("median", "sd", "min", "max")}} for _, r in open_table(contracts).iterrows()]


LAST_MONTH_COLUMNS = [
    {"name": "Days before expiry", "id": "dte"}, {"name": "% of peak still open: median", "id": "open_median"},
    {"name": "lowest", "id": "open_min"}, {"name": "highest", "id": "open_max"},
    {"name": "Change that day, % of peak (median)", "id": "change_median"}, {"name": "Fell that day (median)", "id": "fall_median"},
    {"name": "of which rolled (est.)", "id": "rolled_median"}, {"name": "of which exited (est.)", "id": "exited_median"},
    *({"name": f"Rolled OI to {label}, %", "id": f"share_{k}"} for k, label in OFFSET_LABELS.items()),
]


def last_month_rows(research: ProductResearch, month: int) -> list[dict]:
    rows = []
    for _, r in last_month_table(research, month).iterrows():
        rows.append({"dte": int(r["dte"]), **{c["id"]: _num(r[c["id"]], 1 if c["id"].startswith(("open", "share")) else 2) for c in LAST_MONTH_COLUMNS[1:]}})
    return rows


def status_text(status: dict) -> html.Div:
    """Download state in one or two lines."""
    parts = []
    if status["running"]:
        total = status["total"] or 1
        parts.append(f"⏳ {status['message']} ({status['done']}/{status['total']} requests, {status['done'] / total:.0%})")
    elif status["updated_on"]:
        parts.append(f"Data updated {status['updated_on']} · {status['history_days']} calendar days of history per contract · OI is the previous trading day's")
        if status["message"]:
            parts.append(status["message"])
    else:
        parts.append("No OI data downloaded yet. It downloads automatically, or press the button.")
    children = [html.Span(" · ".join(parts))]
    if status["error"]:
        children.append(html.Div(f"⚠️ {status['error']}", style={"color": COLORS["ACCENT_RED"], "marginTop": "4px"}))
    return html.Div(children)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _section(title: str, hint: str, body) -> html.Div:
    return html.Div([
        html.H5(title, style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "26px", "marginBottom": "2px"}),
        html.Div(hint, style={**_MUTED, "marginBottom": "10px"}),
        body,
    ])


def research_body(research: ProductResearch, product: str, month: int) -> html.Div:
    """The research of ONE product, with the month-specific parts for ONE delivery month."""
    if research is None or research.empty:
        return html.Div((research.error if research is not None else None) or "No research yet: waiting for the OI download.", style=_MUTED)
    graph = {"displayModeBar": False}
    name = MONTH_NAMES[month]
    contracts = month_contracts(research, month)
    month_part = (
        html.Div(f"No finished {name} contracts yet.", style=_MUTED) if contracts.empty else html.Div([
            _section(
                f"{product} {name} contracts: peak OI and when it came",
                f"One row per finished {name} contract (one per year). Only {name} contracts are compared with each other.",
                html.Div([
                    _table("oi-contract-table", CONTRACT_COLUMNS, contract_rows(contracts), page_size=12),
                    dbc.Row([dbc.Col(dcc.Graph(figure=build_peak_timing_figure(research, month), config=graph, style={"height": "320px"}), md=5),
                             dbc.Col(dcc.Graph(figure=build_profile_figure(research, month), config=graph, style={"height": "320px"}), md=7)], className="mt-3"),
                ]),
            ),
            _section(
                f"{name}: how the OI builds up and drains",
                f"Across the {name} contracts: how many days before expiry they reached each share of their peak, and what share of the peak was still open N days before expiry. "
                "The spread across the years is the SD, min and max.",
                dbc.Row([dbc.Col(_table("oi-threshold-table", THRESHOLD_COLUMNS, threshold_rows(contracts), page_size=12), md=6),
                         dbc.Col(_table("oi-open-table", OPEN_COLUMNS, open_rows(contracts), page_size=12), md=6)]),
            ),
            _section(
                f"{name}: the last month, day by day",
                f"One row per calendar day before expiry, across the {name} contracts. As a share of each contract's peak OI: how much is still open, the "
                "fall that day, how much of it is estimated to have rolled into other contracts and how much exited, and where the rolled OI went. "
                "A day with no trading in any of these contracts is blank. The roll split divides each day's fall in proportion to the rises of the other "
                "contracts that day, so it is an estimate, not who actually rolled.",
                _table("oi-last-month-table", LAST_MONTH_COLUMNS, last_month_rows(research, month), page_size=31),
            ),
        ])
    )
    return html.Div([
        _section(
            f"{product} peak OI by delivery month",
            "Each row is one delivery month (Jan to Dec) over all the finished contracts of that month: the median, mean and standard deviation "
            "(SD) of their peak OI, the spread around it, and when in the last months the peak came.",
            _table("oi-month-peak-table", MONTH_PEAK_COLUMNS, month_peak_rows(research), page_size=12),
        ),
        _section(
            f"{product} OI by delivery month, N days before expiry",
            "The OI of the finished contracts of each delivery month at the same number of calendar days before expiry: median (SD). "
            "This is what the live monitor compares today's OI with.",
            _table("oi-month-at-table", month_at_columns(), month_at_rows(research), page_size=12),
        ),
        month_part,
    ])


def oi_layout() -> html.Div:
    service = container.oi_service
    params = service.monitor_params() if service is not None else MonitorParams()
    product_options = [{"label": p.name, "value": code} for code, p in PRODUCTS.items()]
    month_options = [{"label": MONTH_NAMES[m], "value": m} for m in range(1, 13)]
    return html.Div([
        dcc.Interval(id="oi-refresh", interval=REFRESH_MS, n_intervals=0),
        dcc.Store(id="oi-version"),
        dcc.Store(id="oi-params"),
        html.H3("📊 OI · Volume · ATR", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "4px"}),
        dbc.Row([
            dbc.Col([html.Div("Product", style=_LABEL), dcc.Dropdown(id="oi-product", options=product_options, value="CL", clearable=False)], md=3),
            dbc.Col(html.Div(id="oi-status", style={**_MUTED, "minHeight": "20px"}), md=7, className="pt-3"),
            dbc.Col(dbc.Button("Download / update data", id="oi-update-btn", size="sm", color="secondary", outline=True), md=2, className="text-end pt-3"),
        ], className="mb-3 align-items-center"),
        dbc.Tabs([
            dbc.Tab(html.Div([
                monitor_controls(params),
                dcc.Loading(html.Div(id="oi-monitor-wrap"), type="dot"),
                html.H6("One contract against its usual path", style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "22px"}),
                dbc.Row([dbc.Col([html.Div("Contract", style=_LABEL), dcc.Dropdown(id="oi-contract", options=[], clearable=False)], md=3)], className="mb-2"),
                dcc.Graph(id="oi-chart", figure=empty_figure("Pick a contract"), config={"displayModeBar": False}, style={"height": "460px"}),
            ], style={"paddingTop": "14px"}), label="Live monitor", tab_id="monitor"),
            dbc.Tab(html.Div([
                dbc.Row([dbc.Col([html.Div("Delivery month", style=_LABEL), dcc.Dropdown(id="oi-month", options=month_options, value=default_month(), clearable=False)], md=3)],
                        className="mb-1 mt-3"),
                dcc.Loading(html.Div(id="oi-research-wrap")),
            ]), label="Research", tab_id="research"),
            dbc.Tab(atr_tab_content(), label="ATR", tab_id="atr"),
        ], id="oi-tabs", active_tab="monitor"),
    ])
