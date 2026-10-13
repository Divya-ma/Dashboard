"""Live monitor view for the OI page: parameter inputs, summary cards, the contract table, the roll
estimate and the chart of one contract against its two baselines (core.oi_monitor). Colours from COLORS."""

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dash_table.Format import Format, Group, Scheme

from core.oi_monitor import STATUS_SUDDEN, STATUS_SUSTAINED, MonitorParams, MonitorResult
from ui.layouts.curve_tab import _CARD, _LABEL, _MUTED, _layout, _table, empty_figure
from ui.layouts.shell import COLORS

STALE_DATA_DAYS = 4  # warn when the newest OI is older than this many calendar days
STATUS_COLORS = {STATUS_SUDDEN: COLORS["ACCENT_RED"], STATUS_SUSTAINED: COLORS["ACCENT_YELLOW"]}
VERDICT_COLORS = {"ALERT": COLORS["ACCENT_RED"], "WATCH": COLORS["ACCENT_YELLOW"], "INFO": "#5dade2", "OK": COLORS["ACCENT_GREEN"], "": COLORS["TEXT_SECONDARY"]}

# (id, name, parameter field, step, hint)
PARAM_INPUTS = [
    ("oi-p-contracts", "Contracts to watch", "contracts", 1, "The next N live contracts"),
    ("oi-p-sudden", "Sudden window (days)", "sudden_days", 1, "A deviation younger than this many trading days is SUDDEN; older is SUSTAINED"),
    ("oi-p-z", "Deviation threshold (z)", "z_threshold", 0.1, "How many standard deviations from the usual counts as a deviation"),
    ("oi-p-years", "Same month, last N years", "years", 1, "Compare with the same delivery month in the last N years at the same days before expiry"),
    ("oi-p-minoi", "Ignore OI below", "min_oi", 500, "A far contract with almost no OI moves a lot in percent for no reason"),
    ("oi-p-minchg", "Ignore moves below", "min_change", 500, "Smallest OI gap from the usual that counts"),
]


def _number(value, digits=0):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), digits)


def monitor_controls(params: MonitorParams) -> html.Div:
    """The parameter inputs (applied on Enter or when the field is left) and the alerts switch."""
    columns = [
        dbc.Col([
            html.Div(name, style=_LABEL, title=hint),
            dbc.Input(id=component_id, type="number", value=getattr(params, field), step=step, debounce=True, size="sm"),
        ], md=3, lg=True, className="mb-2")
        for component_id, name, field, step, hint in PARAM_INPUTS
    ]
    return html.Div([
        dbc.Row(columns, className="g-2"),
        dbc.Switch(id="oi-p-alerts", label="Teams / in-app alert when a contract suddenly deviates (outside the same-month baseline)", value=params.alerts_enabled, className="mt-1"),
    ], style={**_CARD, "padding": "12px 14px", "borderRadius": "6px", "marginBottom": "12px"})


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

_INT = Format(group=Group.yes, precision=0, scheme=Scheme.fixed)
_ONE = Format(precision=1, scheme=Scheme.fixed)


def _col(name, column_id, fmt=None):
    column = {"name": name, "id": column_id}
    if fmt is not None:
        column.update({"type": "numeric", "format": fmt})
    return column


def monitor_columns(params: MonitorParams) -> list[dict]:
    n = params.sudden_days
    return [
        _col("Contract", "label"), _col("Days to expiry", "dte", _INT), _col("OI", "oi", _INT),
        _col("1-day change", "d1", _INT), _col(f"{n}-day change %", "dn_pct", _ONE), _col("% of peak so far", "pct", _ONE),
        _col(f"Usual (same month, {params.years}y)", "b_median", _INT), _col("Past range", "b_range"),
        _col("z", "b_z", _ONE),
        _col("Share of product OI %", "share", _ONE), _col("share z", "share_z", _ONE),
        _col("Status", "status"), _col("Days deviating", "run", _INT),
        _col("Since", "since"), _col("Last day read", "read1"), _col(f"{n}-day read", "readn"),
    ]


COLUMN_TIPS = {
    "b_z": "Distance of today's figure from the usual of the same delivery month in past years, at the same days before expiry, in standard deviations.",
    "share": "This contract's OI as a share of the total OI of all the product's contracts.",
    "share_z": "Distance of that share from the same delivery month's usual share at the same days before expiry, in standard deviations. "
               "High: the contract holds an unusually large part of the market's OI for this point in its life (market growth removed).",
    "pct": "OI as a share of the highest OI this contract has had so far.",
    "run": "Consecutive trading days up to the latest one that the contract has been deviating from the usual.",
}


def monitor_rows(result: MonitorResult) -> list[dict]:
    rows = []
    for r in result.rows.itertuples():
        b_range = f"{r.b_min:,.0f} to {r.b_max:,.0f}" if np.isfinite(r.b_min) and np.isfinite(r.b_max) and r.b_n else ""
        rows.append({
            "label": r.label, "dte": r.dte, "oi": _number(r.oi), "d1": _number(r.d1_oi), "dn_pct": _number(r.dn_pct, 1),
            "pct": _number(r.pct_of_peak * 100, 1) if np.isfinite(r.pct_of_peak) else None,
            "b_median": _number(r.b_median), "b_range": b_range,
            "b_z": _number(r.b_z, 1) if r.b_reliable else None,
            "share": _number(r.share * 100, 1) if np.isfinite(r.share) else None, "share_z": _number(r.share_z, 1),
            "status": r.status, "run": r.run_days, "since": str(r.since) if r.since else "",
            "read1": r.read_1d, "readn": r.read_n,
        })
    return rows


def monitor_styles(params: MonitorParams) -> list[dict]:
    z = params.z_threshold
    styles = []
    for column in ("b_z", "share_z"):
        styles.append({"if": {"column_id": column, "filter_query": f"{{{column}}} >= {z} || {{{column}}} <= {-z}"}, "color": COLORS["ACCENT_YELLOW"], "fontWeight": "bold"})
    for status, color in STATUS_COLORS.items():
        styles.append({"if": {"column_id": "status", "filter_query": f'{{status}} = "{status}"'}, "color": color, "fontWeight": "bold"})
    styles.append({"if": {"filter_query": f'{{status}} = "{STATUS_SUDDEN}"'}, "backgroundColor": "rgba(231, 76, 60, 0.10)"})
    return styles


def roll_rows(result: MonitorResult) -> list[dict]:
    rows = []
    for r in result.roll.itertuples():
        rows.append({
            "label": r.label, "change": _number(r.change), "exited": _number(r.exited),
            "rolled_in": _number(r.rolled_in), "fresh": _number(r.fresh),
        })
    return rows


ROLL_COLUMNS = [
    _col("Contract", "label"), _col("OI change", "change", _INT), _col("Left: exited (est.)", "exited", _INT),
    _col("Arrived: rolled in from the front contract (est.)", "rolled_in", _INT), _col("Arrived: fresh OI (est.)", "fresh", _INT),
]


# ----------------------------------------------------------------------
# Body
# ----------------------------------------------------------------------


def _card(title: str, value: str, detail: str = "", color: str | None = None) -> dbc.Col:
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(title, style=_LABEL),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "bold", "color": color or COLORS["TEXT_PRIMARY"]}),
        html.Div(detail, style=_MUTED),
    ]), style=_CARD), md=3, className="mb-2")


def monitor_body(result: MonitorResult, params: MonitorParams, today) -> html.Div:
    """Cards, the contract table and the roll estimate for one product."""
    if result.error or result.rows.empty:
        return html.Div(result.error or "Nothing to monitor yet", style=_MUTED)
    rows = result.rows
    sudden = int((rows["status"] == STATUS_SUDDEN).sum())
    sustained = int((rows["status"] == STATUS_SUSTAINED).sum())
    age = (today - result.data_date.date()).days
    totals = result.totals
    d1, d5 = totals.get("d1"), totals.get("d5")
    detail = " · ".join(
        f"{label} {value:+,.0f}" for label, value in (("1d", d1), ("5d", d5)) if value is not None and np.isfinite(value)
    )
    cards = dbc.Row([
        _card("OI as of", str(result.data_date.date()), f"{age} day(s) ago · yesterday's OI is the latest published",
              COLORS["ACCENT_YELLOW"] if age > STALE_DATA_DAYS else None),
        _card(f"{result.product} total OI (tracked contracts)", f"{totals.get('total_oi', float('nan')):,.0f}", detail),
        _card("Sudden deviations", str(sudden), f"within {params.sudden_days} trading days", COLORS["ACCENT_RED"] if sudden else None),
        _card("Sustained deviations", str(sustained), "longer than that: likely a lasting reason", COLORS["ACCENT_YELLOW"] if sustained else None),
    ])
    summary = result.roll_summary
    roll_note = (
        f"{summary['date']}: {summary['decrease']:,.0f} left falling contracts, {summary['increase']:,.0f} arrived in rising ones; "
        f"about {summary['rolled']:,.0f} is estimated to have rolled, {summary['exited']:,.0f} exited and {summary['fresh']:,.0f} is fresh."
        if summary else ""
    )
    return html.Div([
        cards,
        html.Div(
            "Each contract's OI is compared with the same delivery month's finished contracts in the last "
            f"{params.years} years, at the SAME number of days before expiry (never with other months). While OI is still building the "
            "absolute OI is compared; once it has passed its peak, its % of the peak so far. z is the distance from the usual in standard "
            f"deviations. SUDDEN = deviating for at most {params.sudden_days} trading days; SUSTAINED = longer. "
            "Hover over a column heading for what it means.",
            style={**_MUTED, "margin": "6px 0 10px"},
        ),
        _with_tips(_table("oi-monitor-table", monitor_columns(params), monitor_rows(result), monitor_styles(params), page_size=20)),
        front_check_body(result.front, params),
        html.H6("Yesterday's change in OI and where it is estimated to have gone", style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "22px"}),
        html.Div(roll_note + " Rolled-in counts only what is estimated to have come from the front contract; "
                 "fresh OI is the rest of a contract's rise after anything that rolled in from any contract.", style={**_MUTED, "marginBottom": "8px"}),
        _table("oi-roll-table-live", ROLL_COLUMNS, roll_rows(result), page_size=20),
    ])


def _with_tips(table):
    """Hover text for the column headings."""
    table.tooltip_header = {key: {"value": text, "type": "text"} for key, text in COLUMN_TIPS.items()}
    table.tooltip_delay = 0
    table.tooltip_duration = None
    return table


# ----------------------------------------------------------------------
# The front contract's last month
# ----------------------------------------------------------------------

FRONT_COLUMNS = [
    _col("Measured as a share of the contract's peak OI", "metric"), _col("Now %", "now", _ONE),
    _col("Usual: same month in past years, median %", "b_median", _ONE), _col("lowest to highest", "b_range"), _col("z", "b_z", _ONE),
]


def front_rows(front: dict) -> list[dict]:
    def pct(value):
        return _number(value * 100, 1) if value is not None and np.isfinite(value) else None

    def span(low, high):
        return f"{low * 100:.1f} to {high * 100:.1f}" if np.isfinite(low) and np.isfinite(high) else ""

    return [{
        "metric": m["label"], "now": pct(m["now"]),
        "b_median": pct(m["b_median"]), "b_range": span(m["b_min"], m["b_max"]), "b_z": _number(m["b_z"], 1),
    } for m in front["metrics"]]


def front_check_body(front: dict | None, params: MonitorParams) -> html.Div:
    """Is the contract closest to expiry draining like finished contracts did at this point?"""
    if not front:
        return html.Div()
    title = html.H6(
        f"Front contract {front['label']}: {front['dte']} days to expiry. Is it rolling as usual?",
        style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "22px"},
    )
    if front.get("note"):
        return html.Div([title, html.Div(front["note"], style=_MUTED)])
    colour = VERDICT_COLORS.get(front["level"], COLORS["TEXT_SECONDARY"])
    z = params.z_threshold
    styles = [
        {"if": {"column_id": column, "filter_query": f"{{{column}}} >= {z} || {{{column}}} <= {-z}"}, "color": COLORS["ACCENT_YELLOW"], "fontWeight": "bold"}
        for column in ("b_z",)
    ]
    return html.Div([
        title,
        html.Div(
            [html.Span(f"{front['level']}  " if front["level"] else "", style={"color": colour, "fontWeight": "bold"}), front["verdict"]],
            style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "8px"},
        ),
        html.Div(
            "Each measure is a share of the contract's own peak OI, compared with the finished contracts at this same number of days before expiry. "
            "More OI still open and less rolled than usual, together, is what a squeeze or delivery interest looks like; OI leaving early is the opposite. "
            "Rolled is the model estimate (a fall is split over the contracts that rose that day), not who actually rolled.",
            style={**_MUTED, "marginBottom": "8px"},
        ),
        _table("oi-front-table", FRONT_COLUMNS, front_rows(front), styles, page_size=6),
        dcc.Graph(figure=build_front_chart(front.get("chart")), config={"displayModeBar": False}, style={"height": "340px"}),
    ])


def build_front_chart(data: dict | None) -> go.Figure:
    """OI still open (% of peak) in the last month: the front contract against past contracts."""
    if not data:
        return empty_figure("")
    figure = go.Figure()
    days = data["dte"]
    d = data["B"]
    color = "77,163,255"
    figure.add_trace(go.Scatter(x=days, y=d["max"] * 100, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(
        x=days, y=d["min"] * 100, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=f"rgba({color},0.15)",
        name=f"Same month ({', '.join(d['labels'])}): range", hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=days, y=d["median"] * 100, mode="lines", name="Same month: median", line={"color": f"rgba({color},0.95)", "width": 2},
        hovertemplate="%{x} days before expiry: usual %{y:.1f}%<extra></extra>",
    ))
    live_days, live_values = data["live"]
    figure.add_trace(go.Scatter(
        x=live_days, y=np.asarray(live_values) * 100, mode="lines+markers", name=f"{data['label']} (live)",
        line={"color": COLORS["ACCENT_GREEN"], "width": 3}, marker={"size": 5},
        hovertemplate="%{x} days before expiry: %{y:.1f}% of peak<extra></extra>",
    ))
    figure.update_layout(**_layout(
        title={"text": f"{data['label']}: OI still open in the last month, against past contracts", "font": {"size": 14}},
        xaxis={"title": "Calendar days before expiry (expiry on the right)", "range": [30, 0], "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis={"title": "% of the contract's peak OI", "gridcolor": COLORS["BORDER_COLOR"], "rangemode": "tozero"},
        hovermode="x unified", uirevision=data["label"],
    ))
    return figure


# ----------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------


def build_monitor_chart(data: dict | None) -> go.Figure:
    """A live contract's OI by days before expiry with the same month's usual path (median and the band that would count as a deviation)."""
    if data is None:
        return empty_figure("Pick a contract")
    figure = go.Figure()
    horizon = float(np.max(data["dte"])) * 1.04 + 5
    d = data["B"]
    color = "rgba(77,163,255,{})"
    keep = d["dte"] <= horizon
    x, low, high, median = d["dte"][keep], d["low"][keep], d["high"][keep], d["median"][keep]
    figure.add_trace(go.Scatter(x=x, y=high, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    figure.add_trace(go.Scatter(
        x=x, y=low, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=color.format(0.13),
        name=f"band (usual ± {data['threshold']:g} z)", hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=x, y=median, mode="lines", name=f"Same month ({', '.join(d['labels'])}): usual", line={"color": color.format(0.95), "width": 2},
        hovertemplate="%{x} days before expiry: usual %{y:,.0f}<extra></extra>",
    ))
    for label, y in data["B"]["lines"].items():
        keep = data["B"]["dte"] <= horizon
        figure.add_trace(go.Scatter(
            x=data["B"]["dte"][keep], y=np.asarray(y)[keep], mode="lines", name=label, line={"color": "rgba(150,160,170,0.55)", "width": 1},
            hovertemplate=label + ": %{y:,.0f}<extra></extra>", visible="legendonly",
        ))
    figure.add_trace(go.Scatter(
        x=data["dte"], y=data["oi"], mode="lines+markers", name=f"{data['label']} (live)",
        line={"color": COLORS["ACCENT_GREEN"], "width": 3}, marker={"size": 4},
        customdata=[str(d.date()) for d in data["dates"]],
        hovertemplate="%{customdata}: %{y:,.0f} (%{x} days before expiry)<extra></extra>",
    ))
    figure.update_layout(**_layout(
        title={"text": f"{data['label']}: OI by days before expiry against its usual path", "font": {"size": 14}},
        xaxis={"title": "Calendar days before expiry (expiry on the right)", "range": [horizon, 0], "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis={"title": "Open interest", "gridcolor": COLORS["BORDER_COLOR"], "rangemode": "tozero"},
        hovermode="x unified", uirevision=data["symbol"],
    ))
    return figure
