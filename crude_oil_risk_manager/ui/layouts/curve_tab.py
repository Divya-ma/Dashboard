"""Curve Kinks page (route /trade-analyzer): live curves, kinks, data quality, log, backtest, thresholds.

Nothing is typed in to analyse: the engine (core.curve_service) prices the whole 15-month
ladder of CL and BRN every poll and runs the kink maths on every curve in the background;
this page only chooses which curve to display. All colours come from COLORS.
"""

from datetime import datetime, timezone

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from core.curve_calendar import DFLY, FAMILIES, FAMILY_LABELS, OUTRIGHT, PRODUCTS
from core.curve_kinks import METHOD_LABELS, METHODS, trade_side
from core.curve_service import FamilyView
from core.curve_settings import PRIORITIES, CurveParams, load_params
from ui.container import container
from ui.layouts.shell import COLORS

REFRESH_MS = 10_000
PLACEHOLDER = "Waiting for live prices..."

_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_LABEL = {**_MUTED, "textTransform": "uppercase", "letterSpacing": "0.05em", "marginBottom": "6px"}
_CARD = {"backgroundColor": COLORS["CARD_BG"], "border": f"1px solid {COLORS['BORDER_COLOR']}"}

PRIORITY_COLORS = {
    "HIGH": COLORS["ACCENT_RED"],
    "MEDIUM": COLORS["ACCENT_YELLOW"],
    "LOW": "#5dade2",
    "NONE": COLORS["TEXT_SECONDARY"],
}
SEVERITY_COLORS = {"BUFFER": COLORS["ACCENT_RED"], "WARNING": COLORS["ACCENT_YELLOW"], "INFO": COLORS["TEXT_SECONDARY"]}

# (name, label, hint, step) for the threshold form; names match core.curve_settings.CurveParams.
PARAM_FIELDS = [
    ("z_fit", "Fit z", "Robust curve-fit residual; flag when |z| exceeds this", 0.1),
    ("z_neighbour", "Neighbour z", "Interpolation from neighbouring contracts", 0.1),
    ("z_pca", "PCA z", "Residual after rebuilding the curve from principal components", 0.1),
    ("z_history", "History z", "Value vs its own history (spread / fly / dfly)", 0.1),
    ("min_methods_high", "Methods for HIGH", "How many methods must agree for HIGH priority", 1),
    ("max_kink_fraction", "Max kinked share", "If more than this share of a curve is flagged it is treated as a data mismatch or a whole-curve move: no kinks or alerts for it", 0.05),
    ("seasonal_z", "Seasonal z", "|seasonal z| at or below this = seasonally normal (priority drops one level)", 0.1),
    ("seasonal_window_days", "Seasonal window (days)", "+/- days around today's date in earlier years", 1),
    ("lookback_days", "Lookback (days)", "History used for scales, PCA and own-history z", 10),
    ("pca_components", "PCA components", "Level / slope / curvature ...", 1),
    ("poly_degree", "Fit polynomial degree", "Degree of the robust curve fit", 1),
    ("min_scale", "Minimum scale (points)", "Floor on every scale, so a flat curve can't give huge z", 0.001),
    ("cooldown_minutes", "Alert cooldown (min)", "A cleared kink that returns within this window does not re-alert", 5),
    ("stale_seconds", "Stale price (seconds)", "Live prices older than this are flagged", 5),
    ("snapshot_minutes", "Snapshot every (min)", "How often today's curve snapshot is saved", 1),
    ("history_years", "History (years)", "Years of generic history to load", 1),
]


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def _layout(**overrides) -> dict:
    layout = {
        "paper_bgcolor": COLORS["CARD_BG"], "plot_bgcolor": COLORS["CARD_BG"],
        "font": {"color": COLORS["TEXT_PRIMARY"]}, "margin": {"l": 70, "r": 30, "t": 50, "b": 90},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02},
    }
    layout.update(overrides)
    return layout


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font={"size": 16, "color": COLORS["TEXT_SECONDARY"]})
    figure.update_layout(**_layout(xaxis={"visible": False}, yaxis={"visible": False}))
    return figure


def axis_label(structure) -> str:
    """Short x-axis text: a fly/dfly shows first->last month; the full label stays in the hover."""
    if structure.family in ("fly", DFLY):
        parts = structure.label.split("/")
        return f"{parts[0]}→{parts[-1]}"
    return structure.label


def _fmt(value, digits=2, signed=False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _hover(position) -> str:
    structure, result = position.structure, position.result
    lines = [f"<b>{structure.label}</b>", f"Live {_fmt(position.live)}"]
    if position.prev_settle is not None:
        lines.append(f"Prev settle {_fmt(position.prev_settle)} (Δ {_fmt(position.change, signed=True)})")
    if result is not None and result.implied is not None:
        lines.append(f"Implied by neighbours {_fmt(result.implied)}")
    if result is not None and result.n_flags and result.echo_of is None:
        z = ", ".join(f"{METHOD_LABELS[m]} {result.z[m]:+.1f}" for m in METHODS if result.flags[m])
        lines.append(f"<b>KINK {result.priority}: {trade_side(result.direction)}</b> ({result.direction}): {z}")
        if result.seasonal_normal:
            lines.append(f"seasonally normal (z {result.seasonal_z:+.1f})")
    if position.stale:
        lines.append("⚠ stale price")
    return "<br>".join(lines)


def build_curve_figure(view: FamilyView | None, snapshot: dict | None, snapshot_date: str | None = None) -> go.Figure:
    """Live curve with kink markers, previous settlement, optional snapshot and the robust fit."""
    if view is None or not any(p.live is not None for p in view.positions):
        return empty_figure(PLACEHOLDER)
    positions = view.positions
    x = [axis_label(p.structure) for p in positions]
    figure = go.Figure()
    kink = [] if view.suspect else [
        p for p in positions if p.result is not None and p.result.n_flags and p.result.echo_of is None
    ]

    figure.add_trace(go.Scatter(
        x=x, y=[p.live for p in positions], mode="lines+markers", name="Live",
        line={"color": "#4da3ff", "width": 2}, marker={"size": 7, "color": "#4da3ff"},
        hovertext=[_hover(p) for p in positions], hoverinfo="text", connectgaps=False,
    ))
    if any(p.prev_settle is not None for p in positions):
        figure.add_trace(go.Scatter(
            x=x, y=[p.prev_settle for p in positions], mode="lines+markers", name="Previous settlement",
            line={"color": COLORS["TEXT_SECONDARY"], "width": 2, "dash": "dash"}, marker={"size": 5}, connectgaps=False,
            hovertemplate="%{x}<br>Prev settle %{y:.2f}<extra></extra>",
        ))
    if snapshot:
        figure.add_trace(go.Scatter(
            x=x, y=[snapshot.get(p.structure.label) for p in positions], mode="lines", connectgaps=False,
            name=f"Snapshot {snapshot_date}", line={"color": COLORS["ACCENT_GREEN"], "width": 1.5, "dash": "dot"},
            hovertemplate="%{x}<br>Snapshot %{y:.2f}<extra></extra>",
        ))
    fitted = [p.result.fitted if p.result else None for p in positions]
    if any(f is not None for f in fitted):
        figure.add_trace(go.Scatter(
            x=x, y=fitted, mode="lines", name="Robust fit", visible="legendonly",
            line={"color": "#bbbbbb", "width": 1, "dash": "dot"}, hoverinfo="skip",
        ))
    if kink:
        figure.add_trace(go.Scatter(
            x=[axis_label(p.structure) for p in kink], y=[p.result.implied for p in kink], mode="markers",
            name="Implied by neighbours", marker={"symbol": "x", "size": 9, "color": COLORS["TEXT_PRIMARY"]},
            hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter(
            x=[axis_label(p.structure) for p in kink], y=[p.live for p in kink], mode="markers", name="Kink",
            marker={
                "size": [12 + 3 * p.result.n_flags for p in kink], "symbol": "diamond-open",
                "color": [PRIORITY_COLORS[p.result.priority] for p in kink], "line": {"width": 3},
            },
            hovertext=[_hover(p) for p in kink], hoverinfo="text",
        ))
    figure.update_layout(**_layout(
        title={"text": f"{PRODUCTS[view.product].name} — {FAMILY_LABELS[view.family]} curve", "font": {"size": 18}},
        yaxis={"title": "Price (points)", "zeroline": view.family != OUTRIGHT},
        xaxis={"tickangle": -35, "automargin": True}, hovermode="closest",
    ))
    return figure


def build_strength_figure(view: FamilyView | None, params: CurveParams) -> go.Figure:
    """Per contract: the strongest method's |z| as a multiple of its threshold (>= 1 means flagged)."""
    if view is None or not any(p.result is not None for p in view.positions):
        return empty_figure(PLACEHOLDER)
    thresholds = params.thresholds
    x, y, colors, text = [], [], [], []
    for p in view.positions:
        result = p.result
        strengths = (
            {m: abs(result.z[m]) / thresholds[m] for m in METHODS if result.z.get(m) is not None} if result else {}
        )
        top = max(strengths.values(), default=0.0)
        x.append(axis_label(p.structure))
        y.append(top)
        is_kink = not view.suspect and result is not None and result.n_flags and result.echo_of is None
        colors.append(PRIORITY_COLORS[result.priority] if is_kink else COLORS["ACCENT_BLUE"])
        text.append(
            f"{p.structure.label}<br>" + "<br>".join(
                f"{METHOD_LABELS[m]}: z {result.z[m]:+.1f}" for m in METHODS if result and result.z.get(m) is not None
            )
        )
    figure = go.Figure(go.Bar(x=x, y=y, marker={"color": colors}, hovertext=text, hoverinfo="text"))
    figure.add_hline(y=1.0, line_dash="dot", line_color=COLORS["ACCENT_YELLOW"], annotation_text="threshold")
    figure.update_layout(**_layout(
        title={"text": "Kink strength (strongest method, × its threshold)", "font": {"size": 14}},
        yaxis={"title": "× threshold"}, xaxis={"tickangle": -35, "automargin": True},
        margin={"l": 70, "r": 30, "t": 40, "b": 80}, showlegend=False,
    ))
    return figure


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------


def _table(component_id: str, columns: list[dict], rows: list[dict], style_data_conditional=None, page_size=15):
    return dash_table.DataTable(
        id=component_id, columns=columns, data=rows, sort_action="native", page_size=page_size,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": COLORS["SIDEBAR_BG"], "color": COLORS["TEXT_SECONDARY"],
            "border": f"1px solid {COLORS['BORDER_COLOR']}", "fontWeight": "bold",
        },
        style_cell={
            "backgroundColor": COLORS["CARD_BG"], "color": COLORS["TEXT_PRIMARY"],
            "border": f"1px solid {COLORS['BORDER_COLOR']}", "padding": "6px 10px", "textAlign": "left",
            "whiteSpace": "normal", "height": "auto",
        },
        style_data_conditional=style_data_conditional or [],
    )


def _priority_styles(column: str = "priority") -> list[dict]:
    return [
        {"if": {"column_id": column, "filter_query": f'{{{column}}} = "{p}"'}, "color": c, "fontWeight": "bold"}
        for p, c in PRIORITY_COLORS.items()
    ]


def kinks_rows(kinks: list[dict]) -> list[dict]:
    rows = []
    for k in kinks:
        if k["seasonal_z"] is None:
            seasonal = "no baseline"
        else:
            seasonal = f"{'normal' if k['seasonal_normal'] else 'unusual'} (z {k['seasonal_z']:+.1f})"
        rows.append({
            "priority": k["priority"], "trade": k["trade"], "product": k["product"], "family": FAMILY_LABELS[k["family"]],
            "structure": k["label"], "generic": k["generic"], "direction": k["direction"],
            "methods": " · ".join(f"{METHOD_LABELS[m]} {k['z'][m]:+.1f}" for m in METHODS if k["flags"].get(m)),
            "score": round(k["score"], 1), "seasonal": seasonal,
            "live": _fmt(k["value"]), "implied": _fmt(k["implied"]), **_plan_columns(k.get("plan")),
        })
    return rows


def _plan_columns(plan) -> dict:
    """The trade plan's headline numbers as table cells (blank until the plan can be built)."""
    if plan is None:
        return {"entry": "", "lots": "", "stop": "", "target": "", "rr": "", "hedge": ""}
    hedge = plan.hedges[0] if plan.hedges else None
    return {
        "entry": _fmt(plan.entry), "lots": plan.lots, "stop": _fmt(plan.stop),
        "target": _fmt(plan.target), "rr": "" if plan.rr is None else f"{plan.rr:.1f}",
        "hedge": "" if hedge is None else f"{hedge.side} {hedge.lots} {hedge.family} {hedge.label}",
    }


def build_kinks_table(kinks: list[dict]) -> html.Div:
    if not kinks:
        return html.Div("No kinks at the current thresholds.", style={**_MUTED, "padding": "12px"})
    columns = [{"id": c, "name": n} for c, n in (
        ("priority", "Priority"), ("trade", "Trade"), ("product", "Product"), ("family", "Curve"),
        ("structure", "Structure"), ("generic", "Generic"), ("direction", "vs curve"), ("methods", "Methods flagged (z)"), ("score", "Score"),
        ("seasonal", "Seasonality"), ("live", "Live"), ("implied", "Implied"), ("entry", "Entry"), ("lots", "Lots"),
        ("stop", "Stop"), ("target", "Target"), ("rr", "R:R"), ("hedge", "Best hedge"),
    )]
    return _table("curve-kinks-table", columns, kinks_rows(kinks), _priority_styles())


PLAN_CARD_LIMIT = 10


def _plan_card(kink: dict) -> dbc.Card:
    plan = kink.get("plan")
    family = FAMILY_LABELS[kink["family"]]
    colour = PRIORITY_COLORS[kink["priority"]]
    header = html.Div([
        html.Span(f"{kink['priority']} ", style={"color": colour, "fontWeight": "bold"}),
        html.Span(f"{kink['trade']} {kink['product']} {family} {kink['label']}", style={"fontWeight": "bold", "fontSize": "17px"}),
        html.Span(f"   {kink['generic']} is {kink['direction']} vs the curve", style=_MUTED),
    ])
    body: list = [header, html.Div(f"Legs: {kink['legs']}", style={**_MUTED, "margin": "4px 0 8px"})]
    if plan is None:
        body.append(html.Div(
            "Lots, stop, target and hedges need price history, which is not loaded yet.",
            style={"color": COLORS["ACCENT_YELLOW"]},
        ))
        return dbc.Card(dbc.CardBody(body), style={**_CARD, "borderLeft": f"4px solid {colour}"}, className="mb-2")

    def stat(label, value, sub=""):
        return dbc.Col([
            html.Div(label, style=_LABEL),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "bold", "color": COLORS["TEXT_PRIMARY"]}),
            html.Div(sub, style={**_MUTED, "fontSize": "11px"}),
        ], xs=6, md=2, className="mb-2")

    target = "no target" if plan.target is None else _fmt(plan.target)
    target_sub = "" if plan.target is None else f"{plan.target_distance:.2f} pts · {plan.reversion_share:.0%} of gap to fair {_fmt(plan.fair_value)}"
    body.append(dbc.Row([
        stat(f"{plan.side} at", _fmt(plan.entry), "live price"),
        stat("Lots", str(plan.lots), f"risk ${plan.dollar_risk:,.0f}"),
        stat("Stop", _fmt(plan.stop), f"{plan.stop_distance:.2f} pts · {plan.stop_basis}"),
        stat("Target", target, target_sub),
        stat("Reward : risk", "n/a" if plan.rr is None else f"{plan.rr:.1f}",
             "" if plan.dollar_reward is None else f"reward ${plan.dollar_reward:,.0f}"),
        stat("Time stop", "n/a" if not plan.time_stop_days else f"~{plan.time_stop_days}d",
             "" if plan.half_life_days is None else f"half-life {plan.half_life_days:.1f}d"),
    ]))
    if plan.hedges:
        rows = [{
            "side": h.side, "lots": h.lots, "structure": f"{FAMILY_LABELS[h.family]} {h.label}", "legs": h.legs,
            "entry": _fmt(h.entry), "ratio": f"{h.ratio:.2f}", "minvar": f"{abs(h.ratio_minvar):.2f}",
            "corr": f"{h.corr:+.2f}", "var": f"-{h.var_reduction:.0%}", "note": h.warning,
        } for h in plan.hedges]
        columns = [{"id": c, "name": n} for c, n in (
            ("side", "Hedge"), ("lots", "Lots"), ("structure", "Structure"), ("legs", "Legs"), ("entry", "At"),
            ("ratio", "Ratio (VaR)"), ("minvar", "Ratio (min-var)"), ("corr", "Corr"), ("var", "VaR change"), ("note", "Note"),
        )]
        body.append(html.Div("Hedge alternatives, best first (pick one)", style={**_LABEL, "marginTop": "8px"}))
        body.append(_table(f"curve-hedges-{kink['product']}-{kink['family']}-{kink['position']}", columns, rows, page_size=5))
    for warning in plan.warnings:
        body.append(html.Div(f"⚠️ {warning}", style={"color": COLORS["ACCENT_YELLOW"], "fontSize": "13px", "marginTop": "4px"}))
    return dbc.Card(dbc.CardBody(body), style={**_CARD, "borderLeft": f"4px solid {colour}"}, className="mb-2")


def build_plan_cards(kinks: list[dict]) -> html.Div:
    """A card per current kink (strongest first): entry, lots, stop, target, time stop and hedge options."""
    if not kinks:
        return html.Div("No kinks at the current thresholds, so no trade plans.", style={**_MUTED, "padding": "12px"})
    cards = [_plan_card(k) for k in kinks[:PLAN_CARD_LIMIT]]
    if len(kinks) > PLAN_CARD_LIMIT:
        cards.append(html.Div(f"Showing the {PLAN_CARD_LIMIT} strongest of {len(kinks)} kinks; the Kinks tab lists all.", style=_MUTED))
    return html.Div(cards, style={"paddingTop": "10px"})


def build_change_table(view: FamilyView | None) -> html.Div:
    if view is None:
        return html.Div(PLACEHOLDER, style={**_MUTED, "padding": "12px"})
    rows = []
    for p in view.positions:
        kink = not view.suspect and p.result is not None and p.result.n_flags > 0 and p.result.echo_of is None
        rows.append({
            "structure": p.structure.label, "live": _fmt(p.live), "prev": _fmt(p.prev_settle),
            "change": _fmt(p.change, signed=True), "kink": p.result.priority if kink else "",
        })
    columns = [{"id": c, "name": n} for c, n in (
        ("structure", "Structure"), ("live", "Live"), ("prev", "Prev settlement"), ("change", "Change"), ("kink", "Kink"),
    )]
    styles = [
        {"if": {"column_id": "change", "filter_query": '{change} contains "+"'}, "color": COLORS["ACCENT_GREEN"]},
        {"if": {"column_id": "change", "filter_query": '{change} contains "-"'}, "color": COLORS["ACCENT_RED"]},
        *_priority_styles("kink"),
    ]
    return _table("curve-change-table", columns, rows, styles, page_size=20)


def build_quality(flags: list) -> html.Div:
    if not flags:
        return html.Div("✅ No data-quality issues.", style={**_MUTED, "padding": "12px"})
    return html.Div([
        html.Div(
            [html.Span(f"{f.severity} ", style={"color": SEVERITY_COLORS[f.severity], "fontWeight": "bold"}),
             html.Span(f"[{f.product}] ", style=_MUTED), html.Span(f.text)],
            style={"padding": "8px 10px", "marginBottom": "6px", "backgroundColor": COLORS["SIDEBAR_BG"],
                   "borderLeft": f"3px solid {SEVERITY_COLORS[f.severity]}", "borderRadius": "4px"},
        )
        for f in sorted(flags, key=lambda f: ("BUFFER", "WARNING", "INFO").index(f.severity))
    ])


def build_log_table(events: list[dict]) -> html.Div:
    if not events:
        return html.Div("No kinks logged yet.", style={**_MUTED, "padding": "12px"})
    now = datetime.now(timezone.utc)
    rows = []
    for e in events:
        first = datetime.fromisoformat(e["first_seen"])
        end = datetime.fromisoformat(e["cleared_at"] or e["last_seen"]) if (e["cleared_at"] or e["last_seen"]) else now
        minutes = max(0, int((end - first).total_seconds() // 60))
        z = e["detail"].get("z", {})
        rows.append({
            "first": first.strftime("%Y-%m-%d %H:%M"), "status": "active" if e["cleared_at"] is None else "cleared",
            "duration": f"{minutes // 60}h {minutes % 60:02d}m", "product": e["product"],
            "family": FAMILY_LABELS.get(e["family"], e["family"]), "structure": e["label"], "direction": e["direction"],
            "trade": trade_side(e["direction"]), "priority": e["peak_priority"], "score": round(e["peak_score"], 1),
            "methods": " · ".join(f"{METHOD_LABELS[m]} {z[m]:+.1f}" for m in METHODS if z.get(m) is not None),
            "seasonal": "normal" if e["seasonal_normal"] else "—",
        })
    columns = [{"id": c, "name": n} for c, n in (
        ("first", "First seen (UTC)"), ("status", "Status"), ("duration", "Duration"), ("product", "Product"),
        ("family", "Curve"), ("structure", "Structure"), ("direction", "vs curve"), ("trade", "Trade"), ("priority", "Peak priority"),
        ("score", "Peak score"), ("methods", "z at peak"), ("seasonal", "Seasonal"),
    )]
    return _table("curve-log-table", columns, rows, _priority_styles("priority"), page_size=20)


def build_backtest_table(result) -> html.Div:
    if result.error:
        return html.Div(f"⚠️ {result.error}", style={"color": COLORS["ACCENT_YELLOW"], "padding": "12px"})
    rows = [
        {"group": r["group"], "horizon": f"{r['horizon']}d", "events": r["events"],
         "reverted": f"{r['reverted_pct']:.0f}%", "narrower": f"{r['narrower_pct']:.0f}%",
         "avg_residual": f"{r['avg_residual']:.3f}", "avg_reduction": f"{r['avg_reduction']:+.3f}"}
        for r in result.rows
    ]
    columns = [{"id": c, "name": n} for c, n in (
        ("group", "Signal"), ("horizon", "Follow-up"), ("events", "Kinks"), ("reverted", "Residual halved"),
        ("narrower", "Residual smaller"), ("avg_residual", "Avg residual (pts)"), ("avg_reduction", "Avg reduction (pts)"),
    )]
    return html.Div([
        html.Div(f"{result.days_tested} days tested for {PRODUCTS[result.product].name} {FAMILY_LABELS[result.family].lower()} curves "
                 "(each day fitted only on earlier history; seasonality not applied).", style={**_MUTED, "marginBottom": "8px"}),
        _table("curve-backtest-table", columns, rows, page_size=20),
    ])


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _dropdown(component_id: str, options, value, clearable=False, placeholder=None) -> dcc.Dropdown:
    return dcc.Dropdown(id=component_id, options=options, value=value, clearable=clearable, placeholder=placeholder)


def _param_input(name: str, label: str, hint: str, step, value) -> dbc.Col:
    return dbc.Col(
        [html.Div(label, style=_LABEL), dbc.Input(id=f"curve-param-{name}", type="number", value=value, step=step),
         html.Div(hint, style={**_MUTED, "fontSize": "11px", "marginTop": "2px"})],
        md=4, className="mb-3",
    )


def _thresholds_tab(params: CurveParams) -> html.Div:
    values = params.__dict__
    return html.Div([
        html.Div("Thresholds apply to every curve and are used by the next compute (within 10 seconds).", style={**_MUTED, "margin": "8px 0 12px"}),
        dbc.Row([_param_input(n, label, hint, step, values[n]) for n, label, hint, step in PARAM_FIELDS]),
        dbc.Row([
            dbc.Col([html.Div("Alert from priority", style=_LABEL),
                     _dropdown("curve-param-alert_min_priority", [{"label": p, "value": p} for p in PRIORITIES], params.alert_min_priority)], md=4),
            dbc.Col(dbc.Switch(id="curve-param-alerts_enabled", label="Send alerts", value=params.alerts_enabled, className="mt-4"), md=4),
        ], className="mb-3"),
        html.Div("Send alerts for", style=_LABEL),
        html.Div(
            "Kinks are still shown on the page and logged for every curve; this only chooses which ones send an alert.",
            style={**_MUTED, "marginBottom": "8px"},
        ),
        *[
            dbc.Row([
                dbc.Col(html.Div(spec.name, style={"color": COLORS["TEXT_PRIMARY"], "paddingTop": "4px"}), md=3),
                dbc.Col(
                    dbc.Checklist(
                        id=f"curve-alert-{code}",
                        options=[{"label": FAMILY_LABELS[f], "value": f} for f in FAMILIES],
                        value=[f for f in FAMILIES if f"{code}:{f}" in params.alert_structures],
                        inline=True,
                    ),
                    md=9,
                ),
            ], className="mb-1")
            for code, spec in PRODUCTS.items()
        ],
        dbc.Button("Save thresholds", id="curve-params-save", color="primary", className="mt-3"),
        html.Div(id="curve-params-message", style={**_MUTED, "marginTop": "10px", "minHeight": "20px"}),
    ])


def _backtest_tab() -> html.Div:
    return html.Div([
        html.Div("Replays the detector over past days (no look-ahead) and checks whether flagged kinks closed. "
                 "Uses the current thresholds.", style={**_MUTED, "margin": "8px 0 12px"}),
        dbc.Row([
            dbc.Col([html.Div("Product", style=_LABEL), _dropdown("curve-bt-product", [{"label": p.name, "value": c} for c, p in PRODUCTS.items()], "CL")], md=3),
            dbc.Col([html.Div("Curve", style=_LABEL), _dropdown("curve-bt-family", [{"label": FAMILY_LABELS[f], "value": f} for f in FAMILIES], OUTRIGHT)], md=3),
            dbc.Col([html.Div("Days to test", style=_LABEL), dbc.Input(id="curve-bt-days", type="number", value=250, min=30, max=1000, step=10)], md=3),
            dbc.Col(dbc.Button("Run backtest", id="curve-bt-run", color="primary", className="mt-4"), md=3),
        ], className="mb-3"),
        dcc.Loading(html.Div(id="curve-bt-output"), type="default"),
    ])


def trade_analyzer_layout() -> html.Div:
    """The page: curve controls and chart on top, detail tabs below."""
    params = load_params(container.repository) if container.repository else CurveParams()
    product_options = [{"label": p.name, "value": c} for c, p in PRODUCTS.items()]
    return html.Div([
        dcc.Interval(id="curve-refresh", interval=REFRESH_MS, n_intervals=0),
        html.H3("🎯 Curve Kinks", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "4px"}),
        html.Div(id="curve-status", style={**_MUTED, "marginBottom": "12px", "minHeight": "20px"}),
        dbc.Row([
            dbc.Col([html.Div("Product", style=_LABEL), _dropdown("curve-product", product_options, "CL")], md=3),
            dbc.Col([html.Div("Curve", style=_LABEL), _dropdown("curve-family", [{"label": FAMILY_LABELS[f], "value": f} for f in FAMILIES], OUTRIGHT)], md=3),
            dbc.Col([html.Div("Compare with snapshot", style=_LABEL), _dropdown("curve-snapshot", [], None, clearable=True, placeholder="None")], md=3),
        ], className="mb-3"),
        dbc.Card(dbc.CardBody([
            dcc.Graph(id="curve-graph", figure=empty_figure(PLACEHOLDER), config={"displayModeBar": False}, style={"height": "460px"}),
            dcc.Graph(id="curve-strength", figure=empty_figure(PLACEHOLDER), config={"displayModeBar": False}, style={"height": "230px"}),
        ]), style=_CARD, className="mb-3"),
        dbc.Tabs([
            dbc.Tab(html.Div(id="curve-kinks-wrap"), label="Kinks", tab_id="kinks"),
            dbc.Tab(html.Div(id="curve-change-wrap"), label="Change vs settlement", tab_id="change"),
            dbc.Tab(html.Div(id="curve-plans-wrap"), label="Trade plans", tab_id="plans"),
            dbc.Tab(html.Div(id="curve-quality-wrap"), label="Data quality", tab_id="quality"),
            dbc.Tab(html.Div(id="curve-log-wrap"), label="History log", tab_id="log"),
            dbc.Tab(_backtest_tab(), label="Backtest", tab_id="backtest"),
            dbc.Tab(_thresholds_tab(params), label="Thresholds", tab_id="thresholds"),
        ], id="curve-tabs", active_tab="kinks"),
    ])
