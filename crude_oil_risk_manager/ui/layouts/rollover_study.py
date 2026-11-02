"""Trigger study section of the Strategy Rollover page: why a line moves one way around an expiry in some months and not in others.

It studies the product, curve and generic chosen on the page. Pick a window (from X to Y trading days from the front expiry): the
section shows what each earlier expiry's line did over it, which readings on the entry day line up with those moves, what kind of window
it is (trend / mixed regimes / no mechanism found), a rule built on the best trigger and how it scores leave-one-expiry-out, the contract
switches that were not counted, a scan of every window, and the forward test of saved rules. The numbers come from
core.rollover_triggers; this module only lays them out. All colours come from COLORS.
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html

from core.curve_calendar import product_spec
from core.rollover_triggers import (
    CASE_MIXED, CASE_NONE, CASE_TOO_FEW, CASE_TREND, DEFAULT_FROM, DEFAULT_LINES, DEFAULT_LOOKBACK, DEFAULT_TO, DONE, EXTERNAL_TRIGGERS,
    MAX_LINES, MAX_LOOKBACK, MAX_X, MIN_LOOKBACK, MIN_X, OPEN, RolloverStudy, RuleCheck, ScanResult, direction_word,
)
from core.trigger_stats import CONSTANT, NO_EVIDENCE, NO_TREND, POSSIBLE, STRONG, TOO_FEW, TREND, WEAK
from ui.layouts.curve_tab import _CARD, _LABEL, _MUTED, _layout, _table, empty_figure
from ui.layouts.shell import COLORS

PLACEHOLDER = "Choose a product above: the study uses the product, curve and generic selected on this page"
UP_COLOR, DOWN_COLOR = COLORS["ACCENT_GREEN"], COLORS["ACCENT_RED"]
CASE_TITLES = {CASE_TREND: "TREND", CASE_MIXED: "MIXED REGIMES", CASE_NONE: "NO MECHANISM FOUND", CASE_TOO_FEW: "TOO FEW EXPIRIES"}
CASE_COLORS = {CASE_TREND: COLORS["ACCENT_GREEN"], CASE_MIXED: COLORS["ACCENT_YELLOW"], CASE_NONE: COLORS["TEXT_SECONDARY"], CASE_TOO_FEW: COLORS["ACCENT_RED"]}
TRIGGER_VERDICT_LABELS = {STRONG: "strong", POSSIBLE: "possible", NO_EVIDENCE: "no evidence", CONSTANT: "does not vary", TOO_FEW: "too few expiries"}

LINE_COLUMNS = [
    {"name": "Expiry", "id": "label"}, {"name": "Front expiry", "id": "anchor"}, {"name": "Entry day", "id": "entry"},
    {"name": "Exit day", "id": "exit"}, {"name": "At entry", "id": "start", "type": "numeric"}, {"name": "At exit", "id": "end", "type": "numeric"},
    {"name": "Move", "id": "move", "type": "numeric"}, {"name": "Switch not counted", "id": "skipped", "type": "numeric"},
    {"name": "Usual direction?", "id": "usual"},
]
BOX_COLUMNS = [
    {"name": "From CL", "id": "cl", "type": "numeric"}, {"name": "From Brent (BRN)", "id": "brent", "type": "numeric"},
    {"name": "Unexplained", "id": "residual", "type": "numeric"},
]
BOARD_COLUMNS = [
    {"name": "Trigger", "id": "trigger"}, {"name": "Group", "id": "group"}, {"name": "Expiries", "id": "n", "type": "numeric"},
    {"name": "Rank corr.", "id": "rho", "type": "numeric"}, {"name": "p (permutation)", "id": "p", "type": "numeric"},
    {"name": "q (corrected)", "id": "q", "type": "numeric"}, {"name": "Holds leaving any expiry out", "id": "stable"},
    {"name": "Low third: mean move", "id": "low", "type": "numeric"}, {"name": "Middle", "id": "mid", "type": "numeric"},
    {"name": "High third: mean move", "id": "high", "type": "numeric"}, {"name": "Signs differ?", "id": "flips"},
    {"name": "Verdict", "id": "verdict"}, {"name": "Now", "id": "now", "type": "numeric"}, {"name": "Why it might matter", "id": "why"},
]
FORWARD_COLUMNS = [
    {"name": "Rule", "id": "rule_label"}, {"name": "Expiry", "id": "expiry"}, {"name": "Entry day", "id": "entry"},
    {"name": "Reading at entry", "id": "reading", "type": "numeric"}, {"name": "Expected", "id": "expect"}, {"name": "Status", "id": "status"},
    {"name": "Move (so far)", "id": "move", "type": "numeric"}, {"name": "Result", "id": "result"}, {"name": "A real forward test?", "id": "forward"},
]
SCAN_COLUMNS = [
    {"name": "From", "id": "x_from", "type": "numeric"}, {"name": "To", "id": "x_to", "type": "numeric"}, {"name": "Length", "id": "length", "type": "numeric"},
    {"name": "Expiries", "id": "n", "type": "numeric"}, {"name": "Mean move", "id": "mean", "type": "numeric"}, {"name": "Agreeing", "id": "hit"},
    {"name": "Direction", "id": "direction"}, {"name": "t", "id": "t", "type": "numeric"},
]

SIGN_STYLES = [
    {"if": {"column_id": "move", "filter_query": "{move} > 0"}, "color": UP_COLOR},
    {"if": {"column_id": "move", "filter_query": "{move} < 0"}, "color": DOWN_COLOR},
]
BOARD_STYLES = [
    {"if": {"column_id": "verdict", "filter_query": f'{{verdict}} = "{TRIGGER_VERDICT_LABELS[STRONG]}"'}, "color": COLORS["ACCENT_GREEN"], "fontWeight": "bold"},
    {"if": {"column_id": "verdict", "filter_query": f'{{verdict}} = "{TRIGGER_VERDICT_LABELS[POSSIBLE]}"'}, "color": COLORS["ACCENT_YELLOW"], "fontWeight": "bold"},
    {"if": {"column_id": "verdict", "filter_query": f'{{verdict}} != "{TRIGGER_VERDICT_LABELS[STRONG]}" && {{verdict}} != "{TRIGGER_VERDICT_LABELS[POSSIBLE]}"'}, "color": COLORS["TEXT_SECONDARY"]},
]
FORWARD_STYLES = [
    {"if": {"column_id": "result", "filter_query": '{result} = "HIT"'}, "color": UP_COLOR, "fontWeight": "bold"},
    {"if": {"column_id": "result", "filter_query": '{result} = "MISS"'}, "color": DOWN_COLOR, "fontWeight": "bold"},
]


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------


def _num(value, digits: int = 3):
    return None if value is None or value != value else round(float(value), digits)


def _signed(value, digits: int = 3) -> str:
    return "—" if value is None or value != value else f"{value:+.{digits}f}"


def _t(value: float | None) -> str:
    return "—" if value is None else f"{min(abs(value), 99.0) * (1 if value > 0 else -1):.2f}"


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------


def line_columns(study: RolloverStudy | None) -> list[dict]:
    return LINE_COLUMNS + (BOX_COLUMNS if study is not None and product_spec(study.product).is_box else [])


def line_rows(study: RolloverStudy | None) -> list[dict]:
    """The finished expiries, then the one being drawn (or about to start); newest first."""
    if study is None:
        return []
    usual = study.stats.direction if study.stats else 0
    rows = []
    for obs in [*study.pending, *study.finished]:
        move = obs.move if obs.status == DONE else obs.so_far if obs.status == OPEN else None
        agrees = ""
        if obs.status == DONE and usual and obs.move != 0:
            agrees = "with" if (obs.move > 0) == (usual > 0) else "against"
        status = "" if obs.status == DONE else {"open": "running", "ahead": "not started", "missing": "no data"}[obs.status]
        parts = obs.attribution or {}
        rows.append({
            "label": f"{obs.label} (now)" if obs in study.pending else obs.label, "anchor": obs.anchor.isoformat(),
            "entry": obs.entry_date.isoformat() if obs.entry_date else "", "exit": obs.exit_date.isoformat() if obs.exit_date else status,
            "start": _num(obs.start_price), "end": _num(obs.end_price), "move": _num(move),
            "skipped": _num(obs.raw_move - obs.move) if obs.raw_move is not None and obs.skipped else None,
            "usual": agrees or (status if obs in study.pending else ""),
            "cl": _num(parts.get("CL")), "brent": _num(parts.get("Brent")), "residual": _num(parts.get("residual")),
        })
    return rows


def board_rows(study: RolloverStudy | None) -> list[dict]:
    if study is None:
        return []
    rows = []
    for item in study.screens:
        result, split = item.result, item.result.split
        rows.append({
            "trigger": item.spec.label, "group": item.spec.group, "n": result.n, "rho": _num(result.rho, 2), "p": _num(result.p_value, 3),
            "q": _num(result.q_value, 2), "stable": "" if result.rho is None else ("yes" if result.loo_stable else "no"),
            "low": _num(split.low_mean), "mid": _num(split.mid_mean), "high": _num(split.high_mean),
            "flips": "yes" if split.flips else "", "verdict": TRIGGER_VERDICT_LABELS[result.verdict], "now": _num(item.current),
            "why": item.spec.why, "key": item.spec.key,
        })
    return rows


def values_view(study: RolloverStudy | None) -> tuple[list[dict], list[dict]]:
    """(columns, rows) of every trigger's reading on every expiry: triggers down, expiries across, oldest to newest."""
    if study is None:
        return [], []
    observed = [*reversed([o for o in study.obs if o.status in (DONE, OPEN)])]
    columns = [{"name": "Trigger", "id": "trigger"}] + [{"name": o.label, "id": f"e{i}", "type": "numeric"} for i, o in enumerate(observed)]
    moves = {"trigger": "MOVE OVER THE WINDOW"}
    rows = []
    for item in study.screens:
        row = {"trigger": item.spec.label}
        for i, obs in enumerate(observed):
            row[f"e{i}"] = _num(obs.triggers.get(item.spec.key))
        rows.append(row)
    for i, obs in enumerate(observed):
        moves[f"e{i}"] = _num(obs.move if obs.status == DONE else obs.so_far)
    return columns, [moves, *rows]


def forward_view(rows: list[dict], rules: list[dict]) -> tuple[list[dict], list[dict]]:
    """(table rows, dropdown options to delete a rule)."""
    table = [{**r, "reading": _num(r.get("reading")), "move": _num(r.get("move")), "entry": r.get("entry") or ""} for r in rows]
    options = [{"label": f"{r['label']}: {r['trigger_label']}, {r['x_from']}→{r['x_to']} days (saved {r['created_on']})", "value": r["id"]} for r in rules]
    return table, options


def scan_view(result: ScanResult | None) -> tuple[list[dict], str]:
    if result is None:
        return [], ""
    rows = [
        {"id": i, "x_from": r.x_from, "x_to": r.x_to, "length": r.x_to - r.x_from, "n": r.n, "mean": _num(r.mean), "hit": f"{r.hit_rate:.0%}",
         "direction": direction_word(r.direction), "t": _num(None if r.t_stat is None else min(abs(r.t_stat), 99.0) * (1 if r.t_stat > 0 else -1), 2)}
        for i, r in enumerate(result.rows)
    ]
    if result.note:
        return rows, result.note
    note = (
        f"{result.windows} windows scanned: {result.trends} look like a trend and {result.weak} weak. If each expiry's whole line had a "
        f"random sign instead, about {result.chance_trends:.0f} of them would still look like a trend. "
        + ("That is about what chance alone produces, so a window here is a lead to test with triggers, not a finding. "
           if result.trends <= 1.5 * (result.chance_trends or 0) + 2 else
           "The count is well above chance, but the windows overlap (they are not independent). ")
        + "Click a row to study that window."
    )
    return rows, note


# ----------------------------------------------------------------------
# Text blocks
# ----------------------------------------------------------------------


def title_text(study: RolloverStudy | None, message: str = PLACEHOLDER) -> str:
    if study is None:
        return message
    return (
        f"{study.title} · from {study.x_from} to {study.x_to} trading days from the front expiry ({study.window_days} days) · "
        f"triggers read on the entry day, lookback {study.lookback} days"
    )


def _badge(text: str, color: str) -> html.Span:
    return html.Span(text, style={
        "backgroundColor": color, "color": COLORS["DARK_BG"], "fontWeight": "bold", "padding": "3px 10px", "borderRadius": "4px", "marginRight": "10px",
    })


def verdict_block(study: RolloverStudy | None) -> html.Div:
    if study is None:
        return html.Div()
    stats = study.stats
    head = html.Div([_badge(CASE_TITLES[study.case], CASE_COLORS[study.case])])
    if stats is not None:
        head.children.append(html.Span(
            f"{stats.n} earlier expiries: {stats.up} up, {stats.down} down · mean {_signed(stats.mean)} · median {_signed(stats.median)} · "
            f"{stats.hit_rate:.0%} moved {direction_word(stats.direction)} · t = {_t(stats.t_stat)} · sign test p = {stats.sign_p:.3f} · "
            f"window verdict: {stats.verdict}", style={"color": COLORS["TEXT_PRIMARY"]},
        ))
    children = [head, html.Div(study.case_text, style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "8px"})]
    current = study.current
    if current is not None:
        if current.status == OPEN:
            now = f"The {current.label} line is running: entered {current.entry_date.isoformat()}, {_signed(current.so_far)} so far."
        else:
            now = f"The {current.label} line has not reached the entry day (day {study.x_from}); the triggers are read on that day."
        rule = study.rule
        if rule is not None and rule.now_direction is not None:
            now += f" The rule below expects {direction_word(rule.now_direction) if rule.now_direction else 'no call'}."
        children.append(html.Div(now, style={**_MUTED, "marginTop": "6px"}))
    return html.Div(children)


def rule_block(study: RolloverStudy | None) -> html.Div:
    if study is None:
        return html.Div()
    rule: RuleCheck | None = study.rule
    if rule is None:
        return html.Div("No trigger passed the screening, so no rule is proposed. A rule built on a trigger that did not hold up would only fit the past.", style=_MUTED)
    fit, score = rule.fit, rule.score
    side = lambda sign: direction_word(sign) if sign else "no call"  # noqa: E731
    lines = [
        html.Div([html.B("Best candidate: "), rule.spec.label]),
        html.Div(f"Rule: above {fit.cut:.3f} expect {side(fit.high_dir)}; at or below it expect {side(fit.low_dir)}.", style={"marginTop": "4px"}),
    ]
    if score is not None:
        mean = "—" if score.mean_pnl is None else _signed(score.mean_pnl)
        base = "—" if score.baseline_mean_pnl is None else _signed(score.baseline_mean_pnl)
        lines.append(html.Div(
            f"Tested leave-one-expiry-out (each expiry judged by a rule learned from the OTHER expiries): it had a view in {score.trades} of {score.years} "
            f"expiries and was right in {score.hits}, averaging {mean} per trade. Always taking the usual direction was right in "
            f"{score.baseline_hits} of {score.years}, averaging {base}.", style={"marginTop": "4px"}))
        if score.thin:
            lines.append(html.Div("⚠️ Few trades: this is too thin to trust. Forward-test it before using it.", style={"color": COLORS["ACCENT_YELLOW"], "marginTop": "4px"}))
    if rule.now_value is not None:
        lines.append(html.Div(f"Now: the trigger reads {rule.now_value:.3f}, so the rule expects {side(rule.now_direction) if rule.now_direction is not None else '—'}.", style={"marginTop": "4px"}))
    return html.Div(lines, style={"color": COLORS["TEXT_PRIMARY"]})


def _table_of(header: list[str], rows: list[list[str]]) -> dbc.Table:
    return dbc.Table(
        [html.Thead(html.Tr([html.Th(c) for c in header])), html.Tbody([html.Tr([html.Td(c) for c in row]) for row in rows])],
        size="sm", style={"color": COLORS["TEXT_PRIMARY"], "maxWidth": "760px"},
    )


def roll_block(study: RolloverStudy | None) -> html.Div:
    explanation = (
        "When a line changes what it prices, the price jumps by about the spread between the two, and that jump is a switch, not a market move. "
        "The CL-BZ line is the front box: it changes pair on the first trading day after the CL expiry and again after the Brent expiry, so a "
        "naive move across them would show CL|BZ dropping after every expiry. Those days are left out of every move, and the first trading day "
        "after each CL and Brent expiry is left out of the market-movement triggers too. A generic line follows one fixed contract through its "
        "expiry and never switches."
    )
    children: list = [html.Div(explanation, style=_MUTED)]
    if study is not None and study.skips:
        children += [
            html.Div(f"{len(study.skips)} switch days were left out of the lines' moves (latest 12 shown):", style={**_MUTED, "margin": "8px 0 4px"}),
            _table_of(["Day", "Line", "What was priced", "Jump not counted"], [
                [s.day.isoformat(), s.label, f"{s.before} → {s.after}", _signed(s.jump, 2)] for s in study.skips[:12]
            ]),
        ]
    elif study is not None and product_spec(study.product).is_box is False:
        children.append(html.Div("No switch days: this line follows one contract.", style={**_MUTED, "marginTop": "8px"}))
    if study is not None and study.rolls:
        children += [
            html.Div(f"{len(study.rolls)} front-month roll days fell inside the trigger lookbacks and were not counted (latest 12 shown):", style={**_MUTED, "margin": "8px 0 4px"}),
            _table_of(["Day not counted", "Leg", "Contract change", "Jump if it had been counted"], [
                [e.day.isoformat(), e.product, f"{e.from_contract} → {e.to_contract}", _signed(e.jump, 2)] for e in study.rolls[:12]
            ]),
        ]
    return html.Div(children)


def notes_block(study: RolloverStudy | None) -> html.Div:
    notes = [] if study is None else study.notes
    external = html.Details([
        html.Summary("Triggers worth testing that the app has no data for", style=_MUTED),
        html.Ul([html.Li([html.B(name + ": "), why]) for name, why in EXTERNAL_TRIGGERS], style=_MUTED),
    ], style={"marginTop": "8px"})
    return html.Div([*[html.Div(f"⚠️ {n}", style={"color": COLORS["ACCENT_YELLOW"], "fontSize": "13px", "marginTop": "3px"}) for n in notes], external])


# ----------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------


def scatter_figure(study: RolloverStudy | None, key: str | None) -> go.Figure:
    """One trigger against the move: each earlier expiry a point, the line being drawn now a dashed line, the rule's cut a dotted one."""
    if study is None or not key:
        return empty_figure("Pick a trigger to plot it against the moves")
    item = next((s for s in study.screens if s.spec.key == key), None)
    if item is None:
        return empty_figure("This trigger is not part of the study")
    points = [(o.triggers.get(key), o.move, o.label) for o in study.finished if o.triggers.get(key) is not None]
    if len(points) < 3:
        return empty_figure(f"Too few expiries have a reading of '{item.spec.label}' to plot it")
    xs, ys, names = zip(*points)
    figure = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=list(names), textposition="top center", textfont={"size": 9},
        marker={"size": 11, "color": [UP_COLOR if y > 0 else DOWN_COLOR for y in ys], "line": {"width": 1, "color": COLORS["TEXT_PRIMARY"]}},
        hovertemplate="%{text}<br>trigger %{x:.3f}<br>move %{y:+.3f}<extra></extra>", showlegend=False,
    ))
    figure.add_hline(y=0, line={"color": COLORS["TEXT_SECONDARY"], "width": 1})
    if item.current is not None:
        figure.add_vline(x=item.current, line={"color": "#4da3ff", "dash": "dash", "width": 2}, annotation_text="now", annotation_font_color="#4da3ff")
    if study.rule is not None and study.rule.spec.key == key:
        figure.add_vline(x=study.rule.fit.cut, line={"color": COLORS["TEXT_SECONDARY"], "dash": "dot", "width": 1}, annotation_text="rule cut")
    result = item.result
    detail = "" if result.rho is None else f" — rank corr. {result.rho:+.2f}, p = {result.p_value:.3f}, q = {result.q_value:.2f}"
    figure.update_layout(**_layout(
        title={"text": f"{item.spec.label}{detail}", "font": {"size": 14}},
        xaxis={"title": "Trigger reading on the entry day", "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis={"title": f"Move from day {study.x_from} to day {study.x_to}", "gridcolor": COLORS["BORDER_COLOR"], "zeroline": False},
        margin={"l": 70, "r": 30, "t": 50, "b": 60}, hovermode="closest", uirevision=f"{study.product}|{study.title}|{key}",
    ))
    return figure


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _number(component_id: str, value: int, low: int, high: int) -> dcc.Input:
    return dcc.Input(id=component_id, type="number", min=low, max=high, step=1, value=value, debounce=True, className="form-control")


def _heading(text: str) -> html.Div:
    return html.Div(text, style={**_LABEL, "fontSize": "14px", "margin": "18px 0 6px"})


def study_section() -> html.Div:
    """The Trigger study card; its parts are filled by ui.callbacks.rollover_study_callbacks."""
    return html.Div([
        dbc.Card(dbc.CardBody(className="study-tables", children=[
            html.H5("🔬 Trigger study", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "4px"}),
            html.Div(
                "Why does a line move one way around an expiry in some months and not in others? The study uses the product, curve and generic "
                "chosen above. Every earlier expiry gives one move over the window; the readings on the entry day (triggers) are lined up against "
                "those moves to see which ones separate the expiries. Treat a result as a lead to forward-test, not as proof.",
                style={**_MUTED, "marginBottom": "12px"},
            ),
            dbc.Row([
                dbc.Col([html.Div("From (days from the front expiry)", style=_LABEL), _number("rollover-study-from", DEFAULT_FROM, MIN_X, MAX_X - 1)], md=3),
                dbc.Col([html.Div("To (days from the front expiry)", style=_LABEL), _number("rollover-study-to", DEFAULT_TO, MIN_X + 1, MAX_X)], md=3),
                dbc.Col([html.Div(f"Lookback for momentum, OI, volatility ({MIN_LOOKBACK}-{MAX_LOOKBACK} days)", style=_LABEL),
                         _number("rollover-study-lookback", DEFAULT_LOOKBACK, MIN_LOOKBACK, MAX_LOOKBACK)], md=2),
                dbc.Col([html.Div(f"Expiries studied (1-{MAX_LINES})", style=_LABEL), _number("rollover-study-lines", DEFAULT_LINES, 1, MAX_LINES)], md=2),
                dbc.Col(dbc.Button("Scan all windows", id="rollover-scan-run", color="secondary", className="mt-4"), md=2),
            ], className="mb-3"),
            html.Div(PLACEHOLDER, id="rollover-study-title", style={"color": COLORS["TEXT_PRIMARY"], "fontWeight": "bold", "marginBottom": "8px"}),
            html.Div(id="rollover-study-verdict"),
            _heading("Expiry by expiry"),
            html.Div(_table("rollover-study-lines-table", LINE_COLUMNS, [], SIGN_STYLES, page_size=15), style={"overflowX": "auto"}),
            _heading("Triggers: which readings line up with the moves?"),
            html.Div(_table("rollover-study-board", BOARD_COLUMNS, [], BOARD_STYLES, page_size=30), style={"overflowX": "auto"}),
            html.Div(
                "Rank corr.: +1 means the higher the reading the better the move, -1 the opposite. p: how often a random pairing of the expiries would "
                "correlate this strongly. q: the same, corrected for how many triggers were tried. 'Holds leaving any expiry out': the sign survives "
                "removing each single expiry. Low/High third: average move over the expiries with the lowest and highest readings.",
                style={**_MUTED, "fontSize": "12px", "marginTop": "6px"},
            ),
            dbc.Row([
                dbc.Col([html.Div("Plot trigger", style=_LABEL), dcc.Dropdown(id="rollover-study-trigger", options=[], value=None, clearable=False)], md=6),
            ], className="mt-3"),
            dcc.Graph(id="rollover-study-scatter", figure=empty_figure("Pick a trigger to plot it against the moves"), config={"displayModeBar": False}, style={"height": "380px"}),
            html.Details([
                html.Summary("Trigger readings, expiry by expiry", style=_MUTED),
                html.Div(_table("rollover-study-values", [], [], [], page_size=40), style={"overflowX": "auto"}),
            ], className="mt-2"),
            _heading("A rule from the best trigger"),
            html.Div(id="rollover-study-rule"),
            dbc.Button("Save this rule to forward-test", id="rollover-study-save-rule", color="primary", size="sm", className="mt-2"),
            html.Span(id="rollover-study-rule-feedback", style={**_MUTED, "marginLeft": "12px"}),
            _heading("Contract switches (roll days)"),
            html.Div(id="rollover-study-rolls"),
            html.Div(id="rollover-study-notes"),
            _heading("Scan: where do the expiries agree?"),
            dcc.Loading(html.Div([
                html.Div(id="rollover-scan-note", style={**_MUTED, "marginBottom": "6px"}),
                html.Div(_table("rollover-scan-table", SCAN_COLUMNS, [], [], page_size=15), style={"overflowX": "auto"}),
            ])),
            _heading("Forward test of saved rules"),
            html.Div(
                "A saved rule is frozen: it applies to the same product, curve, generic and window on every later expiry. A row only counts as a "
                "forward test when its entry day came after the rule was saved.", style={**_MUTED, "marginBottom": "6px"},
            ),
            html.Div(_table("rollover-forward-table", FORWARD_COLUMNS, [], FORWARD_STYLES, page_size=15), style={"overflowX": "auto"}),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="rollover-forward-pick", options=[], value=None, placeholder="Choose a saved rule to delete"), md=7),
                dbc.Col(dbc.Button("Delete rule", id="rollover-forward-delete", color="danger", size="sm"), md=2),
                dbc.Col(html.Span(id="rollover-forward-feedback", style=_MUTED), md=3),
            ], className="mt-2"),
        ]), style=_CARD, className="mb-3"),
    ])
