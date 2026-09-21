"""New Structure builder: the modal skeleton plus the small render helpers its callbacks use.

Steps 1-4 all stay in the DOM (hidden with display:none) so every field keeps its
value while the user moves back and forth; ui/callbacks/structure_builder_callbacks.py
toggles their visibility.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from core.structure_utils import STRUCTURE_TEMPLATES
from ui.layouts.shell import COLORS

STEPS = ["Template", "Legs", "Details", "Review"]
STEP_COUNT = len(STEPS)
HIDDEN = {"display": "none"}
SHOWN: dict = {}

_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_HEADING = {"color": COLORS["TEXT_PRIMARY"], "marginBottom": "12px"}
_CELL = {"padding": "6px 10px", "borderBottom": f"1px solid {COLORS['BORDER_COLOR']}"}

CLASSIFICATION_DISPLAY = {
    "highly_correlated": ("⚠️ Highly Correlated", COLORS["ACCENT_YELLOW"]),
    "negatively_correlated": ("🔵 Hedge (negative)", COLORS["TEXT_PRIMARY"]),
    "uncorrelated": ("✅ New Exposure", COLORS["ACCENT_GREEN"]),
    "insufficient_data": ("Insufficient data", COLORS["TEXT_SECONDARY"]),
}

EXPOSURE_HINT = "Enter leg symbols to see the net outright equivalent (per 1 lot of the structure)."
CORRELATION_HINT = "Click 'Refresh Correlation' to compare against your active structures."


# ----------------------------------------------------------------------
# Render helpers
# ----------------------------------------------------------------------


def data_table(headers: list[str], rows: list[list], row_styles: list[dict] | None = None) -> html.Table:
    """Small dark table; cells may be strings or components."""
    head = html.Thead(
        html.Tr([html.Th(h, style={**_CELL, **_MUTED, "textAlign": "left"}) for h in headers])
    )
    body = html.Tbody(
        [
            html.Tr(
                [html.Td(cell, style=_CELL) for cell in row],
                style=(row_styles[i] if row_styles else {}),
            )
            for i, row in enumerate(rows)
        ]
    )
    return html.Table([head, body], style={"width": "100%", "color": COLORS["TEXT_PRIMARY"], "fontSize": "14px"})


def render_messages(errors: list[str], warnings: list[str]) -> list:
    """Blocking errors in red, non-blocking warnings in yellow."""
    return [html.Div(f"⛔ {m}", style={"color": COLORS["ACCENT_RED"]}) for m in errors] + [
        html.Div(f"⚠️ {m}", style={"color": COLORS["ACCENT_YELLOW"]}) for m in warnings
    ]


def render_exposure_table(net: dict[str, float], ignored: list[str]):
    """Contract | Direction | Net Lots, or the hint when there is nothing to show."""
    if not net and not ignored:
        return html.Div(EXPOSURE_HINT, style=_MUTED)
    rows = [
        [
            symbol,
            "Long" if lots > 0 else "Short" if lots < 0 else "Flat",
            html.Span(
                f"{lots:+g}",
                style={"color": COLORS["ACCENT_GREEN"] if lots > 0 else COLORS["ACCENT_RED"] if lots < 0 else COLORS["TEXT_PRIMARY"]},
            ),
        ]
        for symbol, lots in sorted(net.items())
    ]
    parts = [data_table(["Contract", "Direction", "Net Lots"], rows)] if rows else []
    if ignored:
        parts.append(html.Div(f"Ignored (incomplete or invalid): {', '.join(ignored)}", style={**_MUTED, "marginTop": "8px"}))
    return html.Div(parts)


def render_correlation_table(rows: list[dict]):
    """Candidate | Existing Structure | Correlation | Classification with colour coding."""
    if not rows:
        return html.Div("No active structures with an exchange-quoted symbol to compare against.", style=_MUTED)
    table_rows, styles = [], []
    for row in rows:
        label, color = CLASSIFICATION_DISPLAY[row["classification"]]
        if row["correlation"] is None:
            correlation = html.Span(f"Insufficient data for {row['candidate']} / {row['existing_symbol']}", style=_MUTED)
            chip = html.Span(label, style={"color": color})
        else:
            correlation = f"{row['correlation']:+.2f}"
            hedge = row["classification"] == "negatively_correlated"
            chip = html.Span(
                label,
                style={
                    "color": color,
                    **({"backgroundColor": COLORS["ACCENT_BLUE"], "padding": "2px 8px", "borderRadius": "10px"} if hedge else {}),
                },
            )
        table_rows.append([row["candidate"], row["existing_structure"], correlation, chip])
        styles.append({"borderLeft": f"3px solid {color}"})
    return data_table(["Candidate", "Existing Structure", "Correlation", "Classification"], table_rows, styles)


def render_step_indicator(step: int, template: str | None) -> dbc.Row:
    """[1.Template] [2.Legs] [3.Details] [4.Review] with the current step highlighted."""
    chips = []
    for number, label in enumerate(STEPS, start=1):
        current, done = number == step, number < step
        text = f"{number}. {label}"
        if number == 1 and template in STRUCTURE_TEMPLATES:
            text = f"{number}. {STRUCTURE_TEMPLATES[template]['label']}"
        chips.append(
            dbc.Col(
                html.Div(
                    ("✓ " if done else "") + text,
                    style={
                        "textAlign": "center",
                        "padding": "8px",
                        "borderRadius": "6px",
                        "fontWeight": "bold" if current else "normal",
                        "backgroundColor": COLORS["ACCENT_BLUE"] if current else "transparent",
                        "color": COLORS["TEXT_PRIMARY"] if current or done else COLORS["TEXT_SECONDARY"],
                        "border": f"1px solid {COLORS['ACCENT_GREEN'] if current else COLORS['BORDER_COLOR']}",
                    },
                )
            )
        )
    return dbc.Row(chips, className="g-2")


def build_leg_row(index: int, symbol, ratio, contract_options: list[str], removable: bool) -> dbc.Row:
    """One leg row: symbol (free text), saved-contract picker, ratio and (Custom only) remove."""
    return dbc.Row(
        [
            dbc.Col(html.Div(f"Leg {index + 1}", style={"color": COLORS["TEXT_PRIMARY"], "paddingTop": "8px"}), width=2),
            dbc.Col(
                [
                    dbc.Input(
                        id={"type": "leg-symbol", "index": index},
                        placeholder="e.g. CLZ26 or CLZ26-F27",
                        value=symbol,
                        debounce=True,
                        className="mb-1",
                    ),
                    dcc.Dropdown(
                        id={"type": "leg-autocomplete", "index": index},
                        options=contract_options,
                        placeholder="Or select saved contract...",
                        className="builder-dropdown",
                    ),
                ],
                width=6,
            ),
            dbc.Col(
                dbc.Input(id={"type": "leg-ratio", "index": index}, type="number", step=1, value=ratio, placeholder="Ratio"),
                width=2,
            ),
            dbc.Col(
                dbc.Button("✕", id={"type": "leg-remove", "index": index}, color="danger", outline=True, size="sm")
                if removable
                else None,
                width=1,
            ),
        ],
        className="mb-3",
    )


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def _template_card(key: str, spec: dict) -> dbc.Col:
    legs = "user-defined" if spec["legs"] is None else str(spec["legs"])
    ratios = "user-defined" if spec["legs"] is None else str(spec["ratios"])
    card = dbc.Card(
        dbc.CardBody(
            [
                html.Div(spec["icon"], style={"fontSize": "40px"}),
                html.H5(spec["label"], style={"color": COLORS["TEXT_PRIMARY"], "marginTop": "8px"}),
                html.Div(spec["description"], style=_MUTED),
                html.Div(f"Legs: {legs}", style={**_MUTED, "marginTop": "8px"}),
                html.Div(f"Ratios: {ratios}", style=_MUTED),
            ],
            style={"textAlign": "center"},
        ),
        style={"backgroundColor": COLORS["CARD_BG"], "height": "100%"},
    )
    # dbc.Card has no n_clicks, so a clickable wrapper carries the id.
    return dbc.Col(
        html.Div(card, id=f"template-card-{key}", n_clicks=0, className="template-card", style={"height": "100%"}),
        md=3,
        className="mb-3",
    )


def _field(label: str, component) -> html.Div:
    return html.Div([html.Label(label, style={**_MUTED, "marginBottom": "4px"}), component], className="mb-3")


def _step_1() -> html.Div:
    return html.Div(
        [
            html.H5("Choose Structure Template", style=_HEADING),
            dbc.Row([_template_card(key, spec) for key, spec in STRUCTURE_TEMPLATES.items()]),
        ],
        id="builder-step-1",
    )


def _step_2() -> html.Div:
    left = [
        html.Div(id="builder-leg-rows"),
        dbc.Button("+ Add Leg", id="btn-add-leg", color="secondary", outline=True, style=HIDDEN),
    ]
    right = [
        html.H6("📊 Net Outright Equivalent", style=_HEADING),
        html.Div(html.Div(EXPOSURE_HINT, style=_MUTED), id="builder-exposure-preview"),
        html.Hr(style={"borderColor": COLORS["BORDER_COLOR"]}),
        html.H6("🔗 Correlation vs Portfolio", style=_HEADING),
        dbc.Button("Refresh Correlation", id="btn-refresh-correlation", color="info", outline=True, size="sm", className="mb-2"),
        dcc.Loading(html.Div(html.Div(CORRELATION_HINT, style=_MUTED), id="builder-correlation-panel"), type="dot"),
    ]
    return html.Div(
        [
            html.H5("Define Structure Legs", style=_HEADING),
            dbc.Row([dbc.Col(left, md=7), dbc.Col(right, md=5)]),
        ],
        id="builder-step-2",
        style=HIDDEN,
    )


def _step_3() -> html.Div:
    return html.Div(
        [
            html.H5("Structure Details", style=_HEADING),
            _field("Structure Name", dbc.Input(id="builder-name", placeholder="e.g. CL Dec26-Jan27 Spread", maxLength=100)),
            dbc.Row(
                [
                    dbc.Col(_field("Multiplier ($/point)", dbc.Input(id="builder-multiplier", type="number", placeholder="e.g. 1000 for CL")), md=4),
                    dbc.Col(_field("Tick Size", dbc.Input(id="builder-tick-size", type="number", placeholder="e.g. 0.01")), md=4),
                    dbc.Col(_field("Tick Value ($)", dbc.Input(id="builder-tick-value", type="number", placeholder="e.g. 10")), md=4),
                ]
            ),
            _field("Notes (optional)", dbc.Textarea(id="builder-notes", placeholder="Optional trade notes", rows=3, maxLength=500)),
        ],
        id="builder-step-3",
        style=HIDDEN,
    )


def _step_4() -> html.Div:
    return html.Div(
        [html.H5("Review Structure", style=_HEADING), html.Div(id="builder-review-body")],
        id="builder-step-4",
        style=HIDDEN,
    )


def new_structure_modal() -> dbc.Modal:
    """The multi-step New Structure builder (Template -> Legs -> Details -> Review)."""
    body = dbc.ModalBody(
        [
            html.Div(render_step_indicator(1, None), id="builder-step-indicator", className="mb-4"),
            _step_1(),
            _step_2(),
            _step_3(),
            _step_4(),
            # Outside the steps so save errors stay visible on the Review step.
            html.Div(id="builder-validation-messages", style={"marginTop": "16px"}),
        ]
    )
    footer = dbc.ModalFooter(
        [
            dbc.Button("Cancel", id="btn-builder-cancel", color="secondary", outline=True, className="me-auto"),
            dbc.Button("← Back", id="btn-builder-back", color="secondary", style=HIDDEN),
            dbc.Button("Next →", id="btn-builder-next", color="primary", disabled=True),
            dbc.Button("💾 Save Structure", id="btn-save-structure", color="success", style=HIDDEN),
        ]
    )
    return dbc.Modal(
        [dbc.ModalHeader(dbc.ModalTitle("New Structure"), close_button=False), body, footer],
        id="modal-new-structure",
        size="xl",
        centered=True,
        scrollable=True,
        backdrop="static",
        keyboard=False,
        is_open=False,
    )


def builder_save_toast() -> dbc.Toast:
    """Confirmation shown after a save (its own toast, so the alert toast container can't wipe it)."""
    return dbc.Toast(
        "",
        id="builder-save-toast",
        header="Structure saved",
        icon="success",
        is_open=False,
        dismissable=True,
        duration=5000,
        style={"position": "fixed", "bottom": "20px", "right": "20px", "width": "360px", "zIndex": 2100},
    )
