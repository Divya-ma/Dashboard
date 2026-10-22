"""Dissolve structures: the button next to '+ New Structure' and the modal behind it (see core.dissolve).

The modal is a static shell; ui/callbacks/dissolve_callbacks.py fills the structure list, the lots and the leg-price
boxes (which depend on what is ticked) and the preview.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from ui.layouts.shell import COLORS

MODE_LEGS = "legs"
MODE_TYPED = "typed"
_MUTED = {"color": COLORS["TEXT_SECONDARY"], "fontSize": "13px"}
_LABEL = {"fontSize": "12px", "color": COLORS["TEXT_SECONDARY"], "textTransform": "uppercase", "marginBottom": "4px"}


def dissolve_button() -> dbc.Button:
    return dbc.Button("🔗 Dissolve", id="btn-dissolve", color="info", outline=True, className="me-2",
                      title="Close several structures against each other without an exit cost")


def dissolve_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Dissolve structures against each other")),
            dbc.ModalBody(
                [
                    html.Div(
                        "For positions you cut with other structures (a d-fly cut with two flies, say): tick all of them and their lots "
                        "are closed together with NO exit cost. Only the entry costs already booked are charged, and they come off "
                        "the gross P&L you give below.",
                        style={**_MUTED, "marginBottom": "12px"},
                    ),
                    html.Div("Structures to dissolve", style=_LABEL),
                    dbc.Checklist(id="dissolve-select", options=[], value=[], style={"maxHeight": "200px", "overflowY": "auto"}),
                    html.Div(
                        [
                            html.Div("Lots to dissolve (partial dissolve)", style=_LABEL),
                            dbc.Input(id="dissolve-lots-all", type="number", min=0, step="any", size="sm", style={"maxWidth": "260px"},
                                      placeholder="Same number of lots for every structure (blank = all lots)"),
                        ],
                        style={"marginTop": "10px"},
                    ),
                    html.Div(id="dissolve-lots-wrap", style={"marginTop": "8px"}),
                    html.Hr(style={"borderColor": COLORS["BORDER_COLOR"]}),
                    html.Div("Gross P&L of the group", style=_LABEL),
                    dbc.RadioItems(
                        id="dissolve-mode", value=MODE_LEGS, inline=True,
                        options=[{"label": "From the leg prices", "value": MODE_LEGS}, {"label": "Type the gross P&L", "value": MODE_TYPED}],
                    ),
                    html.Div(id="dissolve-prices-wrap", style={"marginTop": "8px"}),
                    html.Div(
                        dbc.Input(id="dissolve-gross", type="number", placeholder="Gross P&L in $ (before any transaction cost)"),
                        id="dissolve-gross-wrap", style={"display": "none", "marginTop": "8px"},
                    ),
                    html.Div(dbc.Input(id="dissolve-notes", type="text", maxLength=100, placeholder="Notes (optional)"), style={"marginTop": "10px"}),
                    html.Div(id="dissolve-preview", style={"marginTop": "14px"}),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button("Cancel", id="btn-dissolve-cancel", color="secondary", outline=True),
                    dbc.Button("Dissolve", id="btn-dissolve-confirm", color="info", disabled=True),
                ]
            ),
        ],
        id="modal-dissolve",
        size="lg",
        scrollable=True,
        is_open=False,
    )
