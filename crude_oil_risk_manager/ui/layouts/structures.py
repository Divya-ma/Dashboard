"""Structures tab layout: active structures grid, collapsible closed grid and the detail modal.

ui/callbacks/structures_callbacks.py fills the grids from the repository and the
`store-portfolio-pnl` store. Row actions are custom cell renderers defined in
ui/assets/dashAgGridComponentFunctions.js.
"""

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import html

from ui.layouts.home import GRID_THEME
from ui.layouts.shell import COLORS
from ui.layouts.structure_detail import confirm_edit_modal, confirm_exit_modal, detail_toast

ROW_HEIGHT = 48

_STATUS_OPTIONS = [
    {"label": "All Active", "value": "active"},
    {"label": "Shell Only", "value": "shell"},
    {"label": "Open Only", "value": "open"},
]
_PRODUCT_OPTIONS = [{"label": "All Products", "value": "all"}] + [
    {"label": product, "value": product} for product in ("CL", "BRN", "BZ", "WBS", "G")
]
_SORT_OPTIONS = [
    {"label": "Sort: Total PnL", "value": "total_pnl"},
    {"label": "Sort: Unrealized PnL", "value": "unrealized_pnl"},
    {"label": "Sort: Days Held", "value": "days_held"},
    {"label": "Sort: Name", "value": "name"},
]

_PRICE_FORMATTER = {"function": "params.value == null ? '—' : d3.format(',.2f')(params.value)"}
_INT_FORMATTER = {"function": "params.value == null ? '—' : d3.format(',')(params.value)"}
_TEXT_FORMATTER = {"function": "params.value == null || params.value === '' ? '—' : params.value"}
_PNL_FORMATTER = {
    "function": (
        "params.value == null ? '—' : "
        "(params.value >= 0 ? '+$' : '-$') + d3.format(',.0f')(Math.abs(params.value))"
    )
}


def _pnl_cell_style(bold: bool = False) -> dict:
    """Green for gains, red for losses; blank (no PnL yet) keeps the default text colour."""
    base = {"fontWeight": "bold"} if bold else {}
    return {
        "styleConditions": [
            {"condition": "params.value > 0", "style": {**base, "color": COLORS["ACCENT_GREEN"]}},
            {"condition": "params.value < 0", "style": {**base, "color": COLORS["ACCENT_RED"]}},
        ],
        "defaultStyle": {**base, "color": COLORS["TEXT_PRIMARY"]},
    }


def _pnl_column(field: str, header: str, bold: bool = False) -> dict:
    return {
        "field": field,
        "headerName": header,
        "type": "rightAligned",
        "width": 130,
        "valueFormatter": _PNL_FORMATTER,
        "cellStyle": _pnl_cell_style(bold),
    }


def _number_column(field: str, header: str, formatter: dict, width: int = 110) -> dict:
    return {"field": field, "headerName": header, "type": "rightAligned", "width": width, "valueFormatter": formatter}


ACTIVE_COLUMN_DEFS = [
    {
        "field": "name",
        "headerName": "Name",
        "pinned": "left",
        "width": 230,
        "cellRenderer": "StructureNameLink",
        "cellRendererParams": {"colors": COLORS},
    },
    {"field": "products", "headerName": "Product(s)", "width": 110},
    {"field": "type", "headerName": "Type", "width": 100},
    _number_column("structure_price", "Structure Price", _PRICE_FORMATTER, 130),
    _number_column("live_price", "Live Price", _PRICE_FORMATTER),
    _pnl_column("unrealized", "Unrealized PnL"),
    _pnl_column("realized", "Realized PnL"),
    _pnl_column("total", "Total PnL", bold=True),
    _number_column("lots", "Lots", _INT_FORMATTER, 80),
    _number_column("days_held", "Days Held", _INT_FORMATTER, 100),
    {
        "field": "status",
        "headerName": "Status",
        "width": 110,
        "cellRenderer": "StatusBadge",
        "cellRendererParams": {"colors": COLORS},
    },
    {
        "colId": "actions",
        "headerName": "Actions",
        "width": 240,
        "sortable": False,
        "cellRenderer": "StructureActions",
        "cellRendererParams": {"colors": COLORS},
    },
]

CLOSED_COLUMN_DEFS = [
    {"field": "name", "headerName": "Name", "pinned": "left", "width": 230},
    {"field": "products", "headerName": "Product(s)", "width": 110},
    {"field": "type", "headerName": "Type", "width": 100},
    _number_column("entry_price", "Entry Price", _PRICE_FORMATTER),
    _number_column("exit_price", "Exit Price", _PRICE_FORMATTER),
    _pnl_column("realized", "Realized PnL", bold=True),
    _number_column("lots", "Lots", _INT_FORMATTER, 80),
    _number_column("days_held", "Days Held", _INT_FORMATTER, 100),
    {"field": "closed_at", "headerName": "Closed At", "width": 160, "valueFormatter": _TEXT_FORMATTER},
    {"field": "close_trigger", "headerName": "Close Trigger", "width": 130, "valueFormatter": _TEXT_FORMATTER},
]

_SHELL_ROW_STYLE = {
    "styleConditions": [{"condition": "params.data.status === 'SHELL'", "style": {"opacity": 0.7}}]
}


def _grid(grid_id: str, column_defs: list[dict], empty_message: str, height: str, **kwargs) -> dag.AgGrid:
    return dag.AgGrid(
        id=grid_id,
        columnDefs=column_defs,
        rowData=[],
        getRowId="params.data.structure_id",
        defaultColDef={"sortable": True, "resizable": True},
        dashGridOptions={
            "rowHeight": ROW_HEIGHT,
            "suppressCellFocus": True,
            "animateRows": False,
            "theme": GRID_THEME,
            "localeText": {"noRowsToShow": empty_message},
        },
        style={"width": "100%", "height": height},
        **kwargs,
    )


def _detail_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Structure Detail"), close_button=False),
            dbc.ModalBody(id="modal-structure-detail-body"),
            dbc.ModalFooter(dbc.Button("Close", id="btn-close-structure-detail", color="secondary")),
        ],
        id="modal-structure-detail",
        size="xl",
        scrollable=True,
        is_open=False,
    )


def structures_layout() -> html.Div:
    """Active structures, collapsible closed structures and the structure detail modal."""
    header = html.Div(
        [
            dbc.Button("+ New Structure", id="btn-new-structure", color="success", style={"float": "right"}),
            html.H3("📊 Structures", style={"color": COLORS["TEXT_PRIMARY"], "margin": 0}),
        ],
        style={"overflow": "hidden", "marginBottom": "16px"},
    )

    filter_bar = dbc.Row(
        [
            dbc.Col(dbc.Select(id="filter-structure-status", options=_STATUS_OPTIONS, value="active"), md=3),
            dbc.Col(dbc.Select(id="filter-structure-product", options=_PRODUCT_OPTIONS, value="all"), md=3),
            dbc.Col(dbc.Select(id="filter-structure-sort", options=_SORT_OPTIONS, value="total_pnl"), md=3),
        ],
        className="mb-3",
    )

    active_grid = _grid(
        "structures-active-grid",
        ACTIVE_COLUMN_DEFS,
        "No active structures. Click '+ New Structure' to begin.",
        "520px",
        getRowStyle=_SHELL_ROW_STYLE,
    )

    closed_section = html.Div(
        [
            dbc.Button(
                "📁 Show Closed Structures (0)",
                id="btn-toggle-closed",
                color="secondary",
                outline=True,
                className="mb-3",
            ),
            dbc.Collapse(
                _grid("structures-closed-grid", CLOSED_COLUMN_DEFS, "No closed structures yet.", "360px"),
                id="structures-closed-collapse",
                is_open=False,
            ),
        ],
        style={"marginTop": "32px"},
    )

    return html.Div(
        [
            header, filter_bar, active_grid, closed_section,
            _detail_modal(),
            # Later in the DOM than the detail modal, so they stack above it.
            confirm_exit_modal(), confirm_edit_modal(), detail_toast(),
        ]
    )
