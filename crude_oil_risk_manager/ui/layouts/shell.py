"""App shell layout: fixed sidebar, setup-required banner and page content area.

This module owns the colour palette. Every other layout/callback file must
import colours from here (`from ui.layouts.shell import COLORS`) instead of
defining its own.
"""

import dash_bootstrap_components as dbc
from dash import html

DARK_BG = "#0d0d1a"
SIDEBAR_BG = "#1a1a2e"
CARD_BG = "#16213e"
ACCENT_BLUE = "#0f3460"
ACCENT_GREEN = "#00ff88"
ACCENT_RED = "#ff4444"
ACCENT_YELLOW = "#ffaa00"
TEXT_PRIMARY = "#e0e0e0"
TEXT_SECONDARY = "#888888"
BORDER_COLOR = "#2a2a4a"

COLORS = {
    "DARK_BG": DARK_BG,
    "SIDEBAR_BG": SIDEBAR_BG,
    "CARD_BG": CARD_BG,
    "ACCENT_BLUE": ACCENT_BLUE,
    "ACCENT_GREEN": ACCENT_GREEN,
    "ACCENT_RED": ACCENT_RED,
    "ACCENT_YELLOW": ACCENT_YELLOW,
    "TEXT_PRIMARY": TEXT_PRIMARY,
    "TEXT_SECONDARY": TEXT_SECONDARY,
    "BORDER_COLOR": BORDER_COLOR,
}

APP_VERSION = "v1.0.0"
SIDEBAR_WIDTH_PX = 260

NAV_ITEMS = [
    ("/", "🏠 Home"),
    ("/structures", "📊 Structures"),
    ("/correlation", "🔗 Correlation"),
    ("/exposure", "📈 Exposure Map"),
    ("/var-scenario", "⚠️ VaR & Scenarios"),
    ("/trade-analyzer", "🎯 Trade Analyzer"),
    ("/archive", "📁 Archive"),
    ("/settings", "⚙️ Settings"),
]

SETUP_BANNER_TEXT = (
    "⚠️ API token not configured. Live data is unavailable. "
    "Please go to ⚙️ Settings to enter your Bearer token."
)


def build_global_css() -> str:
    """Global CSS built from COLORS so the palette stays defined in one place."""
    return f"""
body {{ background-color: {DARK_BG}; color: {TEXT_PRIMARY}; }}
:root {{
    --Dash-Stroke-Strong: {BORDER_COLOR};
    --Dash-Stroke-Weak: {BORDER_COLOR};
    --Dash-Fill-Interactive-Strong: {ACCENT_GREEN};
    --Dash-Fill-Interactive-Weak: {SIDEBAR_BG};
    --Dash-Fill-Inverse-Strong: {SIDEBAR_BG};
    --Dash-Text-Primary: {TEXT_PRIMARY};
    --Dash-Text-Strong: {TEXT_PRIMARY};
    --Dash-Text-Weak: {TEXT_SECONDARY};
    --Dash-Text-Disabled: {TEXT_SECONDARY};
    --Dash-Fill-Primary-Hover: {ACCENT_BLUE};
    --Dash-Fill-Primary-Active: {ACCENT_BLUE};
    --Dash-Fill-Disabled: {CARD_BG};
}}
.sidebar-link {{ color: {TEXT_SECONDARY} !important; border-radius: 6px; margin-bottom: 4px; }}
.sidebar-link:hover {{ color: {TEXT_PRIMARY} !important; background-color: {CARD_BG}; }}
.sidebar-link.active {{
    color: {TEXT_PRIMARY} !important;
    background-color: {ACCENT_BLUE} !important;
    border-left: 3px solid {ACCENT_GREEN};
}}
.accordion {{
    --bs-accordion-bg: {CARD_BG};
    --bs-accordion-color: {TEXT_PRIMARY};
    --bs-accordion-border-color: {BORDER_COLOR};
    --bs-accordion-btn-color: {TEXT_PRIMARY};
    --bs-accordion-btn-bg: {CARD_BG};
    --bs-accordion-active-color: {TEXT_PRIMARY};
    --bs-accordion-active-bg: {ACCENT_BLUE};
}}
.accordion-button::after {{ filter: invert(1); }}
.form-control, .form-control:focus {{
    background-color: {SIDEBAR_BG};
    color: {TEXT_PRIMARY};
    border-color: {BORDER_COLOR};
}}
.form-control::placeholder {{ color: {TEXT_SECONDARY}; }}
.table {{
    --bs-table-bg: {CARD_BG};
    --bs-table-color: {TEXT_PRIMARY};
    --bs-table-striped-bg: {SIDEBAR_BG};
    --bs-table-striped-color: {TEXT_PRIMARY};
    --bs-table-border-color: {BORDER_COLOR};
}}
.table thead th {{ color: {TEXT_SECONDARY}; font-weight: normal; }}
.template-card {{ cursor: pointer; transition: transform 0.1s; }}
.template-card:hover {{ transform: translateY(-2px); }}
.template-card:hover .card {{ border-color: {TEXT_SECONDARY}; }}
.template-card.selected .card {{ border: 2px solid {ACCENT_GREEN}; }}
"""


def stale_indicator(is_stale: bool) -> list:
    """Children for the sidebar stale-data indicator: green 'Data Live' or red 'Data Stale'."""
    if is_stale:
        return [html.Span("●", style={"color": ACCENT_RED, "marginRight": "8px"}), "⚠️ Data Stale"]
    return [html.Span("●", style={"color": ACCENT_GREEN, "marginRight": "8px"}), "Data Live"]


def build_alert_container() -> html.Div:
    """Fixed top-right container that in-app alert toasts are rendered into."""
    return html.Div(
        id="alert-toast-container",
        style={"position": "fixed", "top": "20px", "right": "20px", "zIndex": 2000, "width": "360px"},
    )


def build_sidebar() -> html.Div:
    """Fixed 260px sidebar: title, navigation, and a pinned data-status footer."""
    header = html.Div(
        [
            html.Div("🛢 Crude Oil Risk Manager", style={"fontSize": "18px", "fontWeight": "bold", "color": TEXT_PRIMARY}),
            html.Div("Portfolio Risk & Analytics", style={"fontSize": "12px", "color": TEXT_SECONDARY, "marginTop": "4px"}),
        ],
        style={"padding": "24px 20px", "borderBottom": f"1px solid {BORDER_COLOR}"},
    )

    nav = dbc.Nav(
        [
            dbc.NavLink(label, href=path, active="exact", className="sidebar-link")
            for path, label in NAV_ITEMS
        ],
        vertical=True,
        pills=True,
        style={"padding": "16px 12px", "flex": "1", "overflowY": "auto"},
    )

    footer_text = {"fontSize": "12px", "color": TEXT_SECONDARY, "marginBottom": "6px"}
    footer = html.Div(
        [
            html.Div(
                stale_indicator(False),
                id="sidebar-stale-indicator",
                style={"fontSize": "13px", "color": TEXT_PRIMARY, "marginBottom": "10px"},
            ),
            html.Div("Last: --:--:--", id="sidebar-last-updated", style=footer_text),
            html.Div("Next refresh in: 60s", id="sidebar-refresh-countdown", style=footer_text),
            html.Div(APP_VERSION, style={**footer_text, "marginTop": "10px", "marginBottom": "0"}),
        ],
        style={"padding": "16px 20px", "borderTop": f"1px solid {BORDER_COLOR}"},
    )

    return html.Div(
        [header, nav, footer],
        id="sidebar",
        style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "bottom": 0,
            "width": f"{SIDEBAR_WIDTH_PX}px",
            "backgroundColor": SIDEBAR_BG,
            "borderRight": f"1px solid {BORDER_COLOR}",
            "display": "flex",
            "flexDirection": "column",
            "zIndex": 1000,
        },
    )


def build_content_area(token_configured: bool) -> html.Div:
    """Main content area: setup-required banner above the page content container."""
    banner = dbc.Alert(
        SETUP_BANNER_TEXT,
        id="banner-setup-required",
        color="warning",
        dismissable=False,
        style={"display": "none" if token_configured else "block"},
    )
    return html.Div(
        [banner, html.Div(id="page-content")],
        id="content-area",
        style={
            "marginLeft": f"{SIDEBAR_WIDTH_PX}px",
            "padding": "24px",
            "minHeight": "100vh",
            "backgroundColor": DARK_BG,
            "color": TEXT_PRIMARY,
        },
    )


def build_shell(token_configured: bool) -> html.Div:
    """Sidebar plus content area."""
    return html.Div([build_sidebar(), build_content_area(token_configured)])
