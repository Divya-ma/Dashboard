"""Placeholder layouts for tabs that are not built yet (replaced in later phases)."""

from dash import html

from ui.layouts.shell import COLORS


def under_construction() -> html.Div:
    """Generic 'under construction' page."""
    return html.Div(
        "🚧 This tab is under construction",
        style={"color": COLORS["TEXT_PRIMARY"], "padding": "40px", "fontSize": "20px"},
    )


def structures_layout() -> html.Div:
    return under_construction()


def correlation_layout() -> html.Div:
    return under_construction()


def exposure_layout() -> html.Div:
    return under_construction()


def var_scenario_layout() -> html.Div:
    return under_construction()


def trade_analyzer_layout() -> html.Div:
    return under_construction()


def archive_layout() -> html.Div:
    return under_construction()
