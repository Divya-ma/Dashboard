"""The page frame served on every load: the URL, the shared stores, the three global timers, the alert containers and the shell.

Kept out of ui/app.py (which wires the real database and services when imported) so tests can build it and check that every id a
callback uses exists somewhere in the layout.
"""

from dash import dcc, html

from ui.layouts.shell import build_alert_container, build_price_alert_container, build_shell
from ui.layouts.structure_builder import builder_save_toast, new_structure_modal

LIVE_POLL_INTERVAL_MS = 10 * 1000  # /fairvalue/ has its own independent 7 req/min budget
PNL_REFRESH_INTERVAL_MS = 5 * 1000
COUNTDOWN_INTERVAL_MS = 1 * 1000


def build_root_layout(token_configured: bool, show_signout: bool = False) -> html.Div:
    return html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="store-token-configured", data=token_configured),
            dcc.Store(id="store-live-prices", data={}),
            dcc.Store(id="store-portfolio-pnl", data={}),
            dcc.Store(id="store-selected-structure-id", data=None),
            dcc.Store(id="store-pending-delete", data=None),
            dcc.Store(id="store-builder-template", data=None),
            dcc.Store(id="store-builder-legs", data=[]),
            dcc.Store(id="store-builder-step", data=1),
            dcc.Store(id="store-builder-correlation", data=[]),
            dcc.Interval(id="interval-live-poll", interval=LIVE_POLL_INTERVAL_MS, n_intervals=0),
            dcc.Interval(id="interval-pnl-refresh", interval=PNL_REFRESH_INTERVAL_MS, n_intervals=0),
            dcc.Interval(id="interval-countdown", interval=COUNTDOWN_INTERVAL_MS, n_intervals=0),
            build_alert_container(),
            build_price_alert_container(),
            build_shell(token_configured, show_signout),
            new_structure_modal(),
            builder_save_toast(),
        ]
    )
