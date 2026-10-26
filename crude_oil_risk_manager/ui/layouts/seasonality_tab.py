"""Seasonality page (route /seasonality): a structure today against the same structure in earlier years.

Pick a product and a curve type; the table lists the first N contracts' structures (live price, last settlement, the
usual price at this many days before expiry, the deviation, the run of days outside the band, volume). Click a row to
see every earlier year plotted against trading days before expiry, with the band and today's price. All colours come
from COLORS.
"""

from datetime import date

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from core.curve_calendar import DFLY, FAMILIES, FAMILY_LABELS, PRODUCTS
from core.curve_compare import DEFAULT_DFLY_METHOD, DFLY_METHODS, MAX_MONTHS
from core.oi_data import FIRST_CONTRACT_YEAR
from core.seasonality import (
    BAND_METHODS, CHEAP, DEFAULT_DAYS, DEFAULT_MONTHS, DEFAULT_RUN_DAYS, DEFAULT_YEARS, DEFAULT_Z, FLAGGED, MAX_DAYS, MAX_YEARS, MEAN_SD,
    NO_DATA, RICH, WATCH, SeasonRow, SeasonTable, StructureSeason,
)
from ui.layouts.curve_tab import _CARD, _LABEL, _MUTED, _dropdown, _layout, _table, empty_figure
from ui.layouts.shell import COLORS

REFRESH_MS = 10_000
PLACEHOLDER = "Click a structure in the table to see its seasonal curves"
LEVEL_PRICE, LEVEL_CHANGE = "price", "change"
DEFAULT_REFERENCE_DAYS = 250
CURRENT_COLOR, LIVE_COLOR, BAND_FILL, CENTER_COLOR = "#ffffff", "#4da3ff", "rgba(77, 163, 255, 0.16)", "#4da3ff"
YEAR_COLORS = ["#f4a261", "#2a9d8f", "#e76f51", "#a8dadc", "#e9c46a", "#b388eb", "#90be6d", "#f28482", "#84a59d", "#f6bd60", "#9d8189", "#7ec8e3"]
EXCLUDED_COLOR = "#6b6b80"

COLUMN_IDS = [
    ("position", "#", "numeric"), ("contract", "Structure", "text"), ("days", "Days to expiry", "numeric"),
    ("live", "Live", "numeric"), ("source", "Source", "text"), ("settle", "Last settle", "numeric"),
    ("change", "Live − settle", "numeric"), ("avg", "Usual at this time", "numeric"), ("dev", "Deviation (ticks)", "numeric"),
    ("z", "Z", "numeric"), ("above", "Above (years)", "text"), ("run", "Days outside", "numeric"), ("status", "Status", "text"),
    ("also", "Same move in earlier years", "text"), ("volume", "Volume (thinnest leg)", "numeric"),
    ("usual_volume", "Usual volume", "numeric"), ("volume_ratio", "Volume ÷ usual", "numeric"),
]
TABLE_COLUMNS = [{"name": name, "id": cid, "type": kind} for cid, name, kind in COLUMN_IDS]


def _num(value, digits: int = 3):
    return None if value is None or value != value else round(float(value), digits)


def _status_text(row: SeasonRow) -> str:
    evaluation = row.evaluation
    if evaluation.status == NO_DATA:
        return NO_DATA
    suffix = f" {evaluation.direction.upper()}" if evaluation.side else ""
    return f"{evaluation.status}{suffix}"


def _also_text(row: SeasonRow) -> str:
    similar = row.evaluation.similar
    if not row.evaluation.side:
        return ""
    if not similar:
        return f"none of {row.evaluation.years}"
    parts = []
    for item in similar:
        outcome = "?" if item.reverted is None else ("back" if item.reverted else "still out")
        parts.append(f"{item.year} ({item.peak_z:+.1f}σ, {outcome})")
    return f"{len(similar)} of {row.evaluation.years}: " + ", ".join(parts)


def table_rows(table: SeasonTable | None) -> list[dict]:
    if table is None:
        return []
    rows = []
    for row in table.rows:
        season, evaluation = row.season, row.evaluation
        live = evaluation.price if evaluation.used_live else None
        change = None if live is None or season.last_price is None else live - season.last_price
        ratio = None if not season.usual_volume or season.last_volume != season.last_volume or season.last_volume is None else season.last_volume / season.usual_volume
        rows.append({
            "id": season.position, "position": season.position, "contract": season.label, "days": season.td_now,
            "live": _num(live), "source": row.point.source if row.point and evaluation.used_live else "",
            "settle": _num(season.last_price), "change": _num(change), "avg": _num(evaluation.center),
            "dev": _num(evaluation.ticks, 1), "z": _num(evaluation.z, 2),
            "above": "" if evaluation.rank is None else f"{evaluation.rank} of {evaluation.years}",
            "run": evaluation.run if evaluation.status != NO_DATA else None, "status": _status_text(row), "also": _also_text(row),
            "volume": _num(season.last_volume, 0), "usual_volume": _num(season.usual_volume, 0), "volume_ratio": _num(ratio, 2),
        })
    return rows


def build_status(table: SeasonTable | None, error: str | None = None) -> html.Div:
    if error:
        return html.Div(error, style={"color": COLORS["ACCENT_RED"]})
    if table is None:
        return html.Div("Waiting for data...")
    scored = [r for r in table.rows if r.evaluation.status != NO_DATA]
    flagged = sum(1 for r in scored if r.evaluation.status == FLAGGED)
    watch = sum(1 for r in scored if r.evaluation.status == WATCH)
    live = sum(1 for r in scored if r.evaluation.used_live)
    last = max((r.season.last_date for r in table.rows if r.season.last_date), default=None)
    parts = [f"{len(scored)} of {len(table.rows)} structures scored", f"{flagged} flagged", f"{watch} on watch", f"{live} on a live price"]
    if last:
        parts.append(f"last settlement {last.isoformat()}")
    children = [html.Span(" · ".join(parts))]
    for note in table.notes:
        children.append(html.Div(f"⚠️ {note}", style={"color": COLORS["ACCENT_YELLOW"], "marginTop": "4px"}))
    return html.Div(children)


def _short(curve) -> str:
    """The legend name of a year: its front month ("Feb26"); the full structure is in the hover."""
    return curve.label.split("/")[0]


def _y_range(season: StructureSeason, top: int) -> list[float] | None:
    """The 2nd to 98th percentile of every plotted price, padded: the last days before expiry swing by far more than
    the rest of the curve and would flatten it. Zooming out shows everything."""
    curves = [*season.past, *([season.current] if season.current is not None else [])]
    values = np.concatenate([c.frame.loc[c.frame.index <= top, "price"].dropna().to_numpy() for c in curves]) if curves else np.array([])
    if len(values) < 10:
        return None
    low, high = np.percentile(values, [2, 98])
    pad = max((high - low) * 0.15, 1e-9)
    return [float(low - pad), float(high + pad)]


def build_season_figure(row: SeasonRow | None, show_band: bool = True) -> go.Figure:
    """Every earlier year against trading days before expiry, the band, this year's settlements and today's price."""
    if row is None:
        return empty_figure(PLACEHOLDER)
    season, evaluation, options = row.season, row.evaluation, row.season.options
    if not season.past and season.current is None:
        return empty_figure(f"No settlement history for {season.label}")
    figure = go.Figure()
    top = min(season.top, max(options.max_days, season.td_now + 5))
    if show_band:
        bands = season.bands.loc[:top].dropna(subset=["center", "spread"])
        if not bands.empty:
            upper, lower = bands["center"] + options.z * bands["spread"], bands["center"] - options.z * bands["spread"]
            figure.add_trace(go.Scatter(x=bands.index, y=lower, mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
            figure.add_trace(go.Scatter(
                x=bands.index, y=upper, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=BAND_FILL,
                name=f"±{options.z:g} {'SD' if options.band == MEAN_SD else 'MAD'} band", hoverinfo="skip",
            ))
    colour_of = {c.year: YEAR_COLORS[i % len(YEAR_COLORS)] for i, c in enumerate(season.past)}
    for curve in season.past:
        excluded = curve.year in options.excluded
        frame = curve.frame[curve.frame.index <= top]
        figure.add_trace(go.Scatter(
            x=frame.index, y=frame["price"], mode="lines", name=f"{_short(curve)}{' (excluded)' if excluded else ''}",
            line={"color": EXCLUDED_COLOR if excluded else colour_of[curve.year], "width": 1.3, "dash": "dot" if excluded else "solid"},
            opacity=0.6 if excluded else 0.9,
            hovertemplate=f"<b>{curve.label}</b><br>%{{x}} days to expiry<br>%{{y:.3f}}<extra></extra>",
        ))
    if show_band:
        bands = season.bands.loc[:top].dropna(subset=["center"])
        figure.add_trace(go.Scatter(
            x=bands.index, y=bands["center"], mode="lines", name="Average" if options.band == MEAN_SD else "Median",
            line={"color": CENTER_COLOR, "width": 2, "dash": "dash"}, hovertemplate="average %{y:.3f}<extra></extra>",
        ))
    if season.current is not None:
        frame = season.current.frame[season.current.frame.index <= top]
        figure.add_trace(go.Scatter(
            x=frame.index, y=frame["price"], mode="lines", name=f"{_short(season.current)} (settlements)",
            line={"color": CURRENT_COLOR, "width": 3},
            hovertemplate=f"<b>{season.current.label}</b><br>%{{x}} days to expiry<br>%{{y:.3f}}<extra></extra>",
        ))
    if evaluation.price is not None and evaluation.status != NO_DATA:
        marker_colour = COLORS["ACCENT_RED"] if evaluation.status == FLAGGED else COLORS["ACCENT_YELLOW"] if evaluation.status == WATCH else LIVE_COLOR
        x = season.td_now if evaluation.used_live else season.current.last_td
        figure.add_trace(go.Scatter(
            x=[x], y=[evaluation.price], mode="markers", name="Live" if evaluation.used_live else "Last settlement",
            marker={"symbol": "star", "size": 16, "color": marker_colour, "line": {"width": 1, "color": CURRENT_COLOR}},
            hovertemplate=(
                f"<b>{'Live' if evaluation.used_live else 'Last settlement'}</b> %{{y:.3f}}<br>usual {evaluation.center:.3f}<br>"
                f"{evaluation.ticks:+.1f} ticks, z {evaluation.z:+.2f}<extra></extra>"
            ),
        ))
    level = "Change" if options.rebase_days is not None else "Price"
    title = f"{PRODUCTS[season.product].name} {FAMILY_LABELS[season.family]} {season.label}"
    if evaluation.status != NO_DATA:
        title += f" — {_status_text(row)}, z {evaluation.z:+.2f}"
    figure.update_layout(**_layout(
        title={"text": title, "font": {"size": 14}},
        xaxis={"title": "Trading days before expiry", "range": [top, 0], "gridcolor": COLORS["BORDER_COLOR"]},
        yaxis={"title": f"{level} (settlement)" if options.rebase_days is None else f"Change since {options.rebase_days} days before expiry",
               "gridcolor": COLORS["BORDER_COLOR"], "zerolinecolor": COLORS["TEXT_SECONDARY"], "range": _y_range(season, top)},
        legend={"orientation": "h", "yanchor": "top", "y": -0.2},
        margin={"l": 70, "r": 30, "t": 50, "b": 150},
        hovermode="closest",
        uirevision=f"{season.product}|{season.family}|{season.position}|{options.max_days}|{options.rebase_days}",
    ))
    return figure


STATUS_STYLES = [
    {"if": {"column_id": "status", "filter_query": f'{{status}} = "{FLAGGED} {RICH.upper()}"'}, "color": COLORS["ACCENT_RED"], "fontWeight": "bold"},
    {"if": {"column_id": "status", "filter_query": f'{{status}} = "{FLAGGED} {CHEAP.upper()}"'}, "color": COLORS["ACCENT_GREEN"], "fontWeight": "bold"},
    {"if": {"column_id": "status", "filter_query": f'{{status}} contains "{WATCH}"'}, "color": COLORS["ACCENT_YELLOW"]},
    {"if": {"column_id": "status", "filter_query": f'{{status}} = "{NO_DATA}"'}, "color": COLORS["TEXT_SECONDARY"]},
]


def seasonality_layout() -> html.Div:
    product_options = [{"label": p.name, "value": c} for c, p in PRODUCTS.items()]
    years = list(range(FIRST_CONTRACT_YEAR, date.today().year + 1))
    method_options = [{"label": label, "value": key} for key, label in DFLY_METHODS.items()]

    def number(component_id, value, low, high, step=1):
        return dcc.Input(id=component_id, type="number", min=low, max=high, step=step, value=value, debounce=True, className="form-control")

    def field(label, control, width):
        return dbc.Col([html.Div(label, style=_LABEL), control], md=width)

    return html.Div([
        dcc.Interval(id="season-refresh", interval=REFRESH_MS, n_intervals=0),
        dcc.Store(id="season-selected", data=None),
        html.H3("🗓️ Seasonality", style={"color": COLORS["TEXT_PRIMARY"], "marginBottom": "4px"}),
        html.Div(id="season-status", style={**_MUTED, "marginBottom": "12px", "minHeight": "20px"}),
        dbc.Row([
            field("Product", _dropdown("season-product", product_options, "BRN"), 2),
            field("Curve", _dropdown("season-family", [{"label": FAMILY_LABELS[f], "value": f} for f in FAMILIES], DFLY), 2),
            field(f"Months (1-{MAX_MONTHS})", number("season-months", DEFAULT_MONTHS, 1, MAX_MONTHS), 2),
            field("Earlier years", number("season-years", DEFAULT_YEARS, 1, MAX_YEARS), 2),
            field("Days before expiry", number("season-days", DEFAULT_DAYS, 10, MAX_DAYS), 2),
            field("D-fly built from", _dropdown("season-method", method_options, DEFAULT_DFLY_METHOD), 2),
        ], className="mb-3"),
        dbc.Row([
            field("Band", _dropdown("season-band", [{"label": v, "value": k} for k, v in BAND_METHODS.items()], MEAN_SD), 3),
            field("Flag beyond (σ)", number("season-z", DEFAULT_Z, 0.5, 10, 0.1), 2),
            field("…for (days in a row)", number("season-run", DEFAULT_RUN_DAYS, 1, 60), 2),
            field("Leave out of the average", dcc.Dropdown(
                id="season-excluded", options=[{"label": str(y), "value": y} for y in years], value=[], multi=True,
                placeholder="No year excluded",
            ), 5),
        ], className="mb-3"),
        dbc.Row([
            field("Show", _dropdown("season-level", [
                {"label": "Settlement price", "value": LEVEL_PRICE},
                {"label": "Change since a reference day", "value": LEVEL_CHANGE},
            ], LEVEL_PRICE), 3),
            field("Reference (days before expiry)", number("season-reference", DEFAULT_REFERENCE_DAYS, 5, MAX_DAYS), 2),
            dbc.Col(dbc.Switch(id="season-show-band", label="Show the band and average on the chart", value=True, className="mt-4"), md=4),
        ], className="mb-3"),
        dbc.Card(dbc.CardBody(
            html.Div(
                _table("season-table", TABLE_COLUMNS, [], STATUS_STYLES, page_size=40),
                style={"overflowX": "auto"},
            ),
        ), style=_CARD, className="mb-3"),
        dbc.Card(dbc.CardBody([
            dcc.Graph(id="season-graph", figure=empty_figure(PLACEHOLDER), config={"displayModeBar": False}, style={"height": "520px"}),
        ]), style=_CARD, className="mb-3"),
        html.Div(
            "Each structure is compared with the same structure of earlier delivery years at the same number of trading days "
            "before expiry (the front leg's expiry), on settlement prices. The live price is the Curve Compare price (quote VWAP, "
            "fair value as fallback). A structure is FLAGGED when it has stayed outside the band, on one side, for the chosen number "
            "of days in a row (today's live price counts as a day); fewer days is WATCH. 'Same move in earlier years' lists the other "
            "years that left their own band on the same side within 10 trading days of today's point, and whether they were back "
            "inside 20 trading days later. Click a row to plot it. Settings stay only for this visit.",
            style={**_MUTED, "marginTop": "10px"},
        ),
    ])
