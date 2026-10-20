"""Per-contract net exposure for the Exposure Map tab.

Every traded leg of every open structure contributes ratio x side x lots, where side is +1 for a buy and -1 for a
sell (core.positions.signed_lots: the same rule as Home's net lots, the VaR and the beta panels). A leg whose symbol is
itself a spread/fly is split into its outright contracts (core.positions.leg_outrights), so the map is always per outright.
This module adds only what the map needs on top: gross long, gross short and the structures each outright comes from.
"""

import calendar
import logging

from adapters.base import LETTER_TO_MONTH
from core.models import Structure
from core.positions import filter_open_structures, leg_outrights
from core.structure_utils import parse_symbol

logger = logging.getLogger(__name__)

_EPSILON = 1e-9


def _month_label(symbol: str, with_product: bool) -> str:
    leg = parse_symbol(symbol)[0]
    text = f"{calendar.month_abbr[LETTER_TO_MONTH[leg['letter']]]}-{leg['year']}"
    return f"{leg['product']} {text}" if with_product else text


def _sort_key(symbol: str) -> tuple:
    leg = parse_symbol(symbol)[0]
    return (int(leg["year"]), LETTER_TO_MONTH[leg["letter"]], leg["product"])


def build_exposure_map(structures: list[Structure]) -> list[dict]:
    """Rows [{symbol, label, net_lots, gross_long, gross_short, structures}] sorted by contract month.

    Only OPEN structures count. gross_short is a positive number. `label` is "Dec-26",
    prefixed with the product ("CL Dec-26") when more than one product is present.
    Structures without legs, untraded legs and legs that cannot be decomposed are
    skipped (the last with a logged warning).
    """
    totals: dict[str, dict] = {}
    for structure in filter_open_structures(structures):
        for leg in structure.legs:
            if not leg.is_traded:
                continue
            outrights = leg_outrights(leg, structure.name)
            for symbol, lots in outrights.items():
                entry = totals.setdefault(
                    symbol, {"symbol": symbol, "net_lots": 0.0, "gross_long": 0.0, "gross_short": 0.0, "structures": []}
                )
                entry["net_lots"] += lots
                if lots > 0:
                    entry["gross_long"] += lots
                elif lots < 0:
                    entry["gross_short"] += -lots
                if structure.name not in entry["structures"]:
                    entry["structures"].append(structure.name)

    products = {parse_symbol(symbol)[0]["product"] for symbol in totals}
    rows = sorted(totals.values(), key=lambda row: _sort_key(row["symbol"]))
    for row in rows:
        row["label"] = _month_label(row["symbol"], with_product=len(products) > 1)
        row["net_lots"] = 0.0 if abs(row["net_lots"]) < _EPSILON else round(row["net_lots"], 9)
        row["gross_long"] = round(row["gross_long"], 9)
        row["gross_short"] = round(row["gross_short"], 9)
    return rows


def concentration_warnings(rows: list[dict], threshold: float) -> list[dict]:
    """Rows whose |net_lots| exceeds `threshold`, each with its side (long/short) and size."""
    flagged = []
    for row in rows:
        if abs(row["net_lots"]) > threshold:
            flagged.append(
                {
                    "label": row["label"],
                    "side": "long" if row["net_lots"] > 0 else "short",
                    "lots": abs(row["net_lots"]),
                    "structures": row["structures"],
                }
            )
    return flagged
