"""Per-contract net exposure for the Exposure Map tab.

Every traded leg of every open structure contributes ratio * side * lots, where
side is +1 for a buy and -1 for a sell (the leg ratio carries the leg's own side
within the structure, the direction flips a sold position). A leg whose symbol is
itself a spread/fly is decomposed into its outright contracts with
core.structure_utils.decompose_to_outrights, so the map is always per outright.

core.exposure.calculate_net_exposure is deliberately not used here: it keeps
exchange-quoted symbols intact (VaR needs their own price series) and ignores
Leg.direction, whereas this map needs outrights and the sell sign.
"""

import calendar
import logging

from adapters.base import LETTER_TO_MONTH
from core.exposure import filter_open_structures
from core.models import Structure
from core.structure_utils import decompose_to_outrights, parse_symbol

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
            signed_lots = leg.ratio * (1 if leg.direction == "buy" else -1) * leg.lots
            try:
                outrights = decompose_to_outrights(leg.contract.symbol, signed_lots)
            except ValueError as exc:
                logger.warning("Skipping leg %s of %s in the exposure map: %s", leg.contract.symbol, structure.name, exc)
                continue
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
