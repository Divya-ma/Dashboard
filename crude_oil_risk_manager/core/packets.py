"""Home page packets: the open structures grouped by product and shape, each with side, lots, prices and risk.

A packet is "BRN flies", "CL spreads", "BRN/CL spreads" ...: every open structure of the same products and
type. Per structure:

    side        BUY / SELL = the direction its traded legs were entered on ("MIXED" if they differ)
    lots        the lots of its first traded leg (legs of one structure trade the same lots, scaled by their ratio)
    entry/live  the structure's quoted price = sum of ratio x leg price (average entry price / live price)
    risk        1-day historical VaR at 95% of that structure alone (same method and settlement history as the
                VaR page), a positive dollar loss; a packet's risk is the VaR of its structures TOGETHER, so
                offsetting structures net out.
"""

import logging

from core.models.structure import Structure
from core.positions import dollar_weights
from core.var import historical_var

logger = logging.getLogger(__name__)

RISK_CONFIDENCE = 0.95
RISK_LOOKBACK = 250
_PLURAL = {"outright": "outrights", "spread": "spreads", "fly": "flies", "custom": "custom"}


def packet_name(structure: Structure) -> str:
    products = "/".join(sorted({leg.contract.product for leg in structure.legs}))
    return f"{products} {_PLURAL.get(structure.structure_type.value, structure.structure_type.value)}"


def structure_side(structure: Structure) -> str:
    directions = {leg.direction for leg in structure.legs if leg.is_traded}
    if len(directions) != 1:
        return "MIXED" if directions else ""
    return "BUY" if directions == {"buy"} else "SELL"


def structure_lots(structure: Structure) -> float:
    traded = [leg for leg in structure.legs if leg.is_traded]
    return traded[0].lots if traded else 0.0


def entry_price(structure: Structure) -> float | None:
    traded = [leg for leg in structure.legs if leg.is_traded]
    if not traded:
        return None
    return sum(leg.ratio * (leg.average_entry_price if leg.average_entry_price is not None else leg.entry_price) for leg in traded)


def live_price(structure: Structure, prices: dict[str, float]) -> float | None:
    traded = [leg for leg in structure.legs if leg.is_traded]
    if not traded or any(leg.contract.symbol not in prices for leg in traded):
        return None
    return sum(leg.ratio * prices[leg.contract.symbol] for leg in traded)


class RiskCache:
    """Historical VaR per book, remembered until the book or the stored settlements change (so a poll costs nothing)."""

    def __init__(self, data_loader_getter, version_getter=lambda: None):
        self._loader, self._version = data_loader_getter, version_getter
        self._cache: dict[tuple, float | None] = {}

    def var(self, structures: list[Structure]) -> float | None:
        weights = dollar_weights(structures)
        if not weights:
            return 0.0
        key = (tuple(sorted((s, round(w, 6)) for s, w in weights.items())), self._version())
        if key not in self._cache:
            if len(self._cache) > 500:
                self._cache.clear()
            try:
                result = historical_var(weights, RISK_LOOKBACK, self._loader(), (RISK_CONFIDENCE,))
                self._cache[key] = None if result.error else result.var[RISK_CONFIDENCE]
            except Exception:  # noqa: BLE001
                logger.exception("Structure risk failed")
                self._cache[key] = None
        return self._cache[key]


def build_packets(structures: list[Structure], per_structure: dict, prices: dict[str, float], risk: RiskCache) -> list[dict]:
    """JSON-safe packets, biggest absolute unrealized PnL first. `per_structure` is calculate_portfolio_pnl's."""
    groups: dict[str, list[Structure]] = {}
    for structure in structures:
        if structure.structure_id in per_structure:
            groups.setdefault(packet_name(structure), []).append(structure)
    packets = []
    for name, members in groups.items():
        rows = []
        for s in members:
            info = per_structure[s.structure_id]
            rows.append({
                "structure_id": s.structure_id, "name": s.name, "side": structure_side(s), "lots": structure_lots(s),
                "entry": entry_price(s), "live": live_price(s, prices), "risk": risk.var([s]),
                "unrealized": info["unrealized"], "total": info["total"], "is_stale": info.get("is_stale", False),
            })
        rows.sort(key=lambda r: r["unrealized"], reverse=True)
        packets.append({
            "name": name, "rows": rows, "unrealized": sum(r["unrealized"] for r in rows),
            "risk": risk.var(members),
        })
    packets.sort(key=lambda p: abs(p["unrealized"]), reverse=True)
    return packets
