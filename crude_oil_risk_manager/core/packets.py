"""Home page packets: one per open structure, with the structure's side, lots, prices and risk and a row per leg.

Per structure:

    side        BUY / SELL = the direction its traded legs were entered on ("MIXED" if they differ)
    lots        the lots of its first traded leg (legs of one structure trade the same lots, scaled by their ratio)
    entry/live  the structure's quoted price = sum of ratio x leg price (average entry price / live price)
    risk        1-day historical VaR at 95% of that structure alone (same method and settlement history as the
                VaR page), a positive dollar loss. A leg row shows the same for that leg alone, so the legs'
                risks add up to MORE than the structure's: they offset each other inside it.
"""

import logging

from core.models.structure import Structure
from core.pnl import calculate_structure_unrealized_pnl
from core.positions import dollar_weights, signed_lots
from core.var import historical_var

logger = logging.getLogger(__name__)

RISK_CONFIDENCE = 0.95
RISK_LOOKBACK = 250


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


def latest_level(trades: list, field: str) -> float | None:
    """The stop-loss or target price (`stop_loss_price` / `target_price`) of the most recent trade that set one."""
    set_ones = [t for t in trades or [] if getattr(t, field, None) is not None]
    return float(getattr(max(set_ones, key=lambda t: t.timestamp), field)) if set_ones else None


class RiskCache:
    """Historical VaR per book, remembered until the book or the stored settlements change (so a poll costs nothing)."""

    def __init__(self, data_loader_getter, version_getter=lambda: None):
        self._loader, self._version = data_loader_getter, version_getter
        self._cache: dict[tuple, float | None] = {}

    def var_of_weights(self, weights: dict[str, float]) -> float | None:
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

    def var(self, structures: list[Structure]) -> float | None:
        return self.var_of_weights(dollar_weights(structures))


def _leg_rows(structure: Structure, prices: dict[str, float], risk: RiskCache) -> list[dict]:
    _, breakdown, _ = calculate_structure_unrealized_pnl(structure.legs, prices)
    rows = []
    for leg in structure.legs:
        if not leg.is_traded:
            continue
        signed = signed_lots(leg)
        symbol = leg.contract.symbol
        entry = leg.average_entry_price if leg.average_entry_price is not None else leg.entry_price
        rows.append({
            "symbol": symbol, "side": "BUY" if signed > 0 else "SELL", "lots": abs(signed), "entry": entry,
            "live": prices.get(symbol), "risk": risk.var_of_weights({symbol: signed * leg.contract.multiplier}),
            "unrealized": breakdown.get(leg.leg_id),
        })
    return rows


def build_packets(
    structures: list[Structure], per_structure: dict, prices: dict[str, float], risk: RiskCache, trades_by_structure: dict | None = None,
) -> list[dict]:
    """JSON-safe packets, one per open structure, biggest absolute unrealized PnL first. `per_structure` is calculate_portfolio_pnl's;
    `trades_by_structure` supplies the stop and target (structure prices) last entered with a trade."""
    packets = []
    for s in structures:
        info = per_structure.get(s.structure_id)
        if info is None:
            continue
        packets.append({
            "structure_id": s.structure_id, "name": s.name, "side": structure_side(s), "lots": structure_lots(s),
            "entry": entry_price(s), "live": live_price(s, prices), "risk": risk.var([s]),
            "target": latest_level((trades_by_structure or {}).get(s.structure_id), "target_price"),
            "stop": latest_level((trades_by_structure or {}).get(s.structure_id), "stop_loss_price"),
            "unrealized": info["unrealized"], "unrealized_net": info["unrealized"] - info.get("open_costs", 0.0), "total": info["total"], "is_stale": info.get("is_stale", False),
            "legs": _leg_rows(s, prices, risk),
        })
    packets.sort(key=lambda p: abs(p["unrealized"]), reverse=True)
    return packets
