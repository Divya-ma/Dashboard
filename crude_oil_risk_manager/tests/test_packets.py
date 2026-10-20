"""core.packets: structures grouped by product + shape, with side, lots, entry/live price and risk."""

import numpy as np
import pandas as pd

from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from core.packets import RiskCache, build_packets, entry_price, live_price, packet_name, structure_lots, structure_side


def leg(symbol, ratio, lots=2.0, direction="buy", product="BRN", entry=75.0):
    c = Contract(product=product, contract_month=12, contract_year=2026, symbol=symbol, multiplier=1000.0, tick_size=0.01, tick_value=10.0)
    return Leg(contract=c, ratio=ratio, lots=lots, entry_price=entry, direction=direction)


def fly(name="fly", direction="buy", product="BRN"):
    legs = [leg("BRNZ26", 1, direction=direction, product=product, entry=76.0), leg("BRNF27", -2, direction=direction, product=product, entry=75.5),
            leg("BRNG27", 1, direction=direction, product=product, entry=75.0)]
    return Structure(name=name, structure_type=StructureType.FLY, products=[product], legs=legs, status=StructureStatus.OPEN)


class Loader:
    def __init__(self):
        self.calls = 0
        self.series = {s: pd.Series(np.cumsum(np.random.default_rng(i).normal(0, 0.5, 300)) + 75.0, index=pd.bdate_range("2025-01-01", periods=300))
                       for i, s in enumerate(["BRNZ26", "BRNF27", "BRNG27"])}

    def get_available_date_range(self, symbol):
        return (self.series[symbol].index[0], self.series[symbol].index[-1])

    def load_price_differences(self, symbol, min_rows=2):
        self.calls += 1
        return self.series[symbol].diff().dropna()


def test_name_side_lots_and_prices_of_a_structure():
    s = fly()
    assert packet_name(s) == "BRN flies" and structure_side(s) == "BUY" and structure_lots(s) == 2.0
    assert entry_price(s) == 76.0 - 2 * 75.5 + 75.0
    assert live_price(s, {"BRNZ26": 77.0, "BRNF27": 76.0, "BRNG27": 75.5}) == 77.0 - 152.0 + 75.5
    assert live_price(s, {"BRNZ26": 77.0}) is None  # a missing leg price gives no structure price
    assert structure_side(fly(direction="sell")) == "SELL"


def test_packets_group_by_product_and_shape_and_risk_is_cached():
    loader = Loader()
    risk = RiskCache(lambda: loader, lambda: "v1")
    a, b = fly("a"), fly("b")
    other = Structure(name="spr", structure_type=StructureType.SPREAD, products=["BRN"], legs=[leg("BRNZ26", 1), leg("BRNF27", -1)], status=StructureStatus.OPEN)
    info = {s.structure_id: {"unrealized": u, "total": u} for s, u in ((a, 100.0), (b, -40.0), (other, 5.0))}
    prices = {"BRNZ26": 77.0, "BRNF27": 76.0, "BRNG27": 75.5}
    packets = build_packets([a, b, other], info, prices, risk)
    assert [p["name"] for p in packets] == ["BRN flies", "BRN spreads"]  # biggest |unrealized| first
    flies = packets[0]
    assert [r["name"] for r in flies["rows"]] == ["a", "b"] and flies["unrealized"] == 60.0
    assert flies["rows"][0]["risk"] > 0 and flies["risk"] > 0
    calls = loader.calls
    build_packets([a, b, other], info, prices, risk)
    assert loader.calls == calls  # the second poll is served from the cache
