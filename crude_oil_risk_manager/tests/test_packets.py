"""core.packets: structures grouped by product + shape, with side, lots, entry/live price and risk."""

import numpy as np
import pandas as pd

from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from core.packets import RiskCache, build_packets, entry_price, live_price, structure_lots, structure_side


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
    assert structure_side(s) == "BUY" and structure_lots(s) == 2.0
    assert entry_price(s) == 76.0 - 2 * 75.5 + 75.0
    assert live_price(s, {"BRNZ26": 77.0, "BRNF27": 76.0, "BRNG27": 75.5}) == 77.0 - 152.0 + 75.5
    assert live_price(s, {"BRNZ26": 77.0}) is None  # a missing leg price gives no structure price
    assert structure_side(fly(direction="sell")) == "SELL"


def test_one_packet_per_structure_with_a_row_per_leg_and_cached_risk():
    loader = Loader()
    risk = RiskCache(lambda: loader, lambda: "v1")
    a, b = fly("a"), fly("b", direction="sell")
    info = {a.structure_id: {"unrealized": 100.0, "total": 100.0}, b.structure_id: {"unrealized": -400.0, "total": -400.0}}
    prices = {"BRNZ26": 77.0, "BRNF27": 76.0, "BRNG27": 75.5}
    packets = build_packets([a, b], info, prices, risk)
    assert [p["name"] for p in packets] == ["b", "a"]  # biggest |unrealized| first
    top = packets[1]
    assert top["side"] == "BUY" and top["lots"] == 2.0 and top["risk"] > 0 and top["unrealized"] == 100.0
    assert [(leg["symbol"], leg["side"], leg["lots"]) for leg in top["legs"]] == [("BRNZ26", "BUY", 2.0), ("BRNF27", "SELL", 4.0), ("BRNG27", "BUY", 2.0)]
    assert top["legs"][0]["live"] == 77.0 and top["legs"][0]["entry"] == 76.0 and top["legs"][0]["unrealized"] == 2000.0
    assert packets[0]["legs"][1]["side"] == "BUY"  # a sold fly buys its body
    calls = loader.calls
    build_packets([a, b], info, prices, risk)
    assert loader.calls == calls  # the second poll is served from the cache


def test_a_packet_carries_the_latest_stop_and_target_entered_with_a_trade():
    from datetime import datetime, timedelta, timezone

    from core.models import Trade, TradeEventType
    from core.packets import latest_level

    s = fly("a")
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)

    def trade(days, stop=None, target=None):
        return Trade(structure_id=s.structure_id, leg_id="x", event_type=TradeEventType.TRADE, lots=1, price=1.0, direction="buy",
                     timestamp=t0 + timedelta(days=days), stop_loss_price=stop, target_price=target)

    trades = [trade(0, stop=-1.0, target=2.0), trade(1), trade(2, stop=-0.5)]
    assert latest_level(trades, "stop_loss_price") == -0.5 and latest_level(trades, "target_price") == 2.0  # each from the latest trade that set it
    assert latest_level([trade(0)], "target_price") is None and latest_level(None, "stop_loss_price") is None
    info = {s.structure_id: {"unrealized": 0.0, "total": 0.0}}
    packet = build_packets([s], info, {}, RiskCache(lambda: Loader()), {s.structure_id: trades})[0]
    assert packet["stop"] == -0.5 and packet["target"] == 2.0
