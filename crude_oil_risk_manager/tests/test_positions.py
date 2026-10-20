"""core.positions: the one place a book's legs become lots and dollar weights (sold legs count negative everywhere)."""

import pytest

from core import pnl as pnl_module
from core.exposure_map import build_exposure_map
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from core.positions import (
    candidate_dollar_weights,
    dollar_weights,
    filter_open_structures,
    net_lots,
    net_lots_by_product,
    net_lots_by_product_and_symbol,
    signed_lots,
)


def contract(symbol, product="CL", multiplier=1000.0):
    return Contract(product=product, contract_month=12, contract_year=2026, symbol=symbol, multiplier=multiplier, tick_size=0.01, tick_value=10.0)


def leg(symbol, ratio=1, lots=1.0, direction="buy", product="CL", multiplier=1000.0, entry=1.0):
    return Leg(contract=contract(symbol, product, multiplier), ratio=ratio, lots=lots, entry_price=entry, direction=direction)


def structure(name, legs, status=StructureStatus.OPEN):
    return Structure(name=name, structure_type=StructureType.CUSTOM, products=sorted({l.contract.product for l in legs}), legs=legs, status=status)


def test_a_leg_counts_ratio_times_side_times_lots():
    assert signed_lots(leg("CLZ26", ratio=1, lots=3)) == 3
    assert signed_lots(leg("CLZ26", ratio=1, lots=3, direction="sell")) == -3
    assert signed_lots(leg("CLZ26", ratio=-2, lots=3)) == -6  # a fly's body, bought
    assert signed_lots(leg("CLZ26", ratio=-2, lots=3, direction="sell")) == 6  # and sold
    assert signed_lots(Leg(contract=contract("CLZ26"), ratio=1, lots=0.0)) == 0.0  # a shell leg carries nothing


def test_only_open_structures_and_traded_legs_count():
    book = [structure("open", [leg("CLZ26", lots=2)]), structure("closed", [leg("CLZ26", lots=5)], StructureStatus.CLOSED),
            structure("shell", [Leg(contract=contract("CLZ26"), ratio=1, lots=0.0)], StructureStatus.SHELL),
            structure("part", [leg("CLF27", lots=1)], StructureStatus.PARTIALLY_CLOSED)]
    assert [s.name for s in filter_open_structures(book)] == ["open", "part"]
    assert net_lots(book) == {"CLZ26": 2.0, "CLF27": 1.0}
    assert dollar_weights(book) == {"CLZ26": 2000.0, "CLF27": 1000.0}


def test_a_sold_leg_is_negative_in_every_view():
    """The bug this module fixed: Home's net lots and the net-exposure roll-up counted a sold leg as bought."""
    book = [structure("long", [leg("CLZ26", lots=4)]), structure("short", [leg("CLZ26", lots=1, direction="sell"), leg("BRNZ26", lots=2, direction="sell", product="BRN")])]
    assert net_lots(book) == {"CLZ26": 3.0, "BRNZ26": -2.0}
    assert net_lots_by_product(book) == {"CL": 3.0, "BRN": -2.0}
    assert net_lots_by_product_and_symbol(book) == {"CL": {"CLZ26": 3.0}, "BRN": {"BRNZ26": -2.0}}
    assert pnl_module.calculate_net_exposure(book) == {"CL": {"CLZ26": 3.0}, "BRN": {"BRNZ26": -2.0}}
    assert pnl_module.calculate_portfolio_pnl(book, {}, {"CLZ26": 70.0, "BRNZ26": 75.0}, [])["net_lots_by_product"] == {"CL": 3.0, "BRN": -2.0}
    rows = {r["symbol"]: r for r in build_exposure_map(book)}
    assert rows["CLZ26"]["net_lots"] == 3.0 and rows["BRNZ26"]["net_lots"] == -2.0  # the Exposure Map agrees


def test_legs_that_cancel_drop_out_and_the_two_views_split_a_fly_the_same_way():
    book = [structure("a", [leg("CLZ26-F27-G27", lots=2)]), structure("b", [leg("CLF27", ratio=2, lots=2, direction="buy")])]
    assert net_lots(book) == {"CLZ26-F27-G27": 2.0, "CLF27": 4.0}  # as quoted
    outrights = net_lots(book, decompose=True)
    assert outrights == {"CLZ26": 2.0, "CLG27": 2.0}  # the fly's -2 body against 4 bought F27 cancels exactly
    assert dollar_weights(book, decompose=True) == {"CLZ26": 2000.0, "CLG27": 2000.0}


def test_dollar_weights_use_each_legs_multiplier():
    book = [structure("mixed", [leg("CLZ26", lots=1, multiplier=1000.0), leg("GZ26", lots=2, product="G", multiplier=100.0)])]
    assert dollar_weights(book) == {"CLZ26": 1000.0, "GZ26": 200.0}


def test_a_symbol_that_cannot_be_split_is_skipped_in_the_outright_view_only():
    book = [structure("odd", [leg("WEIRD", lots=1), leg("CLZ26", lots=2)])]
    assert net_lots(book, decompose=True) == {"CLZ26": 2.0}
    assert net_lots(book) == {"WEIRD": 1.0, "CLZ26": 2.0}


def test_candidate_weights_are_for_one_lot_bought():
    saved = {"CLZ26": contract("CLZ26", multiplier=500.0)}
    legs = [{"symbol": "clz26", "ratio": 1}, {"symbol": "CLF27", "ratio": -2}, {"symbol": "", "ratio": 1}, {"symbol": "CLG27", "ratio": 0}, {"symbol": "CLH27", "ratio": "x"}]
    assert candidate_dollar_weights(legs, saved.get, default_multiplier=250.0) == {"CLZ26": 500.0, "CLF27": -500.0}
    assert candidate_dollar_weights([{"symbol": "CLF27", "ratio": 1}], lambda s: None) == {"CLF27": 1000.0}  # no saved contract, nothing typed: 1,000 barrels
    assert candidate_dollar_weights([{"symbol": "CLZ26-F27", "ratio": 2}], lambda s: None, decompose=True) == {"CLZ26": 2000.0, "CLF27": -2000.0}
