"""Gasoil (G, API code GO) next to CL and Brent: its calendar, symbols, contract size and tick, and the places that scale by them."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from adapters.base import SymbolTranslator
from core import structure_builder as sb
from core.atr_engine import pair_hedge
from core.contract_specs import spec_for_product, spec_for_symbol
from core.curve_calendar import (
    PRODUCTS, api_symbol, build_structures, contract_expiry, front_month, generic_contracts, product_of_symbol,
)
from core.curve_settings import CurveParams
from core.curve_trade import point_value_for
from core.positions import candidate_dollar_weights
from db.repository import Repository


def test_gasoil_is_a_registered_curve_product_with_its_own_size_and_tick():
    assert PRODUCTS["G"].api_code == "GO" and list(PRODUCTS) == ["CL", "BRN", "G"]
    g, cl = spec_for_product("G"), spec_for_product("CL")
    assert (g.multiplier, g.tick_size, g.tick_value) == (100.0, 0.25, 25.0)
    assert (cl.multiplier, cl.tick_size, cl.tick_value) == (1000.0, 0.01, 10.0)
    assert g.tick_size * g.multiplier == g.tick_value and cl.tick_size * cl.multiplier == cl.tick_value
    assert spec_for_symbol("GZ26-F27") is g and spec_for_symbol("BRNZ26") is spec_for_product("BRN") and spec_for_symbol("WBSZ26").multiplier == 1000.0


def test_gasoil_symbols_translate_both_ways_and_are_recognised():
    assert SymbolTranslator.internal_to_api("GZ26") == "GOZ26" and SymbolTranslator.api_to_internal("GOZ26") == "GZ26"
    assert SymbolTranslator.internal_to_api("GZ26-F27-G27") == "GOZ26-F27-G27"
    assert api_symbol("G", 12, 2026) == "GOZ26" and product_of_symbol("GOZ26-F27") == "G" and product_of_symbol("COZ26") == "BRN"


def test_gasoil_expires_two_business_days_before_the_14th_of_its_delivery_month():
    assert contract_expiry("G", 12, 2026) == date(2026, 12, 10)  # the 14th is a Monday: Thursday before
    assert contract_expiry("G", 1, 2027) == date(2027, 1, 12)  # the 14th is a Thursday
    assert contract_expiry("G", 5, 2026) == date(2026, 5, 12)  # the 14th is a Thursday... and 1 May bank holiday is irrelevant
    for month in range(1, 13):
        expiry = contract_expiry("G", month, 2026)
        assert expiry.month == month and expiry.weekday() < 5 and 8 <= expiry.day <= 12  # inside its own delivery month, before the 14th


def test_the_gasoil_ladder_starts_with_the_contract_still_trading():
    assert front_month("G", date(2026, 10, 20)) == (11, 2026)  # October gasoil expired on the 12th
    assert [c.symbol for c in generic_contracts("G", date(2026, 10, 20), 3)] == ["GOX26", "GOZ26", "GOF27"]
    ladder = build_structures("G", date(2026, 10, 20))
    assert ladder["fly"][0].symbol == "GOX26-Z26-F27" and ladder["fly"][0].generic_code == "GO1-2-3"
    assert ladder["spread"][1].generic_code == "GO2-3" and len(ladder["dfly"]) == 12


def test_a_new_gasoil_structure_takes_gasoil_specs_when_the_boxes_are_blank(tmp_path):
    repo = Repository(str(tmp_path / "db.sqlite"))
    built = sb.build_shell_structure(
        "G spread", "spread", ["GZ26", "GF27"], [1, -1], None, "", None, "", repo.get_contract)
    assert {(c.multiplier, c.tick_size, c.tick_value) for c in built.new_contracts} == {(100.0, 0.25, 25.0)}
    cross = sb.build_shell_structure(
        "x", "custom", ["GZ26", "CLZ26"], [1, -1], None, None, None, "", repo.get_contract)
    sized = {c.symbol: c.multiplier for c in cross.new_contracts}
    assert sized == {"GZ26": 100.0, "CLZ26": 1000.0}  # each leg its own product's size
    typed = sb.build_shell_structure("t", "spread", ["GZ26", "GF27"], [1, -1], 50, 0.5, 25, "", repo.get_contract)
    assert {c.multiplier for c in typed.new_contracts} == {50.0}  # what is typed still wins


def test_an_unsaved_gasoil_leg_is_weighted_by_100_dollars_a_point():
    weights = candidate_dollar_weights([{"symbol": "GZ26", "ratio": 1}, {"symbol": "CLZ26", "ratio": -1}], lambda s: None)
    assert weights == {"GZ26": 100.0, "CLZ26": -1000.0}


def test_the_trade_plan_point_value_scales_with_contract_size():
    params = CurveParams(point_value=1000.0)
    assert point_value_for(params, "CL") == 1000.0 and point_value_for(params, "BRN") == 1000.0 and point_value_for(params, "G") == 100.0
    assert point_value_for(CurveParams(point_value=2000.0), "G") == 200.0


def _frame(changes):
    close = 75.0 + np.cumsum(changes)
    index = pd.bdate_range("2026-01-01", periods=len(close))
    return pd.DataFrame({"close": close, "atr": 1.0}, index=index)


def test_a_cross_product_hedge_ratio_is_in_lots_not_points():
    rng = np.random.default_rng(3)
    brent = rng.normal(0, 1, 120)
    gasoil = 2.0 * brent + rng.normal(0, 0.05, 120)  # gasoil moves two points per Brent point
    same_size = pair_hedge(_frame(gasoil), _frame(brent), "G", "BRN", 100)
    assert same_size.beta_window == pytest.approx(2.0, abs=0.05) and same_size.size_ratio == 1.0
    lots = pair_hedge(_frame(gasoil), _frame(brent), "G", "BRN", 100, size_ratio=100.0 / 1000.0)
    # a gasoil lot is a tenth the size of a Brent lot, so 1 gasoil lot needs 0.2 Brent lots
    assert lots.beta_window == pytest.approx(0.2, abs=0.005) and lots.beta_all == pytest.approx(0.2, abs=0.005)
    assert lots.residual_window == pytest.approx(same_size.residual_window)  # the risk left over does not depend on the lot size
