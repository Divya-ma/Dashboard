"""Contract calendar: expiries, buffer-day rolls and the 15-month structure ladder."""

from datetime import date

import pytest

from core.curve_calendar import (
    DFLY, FLY, OUTRIGHT, SPREAD,
    add_business_days, buffer_day, buffer_rolls, build_structures, contract_expiry,
    front_month, front_symbol_by_expiry, generic_contracts, is_business_day, live_symbols, parse_contract,
)


def test_cl_expiry_follows_the_three_business_days_before_the_25th_rule():
    assert contract_expiry("CL", 11, 2026) == date(2026, 10, 20)  # 25 Oct is a Sunday -> Fri 23 -> -3bd
    assert contract_expiry("CL", 12, 2026) == date(2026, 11, 20)
    assert contract_expiry("CL", 1, 2027) == date(2026, 12, 21)  # 25 Dec holiday -> Thu 24 -> -3bd


def test_brent_expiry_is_last_business_day_of_second_preceding_month():
    assert contract_expiry("BRN", 12, 2026) == date(2026, 10, 30)  # matches the API's own expiry for BRN Dec26
    assert contract_expiry("BRN", 1, 2027) == date(2026, 11, 30)
    assert contract_expiry("BRN", 3, 2027) == date(2027, 1, 29)  # 31 Jan 2027 is a Sunday


def test_buffer_day_is_one_business_day_before_expiry():
    assert buffer_day("BRN", 12, 2026) == date(2026, 10, 29)
    assert buffer_day("CL", 11, 2026) == date(2026, 10, 19)


def test_holidays_are_not_business_days():
    assert not is_business_day(date(2026, 12, 25), "nymex")
    assert not is_business_day(date(2026, 4, 3), "nymex")  # Good Friday
    assert not is_business_day(date(2026, 12, 28), "ice_uk")  # Boxing Day substitute (26th is a Saturday)
    assert is_business_day(date(2026, 7, 3), "ice_uk")
    assert add_business_days(date(2026, 10, 23), -3, "nymex") == date(2026, 10, 20)


def test_front_month_matches_the_exchange_generic_mapping():
    assert front_month("BRN", date(2026, 10, 6)) == (12, 2026)  # CO1 was COZ26
    assert front_month("CL", date(2026, 10, 6)) == (11, 2026)  # CL1 was CLX26


def test_front_rolls_on_the_buffer_day_not_the_expiry_day():
    assert front_month("BRN", date(2026, 10, 28)) == (12, 2026)
    assert front_month("BRN", date(2026, 10, 29)) == (1, 2027)  # buffer day: Dec already rolled out


def test_buffer_rolls_flags_only_the_buffer_day():
    assert buffer_rolls("BRN", date(2026, 10, 28)) == []
    rolled = buffer_rolls("BRN", date(2026, 10, 29))
    assert [c.symbol for c in rolled] == ["COZ26"] and rolled[0].expiry == date(2026, 10, 30)
    assert buffer_rolls("BRN", date(2026, 10, 30)) == []


def test_generic_contracts_are_consecutive_months_across_the_year_end():
    contracts = generic_contracts("BRN", date(2026, 10, 6))
    assert [c.symbol for c in contracts][:5] == ["COZ26", "COF27", "COG27", "COH27", "COJ27"]
    assert len(contracts) == 15 and contracts[-1].symbol == "COG28"
    assert [c.position for c in contracts] == list(range(1, 16))


def test_structure_ladder_sizes_and_symbols():
    ladder = build_structures("BRN", date(2026, 10, 6))
    assert {f: len(v) for f, v in ladder.items()} == {OUTRIGHT: 15, SPREAD: 14, FLY: 13, DFLY: 12}
    spread, fly, dfly = ladder[SPREAD][0], ladder[FLY][0], ladder[DFLY][0]
    assert spread.symbol == "COZ26-F27" and spread.generic_code == "CO1-2" and spread.leg_weights == (1, -1)
    assert fly.symbol == "COZ26-F27-G27" and fly.generic_code == "CO1-2-3" and fly.leg_weights == (1, -2, 1)
    assert dfly.symbol == "" and dfly.generic_code == ""
    assert dfly.components == ("COZ26-F27-G27", "COF27-G27-H27") and dfly.component_weights == (1, -1)
    assert dfly.leg_weights == (1, -3, 3, -1) and dfly.front_month == 12


def test_live_symbols_cover_outrights_spreads_and_flies_only():
    symbols = live_symbols("CL", date(2026, 10, 6))
    assert len(symbols) == 15 + 14 + 13 and len(set(symbols)) == len(symbols)
    assert symbols[0] == "CLX26" and "CLX26-Z26" in symbols and "CLX26-Z26-F27" in symbols


def test_parse_contract_reads_the_front_leg():
    assert parse_contract("COZ26-F27-G27") == ("CO", 12, 2026)


def test_dfly_leg_weights_equal_fly_minus_next_fly():
    prices = [100.0, 99.0, 97.5, 95.0]
    fly1 = prices[0] - 2 * prices[1] + prices[2]
    fly2 = prices[1] - 2 * prices[2] + prices[3]
    dfly = sum(w * p for w, p in zip((1, -3, 3, -1), prices))
    assert dfly == pytest.approx(fly1 - fly2)


def test_front_symbol_by_expiry_is_live_through_the_expiry_day():
    assert front_symbol_by_expiry("BRN", date(2026, 10, 30)) == "COZ26"
    assert front_symbol_by_expiry("BRN", date(2026, 11, 2)) == "COF27"
