"""Tests for core.exposure."""

from datetime import datetime

import pytest

from core.exposure import calculate_dollar_exposure, get_exposure_summary
from core.models import Contract, Leg, Structure, StructureStatus, StructureType


def make_contract(**overrides) -> Contract:
    defaults = dict(
        product="CL",
        contract_month=12,
        contract_year=2026,
        symbol="CLZ26",
        multiplier=1000.0,
        tick_size=0.01,
        tick_value=10.0,
    )
    defaults.update(overrides)
    return Contract(**defaults)


def make_structure(contract: Contract, lots: float, status=StructureStatus.OPEN) -> Structure:
    leg = Leg(contract=contract, ratio=1, lots=lots, entry_price=75.0)
    return Structure(
        name=f"{contract.symbol} outright",
        structure_type=StructureType.OUTRIGHT,
        products=[contract.product],
        legs=[leg],
        status=status,
    )


CL_Z26 = make_contract()
BRN_Z26 = make_contract(product="BRN", symbol="BRNZ26")


# ---------- calculate_dollar_exposure ----------


def test_dollar_exposure_arithmetic():
    result = calculate_dollar_exposure(
        {"CL": {"CLZ26": 5.0}}, {"CLZ26": 75.0}, {"CLZ26": CL_Z26}
    )
    entry = result["CL"]["CLZ26"]
    assert entry["net_lots"] == 5.0
    assert entry["price"] == 75.0
    assert entry["dollar_exposure"] == 5.0 * 75.0 * 1000.0
    assert entry["price_is_stale"] is False


def test_dollar_exposure_short_position_is_negative():
    result = calculate_dollar_exposure(
        {"CL": {"CLZ26": -2.0}}, {"CLZ26": 75.0}, {"CLZ26": CL_Z26}
    )
    assert result["CL"]["CLZ26"]["dollar_exposure"] == -150_000.0


def test_dollar_exposure_missing_price_gives_none_not_zero():
    result = calculate_dollar_exposure({"CL": {"CLZ26": 5.0}}, {}, {"CLZ26": CL_Z26})
    entry = result["CL"]["CLZ26"]
    assert entry["price"] is None
    assert entry["dollar_exposure"] is None
    assert entry["price_is_stale"] is True
    assert entry["net_lots"] == 5.0


def test_dollar_exposure_nan_price_treated_as_missing():
    result = calculate_dollar_exposure(
        {"CL": {"CLZ26": 5.0}}, {"CLZ26": float("nan")}, {"CLZ26": CL_Z26}
    )
    assert result["CL"]["CLZ26"]["dollar_exposure"] is None
    assert result["CL"]["CLZ26"]["price_is_stale"] is True


def test_dollar_per_bp_cl_known_example():
    # CL: tick_size 0.01, tick_value $10 -> 1bp (0.01) is one tick = $10 per lot.
    result = calculate_dollar_exposure(
        {"CL": {"CLZ26": 5.0}}, {"CLZ26": 75.0}, {"CLZ26": CL_Z26}
    )
    assert result["CL"]["CLZ26"]["dollar_per_bp"] == 50.0


def test_dollar_per_bp_uses_tick_size_ratio():
    contract = make_contract(tick_size=0.05, tick_value=12.5)
    result = calculate_dollar_exposure({"CL": {"CLZ26": 4.0}}, {"CLZ26": 75.0}, {"CLZ26": contract})
    # 4 lots * 12.5 * (0.01 / 0.05) = 10.0
    assert result["CL"]["CLZ26"]["dollar_per_bp"] == pytest.approx(10.0)


def test_dollar_per_bp_available_when_price_missing():
    result = calculate_dollar_exposure({"CL": {"CLZ26": 5.0}}, {}, {"CLZ26": CL_Z26})
    assert result["CL"]["CLZ26"]["dollar_per_bp"] == 50.0


def test_dollar_exposure_missing_contract_gives_none_values():
    result = calculate_dollar_exposure({"CL": {"CLZ26": 5.0}}, {"CLZ26": 75.0}, {})
    entry = result["CL"]["CLZ26"]
    assert entry["dollar_exposure"] is None
    assert entry["dollar_per_bp"] is None


# ---------- get_exposure_summary ----------


def test_exposure_summary_has_all_required_keys():
    summary = get_exposure_summary(
        [make_structure(CL_Z26, 5.0)], {"CLZ26": 75.0}, {"CLZ26": CL_Z26}
    )
    assert set(summary) == {
        "by_product",
        "totals_by_product",
        "grand_total_dollar_exposure",
        "has_stale_prices",
        "calculated_at",
    }
    assert set(summary["by_product"]["CL"]["CLZ26"]) == {
        "net_lots",
        "dollar_exposure",
        "dollar_per_bp",
        "price_is_stale",
    }
    assert set(summary["totals_by_product"]["CL"]) == {"net_lots", "dollar_exposure"}
    assert summary["grand_total_dollar_exposure"] == 375_000.0
    assert summary["has_stale_prices"] is False
    assert isinstance(summary["calculated_at"], datetime)


def test_exposure_summary_grand_total_none_if_any_component_none():
    structures = [make_structure(CL_Z26, 5.0), make_structure(BRN_Z26, 3.0)]
    summary = get_exposure_summary(
        structures,
        {"CLZ26": 75.0},  # BRNZ26 price missing
        {"CLZ26": CL_Z26, "BRNZ26": BRN_Z26},
    )
    assert summary["totals_by_product"]["CL"]["dollar_exposure"] == 375_000.0
    assert summary["totals_by_product"]["BRN"]["dollar_exposure"] is None
    assert summary["grand_total_dollar_exposure"] is None
    assert summary["has_stale_prices"] is True


def test_exposure_summary_sums_across_products():
    structures = [make_structure(CL_Z26, 5.0), make_structure(BRN_Z26, 2.0)]
    summary = get_exposure_summary(
        structures,
        {"CLZ26": 75.0, "BRNZ26": 80.0},
        {"CLZ26": CL_Z26, "BRNZ26": BRN_Z26},
    )
    assert summary["grand_total_dollar_exposure"] == 375_000.0 + 160_000.0
    assert summary["totals_by_product"]["BRN"]["net_lots"] == 2.0


def test_exposure_summary_excludes_closed_structures():
    closed = make_structure(CL_Z26, 5.0, status=StructureStatus.CLOSED)
    summary = get_exposure_summary([closed], {"CLZ26": 75.0}, {"CLZ26": CL_Z26})
    assert summary["by_product"] == {}
    assert summary["grand_total_dollar_exposure"] == 0.0


def test_exposure_summary_empty_portfolio():
    summary = get_exposure_summary([], {}, {})
    assert summary["by_product"] == {}
    assert summary["grand_total_dollar_exposure"] == 0.0
    assert summary["has_stale_prices"] is False
