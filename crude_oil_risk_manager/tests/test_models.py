"""Tests for core.models: validators and computed properties."""

import pytest
from pydantic import ValidationError

from core.models import (
    Alert,
    AlertLevel,
    Contract,
    Leg,
    PnLRecord,
    Structure,
    StructureStatus,
    StructureType,
    Trade,
    TradeEventType,
)


def make_contract(**overrides) -> Contract:
    defaults = dict(
        product="CL",
        contract_month=12,
        contract_year=2025,
        symbol="CLZ25",
        multiplier=1000.0,
        tick_size=0.01,
        tick_value=10.0,
    )
    defaults.update(overrides)
    return Contract(**defaults)


def make_leg(**overrides) -> Leg:
    defaults = dict(contract=make_contract(), ratio=1)
    defaults.update(overrides)
    return Leg(**defaults)


# ---------- Contract ----------


def test_contract_valid():
    c = make_contract()
    assert c.product == "CL"


def test_contract_invalid_product_raises():
    with pytest.raises(ValidationError):
        make_contract(product="XYZ")


def test_contract_invalid_month_zero_raises():
    with pytest.raises(ValidationError):
        make_contract(contract_month=0)


def test_contract_invalid_month_thirteen_raises():
    with pytest.raises(ValidationError):
        make_contract(contract_month=13)


def test_contract_year_out_of_range_raises():
    with pytest.raises(ValidationError):
        make_contract(contract_year=2019)
    with pytest.raises(ValidationError):
        make_contract(contract_year=2051)


def test_contract_multiplier_not_positive_raises():
    with pytest.raises(ValidationError):
        make_contract(multiplier=0)
    with pytest.raises(ValidationError):
        make_contract(multiplier=-5)


def test_contract_tick_size_not_positive_raises():
    with pytest.raises(ValidationError):
        make_contract(tick_size=0)


def test_contract_tick_value_not_positive_raises():
    with pytest.raises(ValidationError):
        make_contract(tick_value=0)


def test_contract_display_name():
    c = make_contract(product="CL", contract_month=12, contract_year=2025, symbol="CLZ25")
    assert c.display_name == "CL Dec 2025 (CLZ25)"


def test_contract_from_symbol_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        Contract.from_symbol("CLZ25")


# ---------- Leg ----------


def test_leg_ratio_zero_raises():
    with pytest.raises(ValidationError):
        make_leg(ratio=0)


def test_leg_negative_lots_raises():
    with pytest.raises(ValidationError):
        make_leg(lots=-1.0)


def test_leg_is_traded_false_when_shell():
    leg = make_leg(lots=0.0, entry_price=None)
    assert leg.is_traded is False
    assert leg.notional_value is None


def test_leg_is_traded_true_when_traded():
    leg = make_leg(lots=10.0, entry_price=75.5)
    assert leg.is_traded is True
    assert leg.notional_value == 10.0 * 1000.0 * 75.5


# ---------- Structure ----------


def test_structure_empty_legs_raises():
    with pytest.raises(ValidationError):
        Structure(
            name="Empty",
            structure_type=StructureType.OUTRIGHT,
            products=["CL"],
            legs=[],
        )


def test_structure_leg_with_zero_ratio_raises():
    contract = make_contract()
    with pytest.raises(ValidationError):
        Structure(
            name="Bad leg",
            structure_type=StructureType.OUTRIGHT,
            products=["CL"],
            legs=[Leg(contract=contract, ratio=1).model_copy(update={"ratio": 0})],
        )


def test_structure_days_held_none_for_shell():
    structure = Structure(
        name="Shell Structure",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[make_leg()],
        status=StructureStatus.SHELL,
    )
    assert structure.days_held is None


def test_structure_days_held_int_when_open():
    structure = Structure(
        name="Open Structure",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[make_leg()],
        status=StructureStatus.OPEN,
    )
    assert isinstance(structure.days_held, int)
    assert structure.days_held >= 0


def test_structure_is_cross_product_single():
    structure = Structure(
        name="Single Product",
        structure_type=StructureType.SPREAD,
        products=["CL"],
        legs=[
            make_leg(contract=make_contract(symbol="CLZ25"), ratio=1),
            make_leg(contract=make_contract(symbol="CLF26", contract_month=1, contract_year=2026), ratio=-1),
        ],
    )
    assert structure.is_cross_product is False


def test_structure_is_cross_product_multi():
    structure = Structure(
        name="Cross Product",
        structure_type=StructureType.SPREAD,
        products=["CL", "BRN"],
        legs=[
            make_leg(contract=make_contract(product="CL", symbol="CLZ25"), ratio=1),
            make_leg(
                contract=make_contract(
                    product="BRN", symbol="BRNF26", contract_month=1, contract_year=2026
                ),
                ratio=-1,
            ),
        ],
    )
    assert structure.is_cross_product is True


def test_structure_duplicate_tenors_flagged():
    structure = Structure(
        name="Duplicate Tenor",
        structure_type=StructureType.CUSTOM,
        products=["CL"],
        legs=[
            make_leg(contract=make_contract(symbol="CLZ25"), ratio=1),
            make_leg(contract=make_contract(symbol="CLZ25"), ratio=-1),
        ],
    )
    assert structure.has_duplicate_tenors is True


def test_structure_net_lots():
    structure = Structure(
        name="Net Lots",
        structure_type=StructureType.SPREAD,
        products=["CL"],
        legs=[
            make_leg(ratio=1, lots=10.0),
            make_leg(ratio=-1, lots=10.0),
        ],
    )
    assert structure.net_lots == 0.0


def test_structure_leg_count():
    structure = Structure(
        name="Leg Count",
        structure_type=StructureType.FLY,
        products=["CL"],
        legs=[make_leg(ratio=1), make_leg(ratio=-2), make_leg(ratio=1)],
    )
    assert structure.leg_count == 3


def test_structure_name_too_long_raises():
    with pytest.raises(ValidationError):
        Structure(
            name="x" * 101,
            structure_type=StructureType.OUTRIGHT,
            products=["CL"],
            legs=[make_leg()],
        )


# ---------- Trade ----------


def test_trade_invalid_direction_raises():
    with pytest.raises(ValidationError):
        Trade(
            structure_id="s1",
            leg_id="l1",
            event_type=TradeEventType.TRADE,
            lots=10.0,
            price=75.0,
            direction="hold",
        )


def test_trade_valid_direction():
    trade = Trade(
        structure_id="s1",
        leg_id="l1",
        event_type=TradeEventType.TRADE,
        lots=10.0,
        price=75.0,
        direction="buy",
    )
    assert trade.direction == "buy"


def test_trade_lots_not_positive_raises():
    with pytest.raises(ValidationError):
        Trade(
            structure_id="s1",
            leg_id="l1",
            event_type=TradeEventType.TRADE,
            lots=0,
            price=75.0,
            direction="buy",
        )


# ---------- PnLRecord ----------


def test_pnlrecord_total_mismatch_raises():
    with pytest.raises(ValidationError):
        PnLRecord(
            structure_id="s1",
            unrealized_pnl=100.0,
            realized_pnl=50.0,
            total_pnl=200.0,
        )


def test_pnlrecord_total_match_ok():
    record = PnLRecord(
        structure_id="s1",
        unrealized_pnl=100.0,
        realized_pnl=50.0,
        total_pnl=150.0,
    )
    assert record.total_pnl == 150.0


# ---------- Alert ----------


def test_alert_valid():
    alert = Alert(level=AlertLevel.WARNING, title="Loss limit", body="Structure exceeded loss limit.")
    assert alert.level == AlertLevel.WARNING
    assert alert.acknowledged is False


def test_alert_title_too_long_raises():
    with pytest.raises(ValidationError):
        Alert(level=AlertLevel.INFO, title="x" * 101, body="body")
