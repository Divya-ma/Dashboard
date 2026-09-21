"""Tests for core.pnl — the pure P&L calculation engine."""

from datetime import date, datetime, timedelta, timezone

import pytest

from core.models import (
    Contract,
    Leg,
    PnLRecord,
    Structure,
    StructureStatus,
    StructureType,
    Trade,
    TradeEventType,
)
from core.pnl import (
    build_pnl_record,
    calculate_average_entry_price,
    calculate_leg_realized_pnl,
    calculate_leg_unrealized_pnl,
    calculate_margin_efficiency,
    calculate_net_exposure,
    calculate_portfolio_pnl,
    calculate_structure_realized_pnl,
    calculate_structure_unrealized_pnl,
    calculate_todays_pnl,
    calculate_todays_realized_pnl,
)


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


def make_leg(**overrides) -> Leg:
    defaults = dict(contract=make_contract(), ratio=1, lots=10.0, entry_price=75.0)
    defaults.update(overrides)
    return Leg(**defaults)


def make_structure(**overrides) -> Structure:
    defaults = dict(
        name="Test Structure",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[make_leg()],
        status=StructureStatus.OPEN,
    )
    defaults.update(overrides)
    return Structure(**defaults)


# ---------- calculate_leg_unrealized_pnl ----------


def test_leg_unrealized_pnl_long_profit():
    assert calculate_leg_unrealized_pnl(75.0, 76.0, 1, 10, 1000, "buy") == 10_000.0


def test_leg_unrealized_pnl_short_profit():
    assert calculate_leg_unrealized_pnl(75.0, 74.0, -1, 10, 1000, "buy") == 10_000.0


def test_leg_unrealized_pnl_long_loss():
    assert calculate_leg_unrealized_pnl(75.0, 74.0, 1, 10, 1000, "buy") == -10_000.0


def test_leg_unrealized_pnl_zero_ratio_raises():
    with pytest.raises(ValueError):
        calculate_leg_unrealized_pnl(75.0, 76.0, 0, 10, 1000)


def test_leg_unrealized_pnl_negative_lots_raises():
    with pytest.raises(ValueError):
        calculate_leg_unrealized_pnl(75.0, 76.0, 1, -10, 1000)


def test_leg_unrealized_pnl_zero_multiplier_raises():
    with pytest.raises(ValueError):
        calculate_leg_unrealized_pnl(75.0, 76.0, 1, 10, 0)


def test_leg_unrealized_pnl_nan_input_raises():
    with pytest.raises(ValueError):
        calculate_leg_unrealized_pnl(float("nan"), 76.0, 1, 10, 1000)


# Note: (0.45 - 0.37) * 10 lots * 1000 = 800, i.e. the direction tests below use 800 / 500.


def test_sell_trade_profit_when_price_falls():
    assert calculate_leg_unrealized_pnl(0.45, 0.37, 1, 10, 1000, direction="sell") == pytest.approx(800.0)


def test_sell_trade_loss_when_price_rises():
    assert calculate_leg_unrealized_pnl(0.45, 0.50, 1, 10, 1000, direction="sell") == pytest.approx(-500.0)


def test_buy_trade_profit_when_price_rises():
    assert calculate_leg_unrealized_pnl(0.37, 0.45, 1, 10, 1000, direction="buy") == pytest.approx(800.0)


def test_direction_defaults_to_buy_and_is_validated():
    assert calculate_leg_unrealized_pnl(75.0, 76.0, 1, 10, 1000) == calculate_leg_unrealized_pnl(75.0, 76.0, 1, 10, 1000, "buy")
    with pytest.raises(ValueError, match="direction"):
        calculate_leg_unrealized_pnl(75.0, 76.0, 1, 10, 1000, direction="hold")


def test_sell_direction_combines_with_negative_ratio():
    # A sold short leg is a long exposure: falling-price loss flips twice.
    assert calculate_leg_unrealized_pnl(75.0, 74.0, -1, 10, 1000, direction="sell") == pytest.approx(-10_000.0)


def test_leg_realized_pnl_honours_direction():
    assert calculate_leg_realized_pnl(0.45, 0.37, 1, 10, 1000, direction="sell") == pytest.approx(800.0)


def test_structure_unrealized_pnl_reads_direction_from_each_leg():
    sold = make_structure(legs=[make_leg(entry_price=75.0, direction="sell")])
    bought = make_structure(legs=[make_leg(entry_price=75.0, direction="buy")])
    assert calculate_structure_unrealized_pnl(sold.legs, {"CLZ26": 74.0})[0] == pytest.approx(10_000.0)
    assert calculate_structure_unrealized_pnl(bought.legs, {"CLZ26": 74.0})[0] == pytest.approx(-10_000.0)


# ---------- calculate_leg_realized_pnl ----------


def test_leg_realized_pnl_matches_unrealized_formula():
    assert calculate_leg_realized_pnl(75.0, 76.0, 1, 10, 1000, "buy") == 10_000.0


# ---------- calculate_average_entry_price ----------


def test_average_entry_price_equal_lots():
    assert calculate_average_entry_price(10, 75.0, 10, 77.0) == 76.0


def test_average_entry_price_unequal_lots():
    result = calculate_average_entry_price(5, 75.0, 10, 77.0)
    assert result == pytest.approx(76.333333, rel=1e-5)


def test_average_entry_price_zero_new_lots_raises():
    with pytest.raises(ValueError):
        calculate_average_entry_price(10, 75.0, 0, 77.0)


def test_average_entry_price_negative_existing_lots_raises():
    with pytest.raises(ValueError):
        calculate_average_entry_price(-1, 75.0, 10, 77.0)


# ---------- calculate_structure_unrealized_pnl ----------


def test_structure_unrealized_pnl_two_leg_spread():
    front = make_leg(
        contract=make_contract(symbol="CLZ26"), ratio=1, lots=10.0, entry_price=75.0
    )
    back = make_leg(
        contract=make_contract(symbol="CLF27", contract_month=1, contract_year=2027),
        ratio=-1,
        lots=10.0,
        entry_price=76.0,
    )
    total, breakdown, missing = calculate_structure_unrealized_pnl(
        [front, back], {"CLZ26": 76.0, "CLF27": 75.5}
    )
    expected_front = (76.0 - 75.0) * 1 * 10.0 * 1000.0
    expected_back = (75.5 - 76.0) * -1 * 10.0 * 1000.0
    assert total == pytest.approx(expected_front + expected_back)
    assert breakdown[front.leg_id] == pytest.approx(expected_front)
    assert breakdown[back.leg_id] == pytest.approx(expected_back)
    assert missing == []


def test_structure_unrealized_pnl_missing_price_excluded():
    leg1 = make_leg(contract=make_contract(symbol="CLZ26"), entry_price=75.0)
    leg2 = make_leg(
        contract=make_contract(symbol="CLF27", contract_month=1, contract_year=2027),
        entry_price=76.0,
    )
    total, breakdown, missing = calculate_structure_unrealized_pnl(
        [leg1, leg2], {"CLZ26": 76.0}
    )
    assert missing == ["CLF27"]
    assert leg2.leg_id not in breakdown
    assert total == pytest.approx((76.0 - 75.0) * 1 * 10.0 * 1000.0)


def test_structure_unrealized_pnl_no_traded_legs_returns_zero():
    shell_leg = make_leg(lots=0.0, entry_price=None)
    total, breakdown, missing = calculate_structure_unrealized_pnl([shell_leg], {"CLZ26": 76.0})
    assert total == 0.0
    assert breakdown == {}
    assert missing == []


def test_structure_unrealized_pnl_uses_average_entry_price_when_set():
    leg = make_leg(entry_price=75.0, average_entry_price=74.0)
    total, breakdown, _ = calculate_structure_unrealized_pnl([leg], {"CLZ26": 76.0})
    assert total == pytest.approx((76.0 - 74.0) * 1 * 10.0 * 1000.0)


# ---------- calculate_structure_realized_pnl ----------


def make_trade(**overrides) -> Trade:
    defaults = dict(
        structure_id="s1",
        leg_id="l1",
        event_type=TradeEventType.TRADE,
        lots=10.0,
        price=75.0,
        direction="buy",
    )
    defaults.update(overrides)
    return Trade(**defaults)


def test_structure_realized_pnl_no_exit_trades_returns_zero():
    trades = [make_trade(event_type=TradeEventType.TRADE)]
    assert calculate_structure_realized_pnl(trades) == 0.0


def test_structure_realized_pnl_one_partial_exit():
    trades = [make_trade(event_type=TradeEventType.PARTIAL_EXIT, realized_pnl=500.0)]
    assert calculate_structure_realized_pnl(trades) == 500.0


def test_structure_realized_pnl_only_exits_counted():
    trades = [
        make_trade(event_type=TradeEventType.TRADE, realized_pnl=None),
        make_trade(event_type=TradeEventType.ADD, realized_pnl=None),
        make_trade(event_type=TradeEventType.FULL_EXIT, realized_pnl=1000.0),
    ]
    assert calculate_structure_realized_pnl(trades) == 1000.0


# ---------- build_pnl_record ----------


def test_build_pnl_record_is_stale_true():
    leg = make_leg(entry_price=75.0)
    record = build_pnl_record(
        structure_id="s1",
        legs=[leg],
        trades=[],
        live_prices={"CLZ26": 76.0},
        stale_symbols=["CLZ26"],
    )
    assert record.is_stale is True
    assert record.stale_symbols == ["CLZ26"]


def test_build_pnl_record_is_stale_false():
    leg = make_leg(entry_price=75.0)
    record = build_pnl_record(
        structure_id="s1",
        legs=[leg],
        trades=[],
        live_prices={"CLZ26": 76.0},
        stale_symbols=[],
    )
    assert record.is_stale is False


def test_build_pnl_record_total_matches_sum():
    leg = make_leg(entry_price=75.0)
    trades = [make_trade(event_type=TradeEventType.PARTIAL_EXIT, realized_pnl=200.0)]
    record = build_pnl_record(
        structure_id="s1",
        legs=[leg],
        trades=trades,
        live_prices={"CLZ26": 76.0},
        stale_symbols=[],
    )
    assert record.total_pnl == pytest.approx(record.unrealized_pnl + record.realized_pnl)
    assert isinstance(record, PnLRecord)


# ---------- calculate_portfolio_pnl ----------


def test_portfolio_pnl_single_open_structure():
    structure = make_structure(legs=[make_leg(entry_price=75.0)])
    result = calculate_portfolio_pnl(
        structures=[structure],
        trades_by_structure={},
        live_prices={"CLZ26": 76.0},
        stale_symbols=[],
    )
    expected_unrealized = (76.0 - 75.0) * 1 * 10.0 * 1000.0
    assert result["total_unrealized"] == pytest.approx(expected_unrealized)
    assert result["total_realized"] == 0.0
    assert result["total_pnl"] == pytest.approx(expected_unrealized)
    assert result["open_structure_count"] == 1
    assert result["open_leg_count"] == 1
    assert result["net_lots_by_product"] == {"CL": 10.0}
    assert result["largest_winner"]["structure_id"] == structure.structure_id
    assert result["largest_loser"]["structure_id"] == structure.structure_id
    assert result["has_stale_data"] is False
    assert result["stale_symbols"] == []
    assert isinstance(result["calculated_at"], datetime)
    assert structure.structure_id in result["per_structure"]


def test_portfolio_pnl_empty_structures():
    result = calculate_portfolio_pnl(
        structures=[], trades_by_structure={}, live_prices={}, stale_symbols=[]
    )
    assert result["total_unrealized"] == 0.0
    assert result["total_realized"] == 0.0
    assert result["total_pnl"] == 0.0
    assert result["open_structure_count"] == 0
    assert result["open_leg_count"] == 0
    assert result["net_lots_by_product"] == {}
    assert result["largest_winner"] is None
    assert result["largest_loser"] is None


def test_portfolio_pnl_excludes_shell_and_closed_structures():
    shell = make_structure(status=StructureStatus.SHELL)
    closed = make_structure(status=StructureStatus.CLOSED)
    result = calculate_portfolio_pnl(
        structures=[shell, closed],
        trades_by_structure={},
        live_prices={"CLZ26": 76.0},
        stale_symbols=[],
    )
    assert result["open_structure_count"] == 0


def closed_with_exit(pnl: float, when: datetime | None = None):
    structure = make_structure(status=StructureStatus.CLOSED)
    trade = Trade(
        structure_id=structure.structure_id,
        leg_id=structure.legs[0].leg_id,
        event_type=TradeEventType.FULL_EXIT,
        lots=10,
        price=77.0,
        direction="sell",
        realized_pnl=pnl,
        **({"timestamp": when} if when else {}),
    )
    return structure, trade


def test_portfolio_realized_keeps_closed_structures_in_all_time_total():
    open_structure = make_structure()
    closed, exit_trade = closed_with_exit(2500.0)
    result = calculate_portfolio_pnl(
        structures=[open_structure],
        trades_by_structure={},
        live_prices={"CLZ26": 76.0},
        stale_symbols=[],
        closed_structures=[closed],
        closed_trades_by_structure={closed.structure_id: [exit_trade]},
    )
    assert result["total_realized"] == 0.0  # open structures only
    assert result["total_realized_all_time"] == pytest.approx(2500.0)
    assert result["total_unrealized"] == pytest.approx(10_000.0)
    assert result["open_structure_count"] == 1 and len(result["per_structure"]) == 1


def test_portfolio_realized_all_time_sums_open_and_closed():
    open_structure = make_structure()
    partial = Trade(
        structure_id=open_structure.structure_id, leg_id=open_structure.legs[0].leg_id,
        event_type=TradeEventType.PARTIAL_EXIT, lots=1, price=76.0, direction="sell", realized_pnl=400.0,
    )
    closed, exit_trade = closed_with_exit(-100.0)
    result = calculate_portfolio_pnl(
        [open_structure], {open_structure.structure_id: [partial]}, {"CLZ26": 75.0}, [],
        [closed], {closed.structure_id: [exit_trade]},
    )
    assert result["total_realized"] == pytest.approx(400.0)
    assert result["total_realized_all_time"] == pytest.approx(300.0)


def test_portfolio_without_closed_arguments_is_unchanged():
    result = calculate_portfolio_pnl([make_structure()], {}, {"CLZ26": 76.0}, [])
    assert result["total_realized_all_time"] == result["total_realized"] == 0.0


# ---------- calculate_todays_realized_pnl ----------

_NOW = datetime(2026, 9, 21, 15, 0, tzinfo=timezone.utc)


def test_todays_realized_no_exits_is_zero():
    assert calculate_todays_realized_pnl([], _NOW.date()) == 0.0


def test_todays_realized_sums_todays_full_exit():
    _, exit_trade = closed_with_exit(1200.0, _NOW)
    assert calculate_todays_realized_pnl([exit_trade], _NOW.date()) == pytest.approx(1200.0)


def test_todays_realized_excludes_yesterday():
    _, exit_trade = closed_with_exit(1200.0, _NOW - timedelta(days=1))
    assert calculate_todays_realized_pnl([exit_trade], _NOW.date()) == 0.0


def test_todays_realized_mix_only_counts_today_and_ignores_non_exits():
    _, today_a = closed_with_exit(1000.0, _NOW)
    _, today_b = closed_with_exit(-300.0, _NOW - timedelta(hours=2))
    _, yesterday = closed_with_exit(5000.0, _NOW - timedelta(days=1))
    entry = Trade(structure_id="s", leg_id="l", event_type=TradeEventType.TRADE, lots=1, price=1.0, direction="buy", timestamp=_NOW)
    assert calculate_todays_realized_pnl([today_a, today_b, yesterday, entry], _NOW.date()) == pytest.approx(700.0)


def test_todays_realized_defaults_to_today_utc():
    _, exit_trade = closed_with_exit(50.0)
    assert calculate_todays_realized_pnl([exit_trade]) == pytest.approx(50.0)


def test_portfolio_pnl_net_lots_by_product_spread_nets_to_zero():
    front = make_leg(contract=make_contract(symbol="CLZ26"), ratio=1, lots=10.0, entry_price=75.0)
    back = make_leg(
        contract=make_contract(symbol="CLF27", contract_month=1, contract_year=2027),
        ratio=-1,
        lots=10.0,
        entry_price=76.0,
    )
    structure = make_structure(structure_type=StructureType.SPREAD, legs=[front, back])
    result = calculate_portfolio_pnl(
        structures=[structure],
        trades_by_structure={},
        live_prices={"CLZ26": 76.0, "CLF27": 75.5},
        stale_symbols=[],
    )
    assert result["net_lots_by_product"]["CL"] == 0.0


def test_portfolio_pnl_largest_winner_and_loser():
    winner = make_structure(
        name="Winner",
        legs=[make_leg(contract=make_contract(symbol="CLZ26"), entry_price=75.0, lots=10.0)],
    )
    loser = make_structure(
        name="Loser",
        legs=[
            make_leg(
                contract=make_contract(symbol="CLF27", contract_month=1, contract_year=2027),
                entry_price=80.0,
                lots=10.0,
            )
        ],
    )
    result = calculate_portfolio_pnl(
        structures=[winner, loser],
        trades_by_structure={},
        live_prices={"CLZ26": 76.0, "CLF27": 75.0},
        stale_symbols=[],
    )
    assert result["largest_winner"]["structure_id"] == winner.structure_id
    assert result["largest_loser"]["structure_id"] == loser.structure_id


# ---------- calculate_todays_pnl ----------


def make_pnl_record(total_pnl: float, timestamp: datetime) -> PnLRecord:
    return PnLRecord(
        structure_id="s1",
        unrealized_pnl=total_pnl,
        realized_pnl=0.0,
        total_pnl=total_pnl,
        timestamp=timestamp,
        last_price_used={},
    )


def test_todays_pnl_fewer_than_two_records_returns_zero():
    today = datetime.now(timezone.utc)
    records = [make_pnl_record(100.0, today)]
    assert calculate_todays_pnl(records) == 0.0


def test_todays_pnl_computes_difference():
    today = datetime.now(timezone.utc)
    records = [
        make_pnl_record(100.0, today.replace(hour=1, minute=0, second=0, microsecond=0)),
        make_pnl_record(300.0, today.replace(hour=10, minute=0, second=0, microsecond=0)),
    ]
    assert calculate_todays_pnl(records, reference_date=today.date()) == 200.0


def test_todays_pnl_ignores_records_from_other_days():
    today = datetime.now(timezone.utc)
    yesterday = today - timedelta(days=1)
    records = [
        make_pnl_record(50.0, yesterday),
        make_pnl_record(999.0, yesterday),
        make_pnl_record(100.0, today),
    ]
    assert calculate_todays_pnl(records, reference_date=today.date()) == 0.0


# ---------- calculate_margin_efficiency ----------


def test_margin_efficiency_basic():
    assert calculate_margin_efficiency(5000.0, 10000.0) == 0.5


def test_margin_efficiency_zero_margin_returns_none():
    assert calculate_margin_efficiency(5000.0, 0.0) is None


# ---------- calculate_net_exposure ----------


def test_net_exposure_single_spread():
    front = make_leg(contract=make_contract(symbol="CLZ26"), ratio=1, lots=10.0, entry_price=75.0)
    back = make_leg(
        contract=make_contract(symbol="CLF27", contract_month=1, contract_year=2027),
        ratio=-1,
        lots=3.0,
        entry_price=76.0,
    )
    structure = make_structure(structure_type=StructureType.SPREAD, legs=[front, back])
    exposure = calculate_net_exposure([structure])
    assert exposure == {"CL": {"CLZ26": 10.0, "CLF27": -3.0}}


def test_net_exposure_accumulates_across_structures():
    leg_a = make_leg(contract=make_contract(symbol="CLZ26"), ratio=1, lots=5.0, entry_price=75.0)
    leg_b = make_leg(contract=make_contract(symbol="CLZ26"), ratio=1, lots=3.0, entry_price=76.0)
    structure_a = make_structure(name="A", legs=[leg_a])
    structure_b = make_structure(name="B", legs=[leg_b])
    exposure = calculate_net_exposure([structure_a, structure_b])
    assert exposure == {"CL": {"CLZ26": 8.0}}


def test_net_exposure_excludes_untraded_legs():
    traded = make_leg(contract=make_contract(symbol="CLZ26"), ratio=1, lots=5.0, entry_price=75.0)
    untraded = make_leg(
        contract=make_contract(symbol="CLF27", contract_month=1, contract_year=2027),
        ratio=1,
        lots=0.0,
        entry_price=None,
    )
    structure = make_structure(legs=[traded, untraded])
    exposure = calculate_net_exposure([structure])
    assert exposure == {"CL": {"CLZ26": 5.0}}
