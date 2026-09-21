"""Tests for core.var."""

import numpy as np
import pytest

from adapters.mock.mock_historical import MockHistoricalAdapter
from core.data_loader import DataLoader
from core.exceptions import DataNotAvailableError
from core.exposure import get_exposure_summary
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from core.var import (
    calculate_historical_var,
    calculate_portfolio_var,
    get_var_term_structure,
)


def make_contract(symbol: str, product: str = "CL", month: int = 12, year: int = 2026) -> Contract:
    return Contract(
        product=product,
        contract_month=month,
        contract_year=year,
        symbol=symbol,
        multiplier=1000.0,
        tick_size=0.01,
        tick_value=10.0,
    )


CLZ26 = make_contract("CLZ26")
CLF27 = make_contract("CLF27", month=1, year=2027)
BRNZ26 = make_contract("BRNZ26", product="BRN")
CONTRACTS = {c.symbol: c for c in (CLZ26, CLF27, BRNZ26)}


def leg(contract: Contract, ratio: int, lots: float) -> Leg:
    return Leg(contract=contract, ratio=ratio, lots=lots, entry_price=75.0)


def structure(name: str, legs: list[Leg], stype=StructureType.OUTRIGHT) -> Structure:
    return Structure(
        name=name,
        structure_type=stype,
        products=sorted({lg.contract.product for lg in legs}),
        legs=legs,
        status=StructureStatus.OPEN,
    )


@pytest.fixture
def loader(tmp_path):
    return DataLoader(str(tmp_path), MockHistoricalAdapter())


@pytest.fixture
def market(tmp_path, create_synthetic_parquet):
    """300 days of synthetic data for CLZ26 and CLF27."""
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=300, seed=11)
    create_synthetic_parquet(tmp_path, "CLF27", n_days=300, seed=12)
    return tmp_path


PRICES = {"CLZ26": 75.0, "CLF27": 74.0, "BRNZ26": 80.0}


# ---------- calculate_historical_var ----------


def test_var_is_positive(market, loader):
    result = calculate_historical_var("CLZ26", 10.0, 1000.0, 0.95, 1, loader)
    assert result["var_1day"] > 0


def test_var_positive_for_short_position(market, loader):
    result = calculate_historical_var("CLZ26", -10.0, 1000.0, 0.95, 1, loader)
    assert result["var_1day"] > 0


def test_var_99_at_least_var_95(market, loader):
    var_95 = calculate_historical_var("CLZ26", 10.0, 1000.0, 0.95, 1, loader)["var_1day"]
    var_99 = calculate_historical_var("CLZ26", 10.0, 1000.0, 0.99, 1, loader)["var_1day"]
    assert var_99 >= var_95


def test_var_5day_exceeds_1day_and_uses_sqrt_scaling(market, loader):
    result = calculate_historical_var("CLZ26", 10.0, 1000.0, 0.95, 5, loader)
    assert result["var_Nday"] > result["var_1day"]
    assert result["var_Nday"] == pytest.approx(result["var_1day"] * np.sqrt(5))


def test_var_data_warning_present_below_252_scenarios(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=100, seed=11)
    result = calculate_historical_var("CLZ26", 10.0, 1000.0, 0.95, 1, loader)
    assert result["scenarios_used"] == 99
    assert "Only 99 scenarios available" in result["data_warning"]


def test_var_no_data_warning_with_full_history(market, loader):
    result = calculate_historical_var("CLZ26", 10.0, 1000.0, 0.95, 1, loader)
    assert result["data_warning"] is None
    assert result["scenarios_used"] == 299


def test_var_zero_net_lots_returns_zero(market, loader):
    result = calculate_historical_var("CLZ26", 0.0, 1000.0, 0.95, 5, loader)
    assert result["var_1day"] == 0.0
    assert result["var_Nday"] == 0.0


def test_var_result_matches_manual_percentile(market, loader):
    result = calculate_historical_var("CLZ26", 10.0, 1000.0, 0.95, 1, loader)
    pnl = loader.load_price_differences("CLZ26").to_numpy() * 10.0 * 1000.0
    assert result["var_1day"] == pytest.approx(-np.percentile(pnl, 5.0))
    assert result["percentile_cutoff"] == pytest.approx(np.percentile(pnl, 5.0))
    assert result["min_scenario_pnl"] == pytest.approx(pnl.min())
    assert result["max_scenario_pnl"] == pytest.approx(pnl.max())
    assert result["mean_scenario_pnl"] == pytest.approx(pnl.mean())


def test_var_invalid_inputs_raise(market, loader):
    with pytest.raises(ValueError):
        calculate_historical_var("CLZ26", 10.0, 1000.0, 1.5, 1, loader)
    with pytest.raises(ValueError):
        calculate_historical_var("CLZ26", 10.0, 1000.0, 0.95, 0, loader)
    with pytest.raises(ValueError):
        calculate_historical_var("CLZ26", 10.0, 0.0, 0.95, 1, loader)


# ---------- calculate_portfolio_var ----------


def build_portfolio():
    outright = structure("CLZ26 outright", [leg(CLZ26, 1, 10.0)])
    spread = structure(
        "CLZ26-F27 spread", [leg(CLZ26, 1, 5.0), leg(CLF27, -1, 5.0)], StructureType.SPREAD
    )
    return [outright, spread]


def test_portfolio_var_has_all_required_keys(market, loader):
    structures = build_portfolio()
    summary = get_exposure_summary(structures, PRICES, CONTRACTS)
    result = calculate_portfolio_var(structures, summary, 0.95, 1, loader, CONTRACTS)
    assert set(result) == {
        "portfolio_var_1day",
        "portfolio_var_Nday",
        "confidence",
        "horizon_days",
        "scenarios_used",
        "common_dates_used",
        "per_structure_var",
        "marginal_var",
        "marginal_var_note",
        "data_warnings",
        "calculated_at",
    }
    assert result["portfolio_var_1day"] > 0
    assert result["common_dates_used"] == 299


def test_portfolio_var_per_structure_keys_match_structure_ids(market, loader):
    structures = build_portfolio()
    summary = get_exposure_summary(structures, PRICES, CONTRACTS)
    result = calculate_portfolio_var(structures, summary, 0.95, 1, loader, CONTRACTS)
    assert set(result["per_structure_var"]) == {s.structure_id for s in structures}
    entry = result["per_structure_var"][structures[1].structure_id]
    assert set(entry) == {
        "standalone_var_1day",
        "standalone_var_Nday",
        "symbols_included",
        "data_warning",
    }
    assert set(entry["symbols_included"]) == {"CLZ26", "CLF27"}
    assert entry["standalone_var_1day"] > 0


def test_portfolio_var_marginal_is_none_with_note(market, loader):
    structures = build_portfolio()
    summary = get_exposure_summary(structures, PRICES, CONTRACTS)
    result = calculate_portfolio_var(structures, summary, 0.95, 1, loader, CONTRACTS)
    assert result["marginal_var"] is None
    assert "Marginal VaR" in result["marginal_var_note"]
    assert result["marginal_var_note"].startswith("TODO")


def test_portfolio_var_matches_manual_combined_percentile(market, loader):
    structures = build_portfolio()
    summary = get_exposure_summary(structures, PRICES, CONTRACTS)
    result = calculate_portfolio_var(structures, summary, 0.95, 5, loader, CONTRACTS)

    d_z = loader.load_price_differences("CLZ26")
    d_f = loader.load_price_differences("CLF27")
    # net lots: CLZ26 = 10 + 5 = 15, CLF27 = -5
    combined = (d_z * 15.0 * 1000.0 + d_f * -5.0 * 1000.0).to_numpy()
    expected_1day = -np.percentile(combined, 5.0)
    assert result["portfolio_var_1day"] == pytest.approx(expected_1day)
    assert result["portfolio_var_Nday"] == pytest.approx(expected_1day * np.sqrt(5))
    assert result["horizon_days"] == 5


def test_portfolio_var_excludes_symbol_without_data_and_warns(market, loader):
    structures = build_portfolio() + [structure("BRN outright", [leg(BRNZ26, 1, 4.0)])]
    summary = get_exposure_summary(structures, PRICES, CONTRACTS)
    result = calculate_portfolio_var(structures, summary, 0.95, 1, loader, CONTRACTS)

    assert any("BRNZ26" in w for w in result["data_warnings"])
    brn_entry = result["per_structure_var"][structures[2].structure_id]
    assert brn_entry["standalone_var_1day"] is None
    assert brn_entry["symbols_included"] == []
    assert "BRNZ26" in brn_entry["data_warning"]
    # the rest of the portfolio is unaffected
    assert result["portfolio_var_1day"] > 0


def test_portfolio_var_raises_when_no_position_has_data(tmp_path, loader):
    structures = [structure("BRN outright", [leg(BRNZ26, 1, 4.0)])]
    summary = get_exposure_summary(structures, PRICES, CONTRACTS)
    with pytest.raises(DataNotAvailableError):
        calculate_portfolio_var(structures, summary, 0.95, 1, loader, CONTRACTS)


def test_portfolio_var_empty_portfolio_is_zero(loader):
    summary = get_exposure_summary([], PRICES, CONTRACTS)
    result = calculate_portfolio_var([], summary, 0.95, 1, loader, CONTRACTS)
    assert result["portfolio_var_1day"] == 0.0
    assert result["scenarios_used"] == 0
    assert result["per_structure_var"] == {}


def test_portfolio_var_ignores_closed_structures(market, loader):
    closed = structure("closed", [leg(CLZ26, 1, 10.0)])
    closed.status = StructureStatus.CLOSED
    summary = get_exposure_summary([closed], PRICES, CONTRACTS)
    result = calculate_portfolio_var([closed], summary, 0.95, 1, loader, CONTRACTS)
    assert result["per_structure_var"] == {}
    assert result["portfolio_var_1day"] == 0.0


# ---------- get_var_term_structure ----------


def test_term_structure_keys(market, loader):
    result = get_var_term_structure("CLZ26", 10.0, 1000.0, 0.95, loader)
    assert list(result.keys()) == [1, 2, 3, 5, 10]


def test_term_structure_monotonically_increasing(market, loader):
    result = get_var_term_structure("CLZ26", 10.0, 1000.0, 0.95, loader)
    values = [result[h] for h in [1, 2, 3, 5, 10]]
    assert all(later > earlier for earlier, later in zip(values, values[1:]))
    assert result[10] == pytest.approx(result[1] * np.sqrt(10))
