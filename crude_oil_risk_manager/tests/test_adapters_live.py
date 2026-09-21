"""Tests for adapters.base (SymbolTranslator) and adapters.mock.mock_live (MockLiveAdapter)."""

import pytest

from adapters.base import SymbolTranslator
from adapters.mock.mock_live import MockLiveAdapter
from core.models import Leg
from core.models.contract import Contract


def make_contract(product: str, month: int, year: int, symbol: str) -> Contract:
    return Contract(
        product=product,
        contract_month=month,
        contract_year=year,
        symbol=symbol,
        multiplier=1000.0,
        tick_size=0.01,
        tick_value=10.0,
    )


def make_leg(product: str, month: int, year: int, symbol: str, ratio: int) -> Leg:
    return Leg(contract=make_contract(product, month, year, symbol), ratio=ratio)


# ---------- SymbolTranslator.internal_to_api ----------


@pytest.mark.parametrize(
    "internal_symbol,expected_api_symbol",
    [
        ("CLZ26", "CLZ26"),
        ("BRNZ26", "COZ26"),
        ("BZZ26", "BZZZ26"),
        ("WBSZ26", "WTCLZ26"),
        ("GZ26", "GOZ26"),
    ],
)
def test_internal_to_api_translates_product_codes(internal_symbol, expected_api_symbol):
    assert SymbolTranslator.internal_to_api(internal_symbol) == expected_api_symbol


def test_internal_to_api_unknown_product_raises_value_error():
    with pytest.raises(ValueError):
        SymbolTranslator.internal_to_api("XYZZ26")


def test_api_to_internal_roundtrip():
    assert SymbolTranslator.api_to_internal("COZ26") == "BRNZ26"
    assert SymbolTranslator.api_to_internal("CLZ26") == "CLZ26"


# ---------- SymbolTranslator.structure_to_api_symbol ----------


def test_structure_to_api_symbol_outright():
    legs = [make_leg("BRN", 12, 2026, "BRNZ26", 1)]
    assert SymbolTranslator.structure_to_api_symbol(legs) == "COZ26"


def test_structure_to_api_symbol_spread():
    legs = [
        make_leg("BRN", 12, 2026, "BRNZ26", 1),
        make_leg("BRN", 1, 2027, "BRNF27", -1),
    ]
    assert SymbolTranslator.structure_to_api_symbol(legs) == "COZ26-F27"


def test_structure_to_api_symbol_fly():
    legs = [
        make_leg("WBS", 12, 2026, "WBSZ26", 1),
        make_leg("WBS", 1, 2027, "WBSF27", -2),
        make_leg("WBS", 2, 2027, "WBSG27", 1),
    ]
    assert SymbolTranslator.structure_to_api_symbol(legs) == "WTCLZ26-F27-G27"


def test_structure_to_api_symbol_condor():
    legs = [
        make_leg("BZ", 12, 2026, "BZZ26", 1),
        make_leg("BZ", 1, 2027, "BZF27", -1),
        make_leg("BZ", 2, 2027, "BZG27", -1),
        make_leg("BZ", 3, 2027, "BZH27", 1),
    ]
    assert SymbolTranslator.structure_to_api_symbol(legs) == "BZZZ26-F27-G27+H27"


def test_structure_to_api_symbol_more_than_four_legs_raises():
    legs = [
        make_leg("CL", m, 2026, f"CL{letter}26", 1)
        for m, letter in [(9, "U"), (10, "V"), (11, "X"), (12, "Z"), (1, "F")]
    ]
    with pytest.raises(NotImplementedError):
        SymbolTranslator.structure_to_api_symbol(legs)


def test_structure_to_api_symbol_cross_product_spread_raises():
    legs = [
        make_leg("CL", 12, 2026, "CLZ26", 1),
        make_leg("BRN", 12, 2026, "BRNZ26", -1),
    ]
    with pytest.raises(NotImplementedError):
        SymbolTranslator.structure_to_api_symbol(legs)


# ---------- SymbolTranslator.parse_api_symbol ----------


def test_parse_api_symbol_outright():
    assert SymbolTranslator.parse_api_symbol("CLZ26") == [{"product": "CL", "month": "Z", "year": "26"}]


def test_parse_api_symbol_spread():
    parsed = SymbolTranslator.parse_api_symbol("COZ26-F27")
    assert parsed == [
        {"product": "BRN", "month": "Z", "year": "26"},
        {"product": "BRN", "month": "F", "year": "27"},
    ]


# ---------- MockLiveAdapter ----------


def test_mock_live_adapter_returns_price_for_each_symbol():
    adapter = MockLiveAdapter()
    symbols = ["CLZ26", "COZ26"]
    prices = adapter.get_live_prices(symbols)
    assert set(prices.keys()) == set(symbols)


def test_mock_live_adapter_returns_internal_symbols_not_api_codes():
    adapter = MockLiveAdapter()
    prices = adapter.get_live_prices(["BRNZ26"])
    assert "BRNZ26" in prices
    assert prices["BRNZ26"].symbol == "BRNZ26"


def test_mock_live_adapter_get_access_token():
    adapter = MockLiveAdapter()
    assert adapter.get_access_token() == "mock_token"


def test_live_price_is_stale_false_when_recent():
    adapter = MockLiveAdapter()
    prices = adapter.get_live_prices(["CLZ26"])
    assert prices["CLZ26"].is_stale is False


# ---------- VendorLiveAdapter.check_connection / staleness ----------


def _vendor(tmp_db_path, staleness=10.0):
    from adapters.live.vendor import VendorLiveAdapter
    from db.repository import Repository

    repo = Repository(str(tmp_db_path))
    return VendorLiveAdapter(repo, staleness), repo


def _candle(age_seconds=0.0, product="CLZ26"):
    from datetime import datetime, timezone

    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    return {
        "product": product, "time": now_ms - age_seconds * 1000,
        "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10,
    }


def test_check_connection_uses_given_token_and_leaves_saved_token_alone(tmp_db_path, mocker):
    adapter, repo = _vendor(tmp_db_path)
    repo.set_setting("api_access_token", "saved-token")
    get = mocker.patch(
        "adapters.live.vendor.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: [_candle()]),
    )

    assert adapter.check_connection("typed-token", "CLZ26") == 1

    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer typed-token"
    assert get.call_args.kwargs["params"]["instruments"] == "CLZ26"
    assert repo.get_setting("api_access_token") == "saved-token"


def test_check_connection_raises_authentication_error_on_401(tmp_db_path, mocker):
    from adapters.base import AuthenticationError

    adapter, _ = _vendor(tmp_db_path)
    mocker.patch("adapters.live.vendor.requests.get", return_value=mocker.Mock(status_code=401))
    with pytest.raises(AuthenticationError):
        adapter.check_connection("bad-token")


def test_check_connection_returns_zero_when_no_candles(tmp_db_path, mocker):
    adapter, _ = _vendor(tmp_db_path)
    mocker.patch("adapters.live.vendor.requests.get", return_value=mocker.Mock(status_code=200, json=lambda: []))
    assert adapter.check_connection("good-token") == 0


def test_set_staleness_threshold_changes_what_is_flagged_stale(tmp_db_path, mocker):
    adapter, repo = _vendor(tmp_db_path, staleness=10.0)
    repo.set_setting("api_access_token", "tok")
    mocker.patch(
        "adapters.live.vendor.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: [_candle(age_seconds=45)]),
    )
    assert adapter.get_live_prices(["CLZ26"])["CLZ26"].is_stale is True

    adapter.set_staleness_threshold(120.0)
    assert adapter.get_live_prices(["CLZ26"])["CLZ26"].is_stale is False
