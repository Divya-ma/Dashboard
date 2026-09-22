"""Tests for adapters.live.fairvalue.FairValueLiveAdapter (/fairvalue/ endpoint)."""

from datetime import datetime, timezone

import pytest

from adapters.base import AuthenticationError, RateLimitError
from adapters.live.fairvalue import WATCHLIST_SETTING_KEY, FairValueLiveAdapter
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from db.repository import Repository


def make_contract(**overrides) -> Contract:
    defaults = dict(
        product="CL", contract_month=12, contract_year=2026, symbol="CLZ26",
        multiplier=1000.0, tick_size=0.01, tick_value=10.0,
    )
    defaults.update(overrides)
    return Contract(**defaults)


def make_leg(**overrides) -> Leg:
    defaults = dict(contract=make_contract(), ratio=1, lots=5.0, entry_price=75.0, average_entry_price=75.0)
    defaults.update(overrides)
    return Leg(**defaults)


def save_open_structure(repo: Repository, name: str, legs: list[Leg], status=StructureStatus.OPEN) -> Structure:
    structure = Structure(
        name=name, structure_type=StructureType.OUTRIGHT if len(legs) == 1 else StructureType.CUSTOM,
        products=sorted({leg.contract.symbol[:2] for leg in legs}), legs=legs, status=status,
    )
    for leg in legs:
        repo.save_contract(leg.contract)
    repo.save_structure(structure)
    return structure


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture
def adapter(repo):
    return FairValueLiveAdapter(repo, staleness_threshold_seconds=10.0)


def _fairvalue_row(contract="CLZ26", price=75.5, age_seconds=0.0):
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    return {"Timestamp": now_ms - age_seconds * 1000, "Price": price, "Contract": contract}


# ---------- contract selection ----------


def test_no_open_structures_or_watchlist_means_no_request(adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    get = mocker.patch("adapters.live.fairvalue.requests.get")
    assert adapter.get_live_prices(["ignored"]) == {}
    get.assert_not_called()


def test_requests_leg_symbols_of_open_structures_only(adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    save_open_structure(repo, "Open CL", [make_leg()], status=StructureStatus.OPEN)
    save_open_structure(
        repo, "Shell BRN",
        [make_leg(contract=make_contract(product="BRN", symbol="BRNZ26"), lots=0.0, entry_price=None, average_entry_price=None)],
        status=StructureStatus.SHELL,
    )
    save_open_structure(
        repo, "Closed WBS",
        [make_leg(contract=make_contract(product="WBS", symbol="WBSZ26"))],
        status=StructureStatus.CLOSED,
    )
    get = mocker.patch(
        "adapters.live.fairvalue.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: {"data": [_fairvalue_row()]}),
    )
    prices = adapter.get_live_prices([])
    assert get.call_args.kwargs["params"]["products"] == "CLZ26"
    assert set(prices) == {"CLZ26"}


def test_partially_closed_structures_are_treated_as_open(adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    save_open_structure(repo, "Legacy", [make_leg()], status=StructureStatus.PARTIALLY_CLOSED)
    get = mocker.patch(
        "adapters.live.fairvalue.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: {"data": []}),
    )
    adapter.get_live_prices([])
    assert get.call_args.kwargs["params"]["products"] == "CLZ26"


def test_watchlist_symbols_are_included_and_deduplicated(adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    save_open_structure(repo, "Open CL", [make_leg()])
    repo.set_setting(WATCHLIST_SETTING_KEY, ["CLZ26", "BRNZ26"])  # CLZ26 overlaps the open leg
    get = mocker.patch(
        "adapters.live.fairvalue.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: {"data": []}),
    )
    adapter.get_live_prices([])
    requested = set(get.call_args.kwargs["params"]["products"].split(","))
    assert requested == {"CLZ26", "COZ26"}  # BRNZ26 -> COZ26


def test_ignores_the_symbols_argument_entirely(adapter, repo, mocker):
    """Per spec: the caller's `symbols` list is not what gets requested."""
    repo.set_setting("api_access_token", "tok")
    save_open_structure(repo, "Open CL", [make_leg()])
    get = mocker.patch(
        "adapters.live.fairvalue.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: {"data": []}),
    )
    adapter.get_live_prices(["XYZ99", "totally-unrelated"])
    assert get.call_args.kwargs["params"]["products"] == "CLZ26"


def test_untranslatable_watchlist_symbol_is_skipped_not_fatal(adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    repo.set_setting(WATCHLIST_SETTING_KEY, ["CLZ26", "NOTAPRODUCT99"])
    get = mocker.patch(
        "adapters.live.fairvalue.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: {"data": []}),
    )
    adapter.get_live_prices([])
    assert get.call_args.kwargs["params"]["products"] == "CLZ26"


# ---------- request / response ----------


def test_get_live_prices_parses_response_and_translates_symbols(adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    save_open_structure(repo, "Open BRN", [make_leg(contract=make_contract(product="BRN", symbol="BRNZ26"))])
    mocker.patch(
        "adapters.live.fairvalue.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: {"data": [_fairvalue_row(contract="COZ26", price=42.5)]}),
    )
    prices = adapter.get_live_prices([])
    assert prices["BRNZ26"].symbol == "BRNZ26"
    assert prices["BRNZ26"].price == 42.5
    assert prices["BRNZ26"].raw_open == prices["BRNZ26"].raw_high == prices["BRNZ26"].raw_low == 42.5
    assert prices["BRNZ26"].raw_volume == 0.0


def test_sends_bearer_token_and_products_param(adapter, repo, mocker):
    repo.set_setting("api_access_token", "secret-token")
    save_open_structure(repo, "Open CL", [make_leg()])
    get = mocker.patch(
        "adapters.live.fairvalue.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: {"data": []}),
    )
    adapter.get_live_prices([])
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-token"
    assert get.call_args.args[0] == "https://qh-api.corp.hertshtengroup.com/apis/fairvalue/"


def test_no_token_raises_runtime_error(adapter, repo):
    save_open_structure(repo, "Open CL", [make_leg()])
    with pytest.raises(RuntimeError):
        adapter.get_live_prices([])


def test_response_product_not_requested_is_ignored(adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    save_open_structure(repo, "Open CL", [make_leg()])
    mocker.patch(
        "adapters.live.fairvalue.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: {"data": [_fairvalue_row(contract="XXXX99")]}),
    )
    assert adapter.get_live_prices([]) == {}


def test_staleness_flag(adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    save_open_structure(repo, "Open CL", [make_leg()])
    mocker.patch(
        "adapters.live.fairvalue.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: {"data": [_fairvalue_row(age_seconds=45)]}),
    )
    assert adapter.get_live_prices([])["CLZ26"].is_stale is True

    adapter.set_staleness_threshold(120.0)
    assert adapter.get_live_prices([])["CLZ26"].is_stale is False


# ---------- errors ----------


def test_401_raises_authentication_error(adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    save_open_structure(repo, "Open CL", [make_leg()])
    mocker.patch("adapters.live.fairvalue.requests.get", return_value=mocker.Mock(status_code=401))
    with pytest.raises(AuthenticationError):
        adapter.get_live_prices([])


def test_429_raises_rate_limit_error(adapter, repo, mocker):
    repo.set_setting("api_access_token", "tok")
    save_open_structure(repo, "Open CL", [make_leg()])
    mocker.patch("adapters.live.fairvalue.requests.get", return_value=mocker.Mock(status_code=429))
    with pytest.raises(RateLimitError):
        adapter.get_live_prices([])


# ---------- check_connection ----------


def test_check_connection_uses_given_token_and_leaves_saved_token_alone(adapter, repo, mocker):
    repo.set_setting("api_access_token", "saved-token")
    get = mocker.patch(
        "adapters.live.fairvalue.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: {"data": [_fairvalue_row()]}),
    )
    assert adapter.check_connection("typed-token", "CLZ26") == 1
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer typed-token"
    assert get.call_args.kwargs["params"]["products"] == "CLZ26"
    assert repo.get_setting("api_access_token") == "saved-token"


def test_check_connection_returns_zero_when_no_data(adapter, repo, mocker):
    mocker.patch(
        "adapters.live.fairvalue.requests.get",
        return_value=mocker.Mock(status_code=200, json=lambda: {"data": []}),
    )
    assert adapter.check_connection("good-token") == 0
