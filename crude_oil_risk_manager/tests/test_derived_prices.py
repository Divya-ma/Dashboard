"""core.derived_prices and the live poll's use of it for composites /fairvalue/ does not return."""

import logging
from datetime import datetime, timezone

import pytest

from adapters.live.fairvalue import FairValueLiveAdapter
from core.derived_prices import components_needed, derivation_options, derive_price
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from db.repository import Repository

NOW = 1_790_000_000.0
FLY = "BRNV28-X28-Z28"


# ---------- the pure maths ----------


def test_options_for_each_kind_of_symbol():
    assert derivation_options("CLZ26") == []
    assert derivation_options("CLZ26-F27") == [[("CLZ26", 1.0), ("CLF27", -1.0)]]
    fly = derivation_options(FLY)
    assert fly[0] == [("BRNV28-X28", 1.0), ("BRNX28-Z28", -1.0)]  # two spreads first
    assert fly[1] == [("BRNV28", 1.0), ("BRNX28", -2.0), ("BRNZ28", 1.0)]  # then the three outrights
    condor = derivation_options("CLZ26-F27-G27+H27")
    assert condor[0] == [("CLZ26-F27", 1.0), ("CLG27-H27", -1.0)]
    assert condor[1] == [("CLZ26", 1.0), ("CLF27", -1.0), ("CLG27", -1.0), ("CLH27", 1.0)]
    assert derivation_options("nonsense") == []
    assert components_needed(FLY) == ["BRNV28-X28", "BRNX28-Z28", "BRNV28", "BRNX28", "BRNZ28"]


def test_a_fly_is_built_from_spreads_when_they_are_priced():
    prices = {"BRNV28-X28": (0.40, NOW), "BRNX28-Z28": (0.31, NOW - 60), "BRNV28": (70.0, NOW), "BRNX28": (69.6, NOW), "BRNZ28": (69.29, NOW)}
    value, when, how = derive_price(FLY, prices)
    assert value == pytest.approx(0.09) and when == NOW - 60 and "BRNV28-X28" in how


def test_it_falls_back_to_outrights_when_a_spread_is_missing():
    prices = {"BRNV28-X28": (0.40, NOW), "BRNV28": (70.0, NOW), "BRNX28": (69.6, NOW), "BRNZ28": (69.29, NOW)}
    assert derive_price(FLY, prices)[0] == pytest.approx(70.0 - 2 * 69.6 + 69.29)


def test_pieces_priced_hours_apart_are_not_combined():
    stale_leg = {"BRNV28-X28": (0.40, NOW), "BRNX28-Z28": (0.31, NOW - 4 * 3600)}
    assert derive_price(FLY, stale_leg) is None  # would mix a price from now with one from 4 hours ago
    outrights_ok = {**stale_leg, "BRNV28": (70.0, NOW), "BRNX28": (69.6, NOW - 30), "BRNZ28": (69.29, NOW - 90)}
    assert derive_price(FLY, outrights_ok)[0] == pytest.approx(0.09)  # the consistent route is used instead
    assert derive_price(FLY, {}) is None


# ---------- the live poll ----------


def make_contract(symbol):
    return Contract(product="BRN", contract_month=12, contract_year=2028, symbol=symbol,
                    multiplier=1000.0, tick_size=0.01, tick_value=10.0)


@pytest.fixture
def repo(tmp_db_path):
    repository = Repository(str(tmp_db_path))
    repository.set_setting("api_access_token", "tok")
    leg = Leg(contract=make_contract(FLY), ratio=1, lots=5.0, entry_price=0.1, average_entry_price=0.1)
    repository.save_contract(leg.contract)
    repository.save_structure(Structure(name="far fly", structure_type=StructureType.OUTRIGHT, products=["BRN"], legs=[leg], status=StructureStatus.OPEN))
    return repository


def row(contract, price, age=0.0):
    return {"Contract": contract, "Price": price, "Timestamp": (datetime.now(timezone.utc).timestamp() - age) * 1000}


class Replies:
    """Successive /fairvalue/ replies; remembers what was asked."""

    def __init__(self, mocker, *replies):
        self.requested: list[set[str]] = []
        self._replies = list(replies)

        def request(method, url, params=None, **kwargs):
            self.requested.append(set(params["products"].split(",")))
            data = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
            return mocker.Mock(headers={}, status_code=200, json=lambda: {"data": data})

        mocker.patch("adapters.qh_api.client.requests.Session.request", side_effect=request)


def test_a_fly_fairvalue_does_not_return_gets_a_derived_price_in_the_same_request(repo, mocker):
    api = Replies(mocker, [row("COV28-X28", 0.40), row("COX28-Z28", 0.31)])
    adapter = FairValueLiveAdapter(repo, staleness_threshold_seconds=60.0)
    prices = adapter.get_live_prices([])
    assert api.requested[0] >= {"COV28-X28-Z28", "COV28-X28", "COX28-Z28", "COV28", "COX28", "COZ28"}  # components ride along
    assert len(api.requested) == 1  # no extra call
    price = prices[FLY]
    assert price.derived is True and price.price == pytest.approx(0.09) and not price.is_stale


def test_a_price_fairvalue_does_return_is_used_as_it_is(repo, mocker):
    Replies(mocker, [row("COV28-X28-Z28", 0.12), row("COV28-X28", 0.40), row("COX28-Z28", 0.31)])
    price = FairValueLiveAdapter(repo, staleness_threshold_seconds=60.0).get_live_prices([])[FLY]
    assert price.derived is False and price.price == pytest.approx(0.12)


def test_components_are_only_requested_again_for_symbols_that_were_missing(repo, mocker):
    api = Replies(mocker, [row("COV28-X28", 0.40), row("COX28-Z28", 0.31)], [row("COV28-X28-Z28", 0.12)])
    adapter = FairValueLiveAdapter(repo, staleness_threshold_seconds=60.0)
    adapter.get_live_prices([])
    assert api.requested[0] > {"COV28-X28-Z28"}  # first poll: components for every composite
    assert adapter.get_live_prices([])[FLY].derived is False  # second reply carries the fly itself
    assert adapter.get_live_prices([])[FLY].derived is False
    assert api.requested[2] == {"COV28-X28-Z28"}  # it came back, so nothing extra is asked for any more


def test_old_components_make_the_derived_price_stale(repo, mocker):
    Replies(mocker, [row("COV28-X28", 0.40, age=4 * 3600), row("COX28-Z28", 0.31, age=4 * 3600 + 20)])
    price = FairValueLiveAdapter(repo, staleness_threshold_seconds=60.0).get_live_prices([])[FLY]
    assert price.derived and price.is_stale


def test_a_symbol_that_cannot_be_derived_is_logged_once_not_every_poll(repo, mocker, caplog):
    Replies(mocker, [row("COV28-X28", 0.40)])  # the second spread and every outright are missing
    adapter = FairValueLiveAdapter(repo, staleness_threshold_seconds=60.0)
    with caplog.at_level(logging.WARNING, logger="adapters.live.fairvalue"):
        for _ in range(4):
            assert adapter.get_live_prices([]) == {}
    assert sum("No fair value data returned" in r.message for r in caplog.records) == 1


def test_pieces_priced_hours_apart_leave_the_symbol_missing(repo, mocker):
    Replies(mocker, [row("COV28-X28", 0.40), row("COX28-Z28", 0.31, age=4 * 3600)])
    assert FairValueLiveAdapter(repo, staleness_threshold_seconds=60.0).get_live_prices([]) == {}
