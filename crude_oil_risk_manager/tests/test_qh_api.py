"""QH API client: rate limiting, retry rules, validation and endpoint wrappers (no network)."""

import pytest
import requests

from adapters.base import (
    APIError,
    APIServerError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from adapters.qh_api import ENDPOINTS, QHApi, QHClient, list_endpoints
from adapters.qh_api.rate_limiter import DEFAULT_LIMIT, EndpointRateLimiter, RateLimit


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code = status
        self._payload = {"results": []} if payload is None else payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params, "json": json, "headers": headers})
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def make_client(*responses, token="tok", max_wait=90.0):
    clock = FakeClock()
    session = FakeSession(*(responses or (FakeResponse(),)))
    client = QHClient(
        token_provider=lambda: token, session=session, max_wait_seconds=max_wait,
        sleep=clock.sleep, limiter_clock=clock.time,
    )
    return client, session, clock


# ---------- limiter ----------


def test_limiter_blocks_the_eighth_request_in_a_minute():
    clock = FakeClock()
    limiter = EndpointRateLimiter(DEFAULT_LIMIT, clock=clock.time, sleep=clock.sleep)
    for _ in range(7):
        limiter.acquire()
    assert clock.slept == []
    limiter.acquire()
    assert 59 < sum(clock.slept) < 62  # waited for the minute window to free a slot


def test_limiter_refuses_to_wait_out_an_exhausted_hour():
    clock = FakeClock()
    limiter = EndpointRateLimiter(RateLimit(100, 2, 1000), name="/x/", max_wait_seconds=90, clock=clock.time, sleep=clock.sleep)
    limiter.acquire()
    limiter.acquire()
    with pytest.raises(RateLimitError):
        limiter.acquire()


def test_limiter_honours_remaining_zero_header():
    clock = FakeClock()
    wall = {"t": 5000.0}
    limiter = EndpointRateLimiter(DEFAULT_LIMIT, clock=clock.time, sleep=clock.sleep, wall_clock=lambda: wall["t"])
    limiter.observe({"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "5020"})
    limiter.acquire()
    assert 19 < sum(clock.slept) < 22


# ---------- client ----------


def test_request_sends_bearer_token_and_drops_none_params():
    client, session, _ = make_client()
    client.request(ENDPOINTS["gtc"], {"products": ["CL", "BZ"], "date": None, "limit": 5})
    call = session.calls[0]
    assert call["headers"]["Authorization"] == "Bearer tok"
    assert call["params"] == {"products": "CL,BZ", "limit": "5"}
    assert call["url"].endswith("/apis/gtc/")


def test_public_endpoint_needs_no_token_and_is_not_limited():
    client, session, clock = make_client(token="")
    for _ in range(20):
        client.request(ENDPOINTS["instruments"], {"qh_codes": "CLZ26"})
    assert "Authorization" not in session.calls[0]["headers"]
    assert clock.slept == []
    assert client.limiter_for(ENDPOINTS["instruments"]) is None


def test_missing_token_raises_before_any_request():
    client, session, _ = make_client(token="")
    with pytest.raises(AuthenticationError):
        client.request(ENDPOINTS["gtc"], {"products": "*"})
    assert session.calls == []


def test_401_raises_authentication_error():
    client, _, _ = make_client(FakeResponse(401))
    with pytest.raises(AuthenticationError):
        client.request(ENDPOINTS["gtc"], {"products": "*"})


def test_429_waits_retry_after_then_retries():
    client, session, clock = make_client(FakeResponse(429, headers={"Retry-After": "30"}), FakeResponse(200, {"ok": 1}))
    assert client.request(ENDPOINTS["gtc"], {"products": "*"}) == {"ok": 1}
    assert len(session.calls) == 2
    assert any(abs(s - 30) < 1 for s in clock.slept)


def test_429_gives_up_after_a_few_attempts_without_hammering():
    client, session, _ = make_client(FakeResponse(429, headers={"Retry-After": "5"}))
    with pytest.raises(RateLimitError):
        client.request(ENDPOINTS["gtc"], {"products": "*"})
    assert len(session.calls) == 3


def test_429_with_long_retry_after_fails_fast():
    client, session, clock = make_client(FakeResponse(429, headers={"Retry-After": "3000"}))
    with pytest.raises(RateLimitError):
        client.request(ENDPOINTS["gtc"], {"products": "*"})
    assert len(session.calls) == 1 and clock.slept == []


def test_5xx_retries_with_exponential_backoff_then_succeeds():
    client, session, clock = make_client(FakeResponse(503), FakeResponse(500), FakeResponse(200, {"ok": 1}))
    assert client.request(ENDPOINTS["economies"]) == {"ok": 1}
    backoffs = [s for s in clock.slept if s in (1.0, 2.0, 4.0)]
    assert backoffs[:2] == [1.0, 2.0]


def test_5xx_gives_up_after_five_attempts():
    client, session, _ = make_client(FakeResponse(502))
    with pytest.raises(APIServerError):
        client.request(ENDPOINTS["economies"])
    assert len(session.calls) == 5


def test_timeout_retries_then_raises():
    client, session, _ = make_client(requests.exceptions.Timeout())
    with pytest.raises(APITimeoutError):
        client.request(ENDPOINTS["economies"])
    assert len(session.calls) == 5


def test_other_4xx_is_not_retried():
    client, session, _ = make_client(FakeResponse(400, text="bad param"))
    with pytest.raises(APIError, match="400"):
        client.request(ENDPOINTS["economies"])
    assert len(session.calls) == 1


def test_get_and_post_on_same_path_share_one_budget():
    client, _, _ = make_client()
    assert client.limiter_for(ENDPOINTS["fairvalue"]) is client.limiter_for(ENDPOINTS["fairvalue_post"])


def test_seven_calls_per_minute_then_the_eighth_waits():
    client, session, clock = make_client()
    for _ in range(8):
        client.request(ENDPOINTS["economies"])
    assert len(session.calls) == 8
    assert sum(clock.slept) > 55


def test_ohlc_has_its_own_higher_limit_and_tas_its_own():
    client, _, _ = make_client()
    assert client.limiter_for(ENDPOINTS["ohlc"]).remaining_this_minute() == 30
    assert client.limiter_for(ENDPOINTS["tas"]).remaining_this_minute() == 10
    assert client.limiter_for(ENDPOINTS["economies"]).remaining_this_minute() == 7


def test_request_all_walks_pages_and_stops_at_the_last():
    pages = [
        FakeResponse(200, {"results": [1, 2], "next": "x"}),
        FakeResponse(200, {"results": [3], "next": None}),
    ]
    client, session, _ = make_client(*pages)
    assert client.request_all(ENDPOINTS["gtc"], {"products": "*"}, page_size=2) == [1, 2, 3]
    assert [c["params"]["offset"] for c in session.calls] == ["0", "2"]


def test_request_all_respects_max_pages():
    client, session, _ = make_client(FakeResponse(200, {"results": [1, 2], "next": "more"}))
    assert len(client.request_all(ENDPOINTS["gtc"], {"products": "*"}, page_size=2, max_pages=3)) == 6
    assert len(session.calls) == 3


def test_request_all_rejects_unpaginated_endpoint():
    client, _, _ = make_client()
    with pytest.raises(ValueError):
        client.request_all(ENDPOINTS["economies"])


# ---------- endpoint wrappers ----------


def make_api(*responses):
    client, session, clock = make_client(*responses)
    return QHApi(client), session, clock


def test_registry_covers_every_swagger_endpoint():
    assert len(list_endpoints()) == 37
    assert {e["name"] for e in list_endpoints() if e["per_minute"] is None} == {
        "instruments", "fundamental_codes", "taq_products",
    }


def test_ohlc_validates_and_maps_params():
    api, session, _ = make_api()
    api.ohlc("1D", instruments=["CLZ26", "COZ26"], count=10, extra_fields="buyvolume")
    assert session.calls[0]["params"] == {
        "instruments": "CLZ26,COZ26", "interval": "1D", "count": "10", "extraFields": "buyvolume",
    }
    with pytest.raises(ValueError):
        api.ohlc("2D", instruments="CLZ26")
    with pytest.raises(ValueError):
        api.ohlc("1D", instruments="CLZ26", hg_instrument_ids="1")
    with pytest.raises(ValueError):
        api.ohlc("1D")
    with pytest.raises(ValueError):
        api.ohlc("1D", instruments=[f"CL{i}" for i in range(51)])


def test_vap_and_fairvalue_historical_limits():
    api, _, _ = make_api()
    with pytest.raises(ValueError):
        api.vap("CLZ26", "1H")
    with pytest.raises(ValueError):
        api.fairvalue_historical([f"CL{i}" for i in range(51)])
    with pytest.raises(ValueError):
        api.fairvalue_historical("CLZ26", fields="volume")


def test_fundamental_data_uses_semicolons_and_caps_codes():
    api, session, _ = make_api()
    api.fundamental_data(["AU CPI YoY NSA_1M", "AU BoP Current Account Balance_1Q"], start_date=["2025-01-01", "2025-02-01"])
    params = session.calls[0]["params"]
    assert params["qhcode"] == "AU CPI YoY NSA_1M;AU BoP Current Account Balance_1Q"
    assert params["start_date"] == "2025-01-01;2025-02-01"
    with pytest.raises(ValueError):
        api.fundamental_data([f"c{i}" for i in range(11)])
    with pytest.raises(ValueError):
        api.fundamental_data("c", page_size=1001)


def test_options_oi_caps_contracts_at_25():
    api, _, _ = make_api()
    with pytest.raises(ValueError):
        api.options_oi(contracts=[f"c{i}" for i in range(26)])


def test_unknown_filter_is_rejected_instead_of_silently_ignored():
    api, session, _ = make_api()
    with pytest.raises(ValueError, match="unknown parameter"):
        api.platts_commentary(dat="2026-01-01")
    assert session.calls == []


def test_post_endpoints_send_json_bodies():
    api, session, _ = make_api()
    api.taq([{"id": "CLZ26", "dates": ["2026-10-01"], "start": "09:00:00", "end": "10:00:00"}])
    api.tas([{"id": "CLZ26", "dates": ["2026-10-01"]}])
    api.generic_chart_data([{"product": "DIJF29", "interval": "1D"}])
    api.fairvalue_post(["CLZ26"])
    assert [c["method"] for c in session.calls] == ["POST"] * 4
    assert session.calls[0]["json"]["products"][0]["id"] == "CLZ26"
    assert session.calls[2]["json"] == {"groups": [{"product": "DIJF29", "interval": "1D"}], "end_inclusive": 1}
    with pytest.raises(ValueError):
        api.taq([{"id": "CLZ26", "dates": []}])


def test_ltp_by_id_fills_the_path_and_shares_the_budget():
    api, session, _ = make_api()
    api.platts_ltp_historical_by_id(42, market="X")
    assert session.calls[0]["url"].endswith("/platts/ltp/historical/42/")
    assert api.client.limiter_for(ENDPOINTS["platts_ltp_historical_by_id"]) is not api.client.limiter_for(
        ENDPOINTS["platts_ltp_historical"]
    )  # different endpoint path, own budget


def test_every_registry_entry_has_a_wrapper_method():
    wrapper_names = {n for n in dir(QHApi) if not n.startswith("_")}
    for name in ENDPOINTS:
        assert name in wrapper_names, name
