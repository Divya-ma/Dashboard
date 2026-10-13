"""Curve Compare: VWAP, quoted-vs-derived curves, the TAQ quote store, the service and the page."""

from datetime import date, datetime, timezone

import pytest

from adapters.base import RateLimitError
from core.curve_calendar import DFLY, FLY, OUTRIGHT, SPREAD, generic_contracts
from core.curve_compare import (
    build_curve_views, DFLY_FLIES, DFLY_SPREAD3, DFLY_SPREADS, SOURCE_DERIVED, SOURCE_FAIR, SOURCE_QUOTE, CurveCompare, Quote,
    build_curve, build_price_book, settlement_curve, symbol_tiers, symbols_needed, vwap,
)
from core.quote_store import FairValueStore, QuoteStore, parse_quotes
from ui.callbacks import curve_compare_callbacks as ccc
from ui.container import Container
from ui.layouts import curve_compare_tab as tab

AS_OF = date(2026, 10, 6)
NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc).timestamp()


def outright_prices(product="CL", n=8):
    """A curved, non-linear set of outright prices so every family has a distinct, checkable value."""
    contracts = generic_contracts(product, AS_OF, n)
    return {c.symbol: 80.0 - 0.7 * i + 0.05 * i * i for i, c in enumerate(contracts)}


def book_from(prices, ts=NOW):
    return {s: (p, ts, SOURCE_QUOTE) for s, p in prices.items()}


# ----------------------------------------------------------------------
# VWAP
# ----------------------------------------------------------------------


def test_vwap_weights_by_the_opposite_side_size():
    # More size on the bid -> the ask is likelier to be hit last, price leans towards the ask side's
    # smaller size: (90.26 * 1 + 90.28 * 4) / 5
    assert vwap(90.26, 90.28, 4, 1) == pytest.approx((90.26 * 1 + 90.28 * 4) / 5)
    assert vwap(90.26, 90.28, 1, 4) == pytest.approx((90.26 * 4 + 90.28 * 1) / 5)


def test_vwap_without_sizes_is_the_plain_mid_and_without_a_two_sided_quote_is_none():
    assert vwap(1.0, 2.0, 0, 0) == 1.5
    assert vwap(1.0, 2.0, None, None) == 1.5
    assert vwap(None, 2.0, 1, 1) is None
    assert vwap(1.0, float("nan"), 1, 1) is None


def test_quote_price_and_book_prefers_quotes_over_fair_value():
    book = build_price_book({"A": Quote(1.0, 1.1, 1, 1, 100.0)}, {"A": (9.0, 90.0), "B": (5.0, 90.0)})
    assert book["A"] == (pytest.approx(1.05), 100.0, SOURCE_QUOTE)
    assert book["B"] == (5.0, 90.0, SOURCE_FAIR)


# ----------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------


def test_each_family_has_the_right_number_of_points_and_values():
    out = outright_prices()
    vals = list(out.values())
    book = book_from(out)
    for family, length in ((OUTRIGHT, 1), (SPREAD, 2), (FLY, 3), (DFLY, 4)):
        points = build_curve("CL", AS_OF, 8, family, book, NOW)
        assert len(points) == 8 - length + 1
        assert all(p.value is not None for p in points)
    spread = build_curve("CL", AS_OF, 8, SPREAD, book, NOW)
    fly = build_curve("CL", AS_OF, 8, FLY, book, NOW)
    assert spread[0].value == pytest.approx(vals[0] - vals[1])
    assert fly[0].value == pytest.approx(vals[0] - 2 * vals[1] + vals[2])


def test_all_three_dfly_methods_agree_on_the_same_underlying_prices():
    out = outright_prices()
    v = list(out.values())
    expected = [v[i] - 3 * v[i + 1] + 3 * v[i + 2] - v[i + 3] for i in range(5)]
    for method in (DFLY_FLIES, DFLY_SPREAD3, DFLY_SPREADS):
        points = build_curve("CL", AS_OF, 8, DFLY, book_from(out), NOW, method)
        assert [p.value for p in points] == pytest.approx(expected)
        assert {p.source for p in points} == {SOURCE_DERIVED}


def test_a_quoted_fly_is_used_as_is_and_a_missing_one_is_derived_from_spreads():
    out = outright_prices()
    contracts = generic_contracts("CL", AS_OF, 8)
    quoted_symbol = f"{contracts[0].symbol}-{contracts[1].symbol[-3:]}-{contracts[2].symbol[-3:]}"
    book = book_from(out)
    book[quoted_symbol] = (99.0, NOW, SOURCE_QUOTE)
    fly = build_curve("CL", AS_OF, 8, FLY, book, NOW)
    assert (fly[0].value, fly[0].source) == (99.0, SOURCE_QUOTE)
    assert fly[1].source == SOURCE_DERIVED and fly[1].how == "spread - next spread"


def test_back_months_with_no_prices_are_left_empty():
    out = outright_prices()
    contracts = generic_contracts("CL", AS_OF, 8)
    del out[contracts[7].symbol]  # the last outright has no price
    points = build_curve("CL", AS_OF, 8, DFLY, book_from(out), NOW)
    assert [p.value is not None for p in points] == [True, True, True, True, False]
    spread = build_curve("CL", AS_OF, 8, SPREAD, book_from(out), NOW)
    assert spread[-1].value is None and spread[-2].value is not None


def test_a_quiet_contracts_hours_old_price_is_still_shown_with_its_age():
    out = outright_prices()
    points = build_curve("CL", AS_OF, 8, OUTRIGHT, book_from(out, ts=NOW - 3 * 3600), NOW)
    assert all(p.source == SOURCE_QUOTE and p.how == "" and p.age_seconds == 3 * 3600 for p in points)


def test_a_price_over_a_day_old_loses_to_a_build_from_fresh_pieces_but_beats_nothing():
    out = outright_prices()
    contracts = generic_contracts("CL", AS_OF, 8)
    spread_symbol = f"{contracts[0].symbol}-{contracts[1].symbol[-3:]}"
    book = book_from(out)
    book[spread_symbol] = (55.0, NOW - 2 * 86400, SOURCE_FAIR)  # stale direct spread, fresh outrights
    spread = build_curve("CL", AS_OF, 8, SPREAD, book, NOW)
    assert spread[0].source == SOURCE_DERIVED and spread[0].value != 55.0
    for symbol in out:  # nothing fresh to build from: the old price is the last resort
        del book[symbol]
    old = build_curve("CL", AS_OF, 8, SPREAD, book, NOW)
    assert (old[0].value, old[0].source, old[0].how) == (55.0, SOURCE_FAIR, "old price")
    assert build_curve("CL", AS_OF, 8, SPREAD, {spread_symbol: (55.0, NOW - 10 * 86400, SOURCE_FAIR)}, NOW)[0].value is None


def test_the_newer_of_quote_and_fair_value_wins():
    book = build_price_book({"A": Quote(1.0, 1.1, 1, 1, 100.0)}, {"A": (9.0, 500.0)})
    assert book["A"] == (9.0, 500.0, SOURCE_FAIR)
    book = build_price_book({"A": Quote(1.0, 1.1, 1, 1, 500.0)}, {"A": (9.0, 500.0)})
    assert book["A"][2] == SOURCE_QUOTE


def test_fair_value_fills_in_where_there_is_no_quote():
    out = outright_prices()
    book = build_price_book({}, {s: (p, NOW) for s, p in out.items()})
    points = build_curve("CL", AS_OF, 8, OUTRIGHT, book, NOW)
    assert {p.source for p in points} == {SOURCE_FAIR}


def test_symbols_needed_lists_the_curves_own_quotes_first():
    tiers = symbol_tiers("CL", AS_OF, 6, FLY)
    assert [len(t) for t in tiers] == [4, 5, 6]
    assert "-" in tiers[0][0] and tiers[0][0].count("-") == 2
    dfly_spread3 = symbol_tiers("CL", AS_OF, 6, DFLY, DFLY_SPREAD3)
    assert [len(t) for t in dfly_spread3] == [3, 5, 6]
    assert dfly_spread3[0][0].count("-") == 1  # a 3-month spread, e.g. CLZ26-G27
    assert symbols_needed("CL", AS_OF, 6, DFLY, DFLY_SPREADS) == [s for t in symbol_tiers("CL", AS_OF, 6, DFLY, DFLY_SPREADS) for s in t]


def test_two_year_ladder_for_both_products():
    for product in ("CL", "BRN"):
        assert len(build_curve(product, AS_OF, 24, OUTRIGHT, {}, NOW)) == 24
        assert len(build_curve(product, AS_OF, 24, DFLY, {}, NOW)) == 21


def test_settlement_curve_builds_every_family_from_outrights():
    out = outright_prices()
    v = list(out.values())
    assert settlement_curve("CL", AS_OF, 8, SPREAD, out)[0] == pytest.approx(v[0] - v[1])
    assert settlement_curve("CL", AS_OF, 8, DFLY, out)[0] == pytest.approx(v[0] - 3 * v[1] + 3 * v[2] - v[3])
    del out[list(out)[2]]
    assert settlement_curve("CL", AS_OF, 8, FLY, out)[0] is None


# ----------------------------------------------------------------------
# Quote store (/taq/)
# ----------------------------------------------------------------------


def taq_row(symbol, ts_seconds, bid, ask, bid_size=1, ask_size=1):
    return {"timestamp": int(ts_seconds * 1e6), "instrument": symbol, "best_bid": bid, "best_ask": ask,
            "bid_size": bid_size, "ask_size": ask_size, "trade_price": ask}


class FakeTaq:
    """Answers /taq/ like the real API: rows sorted oldest first, `pages` of them (page 1 = the oldest)."""

    def __init__(self, rows=None, total_pages=1, error=None, pages=None):
        self.rows, self.total_pages, self.error, self.calls, self.pages = rows or [], total_pages, error, [], pages
        self.page_calls: list[int] = []

    def taq(self, products, page=None):
        self.calls.append(products)
        self.page_calls.append(page or 1)
        if self.error:
            raise self.error
        wanted = {p["id"] for p in products}
        if self.pages is not None:  # explicit pages, oldest first
            data = self.pages[(page or 1) - 1]
            return {"status": "SUCCESS", "page": page or 1, "total_pages": len(self.pages), "data": [r for r in data if r["instrument"] in wanted]}
        return {"status": "SUCCESS", "page": page or 1, "total_pages": self.total_pages,
                "data": [r for r in self.rows if r["instrument"] in wanted]}


def test_parse_quotes_keeps_the_newest_two_sided_row_per_instrument():
    rows = [taq_row("CLZ26", 100, 1.0, 1.2), taq_row("CLZ26", 200, 2.0, 2.2), taq_row("CLZ26", 150, 9, 9),
            taq_row("CLF27", 100, None, 5.0), {"instrument": "CLG27", "best_bid": "x", "best_ask": 1}]
    quotes = parse_quotes(rows)
    assert set(quotes) == {"CLZ26"} and quotes["CLZ26"].bid == 2.0 and quotes["CLZ26"].ts == 200


def test_refresh_requests_utc_windows_and_stores_the_quotes():
    api = FakeTaq([taq_row("CLZ26", NOW - 30, 90.26, 90.28, 4, 1)])
    store = QuoteStore(api, clock=lambda: NOW, autostart=False)
    store.request(("CL", 2, OUTRIGHT, "flies"), ["CLZ26", "CLF27"], primary=2)
    assert store.refresh_once() == 1
    product = {p["id"]: p for p in api.calls[0]}
    assert product["CLZ26"]["dates"] == ["2026-10-06"]
    assert (product["CLZ26"]["start"], product["CLZ26"]["end"]) == ("11:45:00", "12:00:00")  # 15 min bootstrap
    assert store.quotes()["CLZ26"].price == pytest.approx(vwap(90.26, 90.28, 4, 1))
    store.refresh_once()
    again = {p["id"]: p for p in api.calls[1]}
    assert again["CLZ26"]["start"] == "11:58:00"  # now only the short window; CLF27 is still retried later


def test_window_never_starts_before_midnight_utc():
    just_after = datetime(2026, 10, 6, 0, 3, tzinfo=timezone.utc).timestamp()
    api = FakeTaq()
    store = QuoteStore(api, clock=lambda: just_after, autostart=False)
    store.request(("k",), ["CLZ26"])
    store.refresh_once()
    assert api.calls[0][0]["start"] == "00:00:00"


def test_an_older_reply_does_not_replace_a_newer_quote():
    api = FakeTaq([taq_row("CLZ26", NOW - 10, 5.0, 5.2)])
    store = QuoteStore(api, clock=lambda: NOW, autostart=False)
    store.request(("k",), ["CLZ26"])
    store.refresh_once()
    api.rows = [taq_row("CLZ26", NOW - 500, 1.0, 1.2)]
    store.refresh_once()
    assert store.quotes()["CLZ26"].bid == 5.0


def test_errors_are_reported_not_raised_and_extra_pages_shrink_the_chunk():
    store = QuoteStore(FakeTaq(error=RateLimitError("slow down")), clock=lambda: NOW, autostart=False)
    store.request(("k",), ["CLZ26"])
    assert store.refresh_once() == 0 and "rate limit" in store.last_error.lower()
    paged = QuoteStore(FakeTaq([taq_row("CLZ26", NOW - 5, 1.0, 1.2)], total_pages=3), clock=lambda: NOW, autostart=False, chunk_size=20)
    paged.request(("k",), ["CLZ26"])
    paged.refresh_once()
    assert paged._chunk == 10


def test_demand_lapses_and_the_most_overdue_symbols_go_first():
    clock = {"t": NOW}
    store = QuoteStore(FakeTaq(), clock=lambda: clock["t"], autostart=False, chunk_size=2)
    store.request(("k",), ["A", "B", "C"], primary=0)
    store.refresh_once()
    chunk2, _ = store._pick()
    assert chunk2[0] == "C"  # A and B were just attempted
    clock["t"] += 10_000
    assert store._active() == ([], set())


# ----------------------------------------------------------------------
# Service and page
# ----------------------------------------------------------------------


class FakeSettlements:
    def __init__(self, prices):
        self._prices, self.fetched_on, self.fetches = prices, None, 0

    def prices(self):
        return self._prices

    def settlement_date(self):
        return date(2026, 10, 5)

    def fetch(self, api, symbols, today, fetched_on):
        self.fetches += 1
        self.fetched_on = fetched_on.isoformat()
        return {"fetched": len(symbols), "missing": []}


def make_service(rows, prev=None, fallback=None):
    api = FakeTaq(rows)
    store = QuoteStore(api, clock=lambda: NOW, autostart=False)
    settlements = FakeSettlements(prev or {})
    return CurveCompare(store, settlements, api, fallback_prices=fallback, clock=lambda: NOW, background=False), store, settlements


def test_view_compares_live_with_previous_and_registers_demand():
    out = outright_prices()
    rows = [taq_row(s, NOW - 5, p - 0.01, p + 0.01) for s, p in out.items()]
    service, store, settlements = make_service(rows, prev={s: p - 1.0 for s, p in out.items()})
    first = service.view("CL", 8, OUTRIGHT)
    assert all(r.point.value is None for r in first.rows)  # nothing polled yet
    store.refresh_once()
    view = service.view("CL", 8, OUTRIGHT)
    assert all(r.point.source == SOURCE_QUOTE for r in view.rows)
    assert view.rows[0].change == pytest.approx(1.0)
    assert view.settlement_date == date(2026, 10, 5) and settlements.fetches == 1
    service.view("CL", 8, OUTRIGHT)
    assert settlements.fetches == 1  # fetched once per day


def test_view_uses_fair_value_when_there_are_no_quotes():
    out = outright_prices()
    service, _, _ = make_service([], fallback=lambda: {s: (p, NOW - 3) for s, p in out.items()})
    view = service.view("CL", 8, OUTRIGHT)
    assert {r.point.source for r in view.rows} == {SOURCE_FAIR}


def test_page_callback_validates_months_and_family(monkeypatch):
    out = outright_prices()
    service, store, _ = make_service([taq_row(s, NOW - 5, p, p + 0.02) for s, p in out.items()])
    monkeypatch.setattr(ccc, "container", Container(curve_compare=service))
    store.request(("seed",), list(out))
    store.refresh_once()
    figure, change, status, table = ccc.render_compare(0, "CL", 8, OUTRIGHT, DFLY_FLIES)
    assert len(figure.data) >= 2 and "8 of 8 symbols priced" in status.children[0].children
    for bad in (0, 37, "", None, 2.5):
        assert "whole number" in ccc.render_compare(0, "CL", bad, OUTRIGHT, DFLY_FLIES)[0].layout.annotations[0].text
    assert "at least 4" in ccc.render_compare(0, "CL", 3, DFLY, DFLY_FLIES)[0].layout.annotations[0].text
    assert ccc.method_disabled(OUTRIGHT) is True and ccc.method_disabled(DFLY) is False


def test_page_callback_without_a_service(monkeypatch):
    monkeypatch.setattr(ccc, "container", Container())
    assert "not running" in ccc.render_compare(0, "CL", 8, OUTRIGHT, DFLY_FLIES)[0].layout.annotations[0].text


def test_layout_and_table_rows_build():
    assert tab.curve_compare_layout() is not None
    assert tab.table_rows(None) == []
    out = outright_prices()
    service, store, _ = make_service([taq_row(s, NOW - 5, p, p + 0.02) for s, p in out.items()])
    view = service.view("CL", 8, SPREAD)
    store.refresh_once()
    rows = tab.table_rows(service.view("CL", 8, SPREAD))
    assert len(rows) == 7 and rows[0]["source"] == SOURCE_DERIVED and rows[0]["how"] == "outright - next outright"


# ----------------------------------------------------------------------
# Staged look-back and the fair value feed
# ----------------------------------------------------------------------


def test_a_symbol_with_no_recent_quote_gets_a_two_day_look_back_a_few_at_a_time():
    api = FakeTaq([])
    store = QuoteStore(api, clock=lambda: NOW, autostart=False)
    symbols = [f"SYM{i}" for i in range(9)]
    store.request(("k",), symbols)
    store.refresh_once()  # short bootstrap for all nine: nothing comes back
    assert all(p["dates"] == ["2026-10-06"] for p in api.calls[0])
    store.refresh_once()
    long_products = [p for p in api.calls[1] if p["dates"] == ["2026-10-05", "2026-10-06"]]
    assert len(long_products) == 5  # at most LONG_CHUNK per request
    assert all((p["start"], p["end"]) == ("00:00:00", "23:59:59") for p in long_products)
    store.refresh_once()
    assert sum(p["dates"] == ["2026-10-05", "2026-10-06"] for p in api.calls[2]) == 4  # the rest, none repeated


def test_a_quote_found_in_the_long_look_back_is_kept_and_then_polled_cheaply():
    api = FakeTaq([taq_row("CLG28", NOW - 20 * 3600, 74.93, 75.51)])
    store = QuoteStore(api, clock=lambda: NOW, autostart=False)
    store.request(("k",), ["CLG28"])
    store.refresh_once()  # short window: the 20-hour-old row is outside it ... but FakeTaq ignores windows
    assert store.quotes()["CLG28"].ts == NOW - 20 * 3600
    store.refresh_once()
    assert api.calls[1][0]["dates"] == ["2026-10-06"] and api.calls[1][0]["start"] == "11:58:00"


class FakeFair:
    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.calls = rows or [], error, []

    def fairvalue(self, products, **kwargs):
        self.calls.append(list(products))
        if self.error:
            raise self.error
        return {"data": [r for r in self.rows if r["Contract"] in products]}


def fair_row(symbol, price, age=14000):
    return {"Contract": symbol, "Price": price, "Timestamp": int((NOW - age) * 1000)}


def test_fair_values_keep_their_own_timestamps_and_are_refreshed_only_when_due():
    clock = {"t": NOW}
    api = FakeFair([fair_row("CLZ28", 71.08), fair_row("CLK29", 69.28, age=999)])
    store = FairValueStore(api, clock=lambda: clock["t"], autostart=False, refresh_seconds=600)
    store.request(["CLZ28", "CLK29", "CLV29"])
    assert store.refresh_once() == 2
    assert store.prices()["CLZ28"] == (71.08, NOW - 14000)
    assert store.refresh_once() == 0 and len(api.calls) == 1  # not due yet
    clock["t"] += 601
    store.request(["CLZ28", "CLK29", "CLV29"])
    assert store.refresh_once() == 2 and len(api.calls) == 2


def test_fair_value_chunks_and_error_backoff():
    api = FakeFair([fair_row(f"S{i}", float(i)) for i in range(250)])
    store = FairValueStore(api, clock=lambda: NOW, autostart=False, chunk_size=100)
    store.request([f"S{i}" for i in range(250)])
    assert store.refresh_once() == 250 and [len(c) for c in api.calls] == [100, 100, 50]
    failing = FairValueStore(FakeFair(error=RateLimitError("slow")), clock=lambda: NOW, autostart=False)
    failing.request(["CLZ28"])
    assert failing.refresh_once() == 0 and "failed" in failing.last_error
    assert failing.refresh_once() == 0 and len(failing._api.calls) == 1  # backs off instead of retrying at once


def test_service_merges_fair_value_store_quotes_and_the_live_poll():
    out = outright_prices()
    contracts = generic_contracts("CL", AS_OF, 8)
    far = contracts[7].symbol
    store_fair = FakeFairStore({far: (50.0, NOW - 9 * 3600)})
    service, _, _ = make_service([], fallback=lambda: {contracts[0].symbol: (81.0, NOW - 2)})
    service._fair_store = store_fair
    view = service.view("CL", 8, OUTRIGHT)
    assert view.rows[0].point.value == 81.0 and view.rows[0].point.age_seconds == 2
    assert view.rows[7].point.value == 50.0 and view.rows[7].point.age_seconds == 9 * 3600
    assert view.rows[3].point.value is None and view.symbols_priced == 2
    assert store_fair.requested == symbols_needed("CL", AS_OF, 8, OUTRIGHT)


class FakeFairStore:
    last_error = None

    def __init__(self, prices):
        self._prices, self.requested = prices, None

    def request(self, symbols):
        self.requested = list(symbols)

    def prices(self, symbols=None):
        return {s: v for s, v in self._prices.items() if symbols is None or s in symbols}


# ----------------------------------------------------------------------
# /taq/ pages: oldest first, so the newest quotes are on the LAST page
# ----------------------------------------------------------------------


def test_a_multi_page_reply_is_read_from_the_newest_page_backwards():
    pages = [
        [taq_row("CLZ26", NOW - 3000, 10.0, 10.2), taq_row("CLF27", NOW - 2900, 20.0, 20.2)],  # page 1: the oldest rows
        [taq_row("CLZ26", NOW - 1500, 11.0, 11.2)],
        [taq_row("CLZ26", NOW - 10, 12.0, 12.2), taq_row("CLF27", NOW - 8, 21.0, 21.2)],  # last page: the newest
    ]
    api = FakeTaq(pages=pages)
    store = QuoteStore(api, clock=lambda: NOW, autostart=False)
    store.request(("k",), ["CLZ26", "CLF27"])
    store.refresh_once()
    assert store.quotes()["CLZ26"].bid == 12.0 and store.quotes()["CLF27"].bid == 21.0  # not page 1's old quotes
    assert api.page_calls == [1, 3]  # page 1, then the last page; both symbols seen, so page 2 is not read


def test_older_pages_are_read_only_while_a_symbol_is_still_unseen():
    pages = [
        [taq_row("CLF27", NOW - 3000, 20.0, 20.2)],  # CLF27 only quoted early in the window
        [taq_row("CLZ26", NOW - 1500, 11.0, 11.2)],
        [taq_row("CLZ26", NOW - 10, 12.0, 12.2)],
    ]
    api = FakeTaq(pages=pages)
    store = QuoteStore(api, clock=lambda: NOW, autostart=False)
    store.request(("k",), ["CLZ26", "CLF27", "CLG27"])  # CLG27 has no rows at all
    store.refresh_once()
    assert api.page_calls == [1, 3, 2]  # newest first, then back; never more than MAX_EXTRA_PAGES
    assert store.quotes()["CLZ26"].bid == 12.0 and store.quotes()["CLF27"].bid == 20.0 and "CLG27" not in store.quotes()


def test_at_most_a_few_extra_pages_are_read():
    from core import quote_store

    pages = [[taq_row("OTHER", NOW - i, 1.0, 1.2)] for i in range(20)]
    api = FakeTaq(pages=pages)
    store = QuoteStore(api, clock=lambda: NOW, autostart=False)
    store.request(("k",), ["CLZ26"])
    store.refresh_once()
    assert len(api.page_calls) == 1 + quote_store.MAX_EXTRA_PAGES


def test_the_api_wrapper_sends_the_page_number():
    from adapters.qh_api import make_api

    class Session:
        def __init__(self):
            self.calls = []

        def request(self, method, url, params=None, json=None, headers=None, timeout=None):
            self.calls.append({"method": method, "url": url, "params": params, "json": json})

            class Response:
                status_code, headers, text = 200, {}, ""

                @staticmethod
                def json():
                    return {"data": []}

            return Response()

    from adapters.qh_api.client import QHClient
    from adapters.qh_api import QHApi

    session = Session()
    api = QHApi(QHClient(token_provider=lambda: "t", session=session, sleep=lambda s: None))
    api.taq([{"id": "CLZ26", "dates": ["2026-10-01"]}])
    api.taq([{"id": "CLZ26", "dates": ["2026-10-01"]}], page=3)
    assert session.calls[0]["params"] is None and session.calls[1]["params"] == {"page": "3"}


# ----------------------------------------------------------------------
# Derived values must combine prices from the same moment
# ----------------------------------------------------------------------


def test_a_spread_is_not_built_from_outrights_priced_hours_apart():
    out = outright_prices()
    contracts = generic_contracts("CL", AS_OF, 8)
    book = {s: (p, NOW - 4 * 3600, SOURCE_FAIR) for s, p in out.items()}  # the fair-value batch, 4 hours old
    book[contracts[0].symbol] = (out[contracts[0].symbol] + 1.5, NOW - 5, SOURCE_QUOTE)  # one outright quoted just now, the market moved
    spread = build_curve("CL", AS_OF, 8, SPREAD, book, NOW)
    assert spread[0].value is None  # the mix is rejected: it would be off by the 1.5 move
    assert spread[1].value == pytest.approx(out[contracts[1].symbol] - out[contracts[2].symbol])  # a consistent pair is fine


def test_the_mixed_point_is_rebuilt_from_a_single_consistent_basis():
    out = outright_prices()
    contracts = generic_contracts("CL", AS_OF, 8)
    fair = {s: (p, NOW - 4 * 3600) for s, p in out.items()}
    quotes = {contracts[0].symbol: Quote(out[contracts[0].symbol] + 1.4, out[contracts[0].symbol] + 1.6, 1, 1, NOW - 5)}
    mixed = build_price_book(quotes, fair)
    views = [mixed, build_price_book({}, fair), build_price_book(quotes, {})]
    spread = build_curve_views("CL", AS_OF, 8, SPREAD, views, NOW)
    expected = out[contracts[0].symbol] - out[contracts[1].symbol]  # the fair-value batch alone, not the mix
    assert spread[0].value == pytest.approx(expected) and spread[0].source == SOURCE_DERIVED
    assert build_curve("CL", AS_OF, 8, SPREAD, mixed, NOW)[0].value is None  # the mixed book alone cannot


def test_a_proper_value_from_a_later_book_beats_an_old_price_from_the_first():
    out = outright_prices()
    contracts = generic_contracts("CL", AS_OF, 8)
    spread_symbol = f"{contracts[0].symbol}-{contracts[1].symbol[-3:]}"
    old = {spread_symbol: (50.0, NOW - 3 * 86400, SOURCE_FAIR)}
    fresh = {s: (p, NOW - 60, SOURCE_QUOTE) for s, p in out.items()}
    spread = build_curve_views("CL", AS_OF, 8, SPREAD, [old, fresh], NOW)
    assert spread[0].value == pytest.approx(out[contracts[0].symbol] - out[contracts[1].symbol])


def test_a_clearly_fresher_built_price_replaces_a_stale_own_price_but_not_a_similar_one():
    out = outright_prices()
    contracts = generic_contracts("CL", AS_OF, 8)
    spread_symbol = f"{contracts[0].symbol}-{contracts[1].symbol[-3:]}"
    built = out[contracts[0].symbol] - out[contracts[1].symbol]
    fresh_outrights = {s: (p, NOW - 30, SOURCE_QUOTE) for s, p in out.items()}
    stale_own = {**fresh_outrights, spread_symbol: (built + 0.5, NOW - 4 * 3600, SOURCE_FAIR)}
    assert build_curve("CL", AS_OF, 8, SPREAD, stale_own, NOW)[0].value == pytest.approx(built)
    similar_own = {**fresh_outrights, spread_symbol: (built + 0.5, NOW - 60, SOURCE_QUOTE)}
    assert build_curve("CL", AS_OF, 8, SPREAD, similar_own, NOW)[0].value == pytest.approx(built + 0.5)


def test_the_page_warns_when_taq_runs_behind():
    out = outright_prices()
    old = NOW - 15 * 3600  # yesterday evening's quotes
    service, store, _ = make_service([taq_row(s, old, p, p + 0.02) for s, p in out.items()])
    service.view("CL", 8, OUTRIGHT)
    store.refresh_once()
    view = service.view("CL", 8, OUTRIGHT)
    assert view.quote_lag_seconds == pytest.approx(15 * 3600, abs=1)
    assert "The newest TAQ quote is 15.0h old" in str(tab.build_status(view))
    no_quotes_service, _, _ = make_service([])
    empty = no_quotes_service.view("CL", 8, OUTRIGHT)
    assert empty.quote_lag_seconds is None and "No TAQ quote yet" in str(tab.build_status(empty))
    fresh_service, fresh_store, _ = make_service([taq_row(s, NOW - 5, p, p + 0.02) for s, p in out.items()])
    fresh_service.view("CL", 8, OUTRIGHT)
    fresh_store.refresh_once()
    status = str(tab.build_status(fresh_service.view("CL", 8, OUTRIGHT)))
    assert "TAQ can run behind" not in status and "No TAQ quote yet" not in status


# ----------------------------------------------------------------------
# Quote quality: junk-wide quotes are ignored
# ----------------------------------------------------------------------


def test_a_junk_wide_outright_quote_is_ignored_and_the_fair_value_is_used():
    wide = Quote(87.05, 93.45, 1, 1, 100.0)  # COU27 on a real day: midpoint 1.9 away from the fair value
    book = build_price_book({"COU27": wide}, {"COU27": (88.37, 90.0)})
    assert book["COU27"] == (88.37, 90.0, SOURCE_FAIR)  # the quote is newer but not trustworthy
    assert "COU27" not in build_price_book({"COU27": wide}, {})  # and with no fair value, nothing rather than junk


def test_the_width_limit_depends_on_the_kind_of_symbol():
    from core.curve_compare import MAX_QUOTE_WIDTH, symbol_kind, usable_quote

    assert [symbol_kind(s) for s in ("CLZ26", "CLZ26-F27", "CLZ26-F27-G27")] == ["outright", "spread", "fly"]
    assert usable_quote("CLZ26", Quote(80.0, 80.0 + MAX_QUOTE_WIDTH["outright"], 1, 1, 1.0))
    assert not usable_quote("CLZ26", Quote(80.0, 80.5, 1, 1, 1.0))
    assert usable_quote("CLZ26-F27", Quote(1.0, 1.05, 1, 1, 1.0)) and not usable_quote("CLZ26-F27", Quote(1.0, 1.2, 1, 1, 1.0))
    assert not usable_quote("CLZ26-F27-G27", Quote(0.0, 0.3, 1, 1, 1.0))
    assert not usable_quote("CLZ26", Quote(80.1, 80.0, 1, 1, 1.0))  # crossed


def test_a_spread_is_not_derived_from_a_junk_wide_outright_quote():
    out = outright_prices()
    contracts = generic_contracts("CL", AS_OF, 8)
    quotes = {contracts[1].symbol: Quote(out[contracts[1].symbol] - 3.0, out[contracts[1].symbol] + 3.0 + 0.4, 1, 1, NOW - 5)}  # wide, lopsided
    fair = {s: (p, NOW - 60) for s, p in out.items()}
    views = [build_price_book(quotes, fair), build_price_book({}, fair), build_price_book(quotes, {})]
    spread = build_curve_views("CL", AS_OF, 8, SPREAD, views, NOW)
    assert spread[0].value == pytest.approx(out[contracts[0].symbol] - out[contracts[1].symbol])  # from the fair values


def test_the_status_counts_the_quotes_it_ignored():
    out = outright_prices()
    service, store, _ = make_service([taq_row(s, NOW - 5, p - 2.0, p + 2.0) for s, p in out.items()])
    service.view("CL", 8, OUTRIGHT)
    store.refresh_once()
    view = service.view("CL", 8, OUTRIGHT)
    assert view.symbols_wide == 8 and view.symbols_quoted == 8 and view.count(SOURCE_QUOTE) == 0
    assert "8 quote(s) ignored as too wide" in str(tab.build_status(view))


def test_default_months_is_32_and_zoom_is_kept_across_refreshes():
    layout = tab.curve_compare_layout()
    months = next(x for x in _walk(layout) if getattr(x, "id", None) == "compare-months")
    assert months.value == 32 and tab.DEFAULT_MONTHS == 32
    out = outright_prices()
    service, store, _ = make_service([taq_row(s, NOW - 5, p, p + 0.02) for s, p in out.items()])
    service.view("CL", 8, OUTRIGHT)
    store.refresh_once()
    first = tab.build_compare_figure(service.view("CL", 8, OUTRIGHT))
    store.refresh_once()
    again = tab.build_compare_figure(service.view("CL", 8, OUTRIGHT))
    assert first.layout.uirevision == again.layout.uirevision  # same selection: the zoom survives the refresh
    other = tab.build_compare_figure(service.view("CL", 8, SPREAD))
    assert other.layout.uirevision != first.layout.uirevision  # a different curve starts unzoomed


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and not isinstance(child, str):
            yield from _walk(child)
