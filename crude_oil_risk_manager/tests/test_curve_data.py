"""Curve history cache, previous-settlement store and snapshot store (no network)."""

from datetime import date

import numpy as np
import pytest

from adapters.base import APIError
from core.curve_calendar import DFLY, FLY, OUTRIGHT, SPREAD, build_structures
from core.curve_history import GROUPS_PER_CALL, CurveHistoryStore, generic_code
from core.curve_store import PrevSettlementStore, SnapshotStore
from tests.curve_fakes import FakeApi

TODAY = date(2026, 10, 6)


def test_generic_codes():
    assert generic_code("BRN", OUTRIGHT, 3) == "CO3"
    assert generic_code("BRN", SPREAD, 1) == "CO1-2"
    assert generic_code("CL", FLY, 2) == "CL2-3-4"
    with pytest.raises(ValueError):
        generic_code("CL", DFLY, 1)


def test_only_position_1_and_the_spreads_are_downloaded_in_chunks_and_todays_bar_is_dropped(tmp_path):
    api = FakeApi()
    store = CurveHistoryStore(tmp_path, api)
    report = store.refresh("BRN", years=1, today=TODAY)
    assert report.ok and report.series_total == 15 == report.series_updated
    assert len(api.generic_calls) == 2  # 15 series: 14 + 1 per call
    assert all(len(call) <= GROUPS_PER_CALL for call in api.generic_calls)
    requested = {g["product"] for call in api.generic_calls for g in call}
    assert requested == {"CO1"} | {f"CO{k}-{k + 1}" for k in range(1, 15)}  # no far outrights, no flies
    group = api.generic_calls[0][0]
    assert group["resolve"] == 1 and group["gap"] == 0 and group["interval"] == "1D" and group["count"] == 290
    frame = store.read_series("BRN", "CO1")
    assert frame["date"].max().date() == date(2026, 10, 5)  # 6 Oct is today's unfinished bar
    assert (frame["contract"] == "COZ26").all()


def test_second_refresh_only_pulls_the_gap_and_adds_new_days(tmp_path):
    api = FakeApi()
    store = CurveHistoryStore(tmp_path, api)
    store.refresh("BRN", years=1, today=TODAY)
    api.generic_calls.clear()
    assert store.refresh("BRN", years=1, today=TODAY).rows_added == 0
    assert api.generic_calls[0][0]["count"] == 8  # (6 Oct - 5 Oct) + 7, not a full reload

    api.today = date(2026, 10, 7)
    report = store.refresh("BRN", years=1, today=date(2026, 10, 7))
    assert report.rows_added == 15  # 6 Oct, one new day for every downloaded series
    assert store.last_date("BRN", "CO1") == date(2026, 10, 6)


def test_version_changes_when_data_is_stored(tmp_path):
    store = CurveHistoryStore(tmp_path, FakeApi())
    before = store.version
    store.refresh("CL", years=1, today=TODAY)
    assert store.version == before + 1


def test_failed_calls_are_reported_per_series_and_nothing_is_stored(tmp_path):
    store = CurveHistoryStore(tmp_path, FakeApi(fail_generic=True))
    report = store.refresh("BRN", years=1, today=TODAY)
    assert not report.ok and len(report.errors) == 15 and report.series_updated == 0
    assert store.read_series("BRN", "CO1") is None and store.matrix("BRN", OUTRIGHT) is None


def test_every_curve_type_is_derived_with_the_full_history(tmp_path):
    store = CurveHistoryStore(tmp_path, FakeApi())
    store.refresh("BRN", years=1, today=TODAY)
    shapes = {family: store.matrix("BRN", family).values.shape[1] for family in (OUTRIGHT, SPREAD, FLY, DFLY)}
    assert shapes == {OUTRIGHT: 15, SPREAD: 14, FLY: 13, DFLY: 12}
    rows = {store.matrix("BRN", family).rows for family in (OUTRIGHT, SPREAD, FLY, DFLY)}
    assert len(rows) == 1 and rows.pop() > 250  # all four share one complete set of days
    assert store.matrix("BRN", OUTRIGHT).last_date == date(2026, 10, 5)


def test_delivery_months_come_from_the_spread_contracts(tmp_path):
    store = CurveHistoryStore(tmp_path, FakeApi())
    store.refresh("BRN", years=1, today=TODAY)
    outright, spread = store.matrix("BRN", OUTRIGHT), store.matrix("BRN", SPREAD)
    assert outright.months.shape == outright.values.shape and spread.months.shape == spread.values.shape
    assert set(np.unique(spread.months)) == {12}  # the fake spreads are all COZ26-F27
    assert set(np.unique(outright.months[:, :14])) == {12} and set(np.unique(outright.months[:, 14])) == {1}
    assert store.matrix("BRN", FLY).months.shape[1] == 13 and store.matrix("BRN", DFLY).months.shape[1] == 12


def test_derived_curves_follow_the_defining_identities(tmp_path):
    store = CurveHistoryStore(tmp_path, FakeApi())
    store.refresh("BRN", years=1, today=TODAY)
    p, s = store.matrix("BRN", OUTRIGHT).values, store.matrix("BRN", SPREAD).values
    fly, dfly = store.matrix("BRN", FLY).values, store.matrix("BRN", DFLY).values
    assert np.allclose(p[:, :-1] - p[:, 1:], s)  # spread k = outright k - outright k+1
    assert np.allclose(fly, p[:, :-2] - 2 * p[:, 1:-1] + p[:, 2:])  # fly k = P_k - 2 P_k+1 + P_k+2
    assert np.allclose(dfly[:, 0], fly[:, 0] - fly[:, 1]) and np.allclose(dfly[:, 5], fly[:, 5] - fly[:, 6])
    assert np.allclose(dfly, s[:, :-2] - 2 * s[:, 1:-1] + s[:, 2:])  # = S_k - 2 S_k+1 + S_k+2


def test_the_derived_far_outright_matches_the_curve_it_was_generated_from(tmp_path):
    """The fake generates smooth curves, so the far outright rebuilt from position 1 and 14 spreads must land on it."""
    store = CurveHistoryStore(tmp_path, FakeApi())
    store.refresh("BRN", years=1, today=TODAY)
    outright = store.matrix("BRN", OUTRIGHT)
    level = 6 * np.sin(date(2026, 10, 5).toordinal() / 45)
    expected = [100.0 - 1.2 * k + 0.03 * k * k + level for k in (0, 7, 14)]
    got = outright.values[-1, [0, 7, 14]]
    assert np.allclose(got, expected, atol=0.4)


# ---------- previous settlements ----------


def test_settlement_fetch_keeps_the_latest_day_before_today(tmp_path):
    store = PrevSettlementStore(tmp_path / "prev.json")
    ladder = build_structures("BRN", TODAY)
    symbols = [s.symbol for s in ladder[OUTRIGHT]]
    api = FakeApi()
    report = store.fetch(api, symbols, TODAY, TODAY)
    assert report == {"fetched": 15, "missing": []}
    assert api.settlement_calls[0]["qhcode"] == ",".join(symbols) and api.settlement_calls[0]["end"] == "2026-10-06"
    assert store.settlement_date() == date(2026, 10, 5)  # the 6 Oct row (today) was ignored
    assert store.fetched_on == "2026-10-06"
    assert store.prices()[symbols[0]] == pytest.approx(99.8)  # 100 - 0.2: the 5 Oct row


def test_structure_settlements_come_from_the_outright_legs(tmp_path):
    store = PrevSettlementStore(tmp_path / "prev.json")
    ladder = build_structures("BRN", TODAY)
    store.fetch(FakeApi(), [s.symbol for s in ladder[OUTRIGHT]], TODAY, TODAY)
    prices = store.prices()
    legs = [s.symbol for s in ladder[OUTRIGHT]]
    spread = store.structure_settlement(ladder[SPREAD][0])
    assert spread == pytest.approx(prices[legs[0]] - prices[legs[1]])
    fly = store.structure_settlement(ladder[FLY][0])
    assert fly == pytest.approx(prices[legs[0]] - 2 * prices[legs[1]] + prices[legs[2]])
    dfly = store.structure_settlement(ladder[DFLY][0])
    assert dfly == pytest.approx(
        prices[legs[0]] - 3 * prices[legs[1]] + 3 * prices[legs[2]] - prices[legs[3]]
    )


def test_missing_settlements_are_reported_and_block_derived_structures(tmp_path):
    store = PrevSettlementStore(tmp_path / "prev.json")
    ladder = build_structures("BRN", TODAY)
    only_first = [ladder[OUTRIGHT][0].symbol]

    class Partial(FakeApi):
        def get_all(self, name, **params):
            return super().get_all(name, **{**params, "qhcode": ",".join(only_first)})

    report = store.fetch(Partial(), [s.symbol for s in ladder[OUTRIGHT]], TODAY, TODAY)
    assert report["fetched"] == 1 and len(report["missing"]) == 14
    assert store.structure_settlement(ladder[SPREAD][0]) is None


def test_settlements_persist_across_restarts_and_survive_a_failed_fetch(tmp_path):
    path = tmp_path / "prev.json"
    store = PrevSettlementStore(path)
    ladder = build_structures("BRN", TODAY)
    symbols = [s.symbol for s in ladder[OUTRIGHT]]
    store.fetch(FakeApi(), symbols, TODAY, TODAY)

    class Broken(FakeApi):
        def get_all(self, *args, **kwargs):
            raise APIError("down")

    with pytest.raises(APIError):
        store.fetch(Broken(), symbols, TODAY, TODAY)
    reloaded = PrevSettlementStore(path)
    assert reloaded.prices() == store.prices() and reloaded.fetched_on == "2026-10-06"


# ---------- snapshots ----------


def test_snapshots_are_upserted_per_day_and_listed_newest_first(tmp_path):
    store = SnapshotStore(tmp_path)
    rows = [{"family": "outright", "position": 1, "label": "Dec26", "value": 100.0},
            {"family": "spread", "position": 1, "label": "Dec26/Jan27", "value": 1.5}]
    store.save("BRN", date(2026, 10, 5), rows)
    store.save("BRN", date(2026, 10, 6), rows)
    store.save("BRN", date(2026, 10, 6), [{**rows[0], "value": 101.0}])  # replaces that day only
    assert store.dates("BRN") == ["2026-10-06", "2026-10-05"]
    assert store.load("BRN", "2026-10-06", "outright") == {"Dec26": 101.0}
    assert store.load("BRN", "2026-10-06", "spread") == {}
    assert store.load("BRN", "2026-10-05", "spread") == {"Dec26/Jan27": 1.5}
    assert store.dates("CL") == []


# ---------- the API can fail a whole batch for one awkward series ----------


def test_calls_are_sized_by_total_days_so_a_five_year_load_does_not_trip_the_apis_row_limit(tmp_path):
    from core.curve_history import MAX_ROWS_PER_CALL
    api = FakeApi(max_rows_per_call=10_000)  # the real API failed 14 series x 1,330 days (18,620) as a whole
    store = CurveHistoryStore(tmp_path, api)
    report = store.refresh("BRN", years=5, today=TODAY)
    assert report.ok and report.series_updated == 15
    assert len(api.generic_calls) == 5  # 3 series of 5-year depth per call: no failed call, no retries
    assert all(sum(g["count"] for g in call) <= MAX_ROWS_PER_CALL for call in api.generic_calls)
    assert not any(len(call) == 1 for call in api.generic_calls)


def test_batches_respect_both_the_series_limit_and_the_days_limit():
    pairs = [(f"C{i}", 1330) for i in range(15)]
    sizes = [len(b) for b in CurveHistoryStore._batches(pairs)]
    assert sizes == [3, 3, 3, 3, 3]
    small = [(f"C{i}", 50) for i in range(30)]
    assert [len(b) for b in CurveHistoryStore._batches(small)] == [14, 14, 2]  # many short series: the series limit
    assert CurveHistoryStore._batches([]) == []


def test_a_batch_the_api_fails_as_a_whole_is_retried_one_series_at_a_time(tmp_path):
    api = FakeApi(batch_bug_codes={"CO5-6"})  # the real API's bug: a batch holding this code fails together
    store = CurveHistoryStore(tmp_path, api)
    report = store.refresh("BRN", years=5, today=TODAY)
    assert report.ok and report.series_updated == 15  # nothing lost
    assert store.matrix("BRN", FLY).values.shape[1] == 13
    single_calls = [call for call in api.generic_calls if len(call) == 1]
    assert len(single_calls) == 3  # only the failed batch (3 series) was retried, one by one


def test_the_apis_own_error_text_is_reported_when_a_series_cannot_be_loaded(tmp_path):
    api = FakeApi(code_errors={"CO5-6": "Error fetching generic data: boom"})
    store = CurveHistoryStore(tmp_path, api)
    report = store.refresh("BRN", years=1, today=TODAY)
    assert list(report.errors) == ["CO5-6"] and "boom" in report.errors["CO5-6"]
    assert report.series_updated == 14  # every other series was stored
    assert store.matrix("BRN", OUTRIGHT) is None  # a missing spread means the curve cannot be built, so no half-curve


def test_an_empty_answer_is_not_an_error_when_the_series_is_already_up_to_date(tmp_path):
    api = FakeApi()
    store = CurveHistoryStore(tmp_path, api)
    store.refresh("BRN", years=1, today=TODAY)
    report = store.refresh("BRN", years=1, today=TODAY)  # only today's unfinished bar comes back, which is dropped
    assert report.ok and report.rows_added == 0
