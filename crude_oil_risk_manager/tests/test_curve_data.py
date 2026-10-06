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


def test_first_refresh_loads_every_series_in_chunks_and_drops_todays_unfinished_bar(tmp_path):
    api = FakeApi()
    store = CurveHistoryStore(tmp_path, api)
    report = store.refresh("BRN", years=1, today=TODAY)
    assert report.ok and report.series_total == 15 + 14 + 13 == report.series_updated
    assert len(api.generic_calls) == 3  # 42 series / 14 per call
    assert all(len(call) <= GROUPS_PER_CALL for call in api.generic_calls)
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
    assert report.rows_added == 42  # 6 Oct, one new day for every series
    assert store.last_date("BRN", "CO1") == date(2026, 10, 6)


def test_version_changes_when_data_is_stored(tmp_path):
    store = CurveHistoryStore(tmp_path, FakeApi())
    before = store.version
    store.refresh("CL", years=1, today=TODAY)
    assert store.version == before + 1


def test_failed_calls_are_reported_per_series_and_nothing_is_stored(tmp_path):
    store = CurveHistoryStore(tmp_path, FakeApi(fail_generic=True))
    report = store.refresh("BRN", years=1, today=TODAY)
    assert not report.ok and len(report.errors) == 42 and report.series_updated == 0
    assert store.read_series("BRN", "CO1") is None and store.matrix("BRN", OUTRIGHT) is None


def test_matrix_is_aligned_with_delivery_months(tmp_path):
    store = CurveHistoryStore(tmp_path, FakeApi())
    store.refresh("BRN", years=1, today=TODAY)
    outright = store.matrix("BRN", OUTRIGHT)
    assert outright.values.shape[1] == 15 and outright.rows > 250
    assert outright.months.shape == outright.values.shape and set(np.unique(outright.months)) == {12}
    assert outright.last_date == date(2026, 10, 5)
    assert store.matrix("BRN", SPREAD).values.shape[1] == 14
    assert store.matrix("BRN", FLY).values.shape[1] == 13


def test_dfly_history_is_fly_minus_next_fly(tmp_path):
    store = CurveHistoryStore(tmp_path, FakeApi())
    store.refresh("BRN", years=1, today=TODAY)
    fly, dfly = store.matrix("BRN", FLY), store.matrix("BRN", DFLY)
    assert dfly.values.shape[1] == 12
    assert np.allclose(dfly.values[:, 0], fly.values[:, 0] - fly.values[:, 1])
    assert np.allclose(dfly.values[:, 5], fly.values[:, 5] - fly.values[:, 6])


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
