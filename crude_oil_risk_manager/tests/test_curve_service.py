"""CurveService: states, kinks, alerts and their dedupe, event log, snapshots, buffer day, jobs, data quality."""

from datetime import date, datetime, timezone

import pytest

from core.curve_calendar import DFLY, FAMILIES, FLY, OUTRIGHT, SPREAD
from core.curve_history import CurveHistoryStore
from core.curve_service import CurveService, structure_value
from core.curve_settings import CurveParams, save_params
from core.curve_store import PrevSettlementStore, SnapshotStore
from db.repository import Repository
from tests.curve_fakes import NOW, FakeAlerts, FakeApi, MutableClock, make_prices

PRODUCTS = ("BRN", "CL")


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture
def clock():
    return MutableClock()


@pytest.fixture
def alerts():
    return FakeAlerts()


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def service(repo, tmp_path, clock, alerts, api):
    return CurveService(
        repo, api, CurveHistoryStore(tmp_path / "curves", api),
        PrevSettlementStore(tmp_path / "curves" / "prev.json"), SnapshotStore(tmp_path / "curves"),
        alerts, products=PRODUCTS, clock=clock, run_in_background=False,
    )


def feed(service, clock, **kwargs):
    service.on_prices(make_prices(PRODUCTS, clock().date(), clock(), **kwargs))
    return service.state()


def test_symbols_to_poll_cover_both_products_ladders(service):
    symbols = service.symbols_to_poll()
    assert len(symbols) == 2 * 42 and "COZ26-F27-G27" in symbols and "CLX26" in symbols


def test_state_has_every_product_and_family_with_live_values(service, clock):
    state = feed(service, clock)
    assert set(state.families) == {(p, f) for p in PRODUCTS for f in FAMILIES}
    sizes = {f: len(state.families[("BRN", f)].positions) for f in FAMILIES}
    assert sizes == {OUTRIGHT: 15, SPREAD: 14, FLY: 13, DFLY: 12}
    assert all(p.live is not None for p in state.families[("BRN", OUTRIGHT)].positions)
    assert state.updated_at == NOW and state.as_of == date(2026, 10, 6)


def test_dfly_live_value_is_fly_minus_next_fly(service, clock):
    state = feed(service, clock)
    flies = state.families[("BRN", FLY)].positions
    dfly = state.families[("BRN", DFLY)].positions[0]
    assert dfly.live == pytest.approx(flies[0].live - flies[1].live)


def test_structure_value_is_none_when_a_component_is_missing(service, clock):
    ladder = service.ladder("BRN", clock().date())
    assert structure_value(ladder[FLY][0], {}) is None
    assert structure_value(ladder[DFLY][0], {ladder[FLY][0].symbol: (1.0, 0.0)}) is None


def test_empty_poll_leaves_the_previous_state(service, clock):
    state = feed(service, clock)
    service.on_prices({})
    assert service.state() is state


def test_an_injected_kink_is_found_alerted_once_and_logged(service, clock, alerts, repo):
    state = feed(service, clock, kinks={"COG27": 0.4})  # BRN Feb27 outright
    outright_kinks = [k for k in state.kinks if k["product"] == "BRN" and k["family"] == OUTRIGHT]
    assert [k["label"] for k in outright_kinks] == ["Feb27"] and outright_kinks[0]["direction"] == "rich"
    assert 1 <= len(alerts.sent) <= 2  # batched: one alert per product, not one per related kink
    assert any("Feb27" in title + body for _, title, body in alerts.sent)
    first_count = len(alerts.sent)

    feed(service, clock, kinks={"COG27": 0.4})  # still there: no repeat
    assert len(alerts.sent) == first_count

    events = repo.get_kink_events()
    open_events = [e for e in events if e["cleared_at"] is None]
    assert open_events and any(e["label"] == "Feb27" and e["family"] == OUTRIGHT for e in open_events)

    feed(service, clock)  # kink gone: events close
    assert all(e["cleared_at"] is not None for e in repo.get_kink_events())
    assert not service.state().kinks


def test_a_cleared_kink_returning_inside_the_cooldown_does_not_realert(service, clock, alerts):
    feed(service, clock, kinks={"COG27": 0.4})
    sent = len(alerts.sent)
    feed(service, clock)
    clock.advance(minutes=10)
    feed(service, clock, kinks={"COG27": 0.4})
    assert len(alerts.sent) == sent  # inside the 60 minute cooldown
    feed(service, clock)
    clock.advance(minutes=61)
    feed(service, clock, kinks={"COG27": 0.4})
    assert len(alerts.sent) > sent


def test_a_curve_with_most_points_flagged_is_treated_as_a_data_mismatch_not_kinks(service, clock, alerts, repo):
    repo.set_setting("api_access_token", "tok")
    feed(service, clock)  # loads the (consistent) history
    clock.advance(minutes=20)
    # A saw-tooth the history never shows: against that history nearly every point is off.
    saw = {s.symbol: (0.6 if i % 2 else -0.6) for i, s in enumerate(service.ladder("BRN", clock().date())[OUTRIGHT])}
    alerts.sent.clear()
    state = feed(service, clock, kinks=saw)
    view = state.families[("BRN", OUTRIGHT)]
    assert view.suspect
    assert not [k for k in state.kinks if k["product"] == "BRN" and k["family"] == OUTRIGHT]
    assert not any("BRN outright" in t for t in alerts.titles())
    assert any("data mismatch" in f.text and f.product == "BRN" for f in state.quality)
    assert not state.families[("CL", OUTRIGHT)].suspect  # CL was fine


def test_related_kinks_are_batched_into_one_alert_per_product(service, clock, alerts):
    state = feed(service, clock, kinks={"COG27": 0.4})
    brn_kinks = [k for k in state.kinks if k["product"] == "BRN"]
    assert len(brn_kinks) > 1  # the outright kink also shows in the spread / fly / dfly curves built from it
    brn_alerts = [(t, b) for _, t, b in alerts.sent if "BRN" in t]
    assert len(brn_alerts) == 1
    title, body = brn_alerts[0]
    assert f"+{len(brn_kinks) - 1} related" in title and "Also:" in body


def test_alerts_can_be_switched_off_but_kinks_still_show(service, clock, alerts, repo):
    save_params(repo, CurveParams(alerts_enabled=False))
    state = feed(service, clock, kinks={"COG27": 0.4})
    assert state.kinks and alerts.sent == []


def test_minimum_priority_filters_alerts(service, clock, alerts, repo):
    save_params(repo, CurveParams(alert_min_priority="HIGH", min_methods_high=4))  # HIGH needs all four methods
    state = feed(service, clock, kinks={"COG27": 0.4})
    assert state.kinks and alerts.sent == []


def test_thresholds_apply_on_the_next_compute(service, clock, repo):
    loose = feed(service, clock, kinks={"COG27": 0.05})
    save_params(repo, CurveParams(z_fit=15, z_neighbour=15, z_pca=15, z_history=15))
    strict = feed(service, clock, kinks={"COG27": 0.05})
    assert len(strict.kinks) < len(loose.kinks)


def test_buffer_day_alert_and_flag_are_raised_once(service, clock, alerts):
    clock.now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)  # Brent Dec26 buffer day
    state = feed(service, clock)
    assert any(f.severity == "BUFFER" and "COZ26" in f.text for f in state.quality)
    assert sum("Buffer day" in t for t in alerts.titles()) == 1
    feed(service, clock)
    assert sum("Buffer day" in t for t in alerts.titles()) == 1
    clock.now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
    assert not any(f.severity == "BUFFER" for f in feed(service, clock).quality)


def test_curves_roll_forward_on_their_own_at_expiry(service, clock):
    before = feed(service, clock).families[("BRN", OUTRIGHT)].positions[0].structure.symbol
    clock.now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    after = feed(service, clock).families[("BRN", OUTRIGHT)].positions[0].structure.symbol
    assert (before, after) == ("COZ26", "COF27")


def test_missing_prices_become_gaps_and_a_quality_warning(service, clock):
    state = feed(service, clock, skip={"COH27"})
    outright = state.families[("BRN", OUTRIGHT)].positions
    assert [p.live is None for p in outright].count(True) == 1
    assert any("live prices missing" in f.text and f.product == "BRN" for f in state.quality)


def test_stale_prices_are_flagged(service, clock):
    prices = make_prices(PRODUCTS, clock().date(), clock())
    prices["COZ26"] = (prices["COZ26"][0], clock().timestamp() - 600)
    service.on_prices(prices)
    state = service.state()
    assert state.families[("BRN", OUTRIGHT)].positions[0].stale
    assert any("older than" in f.text for f in state.quality)


def test_quality_notes_missing_history_and_settlements_before_they_are_loaded(service, clock):
    texts = [f.text for f in feed(service, clock).quality]
    assert any("No history loaded" in t for t in texts)
    assert any("settlements not fetched" in t for t in texts)


def test_snapshots_are_saved_and_throttled(service, clock):
    feed(service, clock)
    assert service.snapshots.dates("BRN") == ["2026-10-06"]
    saved = service.snapshots.load("BRN", "2026-10-06", OUTRIGHT)
    assert len(saved) == 15 and "Dec26" in saved
    feed(service, clock, kinks={"COG27": 1.0})
    assert service.snapshots.load("BRN", "2026-10-06", OUTRIGHT) == saved  # inside the 15 minute window
    clock.advance(minutes=16)
    feed(service, clock, kinks={"COG27": 1.0})
    assert service.snapshots.load("BRN", "2026-10-06", OUTRIGHT) != saved


def test_startup_closes_events_left_open_by_a_previous_run(repo, tmp_path, clock, alerts, api):
    repo.open_kink_event("BRN", OUTRIGHT, 3, "Feb27", "COG27", "rich", 5.0, "HIGH", 3, False, {})
    CurveService(repo, api, CurveHistoryStore(tmp_path, api), PrevSettlementStore(tmp_path / "p.json"),
                 SnapshotStore(tmp_path), alerts, clock=clock, run_in_background=False)
    assert repo.get_kink_events()[0]["cleared_at"] is not None


# ---------- background jobs ----------


def test_jobs_do_nothing_without_a_token(service, clock, api):
    feed(service, clock)
    assert api.generic_calls == [] and api.settlement_calls == []


def test_history_and_settlements_load_once_when_a_token_is_set(service, clock, api, repo):
    repo.set_setting("api_access_token", "tok")
    feed(service, clock)
    assert len(api.generic_calls) == 6 and len(api.settlement_calls) == 1  # 2 products x 3 chunks, one settlement call
    assert service.history.matrix("BRN", OUTRIGHT) is not None
    assert service.settlements.fetched_on == "2026-10-06"
    assert "refreshed" in service.job_messages["history"]
    feed(service, clock)
    assert len(api.generic_calls) == 6 and len(api.settlement_calls) == 1  # not again the same day

    state = service.state()
    prev = state.families[("BRN", OUTRIGHT)].positions[0].prev_settle
    assert prev is not None and state.families[("BRN", OUTRIGHT)].history_rows > 200
    assert state.families[("BRN", SPREAD)].positions[0].prev_settle is not None


def test_settlements_refetch_the_next_local_day_after_the_opening_time(service, clock, api, repo):
    repo.set_setting("api_access_token", "tok")
    repo.set_setting("curve_open_time", "07:00")
    feed(service, clock)
    clock.now = datetime(2026, 10, 7, 6, 0, tzinfo=timezone.utc)  # before the 07:00 opening
    feed(service, clock)
    assert len(api.settlement_calls) == 1
    clock.now = datetime(2026, 10, 7, 7, 30, tzinfo=timezone.utc)
    feed(service, clock)
    assert len(api.settlement_calls) == 2


def test_a_failed_history_refresh_is_retried_later_not_marked_done(service, clock, api, repo):
    repo.set_setting("api_access_token", "tok")
    api.fail_generic = True
    feed(service, clock)
    assert "incomplete" in service.job_messages["history"]
    assert repo.get_setting("curve_history_refreshed_on") is None
    calls = len(api.generic_calls)
    clock.advance(minutes=5)
    feed(service, clock)
    assert len(api.generic_calls) == calls  # waiting out the retry delay
    api.fail_generic = False
    clock.advance(minutes=30)
    feed(service, clock)
    assert repo.get_setting("curve_history_refreshed_on") == "2026-10-06"


def test_history_makes_the_pca_and_history_methods_available(service, clock, repo):
    repo.set_setting("api_access_token", "tok")
    state = feed(service, clock)
    state = feed(service, clock)
    assert "pca" in state.families[("BRN", SPREAD)].methods_available
    assert "history" in state.families[("BRN", SPREAD)].methods_available
    assert "history" not in state.families[("BRN", OUTRIGHT)].methods_available
