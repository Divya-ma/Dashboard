"""Trigger study on the Strategy Rollover lines: moves that do not count contract switches, triggers, verdicts, scan, rules, forward test."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from core.curve_calendar import contract_expiry
from core.rollover import RolloverOptions, RolloverService, _index, _month_year
from core.rollover_triggers import (
    AHEAD, CASE_MIXED, CASE_NONE, CASE_TOO_FEW, CASE_TREND, DONE, MAX_LINES, MISSING, OPEN, RolloverStudyService, RuleBook, forward_rows,
    leg_products, prepare, rule_from_study, scan_windows, trigger_specs, value_at,
)
from core.seasonality import trading_days_to
from core.trigger_stats import NO_TREND, POSSIBLE, STRONG, TREND
from tests.test_rollover import CLOCK, TODAY, FakeStore, box_store, settlements

FIRST = _index(1, 2024)


def level(index):
    return 60.0 + 3.0 * (index - FIRST)


def planted_store(regime_of, product="CL"):
    """CL contracts of 2024-2027. Around the expiry of the contract before it (day 0 of its generic-2 line) each contract's price ramps by
    regime_of(index) * 0.5 between day -10 and day +10; the rest of the price is a level that rises with the contract."""
    def price_of(index, day):
        anchor = contract_expiry(product, *_month_year(index - 1))
        x = -int(trading_days_to(pd.DatetimeIndex([day]), anchor, "nymex")[0])
        return level(index) + regime_of(index) * 0.5 * min(max((x + 10) / 20.0, 0.0), 1.0)

    return FakeStore({product: settlements(product, price_of)})


def study_service(store):
    return RolloverStudyService(RolloverService(store, clock=CLOCK), FakeRepo())


class FakeRepo:
    def __init__(self):
        self.values = {}

    def get_setting(self, key, default=None):
        return self.values.get(key, default)

    def set_setting(self, key, value):
        self.values[key] = value


GENERIC = RolloverOptions(generic=2, expiries=1)
MEDIAN = FIRST + 22  # the contracts after it rise over the window, the ones before it fall


def mixed_regime(index):
    return 1.0 if index > MEDIAN else -1.0


def study_of(regime_of, lines=36, options=GENERIC, window=(-10, 10)):
    return study_service(planted_store(regime_of)).study("CL", options, window[0], window[1], 10, lines)


# ----------------------------------------------------------------------
# A line as a series that does not count its switches
# ----------------------------------------------------------------------


def box_line():
    result = RolloverService(box_store(), clock=CLOCK).view("CLBZ", RolloverOptions(generic=1, comparisons=3, before_days=20))
    return next(line for line in result.lines if not line.live)


def test_the_clbz_line_switches_pair_after_the_cl_expiry_and_again_after_the_brent_expiry():
    line = box_line()
    prep = prepare(line)
    assert prep.is_switch.sum() == 2
    first, second = (int(prep.x[r]) for r in np.flatnonzero(prep.is_switch))
    assert first == 1 and second > first  # the first trading day after the CL expiry, then after the Brent expiry
    assert not prep.is_switch[0] and np.isnan(prep.change[prep.is_switch]).all()
    # the plain price difference includes both jumps; the adjusted series does not
    i, j = prep.pos(-10), prep.pos(10)
    assert prep.price[j] - prep.price[i] == pytest.approx(0.2)  # +1.0 at the CL expiry, -0.8 at the Brent expiry
    assert prep.adjusted[j] - prep.adjusted[i] == pytest.approx(0.0)


def test_a_clbz_move_is_what_the_line_did_without_the_two_switch_days():
    study = study_service(box_store_with_brent()).study("CLBZ", RolloverOptions(generic=1), -10, 10, 10, 8)
    done = study.finished[0]
    assert done.status == DONE and done.move == pytest.approx(0.0) and done.raw_move == pytest.approx(0.2)
    assert [round(s.jump, 6) for s in done.skipped] == [1.0, -0.8]
    assert done.skipped[0].before != done.skipped[0].after and "CL" in done.skipped[0].after
    assert all(s.label == done.label for s in done.skipped)
    assert len(study.skips) == 2 * len(study.finished) and study.stats.mean == pytest.approx(0.0)


def test_a_generic_line_follows_one_contract_so_nothing_is_left_out():
    study = study_of(lambda index: 1.0, lines=8)
    done = study.finished[0]
    assert done.skipped == [] and done.move == pytest.approx(done.raw_move)
    assert done.move == pytest.approx(0.5, abs=0.06)  # (lines are stretched so every expiry shares an x, so day 10 can be a day early)
    assert study.skips == []


def box_store_with_brent():
    frames = dict(box_store()._frames)
    frames["BRN"] = settlements("BRN", lambda index, day: 95.0 + (index - FIRST))
    return FakeStore(frames)


# ----------------------------------------------------------------------
# Windows, statuses and options
# ----------------------------------------------------------------------


def test_lines_that_have_not_finished_the_window_are_pending_and_the_upcoming_expiry_is_included():
    study = study_of(mixed_regime)
    statuses = {o.label: o.status for o in study.pending}
    assert OPEN in statuses.values() and AHEAD in statuses.values()
    assert study.current is study.pending[0] and study.current.status == AHEAD  # the expiry still ahead is the newest
    running = next(o for o in study.pending if o.status == OPEN)
    assert running.so_far is not None and running.move is None and running.entry_date < TODAY
    assert all(o.status == DONE for o in study.finished)


def test_the_study_draws_the_page_selection_with_more_expiries_no_zero_base_and_the_upcoming_line():
    used = RolloverStudyService.study_options(RolloverOptions(generic=3, before_days=5, zero_base=True), -10, 10, 30)
    assert used.generic == 3 and used.comparisons == 30 and not used.zero_base and used.upcoming
    assert used.before_days == 22  # room for the lookback before the entry day
    assert RolloverStudyService.study_options(RolloverOptions(before_days=60), -10, 10, 5).before_days == 60


def test_bad_windows_are_refused_with_a_readable_message():
    service = study_service(planted_store(mixed_regime))
    for window, lookback, lines in [((10, -10), 10, 36), ((0, 0), 10, 36), ((-10, 10), 2, 36), ((-10, 10), 99, 36), ((-10, 10), 10, 0),
                                    ((-10, 10), 10, MAX_LINES + 1), ((-500, 10), 10, 36), ((-10, 500), 10, 36)]:
        with pytest.raises(ValueError):
            service.study("CL", GENERIC, window[0], window[1], lookback, lines)


def test_a_window_the_line_does_not_reach_leaves_those_expiries_out_and_says_so():
    study = study_of(mixed_regime, window=(-10, 60))  # generic 2 with one expiry ends about a month after the front expiry
    assert not study.finished and study.case == CASE_TOO_FEW
    assert any("does not cover this window" in n for n in study.notes) and all(o.status in (MISSING, OPEN, AHEAD) for o in study.obs)


def test_the_study_is_cached_until_the_settlements_change():
    store = planted_store(mixed_regime)
    service = study_service(store)
    first = service.study("CL", GENERIC, -10, 10, 10, 36)
    assert service.study("CL", GENERIC, -10, 10, 10, 36) is first
    store.version = "v2"
    assert service.study("CL", GENERIC, -10, 10, 10, 36) is not first


# ----------------------------------------------------------------------
# Triggers and verdicts
# ----------------------------------------------------------------------


def test_a_trigger_that_decides_the_direction_is_found_and_the_window_is_called_mixed_regimes():
    study = study_of(mixed_regime)
    assert study.stats.verdict != TREND and study.stats.up > 5 and study.stats.down > 5  # the expiries do not agree
    price = next(s for s in study.screens if s.spec.key == "price_CL")
    assert price.result.verdict in (STRONG, POSSIBLE) and price.result.split.flips and price.result.rho > 0.6  # ties among +/-0.5 moves cap it near 0.87
    assert study.case == CASE_MIXED and "hypothesis" in study.case_text
    assert study.rule is not None and study.rule.fit.high_dir == 1 and study.rule.fit.low_dir == -1
    assert study.rule.score.hits >= study.rule.score.trades - 2  # judged by the other expiries, it is nearly always right
    assert study.tested >= 3 and any("by chance" in n for n in study.notes)


def test_a_move_every_expiry_agrees_on_is_a_trend():
    study = study_of(lambda index: 1.0)
    assert study.stats.verdict == TREND and study.stats.hit_rate == 1.0 and study.case == CASE_TREND
    assert "consistent move" in study.case_text


def test_a_move_that_alternates_has_no_trend_and_the_price_level_does_not_explain_it():
    study = study_of(lambda index: 1.0 if index % 2 else -1.0)
    assert study.stats.verdict == NO_TREND and study.case in (CASE_NONE, CASE_MIXED)
    price = next(s for s in study.screens if s.spec.key == "price_CL")
    assert price.result.verdict not in (STRONG, POSSIBLE)


def test_too_few_expiries_is_said_plainly():
    study = study_of(lambda index: 1.0, lines=4)
    assert study.case == CASE_TOO_FEW and "at least 5" in study.case_text and study.rule is None


def test_readings_on_the_entry_day_come_from_the_expiry_the_line_belongs_to():
    study = study_of(mixed_regime)
    obs = study.finished[0]
    assert obs.triggers["curve_leg_CL"] == pytest.approx(-3.0, abs=0.01)  # the contract's price less the next month's (the level rises 3 a month)
    assert obs.triggers["curve_mkt_CL"] == pytest.approx(-3.0, abs=0.51)  # M1 - M2 of the contract that is front on the entry day (it carries the previous ramp)
    assert obs.triggers["momentum"] is not None and obs.entry_date < obs.anchor


def test_triggers_that_compare_with_other_expiries_use_only_earlier_ones():
    study = study_of(mixed_regime)
    chronological = sorted(study.obs, key=lambda o: o.anchor)
    readings = [o.triggers["level_hist"] for o in chronological if o.status in (DONE, OPEN)]
    assert readings[:3] == [None, None, None] and readings[3] is not None  # three earlier expiries are needed


def test_the_trigger_catalogue_for_a_box_has_both_legs_and_the_headline_spread():
    box = {s.key for s in trigger_specs("CLBZ", 10)}
    assert leg_products("CLBZ") == ("CL", "BRN") and leg_products("CL") == ("CL",) and leg_products("BZ") == ("BZ",)
    assert {"mkt_move_box", "price_CL", "price_BRN", "oi_chg_BRN", "curve_mkt_CL"} <= box and "curve_leg_CL" not in box
    plain = {s.key for s in trigger_specs("CL", 10)}
    assert "mkt_move_box" not in plain and "curve_leg_CL" in plain
    assert len(trigger_specs("CLBZ", 10)) == 18 and len(trigger_specs("CL", 10)) == 11


def test_a_box_study_reads_both_legs_front_months_without_their_roll_days_and_splits_the_move():
    study = study_service(box_store_with_brent()).study("CLBZ", RolloverOptions(generic=1), -10, 10, 10, 8)
    done = study.finished[0]
    assert done.triggers["mkt_move_CL"] is not None and done.triggers["mkt_move_BRN"] is not None and done.triggers["mkt_move_box"] is not None
    assert study.rolls and all(e.jump is not None for e in study.rolls)  # roll days of the front-month series that were left out
    parts = done.attribution
    assert set(parts) == {"CL", "Brent", "residual"} and parts["residual"] == pytest.approx(done.move - (parts["CL"] - parts["Brent"]))
    assert "Brent triggers use ICE Brent" in " ".join(study.notes)


def test_value_at_carries_a_value_a_few_days_and_no_further():
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-05", "2026-01-06"]))
    assert value_at(series, pd.Timestamp("2026-01-06")) == 2.0
    assert value_at(series, pd.Timestamp("2026-01-09")) == 2.0
    assert value_at(series, pd.Timestamp("2026-02-01")) is None and value_at(series, pd.Timestamp("2025-01-01")) is None
    assert value_at(None, pd.Timestamp("2026-01-01")) is None


# ----------------------------------------------------------------------
# Scan
# ----------------------------------------------------------------------


def test_the_scan_finds_a_planted_window_and_says_how_many_would_pass_by_luck():
    service = study_service(planted_store(lambda index: 1.0))
    result = service.scan("CL", GENERIC, 10, 36)
    assert result.windows > 20 and result.trends >= 1 and result.rows
    best = result.rows[0]
    assert best.hit_rate == 1.0 and best.direction == 1 and best.x_from < best.x_to
    assert result.chance_trends is not None and 0 <= result.chance_trends < result.windows
    assert all(r.verdict == "trend" for r in result.rows)


def test_a_scan_with_too_few_expiries_says_so():
    result = scan_windows([])
    assert result.rows == [] and "needs 5" in result.note


# ----------------------------------------------------------------------
# Saved rules and the forward test
# ----------------------------------------------------------------------


def test_rules_are_saved_listed_and_deleted():
    service = study_service(planted_store(mixed_regime))
    study = service.study("CL", GENERIC, -10, 10, 10, 36)
    rule = service.save_rule(study)
    assert service.rules() == [rule] and rule["created_on"] == TODAY.isoformat()
    assert (rule["product"], rule["generic"], rule["x_from"], rule["x_to"]) == ("CL", 2, -10, 10) and rule["high_dir"] == 1 and rule["low_dir"] == -1
    assert "position" not in rule and service.delete_rule(rule["id"]) and service.rules() == [] and not service.delete_rule("nope")


def test_a_study_with_no_rule_cannot_be_saved():
    study = study_of(lambda index: 1.0 if index % 2 else -1.0, lines=4)
    with pytest.raises(ValueError, match="no rule"):
        rule_from_study(study, TODAY)
    assert RuleBook(FakeRepo()).all() == []


def test_a_saved_rule_is_followed_through_waiting_open_and_finished_expiries():
    service = study_service(planted_store(mixed_regime))
    rule = service.save_rule(service.study("CL", GENERIC, -10, 10, 10, 36))
    study = service.study("CL", GENERIC, -10, 10, 10, 36)

    today_rows = {r["expiry"]: r for r in forward_rows(rule, study.obs)}
    assert {r["status"] for r in today_rows.values()} == {"open", "waiting: the window has not started"}  # nothing finished since it was saved
    open_row = next(r for r in today_rows.values() if r["status"] == "open")
    assert open_row["forward"].startswith("no") and open_row["expect"] in ("up", "down") and open_row["result"] == ""

    older = {**rule, "created_on": "2024-01-01"}  # as if it had been saved long ago
    rows = forward_rows(older, study.obs)
    finished = [r for r in rows if r["status"] == "finished"]
    assert len(finished) > 20 and all(r["forward"] == "yes" for r in finished)
    assert {r["result"] for r in finished} <= {"HIT", "MISS"}
    assert sum(r["result"] == "HIT" for r in finished) > sum(r["result"] == "MISS" for r in finished)  # the planted rule works


def test_the_service_forward_test_labels_every_row_with_its_rule():
    service = study_service(planted_store(mixed_regime))
    rule = service.save_rule(service.study("CL", GENERIC, -10, 10, 10, 36))
    rows = service.forward_test()
    assert rows and all(r["rule"] == rule["id"] and r["rule_label"].startswith("WTI (CL)") for r in rows)
    service._rules.remove(rule["id"])
    assert service.forward_test() == []


def test_a_broken_saved_rule_is_skipped_not_fatal():
    service = study_service(planted_store(mixed_regime))
    service._rules._repository.values["rollover_rules"] = [{"id": "x", "product": "CL"}]
    assert service.forward_test() == []
