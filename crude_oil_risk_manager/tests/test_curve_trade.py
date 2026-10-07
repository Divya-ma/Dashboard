"""Trade plan maths (entry, lots, stop, target, hedges) and its use in the service and alerts."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.curve_calendar import DFLY, FLY, OUTRIGHT, SPREAD, build_structures
from core.curve_history import CurveHistoryStore
from core.curve_service import CurveService
from core.curve_settings import CurveParams, load_params, save_params, validate_params
from core.curve_store import PrevSettlementStore, SnapshotStore
from core.curve_trade import (
    HedgeCandidate, build_plan, change_frame, daily_move, fair_value, half_life, plan_summary, rank_hedges,
)
from db.repository import Repository
from tests.curve_fakes import FakeAlerts, FakeApi, MutableClock, make_prices

TODAY = date(2026, 10, 6)
PRODUCTS = ("BRN", "CL")


def synthetic_frame(rows=200, seed=3, sigma=0.1):
    """Daily changes: outright 0 is the trade; spread 0 mirrors it (hedge), fly 0 is unrelated noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, sigma, rows)
    index = pd.date_range("2026-01-01", periods=rows, freq="B")
    columns = pd.MultiIndex.from_tuples([("outright", 0), ("spread", 0), ("fly", 0), ("spread", 1)])
    data = np.column_stack([
        base, base + rng.normal(0, sigma * 0.1, rows), rng.normal(0, sigma, rows), -0.5 * base + rng.normal(0, sigma * 0.5, rows),
    ])
    return pd.DataFrame(data, index=index, columns=columns)


@pytest.fixture
def ladder():
    return build_structures("BRN", TODAY)


def candidates_for(ladder, picks):
    return [HedgeCandidate(family, ladder[family][i], live) for family, i, live in picks]


# ---------- stop, lots, target ----------


def test_volatility_stop_lots_and_target(ladder):
    frame = synthetic_frame(sigma=0.1)
    sigma = daily_move(frame, "outright", 0, 60)
    params = CurveParams(risk_per_trade=5000, risk_per_lot=1000, point_value=1000, vol_stop_mult=1.5)
    plan = build_plan(
        product="BRN", structure=ladder[OUTRIGHT][0], side="SELL", live=100.0,
        fair={"neighbour": 99.0, "fit": 99.2}, flags={"neighbour": True, "fit": True}, frame=frame,
        half_life_days=3.2, reversion_share=0.8, reversion_events=40, candidates=[], params=params,
    )
    assert plan.stop_basis == "volatility" and plan.stop_distance == pytest.approx(1.5 * sigma)
    assert plan.stop == pytest.approx(100.0 + 1.5 * sigma)  # a short is stopped ABOVE the entry
    assert plan.lots == int(5000 // (1.5 * sigma * 1000))
    assert plan.dollar_risk == pytest.approx(plan.lots * 1.5 * sigma * 1000)
    assert plan.fair_value == pytest.approx(99.1)  # median of the two flagged methods
    assert plan.target == pytest.approx(100.0 + (99.1 - 100.0) * 0.8)
    assert plan.rr == pytest.approx(0.72 / (1.5 * sigma)) and plan.time_stop_days == 7
    assert plan.dollar_reward == pytest.approx(plan.lots * 0.72 * 1000)


def test_a_long_has_its_stop_below_and_target_above(ladder):
    plan = build_plan(
        product="BRN", structure=ladder[OUTRIGHT][0], side="BUY", live=100.0, fair={"neighbour": 101.0},
        flags={"neighbour": True}, frame=synthetic_frame(), half_life_days=None, reversion_share=1.0,
        reversion_events=50, candidates=[], params=CurveParams(),
    )
    assert plan.stop < 100.0 < plan.target == pytest.approx(101.0) and plan.time_stop_days is None


def test_per_lot_risk_caps_the_stop_and_the_plan_says_so(ladder):
    params = CurveParams(risk_per_trade=5000, risk_per_lot=100, point_value=1000)  # cap = 0.10 pts
    plan = build_plan(
        product="BRN", structure=ladder[OUTRIGHT][0], side="SELL", live=100.0, fair={"neighbour": 99.0},
        flags={"neighbour": True}, frame=synthetic_frame(sigma=0.2), half_life_days=None, reversion_share=0.8,
        reversion_events=40, candidates=[], params=params,
    )
    assert plan.stop_basis == "per-lot risk cap" and plan.stop_distance == pytest.approx(0.10)
    assert plan.lots == 50 and plan.dollar_risk == pytest.approx(5000)  # no lot risks more than $100
    assert any("tighter" in w for w in plan.warnings)


def test_lots_are_zero_when_one_lot_risks_more_than_the_trade_budget(ladder):
    params = CurveParams(risk_per_trade=50, risk_per_lot=1000, point_value=1000)
    plan = build_plan(
        product="BRN", structure=ladder[OUTRIGHT][0], side="SELL", live=100.0, fair={"neighbour": 99.0},
        flags={"neighbour": True}, frame=synthetic_frame(), half_life_days=None, reversion_share=0.8,
        reversion_events=40, candidates=[], params=params,
    )
    assert plan.lots == 0 and plan.dollar_risk == 0 and any("no lots" in w for w in plan.warnings)


def test_max_lots_caps_a_tiny_stop(ladder):
    params = CurveParams(risk_per_trade=1_000_000, risk_per_lot=1000, point_value=1000, max_lots=40)
    plan = build_plan(
        product="BRN", structure=ladder[OUTRIGHT][0], side="SELL", live=100.0, fair={"neighbour": 99.0},
        flags={"neighbour": True}, frame=synthetic_frame(), half_life_days=None, reversion_share=0.8,
        reversion_events=40, candidates=[], params=params,
    )
    assert plan.lots == 40 and any("capped at 40" in w for w in plan.warnings)


def test_without_history_the_stop_falls_back_to_the_risk_cap(ladder):
    plan = build_plan(
        product="BRN", structure=ladder[OUTRIGHT][0], side="SELL", live=100.0, fair={"neighbour": 99.0},
        flags={"neighbour": True}, frame=pd.DataFrame(), half_life_days=None, reversion_share=None,
        reversion_events=0, candidates=[], params=CurveParams(),
    )
    assert plan.stop_basis.startswith("per-lot risk cap") and plan.stop_distance == pytest.approx(1.0)
    assert plan.reversion_share == 0.5 and any("Only 0 similar" in w for w in plan.warnings)
    assert plan.hedges == [] and any("No history" in w for w in plan.warnings)


def test_reversion_share_is_never_below_the_minimum_or_above_one(ladder):
    for given, expected in ((0.2, 0.5), (0.7, 0.7), (1.7, 1.0)):
        plan = build_plan(
            product="BRN", structure=ladder[OUTRIGHT][0], side="SELL", live=100.0, fair={"neighbour": 99.0},
            flags={"neighbour": True}, frame=synthetic_frame(), half_life_days=None, reversion_share=given,
            reversion_events=40, candidates=[], params=CurveParams(),
        )
        assert plan.reversion_share == pytest.approx(expected)


def test_no_target_when_fair_value_is_on_the_wrong_side_of_the_entry(ladder):
    plan = build_plan(
        product="BRN", structure=ladder[OUTRIGHT][0], side="SELL", live=100.0, fair={"neighbour": 101.0},
        flags={"neighbour": True}, frame=synthetic_frame(), half_life_days=None, reversion_share=0.8,
        reversion_events=40, candidates=[], params=CurveParams(),
    )
    assert plan.target is None and plan.rr is None and any("wrong side" in w for w in plan.warnings)


def test_fair_value_uses_the_methods_that_flagged_it():
    fair = {"fit": 99.0, "neighbour": 98.0, "pca": 97.0, "history": None}
    assert fair_value(fair, {"fit": True, "pca": True}) == pytest.approx(98.0)
    assert fair_value(fair, {}) == pytest.approx(98.5)  # falls back to neighbour and fit
    assert fair_value({"fit": None, "neighbour": None}, {}) is None


def test_half_life_of_an_ar1_residual():
    rng = np.random.default_rng(0)
    phi, rows = 0.8, 600
    series = np.zeros((rows, 6))
    for t in range(1, rows):
        series[t] = phi * series[t - 1] + rng.normal(0, 0.05, 6)
    assert half_life(series, 500) == pytest.approx(np.log(2) / -np.log(phi), rel=0.15)
    assert half_life(rng.normal(0, 1, (300, 6)), 250) < 1.0  # no persistence: a residual is gone within a day
    assert half_life(series[:10], 500) is None


# ---------- hedges ----------


def test_the_best_hedge_is_the_correlated_one_and_its_ratio_matches_the_beta(ladder):
    frame = synthetic_frame()
    params = CurveParams(hedge_min_corr=0.5, hedge_count=2)
    picks = [(SPREAD, 0, 1.2), (FLY, 0, 0.3), (SPREAD, 1, 0.9)]
    hedges, warnings = rank_hedges(frame, OUTRIGHT, 0, "BUY", "BRN", candidates_for(ladder, picks), params)
    assert warnings == []
    best = hedges[0]
    assert (best.family, best.label) == (SPREAD, ladder[SPREAD][0].label)
    assert best.side == "SELL" and best.ratio == pytest.approx(1.0, abs=0.1)  # long A is hedged by selling its twin
    assert best.var_reduction > 0.8 and best.corr > 0.9
    assert best.ratio_minvar == pytest.approx(-1.0, abs=0.1) and "sell" in best.legs or "buy" in best.legs
    assert all(abs(h.corr) >= 0.5 for h in hedges) and FLY not in [h.family for h in hedges]  # the noise fly is out


def test_a_short_trade_flips_the_hedge_side(ladder):
    hedges, _ = rank_hedges(synthetic_frame(), OUTRIGHT, 0, "SELL", "BRN", candidates_for(ladder, [(SPREAD, 0, 1.0)]), CurveParams())
    assert hedges[0].side == "BUY"


def test_a_negatively_correlated_hedge_is_bought_when_the_trade_is_bought(ladder):
    hedges, _ = rank_hedges(synthetic_frame(), OUTRIGHT, 0, "BUY", "BRN", candidates_for(ladder, [(SPREAD, 1, 1.0)]), CurveParams(hedge_min_corr=0.3))
    assert hedges[0].corr < 0 and hedges[0].side == "BUY"  # b moves against a, so a long is hedged by buying b


def test_at_least_one_hedge_is_always_listed_with_a_warning_when_none_qualifies(ladder):
    params = CurveParams(hedge_min_corr=0.99, hedge_count=2)
    hedges, warnings = rank_hedges(
        synthetic_frame(), OUTRIGHT, 0, "BUY", "BRN", candidates_for(ladder, [(FLY, 0, 0.3), (SPREAD, 1, 0.9)]), params
    )
    assert len(hedges) == 1 and hedges[0].warning and warnings
    assert hedges[0].var_reduction >= 0  # still the best available


def test_hedge_count_limits_the_alternatives(ladder):
    picks = [(SPREAD, 0, 1.0), (SPREAD, 1, 1.0)]
    one, _ = rank_hedges(synthetic_frame(), OUTRIGHT, 0, "BUY", "BRN", candidates_for(ladder, picks), CurveParams(hedge_count=1, hedge_min_corr=0.3))
    two, _ = rank_hedges(synthetic_frame(), OUTRIGHT, 0, "BUY", "BRN", candidates_for(ladder, picks), CurveParams(hedge_count=2, hedge_min_corr=0.3))
    assert len(one) == 1 and len(two) == 2 and two[0].var_reduction >= two[1].var_reduction


def test_hedge_lots_follow_the_ratio_and_are_at_least_one(ladder):
    plan = build_plan(
        product="BRN", structure=ladder[OUTRIGHT][0], side="BUY", live=100.0, fair={"neighbour": 101.0},
        flags={"neighbour": True}, frame=synthetic_frame(), half_life_days=None, reversion_share=1.0,
        reversion_events=50, candidates=candidates_for(ladder, [(SPREAD, 0, 1.2), (SPREAD, 1, 0.9)]),
        params=CurveParams(hedge_min_corr=0.3),
    )
    assert plan.lots > 0
    for hedge in plan.hedges:
        assert hedge.lots == max(1, round(hedge.ratio * plan.lots)) and hedge.entry in (1.2, 0.9)


def test_hedge_ranking_is_cached_but_entry_prices_stay_live(ladder):
    cache = {}
    frame = synthetic_frame()
    for live in (1.2, 1.5):
        plan = build_plan(
            product="BRN", structure=ladder[OUTRIGHT][0], side="BUY", live=100.0, fair={"neighbour": 101.0},
            flags={"neighbour": True}, frame=frame, half_life_days=None, reversion_share=1.0, reversion_events=50,
            candidates=candidates_for(ladder, [(SPREAD, 0, live)]), params=CurveParams(), hedge_cache=cache,
        )
        assert plan.hedges[0].entry == live
    assert len(cache) == 1


def test_roll_days_are_blanked_out_of_the_change_frame(tmp_path):
    api = FakeApi()
    store = CurveHistoryStore(tmp_path, api)
    store.refresh("BRN", 1, TODAY)
    matrix = store.matrix("BRN", OUTRIGHT)
    frame = change_frame(store, "BRN", 15)
    assert frame[(OUTRIGHT, 0)].isna().sum() == 1  # only the first row (no previous day): the fake never rolls
    assert (OUTRIGHT, 14) in frame.columns and (DFLY, 11) in frame.columns and len(frame) == matrix.rows


def test_plan_summary_reads_like_an_order(ladder):
    plan = build_plan(
        product="BRN", structure=ladder[OUTRIGHT][0], side="SELL", live=98.12, fair={"neighbour": 97.72},
        flags={"neighbour": True}, frame=synthetic_frame(), half_life_days=4.0, reversion_share=0.8,
        reversion_events=40, candidates=candidates_for(ladder, [(SPREAD, 0, 1.2)]), params=CurveParams(hedge_min_corr=0.3),
    )
    text = plan_summary("Dec26", "outright", "sell Dec26", plan)
    assert text.startswith("SELL outright Dec26 (sell Dec26) @ 98.12,")
    assert "Stop" in text and "Target" in text and "R:R" in text and "HEDGE:" in text and "time stop ~8d" in text


# ---------- in the service and the alerts ----------


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
def service(repo, tmp_path, clock, alerts, monkeypatch):
    # The reversion backtest takes seconds per curve; its own tests cover it (test_the_reversion_estimate_*).
    monkeypatch.setattr(CurveService, "_reversion", lambda self, product, family, history, params: (0.8, 40))
    api = FakeApi()
    repo.set_setting("api_access_token", "tok")  # lets the service load history and settlements
    return CurveService(
        repo, api, CurveHistoryStore(tmp_path / "c", api), PrevSettlementStore(tmp_path / "c" / "p.json"),
        SnapshotStore(tmp_path / "c"), alerts, products=PRODUCTS, clock=clock, run_in_background=False,
    )


def feed(service, clock, **kwargs):
    service.on_prices(make_prices(PRODUCTS, clock().date(), clock(), **kwargs))
    return service.state()


def outright_kink(state, product="BRN"):
    return next(k for k in state.kinks if k["product"] == product and k["family"] == OUTRIGHT)


def test_a_kink_gets_a_full_plan_with_hedges_that_share_no_contract(service, clock):
    feed(service, clock)  # loads history
    state = feed(service, clock, kinks={"COG27": 0.4})
    row = outright_kink(state)
    plan = row["plan"]
    assert plan is not None and plan.side == "SELL" and plan.entry == pytest.approx(row["value"])
    assert plan.stop > plan.entry > plan.target and plan.lots >= 1 and plan.rr and plan.rr > 0
    assert 1 <= len(plan.hedges) <= 2
    kinked = set(row["structure"].legs)
    ladder = service.ladder("BRN", clock().date())
    for hedge in plan.hedges:
        structure = next(s for s in ladder[hedge.family] if s.label == hedge.label)
        assert not (set(structure.legs) & kinked)  # no hedge carries the kinked contract
        assert hedge.lots >= 1 and hedge.side in ("BUY", "SELL") and hedge.entry is not None


def test_the_alert_carries_entry_lots_stop_target_and_hedge(service, clock, alerts):
    feed(service, clock)
    alerts.sent.clear()
    state = feed(service, clock, kinks={"COG27": 0.4})
    title, body = service.alert_text(outright_kink(state))  # the outright's own alert text
    assert title.endswith("- SELL")
    assert body.startswith("TRADE: SELL outright Feb27 (sell Feb27) @ ")
    for needle in ("lots", "Stop", "Target", "R:R", "HEDGE:", "ratio", "corr", "VaR", "WHY:"):
        assert needle in body, needle
    sent_title, sent_body = next((t, b) for _, t, b in alerts.sent if "BRN" in t)  # and the batched one that is sent
    assert sent_body.startswith("TRADE: ") and "HEDGE:" in sent_body and len(sent_body) <= 1800
    assert "related" in sent_title


def test_hedge_types_are_respected_and_can_exclude_everything_but_flies(service, clock, repo):
    save_params(repo, CurveParams(hedge_types={"outright": ["fly"], "spread": ["spread", "fly", "dfly"],
                                               "fly": ["spread", "fly", "dfly"], "dfly": ["fly", "dfly"]}))
    feed(service, clock)
    plan = outright_kink(feed(service, clock, kinks={"COG27": 0.4}))["plan"]
    assert plan.hedges and all(h.family == FLY for h in plan.hedges)


def test_a_dfly_kink_can_only_be_hedged_with_flies_or_dflies_by_default(service, clock):
    feed(service, clock)
    state = feed(service, clock, kinks={"COG27": 0.4})
    dflies = [k for k in state.kinks if k["product"] == "BRN" and k["family"] == DFLY and k["plan"] and k["plan"].hedges]
    assert dflies
    for row in dflies:
        assert all(h.family in (FLY, DFLY) for h in row["plan"].hedges)


def test_the_minimum_reward_risk_filters_alerts_but_not_the_page(service, clock, alerts, repo):
    save_params(repo, CurveParams(min_reward_risk=20.0))  # no plan here reaches 20:1
    feed(service, clock)
    alerts.sent.clear()
    state = feed(service, clock, kinks={"COG27": 0.4})
    assert state.kinks and outright_kink(state)["plan"] is not None  # still on the page, with its plan
    assert alerts.sent == []


def test_without_history_the_alert_still_gives_direction_and_price_and_explains_the_rest(repo, tmp_path, clock, alerts):
    api = FakeApi()
    plain = CurveService(
        repo, api, CurveHistoryStore(tmp_path / "n", api), PrevSettlementStore(tmp_path / "n" / "p.json"),
        SnapshotStore(tmp_path / "n"), alerts, products=PRODUCTS, clock=clock, run_in_background=False,
    )
    state = feed(plain, clock, kinks={"COG27": 0.4})
    assert outright_kink(state)["plan"] is None
    body = next(b for _, t, b in alerts.sent if "BRN" in t)
    assert body.startswith("TRADE: SELL outright Feb27 (sell Feb27) @ ") and "need price history" in body


def test_the_event_log_keeps_the_plan_summary(service, clock, repo):
    feed(service, clock)
    feed(service, clock, kinks={"COG27": 0.4})
    event = next(e for e in repo.get_kink_events() if e["label"] == "Feb27" and e["family"] == OUTRIGHT)
    assert event["detail"]["plan"]["side"] == "SELL" and event["detail"]["plan"]["lots"] >= 1


def test_the_reversion_estimate_is_cached_and_computed_off_the_poll(repo, tmp_path, clock, alerts, monkeypatch):
    import threading
    from core import curve_service as cs
    from core.curve_backtest import BacktestResult

    calls = []

    def fake_backtest(product, family, history, params, test_days, horizons):
        calls.append((product, family, horizons))
        return BacktestResult(product, family, reversion={horizons[0]: 0.7}, reversion_events={horizons[0]: 12})

    monkeypatch.setattr(cs, "run_backtest", fake_backtest)
    api = FakeApi()
    real = CurveService(
        repo, api, CurveHistoryStore(tmp_path / "r", api), PrevSettlementStore(tmp_path / "r" / "p.json"),
        SnapshotStore(tmp_path / "r"), alerts, products=PRODUCTS, clock=clock, run_in_background=True,
    )
    real.history.refresh("BRN", 1, TODAY)
    history, params = real.history.matrix("BRN", OUTRIGHT), CurveParams(reversion_horizon=10)
    assert real._reversion("BRN", OUTRIGHT, history, params) == (None, 0)  # not ready: the poll is not held up
    for thread in [t for t in threading.enumerate() if t.name == "curve-reversion"]:
        thread.join(timeout=5)
    assert real._reversion("BRN", OUTRIGHT, history, params) == (0.7, 12)
    assert real._reversion("BRN", OUTRIGHT, history, params) == (0.7, 12) and len(calls) == 1  # cached


def test_only_a_couple_of_reversion_backtests_run_at_once(repo, tmp_path, clock, alerts, monkeypatch):
    import threading
    from core import curve_service as cs
    from core.curve_backtest import BacktestResult

    release = threading.Event()
    started = []

    def slow_backtest(product, family, history, params, test_days, horizons):
        started.append(family)
        release.wait(timeout=5)
        return BacktestResult(product, family)

    monkeypatch.setattr(cs, "run_backtest", slow_backtest)
    api = FakeApi()
    real = CurveService(
        repo, api, CurveHistoryStore(tmp_path / "q", api), PrevSettlementStore(tmp_path / "q" / "p.json"),
        SnapshotStore(tmp_path / "q"), alerts, products=PRODUCTS, clock=clock, run_in_background=True,
    )
    real.history.refresh("BRN", 1, TODAY)
    params = CurveParams()
    for family in (OUTRIGHT, SPREAD, FLY, DFLY):
        real._reversion("BRN", family, real.history.matrix("BRN", family), params)
    release.set()
    for thread in [t for t in threading.enumerate() if t.name == "curve-reversion"]:
        thread.join(timeout=5)
    assert len(started) == 2


def test_trade_settings_validate_and_default_sensibly():
    defaults = CurveParams()
    assert defaults.hedge_types["dfly"] == ["fly", "dfly"] and defaults.hedge_types["outright"][0] == "outright"
    assert validate_params({**defaults.__dict__, "risk_per_trade": 2500}).risk_per_trade == 2500
    for bad in ({"risk_per_trade": 0}, {"risk_per_lot": -5}, {"hedge_count": 0}, {"hedge_min_corr": 1.5},
                {"hedge_types": {"dfly": []}}, {"hedge_types": {"dfly": ["swap"]}}):
        with pytest.raises(ValueError):
            validate_params({**defaults.__dict__, **bad})
    partial = validate_params({**defaults.__dict__, "hedge_types": {"dfly": ["fly"]}})
    assert partial.hedge_types["dfly"] == ["fly"] and partial.hedge_types["spread"] == defaults.hedge_types["spread"]


def test_old_saved_settings_without_trade_fields_still_load(repo):
    repo.set_setting("curve_params", {"z_fit": 4.0})
    loaded = load_params(repo)
    assert loaded.z_fit == 4.0 and loaded.risk_per_trade == 5000.0 and loaded.hedge_count == 2
