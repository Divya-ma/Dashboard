"""Curve Kinks page: backtest, figure/table builders, callbacks, settings section, live-adapter extras."""

import numpy as np
import pandas as pd
import pytest
from dash.exceptions import PreventUpdate

from core.curve_backtest import run_backtest
from core.curve_calendar import DFLY, FAMILIES, OUTRIGHT, SPREAD
from core.curve_history import CurveHistoryStore, HistoryMatrix
from core.curve_service import CurveService
from core.curve_settings import CurveParams, load_params
from core.curve_store import PrevSettlementStore, SnapshotStore
from db.repository import Repository
from tests.curve_fakes import FakeAlerts, FakeApi, MutableClock, make_prices
from tests.test_curve_kinks import smooth_history
from ui.callbacks import curve_callbacks as cc
from ui.callbacks import settings_callbacks as sc
from ui.container import Container
from ui.layouts import curve_tab
from ui.layouts import settings as settings_layout_module

PRODUCTS = ("BRN", "CL")


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture
def clock():
    return MutableClock()


@pytest.fixture
def env(repo, tmp_path, clock, monkeypatch):
    monkeypatch.setattr(CurveService, "_reversion", lambda self, product, family, history, params: (0.8, 40))
    api = FakeApi()
    service = CurveService(
        repo, api, CurveHistoryStore(tmp_path / "curves", api),
        PrevSettlementStore(tmp_path / "curves" / "prev.json"), SnapshotStore(tmp_path / "curves"),
        FakeAlerts(), products=PRODUCTS, clock=clock, run_in_background=False,
    )
    fresh = Container(repository=repo, curve_service=service)
    for module in (cc, curve_tab, sc, settings_layout_module):
        monkeypatch.setattr(module, "container", fresh)
    return fresh


def feed(env, clock, **kwargs):
    env.curve_service.on_prices(make_prices(PRODUCTS, clock().date(), clock(), **kwargs))


# ---------- backtest ----------


def history_with_reverting_spikes(rows=700):
    values, dates, months = smooth_history(rows=rows)
    values = values.copy()
    for t in (450, 500, 550, 600):  # one-day spikes that revert at once
        values[t, 6] += 0.4
    return HistoryMatrix(values=values, dates=dates, months=months, last_date=pd.Timestamp(dates[-1]).date())


def test_backtest_finds_the_spikes_and_sees_them_revert():
    result = run_backtest("BRN", OUTRIGHT, history_with_reverting_spikes(), CurveParams(), test_days=300)
    assert result.error is None and result.days_tested > 200
    consensus = [r for r in result.rows if r["group"] == "3+ methods" and r["horizon"] == 5]
    assert consensus and consensus[0]["events"] >= 3
    assert consensus[0]["reverted_pct"] >= 90 and consensus[0]["avg_reduction"] > 0.2


def test_backtest_reports_when_there_is_too_little_history():
    values, dates, months = smooth_history(rows=60)
    short = HistoryMatrix(values=values, dates=dates, months=months)
    assert "Not enough history" in run_backtest("BRN", OUTRIGHT, short, CurveParams()).error
    assert "Not enough history" in run_backtest("BRN", OUTRIGHT, None, CurveParams()).error


def test_backtest_on_a_clean_history_says_nothing_was_flagged():
    values, dates, months = smooth_history(rows=500)
    clean = HistoryMatrix(values=values, dates=dates, months=months)
    strict = CurveParams(z_fit=12, z_neighbour=12, z_pca=12, z_history=12)
    assert "No kinks were flagged" in run_backtest("BRN", OUTRIGHT, clean, strict, test_days=150).error


# ---------- render ----------


def test_render_curve_returns_figures_and_tables_for_every_family(env, clock):
    feed(env, clock, kinks={"COG27": 0.4})
    for family in FAMILIES:
        figure, strength, status, kinks, change, quality, plans = cc.render_curve(1, "BRN", family, None)
        assert len(figure.data) >= 1 and figure.data[0].name == "Live"
        assert len(strength.data) == 1
    figure, *_ = cc.render_curve(1, "BRN", OUTRIGHT, None)
    names = [t.name for t in figure.data]
    assert "Kink" in names and "Implied by neighbours" in names
    kink_trace = next(t for t in figure.data if t.name == "Kink")
    assert list(kink_trace.x) == ["Feb27"]


def test_dfly_curve_has_twelve_points(env, clock):
    feed(env, clock)
    figure, *_ = cc.render_curve(1, "CL", DFLY, None)
    assert len(figure.data[0].x) == 12 and figure.data[0].x[0] == "Nov26→Feb27"


def test_previous_settlement_line_and_snapshot_overlay(env, clock):
    env.curve_service.settlements.fetch(
        env.curve_service._api, env.curve_service.outright_symbols(), clock().date(), clock().date()
    )
    feed(env, clock)
    assert cc.snapshot_options("BRN") == [{"label": "2026-10-06", "value": "2026-10-06"}]
    figure, *_ = cc.render_curve(1, "BRN", SPREAD, "2026-10-06")
    names = [t.name for t in figure.data]
    assert "Previous settlement" in names and "Snapshot 2026-10-06" in names


def test_change_table_shows_live_vs_settlement(env, clock):
    env.curve_service.settlements.fetch(
        env.curve_service._api, env.curve_service.outright_symbols(), clock().date(), clock().date()
    )
    feed(env, clock)
    *_, change, _quality, _plans = cc.render_curve(1, "BRN", OUTRIGHT, None)
    rows = change.data if hasattr(change, "data") else change.children[0].data
    assert len(rows) == 15 and rows[0]["structure"] == "Dec26" and rows[0]["prev"] != "n/a"


def test_before_the_first_poll_the_page_shows_a_waiting_message(env):
    figure, strength, status, *_ = cc.render_curve(1, "BRN", OUTRIGHT, None)
    assert "Waiting" in figure.layout.annotations[0].text


def test_without_a_service_the_page_says_so(env):
    env.curve_service = None
    figure, *_ = cc.render_curve(1, "BRN", OUTRIGHT, None)
    assert "not running" in figure.layout.annotations[0].text


def test_kinks_table_lists_priority_methods_and_seasonality(env, clock):
    feed(env, clock, kinks={"COG27": 0.4})
    state = env.curve_service.state()
    rows = curve_tab.kinks_rows(state.kinks)
    top = next(r for r in rows if r["product"] == "BRN" and r["family"] == "Outright")
    assert top["structure"] == "Feb27" and top["priority"] == "HIGH" and "Fit" in top["methods"]
    assert top["seasonal"] == "no baseline" and top["direction"] == "rich" and top["trade"] == "SELL"


def test_quality_panel_shows_buffer_flags_first(env, clock):
    from datetime import datetime, timezone
    clock.now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    feed(env, clock)
    panel = curve_tab.build_quality(env.curve_service.state().quality)
    assert "BUFFER" in str(panel.children[0])


def test_log_callback_reads_the_event_log(env, clock):
    feed(env, clock, kinks={"COG27": 0.4})
    table = cc.render_log(1)
    assert any(r["structure"] == "Feb27" and r["status"] == "active" for r in table.data)


def test_backtest_callback_with_and_without_history(env, clock):
    with pytest.raises(PreventUpdate):
        cc.run_backtest_callback(None, "BRN", OUTRIGHT, 250)
    assert "Not enough history" in str(cc.run_backtest_callback(1, "BRN", OUTRIGHT, 250).children)
    env.curve_service.history.refresh("BRN", 3, clock().date())
    result = cc.run_backtest_callback(1, "BRN", OUTRIGHT, 100)
    assert "days tested" in str(result)


# ---------- thresholds form ----------


def form_values(**overrides):
    params = {**CurveParams().__dict__, **overrides}
    names = [name for name, *_ in curve_tab.PARAM_FIELDS]
    by_product = [
        [family for family in FAMILIES if f"{code}:{family}" in params["alert_structures"]] for code in ("CL", "BRN")
    ]
    return [params[n] for n in names] + [params["alert_min_priority"], params["alerts_enabled"], *by_product]


def test_saving_thresholds_stores_them_and_the_service_uses_them(env, repo, clock):
    message = cc.save_thresholds(1, *form_values(z_fit=5.5, alert_min_priority="HIGH", alerts_enabled=False))
    assert "Saved" in message.children
    loaded = load_params(repo)
    assert loaded.z_fit == 5.5 and loaded.alert_min_priority == "HIGH" and loaded.alerts_enabled is False
    assert env.curve_service.params().z_fit == 5.5


def test_choosing_which_structures_may_alert_is_saved_per_product(env, repo):
    keep = ["CL:spread", "CL:fly", "BRN:outright"]
    message = cc.save_thresholds(1, *form_values(alert_structures=keep))
    assert "Saved" in message.children
    assert load_params(repo).alert_structures == ["CL:spread", "CL:fly", "BRN:outright"]  # canonical order
    layout = str(curve_tab.trade_analyzer_layout())
    assert "curve-alert-CL" in layout and "curve-alert-BRN" in layout


def test_invalid_thresholds_are_rejected_with_a_message(env, repo):
    message = cc.save_thresholds(1, *form_values(z_fit=0.01))
    assert "Not saved" in message.children and load_params(repo) == CurveParams()
    with pytest.raises(PreventUpdate):
        cc.save_thresholds(None, *form_values())


def test_the_form_starts_with_the_saved_values(env, repo):
    repo.set_setting("curve_params", {"z_pca": 4.2})
    layout = str(curve_tab.trade_analyzer_layout())
    assert "curve-param-z_pca" in layout and "value=4.2" in layout


# ---------- settings tab: opening time ----------


def test_curve_settings_load_and_save(env, repo):
    with pytest.raises(PreventUpdate):
        sc.load_curve_settings("/home")
    assert sc.load_curve_settings("/settings") == ("07:00", "UTC")
    assert "Saved" in sc.save_curve_settings(1, "6:45", "America/New_York")
    assert sc.load_curve_settings("/settings") == ("06:45", "America/New_York")
    assert "Not saved" in sc.save_curve_settings(1, "99:99", "UTC")
    assert sc.load_curve_settings("/settings") == ("06:45", "America/New_York")


def test_settings_layout_has_the_curve_section(env):
    assert "settings-curve-open-time" in str(settings_layout_module.settings_layout())


# ---------- trade plan on the dashboard ----------


def planned_state(env, repo, clock):
    repo.set_setting("api_access_token", "tok")  # lets the service load history, so plans can be built
    feed(env, clock)
    feed(env, clock, kinks={"COG27": 0.4})
    return env.curve_service.state()


def test_kinks_table_shows_the_plan_columns(env, repo, clock):
    state = planned_state(env, repo, clock)
    row = next(r for r in curve_tab.kinks_rows(state.kinks) if r["product"] == "BRN" and r["family"] == "Outright")
    assert row["trade"] == "SELL" and row["entry"] != "" and int(row["lots"]) >= 1
    assert float(row["stop"]) > float(row["entry"]) > float(row["target"])  # a short: stop above, target below
    assert row["hedge"].split()[0] in ("BUY", "SELL") and float(row["rr"]) > 0
    table = curve_tab.build_kinks_table(state.kinks)
    assert {"entry", "lots", "stop", "target", "rr", "hedge"} <= {c["id"] for c in table.columns}


def test_plan_cards_show_entry_lots_stop_target_and_hedge_options(env, repo, clock):
    state = planned_state(env, repo, clock)
    text = str(curve_tab.build_plan_cards(state.kinks))
    for needle in ("SELL BRN Outright Feb27", "sell Feb27", "Stop", "Target", "Reward : risk", "Time stop",
                   "Hedge alternatives", "Ratio (VaR)", "Ratio (min-var)", "VaR change"):
        assert needle in text, needle


def test_plan_cards_explain_when_history_is_missing(env, clock):
    feed(env, clock, kinks={"COG27": 0.4})  # no token, no history
    text = str(curve_tab.build_plan_cards(env.curve_service.state().kinks))
    assert "need price history" in text and "Stop" not in text


def test_plan_cards_are_limited_to_the_strongest_ten(env, repo, clock):
    state = planned_state(env, repo, clock)
    many = (state.kinks * 5)[: curve_tab.PLAN_CARD_LIMIT + 3]
    assert "strongest" in str(curve_tab.build_plan_cards(many))
    assert "no trade plans" in str(curve_tab.build_plan_cards([]))


def test_render_curve_returns_the_plan_tab_content(env, repo, clock):
    planned_state(env, repo, clock)
    *_, plans = cc.render_curve(1, "BRN", OUTRIGHT, None)
    assert "Hedge alternatives" in str(plans)


# ---------- settings tab: risk appetite and hedge rules ----------


TRADE_FORM = [5000, 1000, 1000, 100, 1.5, 60, 10, 0.5, 0.0, 2, 0.5, 120]  # in CURVE_TRADE_FIELDS order


def test_trade_settings_load_defaults_and_save(env, repo):
    with pytest.raises(PreventUpdate):
        sc.load_curve_trade_settings("/home")
    loaded = sc.load_curve_trade_settings("/settings")
    assert list(loaded[:12]) == TRADE_FORM and loaded[12] is True
    assert loaded[16] == ["fly", "dfly"] and loaded[13][0] == "outright"  # dfly kinks: fly or dfly hedges

    form = [2500, 400, 1000, 50, 2.0, 30, 5, 0.6, 1.5, 3, 0.6, 90]
    hedge_types = [["fly"], ["spread", "fly", "dfly"], ["fly", "dfly"], ["dfly"]]
    message = sc.save_curve_trade_settings(1, *form, False, *hedge_types)
    assert "Saved" in message
    saved = load_params(repo)
    assert (saved.risk_per_trade, saved.risk_per_lot, saved.max_lots, saved.min_reward_risk, saved.hedge_count) == (2500, 400, 50, 1.5, 3)
    assert saved.hedge_exclude_overlap is False and saved.hedge_types["outright"] == ["fly"] and saved.hedge_types["dfly"] == ["dfly"]


def test_saving_trade_settings_keeps_the_thresholds_set_elsewhere(env, repo):
    from core.curve_settings import save_params
    save_params(repo, CurveParams(z_fit=4.4, alert_min_priority="HIGH", alert_structures=["CL:spread"]))
    sc.save_curve_trade_settings(1, *TRADE_FORM, True, *[["fly"]] * 4)
    kept = load_params(repo)
    assert kept.z_fit == 4.4 and kept.alert_min_priority == "HIGH" and kept.alert_structures == ["CL:spread"]


def test_bad_trade_settings_are_rejected_and_nothing_changes(env, repo):
    for bad_form, hedge_types in (
        ([0] + TRADE_FORM[1:], [["fly"]] * 4),  # risk per trade of zero
        (TRADE_FORM[:9] + [0] + TRADE_FORM[10:], [["fly"]] * 4),  # no hedges shown
        (TRADE_FORM, [[], ["fly"], ["fly"], ["fly"]]),  # an outright kink with no allowed hedge type
        ([None] + TRADE_FORM[1:], [["fly"]] * 4),
    ):
        assert "Not saved" in sc.save_curve_trade_settings(1, *bad_form, True, *hedge_types)
    assert load_params(repo) == CurveParams()
    with pytest.raises(PreventUpdate):
        sc.save_curve_trade_settings(None, *TRADE_FORM, True, *[["fly"]] * 4)


def test_settings_layout_has_the_trade_plan_fields(env):
    layout = str(settings_layout_module.settings_layout())
    for needle in ("settings-curve-risk_per_trade", "settings-curve-risk_per_lot", "settings-curve-hedge_count",
                   "settings-curve-hedge-dfly", "settings-save-curve-trade", "settings-curve-hedge_exclude_overlap"):
        assert needle in layout, needle
