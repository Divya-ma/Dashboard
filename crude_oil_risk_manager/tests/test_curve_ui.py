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
        figure, strength, status, kinks, change, quality = cc.render_curve(1, "BRN", family, None)
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
    *_, change, _quality = cc.render_curve(1, "BRN", OUTRIGHT, None)
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
    assert top["seasonal"] == "no baseline" and top["direction"] == "rich"


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
    return [params[n] for n in names] + [params["alert_min_priority"], params["alerts_enabled"]]


def test_saving_thresholds_stores_them_and_the_service_uses_them(env, repo, clock):
    message = cc.save_thresholds(1, *form_values(z_fit=5.5, alert_min_priority="HIGH", alerts_enabled=False))
    assert "Saved" in message.children
    loaded = load_params(repo)
    assert loaded.z_fit == 5.5 and loaded.alert_min_priority == "HIGH" and loaded.alerts_enabled is False
    assert env.curve_service.params().z_fit == 5.5


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
