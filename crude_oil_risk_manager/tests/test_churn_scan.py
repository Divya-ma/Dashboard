"""Churn morning scan: the filters, the levels and the nearest-level distance, then the same through the service and the page."""

import numpy as np
import pandas as pd
import pytest

from core.churn_scan import AT_LEVEL_TICKS, build_scan, levels_of, nearest_level
from tests.test_churn_live import PARAMS, _fill_live, service, shared  # noqa: F401 - the module fixtures
from ui.callbacks import churn_callbacks as cb
from ui.layouts import churn_tab


def row(symbol="S1", score=40.0, osc=5.0, price=70.00, week_low=69.50, week_high=70.50, atr=1.0, direction="long", hours="10:00"):
    return dict(symbol=symbol, product="CL", label=symbol, family="spread", direction=direction, score=score, osc=osc, price=price, week_low=week_low,
                week_high=week_high, atr=atr, hours=hours, verdict="GO")


def test_levels_are_built_from_what_is_known():
    assert levels_of(70.2, 69.5, 70.5) == {"Yesterday's close": 70.2, "Week mid": 70.0, "Week low": 69.5, "Week high": 70.5}
    assert levels_of(None, np.nan, 70.5) == {"Week high": 70.5}
    assert nearest_level({}, 70.0, 0.01) is None and nearest_level({"a": 1.0}, np.nan, 0.01) is None


def test_the_nearest_level_and_its_signed_distance_in_ticks():
    name, level, ticks = nearest_level({"Week low": 69.5, "Week mid": 70.0}, 70.03, 0.01)
    assert name == "Week mid" and level == 70.0 and ticks == pytest.approx(-3.0)  # the level is below the price
    assert nearest_level({"Week high": 70.5}, 70.4, 0.01)[2] == pytest.approx(10.0)  # above


def test_the_scan_keeps_profitable_oscillating_tight_rows_tightest_first():
    table = pd.DataFrame([
        row("LOSER", score=-30.0),                      # this side loses
        row("QUIET", osc=1.0),                          # came back too seldom
        row("WIDE", week_low=68.0, week_high=72.0),     # 4 ATRs of range
        row("OK_WIDER", week_low=69.0, week_high=71.0), # 2 ATRs
        row("OK_TIGHT", week_low=69.8, week_high=70.2), # 0.4 ATRs
        row("NOATR", atr=np.nan),                       # no ATR: kept, sorted last
        row("NOPRICE", price=np.nan, osc=np.nan),       # no live price: kept, no level
    ])
    extras = {s: {"tick": 0.01, "prior_close": 70.0} for s in table["symbol"]}
    scan = build_scan(table, extras, min_osc=3, max_range_atr=3.0)
    assert list(scan["contract"]) == ["CL OK_TIGHT", "CL NOPRICE", "CL OK_WIDER", "CL NOATR"]  # 0.4, 1.0, 2.0 ATRs, then the unknown range
    nop = scan[scan["contract"] == "CL NOPRICE"].iloc[0]
    assert nop["status"] == "No live price" and nop["level"] == ""
    assert len(build_scan(pd.DataFrame(), {})) == 0


def test_a_price_on_a_level_is_at_level_and_otherwise_says_how_far():
    table = pd.DataFrame([row("AT", price=70.01), row("FAR", price=70.30)])
    scan = build_scan(table, {"AT": {"tick": 0.01, "prior_close": 70.0}, "FAR": {"tick": 0.01, "prior_close": None}}).set_index("contract")
    assert scan.loc["CL AT", "status"] == "AT LEVEL" and abs(scan.loc["CL AT", "distance"]) <= AT_LEVEL_TICKS
    assert scan.loc["CL FAR", "level"] == "Week high" and scan.loc["CL FAR", "status"] == "20 ticks above"


def test_the_service_scan_is_none_until_built_and_then_lists_rows_with_levels(service):
    assert service.morning_scan(PARAMS.__class__(contracts=5, start_from=1)) is None  # a different setting: not built
    _fill_live(service)
    scan = service.morning_scan(PARAMS, min_osc=0, max_range_atr=1e9)
    assert not scan.empty and (scan["score"] > 0).all()
    assert (scan["level"] != "").all() and scan["distance"].notna().all()
    assert set(service.morning_scan(PARAMS, 0, 1e9, family="dfly")["family"]) <= {"dfly"}
    assert len(service.morning_scan(PARAMS, min_osc=10_000, max_range_atr=1e9)) <= len(scan)


def test_the_page_callback_fills_the_scan_table(service, monkeypatch):
    from dataclasses import asdict
    from ui.container import container

    monkeypatch.setattr(container, "churn_service", service, raising=False)
    _fill_live(service)
    data, note = cb.render_scan(0, asdict(PARAMS), 1, "all", "all", 0, 1e9)
    assert data and {c["id"] for c in churn_tab.SCAN_COLUMNS} <= set(data[0]) and "pass" in note
    assert cb.render_scan(0, None, 1, "all", "all", 0, 1e9) == (cb.no_update, cb.no_update)
    assert "churn-scan-table" in str(churn_tab.churn_layout())
