"""Editable journal lists, the required stop and target, and the extra market fields (volatility, volume, bid-ask, crack, curve z-score)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.curve_service import CurveService, CurveState
from core.journal import EXIT_REASONS, MISTAKES
from core.journal_analysis import group_table, trade_table
from core.journal_service import JournalService
from core.journal_snapshot import MarketSnapshotter, activity_ratios
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from db.repository import Repository
from tests.test_journal import NOW, FakeOhlc, fly_structure
from tests.test_journal_analysis import row
from ui.callbacks import settings_callbacks as sec
from ui.callbacks import structure_detail_callbacks as dc
from ui.callbacks import structures_callbacks as sc
from ui.container import Container
from core.user_settings import KEY_REQUIRE_STOP_TARGET

GRID = ("active", "all", "name", {}, {})
LIVE_STORE = {"CLX26": {"price": 75.0}, "CLZ26": {"price": 74.0}}


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


class Quiet:
    def snapshot(self, structure, when):
        return {"at": when.isoformat(), "prices": {}, "curve": {}}


@pytest.fixture(autouse=True)
def env(repo, monkeypatch):
    journal = JournalService(repo, Quiet(), clock=lambda: NOW, background=False)
    fresh = Container(repository=repo, journal=journal)
    import ui.layouts.settings as settings_layout_module

    for module in (dc, sc, sec, settings_layout_module):
        monkeypatch.setattr(module, "container", fresh)
    return fresh


def spread():
    legs = [Leg(contract=Contract(product="CL", contract_month=m, contract_year=2026, symbol=s, multiplier=1000.0, tick_size=0.01, tick_value=10.0), ratio=r)
            for s, m, r in (("CLX26", 11, 1), ("CLZ26", 12, -1))]
    return Structure(name="spr", structure_type=StructureType.SPREAD, products=["CL"], legs=legs, status=StructureStatus.SHELL)


def save(repo, s):
    for leg in s.legs:
        repo.save_contract(leg.contract)
    repo.save_structure(s)
    return s.structure_id


# ---------- editable lists ----------


def test_exit_reasons_and_mistakes_start_as_the_built_in_lists_and_can_be_edited(env):
    assert env.journal.exit_reason_options() == EXIT_REASONS and env.journal.mistake_options() == MISTAKES
    assert env.journal.save_exit_reason_options("Took profit\nStopped out") == ["Took profit", "Stopped out"]
    assert env.journal.save_mistake_options("Late\nEarly, Oversized") == ["Late", "Early", "Oversized"]
    assert env.journal.exit_reason_options() == ["Took profit", "Stopped out"] and env.journal.mistake_options() == ["Late", "Early", "Oversized"]
    assert env.journal.save_mistake_options("") == [] and env.journal.mistake_options() == []  # emptied on purpose: nothing offered


def test_the_settings_save_covers_all_three_lists_and_the_switch(env, repo):
    message = sec.save_trade_ideas(1, "Carry", "Took profit", "Late\nEarly", True)
    assert "1 trade idea" in message and "1 exit reason" in message and "2 mistake tag" in message and "required" in message
    assert repo.get_setting(KEY_REQUIRE_STOP_TARGET, False) is True and env.journal.exit_reason_options() == ["Took profit"]
    assert "optional" in sec.save_trade_ideas(1, "Carry", None, None, False) and env.journal.exit_reason_options() == ["Took profit"]  # a list left out is not touched
    from ui.layouts.settings import settings_layout

    text = str(settings_layout())
    for component_id in ("settings-exit-reasons", "settings-mistakes", "settings-require-stop-target", "Took profit"):
        assert component_id in text, component_id


def test_a_reason_or_tag_taken_off_the_list_stays_valid_on_the_trade_that_has_it(env, repo):
    sid = save(repo, spread())
    env.journal.save_exit_reason_options("Took profit\nStopped out")
    env.journal.save_mistake_options("Late")
    env.journal.set_review(sid, 3, "Stopped out", ["Late"])
    env.journal.save_exit_reason_options("Took profit")
    env.journal.save_mistake_options("Early")
    again = env.journal.set_review(sid, 3, "Stopped out", ["Late", "Early"])
    assert again["exit_reason"] == "Stopped out" and again["mistakes"] == ["Late", "Early"]
    with pytest.raises(ValueError, match="exit reason must be one of"):
        env.journal.set_review(sid, 3, "Made up", [])
    with pytest.raises(ValueError):
        env.journal.add_exit_payups(sid, None, None, "Made up")


# ---------- required stop and target ----------


def test_a_first_entry_needs_a_stop_and_a_target_when_the_setting_is_on(env, repo):
    repo.set_setting(KEY_REQUIRE_STOP_TARGET, True)
    sid = save(repo, spread())
    refused = dc.confirm_trade_entry(1, 1.25, 5, "buy", None, None, 2.5, [], sid, LIVE_STORE, *GRID)
    assert "danger" in refused and repo.get_structure(sid).status == StructureStatus.SHELL
    refused = dc.confirm_trade_entry(1, 1.25, 5, "buy", None, 0.75, "", [], sid, LIVE_STORE, *GRID)
    assert "danger" in refused and repo.get_trades_for_structure(sid) == []
    accepted = dc.confirm_trade_entry(1, 1.25, 5, "buy", None, 0.75, 2.5, [], sid, LIVE_STORE, *GRID)
    assert "danger" not in accepted and repo.get_structure(sid).status == StructureStatus.OPEN
    added = dc.confirm_trade_entry(1, 1.30, 2, "buy", None, None, None, [], sid, LIVE_STORE, *GRID)  # an add does not need them again
    assert "danger" not in added and repo.get_structure(sid).legs[0].lots == 7


def test_the_requirement_is_off_unless_it_is_switched_on(env, repo):
    sid = save(repo, spread())
    assert repo.get_setting(KEY_REQUIRE_STOP_TARGET, False) is False
    assert "danger" not in dc.confirm_trade_entry(1, 1.25, 5, "buy", None, None, None, [], sid, LIVE_STORE, *GRID)


# ---------- the extra market fields ----------


def hourly(n_recent=24, n_base=100, vol_recent=0.2, vol_base=0.1, volume_recent=200, volume_base=100):
    rng = np.random.default_rng(5)
    base = rng.normal(0, vol_base, n_base)
    recent = rng.normal(0, vol_recent, n_recent)
    close = 70 + np.cumsum(np.concatenate([base, recent]))
    index = pd.date_range(end=pd.Timestamp(NOW), periods=n_base + n_recent, freq="1h")
    volume = np.concatenate([np.full(n_base, volume_base), np.full(n_recent, volume_recent)])
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=index)


def test_volatility_and_volume_are_measured_against_the_days_before():
    vol, volume = activity_ratios(hourly(), pd.Timestamp(NOW))
    assert 1.4 < vol < 2.8 and volume == pytest.approx(2.0)  # twice the hourly noise, twice the volume
    calm_vol, calm_volume = activity_ratios(hourly(vol_recent=0.1, volume_recent=100), pd.Timestamp(NOW))
    assert 0.7 < calm_vol < 1.4 and calm_volume == pytest.approx(1.0)
    assert activity_ratios(hourly(n_base=20), pd.Timestamp(NOW)) == (None, None)  # too little history
    assert activity_ratios(None, pd.Timestamp(NOW)) == (None, None)


class QuoteApi(FakeOhlc):
    def __init__(self, spare=30, bid=70.00, ask=70.03, rows=True):
        super().__init__()
        self.taq_calls, self.spare, self.bid, self.ask, self.rows = [], spare, bid, ask, rows

    def spare_requests(self, name="ohlc"):
        return self.spare

    def taq(self, products):
        self.taq_calls.append(products)
        return {"data": [{"best_bid": self.bid, "best_ask": self.ask}] if self.rows else []}


def test_the_snapshot_adds_the_crack_the_brent_wti_spread_and_a_live_trades_bid_ask_and_curve_score():
    structure = fly_structure()
    api = QuoteApi()
    seen = []
    snap = MarketSnapshotter(api, curve_z=lambda s: seen.append(s.name) or {"z": -2.4, "methods": {"fit": -2.4}}, clock=lambda: NOW).snapshot(structure, NOW)
    co, cl, go = snap["prices"]["CO"], snap["prices"]["CL"], snap["prices"]["GO"]
    assert snap["brent_wti"] == pytest.approx(co - cl, abs=1e-3) and snap["crack"] == pytest.approx(go / 7.45 - co, abs=1e-3)
    assert snap["spread_ticks"] == pytest.approx(3.0) and snap["kink_z"] == -2.4 and seen == ["fly"]
    assert api.taq_calls[0][0]["id"] == "CLF27-G27-H27" and api.taq_calls[0][0]["start"] < api.taq_calls[0][0]["end"]  # the quoted fly, a two-minute window


def test_a_past_trade_has_no_bid_ask_or_curve_score_and_a_busy_taq_budget_skips_the_bid_ask():
    api = QuoteApi()
    snap = MarketSnapshotter(api, curve_z=lambda s: {"z": 1.0}, clock=lambda: NOW + timedelta(days=3)).snapshot(fly_structure(), NOW)
    assert snap["spread_ticks"] is None and snap["kink_z"] is None and api.taq_calls == []
    busy = QuoteApi(spare=1)
    assert MarketSnapshotter(busy, clock=lambda: NOW).snapshot(fly_structure(), NOW)["spread_ticks"] is None and busy.taq_calls == []
    nothing = QuoteApi(rows=False)
    assert MarketSnapshotter(nothing, clock=lambda: NOW).snapshot(fly_structure(), NOW)["spread_ticks"] is None


def test_the_curve_score_of_a_structure_comes_from_the_kinks_pages_ladder():
    structure = SimpleNamespace(legs=("CLX26", "CLZ26"), leg_weights=(1, -1))
    position = SimpleNamespace(structure=structure, result=SimpleNamespace(z={"fit": 1.2, "neighbour": -3.1, "pca": None, "history": 0.4}))
    service = CurveService.__new__(CurveService)
    service._state = CurveState(families={("CL", "spread"): SimpleNamespace(positions=[position])})
    assert service.kink_z(("CLX26", "CLZ26"), (1, -1))["z"] == -3.1  # the largest |z|, with its sign
    assert service.kink_z(("CLX26", "CLZ26"), (-1, 1))["z"] == -3.1  # the same structure the other way round
    assert service.kink_z(("CLX26", "CLF27"), (1, -1)) is None
    empty = SimpleNamespace(structure=structure, result=None)
    service._state = CurveState(families={("CL", "spread"): SimpleNamespace(positions=[empty])})
    assert service.kink_z(("CLX26", "CLZ26"), (1, -1)) is None


def test_the_new_fields_are_columns_and_groupings():
    from core.archive import SNAPSHOT_FIELDS
    from core.journal_analysis import DIMENSIONS
    from ui.layouts.archive_tab import TABLE_COLUMNS

    names = [c for c, _k, _s in SNAPSHOT_FIELDS]
    assert {"vol_ratio", "volume_ratio", "spread_ticks", "brent_wti", "crack", "kink_z"} <= set(names)
    ids = {c["id"] for c in TABLE_COLUMNS}
    assert {"e_vol_ratio", "x_crack", "e_kink_z", "x_spread_ticks"} <= ids
    rows = [row(f"t{i}", 10.0 * (i - 5), idea=()) for i in range(12)]
    for i, r in enumerate(rows):
        r.update(e_vol_ratio=0.5 + i * 0.15, e_volume_ratio=0.6 + i * 0.12, e_spread_ticks=1 + i % 3, e_brent_wti=10 + i, e_crack=20 + i,
                 kink_edge=-3 + i * 0.5)
    frame = trade_table(rows)
    assert {"quiet (under 0.7x)", "normal", "fast (over 1.3x)"} == set(frame["vol"].dropna())
    assert set(frame["spread"].dropna()) == {"tight (1 tick)", "2 ticks", "wide (over 2 ticks)"}
    assert frame["brent_wti"].iloc[0].startswith("Brent-WTI low") and frame["crack"].iloc[-1].startswith("crack high")
    assert set(frame["kink"].dropna()) == {"against the curve (z under -2)", "near fair (z within 2)", "with the curve (z over 2)"}
    for dimension in ("vol", "volume", "spread", "brent_wti", "crack", "kink"):
        assert dimension in DIMENSIONS and not group_table(frame, dimension).empty
