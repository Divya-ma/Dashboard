"""Analysis of the trade journal: measures, grouping, equity curve, excursion, and the new review fields."""

import json
import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.journal import clean_conviction, clean_exit_reason, clean_mistakes, excursion
from core.journal_analysis import MIN_SAMPLE, equity_curve, group_table, headline, measures, streaks, trade_table
from core.journal_service import JournalService
from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType
from db.repository import Repository
from ui.callbacks import archive_callbacks as ac
from ui.callbacks import structure_detail_callbacks as dc
from ui.callbacks import structures_callbacks as sc
from ui.container import Container
from ui.layouts import archive_analysis as aa
from ui.layouts import archive_tab as at

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def row(name, pnl, *, type_="Spread", products=("CL",), tc=2.0, r=None, idea=(), curve="Contango", price=70.0, day="Bullish", minute="Stalemate", held=2,
        day_index=0, hour=9, conviction=None, reason=None, mistakes=(), payup_entry=None, payup_exit=None, rr=None, atr=None, risk=None):
    entry_at = (T0 + timedelta(days=day_index)).replace(hour=hour)
    snap = {"product": products[0], "prices": {}, "curve": {}}
    return {
        "structure_id": name, "name": name, "type": type_, "products": list(products), "realized_pnl": pnl, "transaction_costs": tc, "r_multiple": r,
        "rr": rr, "rr_detail": {"risk": risk} if risk else None, "e_atr": atr, "ideas": list(idea), "mistakes": list(mistakes),
        "e_px_cl": price, "e_px_co": price, "e_px_go": price, "e_curve_cl": curve, "e_curve_brn": curve, "e_curve_go": curve,
        "e_move_day": day, "e_move_min": minute, "days_held": held, "entry_at": entry_at.isoformat(),
        "exit_at": (entry_at + timedelta(days=held)).isoformat(), "snapshots": {"entry": snap, "exit": None},
        "conviction": conviction, "exit_reason": reason, "payup_entry": payup_entry, "payup_exit": payup_exit, "risk_usd": None, "mfe": None, "mae": None,
    }


# ---------- measures ----------


def test_the_measures_of_a_set_of_trades():
    rows = [row("a", 300.0, r=1.5), row("b", -100.0, r=-0.5), row("c", 200.0, r=1.0), row("d", -100.0), row("e", 0.0)]
    m = measures(trade_table(rows))
    assert m["trades"] == 5 and m["win_rate"] == pytest.approx(0.4)  # a flat trade is not a win
    assert m["avg_win"] == pytest.approx(250.0) and m["avg_loss"] == pytest.approx(-100.0) and m["payoff"] == pytest.approx(2.5)
    assert m["expectancy"] == pytest.approx(60.0) and m["expectancy_r"] == pytest.approx(2 / 3) and m["r_trades"] == 3
    assert m["profit_factor"] == pytest.approx(500 / 200) and m["total"] == pytest.approx(300.0) and m["costs"] == pytest.approx(10.0)


def test_no_losses_is_an_infinite_profit_factor_and_no_trades_is_empty():
    assert math.isinf(measures(trade_table([row("a", 10.0), row("b", 5.0)]))["profit_factor"])
    h = headline(trade_table([]))
    assert h["trades"] == 0 and h["win_rate"] is None and h["max_drawdown"] == 0.0 and h["total"] == 0.0


def test_equity_curve_drawdown_and_streaks():
    rows = [row("a", 100.0, day_index=0), row("b", -50.0, day_index=1), row("c", -80.0, day_index=2), row("d", 200.0, day_index=3), row("e", -10.0, day_index=4)]
    frame = trade_table(rows)
    curve = equity_curve(frame)
    assert list(curve["cum"]) == [100.0, 50.0, -30.0, 170.0, 160.0]
    assert list(curve["drawdown"]) == [0.0, -50.0, -130.0, 0.0, -10.0]
    h = headline(frame)
    assert h["max_drawdown"] == -130.0 and h["longest_loss"] == 2 and h["longest_win"] == 1
    assert streaks(trade_table([row("x", 0.0), row("y", 5.0)])) == (1, 0)
    first_loss = equity_curve(trade_table([row("l", -40.0)]))
    assert first_loss["drawdown"].iloc[0] == -40.0  # the start counts as a peak


# ---------- grouping ----------


def test_groups_split_the_trades_and_flag_small_samples():
    rows = [row(f"w{i}", 100.0, curve="Contango") for i in range(MIN_SAMPLE)] + [row(f"l{i}", -50.0, curve="Backwardation") for i in range(3)]
    table = group_table(trade_table(rows), "curve")
    assert list(table["Group"]) == ["Contango", "Backwardation"]  # best expectancy first
    assert table["Sample"].tolist() == ["ok", f"low (3 of {MIN_SAMPLE})"]
    assert table.loc[0, "Win rate"] == 1.0 and table.loc[1, "Expectancy $"] == -50.0


def test_ideas_and_mistakes_count_a_trade_under_each_tag_and_untagged_ones_show_as_none():
    rows = [row("a", 100.0, idea=["Carry", "Breakout"]), row("b", -40.0, idea=["Carry"]), row("c", 10.0), row("d", -5.0, mistakes=["Poor fill"])]
    table = group_table(trade_table(rows), "idea").set_index("Group")
    assert table.loc["Carry", "Trades"] == 2 and table.loc["Breakout", "Trades"] == 1 and table.loc["(none)", "Trades"] == 2
    assert group_table(trade_table(rows), "mistake").set_index("Group").loc["Poor fill", "Total $"] == -5.0


def test_price_bands_are_thirds_of_each_products_own_prices():
    cl = [row(f"c{i}", 10.0, products=("CL",), price=60.0 + i) for i in range(9)]
    go = [row(f"g{i}", -10.0, products=("G",), price=1400.0 + i) for i in range(9)]
    bands = trade_table(cl + go)["price_band"]
    assert bands.iloc[0].startswith("CL low") and bands.iloc[4].startswith("CL middle") and bands.iloc[8].startswith("CL high")
    assert bands.iloc[9].startswith("GO low") and bands.iloc[17].startswith("GO high")  # gasoil is judged against gasoil
    few = trade_table([row("a", 1.0, price=70.0), row("b", 1.0, price=71.0)])["price_band"]
    assert "too few" in few.iloc[0]


def test_buckets_for_stop_distance_risk_reward_hold_time_and_payups():
    frame = trade_table([
        row("a", 1.0, risk=0.3, atr=1.0, rr=0.8, held=0, payup_entry=0, payup_exit=2, hour=7, day_index=0),
        row("b", 1.0, risk=1.5, atr=1.0, rr=2.5, held=5, payup_entry=1, payup_exit=0, hour=15, day_index=1),
    ])
    assert list(frame["stop_atr"]) == ["under 0.5 ATR", "1 to 2 ATR"] and list(frame["rr_band"]) == ["under 1 : 1", "1 : 2 to 1 : 3"]
    assert list(frame["held"]) == ["same day", "4 to 7 days"] and list(frame["payup_entry"]) == ["0", "1"] and list(frame["payup_exit"]) == ["2 or more", "0"]
    assert list(frame["hour"]) == ["08:00", "16:00"]  # October: London is an hour ahead of UTC
    assert list(frame["weekday"]) == ["Thu", "Fri"]
    assert frame["payups"].tolist() == [2.0, 1.0]
    assert group_table(frame, "nonsense").empty


# ---------- excursion ----------


def structure_for(lots=4.0):
    leg = Leg(contract=Contract(product="CL", contract_month=12, contract_year=2026, symbol="CLZ26", multiplier=1000.0, tick_size=0.01, tick_value=10.0),
              ratio=1, lots=lots, entry_price=70.0, direction="buy")
    return Structure(name="s", structure_type=StructureType.OUTRIGHT, products=["CL"], legs=[leg], status=StructureStatus.CLOSED)


def trades_for(s, direction="buy"):
    return [Trade(structure_id=s.structure_id, leg_id=s.legs[0].leg_id, event_type=TradeEventType.TRADE, lots=4, price=70.0, direction=direction, timestamp=T0),
            Trade(structure_id=s.structure_id, leg_id=s.legs[0].leg_id, event_type=TradeEventType.FULL_EXIT, lots=4, price=71.0, direction="sell", timestamp=T0 + timedelta(days=3))]


def test_excursion_is_the_best_and_worst_mark_between_entry_and_exit():
    s = structure_for()
    days = pd.date_range("2026-09-29", periods=8, tz="UTC")
    prices = pd.Series([60.0, 69.0, 70.5, 72.0, 69.5, 71.0, 80.0, 50.0], index=days)  # 29 Sep .. 6 Oct; only 1 to 4 Oct counts
    result = excursion(s, trades_for(s), prices)
    assert result["mfe"] == pytest.approx((72.0 - 70.0) * 4 * 1000) and result["mae"] == pytest.approx((69.5 - 70.0) * 4 * 1000) and result["days"] == 4
    sell = excursion(s, trades_for(s, "sell"), prices)
    assert sell["mfe"] == pytest.approx(2000.0) and sell["mae"] == pytest.approx(-8000.0)  # a short: 72 hurts, 69.5 helps
    flat = excursion(s, trades_for(s), pd.Series([70.0] * 4, index=pd.date_range("2026-10-01", periods=4, tz="UTC")))
    assert flat["mfe"] == 0.0 and flat["mae"] == 0.0
    assert excursion(s, trades_for(s), pd.Series(dtype=float)) is None and excursion(s, [], prices) is None
    assert excursion(s, trades_for(s), pd.Series([1.0], index=pd.date_range("2026-01-01", periods=1, tz="UTC"))) is None  # nothing in the window


# ---------- validators ----------


def test_conviction_exit_reason_and_mistakes_are_validated():
    assert clean_conviction("3") == 3 and clean_conviction(None) is None and clean_conviction("") is None
    for bad in (0, 6, "x"):
        with pytest.raises(ValueError):
            clean_conviction(bad)
    assert clean_exit_reason("Stop hit") == "Stop hit" and clean_exit_reason("") is None
    with pytest.raises(ValueError):
        clean_exit_reason("because")
    assert clean_mistakes(["Poor fill", "Not a tag", "Poor fill"]) == ["Poor fill"]


# ---------- in the app ----------


class FakeSnapshotter:
    def snapshot(self, structure, when):
        return {"at": when.isoformat(), "product": "CL", "curve_family": "outright", "prices": {"CO": 75.0, "CL": 70.0, "GO": 700.0},
                "move_day": "Bullish", "move_min": "Stalemate", "curve": {"CL": "Contango", "BRN": "Contango", "G": "Contango"}, "atr": 0.5}


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture(autouse=True)
def env(repo, monkeypatch):
    days = pd.date_range("2026-09-25", periods=60, tz="UTC")  # covers the real "today" the trades are stamped with
    path = pd.Series([70.0 + i * 0.2 for i in range(60)], index=days)
    journal = JournalService(repo, FakeSnapshotter(), clock=lambda: T0, background=False, price_path=lambda structure: path)
    fresh = Container(repository=repo, journal=journal)
    for module in (dc, sc, ac):
        monkeypatch.setattr(module, "container", fresh)
    import ui.layouts.structure_detail as layout_module

    monkeypatch.setattr(layout_module, "container", fresh)
    return fresh


def spread(name="spr"):
    legs = [Leg(contract=Contract(product="CL", contract_month=m, contract_year=2026, symbol=s, multiplier=1000.0, tick_size=0.01, tick_value=10.0), ratio=r)
            for s, m, r in (("CLX26", 11, 1), ("CLZ26", 12, -1))]
    return Structure(name=name, structure_type=StructureType.SPREAD, products=["CL"], legs=legs, status=StructureStatus.SHELL)


GRID = ("active", "all", "name", {}, {})
LIVE_STORE = {"CLX26": {"price": 75.0}, "CLZ26": {"price": 74.0}}


def save(repo, s):
    for leg in s.legs:
        repo.save_contract(leg.contract)
    repo.save_structure(s)
    return s.structure_id


def test_the_forms_carry_conviction_and_exit_reason_and_the_exit_records_the_excursion(repo, env):
    sid = save(repo, spread())
    dc.confirm_trade_entry(1, 1.25, 4, "buy", None, 0.75, 2.25, [], sid, LIVE_STORE, *GRID, [], None, None, 4)
    assert env.journal.journal(sid)["conviction"] == 4
    dc.execute_full_exit(1, 1.5, 4, None, sid, *GRID, None, None, "Target hit")
    journal = env.journal.journal(sid)
    assert journal["exit_reason"] == "Target hit" and journal["excursion"]["lots"] == 4 and journal["excursion"]["basis"].startswith("daily")
    assert journal["excursion"]["mfe"] >= 0 >= journal["excursion"]["mae"]


def test_archive_rows_carry_r_risk_and_excursion_and_the_analysis_tab_renders(repo, env):
    sid = save(repo, spread())
    for entry in (dc.confirm_trade_entry(1, 1.0, 4, "buy", None, 0.5, 2.0, [], sid, LIVE_STORE, *GRID, ["x"], 1, None, 3),):
        pass
    dc.execute_full_exit(1, 1.5, 4, None, sid, *GRID, None, None, "Stop hit")
    (archived,) = ac._load_rows()
    assert archived["risk_usd"] == pytest.approx(0.5 * 4 * 1000) and archived["r_multiple"] == pytest.approx(archived["realized_pnl"] / 2000.0)
    assert archived["conviction"] == 3 and archived["exit_reason"] == "Stop hit" and archived["mfe"] is not None
    cells = at.build_table_rows([archived])[0]
    assert cells["risk_usd"] == 2000 and cells["conviction"] == 3 and cells["exit_reason"] == "Stop hit"
    assert {"conviction", "exit_reason", "mistakes", "risk_usd", "r_multiple", "mfe_i", "mae_i", "mfe_d", "mae_d"} <= {c["id"] for c in at.TABLE_COLUMNS}
    rows = json.loads(json.dumps([archived]))
    cards, equity, bars, table = ac.render_analysis(rows, "type")
    assert len(cards) == 8 and len(equity.data) == 2 and table[0]["group"] == "Spread" and table[0]["trades"] == 1 and table[0]["sample"].startswith("low")
    for dimension in aa.DIMENSIONS:
        assert isinstance(ac.render_analysis(rows, dimension)[3], list)
    empty = ac.render_analysis([], "curve")
    assert empty[3] == [] and len(empty[0]) == 8


def test_the_review_fields_save_from_the_archive_editor(repo, env):
    sid = save(repo, spread())
    dc.confirm_trade_entry(1, 1.0, 4, "buy", None, None, None, [], sid, LIVE_STORE, *GRID)
    dc.execute_full_exit(1, 1.5, 4, None, sid, *GRID)
    assert ac.save_journal_edits(1, sid, [], None, None, 5, "Discretionary", ["Poor fill", "Bogus"]).startswith("Saved")
    options = ac.fill_journal_editor(sid)
    assert options[4] == 5 and options[5] == "Discretionary" and options[6] == ["Poor fill"]
    assert "Not saved" in ac.save_journal_edits(1, sid, [], None, None, 9, None, [])
    layout = str(at.archive_layout())
    for component_id in ("archive-tabs", "analysis-equity", "analysis-dimension", "analysis-table", "archive-conviction", "archive-exit-reason", "archive-mistakes", "trades"):
        assert component_id in layout, component_id


def test_the_entry_and_exit_forms_offer_the_new_fields(repo, env):
    from ui.layouts.structure_detail import structure_detail_layout

    s = spread()
    save(repo, s)
    text = str(structure_detail_layout(s, {}, [], None))
    assert "trade-entry-conviction" in text
    opened = s.model_copy(update={"status": StructureStatus.OPEN, "legs": [leg.model_copy(update={"lots": 3.0, "entry_price": 70.0}) for leg in s.legs]})
    assert "trade-exit-reason" in str(structure_detail_layout(opened, {}, [], None))


def test_the_backfill_also_fills_a_missing_excursion_without_the_api(repo, env):
    sid = save(repo, spread())
    dc.confirm_trade_entry(1, 1.0, 4, "buy", None, None, None, [], sid, LIVE_STORE, *GRID)
    dc.execute_full_exit(1, 1.5, 4, None, sid, *GRID)
    journal = repo.get_journal(sid)
    journal["excursion"] = None
    repo.save_journal(sid, journal)
    env.journal.backfill()
    assert repo.get_journal(sid)["excursion"] is not None
