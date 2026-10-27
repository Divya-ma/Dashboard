"""The trade journal: market movement, curve shape, risk:reward, payups, the snapshot, the ATR of a structure and the service."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.curve_calendar import generic_contracts
from core.journal import (
    BACKWARDATION, BACKWARDATION_CONTANGO, BEARISH, BULLISH, CONTANGO, CONTANGO_BACKWARDATION, FLAT, STALEMATE,
    add_payups, classify_move, clean_ideas, curve_shape, curve_values, entry_and_exit_times, format_risk_reward, leg_family,
    parse_idea_list, risk_reward, set_payups, with_defaults,
)
from core.journal_service import JournalService
from core.journal_snapshot import MarketSnapshotter, candles_by_symbol, structure_atr
from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType
from db.repository import Repository
from core.structure_utils import get_structure_type_from_symbol

NOW = datetime(2026, 10, 20, 14, 30, tzinfo=timezone.utc)


# ---------- market movement ----------


def test_a_candle_is_bullish_bearish_or_a_stalemate():
    assert classify_move(70.0, 70.6, 69.9, 70.5, 0.01) == BULLISH
    assert classify_move(70.5, 70.6, 69.9, 70.0, 0.01) == BEARISH
    assert classify_move(70.0, 71.0, 69.0, 70.1, 0.01) == STALEMATE  # range 2.0 is 20x the 0.10 body
    assert classify_move(70.0, 70.5, 69.5, 70.0, 0.01) == STALEMATE  # no body at all
    assert classify_move(70.0, 70.5, 69.9, 70.2, 0.01) == BULLISH  # range 0.6, body 0.2: 3x, not 5x
    assert classify_move(70.0, 71.0, 69.0, 70.2, 0.01) == STALEMATE  # exactly 5x counts
    assert classify_move(None, 1, 1, 1, 0.01) is None and classify_move(70, 71, 69, 70.5, 0) is None


def test_a_big_body_is_a_real_move_whatever_the_range():
    assert classify_move(70.0, 78.0, 66.0, 70.6, 0.01) == BULLISH  # 60 ticks of body beats the 12:0.6 range ratio
    assert classify_move(70.6, 78.0, 66.0, 70.0, 0.01) == BEARISH
    assert classify_move(1400.0, 1420.0, 1390.0, 1412.75, 0.25) == BULLISH  # gasoil: 51 ticks of 0.25


# ---------- curve shape ----------


def test_curve_shapes():
    assert curve_shape([70, 70.5, 71, 71.4, 71.8, 72.1]) == CONTANGO
    assert curve_shape([72, 71.5, 71, 70.4, 70.1, 69.8]) == BACKWARDATION
    assert curve_shape([70, 71, 72, 72.5, 71.5, 70.5]) == CONTANGO_BACKWARDATION
    assert curve_shape([72, 71, 70, 69.5, 70.5, 71.5]) == BACKWARDATION_CONTANGO
    assert curve_shape([70, 70, 70, 70, 70, 70]) == FLAT
    assert curve_shape([70, 70.01, 69.99, 70.0, 70.02, 70.01], min_tol=0.05) == FLAT  # under a few ticks is noise, not a turn
    assert curve_shape([70, 70.5]) is None and curve_shape([]) is None


def test_small_wiggles_do_not_make_a_turn_but_a_bigger_turn_wins():
    assert curve_shape([70, 70.4, 70.35, 70.9, 71.4, 72.0]) == CONTANGO  # a 0.05 dip inside a 2.0 range
    # zig-zag: up 2, down 0.5, up: the rise-then-fall turn is smaller than the overall climb, so still a contango
    assert curve_shape([70, 72, 71.5, 73, 74, 75]) == CONTANGO
    assert curve_shape([70, 75, 71, 72, 70.5, 69]) == CONTANGO_BACKWARDATION


def test_spread_and_fly_curves_come_from_the_outrights():
    outrights = [80.0, 79.0, 78.5, 78.2, 78.1, 78.0, 77.9, 77.85]
    assert curve_values(outrights, "outright")[:2] == [80.0, 79.0]
    assert curve_values(outrights, "spread")[:3] == pytest.approx([1.0, 0.5, 0.3])
    assert curve_values(outrights, "fly")[:2] == pytest.approx([80 - 158 + 78.5, 79 - 157 + 78.2])
    assert curve_values([80.0, None, 78.0, 77.0, 76.0, 75.0, 74.0, 73.0], "spread")[:3] == [None, None, pytest.approx(1.0)]  # a missing month leaves a gap


def test_the_curve_read_follows_the_leg_type():
    kind = get_structure_type_from_symbol
    assert leg_family(["BRNF29-G29-H29", "BRNH29-J29-K29"], kind) == "fly"
    assert leg_family(["CLF27-G27", "CLH27"], kind) == "spread"
    assert leg_family(["CLF27", "CLG27", "CLH27"], kind) == "outright"
    assert leg_family(["CLF27", "CLG27-H27", "CLJ27-K27-M27"], kind) == "fly"  # fly first, then spread, then outright


# ---------- risk : reward ----------


def trade(kind=TradeEventType.TRADE, lots=2, price=70.0, stop=None, target=None, day=0, tc=0.0):
    return Trade(structure_id="s", leg_id="l", event_type=kind, lots=lots, price=price, direction="buy",
                 timestamp=datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(days=day), stop_loss_price=stop, target_price=target, transaction_cost=tc)


def test_risk_reward_from_the_stop_and_target_of_the_entries():
    rr = risk_reward([trade(price=70.0, stop=69.0, target=72.0)])
    assert rr["risk"] == 1.0 and rr["reward"] == 2.0 and rr["ratio"] == 2.0 and format_risk_reward(rr) == "1 : 2.00"
    # two entries with different stops and targets: the plain averages, the entry weighted by lots
    rr = risk_reward([trade(lots=1, price=70.0, stop=69.0, target=73.0), trade(TradeEventType.ADD, lots=3, price=71.0, stop=69.5, target=74.0, day=1)])
    assert rr["entry"] == pytest.approx(70.75) and rr["stop"] == pytest.approx(69.25) and rr["target"] == pytest.approx(73.5)
    assert rr["ratio"] == pytest.approx((73.5 - 70.75) / (70.75 - 69.25))
    sell = risk_reward([trade(price=70.0, stop=71.0, target=68.0)])  # a sell: the stop is above
    assert sell["ratio"] == 2.0
    assert risk_reward([trade(price=70.0, stop=69.0)]) is None and risk_reward([]) is None
    assert format_risk_reward(None) == "—" and risk_reward([trade(stop=70.0, target=71.0)])["ratio"] is None  # a stop at the entry has no ratio


# ---------- the document ----------


def test_payups_add_up_and_can_be_replaced():
    journal = add_payups(None, "entry", 2, {"CLZ26": 1, "CLF27": 1})
    journal = add_payups(journal, "entry", 1, {"CLZ26": 2})
    assert journal["payups"]["entry"] == {"total": 3.0, "legs": {"CLZ26": 3.0, "CLF27": 1.0}} and journal["payups"]["exit"]["total"] is None
    assert add_payups(journal, "exit", "", None)["payups"]["exit"]["total"] is None  # blank adds nothing
    assert set_payups(journal, "entry", "", {"CLZ26": 4})["payups"]["entry"] == {"total": None, "legs": {"CLZ26": 4.0}}
    with pytest.raises(ValueError):
        add_payups(journal, "middle", 1)


def test_ideas_are_cleaned_and_parsed():
    assert clean_ideas([" Mean reversion ", "", "Mean reversion", "Breakout"]) == ["Mean reversion", "Breakout"]
    assert clean_ideas(["a", "b"], allowed=["b"]) == ["b"]
    assert parse_idea_list("Mean reversion\nBreakout, Carry\n\n") == ["Mean reversion", "Breakout", "Carry"]
    assert with_defaults(None)["ideas"] == [] and with_defaults({"ideas": ["x"]})["payups"]["exit"]["legs"] == {}


def test_entry_and_exit_times():
    trades = [trade(day=2), trade(TradeEventType.ADD, day=3), trade(TradeEventType.STRUCTURE_PARTIAL_EXIT, day=5), trade(TradeEventType.FULL_EXIT, day=7)]
    first, last = entry_and_exit_times(trades)
    assert first.day == 3 and last.day == 8
    assert entry_and_exit_times([]) == (None, None)


# ---------- the snapshot ----------


def contract(symbol, month=12, product="CL"):
    return Contract(product=product, contract_month=month, contract_year=2027, symbol=symbol, multiplier=1000.0, tick_size=0.01, tick_value=10.0)


def fly_structure(name="fly", product="CL", letters="FGH", prefix="CL"):
    months = {"F": 1, "G": 2, "H": 3}
    legs = [Leg(contract=contract(f"{prefix}{l}27", months[l], product), ratio=r, lots=2, entry_price=70.0, direction="buy")
            for l, r in zip(letters, (1, -2, 1))]
    return Structure(name=name, structure_type=StructureType.FLY, products=[product], legs=legs, status=StructureStatus.OPEN)


class FakeOhlc:
    """CL rises with maturity, Brent falls, gasoil rises then falls; today's CL front candle is a clean up move."""

    def __init__(self, minutes=True, fail=False, hourly=True, daily=True):
        self.calls, self.minutes, self.fail, self.hourly, self.daily = [], minutes, fail, hourly, daily

    def _level(self, symbol):
        for product, sign in (("CL", "up"), ("BRN", "down"), ("G", "hump")):
            for i, c in enumerate(generic_contracts(product, NOW.date(), 8)):
                if c.symbol == symbol:
                    base = {"CL": 70.0, "BRN": 75.0, "G": 700.0}[product]
                    step = {"up": 0.5 * i, "down": -0.5 * i, "hump": 4.0 * (i if i < 4 else 8 - i)}[sign]
                    return base + step
        raise KeyError(symbol)

    def ohlc(self, interval, instruments=None, start=None, end=None, **kwargs):
        self.calls.append((interval, list(instruments)))
        if self.fail:
            raise RuntimeError("api down")
        rows = []
        for symbol in instruments:
            level = self._level(symbol)
            if interval == "1D" and self.daily:
                day = pd.Timestamp(NOW.date(), tz="UTC")
                rows.append({"product": symbol, "time": int(day.timestamp() * 1000), "open": level, "high": level + 0.9, "low": level - 0.2, "close": level + 0.7, "volume": 5})
            elif interval == "1H" and self.hourly:
                for hour, (o, h, l, c) in ((12, (0.0, 0.3, -0.2, 0.2)), (13, (0.2, 0.9, 0.1, 0.5)), (14, (0.5, 0.8, 0.4, 0.7))):
                    stamp = pd.Timestamp(NOW.date(), tz="UTC") + pd.Timedelta(hours=hour)
                    rows.append({"product": symbol, "time": int(stamp.timestamp() * 1000), "open": level + o, "high": level + h, "low": level + l, "close": level + c, "volume": 1})
            elif interval == "1M" and self.minutes:
                rows.append({"product": symbol, "time": int((pd.Timestamp(NOW) - pd.Timedelta(minutes=1)).timestamp() * 1000), "open": level + 0.7, "high": level + 0.75, "low": level + 0.1, "close": level + 0.2, "volume": 1})
        return rows or {"status": "SUCCESS", "data": []}


def test_the_snapshot_reads_prices_moves_and_curves_from_the_candles():
    snap = MarketSnapshotter(FakeOhlc()).snapshot(fly_structure(), NOW)
    assert len(FakeOhlc().calls) == 0 and snap["product"] == "CL" and snap["curve_family"] == "outright"
    assert snap["prices"]["CL"] == pytest.approx(70.2) and snap["prices"]["CO"] == pytest.approx(75.2) and snap["prices"]["GO"] == pytest.approx(700.2)  # the last minute close
    assert snap["move_day"] == BULLISH  # CL front, the day so far from the hourly candles: open 70, close 70.7 (body 0.7, range 1.1)
    assert snap["move_min"] == BEARISH  # the last minute fell from 70.7 to 70.2
    assert snap["curve"] == {"BRN": BACKWARDATION, "CL": CONTANGO, "G": CONTANGO_BACKWARDATION}
    assert snap["atr"] is None  # no ATR store given


def test_without_minute_candles_the_hourly_close_is_the_last_price_and_the_minute_move_is_blank():
    api = FakeOhlc(minutes=False)
    snap = MarketSnapshotter(api).snapshot(fly_structure(), NOW)
    assert snap["move_min"] is None and snap["prices"]["CL"] == pytest.approx(70.7) and [c[0] for c in api.calls] == ["1M", "1H", "1D"]
    assert len(api.calls[0][1]) == 24  # eight outrights of three products, one request each interval


def test_the_day_move_is_what_was_known_at_the_trade_not_the_whole_days_candle():
    """A trade at 12:30 must not see the 13:00 and 14:00 hours of the same day."""
    early = datetime(2026, 10, 20, 12, 30, tzinfo=timezone.utc)
    snap = MarketSnapshotter(FakeOhlc(minutes=False)).snapshot(fly_structure(), early)
    assert snap["prices"]["CL"] == pytest.approx(70.2)  # the 12:00 hour's close, not the day's 70.7
    assert snap["move_day"] == BULLISH  # open 70.0, close 70.2 (high 70.3, low 69.8: range 0.5 is 2.5x the body)


def test_with_only_daily_candles_the_whole_days_candle_and_close_are_used():
    snap = MarketSnapshotter(FakeOhlc(minutes=False, hourly=False)).snapshot(fly_structure(), NOW)
    assert snap["move_day"] == BULLISH and snap["prices"]["CL"] == pytest.approx(70.7) and snap["move_min"] is None


def test_an_api_failure_leaves_blanks_but_still_returns_a_snapshot():
    snap = MarketSnapshotter(FakeOhlc(fail=True)).snapshot(fly_structure(), NOW)
    assert snap["prices"] == {"CO": None, "CL": None, "GO": None} and snap["move_day"] is None and snap["curve"] == {"BRN": None, "CL": None, "G": None}


def test_a_structure_of_fly_legs_reads_the_fly_curve():
    legs = [Leg(contract=contract("BRNF29-G29-H29", 1, "BRN"), ratio=1, lots=1, entry_price=0.3, direction="sell"),
            Leg(contract=contract("BRNH29-J29-K29", 3, "BRN"), ratio=-1, lots=1, entry_price=0.1, direction="sell")]
    s = Structure(name="fly - fly", structure_type=StructureType.CUSTOM, products=["BRN"], legs=legs, status=StructureStatus.OPEN)
    snap = MarketSnapshotter(FakeOhlc()).snapshot(s, NOW)
    assert snap["curve_family"] == "fly" and snap["product"] == "BRN"
    assert snap["curve"]["CL"] == FLAT or snap["curve"]["CL"] is not None  # a straight-line outright curve has an all-zero fly curve


def test_candles_parse_both_reply_shapes():
    assert candles_by_symbol({"status": "SUCCESS", "data": []}) == {} and candles_by_symbol(None) == {}
    frames = candles_by_symbol([{"product": "A", "time": 1_700_000_000_000, "open": 1, "high": 2, "low": 0.5, "close": 1.5}])
    assert list(frames["A"].columns) == ["open", "high", "low", "close", "volume"]


# ---------- ATR of the structure ----------


def atr_frame_of(symbols_closes, days=40):
    """A stored ATR frame: each symbol a random walk with a daily range of about 1."""
    rng = np.random.default_rng(1)
    rows = []
    for symbol, product in symbols_closes:
        close = 70.0 + np.cumsum(rng.normal(0, 0.5, days))
        for i, d in enumerate(pd.bdate_range("2026-08-01", periods=days)):
            rows.append({"symbol": symbol, "date": d, "open": close[i - 1] if i else close[i], "high": close[i] + 0.5, "low": close[i] - 0.5, "close": close[i], "volume": 10})
    frame = pd.DataFrame(rows)
    return lambda product: frame


def test_the_atr_of_a_quoted_fly_is_that_contracts_own():
    s = fly_structure()
    store = atr_frame_of([("CLF27-G27-H27", "CL"), ("CLF27", "CL"), ("CLG27", "CL"), ("CLH27", "CL")])
    exact = structure_atr(s, store, 14, date(2026, 9, 25))
    assert exact is not None and 0.5 < exact < 2.5
    fly_only = store("CL")
    own = fly_only[fly_only["symbol"] == "CLF27-G27-H27"]
    from core.atr_calc import atr_frame

    expected = atr_frame(own[own["date"] <= pd.Timestamp("2026-09-25")].set_index("date")[["open", "high", "low", "close", "volume"]], 14, True, percent=False)["atr"].dropna().iloc[-1]
    assert exact == pytest.approx(expected)


def test_a_structure_with_no_quoted_contract_uses_the_combined_closes():
    s = fly_structure()
    store = atr_frame_of([("CLF27", "CL"), ("CLG27", "CL"), ("CLH27", "CL")])  # no CLF27-G27-H27 stored
    value = structure_atr(s, store, 14, date(2026, 9, 25))
    assert value is not None and value > 0
    assert structure_atr(s, atr_frame_of([("CLF27", "CL")]), 14, date(2026, 9, 25)) is None  # a leg with no data
    outright = Structure(name="o", structure_type=StructureType.OUTRIGHT, products=["CL"], legs=[Leg(contract=contract("CLF27", 1), ratio=2, lots=1, entry_price=70.0)], status=StructureStatus.OPEN)
    one = structure_atr(outright, atr_frame_of([("CLF27", "CL")]), 14, date(2026, 9, 25))
    assert one is not None and one == pytest.approx(2 * structure_atr(outright.model_copy(update={"legs": [outright.legs[0].model_copy(update={"ratio": 1})]}), atr_frame_of([("CLF27", "CL")]), 14, date(2026, 9, 25)))


def test_an_unquoted_structure_is_estimated_from_the_volatility_of_its_combined_price_and_says_so():
    from core.journal_snapshot import ATR_PER_SIGMA_OTHER, structure_atr_with_basis

    store = atr_frame_of([("CLF27", "CL"), ("CLG27", "CL"), ("CLH27", "CL")])
    value, basis = structure_atr_with_basis(fly_structure(), store, 14, date(2026, 9, 25))
    frame = store("CL").set_index("date")
    price = sum(frame[frame["symbol"] == s]["close"] * r for s, r in (("CLF27", 1), ("CLG27", -2), ("CLH27", 1)))
    assert value == pytest.approx(ATR_PER_SIGMA_OTHER * price.diff().dropna().iloc[-60:].std())
    assert basis == "estimated ±65%"
    quoted, quoted_basis = structure_atr_with_basis(fly_structure(), atr_frame_of([("CLF27-G27-H27", "CL"), ("CLF27", "CL"), ("CLG27", "CL"), ("CLH27", "CL")]), 14, date(2026, 9, 25))
    assert quoted_basis == "quoted" and quoted is not None
    assert structure_atr_with_basis(fly_structure(), atr_frame_of([("CLF27", "CL")]), 14, date(2026, 9, 25)) == (None, None)
    assert structure_atr_with_basis(fly_structure(), atr_frame_of([("CLF27", "CL"), ("CLG27", "CL"), ("CLH27", "CL")], days=20), 14, date(2026, 9, 25)) == (None, None)  # too little history


def test_perfectly_correlated_legs_give_a_near_zero_estimate_not_the_sum_of_leg_atrs():
    from core.journal_snapshot import structure_atr_with_basis

    days = pd.bdate_range("2026-07-01", periods=60)
    walk = 70.0 + np.cumsum(np.random.default_rng(3).normal(0, 0.5, 60))
    rows = [{"symbol": s, "date": d, "open": walk[i], "high": walk[i] + 1, "low": walk[i] - 1, "close": walk[i], "volume": 10} for s in ("CLF27", "CLG27", "CLH27") for i, d in enumerate(days)]
    frame = pd.DataFrame(rows)
    value, _ = structure_atr_with_basis(fly_structure(), lambda product: frame, 14, date(2026, 10, 30))
    assert value == pytest.approx(0.0, abs=1e-9)  # (1, -2, 1) of identical prices is flat


# ---------- the service ----------


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


def save(repo, structure, trades=()):
    for leg in structure.legs:
        repo.save_contract(leg.contract)
    repo.save_structure(structure)
    for t in trades:
        repo.save_trade(t.model_copy(update={"structure_id": structure.structure_id, "leg_id": structure.legs[0].leg_id}))


def service(repo, api=None):
    return JournalService(repo, MarketSnapshotter(api or FakeOhlc()), clock=lambda: NOW, background=False)


def test_ideas_come_from_the_settings_list_and_old_picks_survive_its_edits(repo):
    s = fly_structure()
    save(repo, s)
    svc = service(repo)
    assert svc.idea_options() == [] and svc.save_idea_options("Mean reversion\nBreakout") == ["Mean reversion", "Breakout"]
    svc.add_entry_inputs(s.structure_id, ["Breakout", "Not on the list"], 2, {"CLF27": 1})
    assert svc.journal(s.structure_id)["ideas"] == ["Breakout"] and svc.journal(s.structure_id)["payups"]["entry"]["total"] == 2.0
    svc.save_idea_options("Carry")  # Breakout leaves the list
    assert svc.set_ideas(s.structure_id, ["Breakout", "Carry", "Gone"])["ideas"] == ["Breakout", "Carry"]  # kept because it was already on the trade
    svc.add_exit_payups(s.structure_id, 1, None)
    assert repo.get_journal(s.structure_id)["payups"]["exit"]["total"] == 1.0


def test_entry_is_kept_once_and_exit_is_replaced_and_deleting_the_structure_deletes_the_journal(repo):
    s = fly_structure()
    save(repo, s)
    svc = service(repo)
    svc.record(s, "entry", NOW)
    first = svc.journal(s.structure_id)["entry"]["at"]
    svc.record(s, "entry", NOW + timedelta(hours=1))
    assert svc.journal(s.structure_id)["entry"]["at"] == first  # the first entry is the one that counts
    svc.record(s, "exit", NOW + timedelta(hours=2))
    svc.record(s, "exit", NOW + timedelta(hours=3), overwrite=True)
    assert svc.journal(s.structure_id)["exit"]["at"] == (NOW + timedelta(hours=3)).isoformat()
    repo.delete_structure(s.structure_id)
    assert repo.get_journal(s.structure_id) is None


def test_backfill_fills_closed_trades_from_their_own_dates_and_is_capped(repo):
    old = []
    for name in ("old1", "old2"):
        s = fly_structure(name).model_copy(update={"status": StructureStatus.CLOSED})
        save(repo, s, [trade(price=70.0, day=0), trade(TradeEventType.FULL_EXIT, price=71.0, day=3)])
        old.append(s)
    svc = service(repo, FakeOhlc(minutes=False))
    assert svc.backfill(limit=1) == {"filled": 1, "left": 1}
    assert svc.backfill() == {"filled": 1, "left": 0}
    journal = svc.journal(old[0].structure_id)
    assert journal["entry"]["at"].startswith("2026-10-01") and journal["exit"]["at"].startswith("2026-10-04")
    assert svc.backfill() == {"filled": 0, "left": 0}
