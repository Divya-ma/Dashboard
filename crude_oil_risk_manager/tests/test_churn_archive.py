"""Churn page: your real round trips matched to the simulated contract and side, and set beside the simulation."""

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.churn_archive import compare, fit_settings, real_trips, row_key, split_symbol, structure_key
from core.models.trade import TradeEventType as E
from ui.callbacks import churn_callbacks as cb
from ui.container import Container
from ui.layouts import churn_tab

TICKS = {"CL": 10.0, "BRN": 10.0, "G": 25.0, "CLBZ": 10.0}


def _leg(symbol, ratio=1):
    return SimpleNamespace(contract=SimpleNamespace(symbol=symbol), ratio=ratio)


def _trade(event, lots, direction, when, pnl=None, cost=7.12):
    return SimpleNamespace(event_type=event, lots=lots, direction=direction, realized_pnl=pnl, transaction_cost=cost,
                           timestamp=datetime(2026, 11, when[0], when[1], when[2], tzinfo=timezone.utc))


def _structure(sid, name, *legs):
    return SimpleNamespace(structure_id=sid, name=name, legs=list(legs))


def test_symbols_split_into_a_product_code_and_month_codes():
    assert split_symbol("CLK27-M27-N27") == ("CL", "K27-M27-N27") and split_symbol("GM27") == ("G", "M27") and split_symbol("BRNN28-Q28") == ("BRN", "N28-Q28")
    assert split_symbol("CLBZZXF27-G27") == ("CLBZZX", "F27-G27") and split_symbol("hello") is None and split_symbol("") is None
    assert row_key("BRN", "COH27-J27-K27/COJ27-K27-M27") == ("BRN", "H27-J27-K27/J27-K27-M27") and row_key("CLBZ", "CLBZZXF27-G27-H27") == ("CLBZ", "F27-G27-H27")
    assert row_key("CL", "not a symbol") is None


def test_a_structure_is_matched_to_the_one_instrument_it_is():
    assert structure_key([_leg("CLU27-V27")]) == ("CL", "U27-V27")
    assert structure_key([_leg("BRNJ27-K27-M27"), _leg("BRNK27-M27-N27", -1)]) == ("BRN", "J27-K27-M27/K27-M27-N27")  # two flies: the d-fly
    assert structure_key([_leg("CLK27-M27-N27"), _leg("BZK27-M27-N27", -1)]) == ("CLBZ", "K27-M27-N27")  # a CL-BZ box
    assert structure_key([_leg("CLK27-M27-N27"), _leg("BZM27-N27-Q27", -1)]) is None  # different months: not a quoted box
    assert structure_key([_leg("BZK27-M27-N27")]) is None and structure_key([_leg("WBSQ27-U27-V27")]) is None  # no candles for BZ or WBS alone
    assert structure_key([_leg("CLK27-M27-N27"), _leg("CLM27-N27-Q27", -2)]) is None and structure_key([]) is None
    assert structure_key([_leg("CLK27"), _leg("CLM27", -1), _leg("CLN27")]) is None


BOOK = [
    _structure("box", "CL|BZ May27 FlyBox", _leg("CLK27-M27-N27"), _leg("BZK27-M27-N27", -1)),
    _structure("box2", "CL|BZ May27 FlyBox (reuse)", _leg("CLK27-M27-N27"), _leg("BZK27-M27-N27", -1)),
    _structure("dfly", "CL Aug27 DFly", _leg("CLQ27-U27-V27"), _leg("CLU27-V27-X27", -1)),
    _structure("wbs", "WBS Aug27 Fly", _leg("WBSQ27-U27-V27")),
    _structure("open", "CL Sep27 Spread", _leg("CLU27-V27")),
]
TRADES = {
    "box": [_trade(E.TRADE, 3, "sell", (11, 9, 0)), _trade(E.STRUCTURE_PARTIAL_EXIT, 1, "buy", (11, 9, 30), 10.0), _trade(E.ADD, 1, "sell", (11, 10, 0)),
            _trade(E.STRUCTURE_PARTIAL_EXIT, 2, "buy", (11, 11, 0), 40.0, 14.24), _trade(E.FULL_EXIT, 1, "buy", (12, 9, 0), -10.0)],
    "box2": [_trade(E.TRADE, 1, "sell", (12, 10, 0)), _trade(E.FULL_EXIT, 1, "buy", (12, 12, 0), 20.0)],  # the same contract and side: pooled
    "dfly": [_trade(E.TRADE, 2, "buy", (10, 9, 0), cost=14.24), _trade(E.FULL_EXIT, 2, "sell", (10, 15, 0), 60.0, 14.24), _trade(E.DISSOLVE, 1, "sell", (10, 16, 0), None, 0.0)],
    "wbs": [_trade(E.TRADE, 1, "sell", (6, 12, 0)), _trade(E.FULL_EXIT, 1, "buy", (6, 13, 0), 10.0)],
    "open": [_trade(E.TRADE, 2, "buy", (12, 9, 0))],  # nothing exited yet: no round trip
}


def test_round_trips_are_pooled_by_contract_and_side_with_the_cost_of_both_sides():
    real, unmatched = real_trips(BOOK, TRADES.get, TICKS)
    assert unmatched == ["WBS Aug27 Fly"] and len(real) == 2  # the open spread has no exit: neither a row nor unmatched
    box = real[real["product"] == "CLBZ"].iloc[0]
    assert (box["key"], box["direction"], box["trips"], box["lots"], box["days"]) == ("K27-M27-N27", "short", 4, 5.0, 2)
    assert box["gross"] == pytest.approx(60.0) and box["cost"] == pytest.approx(2 * (7.12 + 14.24 + 7.12 + 7.12))  # each exit's cost twice: entry and exit
    assert box["ticks"] == pytest.approx(60.0 / (5 * 10.0)) and box["net_lot"] == pytest.approx((60.0 - 71.2) / 5)
    assert box["win_rate"] == pytest.approx(0.5)  # 40 - 28.48 and 20 - 14.24 made money, 10 - 14.24 and -10 - 14.24 did not
    dfly = real[real["product"] == "CL"].iloc[0]
    assert (dfly["key"], dfly["direction"], dfly["trips"], dfly["lots"]) == ("Q27-U27-V27/U27-V27-X27", "long", 1, 2.0)  # the dissolve has no P&L: left out
    assert dfly["net_lot"] == pytest.approx((60.0 - 28.48) / 2)
    assert box["win_ticks"] == pytest.approx((1 + 2 + 2) / 3) and box["overnight"] == pytest.approx(0.25)  # one exit the day after its last entry
    assert dfly["win_ticks"] == pytest.approx(3.0) and dfly["overnight"] == 0.0
    assert real_trips([], TRADES.get, TICKS)[0].empty


def test_the_simulation_is_put_beside_the_same_contract_and_side():
    real, _ = real_trips(BOOK, TRADES.get, TICKS)
    scored = pd.DataFrame([
        {"product": "CLBZ", "symbol": "CLBZZXK27-M27-N27", "direction": "short", "label": "May27/Jun27/Jul27", "target": 2, "hit_prob": 0.4, "net_trade": -5.0, "score": -20.0},
        {"product": "CLBZ", "symbol": "CLBZZXK27-M27-N27", "direction": "long", "label": "May27/Jun27/Jul27", "target": 3, "hit_prob": 0.6, "net_trade": 4.0, "score": 20.0},
    ])
    out = compare(real, scored).set_index("product")
    assert (out.loc["CLBZ", "label"], out.loc["CLBZ", "sim_target"], out.loc["CLBZ", "sim_net_trade"], out.loc["CLBZ", "sim_score"]) == ("May27/Jun27/Jul27", 2, -5.0, -20.0)  # the short row
    assert out.loc["CL", "label"] == "" and np.isnan(out.loc["CL", "sim_score"])  # not simulated: blank, still listed
    assert compare(real, None)["sim_score"].isna().all()


def test_the_section_reads_the_gap_in_words_and_names_what_is_not_simulated():
    real, unmatched = real_trips(BOOK, TRADES.get, TICKS)
    scored = pd.DataFrame([{"product": "CLBZ", "symbol": "CLBZZXK27-M27-N27", "direction": "short", "label": "May27/Jun27/Jul27", "target": 2, "hit_prob": 0.4, "net_trade": -25.0, "score": -20.0}])
    section = churn_tab.archive_section(compare(real, scored), unmatched)
    text, table = str(section.children[0].children), section.children[1]
    assert "5 round trips on 2 contract sides: you netted $20 ($120 booked, $100 transaction cost at the full TC)" in text
    assert "On the 1 it also simulates you made -$2.24 per lot per round trip; the simulation expects -$25.00" in text and "harsher than your results" in text
    assert "Traded but not simulated (no candles for them): WBS Aug27 Fly." in text
    assert table.id == "churn-archive-table" and [r["contract"] for r in table.data] == ["CLBZ May27/Jun27/Jul27", "CL Aug27 DFly (not simulated)"]
    assert table.data[0]["sim_hit"] == 0.4 and table.data[1]["sim_score"] is None and set(table.tooltip_header) < {c["id"] for c in churn_tab.ARCHIVE_COLUMNS}
    kind = churn_tab.archive_section(compare(real, scored.assign(net_trade=30.0)), [])
    assert "kinder than your results" in str(kind.children[0].children)
    empty = churn_tab.archive_section(real.iloc[0:0], ["WBS Aug27 Fly"])
    assert "No closed round trip" in str(empty.children) and "WBS Aug27 Fly" in str(empty.children)


def test_the_callback_reads_the_book_and_never_fails_the_page(monkeypatch):
    calls = []

    class Service:
        def archive_comparison(self, params):
            calls.append(params.start_from)
            if params.start_from == 2:
                raise RuntimeError("boom")
            return compare(real_trips(BOOK, TRADES.get, TICKS)[0], None), []

        def calibration(self, params):
            if params.start_from == 3:
                raise RuntimeError("boom")
            return FIT

    monkeypatch.setattr(cb, "container", Container())
    assert cb.render_archive({"start_from": 1}, "v", 0)[0] is cb.no_update  # no service yet
    cb.container.churn_service = Service()
    assert cb.render_archive(None, "v", 0)[0] is cb.no_update
    section, changes, style = cb.render_archive({"start_from": 1}, "v", 0)
    assert "5 round trips" in str(section) and "Settings that fit your trades" in str(section) and changes == {"spread_fly": 0.0} and style["display"] == "block"
    failed = cb.render_archive({"start_from": 2}, "v", 0)
    assert "Could not compare" in str(failed[0]) and failed[1] is None and failed[2] == {"display": "none"}
    section, changes, style = cb.render_archive({"start_from": 3}, "v", 0)  # the calibration failing never hides the comparison
    assert "5 round trips" in str(section) and changes is None and style == {"display": "none"} and calls == [1, 2, 3]


def _real(rows):
    return pd.DataFrame(rows, columns=["product", "key", "direction", "trips", "lots", "ticks", "win_ticks", "overnight", "win_rate"])


def test_the_fit_is_the_spread_and_stop_closest_to_your_ticks():
    real = _real([("CLBZ", "K27", "short", 8, 10.0, 0.5, 1.6, 0.25, 0.5), ("CLBZ", "M27", "short", 4, 10.0, 0.7, 1.4, 0.0, 0.5),
                  ("CL", "Q27/U27", "long", 3, 3.0, 2.0, 2.0, 1.0, 1.0), ("BZ", "K27", "long", 5, 5.0, 1.0, 1.0, 0.0, 1.0)])
    asked = []

    def simulate(product, key, direction, target, spread, stop):
        asked.append((product, key, target))
        return 0.6 - spread - (0.3 if stop == 5.0 else 0.0), 0.5  # every tick of spread costs a tick; only the widest stop differs

    fit = fit_settings(real, lambda product, key: {"CLBZ": "fly", "CL": "dfly"}.get(product), simulate, {"outright": 1.0, "spread": 1.0, "fly": 2.0, "dfly": 3.0}, 3.0)
    assert ("CLBZ", "K27", 2) in asked and ("CLBZ", "M27", 1) in asked and not any(a[0] == "BZ" for a in asked)  # your own target; what is not simulated is left out
    fly, dfly = fit["families"]
    assert (fly["family"], fly["trips"], fly["fitted"], fly["spread_now"], fly["spread_fit"]) == ("fly", 12, True, 2.0, 0.0)
    assert fly["ticks"] == pytest.approx(0.6) and fly["sim_now"] == pytest.approx(-1.4) and fly["sim_fit"] == pytest.approx(0.6) and fly["overnight"] == pytest.approx(2 / 12)
    assert (dfly["family"], dfly["fitted"], dfly["spread_fit"], dfly["overnight"]) == ("dfly", False, 3.0, 1.0)  # three round trips: reported, not fitted
    assert fit["changes"] == {"spread_fly": 0.0} and fit["stop_fit"] == fit["stop_now"] == 3.0  # the stop makes no clear difference: kept

    def stop_matters(product, key, direction, target, spread, stop):
        return 0.6 - spread - 0.4 * abs(stop - 1.0), 0.5

    moved = fit_settings(real, lambda product, key: {"CLBZ": "fly"}.get(product), stop_matters, {"fly": 0.0}, 3.0)
    assert moved["changes"] == {"stop_mult": 1.0} and moved["families"][0]["spread_fit"] == 0.0
    assert fit_settings(real, lambda product, key: None, simulate, {"fly": 2.0}, 3.0) == {"families": [], "stop_now": 3.0, "stop_fit": 3.0, "changes": {}}
    assert fit_settings(real, lambda product, key: "fly", lambda *a: None, {"fly": 2.0}, 3.0)["families"] == []  # nothing could be simulated


FIT = {"families": [{"family": "fly", "trips": 12, "lots": 20.0, "ticks": 0.6, "overnight": 0.25, "target": 1.5, "win_rate": 0.5, "spread_now": 2.0, "fitted": True, "sim_now": -1.4, "spread_fit": 0.0, "sim_fit": 0.6, "hit_fit": 0.5},
                    {"family": "dfly", "trips": 3, "lots": 3.0, "ticks": 2.0, "overnight": 1.0, "target": 2.0, "win_rate": 1.0, "spread_now": 3.0, "fitted": False, "sim_now": -2.1, "spread_fit": 3.0, "sim_fit": -2.1, "hit_fit": 0.0}],
       "stop_now": 3.0, "stop_fit": 3.0, "changes": {"spread_fly": 0.0}}


def test_the_fit_is_read_out_in_words_and_applied_by_the_button():
    text = str(churn_tab.fit_section(FIT))
    assert "Fly: 12 round trips, 25% held overnight. You captured +0.60 ticks per lot per round trip (average target 1.5 ticks)" in text
    assert "gives -1.40 with your 2-tick spread and +0.60 with a 0-tick spread: use 0." in text
    assert "D-fly: 3 round trips, 100% held overnight." in text and "Too few round trips to fit a spread" in text and "no spread or stop makes it match them" in text
    assert "Stop: your 3 targets fit as well as any other; keep it." in text and "Suggested: Fly spread 0." in text
    kept = {**FIT, "changes": {}, "stop_fit": 1.0, "families": [{**FIT["families"][0], "spread_fit": 2.0, "sim_fit": -1.4}]}
    text = str(churn_tab.fit_section(kept))
    assert "which no other spread beats by a clear margin: keep it." in text and "Stop: 1 target fits your results better than your 3." in text and "Nothing to change" in text
    assert "Stop: your 1 target fits as well as any other; keep it." in str(churn_tab.fit_section({**FIT, "stop_now": 1.0, "stop_fit": 1.0}))
    assert churn_tab.fit_section(None).children is None
    assert list(churn_tab.FIT_INPUTS.values()) == ["churn-p-sp-out", "churn-p-sp-spr", "churn-p-sp-fly", "churn-p-sp-dfly", "churn-p-stop"]
    assert set(churn_tab.FIT_INPUTS.values()) <= {i[0] for i in churn_tab.PARAM_INPUTS}
    assert cb.apply_fit(None, {"spread_fly": 0.0}) == (cb.no_update,) * 6 and cb.apply_fit(1, None) == (cb.no_update,) * 6
    assert cb.apply_fit(1, {"spread_fly": 0.0, "stop_mult": 1.0}) == (cb.no_update, cb.no_update, 0.0, cb.no_update, 1.0, "Applied: Fly spread 0, Stop 1. The table is being rebuilt with them.")
