"""The front contract's last-month check, the vectorised roll estimate and the trimmed monitor tables."""

import numpy as np
import pandas as pd
import pytest

from core.oi_monitor import MonitorParams, front_verdict
from core.oi_roll import attribute_rolls, daily_roll_matrix
from tests.test_oi import TODAY, synthetic_frame
from tests.test_oi_monitor import HISTORY, _walk, engine
from ui.layouts import oi_monitor_view as view

FRONT = "CLX26"  # Nov26: 8 days from expiry on TODAY


@pytest.fixture(scope="module")
def base_frame():
    return synthetic_frame("CL", 40)


def stuck_open(frame: pd.DataFrame, symbol: str, days: int) -> pd.DataFrame:
    """The contract's OI frozen at its value `days` trading days ago: nothing rolls, nothing leaves."""
    out = frame.copy()
    rows = out[out["contract"] == symbol].sort_values("date")
    out.loc[rows.index[-days:], "oi"] = rows["oi"].iloc[-days - 1]
    return out


def metric(front: dict, key: str) -> dict:
    return next(m for m in front["metrics"] if m["key"] == key)


# ----------------------------------------------------------------------
# The front contract's last month
# ----------------------------------------------------------------------


def test_the_front_contract_is_checked_against_past_contracts_at_the_same_point(base_frame):
    front = engine(base_frame).result().front
    assert front["symbol"] == FRONT and front["dte"] == 8 and front["note"] is None
    assert [m["key"] for m in front["metrics"]] == ["remaining", "drop", "rolled", "to_next"]
    assert front["level"] == "OK" and "in line" in front["verdict"].lower()
    remaining = metric(front, "remaining")
    assert 0.02 < remaining["now"] < 0.5 and remaining["b_n"] >= 3 and "a_n" not in remaining
    assert remaining["b_min"] <= remaining["b_median"] <= remaining["b_max"]
    # the drop is measured from 30 days out, so open + dropped is what was open then: more than now, at most the whole peak
    assert remaining["now"] < remaining["now"] + metric(front, "drop")["now"] <= 1.0 + 1e-9


def test_oi_that_stays_open_and_does_not_roll_is_flagged_as_squeeze_like(base_frame):
    front = engine(stuck_open(base_frame, FRONT, 12)).result().front
    assert metric(front, "remaining")["b_z"] > 2.5 and metric(front, "rolled")["b_z"] < -2.5
    assert front["level"] == "ALERT" and "squeeze" in front["verdict"]


def test_oi_that_leaves_early_is_the_opposite(base_frame):
    early = base_frame.copy()
    rows = early[early["contract"] == FRONT].sort_values("date")
    early.loc[rows.index[-12:], "oi"] *= 0.4
    front = engine(early).result().front
    assert metric(front, "remaining")["b_z"] < -2.5
    assert front["level"] == "INFO"


def test_the_threshold_is_the_one_on_the_page(base_frame):
    frame = stuck_open(base_frame, FRONT, 12)
    assert engine(frame, z_threshold=2.5).result().front["level"] == "ALERT"
    assert engine(frame, z_threshold=50.0).result().front["level"] == "OK"


def test_before_the_last_month_the_check_says_so(base_frame):
    later = base_frame[base_frame["contract"] != FRONT]  # the front is then Dec26, 39 days out
    front = engine(later).result().front
    assert front["label"] == "Dec26" and front["dte"] == 39 and "last 30 days" in front["note"] and front["metrics"] == []
    assert "last 30 days" in str(view.front_check_body(front, MonitorParams()))


def test_verdicts_name_the_reason():
    assert front_verdict(3.0, -3.0, 2.5)[0] == "ALERT"
    assert front_verdict(3.0, 0.0, 2.5)[0] == "WATCH"
    assert front_verdict(-3.0, 3.0, 2.5)[0] == "INFO" and "ahead of schedule" in front_verdict(-3.0, 3.0, 2.5)[1]
    assert "closed out" in front_verdict(-3.0, 0.0, 2.5)[1]
    assert front_verdict(0.0, -3.0, 2.5)[0] == "WATCH" and front_verdict(0.0, 3.0, 2.5)[0] == "INFO"
    assert front_verdict(0.5, 0.5, 2.5)[0] == "OK"
    assert front_verdict(None, 1.0, 2.5)[0] == "" and front_verdict(float("nan"), 1.0, 2.5)[0] == ""
    assert front_verdict(3.0, None, 2.5)[0] == "WATCH"  # no roll estimate: still reports what is open


def test_the_vectorised_roll_estimate_equals_the_day_by_day_one():
    rng = np.random.default_rng(5)
    names = ["A", "B", "C", "D", "E"]
    delta = pd.DataFrame(rng.normal(0, 100, (40, 5)), columns=names, index=pd.date_range("2026-01-01", periods=40))
    delta.iloc[3, 1] = np.nan  # a missing contract that day
    delta.iloc[7] = 5.0  # only rises
    delta.iloc[9] = -5.0  # only falls
    mats = daily_roll_matrix(delta, {"A": "B", "B": "C", "C": "D", "D": "E"})
    for day in delta.index:
        one = attribute_rolls(delta.loc[day])
        for name in names:
            assert mats["rolled_out"].loc[day, name] == pytest.approx(float(one.rolled_out.get(name, 0.0)))
            value = delta.loc[day, name]
            assert mats["fall"].loc[day, name] == pytest.approx(max(-value, 0.0) if np.isfinite(value) else 0.0)
        for donor, following in {"A": "B", "B": "C"}.items():
            has = donor in one.flows.index and following in one.flows.columns
            assert mats["to_next"].loc[day, donor] == pytest.approx(float(one.flows.loc[donor, following]) if has else 0.0)


# ----------------------------------------------------------------------
# The tables
# ----------------------------------------------------------------------


def test_rolled_in_counts_only_what_came_from_the_front_contract(base_frame):
    eng = engine(base_frame)
    roll = eng.result().roll
    assert roll["symbol"].iloc[0] == FRONT and np.isnan(roll["rolled_in"].iloc[0])  # nothing rolls into the front itself
    flows = attribute_rolls(eng.oi.iloc[-1] - eng.oi.iloc[-2]).flows
    for r in roll.iloc[1:].itertuples():
        expected = float(flows.loc[FRONT, r.symbol]) if FRONT in flows.index and r.symbol in flows.columns else 0.0
        assert r.rolled_in == pytest.approx(expected)
    assert roll["rolled_in"].iloc[1:].sum() <= -roll["change"].iloc[0] + 1e-6  # no more than the front lost


def test_the_monitor_table_has_only_the_columns_that_were_asked_for(base_frame):
    result = engine(base_frame).result()
    ids = [c["id"] for c in view.monitor_columns(MonitorParams())]
    for removed in ("b_n", "priority", "direction", "measure", "a_median", "a_z"):
        assert removed not in ids and all(removed not in row for row in view.monitor_rows(result))
    assert {"label", "oi", "b_median", "b_z", "share", "share_z", "status", "run", "since"} <= set(ids)
    assert "oi-p-recent" not in [c for c, *_ in view.PARAM_INPUTS]  # no "last N contracts" input
    assert not any("priority" in str(style) for style in view.monitor_styles(MonitorParams()))
    roll_ids = [c["id"] for c in view.ROLL_COLUMNS]
    assert roll_ids == ["label", "change", "exited", "rolled_in", "fresh"] and "rolled_out" not in roll_ids
    assert "rolled_out" not in view.roll_rows(result)[0]


def test_the_table_headings_explain_themselves(base_frame):
    body = view.monitor_body(engine(base_frame).result(), MonitorParams(), TODAY)
    table = next(x for x in _walk(body) if getattr(x, "id", None) == "oi-monitor-table")
    assert "share of the total OI" in table.tooltip_header["share"]["value"]
    assert "standard deviations" in table.tooltip_header["share_z"]["value"]
    text = str(body)
    assert "oi-front-table" in text and "Front contract Nov26" in text and "oi-roll-table-live" in text
    assert "Priority" not in text


def test_the_front_chart_has_both_bands_and_the_live_path(base_frame):
    figure = view.build_front_chart(engine(base_frame).result().front["chart"])
    names = [t.name for t in figure.data if t.name]
    assert not any("Last contracts" in n for n in names) and any("Same month" in n for n in names) and any("(live)" in n for n in names)
    assert figure.layout.uirevision == "Nov26" and len(figure.data[-1].x) > 10
    assert len(view.build_front_chart(None).data) == 0


def test_the_front_check_is_part_of_the_cached_monitor_result(base_frame):
    assert engine(base_frame).result().front is not None
    empty = pd.DataFrame(columns=["contract", "date", "oi", "close", "volume"])
    assert engine(empty).result().front is None
    assert HISTORY == 900
