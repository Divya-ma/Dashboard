"""Roll days: front-month series that do not count the contract switch on the first trading day after an expiry."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from core.curve_calendar import api_symbol, contract_expiry
from core.roll_days import ChangeSeries, build_leg_data, difference, total_between

TODAY = date(2026, 10, 23)
ALL_DAYS = pd.bdate_range("2021-01-01", "2030-12-31")


def leg_frame(product, last_day, step, first=(2023, 1), last=(2028, 12), drift=0.1):
    """Monthly contracts that share one market (it drifts by `drift` every business day) and differ by `step` of contango per month."""
    rows = []
    for index in range(first[0] * 12 + first[1] - 1, last[0] * 12 + last[1]):
        year, month = index // 12, index % 12 + 1
        expiry = contract_expiry(product, month, year)
        days = pd.bdate_range(pd.Timestamp(expiry) - pd.Timedelta(days=300), min(pd.Timestamp(expiry), pd.Timestamp(last_day)))
        if len(days) == 0:
            continue
        rows.append(pd.DataFrame({
            "contract": api_symbol(product, month, year), "date": days, "oi": 1000.0 + 5.0 * np.arange(len(days)),
            "close": 70.0 + drift * ALL_DAYS.get_indexer(days) + step * (index - first[0] * 12), "volume": 10.0,
        }))
    return pd.concat(rows, ignore_index=True)


def test_the_first_trading_day_after_an_expiry_is_a_roll_and_its_jump_is_not_counted():
    leg = build_leg_data("CL", leg_frame("CL", TODAY - timedelta(days=1), step=0.5))
    assert leg is not None
    series = leg.series
    expiry = contract_expiry("CL", 6, 2025)
    after = next(day for day in series.index if day.date() > expiry)
    event = leg.event_on(after.date())
    assert event is not None and event.from_contract == "CLM25" and event.to_contract == "CLN25"
    pos = series.index.get_loc(after)
    assert series.is_roll[pos] and np.isnan(series.changes[pos])
    assert event.jump == pytest.approx(0.5 + 0.1)  # a naive continuous series would show the contango step as a move
    quiet = series.changes[pos + 5]
    assert quiet == pytest.approx(0.1) and not series.is_roll[pos + 5]


def test_every_expiry_makes_one_roll_day_and_nothing_else_does():
    leg = build_leg_data("BRN", leg_frame("BRN", TODAY - timedelta(days=1), step=0.2))
    assert int(leg.series.is_roll.sum()) == len(leg.rolls) > 20
    assert set(np.round(leg.series.changes[np.isfinite(leg.series.changes)], 6)) == {0.1}  # only same-contract changes remain
    assert int(np.isfinite(leg.series.changes).sum()) > 500


def test_a_window_read_sums_the_valid_changes_and_names_the_roll_days_it_skipped():
    leg = build_leg_data("CL", leg_frame("CL", TODAY - timedelta(days=1), step=0.5))
    expiry = contract_expiry("CL", 6, 2025)
    roll_day = next(day for day in leg.series.index if day.date() > expiry)
    read = leg.series.read(roll_day + pd.Timedelta(days=4), 10)
    assert roll_day in read.rolls and read.valid == 9
    assert read.total == pytest.approx(0.9)  # nine days of 0.1; the 0.6 switch is not in it
    assert read.std == pytest.approx(0.0, abs=1e-9)
    assert leg.series.read(pd.Timestamp("1990-01-01"), 10) is None


def test_total_between_leaves_the_roll_day_out_too():
    leg = build_leg_data("CL", leg_frame("CL", TODAY - timedelta(days=1), step=0.5))
    expiry = contract_expiry("CL", 6, 2025)
    roll_day = next(day for day in leg.series.index if day.date() > expiry)
    start, end = roll_day - pd.Timedelta(days=3), roll_day + pd.Timedelta(days=3)
    days = leg.series.index[(leg.series.index > start) & (leg.series.index <= end)]
    assert roll_day in days
    assert total_between(leg.series, start, end) == pytest.approx(0.1 * (len(days) - 1))  # every day but the switch
    assert total_between(leg.series, pd.Timestamp("2040-01-01"), pd.Timestamp("2040-02-01")) is None


def test_a_window_with_too_few_usable_days_gives_no_reading():
    stamps = pd.bdate_range("2026-01-05", periods=10)
    series = ChangeSeries("X", stamps, np.array([np.nan] * 10), np.zeros(10, dtype=bool))
    read = series.read(stamps[-1], 10)
    assert read.total is None and read.valid == 0


def test_cl_minus_brent_drops_a_day_when_either_leg_rolled():
    cl = build_leg_data("CL", leg_frame("CL", TODAY - timedelta(days=1), step=0.3))
    brent = build_leg_data("BRN", leg_frame("BRN", TODAY - timedelta(days=1), step=0.2))
    box = difference(cl.series, brent.series, "CL-BRN")
    assert int(box.is_roll.sum()) >= len(cl.rolls) + len(brent.rolls) - 5  # (a few coincide)
    usable = box.changes[np.isfinite(box.changes)]
    assert len(usable) > 400 and np.allclose(usable, 0.0)  # same drift on both legs: no move except at a switch
    # the Brent leg expires about three weeks before CL's: the two roll days of a month are different days
    month_cl = next(e for e in cl.rolls if e.from_contract == "CLM25")
    month_brent = next(e for e in brent.rolls if e.from_contract == "COM25")
    assert month_brent.day < month_cl.day


def test_dates_before_the_store_really_starts_are_not_a_front_month_series():
    leg = build_leg_data("CL", leg_frame("CL", TODAY - timedelta(days=1), step=0.5, first=(2024, 6)))
    early = leg.series.index < pd.Timestamp("2024-04-01")
    assert early.any() and np.isnan(leg.series.changes[early]).all() and not leg.series.is_roll[early].any()


def test_an_empty_frame_has_no_leg_data():
    assert build_leg_data("CL", pd.DataFrame(columns=["contract", "date", "oi", "close", "volume"])) is None


def test_settlements_only_products_have_a_calendar_too():
    leg = build_leg_data("BZ", leg_frame("BZ", TODAY - timedelta(days=1), step=0.2))
    assert leg is not None and leg.calendar == "nymex" and leg.rolls
