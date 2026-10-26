"""Seasonality: days before expiry, the structure history, the band, the flag, 'also seen', the service and the page."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.curve_calendar import api_symbol, contract_expiry
from core.curve_compare import CurvePoint
from core.seasonality import (
    CHEAP, FLAGGED, MEDIAN_MAD, NO_DATA, OK, RICH, WATCH, SeasonalityService, SeasonOptions, analyse, band, build_curve, evaluate,
    pivot_frame, structure_frame, structure_symbols, trading_days_to, trailing_run,
)
from ui.callbacks import seasonality_callbacks as callbacks
from ui.container import Container
from ui.layouts import seasonality_tab as tab

TODAY = date(2026, 10, 23)
CLOCK = lambda: datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)  # noqa: E731


def td_of(day, expiry, product="CL") -> int:
    return int(trading_days_to(pd.DatetimeIndex([pd.Timestamp(day)]), expiry, "nymex" if product == "CL" else "ice_uk")[0])


def make_frame(value_fn, product="CL", legs=(4, 5), volume=1000.0):
    """Settlement rows for April (the first leg, priced `value_fn(year, td)`) and May (zero) contracts of 2019..2027, so
    the April/May spread equals value_fn. Rows exist on business days up to yesterday or the expiry."""
    rows = []
    for year in range(2019, 2028):
        expiry = contract_expiry(product, 4, year)
        end = min(expiry, TODAY - timedelta(days=1))
        for day in pd.bdate_range(expiry - timedelta(days=520), end):
            td = td_of(day, expiry, product)
            for month, value in ((4, value_fn(year, td)), (5, 0.0)):
                rows.append({"contract": api_symbol(product, month, year), "date": day, "oi": 1.0, "close": value, "volume": volume})
    return pd.DataFrame(rows)


def base_value(year, td):
    return 1.0 + 0.01 * (year - 2019) + 0.001 * td


class FakeStore:
    def __init__(self, frame, history_days=1500):
        self._frame = frame
        self.version = "v1"
        self.loaded_history_days = history_days

    def frame(self, product):
        return self._frame if product == "CL" else self._frame.iloc[0:0]

    def has_data(self):
        return not self._frame.empty


class FakeCompare:
    def __init__(self, values):
        self.values = values  # position -> price

    def view(self, product, months, family, method):
        points = [CurvePoint(p, f"P{p}", f"P{p}", "", (), value=v, source="quote", age_seconds=5.0) for p, v in self.values.items()]
        return type("View", (), {"rows": [type("Row", (), {"point": p})() for p in points]})()


def season_of(frame, options=SeasonOptions(), family="spread", year=2027):
    close, volume = pivot_frame(frame)
    return analyse(close, volume, "CL", family, 4, year, 1, options, TODAY)


# ----------------------------------------------------------------------
# Days before expiry and the history of a structure
# ----------------------------------------------------------------------


def test_trading_days_to_counts_business_days_and_skips_holidays():
    expiry = date(2026, 12, 24)  # a Thursday
    days = pd.DatetimeIndex(["2026-12-24", "2026-12-23", "2026-12-21", "2026-12-18"])
    assert list(trading_days_to(days, expiry, "nymex")) == [0, 1, 3, 4]  # Fri 18 -> Mon 21 -> Tue 22 -> Wed 23 -> Thu 24
    # Christmas Friday is a NYMEX holiday: from Wed 23 to Mon 28 is two trading days, not three
    assert trading_days_to(pd.DatetimeIndex(["2026-12-23"]), date(2026, 12, 28), "nymex")[0] == 2


def test_structure_symbols_run_across_the_year_end():
    assert structure_symbols("CL", "dfly", 11, 2026) == ["CLX26", "CLZ26", "CLF27", "CLG27"]
    assert structure_symbols("BRN", "fly", 12, 2026) == ["COZ26", "COF27", "COG27"]
    assert structure_symbols("G", "outright", 3, 2027) == ["GOH27"]


def test_structure_frame_weights_the_legs_and_takes_the_thinnest_volume():
    index = pd.to_datetime(["2026-01-05", "2026-01-06"])
    close = pd.DataFrame({"A": [10.0, 11.0], "B": [9.0, 9.5], "C": [8.0, 8.0], "D": [7.0, np.nan]}, index=index)
    volume = pd.DataFrame({"A": [5.0, 5.0], "B": [3.0, 9.0], "C": [8.0, 8.0], "D": [4.0, 4.0]}, index=index)
    built = structure_frame(close, volume, ["A", "B", "C", "D"], (1, -3, 3, -1))
    assert list(built.index) == [index[0]]  # the day D is missing is dropped
    assert built["price"].iloc[0] == pytest.approx(10 - 27 + 24 - 7)
    assert built["volume"].iloc[0] == 3.0
    assert structure_frame(close, volume, ["A", "Z"], (1, -1)) is None


def test_build_curve_indexes_by_days_before_expiry_and_ends_at_the_last_settlement_of_a_finished_contract():
    close, volume = pivot_frame(make_frame(base_value))
    curve = build_curve(close, volume, "CL", "spread", 4, 2025, 400, TODAY, live=False)
    assert curve.frame.index.min() == 0 and curve.expiry == close["CLJ25"].dropna().index.max().date()
    assert curve.frame["price"].loc[0] == pytest.approx(base_value(2025, 0))
    assert curve.label == "Apr25/May25"


# ----------------------------------------------------------------------
# Band, run and flag
# ----------------------------------------------------------------------


def test_band_is_the_mean_and_sample_sd_of_the_earlier_years_and_never_narrower_than_a_tick():
    close, volume = pivot_frame(make_frame(base_value))
    curves = [build_curve(close, volume, "CL", "spread", 4, y, 300, TODAY, False) for y in (2024, 2025, 2026)]
    result = band(curves, 300, "mean_sd", 0.01)
    values = [base_value(y, 50) for y in (2024, 2025, 2026)]
    assert result["center"].loc[50] == pytest.approx(np.mean(values))
    assert result["spread"].loc[50] == pytest.approx(max(np.std(values, ddof=1), 0.01))
    assert result["n"].loc[50] == 3
    robust = band(curves, 300, MEDIAN_MAD, 0.01)
    assert robust["center"].loc[50] == pytest.approx(np.median(values))
    assert band([curves[0]], 300, "mean_sd", 0.01)["center"].isna().all()  # one year is no band


def test_trailing_run_counts_same_side_days_ending_at_the_latest():
    assert trailing_run(np.array([0.0, 2.5, 3.0, 2.2]), 2.0) == (3, 1)
    assert trailing_run(np.array([-3.0, -3.0, 2.5]), 2.0) == (1, 1)  # a switch of side ends the run
    assert trailing_run(np.array([3.0, 1.0]), 2.0) == (0, 0)
    assert trailing_run(np.array([3.0, np.nan, 3.0]), 2.0) == (1, 1)  # a missing day breaks the run


def test_a_structure_far_above_the_band_for_several_days_is_flagged_rich():
    season = season_of(make_frame(lambda y, td: base_value(y, td) + (1.0 if y == 2027 else 0.0)))
    assert season.run >= 3 and season.run_side == 1
    live = season.last_price + 0.05
    result = evaluate(season, live)
    assert result.status == FLAGGED and result.direction == RICH
    assert result.z > 2 and result.run == season.run + 1  # today's live price adds a day
    assert result.rank == result.years == len(season.used)  # above every earlier year
    # live back inside the band: the run ends
    inside = evaluate(season, float(season.bands["center"][season.td_now]))
    assert inside.status == OK and inside.run == 0


def test_a_single_day_outside_is_only_watch_and_cheap_is_the_other_side():
    # every year follows the pattern; only today's live price is displaced
    season = season_of(make_frame(base_value))
    centre = float(season.bands["center"][season.td_now])
    spread = float(season.bands["spread"][season.td_now])
    low = evaluate(season, centre - 3 * spread)
    assert low.status == WATCH and low.direction == CHEAP and low.run == 1 and low.rank == 0
    assert evaluate(season, centre).status == OK


def test_with_no_live_price_the_last_settlement_is_judged():
    season = season_of(make_frame(lambda y, td: base_value(y, td) + (1.0 if y == 2027 else 0.0)))
    result = evaluate(season, None)
    assert not result.used_live and result.price == pytest.approx(season.last_price)
    assert result.status == FLAGGED and result.run == season.run


def test_excluding_a_year_removes_it_from_the_band_but_keeps_it_on_the_chart_data():
    frame = make_frame(lambda y, td: base_value(y, td) + (5.0 if y == 2022 else 0.0))
    everyone = season_of(frame)
    without = season_of(frame, SeasonOptions(excluded=(2022,)))
    assert [c.year for c in without.past] == [c.year for c in everyone.past] and 2022 in [c.year for c in without.past]
    assert 2022 not in [c.year for c in without.used]
    assert without.bands["center"][50] < everyone.bands["center"][50]
    assert without.bands["spread"][50] < everyone.bands["spread"][50]


def test_not_enough_earlier_years_is_no_data():
    season = season_of(make_frame(base_value), SeasonOptions(years=1))
    assert evaluate(season, 1.0).status == NO_DATA
    assert evaluate(season_of(make_frame(base_value), family="outright", year=2031), 1.0).status == NO_DATA  # no history at all


def test_volume_is_the_thinnest_leg_and_the_usual_is_the_median_of_earlier_years():
    season = season_of(make_frame(base_value, volume=500.0))
    assert season.last_volume == 500.0 and season.usual_volume == 500.0


def test_rebasing_shows_the_change_since_the_reference_day():
    season = season_of(make_frame(base_value), SeasonOptions(rebase_days=200))
    year = season.past[0]
    assert year.frame["price"].loc[200] == pytest.approx(0.0)
    assert year.frame.index.max() <= 200
    assert year.frame["price"].loc[150] == pytest.approx(base_value(year.year, 150) - base_value(year.year, 200))
    live = base_value(2027, season.td_now) + 0.3
    assert evaluate(season, live).price == pytest.approx(0.3 - 0.0 + (base_value(2027, season.td_now) - season.current.ref_value))
    too_far = season_of(make_frame(base_value), SeasonOptions(rebase_days=5))  # today is further than 5 days from expiry
    assert evaluate(too_far, 1.0).status == NO_DATA


def test_similar_years_lists_earlier_years_that_made_the_same_move_and_whether_they_came_back():
    spike = lambda y, td: base_value(y, td) + (1.0 if y == 2027 else 0.0) + (1.0 if y == 2023 and 85 <= td <= 140 else 0.0)  # noqa: E731
    season = season_of(make_frame(spike))
    assert 85 <= season.td_now <= 140
    result = evaluate(season, season.last_price + 0.05)
    assert [s.year for s in result.similar] == [2023]
    only = result.similar[0]
    assert only.peak_z >= 2 and only.reverted is True  # back inside the band 20 trading days later
    assert evaluate(season, float(season.bands["center"][season.td_now])).similar == []  # not outside: nothing to compare


# ----------------------------------------------------------------------
# Service
# ----------------------------------------------------------------------


def service_for(frame, prices):
    return SeasonalityService(FakeStore(frame), compare=lambda: FakeCompare(prices), clock=CLOCK)


def test_table_has_one_row_per_structure_and_applies_the_live_prices():
    frame = make_frame(lambda y, td: base_value(y, td) + (1.0 if y == 2027 else 0.0))
    service = service_for(frame, {5: 3.0, 4: 3.0})
    table = service.table("CL", "spread", 6, SeasonOptions())
    assert len(table.rows) == 5  # six contracts -> five spreads (Dec26/Jan27 ... Apr27/May27)
    april = table.rows[4]
    assert april.season.position == 5 and april.season.label == "Apr27/May27"
    assert april.point.value == 3.0 and april.evaluation.used_live
    assert table.rows[0].evaluation.status == NO_DATA  # the synthetic store only holds April/May
    assert service.row("CL", "spread", 6, 5, SeasonOptions()).season.position == 5
    assert service.row("CL", "spread", 6, 99, SeasonOptions()) is None


def test_table_without_live_prices_uses_settlements_and_without_history_says_so():
    table = SeasonalityService(FakeStore(make_frame(base_value)), compare=None, clock=CLOCK).table("CL", "spread", 6, SeasonOptions())
    assert not table.rows[4].evaluation.used_live and table.rows[4].evaluation.status != NO_DATA
    empty = SeasonalityService(FakeStore(pd.DataFrame(columns=["contract", "date", "oi", "close", "volume"])), None, CLOCK)
    result = empty.table("CL", "outright", 3, SeasonOptions())
    assert all(r.evaluation.status == NO_DATA for r in result.rows) and "downloading" in result.notes[0]


def test_a_watch_window_deeper_than_the_stored_history_is_reported():
    service = SeasonalityService(FakeStore(make_frame(base_value), history_days=365), None, CLOCK)
    assert "Raise 'OI history'" in service.notes("CL", SeasonOptions(max_days=1200))[0]
    assert service.notes("CL", SeasonOptions(max_days=200)) == []


def test_a_market_wide_move_is_reported_instead_of_a_set_of_kinks():
    frame = make_frame(lambda y, td: base_value(y, td) + (1.0 if y == 2027 else 0.0))
    # only the Apr/May spread has data in the synthetic store, so build a table by hand from copies of that row
    table = service_for(frame, {5: 3.0}).table("CL", "spread", 6, SeasonOptions())
    row = table.rows[4]
    assert row.evaluation.side == 1
    notes = SeasonalityService._regime_note([row] * 5)
    assert notes and "market-wide" in notes[0]
    assert SeasonalityService._regime_note([row] * 2) == []


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------


@pytest.fixture
def wired():
    frame = make_frame(lambda y, td: base_value(y, td) + (1.0 if y == 2027 else 0.0))
    box = Container()
    box.seasonality = service_for(frame, {5: 3.0})
    original = callbacks.container
    callbacks.container = box
    yield box
    callbacks.container = original


CONTROLS = ("CL", "spread", 6, 7, 250, "flies", "mean_sd", 2.0, 3, [], "price", 250)


def test_table_rows_and_status_line(wired):
    rows, status = callbacks.render_table(0, *CONTROLS)
    first = rows[4]
    assert first["id"] == 5 and first["contract"] == "Apr27/May27" and first["status"] == "FLAGGED RICH"
    assert first["live"] == 3.0 and first["z"] > 2 and first["run"] >= 4
    assert first["also"].startswith("none of") and first["volume"] == 1000.0 and first["volume_ratio"] == 1.0
    assert "flagged" in status.children[0].children


def test_bad_controls_give_a_readable_message_not_a_crash(wired):
    for index, bad in ((2, 0), (3, 99), (4, 5), (7, "x"), (8, 0)):
        controls = list(CONTROLS)
        controls[index] = bad
        rows, status = callbacks.render_table(0, *controls)
        assert rows == [] and "enter" in status.children.lower()


def test_chart_has_a_line_per_year_the_band_and_the_live_marker(wired):
    figure = callbacks.render_chart(0, 5, True, *CONTROLS)
    names = [t.name for t in figure.data if t.name]
    assert "Apr26" in names and "Apr20" in names
    assert any("band" in n for n in names) and "Average" in names and "Live" in names
    assert any("settlements" in n for n in names)
    assert figure.layout.xaxis.range == (pytest.approx(figure.layout.xaxis.range[0]), 0)
    no_band = callbacks.render_chart(0, 5, False, *CONTROLS)
    assert not any("band" in (t.name or "") for t in no_band.data)
    excluded = list(CONTROLS)
    excluded[9] = [2022]
    figure = callbacks.render_chart(0, 5, True, *excluded)
    assert "Apr22 (excluded)" in [t.name for t in figure.data]
    assert callbacks.render_chart(0, None, True, *CONTROLS).layout.annotations[0].text == tab.PLACEHOLDER


def test_selecting_a_row_and_resetting_when_a_control_changes():
    assert callbacks.selected_position("season-table", {"row": 0, "column": 1, "row_id": 3}) == 3
    assert callbacks.selected_position("season-product", {"row": 0, "row_id": 3}) is None
    assert callbacks.selected_position("season-table", None) is None


def test_option_parsing_and_the_dfly_and_reference_inputs():
    options = callbacks.parse_options(5, 400, "bogus", "median_mad", 2.5, 4, [2022, 2020], "change", 300)
    assert options.years == 5 and options.max_days == 400 and options.rebase_days == 300
    assert options.excluded == (2020, 2022) and options.band == MEDIAN_MAD and options.dfly_method == "flies"
    assert callbacks.parse_options(5, 400, "flies", "mean_sd", 2, 3, None, "price", None).rebase_days is None
    assert callbacks.method_disabled("fly") is True and callbacks.method_disabled("dfly") is False
    assert callbacks.reference_disabled("price") is True and callbacks.reference_disabled("change") is False


def test_route_and_sidebar_entry_exist():
    from ui.callbacks.shell_callbacks import _ROUTES
    from ui.layouts.shell import NAV_ITEMS

    assert _ROUTES["/seasonality"] == ("seasonality_tab", "seasonality_layout")
    assert any(path == "/seasonality" for path, _ in NAV_ITEMS)
    assert tab.seasonality_layout() is not None
