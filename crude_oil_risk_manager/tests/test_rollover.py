"""Strategy Rollover: one contract through the front expiries, zero base, the average, the CL-BZ front box, the service and the page."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from core.curve_calendar import PRODUCTS, api_symbol, contract_expiry
from core.oi_data import OiProgress
from core.rollover import RolloverLine, RolloverOptions, RolloverService, _index, _month_year, align, average_of
from core.seasonality import trading_days_to
from ui.callbacks import rollover_callbacks as callbacks
from ui.container import Container
from ui.layouts import rollover_tab as tab

TODAY = date(2026, 10, 23)
CLOCK = lambda: datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)  # noqa: E731


def settlements(product, price_of, first=2024, last=2027) -> pd.DataFrame:
    """Daily settlements of every contract delivering first..last, up to its expiry or yesterday; price_of(index, day) -> close."""
    rows = []
    for index in range(_index(1, first), _index(12, last) + 1):
        month, year = _month_year(index)
        expiry = contract_expiry(product, month, year)
        for day in pd.bdate_range(expiry - timedelta(days=200), min(expiry, TODAY - timedelta(days=1))):
            rows.append({"contract": api_symbol(product, month, year), "date": day, "oi": 1.0, "close": price_of(index, day), "volume": 1.0})
    return pd.DataFrame(rows)


class FakeStore:
    def __init__(self, frames):
        self._frames = frames
        self.version = "v1"
        self.loaded_history_days = 1500
        self.progress = OiProgress()

    def frame(self, product):
        return self._frames.get(product, pd.DataFrame(columns=["contract", "date", "oi", "close", "volume"]))

    def has_data(self):
        return any(not f.empty for f in self._frames.values())

    def contract_expiry_dates(self, product):
        frame = self.frame(product)
        return {} if frame.empty else frame.groupby("contract")["date"].max().to_dict()


def x_of(day, anchor, product="CL") -> int:
    return int(-trading_days_to(pd.DatetimeIndex([pd.Timestamp(day)]), anchor, PRODUCTS[product].calendar)[0])


def cl_store():
    return FakeStore({"CL": settlements("CL", lambda index, day: 100.0 + index - _index(1, 2024) + 0.01 * day.day)})


def service(store=None):
    return RolloverService(store or cl_store(), clock=CLOCK)


def test_a_generic_line_runs_from_before_the_front_expiry_to_the_next_expiry():
    result = service().view("CL", RolloverOptions(generic=2, expiries=1, comparisons=3, before_days=5))
    finished = [line for line in result.lines if not line.live]
    assert finished and len(result.lines) == 3
    line = finished[0]
    followed = datetime.strptime(line.label, "%b%y")  # generic 2 before the front expiry: the contract after the one expiring
    assert line.frame.index.min() == -5
    assert line.frame["date"].max().date() > line.anchor and line.frame.index.max() == line.bounds[-1] >= x_of(line.frame["date"].max(), line.anchor)  # a short gap is stretched to the longest
    assert line.frame["date"].max().date() == contract_expiry("CL", followed.month, followed.year)  # it runs to its own expiry


def test_the_line_follows_one_contract_so_its_price_is_continuous_across_the_expiry():
    result = service().view("CL", RolloverOptions(generic=2, comparisons=2, before_days=5))
    line = next(line for line in result.lines if not line.live)
    daily = line.frame["price"] - 0.01 * line.frame["date"].dt.day  # the synthetic price only moves with the day of the month
    assert daily.nunique() == 1


def test_zero_base_makes_every_line_pass_through_zero_on_the_expiry_day():
    result = service().view("CL", RolloverOptions(comparisons=4, zero_base=True))
    done = [line for line in result.lines if line.anchor <= TODAY]
    assert done and all(abs(line.frame["price"].get(0)) < 1e-9 for line in done if 0 in line.frame.index)
    assert all(line.zero is not None for line in result.lines)


def test_only_contracts_that_crossed_every_expiry_line_are_drawn():
    for expiries in (1, 2, 3):
        options = RolloverOptions(generic=4, expiries=expiries, comparisons=6, before_days=30)
        for zero in (False, True):
            result = service().view("CL", RolloverOptions(**{**options.__dict__, "zero_base": zero}))
            assert result.lines
            for line in result.lines:
                assert line.dates[-2] <= TODAY if len(line.dates) > 1 else True  # all but the last expiry are behind it
                assert line.anchor <= TODAY
            assert sum(line.live for line in result.lines) <= 1  # at most the one whose last expiry is still ahead


def test_more_expiries_extend_the_line_and_mark_the_expiries_it_passes():
    short = next(l for l in service().view("CL", RolloverOptions(generic=3, expiries=1, comparisons=3)).lines if not l.live)
    long = next(l for l in service().view("CL", RolloverOptions(generic=3, expiries=2, comparisons=3)).lines if not l.live)
    assert long.frame.index.max() > short.frame.index.max()
    assert len(long.marks) == 2 and len(short.marks) == 1


def test_the_line_stops_where_the_structure_itself_expires():
    line = next(l for l in service().view("CL", RolloverOptions(generic=2, expiries=4, comparisons=3)).lines if not l.live)
    expiry = max(line.frame["date"]).date()
    first_leg = _index(line.anchor.month, line.anchor.year) + 1  # the contract that was generic 2 expires one expiry after the anchor's
    assert expiry <= contract_expiry("CL", *_month_year(first_leg + 1)) and len(line.marks) <= 3


def test_a_spread_line_is_the_difference_of_its_legs():
    result = service().view("CL", RolloverOptions(family="spread", comparisons=2))
    line = next(l for l in result.lines if not l.live)
    assert "/" in line.label and (line.frame["price"] - 0.0).abs().max() < 1.5  # month-to-month steps of 1 plus the day-of-month wobble


def test_the_average_uses_finished_lines_only_and_needs_two():
    result = service().view("CL", RolloverOptions(comparisons=5))
    finished = [l for l in result.lines if not l.live]
    expected = pd.concat([l.prices for l in finished], axis=1).mean(axis=1)
    assert abs(result.average.loc[0] - expected.loc[0]) < 1e-9
    assert average_of(finished[:1]).empty


def test_fewer_expiries_than_asked_is_reported():
    result = service().view("CL", RolloverOptions(comparisons=24, generic=12))
    assert len(result.lines) < 24 and any("expiries have settlements" in note for note in result.notes)


def test_without_settlements_the_chart_says_it_is_downloading():
    result = RolloverService(FakeStore({}), clock=CLOCK).view("CL", RolloverOptions())
    assert not result.lines and "downloading" in result.notes[0]


# ----------------------------------------------------------------------
# CL-BZ
# ----------------------------------------------------------------------


def box_store():
    return FakeStore({
        "CL": settlements("CL", lambda index, day: 100.0 + index - _index(1, 2024)),
        "CLBZ": settlements("CLBZ", lambda index, day: 3.0 + 0.2 * (index - _index(1, 2024))),
    })


def box_pairs(line):
    return list(dict.fromkeys(line.frame["pair"]))


def test_the_clbz_line_goes_from_different_months_to_the_same_month_and_back():
    result = service(box_store()).view("CLBZ", RolloverOptions(comparisons=3))
    assert result.box and result.lines
    line = next(l for l in result.lines if not l.live)
    pairs = box_pairs(line)
    assert len(pairs) == 3
    first, middle, last = (p.replace("CL ", "").split(" − BZ ") for p in pairs)
    assert first[0] != first[1] and middle[0] == middle[1] and last[0] != last[1]
    assert middle[0] == last[0] and middle[1] == first[1]  # CL moves to BZ's month at its expiry, BZ moves on at its own
    assert line.frame["date"].min().day <= 5  # it opens on the first trading days of the month
    assert line.frame["date"].max().date() > line.anchor + timedelta(days=20)
    assert sorted(m.text.split()[0] for m in line.marks) == ["BZ", "CL"]  # the BZ expiry inside the line and the CL expiry that ends it


def test_the_clbz_line_is_the_front_box_priced_from_the_cl_spread_and_the_stored_box():
    line = next(l for l in service(box_store()).view("CLBZ", RolloverOptions(comparisons=3)).lines if not l.live)
    row = line.frame.iloc[0]  # CL month c - BZ month c+1 = [CL(c) - CL(c+1)] + box(c+1) = -1 + box(c+1)
    cl_month, bz_month = row["pair"].replace("CL ", "").split(" − BZ ")
    bz_index = _index(datetime.strptime(bz_month, "%b%y").month, datetime.strptime(bz_month, "%b%y").year)
    assert abs(row["price"] - (-1.0 + 3.0 + 0.2 * (bz_index - _index(1, 2024)))) < 1e-9


def test_the_clbz_zero_point_is_the_cl_expiry_and_more_expiries_run_further():
    one = next(l for l in service(box_store()).view("CLBZ", RolloverOptions(comparisons=3, expiries=1, zero_base=True)).lines if not l.live)
    two = next(l for l in service(box_store()).view("CLBZ", RolloverOptions(comparisons=3, expiries=2, zero_base=True)).lines if not l.live)
    assert abs(one.frame["price"].get(0)) < 1e-9 and two.frame.index.max() > one.frame.index.max()
    assert len(two.marks) > len(one.marks)


def test_each_expiry_gets_a_vertical_line_and_a_generic_cannot_pass_more_than_n_minus_1():
    three = service().view("CL", RolloverOptions(generic=4, expiries=3, comparisons=3))
    assert [g.text for g in three.guides] == ["Front expiry", "Expiry 2", "Expiry 3"] and three.guides[0].x == 0
    capped = service().view("CL", RolloverOptions(generic=3, expiries=5, comparisons=3))
    assert len(capped.guides) == 2 and any("at most 2 expiries" in n for n in capped.notes)
    assert len(service().view("CL", RolloverOptions(generic=3, expiries=1, comparisons=3)).guides) == 1


def test_every_line_has_its_expiries_on_the_same_days_and_shows_zero_on_each_vertical_line():
    result = service().view("CL", RolloverOptions(generic=4, expiries=3, comparisons=4, zero_base=True))
    xs = [g.x for g in result.guides]
    assert xs == sorted(xs) and len(set(xs)) == 3
    for line in result.lines:
        assert line.bounds[:3] == xs
        for x in xs:
            if x in line.frame.index:
                assert abs(line.frame["price"].at[x]) < 1e-9  # restarts from 0 on each expiry line


def test_a_shorter_gap_between_expiries_is_stretched_to_the_longest_one():
    def line(gap):
        frame = pd.DataFrame({"price": [float(i) for i in range(-3, gap + 1)], "date": pd.bdate_range("2026-01-01", periods=gap + 4), "pair": ""},
                             index=range(-3, gap + 1))
        return RolloverLine("a", date(2026, 1, 5), frame, bounds=[0, gap], names=["", "x"], kinds=["Front expiry", ""], vlines=[0])
    short, long = line(4), line(6)
    guides, moved = align([short, long])
    assert moved and short.bounds == [0, 6] and short.frame.index.max() == 6 and short.frame["price"].at[6] == 4.0
    assert guides[0].x == 0 and list(short.frame.index) == list(long.frame.index)


def test_the_live_line_is_white_and_the_average_is_not():
    figure = tab.build_rollover_figure(service().view("CL", RolloverOptions(comparisons=4)))
    live = next(t for t in figure.data if t.name and "(live)" in t.name)
    average = next(t for t in figure.data if t.name == "Average (finished)")
    assert live.line.color == "#ffffff" and average.line.color != "#ffffff"
    assert len(figure.layout.shapes) == 1


def test_clbz_has_a_vertical_line_on_every_cl_and_brn_expiry():
    result = service(box_store()).view("CLBZ", RolloverOptions(comparisons=4, expiries=2))
    assert [g.text for g in result.guides] == ["CL expiry", "BRN expiry", "CL expiry", "BRN expiry", "CL expiry"] and result.guides[0].x == 0
    zeroed = service(box_store()).view("CLBZ", RolloverOptions(comparisons=4, expiries=2, zero_base=True))
    for line in zeroed.lines:
        assert all(abs(line.frame["price"].at[g.x]) < 1e-9 for g in zeroed.guides[:-1] if g.x in line.frame.index)


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------


def test_the_page_parses_its_controls_and_rejects_bad_ones():
    options = callbacks.parse_options("fly", 3, 2, 8, 15, True)
    assert (options.generic, options.expiries, options.comparisons, options.before_days, options.zero_base, options.family) == (3, 2, 8, 15, True, "fly")
    for bad in ((0, 1, 6, 20), (2, 0, 6, 20), (2, 1, 99, 20), (2, 1, 6, 0), (2.5, 1, 6, 20), (None, 1, 6, 20)):
        try:
            callbacks.parse_options("outright", *bad, False)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_the_page_draws_lines_the_average_and_the_expiry_line():
    result = service().view("CL", RolloverOptions(comparisons=4))
    figure = tab.build_rollover_figure(result)
    names = [t.name for t in figure.data if t.name]
    assert any("(live)" in n for n in names) and "Average (finished)" in names and len(names) >= 5
    assert tab.build_rollover_figure(None).layout.annotations


def test_the_render_callback_uses_the_service_and_survives_a_bad_entry(monkeypatch):
    holder = Container(rollover=service())
    monkeypatch.setattr(callbacks, "container", holder)
    figure, status = callbacks.render(0, "CL", "outright", 2, 1, 4, 20, False)
    assert len(figure.data) >= 4
    figure, status = callbacks.render(0, "CL", "outright", 99, 1, 4, 20, False)
    assert not figure.data
    assert callbacks.controls_disabled("CLBZ") and not callbacks.controls_disabled("CL")


def test_the_page_is_routed_and_in_the_menu():
    from ui.callbacks.shell_callbacks import _ROUTES
    from ui.layouts.shell import NAV_ITEMS

    assert _ROUTES["/strategy-rollover"] == ("rollover_tab", "rollover_layout")
    assert any(path == "/strategy-rollover" for path, _ in NAV_ITEMS)
    assert tab.rollover_layout() is not None


# ----------------------------------------------------------------------
# BZ and WBS: a settlements-only store of their own, downloaded when shown
# ----------------------------------------------------------------------


class FakeApi:
    """daily_market_data for any requested contract: a settlement on every business day up to its expiry or yesterday."""

    def __init__(self):
        self.calls = 0

    def daily_market_data(self, qhcode, start, end, fields, limit, offset):
        self.calls += 1
        rows = []
        for code in qhcode.split(","):
            product = "BZ" if code.startswith("BZZ") else "WBS"
            month = "FGHJKMNQUVXZ".index(code[-3]) + 1
            year = 2000 + int(code[-2:])
            expiry = contract_expiry(product, month, year)
            for day in pd.bdate_range(max(pd.Timestamp(start), pd.Timestamp(expiry) - pd.Timedelta(days=200)), min(pd.Timestamp(expiry), pd.Timestamp(end))):
                rows.append({"qhcode": code, "datetime": day.isoformat(), "oi": 1, "close": 70.0 + month, "volume": 5})
        return {"results": rows, "count": len(rows)}


def extra_service(tmp_path, api):
    from core.oi_data import OiStore

    store = OiStore(tmp_path, clock=CLOCK, products=("BZ", "WBS"))
    return RolloverService(cl_store(), clock=CLOCK, extra=store, api=api, history_days=lambda: 400, background=False), store


def test_the_extra_products_have_their_own_contract_codes_and_expiries():
    assert api_symbol("BZ", 1, 2027) == "BZZF27" and api_symbol("WBS", 1, 2027) == "WTCLF27"
    assert contract_expiry("BZ", 1, 2027) == date(2026, 11, 30)  # last business day of the second month before delivery
    assert "BZ" not in PRODUCTS and "WBS" not in PRODUCTS  # they are not curve products


def test_showing_bz_downloads_its_settlements_and_draws_the_chart(tmp_path):
    api = FakeApi()
    service, store = extra_service(tmp_path, api)
    result = service.view("BZ", RolloverOptions(comparisons=3))
    assert api.calls and store.has_data() and len(result.lines) == 3
    assert not any(line.frame.empty for line in result.lines)
    before = api.calls
    service.view("BZ", RolloverOptions(comparisons=3, zero_base=True))
    assert api.calls == before  # downloaded once for the day


def test_a_failed_download_is_reported_and_retried_later(tmp_path):
    class Broken(FakeApi):
        def daily_market_data(self, *args, **kwargs):
            raise RuntimeError("token expired")

    service, store = extra_service(tmp_path, Broken())
    result = service.view("WBS", RolloverOptions())
    assert not result.lines and "token expired" in result.notes[0]
