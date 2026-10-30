"""OI page: settings, data store, roll attribution, research, service and page callbacks."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from dash import no_update

from adapters.base import APIError
from core.curve_calendar import PRODUCTS, product_of_symbol
from core import oi_data
from core.oi_data import OiStore, contract_universe
from core.oi_research import build_daily, build_research, last_month_table, month_contracts, open_table, threshold_table
from core.oi_roll import attribute_rolls, donor_flows
from core.oi_service import OiService
from core.oi_settings import (
    DEFAULT_OI_HISTORY_DAYS, KEY_OI_HISTORY_DAYS, load_history_days, validate_history_days,
)
from ui.callbacks import oi_callbacks as occ
from ui.callbacks import settings_callbacks as sc
from ui.container import Container
from ui.layouts import oi_tab

TODAY = date(2026, 10, 13)
NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def small_universe(monkeypatch):
    """A handful of contracts keeps the synthetic data small."""
    monkeypatch.setattr(oi_data, "FIRST_CONTRACT_YEAR", 2025)
    monkeypatch.setattr(oi_data, "MONTHS_AHEAD", 4)


def synthetic_oi(expiry: date, as_of: date, peak_dte: int = 40, seed: int = 0) -> pd.DataFrame:
    """Daily rows of a contract: OI builds to a peak `peak_dte` days before expiry, then drains."""
    end = min(pd.Timestamp(expiry), pd.Timestamp(as_of) - pd.Timedelta(days=1))
    days = pd.bdate_range(pd.Timestamp(expiry) - pd.Timedelta(days=oi_data_days()), end)
    dte = (pd.Timestamp(expiry) - days).days.to_numpy()
    rise = np.exp(-(((dte - peak_dte) / 180.0) ** 2))
    drain = np.where(dte < peak_dte, np.clip(dte / peak_dte, 0, 1) ** 1.5, 1.0)
    oi = 200_000 * rise * drain * (1 + np.random.default_rng(seed).normal(0, 0.01, len(days)))
    return pd.DataFrame({"date": days, "oi": oi, "close": 70.0, "volume": 1000.0})


def oi_data_days() -> int:
    return 900


class FakeOiApi:
    """Answers daily_market_data like the real API: comma qhcodes, start/end, limit/offset, count."""

    def __init__(self, today: date = TODAY, fail_after: int | None = None):
        self.today = today
        self.calls: list[dict] = []
        self.fail_after = fail_after
        self._cache: dict[str, pd.DataFrame] = {}

    def _series(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._cache:
            product = product_of_symbol(symbol)
            contract = next(c for c in contract_universe(product, self.today) if c.symbol == symbol)
            self._cache[symbol] = synthetic_oi(contract.expiry, self.today, seed=sum(map(ord, symbol)))
        return self._cache[symbol]

    def daily_market_data(self, **filters):
        self.calls.append(filters)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise APIError("boom")
        rows = []
        for code in filters["qhcode"].split(","):
            frame = self._series(code)
            frame = frame[(frame["date"] >= filters["start"]) & (frame["date"] <= filters["end"])]
            for r in frame.itertuples():
                rows.append({"datetime": r.date.isoformat(), "oi": r.oi, "close": r.close, "volume": r.volume, "qhcode": code})
        offset, limit = filters["offset"], filters["limit"]
        return {"count": len(rows), "next": None, "results": rows[offset:offset + limit]}


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------


def test_history_days_default_is_1500_and_validated():
    assert DEFAULT_OI_HISTORY_DAYS == 1500
    assert validate_history_days("1200") == 1200
    for bad in (99, 5001, 1500.5, None, "x"):
        with pytest.raises(ValueError):
            validate_history_days(bad)


def test_load_history_days_falls_back_and_clamps(tmp_path):
    from db.repository import Repository

    repo = Repository(str(tmp_path / "t.db"))
    assert load_history_days(repo) == 1500
    repo.set_setting(KEY_OI_HISTORY_DAYS, 900)
    assert load_history_days(repo) == 900
    repo.set_setting(KEY_OI_HISTORY_DAYS, "junk")
    assert load_history_days(repo) == 1500
    repo.set_setting(KEY_OI_HISTORY_DAYS, 99999)
    assert load_history_days(repo) == 5000


def test_settings_callbacks_load_and_save_oi_history(tmp_path, monkeypatch):
    from db.repository import Repository

    repo = Repository(str(tmp_path / "t.db"))
    monkeypatch.setattr(sc, "container", Container(repository=repo))
    assert sc.load_oi_settings("/settings") == (1500, 400)
    assert "✅" in sc.save_oi_settings(1, 1200) and repo.get_setting(KEY_OI_HISTORY_DAYS) == 1200
    assert "❌" in sc.save_oi_settings(1, 20) and repo.get_setting(KEY_OI_HISTORY_DAYS) == 1200
    assert "settings-oi-history-days" in str(__import__("ui.layouts.settings", fromlist=["x"]).settings_layout())


# ----------------------------------------------------------------------
# Universe and store
# ----------------------------------------------------------------------


def test_universe_runs_from_january_2019_to_36_months_ahead_for_both_products():
    cl, brn = contract_universe("CL", TODAY), contract_universe("BRN", TODAY)
    assert cl[0].symbol == "CLF19" and brn[0].symbol == "COF19"
    assert cl[-1].label == "Oct29" and len(cl) == len(brn) == 130
    assert [c.order for c in cl] == sorted(c.order for c in cl)


def test_full_download_pages_by_count_and_saves(tmp_path, small_universe, monkeypatch):
    monkeypatch.setattr(oi_data, "PAGE_SIZE", 500)  # force several pages per request
    api = FakeOiApi()
    store = OiStore(tmp_path, clock=lambda: NOW)
    report = store.update(api, 900, TODAY)
    assert len(api.calls) > report["requests"] > 1  # each chunk needed several pages
    assert report["rows"] > 1000 and store.progress.done == store.progress.total == report["requests"]
    assert all(call["fields"] == "oi,close,volume" for call in api.calls)
    frame = store.frame("CL")
    assert set(frame.columns) == {"contract", "date", "oi", "close", "volume"}
    assert frame.duplicated(["contract", "date"]).sum() == 0
    assert store.updated_on == "2026-10-13" and store.loaded_history_days == 900 and store.has_data()
    # the first day requested is `history_days` before the contract's rule expiry
    clf25 = next(c for c in contract_universe("CL", TODAY) if c.symbol == "CLF25")
    first = frame[frame["contract"] == "CLF25"]["date"].min()
    assert (pd.Timestamp(clf25.expiry) - first).days <= 905


def test_finished_contracts_are_never_requested_again_and_live_ones_are_topped_up(tmp_path, small_universe):
    api = FakeOiApi()
    store = OiStore(tmp_path, clock=lambda: NOW)
    store.update(api, 900, TODAY)
    api.calls.clear()
    tomorrow = TODAY + timedelta(days=1)
    store.update(api, 900, tomorrow)
    asked = {code for call in api.calls for code in call["qhcode"].split(",")}
    live = {c.symbol for p in PRODUCTS for c in contract_universe(p, tomorrow) if c.expiry >= tomorrow - timedelta(days=3)}
    assert asked <= live  # nothing that had already expired and been fetched
    assert all(call["start"] >= (TODAY - timedelta(days=oi_data.OVERLAP_DAYS)).isoformat() for call in api.calls)
    assert len(api.calls) <= 4  # one small top-up request per product (CL, BRN, gasoil and the CL-BZ box)


def test_a_longer_history_setting_downloads_the_extra_days(tmp_path, small_universe):
    api = FakeOiApi()
    store = OiStore(tmp_path, clock=lambda: NOW)
    store.update(api, 400, TODAY)
    before = store.frame("CL").groupby("contract")["date"].min()
    api.calls.clear()
    store.update(api, 900, TODAY)
    after = store.frame("CL").groupby("contract")["date"].min()
    assert api.calls and (after <= before).all() and (after < before).any()
    assert store.loaded_history_days == 900


def test_a_failed_download_keeps_what_was_saved_and_resumes(tmp_path, small_universe, monkeypatch):
    monkeypatch.setattr(oi_data, "FULL_CHUNK", 3)
    api = FakeOiApi(fail_after=2)
    store = OiStore(tmp_path, clock=lambda: NOW)
    with pytest.raises(APIError):
        store.update(api, 900, TODAY)
    saved = set(store._meta["contracts"])
    assert saved and store.updated_on is None  # progress kept, but the run did not finish
    api.fail_after = None
    api.calls.clear()
    store.update(api, 900, TODAY)
    asked = {code for call in api.calls for code in call["qhcode"].split(",")}
    finished_before = {s for s in saved if next(c for p in PRODUCTS for c in contract_universe(p, TODAY) if c.symbol == s).expiry < TODAY - timedelta(days=3)}
    assert not asked & finished_before  # what was already saved and finished is not downloaded again
    assert store.updated_on == "2026-10-13"
    # a second store object reads the same files
    again = OiStore(tmp_path, clock=lambda: NOW)
    assert again.updated_on == "2026-10-13" and len(again.frame("CL")) == len(store.frame("CL"))


def test_missing_oi_stays_missing_not_zero(tmp_path):
    rows = [{"datetime": "2019-11-27T00:00:00", "oi": 5, "close": 1.0, "volume": 2, "qhcode": "COF20"},
            {"datetime": "2019-11-29T00:00:00", "oi": None, "close": 1.0, "volume": 2, "qhcode": "COF20"}]
    frame = oi_data._clean(rows)
    assert frame["oi"].isna().tolist() == [False, True]


# ----------------------------------------------------------------------
# Roll attribution
# ----------------------------------------------------------------------


def test_rolled_oi_is_split_over_receivers_in_proportion_to_their_rises():
    result = attribute_rolls(pd.Series({"A": -100.0, "B": 60.0, "C": 20.0, "D": 20.0}))
    assert result.total_decrease == 100 and result.total_increase == 100
    assert result.rolled == pytest.approx(100.0)
    assert result.flows.loc["A"].to_dict() == pytest.approx({"B": 60.0, "C": 20.0, "D": 20.0})
    assert result.fresh.sum() == pytest.approx(0.0) and result.exited.sum() == pytest.approx(0.0)


def test_when_more_leaves_than_arrives_the_rest_exited():
    result = attribute_rolls(pd.Series({"A": -100.0, "B": -50.0, "C": 90.0}))
    assert result.rolled == pytest.approx(90.0)  # cannot roll into more than the receivers gained
    assert result.rolled_out["A"] == pytest.approx(60.0) and result.rolled_out["B"] == pytest.approx(30.0)
    assert result.exited["A"] == pytest.approx(40.0) and result.exited["B"] == pytest.approx(20.0)
    assert result.fresh["C"] == pytest.approx(0.0)


def test_when_more_arrives_than_leaves_the_rest_is_fresh_oi():
    result = attribute_rolls(pd.Series({"A": -30.0, "B": 60.0, "C": 40.0}))
    assert result.rolled == pytest.approx(30.0)
    assert result.rolled_in["B"] == pytest.approx(18.0) and result.fresh["B"] == pytest.approx(42.0)
    assert result.flows.loc["A", "B"] == pytest.approx(18.0) and result.flows.loc["A", "C"] == pytest.approx(12.0)


def test_money_is_conserved_and_missing_contracts_are_ignored():
    delta = pd.Series({"A": -80.0, "B": 10.0, "C": np.nan, "D": 25.0, "E": -5.0})
    result = attribute_rolls(delta)
    assert "C" not in result.rolled_out.index and "C" not in result.rolled_in.index
    assert result.rolled_out.sum() + result.exited.sum() == pytest.approx(85.0)
    assert result.rolled_in.sum() + result.fresh.sum() == pytest.approx(35.0)
    assert result.flows.to_numpy().sum() == pytest.approx(result.rolled)


def test_no_flow_without_both_sides():
    only_down = attribute_rolls(pd.Series({"A": -10.0, "B": -5.0}))
    assert only_down.rolled == 0 and only_down.exited.sum() == 15
    flat = attribute_rolls(pd.Series({"A": 0.0, "B": 0.0}))
    assert flat.rolled == 0 and flat.flows.empty


def test_donor_flows_for_one_contract():
    fall, rolled, destination = donor_flows(pd.Series({"A": -100.0, "B": 30.0, "C": 10.0}), "A")
    assert fall == 100 and rolled == pytest.approx(40.0)
    assert destination.to_dict() == pytest.approx({"B": 30.0, "C": 10.0})
    fall, rolled, destination = donor_flows(pd.Series({"A": 5.0, "B": -1.0}), "A")  # A did not fall
    assert (fall, rolled) == (0.0, 0.0) and destination.empty


# ----------------------------------------------------------------------
# Research
# ----------------------------------------------------------------------


def synthetic_frame(product: str = "CL", peak_dte: int = 40) -> pd.DataFrame:
    frames = []
    for contract in contract_universe(product, TODAY):
        if contract.expiry < date(2022, 1, 1):
            continue
        frame = synthetic_oi(contract.expiry, TODAY, peak_dte, seed=contract.order)
        frame["contract"] = contract.symbol
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[["contract", "date", "oi", "close", "volume"]]


@pytest.fixture(scope="module")
def research_cl():
    return build_research(synthetic_frame("CL", 40), "CL", TODAY)


def test_peak_is_found_with_its_days_before_expiry(research_cl):
    r = research_cl
    assert not r.empty and r.n_finished > 40 and r.error is None
    assert abs(r.contracts["peak_dte"].median() - 40) <= 10  # a flat synthetic peak plus 1% noise
    assert (r.contracts["peak_oi"] > 150_000).all()
    assert (r.contracts["peak_dte_trading"] < r.contracts["peak_dte"]).all()  # fewer trading days than calendar days
    assert r.contracts["expiry"].is_monotonic_increasing


def test_every_statistic_compares_contracts_of_one_delivery_month(research_cl):
    march = month_contracts(research_cl, 3)
    assert not march.empty and (march["month"] == 3).all() and len(march) < research_cl.n_finished / 6
    manual = march["peak_dte"].median()
    assert threshold_table(march).set_index("threshold").loc["100% of peak", "median"] == manual  # the peak is the 100% point
    july = month_contracts(research_cl, 7)
    assert (july["month"] == 7).all() and not set(july["contract"]) & set(march["contract"])  # each month is its own group
    # the by-month table is the March contracts' own statistics
    row = research_cl.month_peak.set_index("month").loc[3]
    assert row["median"] == pytest.approx(march["peak_oi"].median()) and row["sd"] == pytest.approx(march["peak_oi"].std(ddof=1))


def test_thresholds_are_reached_in_order_before_the_peak(research_cl):
    row = month_contracts(research_cl, 3).iloc[0]
    days = [row[f"t{t}"] for t in (10, 25, 50, 75, 90, 100)]
    assert days == sorted(days, reverse=True)  # 10% is reached earliest (most days before expiry)
    table = threshold_table(month_contracts(research_cl, 3))
    assert table["threshold"].tolist()[0] == "10% of peak" and set(table.columns) == {"threshold", "median", "sd", "min", "max"}


def test_open_share_falls_towards_expiry(research_cl):
    open_at = open_table(month_contracts(research_cl, 3)).set_index("days_before_expiry")["median"]
    assert open_at[60] > open_at[30] > open_at[14] > open_at[3]
    assert 0 < open_at[1] < 15


def test_the_percent_of_peak_curve_is_by_days_before_expiry(research_cl):
    matrix = research_cl.pct_matrix
    assert matrix.index.is_monotonic_decreasing and matrix.max().max() <= 1.0 + 1e-9
    march = month_contracts(research_cl, 3)
    assert set(march["contract"]) <= set(matrix.columns)
    median = matrix[list(march["contract"])].median(axis=1)
    assert median.loc[40] > 0.9 and median.loc[5] < median.loc[40]  # the peak sits near 40 days before expiry


def test_the_last_month_is_one_row_per_day(research_cl):
    table = last_month_table(research_cl, 3)
    assert table["dte"].tolist() == list(range(30, 0, -1))  # 30 rows, one per calendar day
    filled = table.dropna(subset=["open_median"])
    assert len(filled) >= 20 and (filled["open_min"] <= filled["open_median"] + 1e-9).all() and (filled["open_median"] <= filled["open_max"] + 1e-9).all()
    assert table.set_index("dte").loc[25, "open_median"] > table.set_index("dte").loc[3, "open_median"]  # it drains towards expiry
    for shares in filled[[f"share_{k}" for k in (1, 2, 3, 4)]].dropna().to_numpy():
        assert shares.sum() == pytest.approx(100.0, abs=0.5)


def test_a_days_row_uses_only_that_days_data_of_that_months_contracts(research_cl):
    march = set(month_contracts(research_cl, 3)["contract"])
    day = research_cl.daily[(research_cl.daily["contract"].isin(march)) & (research_cl.daily["dte"] == 12)]
    row = last_month_table(research_cl, 3).set_index("dte").loc[12]
    assert not day.empty
    assert row["open_median"] == pytest.approx(day["open_pct"].median()) and row["fall_median"] == pytest.approx(day["fall_pct"].median())
    assert row["share_1"] == pytest.approx(day["to_1"].sum() / day["rolled_pct"].sum() * 100) if day["rolled_pct"].sum() > 0 else True
    other = research_cl.daily[(~research_cl.daily["contract"].isin(march)) & (research_cl.daily["dte"] == 12)]
    assert len(other) > len(day)  # the other months' contracts exist and are left out


def test_a_contracts_fall_is_what_rolled_plus_what_exited(research_cl):
    daily = research_cl.daily
    assert (daily["fall_pct"] - daily["rolled_pct"] - daily["exited_pct"]).abs().max() < 1e-9
    assert (daily["rolled_pct"] <= daily["fall_pct"] + 1e-9).all() and set(daily["dte"]) == set(range(1, 31))


def test_no_pooled_statistics_and_no_contract_count_columns_remain(research_cl):
    import core.oi_research as research_module

    assert not hasattr(research_module, "compare_products") and not hasattr(research_cl, "by_year") and not hasattr(research_cl, "roll_buckets")
    assert "n" not in research_cl.month_peak.columns and "n" not in research_cl.month_at.columns
    assert not any(c in threshold_table(month_contracts(research_cl, 3)).columns for c in ("n",))


def test_research_reports_why_it_is_empty():
    empty = build_research(pd.DataFrame(columns=["contract", "date", "oi", "close", "volume"]), "CL", TODAY)
    assert empty.empty and "No OI data" in empty.error
    junk = synthetic_frame("CL")
    junk["oi"] = 10.0  # below the minimum peak
    assert "enough OI history" in build_research(junk, "CL", TODAY).error


# ----------------------------------------------------------------------
# Service and page
# ----------------------------------------------------------------------


@pytest.fixture
def service(tmp_path, small_universe):
    from db.repository import Repository

    repo = Repository(str(tmp_path / "t.db"))
    repo.set_setting(KEY_OI_HISTORY_DAYS, 900)
    clock = {"now": NOW}
    mono = {"t": 0.0}
    api = FakeOiApi()
    svc = OiService(repo, api, OiStore(tmp_path / "oi", clock=lambda: clock["now"]), clock=lambda: clock["now"],
                    background=False, monotonic=lambda: mono["t"])
    svc.clock, svc.mono, svc.fake_api = clock, mono, api
    return svc


def test_ensure_fresh_downloads_once_per_day(service):
    assert service.needs_update()
    service.ensure_fresh()
    assert service.store.updated_on == "2026-10-13" and not service.needs_update()
    calls = len(service.fake_api.calls)
    service.ensure_fresh()
    assert len(service.fake_api.calls) == calls
    service.clock["now"] += timedelta(days=1)
    assert service.needs_update()


def test_a_product_added_after_todays_update_is_downloaded_at_once(service):
    service.ensure_fresh()
    assert not service.needs_update() and service.store.has_every_product()
    (service.store._dir / "G.parquet").unlink()  # as if gasoil had been added to PRODUCTS after today's download
    assert service.needs_update() and not service.store.has_every_product()
    service.ensure_fresh()
    assert not service.needs_update() and (service.store._dir / "G.parquet").exists()


def test_a_longer_setting_makes_the_data_stale(service):
    service.ensure_fresh()
    service._repository.set_setting(KEY_OI_HISTORY_DAYS, 1200)
    assert service.needs_update()


def test_a_failed_update_is_reported_and_not_retried_at_once(service):
    service.fake_api.fail_after = 0
    service.ensure_fresh()
    status = service.status()
    assert "boom" in status["error"] and not status["running"]
    calls = len(service.fake_api.calls)
    service.ensure_fresh()
    assert len(service.fake_api.calls) == calls  # throttled
    service.mono["t"] += 601
    service.fake_api.fail_after = None
    service.ensure_fresh()
    assert service.status()["error"] is None and service.store.updated_on == "2026-10-13"


def test_refresh_now_is_refused_while_running(service):
    service.store.progress.running = True
    assert service.refresh_now() is False


def test_research_is_cached_until_the_data_changes(service, monkeypatch):
    service.ensure_fresh()
    built = []
    real = __import__("core.oi_service", fromlist=["x"]).build_research
    monkeypatch.setattr("core.oi_service.build_research", lambda *a, **k: built.append(a[1]) or real(*a, **k))
    first = service.research()
    second = service.research()
    assert all(first[p] is second[p] for p in ("CL", "BRN", "G")) and built == ["CL", "BRN", "G"]  # each built once


def test_page_callbacks(service, monkeypatch):
    monkeypatch.setattr(occ, "container", Container(oi_service=service))
    status, version = occ.update_status(0, None)  # first load: starts the daily download
    assert service.store.updated_on == "2026-10-13" and version == service.store.version and status is not None
    assert occ.render_research(version, "CL", "monitor", 3) is no_update  # not built while another tab is showing
    body = occ.render_research(version, "CL", "research", 3)
    assert "oi-month-peak-table" in str(body) or "No finished" in str(body) or "enough OI" in str(body)
    service.store.progress.running = True
    assert occ.update_status(1, None)[1] is no_update  # no re-render mid-download
    service.store.progress.running = False
    clicks = []
    monkeypatch.setattr(occ, "_triggered_id", lambda: "oi-update-btn")
    monkeypatch.setattr(service, "refresh_now", lambda: clicks.append(1))
    occ.update_status(2, 1)
    assert clicks == [1]  # the button starts a download
    assert occ.render_research(None, "CL", "research", 3).children is None
    monkeypatch.setattr(occ, "_triggered_id", lambda: None)
    monkeypatch.setattr(occ, "container", Container())
    assert "not running" in occ.update_status(0, None)[0].children


def test_layout_and_status_text_build():
    assert oi_tab.oi_layout() is not None
    idle = {"running": False, "done": 0, "total": 0, "message": "", "error": None, "updated_on": None, "history_days": None, "has_data": False, "version": ""}
    assert "No OI data downloaded yet" in str(oi_tab.status_text(idle))
    running = {**idle, "running": True, "done": 2, "total": 4, "message": "Downloading CL"}
    assert "50%" in str(oi_tab.status_text(running))
    assert "⚠️" in str(oi_tab.status_text({**idle, "error": "ApiError: x"}))
    nothing = oi_tab.research_body(build_research(pd.DataFrame(columns=oi_data.COLUMNS), "CL", TODAY), "CL", 3)
    assert "No OI data downloaded yet" in str(nothing)


def test_research_body_renders_the_same_month_tables_and_charts(research_cl):
    body = oi_tab.research_body(research_cl, "CL", 3)
    text = str(body)
    for component_id in ("oi-month-peak-table", "oi-month-at-table", "oi-contract-table", "oi-threshold-table", "oi-open-table", "oi-last-month-table"):
        assert component_id in text
    for removed in ("oi-compare", "oi-peak-table", "oi-year-table", "oi-roll-table"):
        assert removed not in text  # nothing that pools different months
    assert "Mar contracts" in text or "Mar" in text
    assert "No finished Aug contracts" not in text
    contracts = month_contracts(research_cl, 3)
    assert len(oi_tab.build_profile_figure(research_cl, 3).data) == len(contracts) + 1  # a line per contract and the median
    assert len(oi_tab.build_peak_timing_figure(research_cl, 3).data) == 1
    assert [r["contract"] for r in oi_tab.contract_rows(contracts)] == list(contracts["label"])
    assert len(oi_tab.last_month_rows(research_cl, 3)) == 30 and oi_tab.last_month_rows(research_cl, 3)[0]["dte"] == 30
    assert "Contracts" not in [c["name"] for c in oi_tab.MONTH_PEAK_COLUMNS + oi_tab.month_at_columns() + oi_tab.THRESHOLD_COLUMNS + oi_tab.OPEN_COLUMNS]
    assert "oi-month" in str(oi_tab.oi_layout())  # the month selector


# ----------------------------------------------------------------------
# No flicker: the research is rebuilt only when the data changes
# ----------------------------------------------------------------------


def test_the_version_is_written_only_when_it_changes(service, monkeypatch):
    monkeypatch.setattr(occ, "container", Container(oi_service=service))
    monkeypatch.setattr(occ, "_triggered_id", lambda: None)
    _, version = occ.update_status(0, None, None)  # first tick: downloads, then reports the new version
    assert version != no_update and version == service.store.version
    for tick in range(1, 6):  # every later tick with the same data must leave the research alone
        status, again = occ.update_status(tick, None, version)
        assert again is no_update and status is not None
    service.clock["now"] += timedelta(days=1)  # a new day: the daily top-up writes new data
    _, changed = occ.update_status(9, None, version)
    assert changed == service.store.version and changed != version
    _, after = occ.update_status(10, None, changed)
    assert after is no_update


def test_the_page_wires_the_current_version_as_state():
    import dash
    from ui.callbacks.oi_callbacks import register_oi_callbacks

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    register_oi_callbacks(app)
    callback = next(v for k, v in app.callback_map.items() if "oi-version" in k and "oi-status" in k)
    assert [s["id"] for s in callback["state"]] == ["oi-version"]


def test_every_tab_of_the_page_can_be_opened():
    layout = oi_tab.oi_layout()
    tabs = next(c for c in layout.children if getattr(c, "active_tab", None) == "monitor")  # the live monitor is the main tab
    assert [tab.label for tab in tabs.children] == ["Live monitor", "Research", "ATR"]
    assert not any(getattr(tab, "disabled", False) for tab in tabs.children)


def test_research_is_built_for_the_product_asked_for_only(service, monkeypatch):
    service.ensure_fresh()
    built = []
    real = __import__("core.oi_service", fromlist=["x"]).build_research
    monkeypatch.setattr("core.oi_service.build_research", lambda *a, **k: built.append(a[1]) or real(*a, **k))
    monkeypatch.setattr(occ, "container", Container(oi_service=service))
    occ.render_research(service.store.version, "BRN", "research", 3)
    assert built == ["BRN"]  # CL is not built until something asks for it
    occ.render_research(service.store.version, "BRN", "research", 5)  # another month of the same product: no rebuild
    assert built == ["BRN"]
    service.research()
    assert built == ["BRN", "CL", "G"]
