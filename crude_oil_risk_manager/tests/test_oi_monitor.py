"""Live OI monitor: deviations, sudden versus sustained, floors, the roll estimate, alerts and the page callbacks."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from dash import no_update

from core.models import AlertLevel
from core.oi_monitor import (
    STATUS_NORMAL, STATUS_SUDDEN, STATUS_SUSTAINED, MonitorEngine, MonitorParams, load_params, oi_price_read, save_params,
    validate_params,
)
from core.oi_research import month_oi_tables
from core.oi_service import KEY_ALERTS_SENT, OiService
from tests.test_oi import NOW, TODAY, synthetic_frame
from ui.callbacks import oi_callbacks as occ
from ui.container import Container
from ui.layouts import oi_monitor_view as view
from ui.layouts import oi_tab

HISTORY = 900  # the synthetic contracts have this many calendar days of history
LIVE = "CLF27"  # Jan27: about 70 days to expiry on TODAY, with four finished January contracts behind it


@pytest.fixture(scope="module")
def base_frame():
    return synthetic_frame("CL", 40)


def spiked(frame: pd.DataFrame, symbol: str, days: int, factor: float) -> pd.DataFrame:
    """Copy of the frame with the last `days` trading days of one contract scaled by `factor`."""
    out = frame.copy()
    dates = sorted(out.loc[out["contract"] == symbol, "date"])[-days:]
    mask = (out["contract"] == symbol) & out["date"].isin(dates)
    out.loc[mask, "oi"] *= factor
    return out


def engine(frame, **overrides) -> MonitorEngine:
    return MonitorEngine(frame, "CL", TODAY, HISTORY, MonitorParams(**overrides))


def row(result, symbol):
    return result.rows[result.rows["symbol"] == symbol].iloc[0]


# ----------------------------------------------------------------------
# Parameters and small helpers
# ----------------------------------------------------------------------


def test_parameters_default_validate_and_round_trip(tmp_path):
    from db.repository import Repository

    assert (MonitorParams().contracts, MonitorParams().sudden_days, MonitorParams().years) == (15, 5, 5)
    assert validate_params({"contracts": "20", "sudden_days": 3, "z_threshold": "3"}).contracts == 20
    assert validate_params({}) == MonitorParams()  # blanks fall back to the defaults
    for bad in ({"contracts": 0}, {"contracts": 99}, {"z_threshold": 0}, {"sudden_days": 1.5}, {"years": 1}, {"min_oi": -1}):
        with pytest.raises(ValueError):
            validate_params(bad)
    repo = Repository(str(tmp_path / "t.db"))
    assert load_params(repo) == MonitorParams()
    save_params(repo, MonitorParams(contracts=10, alerts_enabled=False))
    assert load_params(repo) == MonitorParams(contracts=10, alerts_enabled=False)


def test_oi_and_price_together_are_read_like_a_trader_would():
    assert oi_price_read(100, 1.0) == "new longs" and oi_price_read(100, -1.0) == "new shorts"
    assert oi_price_read(-100, 1.0) == "short covering" and oi_price_read(-100, -1.0) == "long liquidation"
    assert oi_price_read(0, 1.0) == "" and oi_price_read(None, 1.0) == "" and oi_price_read(100, float("nan")) == ""


# ----------------------------------------------------------------------
# The monitor
# ----------------------------------------------------------------------


def test_ordinary_contracts_are_not_flagged(base_frame):
    result = engine(base_frame).result()
    assert len(result.rows) == 15 and result.error is None
    assert (result.rows["status"] == STATUS_NORMAL).all()
    assert result.rows["label"].iloc[0] == "Nov26" and result.rows["dte"].is_monotonic_increasing
    assert result.data_date == pd.Timestamp("2026-10-12")  # yesterday's OI is the latest


def test_the_number_of_contracts_is_a_parameter(base_frame):
    assert len(engine(base_frame, contracts=4).result().rows) == 4


def test_a_jump_that_is_a_few_days_old_is_sudden_against_the_same_month(base_frame):
    result = engine(spiked(base_frame, LIVE, 3, 1.8)).result()
    r = row(result, LIVE)
    assert r["status"] == STATUS_SUDDEN and r["run_days"] == 3
    assert r["direction"] == "above" and r["b_z"] > 2.5 and r["b_flag"]
    assert r["since"] == (pd.Timestamp("2026-10-12") - pd.tseries.offsets.BDay(2)).date()
    others = result.rows[result.rows["symbol"] != LIVE]
    assert (others["status"] == STATUS_NORMAL).all()  # only the contract that moved is flagged


def test_the_same_deviation_held_for_long_is_sustained(base_frame):
    r = row(engine(spiked(base_frame, LIVE, 12, 1.8)).result(), LIVE)
    assert r["status"] == STATUS_SUSTAINED and r["run_days"] == 12


def test_the_sudden_window_is_a_parameter(base_frame):
    frame = spiked(base_frame, LIVE, 8, 1.8)
    assert row(engine(frame).result(), LIVE)["status"] == STATUS_SUSTAINED  # 8 days > 5
    assert row(engine(frame, sudden_days=10).result(), LIVE)["status"] == STATUS_SUDDEN


def test_a_drop_is_flagged_below(base_frame):
    r = row(engine(spiked(base_frame, LIVE, 2, 0.45)).result(), LIVE)
    assert r["status"] == STATUS_SUDDEN and r["direction"] == "below" and r["b_z"] < -2.5


def test_a_small_contract_is_ignored_however_large_the_percentage(base_frame):
    result = engine(spiked(base_frame, LIVE, 3, 1.8), min_oi=10_000_000).result()
    r = row(result, LIVE)
    assert r["status"] == STATUS_NORMAL and r["small"]


def test_a_move_smaller_than_the_floor_is_ignored(base_frame):
    result = engine(spiked(base_frame, LIVE, 3, 1.8), min_change=10_000_000).result()
    assert row(result, LIVE)["status"] == STATUS_NORMAL


def test_the_threshold_decides_what_counts(base_frame):
    frame = spiked(base_frame, LIVE, 3, 1.3)  # a 30% move is about 3 z against a 10% scale
    assert row(engine(frame, z_threshold=2.0).result(), LIVE)["status"] == STATUS_SUDDEN
    assert row(engine(frame, z_threshold=6.0).result(), LIVE)["status"] == STATUS_NORMAL


def test_a_contract_past_its_peak_is_judged_by_its_percent_of_peak(base_frame):
    result = engine(base_frame).result()
    front = row(result, "CLX26")  # a week from expiry: OI has fallen well below its peak so far
    assert front["measure"] == "% of peak" and front["pct_of_peak"] < 0.97
    building = row(result, LIVE)
    assert building["measure"] == "OI" and building["pct_of_peak"] > 0.97


def test_the_baseline_is_ignored_when_it_has_too_few_past_contracts_of_that_month(base_frame):
    r = row(engine(spiked(base_frame, LIVE, 3, 1.8), years=2).result(), LIVE)  # two Januaries: not trusted
    assert not r["b_reliable"] and not r["b_flag"] and r["status"] == STATUS_NORMAL


def test_only_contracts_of_the_same_delivery_month_make_the_baseline(base_frame):
    eng = engine(base_frame)
    r = row(eng.result(), LIVE)
    assert r["b_n"] == 4  # four Januaries on record
    assert all(label.startswith("Jan") for label in r["b_symbols"].split(", ")) and r["b_median"] > 0 and r["b_min"] <= r["b_median"] <= r["b_max"]
    for symbol in eng.live_symbols():  # every live contract is compared with its own month
        months = {eng.universe[s].month for s in eng.same_month_symbols(symbol)}
        assert months <= {eng.universe[symbol].month}
    assert not hasattr(eng, "baseline_symbols") and "recent_months" not in MonitorParams().__dict__
    assert "a_median" not in eng.result().rows.columns and "priority" not in eng.result().rows.columns


def test_the_latest_day_with_oi_is_used_when_the_newest_day_has_none(base_frame):
    frame = base_frame.copy()
    last = frame["date"].max()
    frame.loc[frame["date"] == last, "oi"] = np.nan  # ICE publishes OI a day late
    result = engine(frame).result()
    assert result.data_date == pd.Timestamp("2026-10-09") and not result.rows.empty
    assert np.isfinite(result.totals["total_oi"])


def test_changes_read_with_the_price(base_frame):
    r = row(engine(base_frame).result(), LIVE)
    assert np.isfinite(r["d1_oi"]) and np.isfinite(r["dn_oi"]) and np.isfinite(r["dn_pct"])
    assert r["read_1d"] in {"", "new longs", "new shorts", "short covering", "long liquidation"}


def test_yesterdays_roll_estimate_adds_up(base_frame):
    result = engine(base_frame).result()
    summary = result.roll_summary
    assert summary["date"] == date(2026, 10, 12) and summary["rolled"] <= min(summary["decrease"], summary["increase"]) + 1e-6
    assert result.roll["label"].tolist() == result.rows["label"].tolist()
    # a contract that fell split into what rolled and what exited (the rest of its fall)
    fell = result.roll[result.roll["change"] < 0]
    assert ((fell["rolled_out"] + fell["exited"]) - (-fell["change"])).abs().max() < 1e-6


def test_share_of_product_oi_is_a_fraction_that_sums_sensibly(base_frame):
    result = engine(base_frame).result()
    assert ((result.rows["share"] > 0) & (result.rows["share"] < 1)).all()


def test_an_empty_frame_gives_a_clear_message():
    result = MonitorEngine(pd.DataFrame(columns=["contract", "date", "oi", "close", "volume"]), "CL", TODAY, HISTORY, MonitorParams()).result()
    assert result.error == "No OI data downloaded yet" and result.rows.empty


def test_the_chart_data_has_the_same_month_baseline_and_the_live_path(base_frame):
    data = engine(base_frame).chart(LIVE)
    assert data["label"] == "Jan27" and len(data["dte"]) == len(data["oi"]) > 100 and "A" not in data
    d = data["B"]
    assert len(d["dte"]) == len(d["median"]) == len(d["low"]) == len(d["high"]) and (d["low"] <= d["median"] + 1e-9).all()
    assert len(data["B"]["labels"]) == 4 and len(data["B"]["lines"]) == 4
    assert engine(base_frame).chart("CLZ99") is None


# ----------------------------------------------------------------------
# Research tables by month
# ----------------------------------------------------------------------


def test_month_tables_have_a_row_per_delivery_month_with_median_and_sd(base_frame):
    from core.oi_research import build_research

    research = build_research(base_frame, "CL", TODAY)
    peak, at = research.month_peak, research.month_at
    assert peak["month"].tolist() == list(range(1, 13)) and "n" not in peak.columns
    one = research.contracts[research.contracts["month"] == 3]["peak_oi"]
    assert peak.set_index("month").loc[3, "median"] == pytest.approx(one.median())
    assert peak.set_index("month").loc[3, "sd"] == pytest.approx(one.std(ddof=1))
    assert {"median_30", "sd_30", "median_7", "sd_7"} <= set(at.columns)
    rows = oi_tab.month_peak_rows(research)
    assert rows[0]["month"] == "Jan" and rows[0]["median"] > 0
    text = oi_tab.month_at_rows(research)[0]
    assert "(" in text["d30"] and text["d30"].split(" ")[0].replace(",", "").isdigit()
    assert month_oi_tables(research.contracts.iloc[0:0])[0].empty


# ----------------------------------------------------------------------
# Service: caching and alerts
# ----------------------------------------------------------------------


class StubStore:
    def __init__(self, frames):
        self.frames, self.version, self.progress = frames, "v1", type("P", (), {"running": False, "done": 0, "total": 0, "message": "", "error": None, "finished_at": None})()
        self.updated_on, self.loaded_history_days = TODAY.isoformat(), HISTORY

    def frame(self, product):
        return self.frames.get(product, pd.DataFrame(columns=["contract", "date", "oi", "close", "volume"]))

    def has_data(self):
        return True


class FakeAlerts:
    def __init__(self):
        self.sent = []

    def send_alert(self, level, title, body, structure_id=None, max_body_length=500):
        self.sent.append((level, title, body))


@pytest.fixture
def service_with(tmp_path):
    from db.repository import Repository

    def make(frame):
        repo = Repository(str(tmp_path / "svc.db"))
        repo.set_setting("oi_history_days", HISTORY)
        alerts = FakeAlerts()
        svc = OiService(repo, None, StubStore({"CL": frame}), clock=lambda: NOW, background=False, alert_manager=alerts)
        svc.alerts = alerts
        return svc

    return make


def test_the_monitor_is_cached_per_data_version_and_parameters(service_with, base_frame):
    svc = service_with(base_frame)
    a, b = svc.monitor("CL", MonitorParams()), svc.monitor("CL", MonitorParams())
    assert a is b
    assert svc.monitor("CL", MonitorParams(contracts=5))[1].rows.shape[0] == 5
    svc.store.version = "v2"  # new data
    assert svc.monitor("CL", MonitorParams()) is not a


def test_a_sudden_deviation_alerts_once(service_with, base_frame):
    svc = service_with(spiked(base_frame, LIVE, 3, 1.8))
    assert svc.check_alerts() == 1
    level, title, body = svc.alerts.sent[0]
    assert level == AlertLevel.WARNING and "CL" in title and "Jan27" in title
    assert "above its usual" in body and "[" not in body  # no priority label in the alert
    assert svc.check_alerts() == 0 and len(svc.alerts.sent) == 1  # the same deviation does not alert again
    assert any(LIVE in key for key in svc._repository.get_setting(KEY_ALERTS_SENT))


def test_a_new_deviation_alerts_even_after_an_earlier_one(service_with, base_frame):
    svc = service_with(spiked(base_frame, LIVE, 3, 1.8))
    svc.check_alerts()
    svc.store.frames["CL"] = spiked(spiked(base_frame, LIVE, 3, 1.8), "CLH27", 2, 0.4)
    svc.store.version = "v2"
    assert svc.check_alerts() == 1 and "Mar27" in svc.alerts.sent[-1][1]


def test_sustained_untrusted_and_disabled_alerts_stay_quiet(service_with, base_frame):
    assert service_with(spiked(base_frame, LIVE, 12, 1.8)).check_alerts() == 0  # sustained: a lasting reason, not news
    svc = service_with(spiked(base_frame, LIVE, 3, 1.8))
    svc.save_monitor_params(MonitorParams(years=2))  # too few Januaries to trust: no alert
    assert svc.check_alerts() == 0
    svc = service_with(spiked(base_frame, LIVE, 3, 1.8))
    svc.save_monitor_params(MonitorParams(alerts_enabled=False))
    assert svc.check_alerts() == 0 and not svc.alerts.sent
    assert service_with(base_frame).check_alerts() == 0
    no_manager = service_with(base_frame)
    no_manager._alerts = None
    assert no_manager.check_alerts() == 0


def test_the_daily_check_runs_once_a_day_when_the_data_is_already_fresh(service_with, base_frame):
    svc = service_with(spiked(base_frame, LIVE, 3, 1.8))
    svc.ensure_fresh()  # data stored today, no check yet today: it checks
    assert len(svc.alerts.sent) == 1
    svc.ensure_fresh()
    assert len(svc.alerts.sent) == 1


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------


def values(**overrides):
    base = MonitorParams()
    raw = {field: getattr(base, field) for _, _, field, _, _ in view.PARAM_INPUTS}
    raw.update(overrides)
    return [raw[field] for _, _, field, _, _ in view.PARAM_INPUTS]


def test_the_monitor_callback_builds_the_table_and_picks_a_flagged_contract(service_with, base_frame, monkeypatch):
    svc = service_with(spiked(base_frame, LIVE, 3, 1.8))
    monkeypatch.setattr(occ, "container", Container(oi_service=svc))
    body, options, value, params = occ.render_monitor("v1", "CL", *values(), True, None)
    assert "oi-monitor-table" in str(body) and "Sudden deviations" in str(body)
    assert len(options) == 15 and value == LIVE and "SUDDEN" in next(o["label"] for o in options if o["value"] == LIVE)
    assert params["contracts"] == 15
    assert svc.monitor_params() == validate_params(params)  # remembered for the alert check
    assert occ.render_monitor("v1", "CL", *values(), True, "CLH27")[2] == "CLH27"  # the user's contract choice is kept


def test_a_bad_parameter_shows_a_message_and_leaves_the_rest(service_with, base_frame, monkeypatch):
    monkeypatch.setattr(occ, "container", Container(oi_service=service_with(base_frame)))
    body, options, value, params = occ.render_monitor("v1", "CL", *values(contracts=99), True, None)
    assert "Contracts to watch" in body.children and options is no_update and params is no_update
    assert occ.render_monitor(None, "CL", *values(), True, None)[0].children is None  # no data version yet: nothing


def test_the_chart_callback_draws_the_chosen_contract(service_with, base_frame, monkeypatch):
    svc = service_with(base_frame)
    monkeypatch.setattr(occ, "container", Container(oi_service=svc))
    figure = occ.render_chart(LIVE, {"contracts": 15})
    assert figure.layout.uirevision == LIVE and len(figure.data) >= 8
    assert occ.product_of("COZ26") == "BRN" and occ.product_of("CLZ26") == "CL" and occ.product_of(None) is None
    assert "Pick a contract" in occ.render_chart(None, None).layout.annotations[0].text


def test_switching_product_rebuilds_for_that_product(service_with, base_frame, monkeypatch):
    svc = service_with(base_frame)
    monkeypatch.setattr(occ, "container", Container(oi_service=svc))
    body, options, value, _ = occ.render_monitor("v1", "BRN", *values(), True, "CLX26")  # no BRN data in this stub
    assert "No OI data" in str(body) and options == [] and value is None


def test_the_page_layout_has_the_dropdown_the_controls_and_the_monitor_first(service_with, base_frame, monkeypatch):
    svc = service_with(base_frame)
    monkeypatch.setattr(oi_tab, "container", Container(oi_service=svc))
    layout = oi_tab.oi_layout()
    text = str(layout)
    for component_id in ("oi-product", "oi-contract", "oi-chart", "oi-monitor-wrap", "oi-research-wrap", "oi-p-alerts",
                         *(c for c, *_ in view.PARAM_INPUTS)):
        assert component_id in text, component_id
    product = next(x for x in _walk(layout) if getattr(x, "id", None) == "oi-product")
    assert [o["value"] for o in product.options] == ["CL", "BRN"] and product.value == "CL"
    svc.save_monitor_params(MonitorParams(contracts=9))
    contracts = next(x for x in _walk(oi_tab.oi_layout()) if getattr(x, "id", None) == "oi-p-contracts")
    assert contracts.value == 9  # the inputs start from what was last used


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and not isinstance(child, str):
            yield from _walk(child)


def test_the_callbacks_are_wired_to_the_product_dropdown():
    import dash

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    occ.register_oi_callbacks(app)
    wired = " ".join(app.callback_map)
    assert "oi-research-wrap" in wired and "oi-monitor-wrap" in wired and "oi-chart" in wired and "atr-wrap" in wired and "atr-chart" in wired
    assert "oi-compare-wrap" not in wired
    monitor = next(v for k, v in app.callback_map.items() if "oi-monitor-wrap" in k)
    assert "oi-product" in [i["id"] for i in monitor["inputs"]] and [s["id"] for s in monitor["state"]] == ["oi-contract"]
    research = next(v for k, v in app.callback_map.items() if "oi-research-wrap" in k)
    assert {"oi-product", "oi-tabs", "oi-month"} <= {i["id"] for i in research["inputs"]}
