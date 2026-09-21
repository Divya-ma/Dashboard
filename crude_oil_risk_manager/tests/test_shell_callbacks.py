"""Tests for ui.callbacks.shell_callbacks (called as plain functions, no browser)."""

import json

import pytest
from dash import no_update

from adapters.base import APIServerError, AuthenticationError, LiveDataAdapter
from adapters.mock.mock_live import MockLiveAdapter
from core.alerts import AlertManager
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from db.repository import Repository
from ui.callbacks import shell_callbacks as cb
from ui.container import Container
from ui.layouts.shell import COLORS


class CountingAdapter(MockLiveAdapter):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_live_prices(self, symbols):
        self.calls += 1
        return super().get_live_prices(symbols)


class FailingAdapter(LiveDataAdapter):
    def __init__(self, error):
        self.error = error

    def get_access_token(self):
        return "token"

    def get_live_prices(self, symbols):
        raise self.error


def make_contract(symbol="CLZ26") -> Contract:
    return Contract(
        product="CL", contract_month=12, contract_year=2026, symbol=symbol,
        multiplier=1000.0, tick_size=0.01, tick_value=10.0,
    )


def open_structure(entry_price=75.0, lots=10.0) -> Structure:
    leg = Leg(contract=make_contract(), ratio=1, lots=lots, entry_price=entry_price)
    return Structure(
        name="CLZ26 outright",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[leg],
        status=StructureStatus.OPEN,
    )


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture
def env(repo, monkeypatch):
    """A fresh container wired into the callbacks module with a counting mock adapter."""
    fresh = Container(
        repository=repo,
        live_adapter=CountingAdapter(),
        alert_manager=AlertManager(repo),
    )
    monkeypatch.setattr(cb, "container", fresh)
    return fresh


@pytest.fixture
def with_token(repo):
    repo.set_setting("api_access_token", "test-token")


def pnl_payload(total, **extra):
    return {"total_pnl": total, "has_missing_prices": False, **extra}


# ---------- routing ----------


def test_render_page_unbuilt_tabs_show_placeholder():
    for path in ["/correlation", "/exposure", "/var-scenario", "/trade-analyzer", "/archive"]:
        page, _ = cb.render_page(path, True)
        assert page.children == "🚧 This tab is under construction", path
        assert page.style["color"] == COLORS["TEXT_PRIMARY"]


def test_render_page_home_and_settings_use_real_layouts(env):
    home, _ = cb.render_page("/", True)
    assert "home-structures-table" in str(home)
    settings_page, _ = cb.render_page("/settings", True)
    assert "settings-api-token" in str(settings_page)


def test_render_page_structure_detail_path_routes_to_structures_page():
    page, _ = cb.render_page("/structures/abc-123", True)
    assert "structures-active-grid" in str(page)


def test_render_page_unknown_path_is_404():
    page, _ = cb.render_page("/nope", True)
    assert "404" in page.children


def test_render_page_none_and_trailing_slash_route_correctly():
    assert "home-structures-table" in str(cb.render_page(None, True)[0])
    assert "structures-active-grid" in str(cb.render_page("/structures/", True)[0])


def test_render_page_banner_visibility_follows_token_state():
    assert cb.render_page("/", True)[1] == {"display": "none"}
    assert cb.render_page("/", False)[1] == {"display": "block"}


def test_resolve_layout_prefers_real_module_when_it_exists(monkeypatch):
    import types

    fake = types.ModuleType("ui.layouts.structures")
    fake.structures_layout = lambda: "real layout"
    monkeypatch.setitem(__import__("sys").modules, "ui.layouts.structures", fake)
    assert cb._resolve_layout("structures", "structures_layout")() == "real layout"


# ---------- countdown ----------


@pytest.mark.parametrize(
    "ticks,expected",
    [(0, "Next refresh in: 60s"), (1, "Next refresh in: 59s"), (59, "Next refresh in: 1s"), (60, "Next refresh in: 60s")],
)
def test_update_countdown(ticks, expected):
    assert cb.update_countdown(ticks, 0) == expected


# ---------- fetch_live_prices ----------


def test_fetch_without_token_returns_empty_and_false(env):
    prices, last, indicator, token_ok = cb.fetch_live_prices(0)
    assert prices == {}
    assert last == "Last: --:--:--"
    assert token_ok is False
    assert env.live_adapter.calls == 0
    assert "Data Stale" in str(indicator)


def test_fetch_with_token_but_no_contracts_returns_empty(env, with_token):
    prices, last, _, token_ok = cb.fetch_live_prices(0)
    assert prices == {}
    assert token_ok is True
    assert env.live_adapter.calls == 0


def test_fetch_returns_serializable_prices(env, repo, with_token):
    repo.save_contract(make_contract("CLZ26"))
    prices, last, indicator, token_ok = cb.fetch_live_prices(0)

    assert token_ok is True
    assert set(prices) == {"CLZ26"}
    assert set(prices["CLZ26"]) == {"price", "timestamp_iso", "is_stale", "open", "high", "low", "volume"}
    json.dumps(prices)
    assert last.startswith("Last: ") and "--" not in last
    assert "Data Live" in str(indicator)


def test_fetch_skips_unrecognized_symbols_instead_of_failing_the_batch(env, repo, with_token):
    repo.save_contract(make_contract("CLZ26"))
    repo.save_contract(
        Contract(product="CL", contract_month=1, contract_year=2027, symbol="ZZZF27",
                 multiplier=1000.0, tick_size=0.01, tick_value=10.0)
    )
    prices, *_ = cb.fetch_live_prices(0)
    assert set(prices) == {"CLZ26"}


def test_second_fetch_within_ttl_uses_cache_without_api_call(env, repo, with_token):
    repo.save_contract(make_contract("CLZ26"))
    first, *_ = cb.fetch_live_prices(0)
    second, *_ = cb.fetch_live_prices(1)
    assert env.live_adapter.calls == 1
    assert second == first


def test_fetch_saves_pnl_record_for_open_structure(env, repo, with_token):
    structure = open_structure()
    repo.save_structure(structure)
    cb.fetch_live_prices(0)

    record = repo.get_latest_pnl(structure.structure_id)
    assert record is not None
    assert "CLZ26" in record.last_price_used


def test_fetch_api_error_returns_last_known_prices_flagged_stale(env, repo, with_token):
    repo.save_contract(make_contract("CLZ26"))
    good, *_ = cb.fetch_live_prices(0)
    env.live_cache.fetched_at = None  # force the next poll to hit the (now failing) API
    env.live_adapter = FailingAdapter(APIServerError("boom"))

    prices, last, indicator, token_ok = cb.fetch_live_prices(1)

    assert token_ok is True
    assert prices["CLZ26"]["price"] == good["CLZ26"]["price"]
    assert prices["CLZ26"]["is_stale"] is True
    assert "Data Stale" in str(indicator)


def test_fetch_authentication_error_marks_token_unconfigured(env, repo, with_token):
    repo.save_contract(make_contract("CLZ26"))
    env.live_adapter = FailingAdapter(AuthenticationError("Invalid or expired access token."))

    prices, _, indicator, token_ok = cb.fetch_live_prices(0)

    assert token_ok is False
    assert "Data Stale" in str(indicator)


def test_fetch_never_logs_the_token(env, repo, with_token, caplog):
    repo.save_contract(make_contract("CLZ26"))
    env.live_adapter = FailingAdapter(APIServerError("boom"))
    with caplog.at_level("DEBUG"):
        cb.fetch_live_prices(0)
    assert "test-token" not in caplog.text


# ---------- refresh_portfolio_pnl ----------


def test_refresh_portfolio_pnl_is_json_safe_and_uses_cached_prices(env, repo):
    repo.save_structure(open_structure(entry_price=75.0, lots=10.0))
    live = {"CLZ26": {"price": 76.0, "is_stale": False}}

    result = cb.refresh_portfolio_pnl(0, live)

    json.dumps(result)
    assert result["total_pnl"] == pytest.approx(10_000.0)
    assert result["has_missing_prices"] is False
    assert result["open_structure_count"] == 1


def test_refresh_portfolio_pnl_flags_missing_prices(env, repo):
    repo.save_structure(open_structure())
    result = cb.refresh_portfolio_pnl(0, {})
    assert result["has_missing_prices"] is True


# ---------- check_alerts ----------


def test_check_alerts_no_data_returns_empty(env):
    assert cb.check_alerts(0, {}) == []


def test_check_alerts_above_threshold_returns_empty(env):
    assert cb.check_alerts(0, pnl_payload(-1000.0)) == []


def test_check_alerts_breach_creates_alert_and_toast(env, repo):
    result = cb.check_alerts(0, pnl_payload(-60_000.0))

    assert len(result) == 1
    assert result[0].header == "Portfolio P&L stop breached"
    stored = repo.get_unacknowledged_alerts()
    assert len(stored) == 1
    assert stored[0].level.value == "critical"
    assert "-60,000" in stored[0].body


def test_check_alerts_breach_alerts_only_once_until_recovery(env, repo):
    breach = pnl_payload(-60_000.0)
    cb.check_alerts(0, breach)
    assert cb.check_alerts(1, breach) is no_update
    assert cb.check_alerts(2, breach) is no_update
    assert len(repo.get_alert_history()) == 1

    assert cb.check_alerts(3, pnl_payload(-1000.0)) == []  # recovery clears and re-arms
    cb.check_alerts(4, breach)
    assert len(repo.get_alert_history()) == 2


def test_check_alerts_ignores_pnl_built_from_missing_prices(env, repo):
    result = cb.check_alerts(0, {"total_pnl": -999_999.0, "has_missing_prices": True})
    assert result is no_update
    assert repo.get_alert_history() == []


def test_check_alerts_uses_threshold_from_settings_table(env, repo):
    repo.set_setting("alert_portfolio_pnl_stop", -500.0)
    assert len(cb.check_alerts(0, pnl_payload(-1000.0))) == 1


# ---------- enriched portfolio store (used by the Home tab) ----------


def test_refresh_portfolio_pnl_adds_structure_details_for_home_tab(env, repo):
    structure = open_structure(entry_price=75.0, lots=10.0)
    repo.save_structure(structure)
    live = {"CLZ26": {"price": 76.0, "is_stale": True}}

    result = cb.refresh_portfolio_pnl(0, live)

    info = result["per_structure"][structure.structure_id]
    assert info["name"] == "CLZ26 outright"
    assert info["products"] == ["CL"]
    assert info["status"] == "open"
    assert result["stale_symbols_in_use"] == ["CLZ26"]
    assert result["missing_price_symbols"] == []
    json.dumps(result)


def test_refresh_portfolio_pnl_lists_missing_price_symbols(env, repo):
    repo.save_structure(open_structure())
    result = cb.refresh_portfolio_pnl(0, {})
    assert result["missing_price_symbols"] == ["CLZ26"]


def test_todays_pnl_is_change_since_first_snapshot_today(env, repo):
    from datetime import datetime, timezone

    from core.models import PnLRecord

    structure = open_structure(entry_price=75.0, lots=10.0)
    repo.save_structure(structure)
    repo.save_pnl_record(
        PnLRecord(
            structure_id=structure.structure_id, unrealized_pnl=2_000.0, realized_pnl=0.0,
            total_pnl=2_000.0, timestamp=datetime.now(timezone.utc).replace(hour=0, minute=0, second=1),
        )
    )
    # live PnL is (76 - 75) * 10 * 1000 = 10,000, so today's change is 8,000
    result = cb.refresh_portfolio_pnl(0, {"CLZ26": {"price": 76.0, "is_stale": False}})
    assert result["todays_pnl"] == pytest.approx(8_000.0)


def test_todays_pnl_is_zero_without_a_snapshot_today(env, repo):
    repo.save_structure(open_structure())
    result = cb.refresh_portfolio_pnl(0, {"CLZ26": {"price": 76.0, "is_stale": False}})
    assert result["todays_pnl"] == 0.0
