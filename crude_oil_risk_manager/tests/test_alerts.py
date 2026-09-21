"""Tests for core.alerts.AlertManager."""

import requests

import pytest

from core.alerts import AlertManager
from core.models import Alert, AlertLevel
from db.repository import Repository

WEBHOOK = "https://example.invalid/webhook/secret-path"


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture
def manager(repo):
    return AlertManager(repo)


@pytest.fixture
def teams_manager(repo):
    repo.set_setting("teams_webhook_url", WEBHOOK)
    return AlertManager(repo)


def make_alert(**overrides) -> Alert:
    defaults = dict(level=AlertLevel.CRITICAL, title="Portfolio stop hit", body="P&L below stop.")
    defaults.update(overrides)
    return Alert(**defaults)


# ---------- send_alert ----------


def test_send_alert_saves_alert_to_repository(manager, repo):
    alert = manager.send_alert(AlertLevel.WARNING, "Loss limit", "Structure exceeded loss limit.")
    stored = repo.get_alert_history()
    assert [a.alert_id for a in stored] == [alert.alert_id]


def test_send_alert_returns_alert_with_correct_fields(manager):
    alert = manager.send_alert(AlertLevel.CRITICAL, "Stop hit", "Details here", structure_id="s-1")
    assert isinstance(alert, Alert)
    assert alert.level == AlertLevel.CRITICAL
    assert alert.title == "Stop hit"
    assert alert.body == "Details here"
    assert alert.structure_id == "s-1"
    assert alert.acknowledged is False
    assert alert.channels_sent == ["in_app"]
    assert alert.timestamp.tzinfo is not None


def test_send_alert_without_webhook_does_not_call_teams(manager, mocker):
    post = mocker.patch("core.alerts.requests.post")
    manager.send_alert(AlertLevel.INFO, "t", "b")
    post.assert_not_called()


def test_send_alert_with_webhook_records_teams_channel(teams_manager, repo, mocker):
    mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=200))
    alert = teams_manager.send_alert(AlertLevel.CRITICAL, "Stop hit", "body")
    assert alert.channels_sent == ["in_app", "teams"]
    assert repo.get_alert_history()[0].channels_sent == ["in_app", "teams"]


def test_send_alert_teams_failure_still_saves_and_does_not_raise(teams_manager, repo, mocker):
    mocker.patch("core.alerts.requests.post", side_effect=requests.exceptions.ConnectionError("boom"))
    alert = teams_manager.send_alert(AlertLevel.CRITICAL, "Stop hit", "body")
    assert alert.channels_sent == ["in_app"]
    assert len(repo.get_alert_history()) == 1


def test_send_alert_truncates_overlong_text_instead_of_raising(manager):
    alert = manager.send_alert(AlertLevel.INFO, "t" * 250, "b" * 900)
    assert len(alert.title) == 100
    assert len(alert.body) == 500


def test_send_alert_respects_explicit_teams_disable(teams_manager, repo, mocker):
    repo.set_setting("teams_alerts_enabled", False)
    post = mocker.patch("core.alerts.requests.post")
    teams_manager.send_alert(AlertLevel.INFO, "t", "b")
    post.assert_not_called()


def test_send_alert_survives_repository_failure(manager, mocker):
    mocker.patch.object(manager.repository, "save_alert", side_effect=RuntimeError("db down"))
    alert = manager.send_alert(AlertLevel.INFO, "t", "b")
    assert alert.channels_sent == []


# ---------- _send_teams_alert ----------


def test_send_teams_alert_posts_correct_payload(teams_manager, mocker):
    post = mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=200))
    alert = make_alert(structure_id="s-42")

    assert teams_manager._send_teams_alert(alert) is True

    post.assert_called_once()
    args, kwargs = post.call_args
    assert args[0] == WEBHOOK
    assert kwargs["timeout"] == 10
    payload = kwargs["json"]
    assert payload["@type"] == "MessageCard"
    assert payload["@context"] == "http://schema.org/extensions"
    assert payload["themeColor"] == "FF0000"
    assert payload["summary"] == "Portfolio stop hit"
    section = payload["sections"][0]
    assert section["activityTitle"] == "Portfolio stop hit"
    assert section["activitySubtitle"].startswith("Crude Oil Risk Manager — ")
    assert section["markdown"] is True
    assert section["facts"] == [
        {"name": "Level", "value": "critical"},
        {"name": "Details", "value": "P&L below stop."},
        {"name": "Structure", "value": "s-42"},
    ]


@pytest.mark.parametrize(
    "level,color",
    [(AlertLevel.INFO, "0076D7"), (AlertLevel.WARNING, "FF8C00"), (AlertLevel.CRITICAL, "FF0000")],
)
def test_send_teams_alert_theme_color_by_level(teams_manager, mocker, level, color):
    post = mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=200))
    teams_manager._send_teams_alert(make_alert(level=level))
    assert post.call_args.kwargs["json"]["themeColor"] == color


def test_send_teams_alert_uses_portfolio_when_no_structure(teams_manager, mocker):
    post = mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=200))
    teams_manager._send_teams_alert(make_alert(structure_id=None))
    facts = post.call_args.kwargs["json"]["sections"][0]["facts"]
    assert {"name": "Structure", "value": "Portfolio"} in facts


def test_send_teams_alert_returns_false_on_connection_error(teams_manager, mocker):
    mocker.patch("core.alerts.requests.post", side_effect=requests.exceptions.ConnectionError("boom"))
    assert teams_manager._send_teams_alert(make_alert()) is False


def test_send_teams_alert_returns_false_on_timeout(teams_manager, mocker):
    mocker.patch("core.alerts.requests.post", side_effect=requests.exceptions.Timeout("slow"))
    assert teams_manager._send_teams_alert(make_alert()) is False


def test_send_teams_alert_returns_false_on_non_200(teams_manager, mocker):
    mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=400))
    assert teams_manager._send_teams_alert(make_alert()) is False


def test_send_teams_alert_returns_false_without_webhook(manager, mocker):
    post = mocker.patch("core.alerts.requests.post")
    assert manager._send_teams_alert(make_alert()) is False
    post.assert_not_called()


def test_send_teams_alert_does_not_log_webhook_url(teams_manager, mocker, caplog):
    mocker.patch(
        "core.alerts.requests.post",
        side_effect=requests.exceptions.ConnectionError(f"failed for {WEBHOOK}"),
    )
    teams_manager._send_teams_alert(make_alert())
    assert "secret-path" not in caplog.text


# ---------- webhook lookup / cache ----------


def test_webhook_is_none_when_not_configured(manager):
    assert manager._get_teams_webhook() is None


def test_webhook_falls_back_to_configured_default(repo):
    manager = AlertManager(repo, fallback_webhook_url="https://example.invalid/env-hook")
    assert manager._get_teams_webhook() == "https://example.invalid/env-hook"


def test_webhook_cached_after_first_fetch_until_cleared(repo, mocker):
    repo.set_setting("teams_webhook_url", WEBHOOK)
    manager = AlertManager(repo)
    spy = mocker.spy(repo, "get_setting")

    assert manager._get_teams_webhook() == WEBHOOK
    assert manager._get_teams_webhook() == WEBHOOK
    assert spy.call_count == 1

    repo.set_setting("teams_webhook_url", "https://example.invalid/new")
    manager.clear_cache()
    assert manager._get_teams_webhook() == "https://example.invalid/new"


def test_webhook_configured_later_is_picked_up(repo):
    manager = AlertManager(repo)
    assert manager._get_teams_webhook() is None
    repo.set_setting("teams_webhook_url", WEBHOOK)
    assert manager._get_teams_webhook() == WEBHOOK


# ---------- send_test_alert / last_error ----------


def test_send_test_alert_posts_test_message_to_given_url_without_saving(manager, repo, mocker):
    post = mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=200))
    assert manager.send_test_alert("https://typed.example/hook") is True
    assert post.call_args.args[0] == "https://typed.example/hook"
    section = post.call_args.kwargs["json"]["sections"][0]
    assert section["activityTitle"] == "Test Alert from Crude Oil Risk Manager"
    assert {"name": "Details", "value": "This is a test message."} in section["facts"]
    assert repo.get_alert_history() == []


def test_send_test_alert_works_even_when_teams_alerts_disabled(teams_manager, repo, mocker):
    repo.set_setting("teams_alerts_enabled", False)
    post = mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=200))
    assert teams_manager.send_test_alert() is True
    post.assert_called_once()


def test_last_error_describes_each_failure_without_the_url(teams_manager, mocker):
    mocker.patch("core.alerts.requests.post", side_effect=requests.exceptions.Timeout(WEBHOOK))
    teams_manager._send_teams_alert(make_alert())
    assert teams_manager.last_error == "Timeout"

    mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=400))
    teams_manager._send_teams_alert(make_alert())
    assert teams_manager.last_error == "Teams rejected the message (HTTP 400)"
    assert "secret-path" not in teams_manager.last_error


def test_last_error_set_when_no_webhook_is_configured(manager):
    assert manager.send_test_alert() is False
    assert manager.last_error == "no webhook URL configured"


def test_last_error_cleared_on_success(teams_manager, mocker):
    mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=200))
    teams_manager.last_error = "stale"
    assert teams_manager.send_test_alert() is True
    assert teams_manager.last_error is None


# ---------- check_price_alerts ----------

from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType


def alert_structure(name="CL outright"):
    leg = Leg(
        contract=Contract(product="CL", contract_month=12, contract_year=2026, symbol="CLZ26",
                          multiplier=1000, tick_size=0.01, tick_value=10),
        ratio=1, lots=10, entry_price=75.0,
    )
    return Structure(name=name, structure_type=StructureType.OUTRIGHT, products=["CL"], legs=[leg], status=StructureStatus.OPEN)


def alert_trade(structure, direction="buy", stop=None, target=None):
    return Trade(
        structure_id=structure.structure_id, leg_id=structure.legs[0].leg_id, event_type=TradeEventType.TRADE,
        lots=10, price=75.0, direction=direction, stop_loss_price=stop, target_price=target,
    )


def run_price_alerts(manager, trade, structure, live):
    return manager.check_price_alerts({"CLZ26": live}, [trade], {structure.structure_id: structure})


def test_buy_below_stop_raises_critical_alert(manager, repo):
    s = alert_structure()
    (alert,) = run_price_alerts(manager, alert_trade(s, "buy", stop=73.0), s, 72.5)
    assert alert.level == AlertLevel.CRITICAL
    assert "Stop Loss Hit" in alert.title and "CL outright" in alert.title
    assert "72.5" in alert.body and "73" in alert.body and "buy" in alert.body
    assert alert.structure_id == s.structure_id
    assert [a.alert_id for a in repo.get_alert_history()] == [alert.alert_id]  # saved via send_alert


def test_sell_above_stop_raises_critical_alert(manager):
    s = alert_structure()
    (alert,) = run_price_alerts(manager, alert_trade(s, "sell", stop=77.0), s, 77.5)
    assert alert.level == AlertLevel.CRITICAL and "sell" in alert.body


def test_buy_above_target_raises_info_alert(manager):
    s = alert_structure()
    (alert,) = run_price_alerts(manager, alert_trade(s, "buy", target=80.0), s, 80.5)
    assert alert.level == AlertLevel.INFO and "Target Hit" in alert.title


def test_sell_below_target_raises_info_alert(manager):
    s = alert_structure()
    (alert,) = run_price_alerts(manager, alert_trade(s, "sell", target=70.0), s, 69.5)
    assert alert.level == AlertLevel.INFO


def test_levels_not_reached_raise_nothing(manager):
    s = alert_structure()
    assert run_price_alerts(manager, alert_trade(s, "buy", stop=73.0, target=80.0), s, 75.0) == []
    assert run_price_alerts(manager, alert_trade(s, "sell", stop=77.0, target=70.0), s, 75.0) == []


def test_already_sent_alert_is_not_duplicated_even_after_restart(manager, repo):
    s = alert_structure()
    trade = alert_trade(s, "buy", stop=73.0)
    assert len(run_price_alerts(manager, trade, s, 72.0)) == 1
    assert run_price_alerts(manager, trade, s, 71.0) == []
    assert repo.get_setting(f"alert_sent_{trade.trade_id}_stop") is True
    assert run_price_alerts(AlertManager(repo), trade, s, 70.0) == []  # a new manager = an app restart


def test_stop_and_target_are_deduplicated_independently(manager):
    s = alert_structure()
    stop_only = run_price_alerts(manager, alert_trade(s, "buy", stop=73.0), s, 72.0)
    assert len(stop_only) == 1
    trade = alert_trade(s, "buy", stop=73.0, target=80.0)
    both = manager.check_price_alerts({"CLZ26": 72.0}, [trade], {s.structure_id: s})
    assert [a.level for a in both] == [AlertLevel.CRITICAL]  # only the stop is crossed at 72


def test_no_stop_or_target_returns_empty_list(manager):
    s = alert_structure()
    assert run_price_alerts(manager, alert_trade(s, "buy"), s, 1.0) == []


def test_missing_price_or_structure_is_skipped(manager):
    s = alert_structure()
    trade = alert_trade(s, "buy", stop=73.0)
    assert manager.check_price_alerts({}, [trade], {s.structure_id: s}) == []
    assert manager.check_price_alerts({"CLZ26": 70.0}, [trade], {}) == []
