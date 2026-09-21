"""Tests for ui.callbacks.settings_callbacks (plain functions, no browser)."""

import threading
from datetime import datetime, timezone

import pytest
import requests
from dash import no_update
from dash.exceptions import PreventUpdate

from adapters.base import APIServerError, AuthenticationError
from config.settings import settings
from core.alerts import AlertManager
from db.repository import Repository
from ui.callbacks import settings_callbacks as sc
from ui.container import Container
from ui.layouts import settings as settings_layout_module
from ui.layouts.settings import settings_layout


class StubLive:
    def __init__(self, error=None, candles=1):
        self.error = error
        self.candles = candles
        self.tokens_seen = []
        self.staleness = None

    def check_connection(self, token, symbol="CLZ26"):
        self.tokens_seen.append(token)
        if self.error:
            raise self.error
        return self.candles

    def set_staleness_threshold(self, seconds):
        self.staleness = seconds


class StubHistorical:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.runs = 0
        self.summary = {"last_sync_utc": None, "symbols_tracked": 0, "symbols_ok": 0, "symbols_error": 0}
        self.data = {"symbol_count": 0, "total_bytes": 0}

    def run_morning_sync(self, repository):
        self.runs += 1
        self.started.set()
        self.release.wait(timeout=5)
        return {}

    def get_sync_summary(self):
        return self.summary

    def get_local_data_summary(self):
        return self.data


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture
def env(repo, monkeypatch):
    fresh = Container(
        repository=repo,
        live_adapter=StubLive(),
        historical_adapter=StubHistorical(),
        alert_manager=AlertManager(repo),
    )
    monkeypatch.setattr(sc, "container", fresh)
    monkeypatch.setattr(settings_layout_module, "container", fresh)
    return fresh


# ---------- load_settings ----------


def test_load_settings_other_path_does_nothing(env):
    with pytest.raises(PreventUpdate):
        sc.load_settings("/")


def test_load_settings_returns_config_defaults_when_nothing_saved(env):
    result = sc.load_settings("/settings")
    assert len(result) == 11
    (token, pnl_stop, max_loss, staleness, margin, roll, confidence, window, webhook, enabled, status) = result
    assert token == "" and webhook == ""
    assert pnl_stop == settings.ALERT_PORTFOLIO_PNL_STOP
    assert max_loss == settings.ALERT_STRUCTURE_MAX_LOSS
    assert staleness == settings.LIVE_STALENESS_THRESHOLD_SECONDS
    assert margin == settings.DEFAULT_MARGIN_LIMIT
    assert roll == settings.ROLL_WARNING_DAYS_BEFORE_EXPIRY
    assert confidence == settings.DEFAULT_VAR_CONFIDENCE
    assert window == settings.DEFAULT_CORRELATION_WINDOW
    assert status == "❌ Not configured"


def test_load_settings_returns_saved_values(env, repo):
    repo.set_setting("api_access_token", "tok")
    repo.set_setting("alert_portfolio_pnl_stop", -1234.0)
    repo.set_setting("var_confidence", 0.99)
    repo.set_setting("teams_webhook_url", "https://x.example/hook")
    result = sc.load_settings("/settings")
    assert result[0] == "tok"
    assert result[1] == -1234.0
    assert result[6] == 0.99
    assert result[8] == "https://x.example/hook"
    assert result[10] == "✅ Token configured"


def test_load_settings_teams_switch_mirrors_effective_behaviour(env, repo):
    assert sc.load_settings("/settings")[9] is False  # nothing saved
    repo.set_setting("teams_webhook_url", "https://x.example/hook")
    assert sc.load_settings("/settings")[9] is True  # webhook saved, not disabled -> sends
    repo.set_setting("teams_alerts_enabled", False)
    assert sc.load_settings("/settings")[9] is False  # explicitly disabled


# ---------- token ----------


def test_toggle_token_visibility():
    assert sc.toggle_token_visibility(True) == "text"
    assert sc.toggle_token_visibility(False) == "password"


def test_save_token_stores_token_hides_banner_and_invalidates_price_cache(env, repo):
    env.live_cache.fetched_at = 123.0
    status, configured, banner = sc.save_token(1, "  Bearer abc123 ")
    assert status == "✅ Token saved successfully"
    assert configured is True
    assert banner == {"display": "none"}
    assert repo.get_setting("api_access_token") == "abc123"
    assert env.live_cache.fetched_at is None


@pytest.mark.parametrize("bad", [None, "", "   ", "two words"])
def test_save_token_rejects_invalid_and_stores_nothing(env, repo, bad):
    status, configured, banner = sc.save_token(1, bad)
    assert status.startswith("❌")
    assert configured is no_update and banner is no_update
    assert repo.get_setting("api_access_token", "") == ""


def test_save_token_does_not_log_the_token(env, caplog):
    with caplog.at_level("DEBUG"):
        sc.save_token(1, "very-secret-token")
    assert "very-secret-token" not in caplog.text


def test_test_connection_success_uses_typed_token_and_stores_nothing(env, repo):
    repo.set_setting("api_access_token", "saved-token")
    assert sc.test_connection(1, "typed-token") == "✅ Connection successful"
    assert env.live_adapter.tokens_seen == ["typed-token"]
    assert repo.get_setting("api_access_token") == "saved-token"


def test_test_connection_blank_field_falls_back_to_saved_token(env, repo):
    repo.set_setting("api_access_token", "saved-token")
    sc.test_connection(1, "")
    assert env.live_adapter.tokens_seen == ["saved-token"]


def test_test_connection_without_any_token(env):
    assert sc.test_connection(1, None) == "❌ Enter a token first"
    assert env.live_adapter.tokens_seen == []


def test_test_connection_reports_api_errors_without_the_token(env):
    env.live_adapter.error = AuthenticationError("Invalid or expired access token.")
    result = sc.test_connection(1, "typed-token")
    assert result == "❌ Connection failed: Invalid or expired access token."
    assert "typed-token" not in result

    env.live_adapter.error = APIServerError("Live API returned server error 503.")
    assert sc.test_connection(1, "typed-token").startswith("❌ Connection failed:")


def test_test_connection_notes_when_no_candles_returned(env):
    env.live_adapter.candles = 0
    assert "no data returned" in sc.test_connection(1, "typed-token")


def test_test_connection_rejects_malformed_typed_token(env):
    assert sc.test_connection(1, "has space").startswith("❌")


# ---------- thresholds / defaults ----------


def test_save_thresholds_valid_persists_everything_and_applies_staleness(env, repo):
    env.live_cache.fetched_at = 5.0
    button, feedback = sc.save_thresholds(1, -40000, -8000, 90, 2_000_000, 3)
    assert (button, feedback) == ("✅ Saved", "")
    assert repo.get_setting("alert_portfolio_pnl_stop") == -40000.0
    assert repo.get_setting("alert_structure_max_loss") == -8000.0
    assert repo.get_setting("staleness_threshold_seconds") == 90.0
    assert repo.get_setting("margin_limit") == 2_000_000.0
    assert repo.get_setting("roll_warning_days") == 3
    assert env.live_adapter.staleness == 90.0
    assert env.live_cache.fetched_at is None


def test_save_thresholds_invalid_saves_nothing(env, repo):
    button, feedback = sc.save_thresholds(1, 50000, -8000, 90, 2_000_000, 3)
    assert button == "❌ Not saved"
    assert "zero or negative" in feedback
    assert repo.get_all_settings() == {}
    assert env.live_adapter.staleness is None


def test_save_defaults(env, repo):
    assert sc.save_defaults(1, 0.99, 90) == ("✅ Saved", "")
    assert repo.get_setting("var_confidence") == 0.99
    assert repo.get_setting("correlation_window") == 90


def test_save_defaults_invalid(env, repo):
    button, feedback = sc.save_defaults(1, 0.5, 90)
    assert button == "❌ Not saved" and feedback.startswith("❌")
    assert repo.get_all_settings() == {}


def test_reset_label_returns_the_button_label():
    assert sc._reset_label("Save Thresholds")(1, 2, 3) == "Save Thresholds"


# ---------- Teams ----------


def test_save_teams_persists_and_clears_webhook_cache(env, repo):
    env.alert_manager._teams_webhook = "https://old.example/hook"
    button, status = sc.save_teams(1, "https://new.example/hook", True)
    assert (button, status) == ("✅ Saved", "✅ Teams settings saved")
    assert repo.get_setting("teams_webhook_url") == "https://new.example/hook"
    assert repo.get_setting("teams_alerts_enabled") is True
    assert env.alert_manager._get_teams_webhook() == "https://new.example/hook"


def test_save_teams_invalid_url_saves_nothing(env, repo):
    button, status = sc.save_teams(1, "http://insecure.example/hook", True)
    assert button == "❌ Not saved" and "https://" in status
    assert repo.get_all_settings() == {}


def test_save_teams_enabled_without_webhook_warns(env):
    _, status = sc.save_teams(1, "", True)
    assert "no Teams alerts will be sent" in status


def test_test_teams_success_posts_to_typed_url(env, mocker):
    post = mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=200))
    assert sc.test_teams(1, "https://typed.example/hook") == "✅ Test message sent"
    assert post.call_args.args[0] == "https://typed.example/hook"
    assert post.call_args.kwargs["json"]["summary"] == "Test Alert from Crude Oil Risk Manager"


def test_test_teams_does_not_save_an_alert(env, repo, mocker):
    mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=200))
    sc.test_teams(1, "https://typed.example/hook")
    assert repo.get_alert_history() == []


def test_test_teams_blank_field_uses_saved_webhook(env, repo, mocker):
    repo.set_setting("teams_webhook_url", "https://saved.example/hook")
    post = mocker.patch("core.alerts.requests.post", return_value=mocker.Mock(status_code=200))
    sc.test_teams(1, "")
    assert post.call_args.args[0] == "https://saved.example/hook"


def test_test_teams_failure_reports_reason(env, mocker):
    mocker.patch("core.alerts.requests.post", side_effect=requests.exceptions.ConnectionError("x"))
    assert sc.test_teams(1, "https://typed.example/hook") == "❌ Failed: ConnectionError"


def test_test_teams_without_any_webhook(env):
    assert sc.test_teams(1, "") == "❌ Failed: no webhook URL configured"


def test_test_teams_invalid_url(env):
    assert sc.test_teams(1, "nonsense").startswith("❌ Failed:")


# ---------- data management ----------


def test_run_sync_requires_a_token(env):
    assert sc.run_sync_now(1).startswith("❌")
    assert env.historical_adapter.runs == 0


def test_run_sync_starts_background_thread_once(env, repo):
    repo.set_setting("api_access_token", "tok")
    try:
        assert sc.run_sync_now(1) == "🔄 Sync started in background..."
        assert env.historical_adapter.started.wait(timeout=5)
        assert env.sync_thread.daemon is True
        assert sc.run_sync_now(2) == "🔄 A sync is already running..."
        assert env.historical_adapter.runs == 1
    finally:
        env.historical_adapter.release.set()
        env.sync_thread.join(timeout=5)


def test_run_sync_can_run_again_after_finishing(env, repo):
    repo.set_setting("api_access_token", "tok")
    env.historical_adapter.release.set()
    sc.run_sync_now(1)
    env.sync_thread.join(timeout=5)
    assert sc.run_sync_now(2) == "🔄 Sync started in background..."
    env.sync_thread.join(timeout=5)
    assert env.historical_adapter.runs == 2


def test_refresh_data_status_before_any_sync(env):
    sync_text, data_text = sc.refresh_data_status(0)
    assert sync_text == "No sync has run yet"
    assert data_text == "0 cached symbols · 0.0 KB"


def test_refresh_data_status_shows_last_sync_and_sizes(env):
    env.historical_adapter.summary = {
        "last_sync_utc": datetime(2026, 9, 21, 9, 30, tzinfo=timezone.utc),
        "symbols_tracked": 12, "symbols_ok": 11, "symbols_error": 1,
    }
    env.historical_adapter.data = {"symbol_count": 12, "total_bytes": 5 * 1024 * 1024}
    sync_text, data_text = sc.refresh_data_status(0)
    assert sync_text == "Last sync: 2026-09-21 09:30 UTC · 12 symbols (11 ok, 1 with errors)"
    assert data_text == "12 cached symbols · 5.0 MB"


def test_refresh_data_status_reports_running_sync(env):
    env.sync_thread = threading.Thread(target=env.historical_adapter.release.wait, args=(5,), daemon=True)
    env.sync_thread.start()
    try:
        assert sc.refresh_data_status(0)[0] == "🔄 Sync running..."
    finally:
        env.historical_adapter.release.set()
        env.sync_thread.join(timeout=5)


# ---------- layout ----------


def test_settings_layout_contains_all_component_ids(env):
    text = str(settings_layout())
    for component_id in [
        "settings-api-token", "settings-token-show", "settings-save-token", "settings-test-connection",
        "settings-token-status", "settings-portfolio-pnl-stop", "settings-structure-max-loss",
        "settings-staleness-threshold", "settings-margin-limit", "settings-roll-warning-days",
        "settings-save-thresholds", "settings-var-confidence", "settings-correlation-window",
        "settings-save-defaults", "settings-teams-webhook", "settings-teams-enabled",
        "settings-test-teams", "settings-teams-status", "settings-save-teams", "settings-sync-status",
        "settings-run-sync", "settings-data-summary",
    ]:
        assert component_id in text, component_id


def test_settings_layout_opens_api_section_until_token_is_configured(env, repo):
    assert settings_layout().children[1].active_item == ["api"]
    repo.set_setting("api_access_token", "tok")
    assert settings_layout().children[1].active_item == ["thresholds"]


def test_settings_password_fields_are_masked_by_default(env):
    text = str(settings_layout())
    assert text.count("type='password'") == 2  # token + Teams webhook
