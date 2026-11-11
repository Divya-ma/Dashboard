"""Logins, the health check, the API probe, the container settings and the dependency pins."""

import re
import urllib.error
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.access import AccessConfigError, is_loopback, parse_users
from core.backup import BackupScheduler
from core.health import FAIL, INFO, OK, WARN, collect_health, overall
from db.repository import Repository
from tools.probe_api_codes import candidate_codes, probe
from ui.container import LivePriceCache

ROOT = Path(__file__).resolve().parent.parent


# ---------------- logins ----------------


def test_no_login_configured_is_an_empty_list():
    assert parse_users() == {}


def test_one_login_and_a_list_are_combined_and_whitespace_is_ignored():
    users = parse_users(" alice:pw1 , bob:p:w2 ;\ncara:pw3", "admin", "secret")
    assert users == {"admin": "secret", "alice": "pw1", "bob": "p:w2", "cara": "pw3"}  # a colon after the first belongs to the password


@pytest.mark.parametrize("users, username, password, message", [
    ("alice", "", "", "entry 1 must look like"),
    ("alice:pw,:pw", "", "", "entry 2 must look like"),
    ("alice:", "", "", "entry 1 must look like"),
    ("alice:a,alice:b", "", "", "more than once"),
    ("", "admin", "", "set together"),
    ("", "", "secret", "set together"),
    ("admin:x", "admin", "y", "more than once"),
])
def test_a_malformed_login_list_is_refused_and_the_message_never_holds_a_password(users, username, password, message):
    with pytest.raises(AccessConfigError, match=message) as error:
        parse_users(users, username, password)
    assert "secret" not in str(error.value) and "pw1" not in str(error.value)


def test_loopback_hosts():
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and not is_loopback("0.0.0.0")


def test_the_default_listening_address_is_this_pc_only():
    from config.settings import Settings

    assert Settings(_env_file=None).APP_HOST == "127.0.0.1"


# ---------------- health ----------------


class Service:
    def __init__(self, **status):
        self._status = status

    def status(self):
        return self._status


def parts_for(tmp_db_path, tmp_path, host="127.0.0.1", users=False, **overrides):
    repo = Repository(str(tmp_db_path))
    config = SimpleNamespace(APP_HOST=host, APP_PORT=8050, AUTH_USERNAME="a" if users else "", AUTH_PASSWORD="b" if users else "", AUTH_USERS="")
    values = dict(repository=repo, backup=BackupScheduler(tmp_db_path, tmp_path / "bk"), live_cache=LivePriceCache(), qh_api=None,
                  oi_service=None, churn_service=None, curve_service=None, app_settings=config)
    values.update(overrides)
    return SimpleNamespace(**values)


def by_name(checks):
    return {(c.area, c.name): c for c in checks}


def test_a_fresh_install_is_working_but_needs_a_look(tmp_db_path, tmp_path):
    checks = by_name(collect_health(parts_for(tmp_db_path, tmp_path)))
    assert checks[("Access", "Login")].status == OK  # no login, but only this PC can reach it
    assert checks[("Database", "Free disk space")].status == OK
    assert checks[("Backup", "Newest backup")].status == WARN and checks[("API", "Token")].status == WARN
    assert overall(list(checks.values())) == WARN


def test_a_login_less_dashboard_open_to_the_network_is_a_failure(tmp_db_path, tmp_path):
    assert by_name(collect_health(parts_for(tmp_db_path, tmp_path, host="0.0.0.0")))[("Access", "Login")].status == FAIL
    assert by_name(collect_health(parts_for(tmp_db_path, tmp_path, host="0.0.0.0", users=True)))[("Access", "Login")].status == OK


def test_a_failed_backup_a_stopped_download_and_stale_data_are_reported(tmp_db_path, tmp_path):
    backup = BackupScheduler(tmp_path / "missing.db", tmp_path / "bk")
    backup.run_once()
    old = Service(running=False, has_data=True, updated_on="2026-10-01", error=None)
    broken = Service(running=False, has_data=False, updated_on=None, error="HTTP 500")
    checks = by_name(collect_health(parts_for(tmp_db_path, tmp_path, backup=backup, oi_service=old, churn_service=broken), today=date(2026, 11, 11)))
    assert checks[("Backup", "Last scheduled backup")].status == FAIL
    assert checks[("OI / Volume / ATR", "Data")].status == WARN and "41 day" in checks[("OI / Volume / ATR", "Data")].detail
    assert checks[("Churn", "Download")].status == FAIL and "HTTP 500" in checks[("Churn", "Download")].detail
    assert overall(list(checks.values())) == FAIL


def test_a_fresh_backup_live_poll_and_data_are_ok(tmp_db_path, tmp_path):
    import time

    parts = parts_for(tmp_db_path, tmp_path, oi_service=Service(running=False, has_data=True, updated_on=date.today().isoformat(), error=None))
    parts.backup.run_once()
    parts.live_cache.fetched_at, parts.live_cache.updated_label = time.monotonic(), "Last: 10:00:00"
    parts.qh_api = SimpleNamespace(spare_requests=lambda name: 5)
    checks = by_name(collect_health(parts))
    assert checks[("Backup", "Newest backup")].status == OK and checks[("Live prices", "Last poll")].status == OK
    assert checks[("OI / Volume / ATR", "Data")].status == OK and checks[("API", "Free ohlc requests")].status == INFO


def test_a_check_that_cannot_be_read_never_stops_the_others(tmp_db_path, tmp_path):
    def boom():
        raise RuntimeError("boom")

    checks = collect_health(parts_for(tmp_db_path, tmp_path, oi_service=SimpleNamespace(status=boom)))
    failed = [c for c in checks if "could not be read: boom" in c.detail]
    assert len(failed) == 1 and failed[0].status == WARN and any(c.area == "Database" for c in checks)


def test_a_curve_state_with_a_recent_update_is_ok(tmp_db_path, tmp_path):
    state = SimpleNamespace(updated_at=datetime.now(timezone.utc))
    checks = by_name(collect_health(parts_for(tmp_db_path, tmp_path, curve_service=SimpleNamespace(state=lambda: state))))
    assert checks[("Curve Kinks", "Last computation")].status == OK


def test_the_health_panel_callback_fills_the_table(tmp_db_path, tmp_path, monkeypatch):
    import dash
    from dash.exceptions import PreventUpdate

    from ui.callbacks import settings_callbacks as sc
    from ui.container import Container

    repo = Repository(str(tmp_db_path))
    monkeypatch.setattr(sc, "container", Container(repository=repo, backup=BackupScheduler(tmp_db_path, tmp_path / "bk")))
    monkeypatch.setattr(dash, "callback_context", SimpleNamespace(triggered_id="url"))
    summary, rows = sc.health_panel("/settings", None)
    assert summary[0] in "✅⚠❌" and {"mark", "area", "name", "detail"} <= set(rows[0])
    with pytest.raises(PreventUpdate):
        sc.health_panel("/home", None)


# ---------------- API probe ----------------


def test_the_probe_builds_the_box_and_spread_candidates_from_one_delivery_month():
    codes = {code: group for group, code in candidate_codes(12, 2026)}
    assert codes["WTCLCOZ26"] == "box outbox" and codes["WTCLCOZ26-F27"] == "box outbox"
    assert codes["CLBZZXZ26-F27"] == "box spreadbox" and codes["CLBZZXZ26-F27-G27"] == "box flybox"
    assert codes["CLZ26-COZ26"] == "cross-product spread" and codes["CLZ26"] == "known good"


def test_the_probe_reports_ok_empty_and_refused_without_raising():
    class Api:
        def ohlc(self, interval, instruments, count):
            code = instruments[0]
            if code.startswith("WTCLCO"):
                return {code: [1, 2, 3]}
            if code.startswith("CLBZZX"):
                return {}
            raise RuntimeError("unknown instrument " + code)

    outcomes = {code: outcome for _, code, outcome in probe(Api(), candidate_codes(12, 2026))}
    assert outcomes["WTCLCOZ26"] == "OK (3 rows)" and outcomes["CLBZZXZ26-F27"] == "empty"
    assert outcomes["CLZ26-COZ26"].startswith("error: unknown instrument")


# ---------------- container and dependencies ----------------


def test_docker_runs_one_gunicorn_worker_with_threads_and_a_health_check():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert '"--workers", "1"' in dockerfile and '"--threads"' in dockerfile and "WEB_CONCURRENCY=1" in dockerfile
    assert "HEALTHCHECK" in dockerfile and not (ROOT / "docker-compose.yml").read_text(encoding="utf-8").startswith("version")


def test_the_image_never_carries_the_env_file_the_database_or_the_data():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert {".env", "db/*.db", "data", "db/secret.key"} <= set(ignored)


def test_the_health_check_script_treats_a_login_prompt_as_up(monkeypatch):
    from tools import healthcheck

    def answer(code):
        def raiser(*args, **kwargs):
            raise urllib.error.HTTPError("http://x", code, "x", {}, None)

        return raiser

    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", answer(401))
    assert healthcheck.main() == 0
    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", answer(500))
    assert healthcheck.main() == 1


def test_every_requirement_is_locked_and_every_lock_pin_satisfies_it():
    packaging = pytest.importorskip("packaging.requirements")
    locked = {}
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        match = re.match(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match:
            locked[match.group(1).lower()] = match.group(2)
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        requirement = packaging.Requirement(line)
        name = requirement.name.lower()
        assert name in locked, f"{name} is in requirements.txt but not in requirements.lock"
        assert requirement.specifier.contains(locked[name]), f"{name}=={locked[name]} breaks requirements.txt ({requirement.specifier})"


def test_the_market_data_caches_are_not_tracked_but_hand_typed_uploads_are():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for path in ("data/atr/", "data/churn/", "data/curves/", "data/oi/"):
        assert path in ignore.splitlines()
    assert "data/correlation_uploads/ is NOT ignored" in ignore


def test_logout_answers_401_with_a_fresh_challenge_so_the_browser_forgets_the_login():
    import flask

    from ui.access_routes import add_logout_route

    server = flask.Flask("t")
    add_logout_route(server)
    response = server.test_client().get("/logout")
    assert response.status_code == 401 and "Basic" in response.headers["WWW-Authenticate"] and b"Signed out" in response.data


def test_the_sidebar_shows_sign_out_only_when_a_login_is_set():
    from ui.layouts.shell import build_sidebar

    def ids(component):
        found = [getattr(component, "id", None)]
        for child in getattr(component, "children", None) or []:
            found += ids(child) if hasattr(child, "children") or hasattr(child, "id") else []
        return found

    assert "sidebar-signout" in ids(build_sidebar(True)) and "sidebar-signout" not in ids(build_sidebar())
