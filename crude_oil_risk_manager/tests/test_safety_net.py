"""The safety net around the database: SQLite settings, encrypted secrets, the audit log and backups."""

import json
import sqlite3
import sys

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from core.backup import BackupError, BackupScheduler, create_backup, list_backups, restore_backup
from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType
from core.secret_box import SENSITIVE_SETTING_KEYS, SecretBox, is_encrypted
from core.user_settings import KEY_API_TOKEN, KEY_TEAMS_WEBHOOK
from db.repository import Repository


def structure(name="CL Dec25 Outright", status=StructureStatus.OPEN):
    contract = Contract(product="CL", contract_month=12, contract_year=2025, symbol="CLZ25", multiplier=1000.0, tick_size=0.01, tick_value=10.0)
    return Structure(name=name, structure_type=StructureType.OUTRIGHT, products=["CL"], status=status, legs=[Leg(contract=contract, ratio=1, lots=2.0, entry_price=70.0)])


def trade_for(s, lots=2.0, price=70.0, kind=TradeEventType.TRADE):
    return Trade(structure_id=s.structure_id, leg_id=s.legs[0].leg_id, event_type=kind, lots=lots, price=price, direction="buy")


def raw_setting(path, key):
    connection = sqlite3.connect(str(path))
    try:
        row = connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    finally:
        connection.close()
    return None if row is None else json.loads(row[0])


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


# ---------------- SQLite settings ----------------


def test_every_connection_uses_wal_a_busy_timeout_and_enforces_foreign_keys(repo):
    with repo._engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() >= 10_000


def test_a_trade_for_a_structure_that_does_not_exist_is_refused(repo):
    ghost = structure()
    with pytest.raises(IntegrityError):
        repo.save_trade(trade_for(ghost))


def test_saving_a_structure_again_and_deleting_one_still_work_with_foreign_keys_on(repo):
    s = structure()
    repo.save_structure(s)
    repo.save_trade(trade_for(s))
    repo.save_structure(s)  # INSERT OR REPLACE over a structure that has legs and trades
    assert len(repo.get_trades_for_structure(s.structure_id)) == 1
    repo.delete_structure(s.structure_id)
    assert repo.get_structure(s.structure_id) is None


# ---------------- encrypted secrets ----------------


def test_the_api_token_is_encrypted_in_the_database_and_read_back_plain(tmp_db_path):
    repo = Repository(str(tmp_db_path))
    repo.set_setting(KEY_API_TOKEN, "tok-123")
    repo.set_setting(KEY_TEAMS_WEBHOOK, "https://example.test/hook")
    assert repo.get_setting(KEY_API_TOKEN) == "tok-123"
    assert repo.get_all_settings()[KEY_TEAMS_WEBHOOK] == "https://example.test/hook"
    stored = raw_setting(tmp_db_path, KEY_API_TOKEN)
    assert is_encrypted(stored) and "tok-123" not in stored
    assert "tok-123" not in tmp_db_path.read_bytes().decode("latin-1")  # nor anywhere else in the file


def test_other_settings_stay_plain_and_the_secret_key_list_matches_the_settings_keys():
    assert SENSITIVE_SETTING_KEYS == {KEY_API_TOKEN, KEY_TEAMS_WEBHOOK}


def test_a_plain_secret_saved_before_encryption_is_encrypted_when_the_app_starts(tmp_db_path):
    first = Repository(str(tmp_db_path), secret_box=SecretBox(method=None))  # no encryption available: stored as it came
    first.set_setting(KEY_API_TOKEN, "old-plain-token")
    assert raw_setting(tmp_db_path, KEY_API_TOKEN) == "old-plain-token"
    second = Repository(str(tmp_db_path))
    assert is_encrypted(raw_setting(tmp_db_path, KEY_API_TOKEN))
    assert second.get_setting(KEY_API_TOKEN) == "old-plain-token"


def test_a_secret_that_cannot_be_decrypted_here_reads_as_not_configured(tmp_db_path, tmp_path):
    pytest.importorskip("cryptography")
    writer = Repository(str(tmp_db_path), secret_box=SecretBox(tmp_path / "keys_a", "fernet"))
    writer.set_setting(KEY_API_TOKEN, "tok-123")
    reader = Repository(str(tmp_db_path), secret_box=SecretBox(tmp_path / "keys_b", "fernet"))  # another key
    assert reader.get_setting(KEY_API_TOKEN, "") == ""


def test_secret_box_round_trips_with_the_platform_method_and_with_a_fernet_key_file(tmp_path):
    pytest.importorskip("cryptography")
    for box in (SecretBox(tmp_path / "k"), SecretBox(tmp_path / "k2", "fernet")):
        sealed = box.encrypt("tok-123")
        assert is_encrypted(sealed) and "tok-123" not in sealed
        assert box.encrypt(sealed) == sealed  # never twice
        assert box.decrypt(sealed) == "tok-123"
        assert box.decrypt("plain") == "plain" and box.encrypt("") == ""
    assert (tmp_path / "k2" / "secret.key").exists()
    again = SecretBox(tmp_path / "k2", "fernet")  # the key file is reused, so earlier secrets stay readable
    assert again.decrypt(SecretBox(tmp_path / "k2", "fernet").encrypt("x")) == "x"


def test_without_any_method_a_secret_stays_plain_and_saving_still_works(tmp_path):
    box = SecretBox(method=None)
    assert box.encrypt("tok") == "tok" and box.decrypt("tok") == "tok"


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is Windows only")
def test_dpapi_is_the_default_on_windows():
    assert SecretBox().method == "dpapi"


# ---------------- audit log ----------------


def actions(repo, **filters):
    return [row["action"] for row in repo.get_audit_log(**filters)]


def test_structures_and_trades_write_audit_lines(repo):
    s = structure()
    repo.save_structure(s)
    t = trade_for(s)
    repo.save_trade(t)
    edited = t.model_copy(update={"price": 71.5})
    repo.update_trade_fields(edited)
    repo.update_structure_status(s.structure_id, StructureStatus.CLOSED, "manual")
    repo.delete_trade(t.trade_id)
    repo.delete_structure(s.structure_id)
    assert actions(repo) == ["structure.delete", "trade.delete", "structure.status", "trade.edit", "trade.add", "structure.create"]
    edit = repo.get_audit_log(entity_type="trade", entity_id=t.trade_id)[1]
    assert edit["action"] == "trade.edit" and edit["details"]["changed"]["price"] == [70.0, 71.5]
    assert "CL Dec25 Outright" in edit["summary"] and "70.0 -> 71.5" in edit["summary"]
    delete = repo.get_audit_log(entity_type="structure")[0]
    assert delete["details"]["trades"] == 0 and "Deleted CL Dec25 Outright" in delete["summary"]


def test_an_edit_that_changes_nothing_and_a_repeated_status_are_not_audited(repo):
    s = structure()
    repo.save_structure(s)
    t = trade_for(s)
    repo.save_trade(t)
    before = len(repo.get_audit_log())
    repo.update_trade_fields(t)
    repo.update_structure_status(s.structure_id, StructureStatus.OPEN)
    repo.update_structure_name(s.structure_id, s.name)
    assert len(repo.get_audit_log()) == before


def test_setting_changes_are_audited_with_old_and_new_but_secrets_only_say_they_changed(repo):
    repo.set_setting("margin_limit", 1000.0)
    repo.set_setting("margin_limit", 1000.0)  # no change, no line
    repo.set_setting("margin_limit", 2500.0)
    repo.set_setting(KEY_API_TOKEN, "tok-123")
    lines = repo.get_audit_log(entity_type="setting")
    assert [line["entity_id"] for line in lines] == [KEY_API_TOKEN, "margin_limit", "margin_limit"]
    assert lines[1]["details"] == {"before": 1000.0, "after": 2500.0} and "1000.0 -> 2500.0" in lines[1]["summary"]
    assert lines[0]["details"] == {} and "tok-123" not in json.dumps(lines[0])


def test_settings_the_app_rewrites_on_its_own_are_not_audited(repo):
    for key in ("alert_last_sent_t1_stop", "alert_stopped_t1_stop", "oi_alerts_checked_on", "curve_history_refreshed_on", "fifo_migrated_v2"):
        repo.set_setting(key, "x")
    assert repo.get_audit_log() == []


def test_the_audit_line_names_the_signed_in_user_and_a_failing_provider_does_not_break_the_change(repo):
    repo.set_actor_provider(lambda: "divya")
    repo.save_structure(structure())
    assert repo.get_audit_log()[0]["actor"] == "divya"

    def broken():
        raise RuntimeError("no request")

    repo.set_actor_provider(broken)
    repo.set_setting("margin_limit", 5.0)
    assert repo.get_audit_log()[0]["actor"] == "app"


def test_the_audit_line_and_the_change_are_one_transaction(repo):
    s = structure()
    repo.save_structure(s)
    ghost = structure("ghost")
    with pytest.raises(IntegrityError):
        repo.save_trade(trade_for(ghost))
    assert "trade.add" not in actions(repo)


# ---------------- backups ----------------


def test_a_backup_is_a_whole_readable_copy_of_the_database(repo, tmp_db_path, tmp_path):
    s = structure()
    repo.save_structure(s)
    repo.set_setting("margin_limit", 123.0)
    copy = create_backup(tmp_db_path, tmp_path / "backups")
    assert copy.name.startswith("crude_oil_risk_") and copy.suffix == ".db" and not list((tmp_path / "backups").glob("*.tmp"))
    restored = Repository(str(copy))
    assert restored.get_structure(s.structure_id).name == s.name and restored.get_setting("margin_limit") == 123.0


def test_only_the_newest_backups_are_kept_and_they_list_newest_first(repo, tmp_db_path, tmp_path):
    from datetime import datetime

    folder = tmp_path / "backups"
    for day in range(1, 6):
        create_backup(tmp_db_path, folder, keep=3, now=datetime(2026, 1, day, 9, 0, 0))
    names = [p.name for p in list_backups(folder)]
    assert names == ["crude_oil_risk_20260105_090000.db", "crude_oil_risk_20260104_090000.db", "crude_oil_risk_20260103_090000.db"]


def test_a_missing_database_is_a_clear_error_and_leaves_no_file(tmp_path):
    with pytest.raises(BackupError, match="not found"):
        create_backup(tmp_path / "nope.db", tmp_path / "backups")
    assert list_backups(tmp_path / "backups") == []


def test_restore_puts_the_backup_back_and_keeps_the_replaced_database(repo, tmp_db_path, tmp_path):
    repo.set_setting("margin_limit", 1.0)
    copy = create_backup(tmp_db_path, tmp_path / "backups")
    repo.set_setting("margin_limit", 2.0)
    repo._engine.dispose()
    safety = restore_backup(copy, tmp_db_path)
    assert Repository(str(tmp_db_path)).get_setting("margin_limit") == 1.0
    assert safety.exists() and Repository(str(safety)).get_setting("margin_limit") == 2.0
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not a database")
    with pytest.raises(BackupError):
        restore_backup(broken, tmp_db_path)


def test_the_scheduler_backs_up_at_once_when_there_is_none_and_waits_when_the_newest_is_fresh(repo, tmp_db_path, tmp_path):
    scheduler = BackupScheduler(tmp_db_path, tmp_path / "backups", interval_hours=24, keep=3)
    assert scheduler.newest() is None and scheduler._age_seconds() is None
    assert scheduler.run_once() is not None and scheduler.last_error is None
    assert scheduler._age_seconds() < 60  # fresh: the next one is a day away
    failing = BackupScheduler(tmp_path / "missing.db", tmp_path / "other", interval_hours=24)
    assert failing.run_once() is None and "not found" in failing.last_error


# ---------------- Settings: backup and audit panel ----------------


@pytest.fixture
def panel(repo, tmp_db_path, tmp_path, monkeypatch):
    import dash
    from ui.callbacks import settings_callbacks as sc
    from ui.container import Container

    fresh = Container(repository=repo, backup=BackupScheduler(tmp_db_path, tmp_path / "backups", keep=3))
    monkeypatch.setattr(sc, "container", fresh)

    class Ctx:
        triggered_id = "url"

    ctx = Ctx()
    monkeypatch.setattr(dash, "callback_context", ctx)
    return sc, ctx, fresh


def test_the_backup_panel_shows_status_and_the_audit_rows_when_settings_opens(panel, repo):
    sc, ctx, _ = panel
    repo.save_structure(structure())
    status, feedback, rows = sc.backup_panel("/settings", None)
    assert status.startswith("No backup yet") and rows[0]["action"] == "structure.create" and rows[0]["time"][:2] == "20"
    from dash.exceptions import PreventUpdate

    with pytest.raises(PreventUpdate):  # another page: nothing to update
        sc.backup_panel("/home", None)


def test_back_up_now_writes_a_copy_and_the_status_names_it(panel):
    sc, ctx, container = panel
    ctx.triggered_id = "settings-backup-now"
    status, feedback, _ = sc.backup_panel("/settings", 1)
    assert feedback.startswith("✅ Backup written: crude_oil_risk_") and "Newest backup: crude_oil_risk_" in status
    assert len(list_backups(container.backup.folder)) == 1


def test_download_makes_a_fresh_backup_and_audits_it(panel, repo):
    sc, _, container = panel
    sent = sc.download_backup(1)
    assert sent["filename"].startswith("crude_oil_risk_") and len(list_backups(container.backup.folder)) == 1
    assert repo.get_audit_log()[0]["action"] == "backup.download"
