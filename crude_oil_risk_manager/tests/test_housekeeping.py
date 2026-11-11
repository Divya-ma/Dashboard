"""Small guarantees: money in cents, one clock, log files that rotate, and errors that are logged instead of swallowed."""

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

import pytest

from core.archive import RESULT_FLAT, RESULT_LOSS, RESULT_WIN, filter_archive_rows, result_for, summarize, with_rebate
from core.clock import LONDON, london_now, london_today, utc_now
from core.logging_setup import BACKUPS, FILE_NAME, MAX_BYTES, configure_logging
from core.money import cents


# ---------------- money ----------------


def test_cents_rounds_to_two_decimals_and_never_returns_negative_zero():
    assert cents(5.755) == pytest.approx(5.76, abs=0.0101) and cents(1.005 * 100) == 100.5
    assert cents(None) == 0.0 and str(cents(-0.0000001)) == "0.0"
    assert cents(-4.236) == -4.24


def test_float_dust_is_flat_not_a_win_or_a_loss():
    dust = 5.76 - 5.760000000001
    assert dust != 0 and result_for(dust) == RESULT_FLAT and result_for(-1e-12) == RESULT_FLAT
    assert result_for(0.01) == RESULT_WIN and result_for(-0.01) == RESULT_LOSS


def test_totals_and_win_rates_ignore_float_dust():
    rows = [
        {"realized_pnl": 0.1 + 0.2 - 0.3, "pnl_after_rebate": 0.1 + 0.2 - 0.3, "rebate": 0.0},  # 5.5e-17: flat
        {"realized_pnl": 10.0, "pnl_after_rebate": 10.0, "rebate": 0.0},
    ]
    stats = summarize(rows)
    assert stats["winners"] == 1 and stats["losers"] == 0 and stats["win_rate"] == 100.0 and stats["total_pnl"] == 10.0
    negative_dust = [{"realized_pnl": -1e-12, "pnl_after_rebate": -1e-12, "rebate": 0.0, "exit_date": None}]
    assert filter_archive_rows(negative_dust, None, None, None, None, "winners") == negative_dust  # flat is not a loser
    assert summarize([])["total_pnl"] == 0


def test_a_rebate_added_to_a_pnl_is_rounded_to_a_cent():
    class Book:
        def for_trade(self, structure, lots):
            return 13.6

    row = with_rebate({"realized_pnl": -4.24, "lots": 1.0}, None, Book())
    assert row["pnl_after_rebate"] == 9.36 and row["result_after_rebate"] == RESULT_WIN


# ---------------- clock ----------------


def test_london_time_is_london_and_utc_is_utc():
    assert london_now().tzinfo is LONDON and utc_now().utcoffset().total_seconds() == 0
    assert london_today() == london_now().date()
    assert isinstance(utc_now(), datetime)


def test_a_backup_name_uses_the_utc_clock_so_names_sort_through_the_clock_change(tmp_db_path, tmp_path):
    from core.backup import create_backup
    from db.repository import Repository

    Repository(str(tmp_db_path))
    name = create_backup(tmp_db_path, tmp_path / "bk").name
    assert name[len("crude_oil_risk_"):len("crude_oil_risk_") + 8] == utc_now().strftime("%Y%m%d")


# ---------------- logging ----------------


@pytest.fixture
def clean_root_logger():
    root = logging.getLogger()
    before = list(root.handlers)
    yield root
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()


def test_the_log_file_rotates_and_is_added_only_once(tmp_path, clean_root_logger):
    path = configure_logging(tmp_path / "logs")
    again = configure_logging(tmp_path / "logs")
    assert path == again == tmp_path / "logs" / FILE_NAME
    handlers = [h for h in clean_root_logger.handlers if isinstance(h, RotatingFileHandler) and h.baseFilename == str(path.resolve())]
    assert len(handlers) == 1 and handlers[0].maxBytes == MAX_BYTES and handlers[0].backupCount == BACKUPS
    logging.getLogger("housekeeping.test").info("written to the file")
    handlers[0].flush()
    assert "written to the file" in path.read_text(encoding="utf-8")


def test_a_log_folder_that_cannot_be_used_leaves_console_logging_working(tmp_path, clean_root_logger):
    blocked = tmp_path / "not_a_folder"
    blocked.write_text("a file where the folder should be")
    assert configure_logging(blocked / "logs") is None


# ---------------- errors that used to vanish ----------------


def test_a_failure_remembering_the_var_controls_is_logged_at_debug_level(caplog):
    from core.var import save_mc_settings

    class Broken:
        def get_setting(self, key, default=None):
            raise RuntimeError("database is locked")

    with caplog.at_level(logging.DEBUG, logger="core.var"):
        save_mc_settings(Broken(), "normal", 0, 1, 0, 1, 1000)
    assert "could not be remembered" in caplog.text and "database is locked" in caplog.text
