"""Backups of the SQLite database (trades, journals, settings): a consistent copy made while the app keeps running.

The copy uses SQLite's own online backup (sqlite3.Connection.backup), so it is a whole, consistent database even if someone is
saving a trade at that moment; copying the .db file by hand while the app writes is not safe. Each copy is checked
(PRAGMA integrity_check) before it counts, written under a temporary name and renamed, so a half-written file is never taken
for a backup. Only the newest `keep` are kept.

    create_backup(db, folder)          one copy now, returns its path
    BackupScheduler(db, folder, ...)   a background thread: a copy at start-up if the newest one is old, then one every interval
    restore_backup(copy, db)           puts a copy back (the app must be stopped); the current database is kept as ".before-restore"

Stored secrets (API token, Teams webhook) are encrypted in the database (core.secret_box), so they are encrypted in a backup too;
a copy restored on another PC or Windows user asks for them again in Settings. The data/ folder (market-data caches, uploads) is
not part of this: it can be rebuilt from the API.

Restore from a command line, with the dashboard stopped:  python -m core.backup restore <backup file> [--db <database file>]
"""

import argparse
import logging
import shutil
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PREFIX, SUFFIX = "crude_oil_risk_", ".db"
DEFAULT_KEEP = 14
DEFAULT_INTERVAL_HOURS = 24.0
_BACKUP_LOCK = threading.Lock()  # one copy at a time: the scheduler and a Settings download must not clean up each other's temporary file


class BackupError(RuntimeError):
    """A backup that could not be made or restored; the message says why."""


def _check(path: Path) -> None:
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            result = connection.execute("PRAGMA integrity_check").fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise BackupError(f"{path.name} is not a readable database ({exc})") from None
    if not result or result[0] != "ok":
        raise BackupError(f"{path.name} failed the database integrity check")


def list_backups(folder: str | Path) -> list[Path]:
    """The backup files in the folder, newest first."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.glob(f"{PREFIX}*{SUFFIX}") if p.is_file()), key=lambda p: p.name, reverse=True)


def _create_backup(db_path: str | Path, folder: str | Path, keep: int = DEFAULT_KEEP, now: datetime | None = None) -> Path:
    """Copy the database into the folder as crude_oil_risk_<date>_<time>.db, verify it, and drop copies beyond the newest `keep`."""
    db_path, folder = Path(db_path), Path(folder)
    if not db_path.is_file():
        raise BackupError(f"database {db_path} not found")
    folder.mkdir(parents=True, exist_ok=True)
    for stale in (*folder.glob(f"{PREFIX}*.tmp"), *folder.glob(f"{PREFIX}*.tmp-journal")):  # left by a copy that was cut short
        stale.unlink(missing_ok=True)
    target = folder / f"{PREFIX}{(now or datetime.now()).strftime('%Y%m%d_%H%M%S')}{SUFFIX}"
    temporary = target.with_suffix(".tmp")
    try:
        source = sqlite3.connect(str(db_path), timeout=30)
        try:
            destination = sqlite3.connect(str(temporary))
            try:
                source.backup(destination)
            finally:
                destination.close()
        finally:
            source.close()
        _check(temporary)
        temporary.replace(target)
    except (sqlite3.Error, OSError, BackupError) as exc:
        temporary.unlink(missing_ok=True)
        raise BackupError(f"backup failed: {exc}") from None
    for old in list_backups(folder)[max(1, int(keep)):]:
        try:
            old.unlink()
        except OSError:
            logger.warning("Old backup %s could not be removed", old)
    return target


def create_backup(db_path: str | Path, folder: str | Path, keep: int = DEFAULT_KEEP, now: datetime | None = None) -> Path:
    """Copy the database into the folder as crude_oil_risk_<date>_<time>.db, verify it, and drop copies beyond the newest `keep`."""
    with _BACKUP_LOCK:
        return _create_backup(db_path, folder, keep, now)


def restore_backup(backup: str | Path, db_path: str | Path) -> Path:
    """Replace the database with a backup. The dashboard must be stopped. The database being replaced is kept next to it as
    <name>.before-restore (and its -wal / -shm files are cleared so the old log is not replayed over the restored data)."""
    backup, db_path = Path(backup), Path(db_path)
    _check(backup)
    safety = db_path.with_name(db_path.name + ".before-restore")
    if db_path.exists():
        shutil.copy2(db_path, safety)
    for extra in (db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm")):
        extra.unlink(missing_ok=True)
    shutil.copy2(backup, db_path)
    return safety


class BackupScheduler:
    """Makes a backup in the background: at start-up when the newest is older than the interval, then every interval."""

    def __init__(self, db_path, folder, interval_hours: float = DEFAULT_INTERVAL_HOURS, keep: int = DEFAULT_KEEP):
        self.db_path, self.folder, self.keep = Path(db_path), Path(folder), keep
        self.interval_seconds = max(0.05, float(interval_hours)) * 3600.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = None

    def newest(self) -> Path | None:
        found = list_backups(self.folder)
        return found[0] if found else None

    def _age_seconds(self) -> float | None:
        newest = self.newest()
        return None if newest is None else time.time() - newest.stat().st_mtime

    def run_once(self) -> Path | None:
        """A backup now; the failure (never raised) is kept in `last_error` and logged."""
        try:
            path = create_backup(self.db_path, self.folder, self.keep)
        except BackupError as exc:
            self.last_error = str(exc)
            logger.error("Scheduled database backup failed: %s", exc)
            return None
        self.last_error = None
        logger.info("Database backup written: %s", path)
        return path

    def _loop(self) -> None:
        age = self._age_seconds()
        wait = 0.0 if age is None else max(0.0, self.interval_seconds - age)
        while not self._stop.wait(wait):
            self.run_once()
            wait = self.interval_seconds

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="db-backup", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Back up or restore the dashboard database")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("backup", help="make a backup now")
    make.add_argument("--db", default="db/crude_oil_risk.db")
    make.add_argument("--to", default="db/backups")
    restore = sub.add_parser("restore", help="put a backup back (stop the dashboard first)")
    restore.add_argument("backup")
    restore.add_argument("--db", default="db/crude_oil_risk.db")
    args = parser.parse_args(argv)
    try:
        if args.command == "backup":
            print(create_backup(args.db, args.to))
        else:
            print(f"Restored. The previous database is kept as {restore_backup(args.backup, args.db)}")
    except BackupError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
