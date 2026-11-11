"""A health check of the running dashboard: one list of what is working, what needs a look, and what is broken.

`collect_health(parts)` takes the objects the app wires together (the container's repository, backup scheduler, live price cache,
QH client and the OI / Churn / Curve services, plus the app's own settings) and returns rows of (area, check, status, detail).
Every check is independent and never raises: one that cannot be read shows as "warn" with the reason, so a broken service
cannot hide the rest of the page.

    ok      working as expected
    warn    works but needs a look (stale, old, no data yet, not set up)
    fail    broken (a failed backup, a job that stopped with an error)
    info    a figure, not a verdict (free API requests)
"""

import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from core.clock import london_today

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
LOW_DISK_BYTES = 1_000_000_000
STALE_DATA_DAYS = 4  # a daily store older than this (a long weekend is 3) has missed a day


@dataclass(frozen=True)
class Check:
    area: str
    name: str
    status: str
    detail: str


def _size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "?"
    return f"{size / 1_048_576:,.1f} MB"


def _age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172_800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86_400:.1f} days"


def _database(parts) -> list[Check]:
    repository, found = parts.repository, []
    path = Path(repository._db_path)
    wal = path.with_name(path.name + "-wal")
    found.append(Check("Database", "File", OK, f"{path} ({_size(path)}{', log ' + _size(wal) if wal.exists() else ''})"))
    free = shutil.disk_usage(path.parent).free
    found.append(Check("Database", "Free disk space", OK if free >= LOW_DISK_BYTES else WARN, f"{free / 1_073_741_824:,.1f} GB free beside the database"))
    method = repository.secret_method
    found.append(Check("Database", "Secrets at rest", OK if method else WARN, f"encrypted ({method})" if method else "stored as plain text: no encryption available"))
    last = repository.get_audit_log(1)
    found.append(Check("Database", "Audit log", OK, f"last change {str(last[0]['timestamp'])[:19].replace('T', ' ')} UTC by {last[0]['actor']}" if last else "no changes recorded yet"))
    return found


def _backup(parts) -> list[Check]:
    backup = parts.backup
    if backup is None:
        return [Check("Backup", "Scheduled backups", WARN, "not set up")]
    from core.backup import list_backups

    if backup.last_error:
        return [Check("Backup", "Last scheduled backup", FAIL, backup.last_error)]
    found = list_backups(backup.folder)
    if not found:
        return [Check("Backup", "Newest backup", WARN, f"none yet in {backup.folder}")]
    age = time.time() - found[0].stat().st_mtime
    late = age > 2 * backup.interval_seconds
    return [Check("Backup", "Newest backup", WARN if late else OK, f"{found[0].name}, {_age(age)} old; {len(found)} kept")]


def _api(parts) -> list[Check]:
    found = []
    token = bool(parts.repository.get_setting("api_access_token", ""))
    found.append(Check("API", "Token", OK if token else WARN, "saved" if token else "not saved: Settings > API Configuration"))
    if parts.qh_api is not None:
        for endpoint in ("fairvalue", "ohlc"):
            found.append(Check("API", f"Free {endpoint} requests", INFO, f"{parts.qh_api.spare_requests(endpoint)} available right now"))
    return found


def _live(parts) -> list[Check]:
    cache = parts.live_cache
    if cache.fetched_at is None:
        return [Check("Live prices", "Last poll", WARN, "no poll yet since the app started")]
    age = time.monotonic() - cache.fetched_at
    stale = cache.is_stale
    return [Check("Live prices", "Last poll", WARN if stale else OK, f"{cache.updated_label} ({_age(age)} ago)" + (", prices are stale" if stale else ""))]


def _store(area: str, service, today: date) -> list[Check]:
    if service is None:
        return [Check(area, "Service", WARN, "not started")]
    status = service.status()
    if status.get("error"):
        return [Check(area, "Download", FAIL, str(status["error"]))]
    if status.get("running"):
        return [Check(area, "Download", INFO, f"running: {status.get('done', 0)} of {status.get('total', 0)} {status.get('message', '')}".strip())]
    updated = status.get("updated_on")
    if not status.get("has_data") or not updated:
        return [Check(area, "Data", WARN, "no data downloaded yet")]
    try:
        days = (today - (updated if isinstance(updated, date) else date.fromisoformat(str(updated)[:10]))).days
    except ValueError:
        return [Check(area, "Data", OK, f"updated {updated}")]
    return [Check(area, "Data", WARN if days > STALE_DATA_DAYS else OK, f"updated {updated} ({days} day(s) ago)")]


def _curves(parts) -> list[Check]:
    service = parts.curve_service
    if service is None:
        return [Check("Curve Kinks", "Service", WARN, "not started")]
    updated = service.state().updated_at
    if updated is None:
        return [Check("Curve Kinks", "Last computation", WARN, "none yet")]
    age = (datetime.now(updated.tzinfo) - updated).total_seconds()
    return [Check("Curve Kinks", "Last computation", WARN if age > 600 else OK, f"{_age(age)} ago")]


def _access(parts) -> list[Check]:
    config = parts.app_settings
    auth = bool(config.AUTH_USERNAME and config.AUTH_PASSWORD) or bool(getattr(config, "AUTH_USERS", ""))
    open_host = config.APP_HOST not in LOOPBACK_HOSTS
    if auth:
        return [Check("Access", "Login", OK, f"on, listening on {config.APP_HOST}:{config.APP_PORT}")]
    if open_host:
        return [Check("Access", "Login", FAIL, f"OFF while listening on {config.APP_HOST}: anyone who can reach this PC sees positions and P&L")]
    return [Check("Access", "Login", OK, f"off, reachable from this PC only ({config.APP_HOST})")]


def collect_health(parts, today: date | None = None) -> list[Check]:
    """Every check, in page order. `parts` carries: repository, backup, live_cache, qh_api, oi_service, churn_service, curve_service,
    app_settings (config.settings)."""
    today = today or london_today()
    checks: list[Check] = []
    for area, function in (
        ("Access", _access), ("Database", _database), ("Backup", _backup), ("API", _api), ("Live prices", _live),
        ("OI / Volume / ATR", lambda p: _store("OI / Volume / ATR", p.oi_service, today)),
        ("Churn", lambda p: _store("Churn", p.churn_service, today)), ("Curve Kinks", _curves),
    ):
        try:
            checks.extend(function(parts))
        except Exception as exc:  # noqa: BLE001 - a check that cannot be made is reported, never raised
            checks.append(Check(area, "Check", WARN, f"could not be read: {exc}"))
    return checks


def overall(checks: list[Check]) -> str:
    """fail if anything failed, else warn if anything needs a look, else ok."""
    statuses = {c.status for c in checks}
    return FAIL if FAIL in statuses else WARN if WARN in statuses else OK
