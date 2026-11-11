"""Logging for the dashboard: the console, plus a rotating file so a run started from a shell keeps a record too.

logs/app.log rolls over at 10 MB and keeps 5 old files (about 60 MB at most). Under the Windows service NSSM also captures the console
into logs/stdout.log and rotates that itself (service/nssm_install.bat), so this file is the one that exists however the app is started.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
FILE_NAME = "app.log"
MAX_BYTES = 10 * 1024 * 1024
BACKUPS = 5


def configure_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> Path | None:
    """Console logging always; with `log_dir`, also logs/app.log (rotating). Safe to call twice: it never adds a second handler.

    Returns the log file path, or None when no file could be opened (a read-only folder must not stop the dashboard)."""
    root = logging.getLogger()
    root.setLevel(level)
    if not any(type(h) is logging.StreamHandler for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(console)
    if log_dir is None:
        return None
    path = Path(log_dir) / FILE_NAME
    if any(isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == path.resolve() for h in root.handlers):
        return path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    except OSError:
        logging.getLogger(__name__).warning("The log file %s could not be opened; logging to the console only", path)
        return None
    handler.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(handler)
    return path
