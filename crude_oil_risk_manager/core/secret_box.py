"""Encryption of the few secrets the app keeps in its database (the API token and the Teams webhook).

A secret is stored as "enc:<method>:<text>" instead of plain text, so a copy of the database file (a backup, a download, a
screenshot of a table viewer) does not carry it. Two methods:

    dpapi    Windows only. The OS encrypts it for the Windows user running the app; no key to look after. It can only be read by the
             same user on the same PC, so a restored backup on another machine asks for the token again (it is never lost for good:
             the Settings tab just shows "not configured").
    fernet   Anywhere else (Docker, Linux). Needs the `cryptography` package; the key comes from the SECRETS_KEY environment
             variable, or else from a key file next to the database that is created on first use (owner-only permissions).

Without either method the secret stays plain text and a warning is logged; saving never fails because of encryption. Plain text
saved before this existed is still read, and is encrypted the next time the app starts (Repository does that).
Not covered: the .env file, which the operator controls.
"""

import base64
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PREFIX = "enc:"
SENSITIVE_SETTING_KEYS = frozenset({"api_access_token", "teams_webhook_url"})  # core.user_settings KEY_API_TOKEN / KEY_TEAMS_WEBHOOK
KEY_ENV_VAR = "SECRETS_KEY"
KEY_FILE_NAME = "secret.key"


def is_encrypted(value) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buffer = ctypes.create_string_buffer(data, len(data))
    source, target = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), Blob()
    function = ctypes.windll.crypt32.CryptProtectData if protect else ctypes.windll.crypt32.CryptUnprotectData
    if not function(ctypes.byref(source), None, None, None, None, 0, ctypes.byref(target)):
        raise ValueError("Windows could not " + ("encrypt" if protect else "decrypt") + " the value")
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(target.pbData)


class SecretBox:
    """encrypt() and decrypt() for secret strings; `method` is "dpapi", "fernet" or None (plain text)."""

    def __init__(self, key_dir: str | Path | None = None, method: str | None = "auto"):
        self._key_dir = Path(key_dir) if key_dir else None
        self._fernet = None
        if method == "auto":
            method = "dpapi" if sys.platform == "win32" else "fernet"
        self.method = method
        if method == "fernet":
            self._fernet = self._load_fernet()
            if self._fernet is None:
                self.method = None
        if self.method is None:
            logger.warning("Secrets are stored as plain text: no encryption is available (install `cryptography` and set %s)", KEY_ENV_VAR)

    def _load_fernet(self):
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            return None
        key = os.environ.get(KEY_ENV_VAR, "").strip().encode()
        if not key and self._key_dir is not None:
            path = self._key_dir / KEY_FILE_NAME
            try:
                if path.exists():
                    key = path.read_bytes().strip()
                else:
                    key = Fernet.generate_key()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(key)
            except OSError:
                logger.warning("The secrets key file %s could not be read or written", path)
                return None
        try:
            return Fernet(key) if key else None
        except ValueError:
            logger.warning("%s is not a valid Fernet key", KEY_ENV_VAR)
            return None

    def encrypt(self, text: str) -> str:
        """The text as "enc:<method>:<data>"; unchanged (plain) when no method is available or it fails."""
        if not text or is_encrypted(text) or self.method is None:
            return text
        try:
            if self.method == "dpapi":
                return f"{PREFIX}dpapi:" + base64.b64encode(_dpapi(text.encode("utf-8"), True)).decode("ascii")
            return f"{PREFIX}fernet:" + self._fernet.encrypt(text.encode("utf-8")).decode("ascii")
        except (OSError, ValueError, AttributeError):
            logger.warning("A secret could not be encrypted; it is stored as plain text")
            return text

    def decrypt(self, value: str) -> str | None:
        """The plain text of an "enc:..." value; a plain value comes back as it is; None when it cannot be decrypted here."""
        if not is_encrypted(value):
            return value
        _, method, data = value.split(":", 2)
        try:
            if method == "dpapi" and sys.platform == "win32":
                return _dpapi(base64.b64decode(data), False).decode("utf-8")
            if method == "fernet" and self._fernet is not None:
                return self._fernet.decrypt(data.encode("ascii")).decode("utf-8")
        except Exception:  # noqa: BLE001 - another user, another PC or another key: the secret is simply unreadable here
            pass
        logger.warning("A stored secret was encrypted for another user or machine and cannot be read; enter it again in Settings")
        return None
