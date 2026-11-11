"""Who may open the dashboard: the login list and the listening address.

AUTH_USERNAME / AUTH_PASSWORD is one login; AUTH_USERS adds more as "name:password" entries separated by commas or semicolons
("alice:pw1,bob:pw2"), so each person has their own login and the audit log (db/audit_log) names who changed what. A password
may not contain a comma or semicolon; a name may not contain a colon. A malformed list is an error and stops the start: a typo must
never leave the dashboard open. Every user has the same access: HTTP Basic has no roles, and no callback is read-only.
"""

import re

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


class AccessConfigError(ValueError):
    """The login configuration is not usable; the message never contains a password."""


def parse_users(users: str = "", username: str = "", password: str = "") -> dict[str, str]:
    """name -> password from AUTH_USERNAME/AUTH_PASSWORD and the AUTH_USERS list; empty when no login is configured."""
    found: dict[str, str] = {}
    if bool(username) != bool(password):
        raise AccessConfigError("AUTH_USERNAME and AUTH_PASSWORD must be set together")
    if username:
        found[username] = password
    for number, entry in enumerate(re.split(r"[;,\n]", users or ""), start=1):
        entry = entry.strip()
        if not entry:
            continue
        name, separator, secret = entry.partition(":")
        name = name.strip()
        if not separator or not name or not secret:
            raise AccessConfigError(f"AUTH_USERS entry {number} must look like name:password")
        if name in found:
            raise AccessConfigError(f"AUTH_USERS lists {name!r} more than once")
        found[name] = secret
    return found


def is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS
