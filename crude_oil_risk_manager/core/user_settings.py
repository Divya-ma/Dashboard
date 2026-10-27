"""User-configurable settings: SQLite settings-table keys and input validation.

Everything the Settings tab saves goes through the validators here first, so
bad input (text in a number field, a positive P&L stop, a non-https webhook)
is rejected with a readable message instead of being stored. Validators raise
ValueError; the message is safe to show to the user and never contains secrets.
"""

import math
import re
from urllib.parse import urlparse

from core.alerts import (
    ENABLED_SETTING_KEY, MENTION_ALL, MENTION_EMAIL_KEY, MENTION_LEVEL_CHOICES, MENTION_LEVELS_KEY, MENTION_NAME_KEY, WEBHOOK_SETTING_KEY,
)

KEY_API_TOKEN = "api_access_token"
KEY_PNL_BASELINE = "pnl_reset_baseline"
KEY_TRADE_IDEAS = "trade_ideas"  # the list of trade ideas to pick from (one per line in Settings)
KEY_EXIT_REASONS = "journal_exit_reasons"  # the exit reasons to pick from (the built-in list until edited)
KEY_MISTAKES = "journal_mistakes"  # the mistake tags to pick from (the built-in list until edited)
KEY_REQUIRE_STOP_TARGET = "require_stop_target"  # a first entry needs a stop and a target (switch in Settings)
# total_realized_all_time needs its own baseline: it excludes live unrealized PnL, so
# shifting it by the same baseline as total_pnl only reads zero right after a reset if
# unrealized PnL happened to be zero at that moment. See core.pnl.apply_pnl_reset_baseline.
KEY_PNL_REALIZED_BASELINE = "pnl_reset_realized_baseline"
KEY_PNL_STOP = "alert_portfolio_pnl_stop"
KEY_STRUCTURE_MAX_LOSS = "alert_structure_max_loss"
KEY_STALENESS = "staleness_threshold_seconds"
KEY_MARGIN_LIMIT = "margin_limit"
KEY_VAR_CONFIDENCE = "var_confidence"
# Correlation = average over the last LOOKBACK days of the daily ROLLING-day correlation
# (core.correlation). New keys, so values saved under the old single-window settings
# ("correlation_window", "rolling_correlation_window") are not misread as these.
KEY_CORRELATION_ROLLING = "correlation_rolling_window"
KEY_CORRELATION_LOOKBACK = "correlation_lookback"
KEY_TEAMS_WEBHOOK = WEBHOOK_SETTING_KEY
KEY_TEAMS_ENABLED = ENABLED_SETTING_KEY
KEY_TEAMS_MENTION_EMAIL = MENTION_EMAIL_KEY
KEY_TEAMS_MENTION_NAME = MENTION_NAME_KEY
KEY_TEAMS_MENTION_LEVELS = MENTION_LEVELS_KEY

VAR_CONFIDENCE_CHOICES = (0.95, 0.99)
MIN_ROLLING_WINDOW = 2  # a correlation needs at least two days
MIN_CORRELATION_LOOKBACK = 1
MAX_CORRELATION_WINDOW = 5000

_MAX_TOKEN_LENGTH = 4096
_MAX_URL_LENGTH = 2048


def parse_number(
    value,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    integer: bool = False,
) -> float | int:
    """Convert a form value to a finite number, optionally a whole number within bounds."""
    if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{label} is required")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    if integer:
        if number != int(number):
            raise ValueError(f"{label} must be a whole number")
        number = int(number)
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be at most {maximum:g}")
    return number


def validate_token(raw) -> str:
    """Clean and validate a Bearer token: trims whitespace and a pasted 'Bearer ' prefix."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Token cannot be empty")
    token = raw.strip()
    if token.lower().startswith("bearer "):
        token = token[len("bearer "):].strip()
    if not token or token.lower() == "bearer":
        raise ValueError("Token cannot be empty")
    if any(ch.isspace() for ch in token):
        raise ValueError("Token must not contain spaces or line breaks")
    if len(token) > _MAX_TOKEN_LENGTH:
        raise ValueError("Token is too long")
    return token


def validate_thresholds(
    pnl_stop, structure_max_loss, staleness_seconds, margin_limit
) -> dict[str, float | int]:
    """Validate the risk/alert thresholds and return {settings_key: value}."""
    pnl_stop = parse_number(pnl_stop, "Portfolio PnL Stop")
    if pnl_stop > 0:
        raise ValueError(
            "Portfolio PnL Stop must be zero or negative (e.g. -50000): the alert fires when PnL falls below it"
        )
    structure_max_loss = parse_number(structure_max_loss, "Per-Structure Max Loss")
    if structure_max_loss > 0:
        raise ValueError(
            "Per-Structure Max Loss must be zero or negative (e.g. -10000): the alert fires when PnL falls below it"
        )
    return {
        KEY_PNL_STOP: pnl_stop,
        KEY_STRUCTURE_MAX_LOSS: structure_max_loss,
        KEY_STALENESS: parse_number(staleness_seconds, "Data Staleness Threshold", minimum=1, maximum=3600),
        KEY_MARGIN_LIMIT: parse_number(margin_limit, "Portfolio Margin Limit", minimum=0),
    }


def validate_defaults(var_confidence, rolling_window, lookback) -> dict[str, float | int]:
    """Validate the analysis defaults and return {settings_key: value}."""
    confidence = parse_number(var_confidence, "VaR Confidence Level")
    if confidence not in VAR_CONFIDENCE_CHOICES:
        raise ValueError("VaR Confidence Level must be 95% or 99%")
    return {
        KEY_VAR_CONFIDENCE: confidence,
        KEY_CORRELATION_ROLLING: parse_number(
            rolling_window,
            "Rolling Window",
            minimum=MIN_ROLLING_WINDOW,
            maximum=MAX_CORRELATION_WINDOW,
            integer=True,
        ),
        KEY_CORRELATION_LOOKBACK: parse_number(
            lookback,
            "Lookback Period",
            minimum=MIN_CORRELATION_LOOKBACK,
            maximum=MAX_CORRELATION_WINDOW,
            integer=True,
        ),
    }


def correlation_windows(repository) -> tuple[int, int]:
    """(rolling window, lookback) in days from the settings table, falling back to the defaults."""
    from config.settings import settings  # local: keeps this module's import list unchanged for its other users

    def read(key, default, minimum):
        try:
            return max(minimum, int(repository.get_setting(key, default)))
        except (TypeError, ValueError):
            return default

    return (
        read(KEY_CORRELATION_ROLLING, settings.DEFAULT_CORRELATION_ROLLING_WINDOW, MIN_ROLLING_WINDOW),
        read(KEY_CORRELATION_LOOKBACK, settings.DEFAULT_CORRELATION_LOOKBACK, MIN_CORRELATION_LOOKBACK),
    )


_EMAIL = re.compile(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+")


def validate_mention_email(raw) -> str:
    """The email a Teams alert @mentions. Empty is allowed (no mention); otherwise it must look like name@company.com."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValueError("The Teams mention email must be text")
    email = raw.strip()
    if email and (len(email) > 254 or not _EMAIL.fullmatch(email)):
        raise ValueError("The Teams mention email must look like name@company.com")
    return email


def validate_mention_name(raw) -> str:
    """The optional name shown in the mention (blank = made from the email)."""
    name = (raw or "").strip() if isinstance(raw, str) or raw is None else None
    if name is None or len(name) > 60 or any(ch in name for ch in "<>&"):
        raise ValueError("The mention name must be plain text of at most 60 characters")
    return name


def validate_mention_levels(raw) -> str:
    """Which alerts @mention you: 'all', 'warning' (warning and critical) or 'critical'."""
    if raw in MENTION_LEVEL_CHOICES:
        return raw
    if raw in (None, ""):
        return MENTION_ALL
    raise ValueError("Choose which alerts mention you: all, warnings and critical, or critical only")


def validate_webhook_url(raw) -> str:
    """Validate a Teams webhook URL. Empty is allowed (clears it); otherwise it must be https."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValueError("Teams webhook URL must be text")
    url = raw.strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or any(ch.isspace() for ch in url)
        or len(url) > _MAX_URL_LENGTH
    ):
        raise ValueError("Teams webhook URL must be a valid https:// URL")
    return url
