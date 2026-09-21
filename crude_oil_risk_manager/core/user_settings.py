"""User-configurable settings: SQLite settings-table keys and input validation.

Everything the Settings tab saves goes through the validators here first, so
bad input (text in a number field, a positive P&L stop, a non-https webhook)
is rejected with a readable message instead of being stored. Validators raise
ValueError; the message is safe to show to the user and never contains secrets.
"""

import math
from urllib.parse import urlparse

from core.alerts import ENABLED_SETTING_KEY, WEBHOOK_SETTING_KEY

KEY_API_TOKEN = "api_access_token"
KEY_PNL_STOP = "alert_portfolio_pnl_stop"
KEY_STRUCTURE_MAX_LOSS = "alert_structure_max_loss"
KEY_STALENESS = "staleness_threshold_seconds"
KEY_MARGIN_LIMIT = "margin_limit"
KEY_ROLL_WARNING_DAYS = "roll_warning_days"
KEY_VAR_CONFIDENCE = "var_confidence"
KEY_CORRELATION_WINDOW = "correlation_window"
KEY_TEAMS_WEBHOOK = WEBHOOK_SETTING_KEY
KEY_TEAMS_ENABLED = ENABLED_SETTING_KEY

VAR_CONFIDENCE_CHOICES = (0.95, 0.99)
# DataLoader.align_series needs at least 20 common dates, so a smaller window can never work.
MIN_CORRELATION_WINDOW = 20
MAX_CORRELATION_WINDOW = 500

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
    pnl_stop, structure_max_loss, staleness_seconds, margin_limit, roll_warning_days
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
        KEY_ROLL_WARNING_DAYS: parse_number(
            roll_warning_days, "Roll Warning days", minimum=0, maximum=90, integer=True
        ),
    }


def validate_defaults(var_confidence, correlation_window) -> dict[str, float | int]:
    """Validate the analysis defaults and return {settings_key: value}."""
    confidence = parse_number(var_confidence, "VaR Confidence Level")
    if confidence not in VAR_CONFIDENCE_CHOICES:
        raise ValueError("VaR Confidence Level must be 95% or 99%")
    return {
        KEY_VAR_CONFIDENCE: confidence,
        KEY_CORRELATION_WINDOW: parse_number(
            correlation_window,
            "Correlation Window",
            minimum=MIN_CORRELATION_WINDOW,
            maximum=MAX_CORRELATION_WINDOW,
            integer=True,
        ),
    }


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
