"""Settings for the OI / Volume / ATR page, kept in the SQLite settings table.

OI history length: how many CALENDAR days of daily OI before each contract's expiry to download
and analyse (default 1500). A listed contract rarely has more than ~1,850 days of history, so
the default captures the whole build-up while keeping the first download small.
"""

from core.user_settings import parse_number

KEY_OI_HISTORY_DAYS = "oi_history_days"
DEFAULT_OI_HISTORY_DAYS = 1500
MIN_OI_HISTORY_DAYS = 100
MAX_OI_HISTORY_DAYS = 5000


def validate_history_days(raw) -> int:
    """The OI history length in calendar days, or ValueError with a message safe to show."""
    return parse_number(
        raw, "OI history (days)", minimum=MIN_OI_HISTORY_DAYS, maximum=MAX_OI_HISTORY_DAYS, integer=True
    )


def load_history_days(repository) -> int:
    """Saved OI history length, falling back to the default when unset or unusable."""
    try:
        return max(MIN_OI_HISTORY_DAYS, min(MAX_OI_HISTORY_DAYS, int(repository.get_setting(KEY_OI_HISTORY_DAYS, DEFAULT_OI_HISTORY_DAYS))))
    except (TypeError, ValueError):
        return DEFAULT_OI_HISTORY_DAYS


KEY_ATR_HISTORY_DAYS = "atr_history_days"
DEFAULT_ATR_HISTORY_DAYS = 400
MIN_ATR_HISTORY_DAYS = 60
MAX_ATR_HISTORY_DAYS = 1500


def validate_atr_history_days(raw) -> int:
    """The ATR history length in calendar days before expiry, or ValueError with a message safe to show."""
    return parse_number(raw, "ATR history (days)", minimum=MIN_ATR_HISTORY_DAYS, maximum=MAX_ATR_HISTORY_DAYS, integer=True)


def load_atr_history_days(repository) -> int:
    """Saved ATR history length, falling back to the default when unset or unusable."""
    try:
        return max(MIN_ATR_HISTORY_DAYS, min(MAX_ATR_HISTORY_DAYS, int(repository.get_setting(KEY_ATR_HISTORY_DAYS, DEFAULT_ATR_HISTORY_DAYS))))
    except (TypeError, ValueError):
        return DEFAULT_ATR_HISTORY_DAYS
