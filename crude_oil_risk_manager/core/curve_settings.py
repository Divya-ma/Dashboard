"""Settings for the Curve Kinks page: detection thresholds (edited on the page) and the
daily previous-settlement fetch time (edited in Settings).

All values live in the SQLite settings table. Everything the user can type is validated here
first; validators raise ValueError with a message that is safe to show.
"""

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.user_settings import parse_number

KEY_CURVE_PARAMS = "curve_params"
KEY_CURVE_OPEN_TIME = "curve_open_time"  # "HH:MM"
KEY_CURVE_OPEN_TZ = "curve_open_timezone"  # IANA name

DEFAULT_OPEN_TIME = "07:00"
DEFAULT_OPEN_TZ = "UTC"
TIMEZONE_CHOICES = ("UTC", "America/New_York", "America/Chicago", "Europe/London", "Europe/Zurich", "Asia/Dubai", "Asia/Singapore")

PRIORITIES = ("LOW", "MEDIUM", "HIGH")
PRIORITY_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


@dataclass
class CurveParams:
    """Kink-detection thresholds. Defaults: single method |z| > 3, two methods agreeing = HIGH, alerts from MEDIUM up."""

    z_fit: float = 3.0  # robust smooth-curve fit residual (cross-sectional)
    z_neighbour: float = 3.0  # leave-one-out interpolation from neighbouring contracts
    z_pca: float = 3.0  # residual after rebuilding the curve from the top principal components
    z_history: float = 3.0  # value vs its own history (spread / fly / dfly only)
    min_methods_high: int = 2  # methods that must agree for HIGH priority
    max_kink_fraction: float = 0.4  # more of a curve flagged than this = data mismatch / whole-curve move, not kinks
    seasonal_z: float = 1.5  # |seasonal z| at or below this = "seasonally normal"
    seasonal_window_days: int = 15  # +/- calendar days around today's date for the seasonal sample
    lookback_days: int = 504  # history used for scales / PCA / own-history z
    pca_components: int = 3
    poly_degree: int = 3
    min_scale: float = 0.005  # price points; floor on every scale so a flat curve can't give huge z
    alert_min_priority: str = "MEDIUM"
    cooldown_minutes: int = 60
    alerts_enabled: bool = True
    stale_seconds: int = 120  # a live price older than this is flagged stale
    snapshot_minutes: int = 15
    history_years: int = 5

    @property
    def thresholds(self) -> dict[str, float]:
        return {"fit": self.z_fit, "neighbour": self.z_neighbour, "pca": self.z_pca, "history": self.z_history}


_NUMERIC_BOUNDS = {
    "z_fit": (0.5, 20.0, False), "z_neighbour": (0.5, 20.0, False), "z_pca": (0.5, 20.0, False),
    "z_history": (0.5, 20.0, False), "min_methods_high": (1, 4, True), "max_kink_fraction": (0.1, 1.0, False), "seasonal_z": (0.0, 10.0, False),
    "seasonal_window_days": (1, 60, True), "lookback_days": (60, 2000, True), "pca_components": (1, 6, True),
    "poly_degree": (1, 5, True), "min_scale": (0.0001, 5.0, False), "cooldown_minutes": (1, 1440, True),
    "stale_seconds": (5, 3600, True), "snapshot_minutes": (1, 240, True), "history_years": (1, 10, True),
}
_LABELS = {
    "z_fit": "Fit z", "z_neighbour": "Neighbour z", "z_pca": "PCA z", "z_history": "History z",
    "min_methods_high": "Methods for HIGH", "max_kink_fraction": "Max kinked share", "seasonal_z": "Seasonal z", "seasonal_window_days": "Seasonal window",
    "lookback_days": "Lookback days", "pca_components": "PCA components", "poly_degree": "Fit polynomial degree",
    "min_scale": "Minimum scale", "cooldown_minutes": "Alert cooldown", "stale_seconds": "Stale price seconds",
    "snapshot_minutes": "Snapshot interval", "history_years": "History years",
}


def validate_params(raw: dict) -> CurveParams:
    """Build CurveParams from form values (numbers may be strings), rejecting out-of-range input."""
    values = {}
    for name, (low, high, integer) in _NUMERIC_BOUNDS.items():
        values[name] = parse_number(raw.get(name), _LABELS[name], minimum=low, maximum=high, integer=integer)
    priority = raw.get("alert_min_priority")
    if priority not in PRIORITIES:
        raise ValueError("Alert minimum priority must be LOW, MEDIUM or HIGH")
    values["alert_min_priority"] = priority
    values["alerts_enabled"] = bool(raw.get("alerts_enabled"))
    return CurveParams(**values)


def load_params(repository) -> CurveParams:
    """Saved thresholds, with defaults for anything missing or invalid (never raises)."""
    try:
        saved = repository.get_setting(KEY_CURVE_PARAMS, None)
        if isinstance(saved, str):
            saved = json.loads(saved)
        known = {f.name for f in fields(CurveParams)}
        merged = {**asdict(CurveParams()), **{k: v for k, v in (saved or {}).items() if k in known}}
        return validate_params(merged)
    except Exception:  # noqa: BLE001 - bad stored settings must not stop the page
        return CurveParams()


def save_params(repository, params: CurveParams) -> None:
    repository.set_setting(KEY_CURVE_PARAMS, asdict(params))


def validate_open_time(raw_time, raw_tz) -> tuple[str, str]:
    """("07:00", "Europe/London") -> validated pair; raises ValueError on a bad time or time zone."""
    if not isinstance(raw_time, str) or not raw_time.strip():
        raise ValueError("Opening time is required (HH:MM)")
    try:
        parsed = datetime.strptime(raw_time.strip(), "%H:%M").time()
    except ValueError:
        raise ValueError("Opening time must look like 07:00 (24-hour HH:MM)") from None
    tz = (raw_tz or "").strip()
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        raise ValueError("Unknown time zone") from None
    return parsed.strftime("%H:%M"), tz


def load_open_time(repository) -> tuple[time, ZoneInfo]:
    """(opening time, zone) from settings, falling back to the defaults if unset or invalid."""
    try:
        text, tz = validate_open_time(
            repository.get_setting(KEY_CURVE_OPEN_TIME, DEFAULT_OPEN_TIME),
            repository.get_setting(KEY_CURVE_OPEN_TZ, DEFAULT_OPEN_TZ),
        )
    except ValueError:
        text, tz = DEFAULT_OPEN_TIME, DEFAULT_OPEN_TZ
    return datetime.strptime(text, "%H:%M").time(), ZoneInfo(tz)
