"""Settings for the Curve Kinks page: detection thresholds (edited on the page) and the
daily previous-settlement fetch time (edited in Settings).

All values live in the SQLite settings table. Everything the user can type is validated here
first; validators raise ValueError with a message that is safe to show.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.curve_calendar import FAMILIES, PRODUCTS
from core.user_settings import parse_number

KEY_CURVE_PARAMS = "curve_params"
KEY_CURVE_OPEN_TIME = "curve_open_time"  # "HH:MM"
KEY_CURVE_OPEN_TZ = "curve_open_timezone"  # IANA name

DEFAULT_OPEN_TIME = "07:00"
DEFAULT_OPEN_TZ = "UTC"
TIMEZONE_CHOICES = ("UTC", "America/New_York", "America/Chicago", "Europe/London", "Europe/Zurich", "Asia/Dubai", "Asia/Singapore")

# One alert switch per product and curve type, as "PRODUCT:family" (e.g. "BRN:spread").
ALERT_KEYS = tuple(f"{product}:{family}" for product in PRODUCTS for family in FAMILIES)

# Which curve types may hedge a kink of each type (the user can change this in Settings).
DEFAULT_HEDGE_TYPES = {
    "outright": ["outright", "spread", "fly", "dfly"],
    "spread": ["spread", "fly", "dfly"],
    "fly": ["spread", "fly", "dfly"],
    "dfly": ["fly", "dfly"],
}

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
    alert_structures: list[str] = field(default_factory=lambda: list(ALERT_KEYS))  # which product:curve types may alert

    # --- trade plan: risk appetite and sizing (dollars; the alert structure is traded in lots) ---
    risk_per_trade: float = 5000.0  # max $ loss at the stop for the whole entry
    risk_per_lot: float = 1000.0  # max $ loss per lot at the stop: caps how far the stop can be
    point_value: float = 1000.0  # $ per 1.0 price point per lot (1,000 bbl)
    max_lots: int = 100  # never suggest more than this, however tight the stop
    vol_stop_mult: float = 1.5  # volatility stop = this x the structure's daily move
    vol_window: int = 60  # days of daily moves for that volatility
    reversion_horizon: int = 10  # days the backtest allows a kink to close
    min_reversion_fraction: float = 0.5  # the target takes at least this share of the gap to fair value
    min_reward_risk: float = 0.0  # 0 = off; otherwise no alert unless target / stop distance is at least this

    # --- hedges ---
    hedge_count: int = 2  # ranked hedge alternatives shown (at least one is always shown)
    hedge_min_corr: float = 0.5  # |correlation| a hedge should reach to be listed
    hedge_lookback: int = 120  # days of daily changes for correlation and the VaR ratio
    hedge_exclude_overlap: bool = True  # hedge must not share a contract with the kinked structure
    hedge_types: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_HEDGE_TYPES.items()})
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
    "risk_per_trade": (1.0, 1e8, False), "risk_per_lot": (1.0, 1e7, False), "point_value": (0.01, 1e6, False),
    "max_lots": (1, 10000, True), "vol_stop_mult": (0.1, 10.0, False), "vol_window": (10, 500, True),
    "reversion_horizon": (1, 60, True), "min_reversion_fraction": (0.05, 1.0, False),
    "min_reward_risk": (0.0, 20.0, False), "hedge_count": (1, 5, True), "hedge_min_corr": (0.0, 0.99, False),
    "hedge_lookback": (30, 1000, True),
}
_LABELS = {
    "z_fit": "Fit z", "z_neighbour": "Neighbour z", "z_pca": "PCA z", "z_history": "History z",
    "min_methods_high": "Methods for HIGH", "max_kink_fraction": "Max kinked share", "seasonal_z": "Seasonal z", "seasonal_window_days": "Seasonal window",
    "lookback_days": "Lookback days", "pca_components": "PCA components", "poly_degree": "Fit polynomial degree",
    "min_scale": "Minimum scale", "cooldown_minutes": "Alert cooldown", "stale_seconds": "Stale price seconds",
    "snapshot_minutes": "Snapshot interval", "history_years": "History years",
    "risk_per_trade": "Risk per trade", "risk_per_lot": "Risk per lot", "point_value": "Point value",
    "max_lots": "Max lots", "vol_stop_mult": "Volatility stop multiple", "vol_window": "Volatility window",
    "reversion_horizon": "Reversion horizon", "min_reversion_fraction": "Minimum reversion share",
    "min_reward_risk": "Minimum reward:risk", "hedge_count": "Hedges shown", "hedge_min_corr": "Minimum hedge correlation",
    "hedge_lookback": "Hedge lookback",
}


def _validate_hedge_types(raw) -> dict[str, list[str]]:
    """{kinked curve type: [hedge curve types]}; every kinked type needs at least one allowed hedge type."""
    chosen = {k: list(v) for k, v in DEFAULT_HEDGE_TYPES.items()} if raw is None else dict(raw)
    result = {}
    for family in DEFAULT_HEDGE_TYPES:
        allowed = chosen.get(family, DEFAULT_HEDGE_TYPES[family])
        bad = [f for f in allowed if f not in DEFAULT_HEDGE_TYPES]
        if bad:
            raise ValueError(f"Unknown hedge type(s) for {family}: {', '.join(bad)}")
        if not allowed:
            raise ValueError(f"Choose at least one hedge type for a {family} kink")
        result[family] = [f for f in DEFAULT_HEDGE_TYPES if f in allowed]
    return result


def validate_params(raw: dict) -> CurveParams:
    """Build CurveParams from form values (numbers may be strings), rejecting out-of-range input."""
    raw = {**asdict(CurveParams()), **raw}  # anything not supplied keeps its default
    values = {}
    for name, (low, high, integer) in _NUMERIC_BOUNDS.items():
        values[name] = parse_number(raw.get(name), _LABELS[name], minimum=low, maximum=high, integer=integer)
    priority = raw.get("alert_min_priority")
    if priority not in PRIORITIES:
        raise ValueError("Alert minimum priority must be LOW, MEDIUM or HIGH")
    values["alert_min_priority"] = priority
    values["alerts_enabled"] = bool(raw.get("alerts_enabled"))
    chosen = raw.get("alert_structures")
    chosen = list(ALERT_KEYS) if chosen is None else list(chosen)
    unknown = [key for key in chosen if key not in ALERT_KEYS]
    if unknown:
        raise ValueError(f"Unknown alert structure(s): {', '.join(unknown)}")
    values["alert_structures"] = [key for key in ALERT_KEYS if key in chosen]
    values["hedge_exclude_overlap"] = bool(raw.get("hedge_exclude_overlap"))
    values["hedge_types"] = _validate_hedge_types(raw.get("hedge_types"))
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
