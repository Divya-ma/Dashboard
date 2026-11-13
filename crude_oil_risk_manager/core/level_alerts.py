"""Price-level alerts: pick a structure (by its legs), type a price level, and get an alert when its price gets there.

A level alert is NOT a position: it is only the legs, a level and a direction, kept in the `level_alerts` table and shown on the
Price Alerts page. The structure's price is the same composite the Structures page shows (sign of the first leg's ratio times the
sum of ratio x leg price). Each alert fires ONCE: the moment the live price is at or beyond the level in its direction it is marked
triggered (so a flickering price cannot send it twice), one alert is sent (in the app and to Teams, like every other alert) and
it stays on the page as a triggered alert until it is deleted.

DIRECTION. "above" fires at or over the level, "below" at or under it. "auto" (the default on the page) looks at the structure's
price when the alert is made: a level above it is "above", a level below it is "below", and a level equal to it is "above".

STOP / START. An active alert can be stopped (PAUSED): it stays in the list, keeps its level and legs, but is skipped by
check_level_alerts and dropped from the live feed's leg list (adapters.live.fairvalue), so it cannot fire while paused and costs
nothing. Start makes it active again. A triggered alert is already done; Edit, not Start, is what arms it again.
"""

import logging
import math
from core.exceptions import CrudeOilRiskError
from core.models import AlertLevel
from core.models.level_alert import ACTIVE, PAUSED, TRIGGERED, LevelAlert  # noqa: F401 - re-exported for the page
from core.structure_builder import StructureBuildError, build_shell_structure
from core.structure_utils import compose_structure_symbol

logger = logging.getLogger(__name__)

ABOVE, BELOW, AUTO = "above", "below", "auto"
DIRECTION_LABELS = {ABOVE: "rises to / above", BELOW: "falls to / below"}
NAME_LIMIT = 100


class LevelAlertError(CrudeOilRiskError):
    """Blocking problems with a new level alert; `errors` are safe to show the user."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def level_price(legs: list[dict], prices: dict[str, float]) -> float | None:
    """The structure's composite price from outright leg prices; None if any leg has no price (same maths as
    core.structure_view.structure_live_price)."""
    total = 0.0
    for leg in legs:
        price = prices.get(leg["symbol"])
        if price is None:
            return None
        total += leg["ratio"] * price
    return total if legs and legs[0]["ratio"] > 0 else -total


def is_hit(direction: str, level: float, price: float) -> bool:
    return price >= level if direction == ABOVE else price <= level


def default_name(legs: list[dict]) -> str:
    """The exchange-style symbol of the legs ("BRNJ27-K27-M27"), else the leg symbols joined."""
    symbols, ratios = [leg["symbol"] for leg in legs], [leg["ratio"] for leg in legs]
    return (compose_structure_symbol(symbols, ratios) or " / ".join(f"{r:+g}x{s}" for s, r in zip(symbols, ratios)))[:NAME_LIMIT]


def build_level_alert(name, template, symbols, ratios, level, direction, prices, get_contract):
    """Validate the form and return (LevelAlert, new_contracts, warnings); raises LevelAlertError listing every problem.

    The legs are checked like the structure builder checks them (valid symbols, whole non-zero ratios, the template's leg count),
    and any contract the app has not seen is returned in `new_contracts` so the caller can save it: the live price poll covers
    every saved contract, which is what keeps this alert's price coming in. `prices` ({symbol: price}) is used for the direction
    "auto" and to record the price at creation."""
    errors: list[str] = []
    name = (name or "").strip()
    if len(name) > NAME_LIMIT:
        errors.append(f"The name is limited to {NAME_LIMIT} characters.")
    try:
        level = float(str(level).replace(",", "")) if level not in (None, "") else None
    except ValueError:
        level = None
    if level is None or not math.isfinite(level):
        errors.append("Type the price level as a number.")
    if direction not in (ABOVE, BELOW, AUTO):
        errors.append("Choose when the alert should fire.")
    built = None
    try:
        built = build_shell_structure(name or "alert", template, symbols, ratios, None, None, None, "", get_contract)
    except StructureBuildError as exc:
        errors.extend(exc.errors)
    if errors or built is None:
        raise LevelAlertError(errors)
    legs = [{"symbol": leg.contract.symbol, "ratio": leg.ratio} for leg in built.structure.legs]
    current = level_price(legs, prices or {})
    if direction == AUTO:
        if current is None:
            raise LevelAlertError(["There is no live price for this structure yet, so choose 'rises to / above' or 'falls to / below'."])
        direction = ABOVE if level >= current else BELOW
    warnings = list(built.warnings)
    if current is not None and is_hit(direction, level, current):
        warnings.append(f"The price is already {current:g}, so this alert fires on the next price check.")
    alert = LevelAlert(name=name or default_name(legs), legs=legs, level=level, direction=direction, created_price=current)
    return alert, built.new_contracts, warnings


def check_level_alerts(repository, alert_manager, prices: dict[str, float]) -> list:
    """Fire every active level alert whose price has reached its level; returns the alerts sent. Never raises.

    The alert is marked triggered BEFORE it is sent, and only the call that changed the row sends, so two checks that overlap
    (two browser tabs, two polls) cannot send it twice."""
    try:
        active = repository.get_level_alerts(ACTIVE)
    except Exception as exc:  # noqa: BLE001 - a broken alert table must never stop the price checks
        logger.error("Could not read the level alerts: %s", type(exc).__name__)
        return []
    sent = []
    for alert in active:
        price = level_price(alert.legs, prices)
        if price is None or not is_hit(alert.direction, alert.level, price):
            continue
        try:
            if not repository.mark_level_alert_triggered(alert.alert_id, price):
                continue  # another check got there first, or it was deleted a moment ago
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not mark level alert %s as triggered: %s", alert.alert_id, type(exc).__name__)
            continue
        verb = "reached" if alert.direction == ABOVE else "fallen to"
        sent.append(alert_manager.send_alert(
            AlertLevel.WARNING,
            f"🔔 Price level hit: {alert.name}",
            f"{alert.name} has {verb} your level {alert.level:g} (live price {price:g}; alert set {DIRECTION_LABELS[alert.direction]} {alert.level:g}).",
        ))
    return sent

