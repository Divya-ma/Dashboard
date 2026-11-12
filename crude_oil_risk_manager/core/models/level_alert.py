"""A price-level alert as stored (see core.level_alerts for how it is built, priced and fired)."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

ACTIVE, TRIGGERED = "active", "triggered"


@dataclass
class LevelAlert:
    name: str
    legs: list[dict]  # [{"symbol": "BRNJ27-K27-M27", "ratio": 1}, ...]
    level: float
    direction: str  # "above" (fires at or over the level) | "below" (at or under it)
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_price: float | None = None
    triggered_at: datetime | None = None
    triggered_price: float | None = None

    @property
    def symbols(self) -> list[str]:
        return [leg["symbol"] for leg in self.legs]
