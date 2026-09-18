"""Alert model: an in-app or Teams-dispatched risk alert."""

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class AlertLevel(str, Enum):
    """Severity level of an alert."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert(BaseModel):
    """A single alert, in-app and/or dispatched to external channels."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="UUID4, auto-generated."
    )
    level: AlertLevel = Field(..., description="Severity level of this alert.")
    title: str = Field(..., max_length=100, description="Short alert title.")
    body: str = Field(..., max_length=500, description="Full alert message body.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Auto-set to UTC now."
    )
    structure_id: str | None = Field(
        default=None, description="If alert is structure-specific, the structure's ID."
    )
    acknowledged: bool = Field(default=False, description="Whether the trader has acknowledged this alert.")
    channels_sent: list[str] = Field(
        default_factory=list, description="Channels the alert was sent to, e.g. ['in_app', 'teams']."
    )
