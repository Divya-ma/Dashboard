"""Trade model: a single trade event applied to a leg."""

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator

VALID_DIRECTIONS = {"buy", "sell"}


class TradeEventType(str, Enum):
    """The kind of trade event being recorded."""

    TRADE = "trade"
    ADD = "add"
    PARTIAL_EXIT = "partial_exit"
    FULL_EXIT = "full_exit"
    ROLL = "roll"


class Trade(BaseModel):
    """A single execution event against a leg of a structure."""

    trade_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="UUID4, auto-generated."
    )
    structure_id: str = Field(..., description="ID of the structure this trade belongs to.")
    leg_id: str = Field(..., description="ID of the leg this trade belongs to.")
    event_type: TradeEventType = Field(..., description="The kind of trade event.")
    lots: float = Field(..., description="Number of lots in this trade. Must be > 0.")
    price: float = Field(..., description="Execution price.")
    direction: str = Field(..., description="'buy' or 'sell'.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Auto-set to UTC now."
    )
    realized_pnl: float | None = Field(
        default=None, description="Populated only on exit/roll events."
    )
    notes: str = Field(default="", max_length=200, description="Optional trade notes.")

    @field_validator("lots")
    @classmethod
    def validate_lots(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"lots must be > 0, got {v}")
        return v

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        if v not in VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(VALID_DIRECTIONS)}, got {v!r}")
        return v
