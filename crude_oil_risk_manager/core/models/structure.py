"""Structure model: an outright, spread, fly, or custom combination of legs."""

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from core.models.leg import Leg


class StructureType(str, Enum):
    """The shape of a structure's legs."""

    OUTRIGHT = "outright"
    SPREAD = "spread"
    FLY = "fly"
    CUSTOM = "custom"


class StructureStatus(str, Enum):
    """Lifecycle status of a structure."""

    SHELL = "shell"
    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"


class Structure(BaseModel):
    """A traded or shell structure: one or more legs combined by a defined ratio."""

    structure_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="UUID4, auto-generated."
    )
    name: str = Field(
        ..., max_length=100, description="User-defined name, e.g. 'CL Dec25-Jan26 Spread'."
    )
    structure_type: StructureType = Field(..., description="The shape of this structure's legs.")
    products: list[str] = Field(
        ..., description="Product codes involved, e.g. ['CL'] or ['CL', 'BRN'] for cross-product."
    )
    legs: list[Leg] = Field(..., min_length=1, description="Ordered list of legs; minimum 1 leg.")
    status: StructureStatus = Field(
        default=StructureStatus.SHELL, description="Lifecycle status of this structure."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Auto-set to UTC now if not provided.",
    )
    last_modified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Auto-set to UTC now; updated on every change.",
    )
    notes: str = Field(default="", max_length=500, description="Optional trader notes.")
    close_trigger: str | None = Field(
        default=None, description="'manual' | 'alert' | None. Set on full close."
    )
    closed_at: datetime | None = Field(default=None, description="Timestamp of full close, if closed.")
    has_duplicate_tenors: bool = Field(
        default=False,
        description="Set by validator if duplicate contract symbols exist across legs.",
    )

    @field_validator("legs")
    @classmethod
    def validate_legs(cls, v: list[Leg]) -> list[Leg]:
        for leg in v:
            if leg.ratio == 0:
                raise ValueError("no leg may have ratio == 0")
        return v

    @model_validator(mode="after")
    def check_duplicate_tenors(self) -> "Structure":
        symbols = [leg.contract.symbol for leg in self.legs]
        if len(symbols) != len(set(symbols)):
            self.has_duplicate_tenors = True
        return self

    @property
    def leg_count(self) -> int:
        """Number of legs in this structure."""
        return len(self.legs)

    @property
    def is_cross_product(self) -> bool:
        """True if more than one unique product is involved across legs."""
        return len({leg.contract.product for leg in self.legs}) > 1

    @property
    def days_held(self) -> int | None:
        """Days since created_at if status != SHELL, else None."""
        if self.status == StructureStatus.SHELL:
            return None
        return (datetime.now(timezone.utc) - self.created_at).days

    @property
    def net_lots(self) -> float:
        """Sum of (leg.ratio * leg.lots) across all legs."""
        return sum(leg.ratio * leg.lots for leg in self.legs)
