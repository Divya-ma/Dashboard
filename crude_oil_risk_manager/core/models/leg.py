"""Leg model: a single leg of a structure, embedding its Contract."""

import uuid

from pydantic import BaseModel, Field, field_validator

from core.models.contract import Contract


class Leg(BaseModel):
    """A single leg of a Structure (e.g. one side of a spread or fly)."""

    leg_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="UUID4, auto-generated if not provided."
    )
    contract: Contract = Field(..., description="The full embedded Contract object for this leg.")
    ratio: int = Field(
        ...,
        description=(
            "Signed integer leg ratio: +1/-1 for spread, +1/-2/+1 for fly, any "
            "non-zero integer for custom structures."
        ),
    )
    lots: float = Field(
        default=0.0, description="Number of lots traded. 0.0 means shell (not yet traded)."
    )
    entry_price: float | None = Field(
        default=None, description="Execution entry price. None means no trade entered yet (shell state)."
    )
    average_entry_price: float | None = Field(
        default=None,
        description="Updated when adding to a position. Initially same as entry_price.",
    )
    is_naked: bool = Field(
        default=False,
        description=(
            "True when a partial exit leaves this leg without its counterpart "
            "legs. Set explicitly by trade exit logic."
        ),
    )

    @field_validator("ratio")
    @classmethod
    def validate_ratio(cls, v: int) -> int:
        if v == 0:
            raise ValueError("ratio must be non-zero")
        return v

    @field_validator("lots")
    @classmethod
    def validate_lots(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"lots must be >= 0, got {v}")
        return v

    @property
    def is_traded(self) -> bool:
        """True if this leg has an active traded position."""
        return self.lots > 0 and self.entry_price is not None

    @property
    def notional_value(self) -> float | None:
        """lots * contract.multiplier * entry_price, or None if not traded."""
        if not self.is_traded:
            return None
        return self.lots * self.contract.multiplier * self.entry_price
