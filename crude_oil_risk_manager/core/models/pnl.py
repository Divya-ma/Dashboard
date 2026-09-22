"""PnLRecord model: a point-in-time snapshot of a structure's P&L (data contract only)."""

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator

_PNL_TOLERANCE = 1e-6


class PnLRecord(BaseModel):
    """A single P&L snapshot for a structure at a point in time."""

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="UUID4, auto-generated."
    )
    structure_id: str = Field(..., description="ID of the structure this record belongs to.")
    unrealized_pnl: float = Field(..., description="Unrealized P&L at this snapshot.")
    realized_pnl: float = Field(..., description="Realized P&L at this snapshot.")
    transaction_costs: float = Field(
        default=0.0,
        description=(
            "Total transaction cost incurred so far (sum of every trade's transaction_cost). "
            "Already netted out of total_pnl."
        ),
    )
    total_pnl: float = Field(
        ..., description="Must equal unrealized_pnl + realized_pnl - transaction_costs."
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Auto-set to UTC now."
    )
    last_price_used: dict[str, float] = Field(
        default_factory=dict, description="Map of symbol to price used in this calculation."
    )
    is_stale: bool = Field(
        default=False, description="True if any price used was flagged stale."
    )
    stale_symbols: list[str] = Field(
        default_factory=list, description="Which symbols were stale."
    )

    @model_validator(mode="after")
    def validate_total_pnl(self) -> "PnLRecord":
        expected = self.unrealized_pnl + self.realized_pnl - self.transaction_costs
        if abs(self.total_pnl - expected) > _PNL_TOLERANCE:
            raise ValueError(
                f"total_pnl ({self.total_pnl}) must equal unrealized_pnl + realized_pnl - "
                f"transaction_costs ({expected})"
            )
        return self
