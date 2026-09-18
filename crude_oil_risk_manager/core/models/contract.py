"""Contract model: a single exchange-quoted futures contract specification."""

from datetime import date

from pydantic import BaseModel, Field, field_validator

VALID_PRODUCTS = {"CL", "CO", "BRN", "BZ", "BZZ", "WBS", "WTCL", "G", "GO", "LGO"}

_MONTH_ABBREVIATIONS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


class Contract(BaseModel):
    """A single monthly futures contract, entered and spec'd via the UI."""

    product: str = Field(
        ..., description="Product code, one of VALID_PRODUCTS (e.g. 'CL', 'BRN')."
    )
    contract_month: int = Field(
        ..., description="Delivery month of the contract, 1-12."
    )
    contract_year: int = Field(
        ..., description="Delivery year of the contract, e.g. 2025."
    )
    symbol: str = Field(
        ...,
        description=(
            "Exchange-quoted symbol, e.g. 'CLZ25'. Unique identifier used in all "
            "API calls. User-entered; not computed."
        ),
    )
    multiplier: float = Field(
        ..., description="Contract multiplier in USD per point (e.g. 1000 for CL). User-entered."
    )
    tick_size: float = Field(
        ..., description="Minimum price increment (e.g. 0.01 for CL). User-entered."
    )
    tick_value: float = Field(
        ..., description="Dollar value of one tick move (e.g. 10.0 for CL). User-entered."
    )
    currency: str = Field(
        default="USD", description="Contract currency. Default USD; GBP for ICE Gasoil pence products."
    )
    expiry_date: date | None = Field(
        default=None, description="Contract expiry date, when known. Used for roll warnings."
    )
    first_notice_date: date | None = Field(
        default=None, description="First notice date, when known. Used for roll warnings."
    )

    @field_validator("product")
    @classmethod
    def validate_product(cls, v: str) -> str:
        if v not in VALID_PRODUCTS:
            raise ValueError(f"product must be one of {sorted(VALID_PRODUCTS)}, got {v!r}")
        return v

    @field_validator("contract_month")
    @classmethod
    def validate_contract_month(cls, v: int) -> int:
        if not 1 <= v <= 12:
            raise ValueError(f"contract_month must be between 1 and 12, got {v}")
        return v

    @field_validator("contract_year")
    @classmethod
    def validate_contract_year(cls, v: int) -> int:
        if not 2020 <= v <= 2050:
            raise ValueError(f"contract_year must be between 2020 and 2050, got {v}")
        return v

    @field_validator("multiplier")
    @classmethod
    def validate_multiplier(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"multiplier must be > 0, got {v}")
        return v

    @field_validator("tick_size")
    @classmethod
    def validate_tick_size(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"tick_size must be > 0, got {v}")
        return v

    @field_validator("tick_value")
    @classmethod
    def validate_tick_value(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"tick_value must be > 0, got {v}")
        return v

    @property
    def display_name(self) -> str:
        """Human-readable name, e.g. 'CL Dec 2025 (CLZ25)'."""
        month_abbr = _MONTH_ABBREVIATIONS[self.contract_month]
        return f"{self.product} {month_abbr} {self.contract_year} ({self.symbol})"

    @classmethod
    def from_symbol(cls, symbol: str) -> "Contract":
        """Intentionally unimplemented: contract spec entry is UI-driven, not API-pulled."""
        raise NotImplementedError(
            "Contracts must be created via the UI contract entry form. Symbol "
            "lookup is not implemented — the adapter layer will handle this "
            "when the API is connected."
        )
