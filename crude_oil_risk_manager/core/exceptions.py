"""Application-level (domain) exceptions.

Adapter/API errors live in adapters/base.py; the exceptions here describe
domain failures that analysis modules and the UI need to distinguish.
"""


class CrudeOilRiskError(Exception):
    """Base exception for all domain errors."""


class DataNotAvailableError(CrudeOilRiskError):
    """Raised when required price data is not available locally and cannot be fetched.

    The message should include the symbol and the requested date range.
    """


class InsufficientDataError(CrudeOilRiskError):
    """Raised when data exists but has too few rows for the requested calculation.

    The message should include the symbol, the available row count, and the
    required minimum.
    """


class InvalidSymbolError(CrudeOilRiskError):
    """Raised when a symbol cannot be translated or is not recognized."""


class StaleDataError(CrudeOilRiskError):
    """Raised when data is older than the staleness threshold and the caller
    has requested strict freshness.
    """
