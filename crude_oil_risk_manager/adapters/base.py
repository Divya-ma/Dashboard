"""Abstract adapter interfaces, symbol translation, and shared exceptions.

This module is the single source of truth for:
- The internal product code <-> vendor API product code mapping.
- Futures month letter <-> month number mapping.
- Symbol translation between internal and API symbol formats
  (outright, spread, fly, condor).
- The LiveDataAdapter / HistoricalDataAdapter abstract interfaces that all
  concrete adapters (vendor, mock) must implement.
- The API error hierarchy raised by concrete adapters.

Nothing in this module performs network I/O or touches the database.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from core.models.leg import Leg

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Product code translation (single source of truth)
# ----------------------------------------------------------------------

PRODUCT_TO_API_CODE: dict[str, str] = {
    "CL": "CL",
    "BRN": "CO",
    "BZ": "BZZ",
    "WBS": "WTCL",
    "G": "GO",
}

API_CODE_TO_PRODUCT: dict[str, str] = {v: k for k, v in PRODUCT_TO_API_CODE.items()}

# ----------------------------------------------------------------------
# Futures month letter translation
# ----------------------------------------------------------------------

MONTH_TO_LETTER: dict[int, str] = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

LETTER_TO_MONTH: dict[str, int] = {v: k for k, v in MONTH_TO_LETTER.items()}


class SymbolTranslator:
    """Stateless translation between internal and vendor API symbol formats."""

    @staticmethod
    def _match_product_prefix(symbol: str, code_map: dict[str, str]) -> tuple[str, str]:
        """Return (matched_code, remainder) for the longest matching prefix in code_map."""
        for code in sorted(code_map, key=len, reverse=True):
            if symbol.startswith(code):
                return code, symbol[len(code):]
        raise ValueError(f"product code not recognized in symbol {symbol!r}")

    @staticmethod
    def _month_year_token(contract_month: int, contract_year: int) -> str:
        return f"{MONTH_TO_LETTER[contract_month]}{contract_year % 100:02d}"

    @classmethod
    def _internal_symbol_from_contract(cls, contract) -> str:
        """Build the internal outright symbol (product + month letter + 2-digit year)."""
        return f"{contract.product}{cls._month_year_token(contract.contract_month, contract.contract_year)}"

    @staticmethod
    def internal_to_api(symbol: str) -> str:
        """Convert an internal outright symbol to its API symbol.

        Example: "BRNZ26" -> "COZ26", "CLZ26" -> "CLZ26".
        Raises ValueError if the product code is not recognized.
        """
        product, rest = SymbolTranslator._match_product_prefix(symbol, PRODUCT_TO_API_CODE)
        api_code = PRODUCT_TO_API_CODE[product]
        return f"{api_code}{rest}"

    @staticmethod
    def api_to_internal(api_symbol: str) -> str:
        """Convert an API outright symbol back to its internal symbol."""
        api_code, rest = SymbolTranslator._match_product_prefix(api_symbol, API_CODE_TO_PRODUCT)
        product = API_CODE_TO_PRODUCT[api_code]
        return f"{product}{rest}"

    @staticmethod
    def structure_to_api_symbol(legs: list[Leg]) -> str:
        """Build the API symbol for a structure from its ordered legs.

        1 leg -> outright, 2 -> spread, 3 -> fly, 4 -> condor.
        Subsequent legs omit the product code when they share the front
        leg's product. Raises NotImplementedError for cross-product spreads
        (format not yet confirmed) and for structures with >4 legs.
        """
        leg_count = len(legs)
        if leg_count == 0:
            raise ValueError("cannot build an API symbol for a structure with no legs")

        products = {leg.contract.product for leg in legs}
        front = legs[0].contract
        if front.product not in PRODUCT_TO_API_CODE:
            raise ValueError(f"product code not recognized: {front.product!r}")
        front_symbol = (
            f"{PRODUCT_TO_API_CODE[front.product]}"
            f"{SymbolTranslator._month_year_token(front.contract_month, front.contract_year)}"
        )

        if leg_count == 1:
            return front_symbol

        if leg_count == 2:
            SymbolTranslator._warn_if_nonstandard_ratios(legs, expected={1, -1}, structure_name="spread")
            if len(products) > 1:
                # TODO: cross-product spread API symbol format has not been confirmed
                # by the vendor yet. Implement once the format is provided.
                raise NotImplementedError(
                    "Cross-product spread API symbol format not yet confirmed. "
                    "Will be implemented when format is provided."
                )
            second_token = SymbolTranslator._month_year_token(
                legs[1].contract.contract_month, legs[1].contract.contract_year
            )
            return f"{front_symbol}-{second_token}"

        if leg_count == 3:
            SymbolTranslator._warn_if_nonstandard_ratios(
                legs, expected_multiset=[1, -2, 1], structure_name="fly"
            )
            if len(products) > 1:
                raise NotImplementedError(
                    "Cross-product fly API symbol format not yet confirmed. "
                    "Will be implemented when format is provided."
                )
            tokens = [
                SymbolTranslator._month_year_token(leg.contract.contract_month, leg.contract.contract_year)
                for leg in legs[1:]
            ]
            return f"{front_symbol}-{tokens[0]}-{tokens[1]}"

        if leg_count == 4:
            if len(products) > 1:
                raise NotImplementedError(
                    "Cross-product condor API symbol format not yet confirmed. "
                    "Will be implemented when format is provided."
                )
            tokens = [
                SymbolTranslator._month_year_token(leg.contract.contract_month, leg.contract.contract_year)
                for leg in legs[1:]
            ]
            return f"{front_symbol}-{tokens[0]}-{tokens[1]}+{tokens[2]}"

        raise NotImplementedError(
            "Custom structures with >4 legs have no exchange-quoted symbol. "
            "Derive PnL from individual outright legs instead."
        )

    @staticmethod
    def _warn_if_nonstandard_ratios(
        legs: list[Leg],
        structure_name: str,
        expected: set[int] | None = None,
        expected_multiset: list[int] | None = None,
    ) -> None:
        ratios = [leg.ratio for leg in legs]
        if expected is not None and set(ratios) != expected:
            logger.warning(
                "Non-standard ratios %s for %s structure (expected one of %s)",
                ratios, structure_name, sorted(expected),
            )
        elif expected_multiset is not None and sorted(ratios) != sorted(expected_multiset):
            logger.warning(
                "Non-standard ratios %s for %s structure (expected %s)",
                ratios, structure_name, sorted(expected_multiset),
            )

    @staticmethod
    def parse_api_symbol(api_symbol: str) -> list[dict]:
        """Parse an API symbol back into a list of leg dicts.

        Returns e.g. [{"product": "CL", "month": "Z", "year": "26"}, ...].
        """
        import re

        api_code, rest = SymbolTranslator._match_product_prefix(api_symbol, API_CODE_TO_PRODUCT)
        product = API_CODE_TO_PRODUCT[api_code]
        tokens = [t for t in re.split(r"[-+]", rest) if t]
        legs = []
        for token in tokens:
            month_letter, year = token[0], token[1:]
            if month_letter not in LETTER_TO_MONTH:
                raise ValueError(f"unrecognized month letter {month_letter!r} in symbol {api_symbol!r}")
            legs.append({"product": product, "month": month_letter, "year": year})
        return legs


# ----------------------------------------------------------------------
# API error hierarchy
# ----------------------------------------------------------------------


class APIError(Exception):
    """Base exception for all vendor API errors."""


class AuthenticationError(APIError):
    """Raised when the API rejects the access token (HTTP 401)."""


class RateLimitError(APIError):
    """Raised when the API rate limit is exceeded (HTTP 429)."""


class APIServerError(APIError):
    """Raised when the API returns a server error (HTTP 5xx)."""


class APITimeoutError(APIError):
    """Raised when a request to the API times out."""


class APIConnectionError(APIError):
    """Raised when a connection to the API cannot be established."""


# ----------------------------------------------------------------------
# Live price data contract
# ----------------------------------------------------------------------


@dataclass
class LivePrice:
    """A single live price observation for an internal symbol."""

    symbol: str
    price: float
    timestamp: datetime
    is_stale: bool
    raw_open: float
    raw_high: float
    raw_low: float
    raw_volume: float


# ----------------------------------------------------------------------
# Abstract adapter interfaces
# ----------------------------------------------------------------------


class LiveDataAdapter(ABC):
    """Abstract interface for fetching live (most-recent-candle) prices."""

    @abstractmethod
    def get_live_prices(self, symbols: list[str]) -> dict[str, "LivePrice"]:
        """Return a map of internal_symbol -> LivePrice for the given internal symbols.

        Implementations must translate symbols to API format internally and
        never expose API codes to callers.
        """
        raise NotImplementedError

    @abstractmethod
    def get_access_token(self) -> str:
        """Return the current Bearer token used to authenticate API requests."""
        raise NotImplementedError


class HistoricalDataAdapter(ABC):
    """Abstract interface for fetching historical OHLC data."""

    @abstractmethod
    def get_ohlc(
        self,
        symbol: str,
        interval: str,
        start: datetime | None,
        end: datetime | None,
        count: int | None,
    ) -> pd.DataFrame:
        """Return a DataFrame [symbol, timestamp, open, high, low, close, volume]
        for a single internal symbol. timestamp is UTC datetime; symbol is internal.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ohlc_bulk(
        self,
        symbols: list[str],
        interval: str,
        start: datetime | None,
        end: datetime | None,
        count: int | None,
    ) -> dict[str, pd.DataFrame]:
        """Return a map of internal_symbol -> DataFrame, batching requests to
        respect the 50-instrument-per-request API limit.
        """
        raise NotImplementedError
