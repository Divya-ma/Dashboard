"""Shared pytest fixtures for the crude oil risk manager test suite.

Fixtures will be updated in Phase 1.2 when models are defined.
"""

import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adapters.base import API_CODE_TO_PRODUCT, SymbolTranslator


@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a temporary SQLite database file path for use in tests."""
    return tmp_path / "test_crude_oil_risk.db"


@pytest.fixture
def create_synthetic_parquet():
    """Factory that writes a synthetic daily-OHLC Parquet file at the correct path.

    Usage: create_synthetic_parquet(tmp_path, "CLZ26", n_days=120, start_price=75.0,
    daily_vol=0.02, seed=42). Pass `closes=` to supply an explicit close array
    (n_days is then taken from its length). Returns the written Path.
    """

    def _create(
        tmp_path,
        symbol: str,
        n_days: int = 120,
        start_price: float = 75.0,
        daily_vol: float = 0.02,
        seed: int = 42,
        closes: np.ndarray | None = None,
        end_date: date = date(2026, 6, 30),
    ) -> Path:
        if closes is None:
            rng = np.random.default_rng(seed)
            steps = rng.normal(0.0, daily_vol * start_price, n_days)
            closes = start_price + np.cumsum(steps)
        closes = np.asarray(closes, dtype=float)

        timestamps = pd.date_range(end=end_date, periods=len(closes), freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "symbol": symbol,
                "timestamp": timestamps,
                "open": closes,
                "high": closes + 0.5,
                "low": closes - 0.5,
                "close": closes,
                "volume": 1000.0,
            }
        )
        api_symbol = SymbolTranslator.internal_to_api(symbol)
        api_code, _ = SymbolTranslator._match_product_prefix(api_symbol, API_CODE_TO_PRODUCT)
        path = Path(tmp_path) / api_code / f"{api_symbol}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    return _create


@pytest.fixture
def sample_contract():
    """Return a placeholder Contract-shaped dict (CL, Dec 2025).

    Will be replaced with a Contract model instance once core.models
    defines it in Phase 1.2.
    """
    return {
        "symbol": "CL",
        "contract_month": "Z25",
        "expiry_date": "2025-11-20",
        "exchange": "NYMEX",
        "multiplier": 1000,
        "tick_size": 0.01,
        "tick_value": 10.0,
    }


@pytest.fixture
def sample_structure():
    """Return a placeholder Structure-shaped dict.

    Will be replaced with a Structure model instance once core.models
    defines it in Phase 1.2.
    """
    return {
        "name": "CL Dec25/Jan26 Calendar Spread",
        "legs": [
            {"contract": "CLZ25", "side": "BUY", "quantity": 10},
            {"contract": "CLF26", "side": "SELL", "quantity": 10},
        ],
    }
