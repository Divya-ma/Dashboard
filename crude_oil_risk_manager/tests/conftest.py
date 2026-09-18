"""Shared pytest fixtures for the crude oil risk manager test suite.

Fixtures will be updated in Phase 1.2 when models are defined.
"""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a temporary SQLite database file path for use in tests."""
    return tmp_path / "test_crude_oil_risk.db"


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
