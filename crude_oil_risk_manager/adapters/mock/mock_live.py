"""Mock live data adapter: deterministic fake prices for local development and testing."""

import random
from datetime import datetime, timezone

from adapters.base import LiveDataAdapter, LivePrice

_DEFAULT_BASE_PRICES: dict[str, float] = {
    "CLZ26": 75.50,
    "COZ26": 78.20,
    "BZZZ26": 78.35,
    "WTCLZ26": 74.90,
    "GOZ26": 720.00,
}

_RANDOM_WALK_STEP = 0.05
_DEFAULT_SEED = 42


class MockLiveAdapter(LiveDataAdapter):
    """Deterministic in-memory live price adapter for tests and local dev."""

    def __init__(self, base_prices: dict[str, float] | None = None, seed: int = _DEFAULT_SEED):
        self._base_prices = dict(base_prices) if base_prices is not None else dict(_DEFAULT_BASE_PRICES)
        self._random = random.Random(seed)

    def get_access_token(self) -> str:
        """Return a fixed mock token; no real authentication is performed."""
        return "mock_token"

    def get_live_prices(self, symbols: list[str]) -> dict[str, LivePrice]:
        """Return deterministic LivePrice objects for the given internal symbols."""
        now = datetime.now(timezone.utc)
        results: dict[str, LivePrice] = {}
        for symbol in symbols:
            base_price = self._base_prices.get(symbol, 75.00)
            walk = self._random.uniform(-_RANDOM_WALK_STEP, _RANDOM_WALK_STEP)
            close = round(base_price + walk, 2)
            spread = _RANDOM_WALK_STEP
            results[symbol] = LivePrice(
                symbol=symbol,
                price=close,
                timestamp=now,
                is_stale=False,
                raw_open=round(close - spread / 2, 2),
                raw_high=round(close + spread, 2),
                raw_low=round(close - spread, 2),
                raw_volume=1000.0,
            )
        return results
