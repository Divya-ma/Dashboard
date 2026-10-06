"""Rate-limited client and per-endpoint wrappers for the QH API."""

from typing import Callable

from adapters.qh_api.api import ENDPOINTS, QHApi, list_endpoints
from adapters.qh_api.client import QHClient

__all__ = ["ENDPOINTS", "QHApi", "QHClient", "list_endpoints", "make_api"]


def make_api(token_provider: Callable[[], str]) -> QHApi:
    """A QHApi (with its own rate limiters) reading the Bearer token from `token_provider`.

    The app builds ONE of these and shares it, so every caller draws on the same per-endpoint
    budgets; adapters only build their own when none is passed in (tests, standalone use).
    """
    return QHApi(QHClient(token_provider=token_provider))
