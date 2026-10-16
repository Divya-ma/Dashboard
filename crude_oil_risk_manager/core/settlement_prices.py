"""Daily settlement closes of outright contracts, for the analyses that need a long, gap-free history.

Source: the OI page's store (core.oi_data.OiStore): the /common/dailymarketdata/ settlement of every CL and BRN
outright, every trading day, from 2019. The per-contract price files read by core.data_loader.DataLoader hold
candles that only exist on days a contract traded, so a back-month outright can have a handful of rows; joining
several of those leaves almost no shared days. Settlements have no such holes.

BZ (NYMEX Brent) has no stored settlements. Its contract converges to the ICE Brent contract of the same month, so a BZ
outright is priced here from the BRN outright of that month, and the caller is told (`proxied`) so it can say so.
Anything else (WBS, G, a symbol that does not parse) returns None and the caller falls back to the DataLoader.
"""

import pandas as pd

from adapters.base import SymbolTranslator

PROXY_NOTE = "BZ legs are priced from the ICE Brent settlement of the same month (BZ has no stored settlements)."


def settlement_code(symbol: str) -> tuple[str, str, bool] | None:
    """(OI store product, API contract code, priced from a proxy) for an internal OUTRIGHT symbol, or None."""
    if not symbol or "-" in symbol:
        return None
    try:
        api = SymbolTranslator.internal_to_api(symbol)
    except ValueError:
        return None
    if api.startswith("CL"):
        return "CL", api, False
    if api.startswith("BZZ"):
        return "BRN", "CO" + api[3:], True
    if api.startswith("CO"):
        return "BRN", api, False
    return None


class SettlementPrices:
    """Settlement closes by internal outright symbol, read from the OI store (cached until the stored data changes)."""

    def __init__(self, store):
        self._store = store
        self._wide: dict[str, tuple[str, pd.DataFrame]] = {}

    def _matrix(self, product: str) -> pd.DataFrame:
        version = self._store.version
        cached = self._wide.get(product)
        if cached is None or cached[0] != version:
            frame = self._store.frame(product)
            wide = frame.pivot(index="date", columns="contract", values="close").sort_index() if not frame.empty else pd.DataFrame()
            self._wide[product] = (version, wide)
        return self._wide[product][1]

    def closes(self, symbol: str) -> tuple[pd.Series | None, bool]:
        """(daily settlement closes indexed by UTC midnight as the DataLoader's are, whether a proxy was used)."""
        code = settlement_code(symbol)
        if code is None:
            return None, False
        product, api, proxied = code
        wide = self._matrix(product)
        if api not in wide.columns:
            return None, False
        series = wide[api].dropna()
        if len(series) < 2:
            return None, False
        series = series.copy()
        series.index = pd.DatetimeIndex(series.index).tz_localize("UTC")
        series.name = symbol
        return series, proxied
