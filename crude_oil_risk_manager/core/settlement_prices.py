"""Daily settlement closes of outright contracts, for the analyses that need a long, gap-free history.

Source: the OI page's store (core.oi_data.OiStore): the /common/dailymarketdata/ settlement of every CL, BRN and gasoil (G)
outright, every trading day, from 2019. The per-contract price files read by core.data_loader.DataLoader hold
candles that only exist on days a contract traded, so a back-month outright can have a handful of rows; joining
several of those leaves almost no shared days. Settlements have no such holes.

BZ (NYMEX Brent) has no stored settlements. Its contract converges to the ICE Brent contract of the same month, so a BZ
outright is priced here from the BRN outright of that month, and the caller is told (`proxied`) so it can say so.
Anything else (WBS, a symbol that does not parse) returns None and the caller falls back to the DataLoader.
"""

import pandas as pd

from adapters.base import SymbolTranslator
from core.data_loader import DataLoader
from core.structure_utils import decompose_to_outrights

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
    if api.startswith("GO"):
        return "G", api, False
    return None


class SettlementPrices:
    """Settlement closes by internal outright symbol, read from the OI store (cached until the stored data changes)."""

    def __init__(self, store):
        self._store = store
        self._wide: dict[str, tuple[str, pd.DataFrame]] = {}

    @property
    def version(self) -> str:
        """Changes whenever the stored settlements change."""
        return self._store.version

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


class SettlementFirstLoader(DataLoader):
    """The app's one price-history source: official settlements where the store has them, stored candles otherwise.

    Every page that reads daily history through a DataLoader (the Correlation tab, the Monte Carlo and historical VaR, the
    New Structure builder's legs, the beta panels) gets a CL, BRN or G symbol's series from the settlements: an outright is its
    settlement; a spread, fly or condor is built from its outrights' settlements (the structure's PRICE on the dates every
    outright settled, as in core.correlation.structure_series), so a quoted fly has a settlement history on every trading
    day instead of only the days it printed. Anything else (WBS, BZ, or a CL/BRN/G contract the store does not hold) is
    read from the candles exactly as before, and a symbol is never half settlement and half candle. BZ is NOT proxied here:
    its basis to Brent is real risk, so it keeps its own candles (the builder's beta panel may still proxy it, and says so).

    Because `get_available_date_range` and `ensure_cached` read through the same method, a settlement-covered symbol counts
    as cached and never triggers a candle backfill.
    """

    def __init__(self, data_dir: str, historical_adapter, settlements: SettlementPrices):
        super().__init__(data_dir, historical_adapter)
        self._settlements = settlements
        self._derived: dict[str, tuple[str, pd.Series | None]] = {}

    def _from_settlements(self, symbol: str) -> pd.Series | None:
        try:
            parts = decompose_to_outrights(symbol, 1.0)
        except ValueError:
            return None
        levels = []
        for outright, weight in parts.items():
            series, proxied = self._settlements.closes(outright)
            if series is None or proxied:
                return None
            levels.append(series * weight)
        price = pd.concat(levels, axis=1, join="inner").dropna().sum(axis=1).sort_index()
        if len(price) < 2:
            return None
        price.name = symbol
        return price

    def _derived_series(self, symbol: str) -> pd.Series | None:
        version = self._settlements.version
        cached = self._derived.get(symbol)
        if cached is None or cached[0] != version:
            if len(self._derived) > 600:
                self._derived.clear()
            self._derived[symbol] = (version, self._from_settlements(symbol))
        return self._derived[symbol][1]

    def covered_by_settlements(self, symbol: str) -> bool:
        """True if this symbol's history is read from the stored settlements, so its candle file is never used."""
        return self._derived_series(symbol) is not None

    def _read_series(self, symbol: str) -> pd.Series | None:
        derived = self._derived_series(symbol)
        return derived if derived is not None else super()._read_series(symbol)
