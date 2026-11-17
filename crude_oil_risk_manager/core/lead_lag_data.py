"""Glue between the Correlation page's Lead-Lag view and core.lead_lag: which contracts go in a group's matrix,
and turning their intraday candles into the price-difference series core.lead_lag works on.

No math here (see core.lead_lag for the cross-correlation itself) and no caching policy either (see
adapters.historical.vendor.VendorHistoricalAdapter, which now caches any OHLC interval locally, not just daily).
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from adapters.base import HistoricalDataAdapter, SymbolTranslator
from core.curve_calendar import PRODUCTS, generic_contracts
from core.lead_lag import bar_differences

BAR_INTERVAL = "5M"
BAR_MINUTES = 5
GROUP_SIZE = 4  # consecutive outright contracts shown for a single product
DEFAULT_LOOKBACK_DAYS = 5
MAX_LOOKBACK_DAYS = 20
ALL_GROUPS = "ALL"  # one contract per product, instead of consecutive months of one

GROUP_OPTIONS = [{"label": "All (one contract per product)", "value": ALL_GROUPS}] + [
    {"label": spec.name, "value": code} for code, spec in PRODUCTS.items()
]


def group_symbols(group: str, month: int, as_of: date) -> dict[str, str]:
    """{label: internal symbol} for one Lead-Lag matrix.

    `month` is a generic position (1 = front). For a single product (a core.curve_calendar.PRODUCTS code), the
    GROUP_SIZE consecutive outright contracts starting there (e.g. month=3 on "CL" -> CL's 3rd, 4th, 5th, 6th
    live contracts). For ALL_GROUPS, the single contract at that position of every product (CL, BRN, G, CLBZ),
    so the matrix compares one contract per product rather than consecutive months of one; labels are then
    prefixed with the product code since the same month label can recur across products.
    """
    positions = dict.fromkeys(PRODUCTS, month) if group == ALL_GROUPS else {group: month}
    size = 1 if group == ALL_GROUPS else GROUP_SIZE
    out: dict[str, str] = {}
    for product, start in positions.items():
        contracts = generic_contracts(product, as_of, n=start + size - 1)[start - 1 : start - 1 + size]
        for contract in contracts:
            label = f"{product} {contract.label}" if group == ALL_GROUPS else contract.label
            out[label] = SymbolTranslator.api_to_internal(contract.symbol)
    return out


def load_bar_diffs(
    symbols_by_label: dict[str, str], adapter: HistoricalDataAdapter, lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, pd.Series]:
    """Price-difference series of 5-minute bars for each labeled symbol (core.lead_lag.bar_differences).

    Pulls the last `lookback_days` calendar days (cached locally after the first call -- see
    adapters.historical.vendor.VendorHistoricalAdapter). A symbol the adapter has nothing for (no candles in
    range, an unrecognised box leg, ...) is skipped rather than failing the whole matrix.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    by_symbol = adapter.get_ohlc_bulk(list(symbols_by_label.values()), BAR_INTERVAL, start=start, end=end)
    out: dict[str, pd.Series] = {}
    for label, symbol in symbols_by_label.items():
        bars = by_symbol.get(symbol)
        if bars is None or bars.empty:
            continue
        diffs = bar_differences(bars)
        if not diffs.empty:
            out[label] = diffs
    return out
