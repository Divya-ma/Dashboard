"""Correlation matrix of the front structures of two curves, on SETTLEMENT data.

For a product and a family (outright, spread, fly, dfly) the page takes the first N live structures: the
front contract and the ones after it, one month apart (spread k = M_k - M_k+1, fly = M_k - 2 M_k+1 + M_k+2,
dfly = M_k - 3 M_k+1 + 3 M_k+2 - M_k+3, as everywhere else in the app). Each structure is a FIXED set of
contracts (the contracts that are the front N today, or at the end of a historical range), valued every day
from the settlements of its outrights (core.oi_data's `close`, the /common/dailymarketdata/ settlement), so
its price series has no roll jumps in it.

The correlation of two structures is the one used across the Correlation tab (core.correlation):
  1. ROLLING WINDOW R: the Pearson correlation of the two structures' daily settlement changes over the R days
     ending each day.
  2. LOOKBACK L: the plain average of those daily values over the last L days.
With a HISTORICAL date range the lookback becomes the range: every daily value whose day is inside it is
averaged (the daily values themselves are computed from the days before the range too, so the first day of
the range already has a full window). The matrix is target structures (rows) against base structures
(columns).

The hedge ratio is the minimum-variance one: the beta of the target's daily changes on the base's over a
window of W days (lots of base per lot of target); `rolling_beta` gives its path.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from core.correlation import _daily_correlations, beta_window
from core.oi_data import contract_universe

FAMILIES = ("outright", "spread", "fly", "dfly")
WEIGHTS = {"outright": (1,), "spread": (1, -1), "fly": (1, -2, 1), "dfly": (1, -3, 3, -1)}
FAMILY_NAMES = {"outright": "outright", "spread": "spread", "fly": "fly", "dfly": "d-fly"}
MAX_STRUCTURES = 30  # the OI universe reaches 36 months ahead; a dfly needs 3 more months than its count
MIN_DAYS_ALIGNED = 3
FRONT_DATA_GAP_DAYS = 7  # a contract whose settlements stopped this long before the as-of day is not the front


@dataclass
class StructureSet:
    """The first N structures of one product and family as of one day."""

    product: str
    family: str
    as_of: date
    labels: list[str]  # "Dec26/Jan27/Feb27", front first
    contracts: list[tuple[str, ...]]  # API symbols of each structure's legs
    levels: pd.DataFrame  # settlement price of each structure, a column per label
    changes: pd.DataFrame  # day-to-day change of each, a column per label


def key_of(product: str, family: str) -> str:
    return f"{product}|{family}"


def parse_key(key: str) -> tuple[str, str]:
    product, family = key.split("|")
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    return product, family


def key_label(key: str) -> str:
    product, family = parse_key(key)
    return f"{product} {FAMILY_NAMES[family]}"


def structure_options() -> list[dict]:
    from core.curve_calendar import PRODUCTS

    return [{"label": key_label(key_of(p, f)), "value": key_of(p, f)} for p in PRODUCTS for f in FAMILIES]


def _front_index(contracts, prices: pd.DataFrame, as_of: date) -> int:
    """Index of the front contract on `as_of`: the first whose (rule-based) expiry is after it and that still
    settles in the days before it."""
    seen = prices[prices.index <= pd.Timestamp(as_of)]
    cutoff = (seen.index[-1] - pd.Timedelta(days=FRONT_DATA_GAP_DAYS)) if len(seen) else None
    for i, contract in enumerate(contracts):
        if contract.expiry <= as_of:
            continue
        if cutoff is not None and contract.symbol in seen and seen[contract.symbol].loc[cutoff:].notna().any():
            return i
    raise ValueError("no live contract has settlements around that date")


def build_structure_set(frame: pd.DataFrame, product: str, family: str, n: int, as_of: date, today: date) -> StructureSet:
    """Settlement series of the first `n` structures as of `as_of` (data up to and including that day)."""
    if frame.empty:
        raise ValueError(f"no settlement data for {product} yet")
    n = max(1, min(int(n), MAX_STRUCTURES))
    prices = frame.pivot(index="date", columns="contract", values="close").sort_index()
    prices = prices[prices.index <= pd.Timestamp(as_of)]
    contracts = contract_universe(product, max(today, as_of))
    weights = WEIGHTS[family]
    front = _front_index(contracts, prices, as_of)
    labels, legs_list, levels = [], [], {}
    for k in range(n):
        legs = contracts[front + k: front + k + len(weights)]
        if len(legs) < len(weights):
            break
        label = "/".join(c.label for c in legs)
        symbols = tuple(c.symbol for c in legs)
        labels.append(label)
        legs_list.append(symbols)
        if all(s in prices for s in symbols):
            parts = pd.concat([prices[s] * w for s, w in zip(symbols, weights)], axis=1)
            levels[label] = parts.sum(axis=1, min_count=len(weights))
        else:
            levels[label] = pd.Series(np.nan, index=prices.index)
    level_frame = pd.DataFrame(levels, index=prices.index)[labels]
    changes = pd.DataFrame({label: level_frame[label].dropna().diff() for label in labels}).reindex(level_frame.index)
    return StructureSet(product, family, as_of, labels, legs_list, level_frame, changes)


def pair_daily(target: pd.Series, base: pd.Series, rolling: int) -> pd.Series:
    """The daily `rolling`-day correlation of two change series, on the days both have a change."""
    aligned = pd.concat([target.rename("t"), base.rename("b")], axis=1, join="inner").dropna().sort_index()
    if len(aligned) < MIN_DAYS_ALIGNED:
        return pd.Series(dtype=float)
    return _daily_correlations(aligned["t"], aligned["b"], rolling).dropna()


def lookback_value(daily: pd.Series, lookback: int, start: date | None = None) -> tuple[float, int]:
    """The average of the daily values: the last `lookback` ones, or every one on or after `start`."""
    used = daily[daily.index >= pd.Timestamp(start)] if start is not None else daily.iloc[-lookback:]
    return (float(used.mean()), int(len(used))) if len(used) else (np.nan, 0)


def correlation_matrix(
    target: StructureSet, base: StructureSet, rolling: int, lookback: int, start: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(values, days used): rows are the target structures, columns the base structures."""
    values = pd.DataFrame(np.nan, index=target.labels, columns=base.labels)
    days = pd.DataFrame(0, index=target.labels, columns=base.labels)
    cache: dict[tuple[str, str], tuple[float, int]] = {}
    same = (target.product, target.family, target.as_of) == (base.product, base.family, base.as_of)
    for t in target.labels:
        for b in base.labels:
            if same and (b, t) in cache:
                value, used = cache[(b, t)]
            elif same and t == b:
                value, used = (1.0, 1) if target.changes[t].notna().sum() >= MIN_DAYS_ALIGNED else (np.nan, 0)
            else:
                value, used = lookback_value(pair_daily(target.changes[t], base.changes[b], rolling), lookback, start)
            cache[(t, b)] = (value, used)
            values.at[t, b], days.at[t, b] = value, used
    return values, days


def daily_history(target: StructureSet, base: StructureSet, t: str, b: str, rolling: int, days: int) -> pd.Series:
    """The daily rolling correlations of one pair over the last `days` trading days up to the as-of day."""
    return pair_daily(target.changes[t], base.changes[b], rolling).iloc[-max(int(days), 1):]


def rolling_beta(target: pd.Series, base: pd.Series, window: int) -> pd.DataFrame:
    """Per day: the beta of the target's changes on the base's over the last `window` shared days, and the correlation."""
    aligned = pd.concat([target.rename("t"), base.rename("b")], axis=1, join="inner").dropna().sort_index()
    if len(aligned) < max(window, MIN_DAYS_ALIGNED):
        return pd.DataFrame(columns=["beta", "corr"])
    var = aligned["b"].rolling(window).var()
    beta = (aligned["t"].rolling(window).cov(aligned["b"]) / var.where(var > 0)).replace([np.inf, -np.inf], np.nan)
    corr = _daily_correlations(aligned["t"], aligned["b"], window)
    return pd.DataFrame({"beta": beta, "corr": corr}).dropna(subset=["beta"])


def default_beta_window(rolling: int, lookback: int) -> int:
    """The window the rest of the app regresses over (core.correlation.beta_window)."""
    return beta_window(rolling, lookback)


def as_of_for(mode: str, end: date | None, today: date) -> date:
    """The day the structures are chosen on: today, or the end of the historical range."""
    return min(end, today) if mode == "historical" and end else today


def validate_range(mode: str, start: date | None, end: date | None) -> None:
    if mode != "historical":
        return
    if start is None or end is None:
        raise ValueError("Pick both the start and the end of the date range.")
    if start >= end:
        raise ValueError("The date range must start before it ends.")
    if end - start < timedelta(days=2):
        raise ValueError("The date range is too short.")
