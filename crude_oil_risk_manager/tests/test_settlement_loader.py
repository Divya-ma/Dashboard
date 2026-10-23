"""SettlementFirstLoader: one price source for every page, settlements first and candles as the fallback."""

import numpy as np
import pandas as pd
import pytest

from core.settlement_prices import SettlementFirstLoader, SettlementPrices


class FakeStore:
    def __init__(self, frames):
        self.frames, self.version = frames, "v1"

    def frame(self, product):
        return self.frames.get(product, pd.DataFrame(columns=["contract", "date", "oi", "close", "volume"]))


def oi_rows(code, closes, start="2026-01-05"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"contract": code, "date": dates, "oi": 1.0, "close": closes, "volume": 1.0})


def walk(n, seed, vol=0.5):
    return 75 + np.cumsum(np.random.default_rng(seed).normal(0, vol, n))


class CountingAdapter:
    def __init__(self):
        self.backfilled = []

    def backfill_symbols(self, symbols, start, end):
        self.backfilled += list(symbols)
        return {s: 0 for s in symbols}


@pytest.fixture
def parts():
    closes = {"CLZ26": walk(60, 1), "CLF27": walk(60, 2), "CLG27": walk(60, 3), "COZ26": walk(60, 4)}
    store = FakeStore({
        "CL": pd.concat([oi_rows(code, closes[code]) for code in ("CLZ26", "CLF27", "CLG27")], ignore_index=True),
        "BRN": oi_rows("COZ26", closes["COZ26"]),
    })
    return closes, store


@pytest.fixture
def loader(tmp_path, parts):
    return SettlementFirstLoader(str(tmp_path), CountingAdapter(), SettlementPrices(parts[1]))


def test_an_outright_is_its_settlement(loader, parts):
    series = loader.load_close_series("CLZ26", min_rows=20)
    assert series.to_numpy() == pytest.approx(parts[0]["CLZ26"]) and str(series.index.tz) == "UTC" and series.name == "CLZ26"
    assert loader.load_close_series("BRNZ26", min_rows=20).to_numpy() == pytest.approx(parts[0]["COZ26"])  # BRN is read from its CO contract


def test_a_spread_and_a_fly_are_built_from_their_outrights_settlements(loader, parts):
    closes = parts[0]
    spread = loader.load_close_series("CLZ26-F27", min_rows=20)
    fly = loader.load_close_series("CLZ26-F27-G27", min_rows=20)
    assert spread.to_numpy() == pytest.approx(closes["CLZ26"] - closes["CLF27"])
    assert fly.to_numpy() == pytest.approx(closes["CLZ26"] - 2 * closes["CLF27"] + closes["CLG27"])  # every trading day, not just the days it printed
    assert loader.load_price_differences("CLZ26-F27-G27", min_rows=20).to_numpy() == pytest.approx(np.diff(fly.to_numpy()))


def test_a_structure_is_priced_on_the_dates_every_outright_settled(tmp_path, parts):
    closes, store = parts
    gappy = store.frames["CL"]
    store.frames["CL"] = gappy[~((gappy["contract"] == "CLF27") & (gappy["date"] == pd.Timestamp("2026-02-02")))]
    loader = SettlementFirstLoader(str(tmp_path), CountingAdapter(), SettlementPrices(store))
    spread = loader.load_close_series("CLZ26-F27", min_rows=20)
    assert len(spread) == 59 and pd.Timestamp("2026-02-02", tz="UTC") not in spread.index


def test_anything_the_store_does_not_hold_comes_from_the_candles(tmp_path, parts, create_synthetic_parquet):
    closes, store = parts
    create_synthetic_parquet(tmp_path, "WBSZ26", closes=walk(50, 7))  # not a CL or BRN contract
    create_synthetic_parquet(tmp_path, "CLH27", closes=walk(50, 8))  # a CL contract the store has no settlements for
    loader = SettlementFirstLoader(str(tmp_path), CountingAdapter(), SettlementPrices(store))
    assert len(loader.load_close_series("WBSZ26", min_rows=20)) == 50
    assert len(loader.load_close_series("CLH27", min_rows=20)) == 50


def test_a_structure_with_one_uncovered_leg_is_read_whole_from_the_candles(tmp_path, parts, create_synthetic_parquet):
    closes, store = parts
    create_synthetic_parquet(tmp_path, "CLZ26-H27", closes=walk(40, 9))
    loader = SettlementFirstLoader(str(tmp_path), CountingAdapter(), SettlementPrices(store))
    assert len(loader.load_close_series("CLZ26-H27", min_rows=20)) == 40  # CLH27 has no settlements: the quoted spread's own candles


def test_bz_keeps_its_own_candles_and_is_not_proxied_by_brent(tmp_path, parts, create_synthetic_parquet):
    closes, store = parts
    own = walk(50, 11)
    create_synthetic_parquet(tmp_path, "BZZ26", closes=own)
    loader = SettlementFirstLoader(str(tmp_path), CountingAdapter(), SettlementPrices(store))
    series = loader.load_close_series("BZZ26", min_rows=20)
    assert series.to_numpy() == pytest.approx(own) and not np.allclose(series.to_numpy()[:50], closes["COZ26"][:50])


def test_settlement_covered_symbols_never_trigger_a_backfill(tmp_path, parts):
    adapter = CountingAdapter()
    loader = SettlementFirstLoader(str(tmp_path), adapter, SettlementPrices(parts[1]))
    assert loader.get_available_date_range("CLZ26-F27-G27") is not None
    assert loader.ensure_cached(["CLZ26", "CLZ26-F27", "BRNZ26"]) == {}  # all counted as cached
    assert adapter.backfilled == []
    assert "CLH27" in loader.ensure_cached(["CLH27"]) and adapter.backfilled == ["CLH27"]  # the one the store lacks is fetched as before


def test_new_settlements_are_picked_up_when_the_store_version_changes(tmp_path, parts):
    closes, store = parts
    loader = SettlementFirstLoader(str(tmp_path), CountingAdapter(), SettlementPrices(store))
    assert len(loader.load_close_series("CLZ26", min_rows=20)) == 60
    store.frames["CL"] = pd.concat([store.frames["CL"], oi_rows("CLZ26", [80.0], start="2026-04-30")], ignore_index=True)
    assert len(loader.load_close_series("CLZ26", min_rows=20)) == 60  # cached until the version changes
    store.version = "v2"
    assert len(loader.load_close_series("CLZ26", min_rows=20)) == 61


def test_the_loader_says_which_symbols_it_reads_from_the_settlements(tmp_path, parts):
    loader = SettlementFirstLoader(str(tmp_path), CountingAdapter(), SettlementPrices(parts[1]))
    assert loader.covered_by_settlements("CLZ26") and loader.covered_by_settlements("CLZ26-F27-G27") and loader.covered_by_settlements("BRNZ26")
    assert not loader.covered_by_settlements("WBSZ26") and not loader.covered_by_settlements("CLH27") and not loader.covered_by_settlements("BZZ26")
