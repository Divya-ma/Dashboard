"""Tests for core.correlation and core.data_loader."""

import numpy as np
import pandas as pd
import pytest

from adapters.mock.mock_historical import MockHistoricalAdapter
from core.correlation import (
    build_correlation_matrix,
    calculate_correlation,
    calculate_rolling_correlation,
    classify_correlation,
    get_correlation_with_portfolio,
)
from core.data_loader import DataLoader
from core.exceptions import DataNotAvailableError, InsufficientDataError


def random_closes(n: int, seed: int, start: float = 75.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start + np.cumsum(rng.normal(0.0, 1.5, n))


@pytest.fixture
def loader(tmp_path):
    return DataLoader(str(tmp_path), MockHistoricalAdapter())


def make_series(name: str, day_offsets: range) -> pd.Series:
    index = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=d) for d in day_offsets]
    )
    return pd.Series(np.arange(len(index), dtype=float), index=index, name=name)


# ---------- calculate_correlation ----------


def test_correlation_perfectly_correlated(tmp_path, loader, create_synthetic_parquet):
    base = random_closes(100, seed=1)
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    create_synthetic_parquet(tmp_path, "CLF27", closes=2 * base + 10)
    assert calculate_correlation("CLZ26", "CLF27", 60, loader) == pytest.approx(1.0)


def test_correlation_perfectly_anticorrelated(tmp_path, loader, create_synthetic_parquet):
    base = random_closes(100, seed=1)
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    create_synthetic_parquet(tmp_path, "CLF27", closes=200 - base)
    assert calculate_correlation("CLZ26", "CLF27", 60, loader) == pytest.approx(-1.0)


def test_correlation_unrelated_series_near_zero(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=random_closes(500, seed=1))
    create_synthetic_parquet(tmp_path, "CLF27", closes=random_closes(500, seed=2))
    assert abs(calculate_correlation("CLZ26", "CLF27", 400, loader)) < 0.2


def test_correlation_insufficient_data_raises(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=30, seed=1)
    create_synthetic_parquet(tmp_path, "CLF27", n_days=30, seed=2)
    with pytest.raises(InsufficientDataError):
        calculate_correlation("CLZ26", "CLF27", 60, loader)


def test_correlation_zero_variance_raises_instead_of_nan(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=np.full(60, 75.0))
    create_synthetic_parquet(tmp_path, "CLF27", closes=random_closes(60, seed=2))
    with pytest.raises(InsufficientDataError):
        calculate_correlation("CLZ26", "CLF27", 30, loader)


# ---------- calculate_rolling_correlation ----------


def test_rolling_correlation_length(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=random_closes(100, seed=1))
    create_synthetic_parquet(tmp_path, "CLF27", closes=random_closes(100, seed=2))
    result = calculate_rolling_correlation("CLZ26", "CLF27", 20, loader)
    # 100 closes -> 99 diffs -> 99 - 20 + 1 rolling values
    assert len(result) == 80


def test_rolling_correlation_has_no_nan(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=random_closes(100, seed=1))
    create_synthetic_parquet(tmp_path, "CLF27", closes=random_closes(100, seed=2))
    result = calculate_rolling_correlation("CLZ26", "CLF27", 20, loader)
    assert not result.isna().any()
    assert result.between(-1.0, 1.0).all()


# ---------- build_correlation_matrix ----------


def test_matrix_diagonal_is_one(tmp_path, loader, create_synthetic_parquet):
    symbols = ["CLZ26", "CLF27", "BRNZ26"]
    for i, symbol in enumerate(symbols):
        create_synthetic_parquet(tmp_path, symbol, closes=random_closes(100, seed=i))
    matrix = build_correlation_matrix(symbols, 30, loader)
    assert list(matrix.index) == symbols
    assert list(matrix.columns) == symbols
    assert all(matrix.loc[s, s] == 1.0 for s in symbols)


def test_matrix_is_symmetric(tmp_path, loader, create_synthetic_parquet):
    symbols = ["CLZ26", "CLF27", "BRNZ26"]
    for i, symbol in enumerate(symbols):
        create_synthetic_parquet(tmp_path, symbol, closes=random_closes(100, seed=i))
    matrix = build_correlation_matrix(symbols, 30, loader)
    pd.testing.assert_frame_equal(matrix, matrix.T)


def test_matrix_failed_pair_is_nan_and_does_not_raise(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=random_closes(100, seed=1))
    create_synthetic_parquet(tmp_path, "CLF27", closes=random_closes(100, seed=2))
    # BRNZ26 has no Parquet file and the mock adapter cannot backfill.
    matrix = build_correlation_matrix(["CLZ26", "CLF27", "BRNZ26"], 30, loader)
    assert np.isnan(matrix.loc["CLZ26", "BRNZ26"])
    assert np.isnan(matrix.loc["BRNZ26", "CLF27"])
    assert matrix.loc["BRNZ26", "BRNZ26"] == 1.0
    assert not np.isnan(matrix.loc["CLZ26", "CLF27"])


# ---------- classify_correlation ----------


def test_classify_correlation_boundaries():
    assert classify_correlation(0.7) == "highly_correlated"
    assert classify_correlation(-0.7) == "negatively_correlated"
    assert classify_correlation(0.0) == "uncorrelated"
    assert classify_correlation(0.69) == "uncorrelated"
    assert classify_correlation(-0.69) == "uncorrelated"
    assert classify_correlation(float("nan")) == "insufficient_data"


# ---------- get_correlation_with_portfolio ----------


def test_portfolio_correlation_contains_all_portfolio_symbols(tmp_path, loader, create_synthetic_parquet):
    base = random_closes(100, seed=1)
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    create_synthetic_parquet(tmp_path, "CLF27", closes=2 * base + 10)
    portfolio = ["CLF27", "BRNZ26"]  # BRNZ26 has no data

    result = get_correlation_with_portfolio("CLZ26", portfolio, 30, loader)

    assert set(result.keys()) == set(portfolio)
    assert result["CLF27"]["correlation"] == pytest.approx(1.0)
    assert result["CLF27"]["classification"] == "highly_correlated"
    assert result["CLF27"]["window_used"] == 30
    assert result["CLF27"]["common_dates"] == 99
    assert result["BRNZ26"]["correlation"] is None
    assert result["BRNZ26"]["classification"] == "insufficient_data"


def test_portfolio_correlation_missing_candidate_returns_none_for_all(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLF27", closes=random_closes(100, seed=1))
    result = get_correlation_with_portfolio("CLZ26", ["CLF27"], 30, loader)
    assert result["CLF27"]["correlation"] is None


# ---------- DataLoader ----------


def test_align_series_inner_join_drops_non_common_dates(loader):
    a = make_series("A", range(0, 60))
    b = make_series("B", range(10, 70))
    aligned_a, aligned_b = loader.align_series(a, b)
    expected = a.index.intersection(b.index)
    assert len(expected) == 50
    assert aligned_a.index.equals(expected)
    assert aligned_b.index.equals(expected)


def test_align_series_raises_when_fewer_than_20_common(loader):
    a = make_series("A", range(0, 30))
    b = make_series("B", range(20, 50))  # only 10 common dates
    with pytest.raises(InsufficientDataError):
        loader.align_series(a, b)


def test_load_price_differences_has_no_nan(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=50, seed=3)
    diffs = loader.load_price_differences("CLZ26")
    assert not diffs.isna().any()
    assert len(diffs) == 49
    assert diffs.name == "CLZ26"


def test_load_close_series_insufficient_rows_raises(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=10, seed=3)
    with pytest.raises(InsufficientDataError):
        loader.load_close_series("CLZ26", min_rows=20)


def test_load_close_series_missing_file_without_backfill_raises(loader):
    with pytest.raises(DataNotAvailableError):
        loader.load_close_series("CLZ26")


def test_load_close_series_triggers_backfill_for_missing_file(tmp_path, create_synthetic_parquet):
    class BackfillingAdapter(MockHistoricalAdapter):
        def backfill_symbol(self, symbol, start, end=None):
            create_synthetic_parquet(tmp_path, symbol, n_days=60, seed=5)
            return 60

    loader = DataLoader(str(tmp_path), BackfillingAdapter())
    series = loader.load_close_series("CLZ26")
    assert len(series) == 60
    assert series.index.is_monotonic_increasing


def test_load_close_series_backfill_returning_zero_rows_raises(tmp_path):
    class EmptyBackfillAdapter(MockHistoricalAdapter):
        def backfill_symbol(self, symbol, start, end=None):
            return 0

    loader = DataLoader(str(tmp_path), EmptyBackfillAdapter())
    with pytest.raises(DataNotAvailableError):
        loader.load_close_series("CLZ26")


def test_load_close_series_bulk_reports_all_failures(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=50, seed=3)
    with pytest.raises(DataNotAvailableError) as exc_info:
        loader.load_close_series_bulk(["CLZ26", "CLF27", "BRNZ26"])
    message = str(exc_info.value)
    assert "CLF27" in message
    assert "BRNZ26" in message
    assert "CLZ26:" not in message


def test_get_available_date_range(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=10, seed=3)
    date_range = loader.get_available_date_range("CLZ26")
    assert date_range is not None
    earliest, latest = date_range
    assert (latest - earliest).days == 9
    assert loader.get_available_date_range("CLF27") is None
