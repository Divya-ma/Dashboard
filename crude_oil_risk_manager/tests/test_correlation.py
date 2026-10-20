"""Tests for core.correlation and core.data_loader."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from adapters.mock.mock_historical import MockHistoricalAdapter
from core.correlation import (
    build_correlation_matrix,
    calculate_correlation,
    calculate_rolling_correlation,
    classify_correlation,
    beta_window,
    correlate_structures,
    lookback_average_correlation,
    outright_weights,
    structure_return_series,
    structure_series,
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
    assert calculate_correlation("CLZ26", "CLF27", 5, 20, loader) == pytest.approx(1.0)


def test_correlation_perfectly_anticorrelated(tmp_path, loader, create_synthetic_parquet):
    base = random_closes(100, seed=1)
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    create_synthetic_parquet(tmp_path, "CLF27", closes=200 - base)
    assert calculate_correlation("CLZ26", "CLF27", 5, 20, loader) == pytest.approx(-1.0)


def test_correlation_unrelated_series_near_zero(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=random_closes(500, seed=1))
    create_synthetic_parquet(tmp_path, "CLF27", closes=random_closes(500, seed=2))
    assert abs(calculate_correlation("CLZ26", "CLF27", 10, 300, loader)) < 0.2


def test_correlation_insufficient_data_raises(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=15, seed=1)
    create_synthetic_parquet(tmp_path, "CLF27", n_days=15, seed=2)
    with pytest.raises(InsufficientDataError):  # 14 days of changes: fewer than the 20 every correlation needs
        calculate_correlation("CLZ26", "CLF27", 5, 60, loader)
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=30, seed=1)
    create_synthetic_parquet(tmp_path, "CLF27", n_days=30, seed=2)
    assert -1.0 <= calculate_correlation("CLZ26", "CLF27", 5, 60, loader) <= 1.0  # 29 days of changes: the lookback average uses the days there are


def test_correlation_zero_variance_raises_instead_of_nan(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=np.full(60, 75.0))
    create_synthetic_parquet(tmp_path, "CLF27", closes=random_closes(60, seed=2))
    with pytest.raises(InsufficientDataError):
        calculate_correlation("CLZ26", "CLF27", 5, 20, loader)


# ---------- the two-step measure: rolling window, then lookback average ----------


def _pair(n: int, seed: int) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    a = pd.Series(rng.normal(size=n), index=index, name="A")
    b = pd.Series(0.5 * a.to_numpy() + rng.normal(size=n), index=index, name="B")
    return a, b


def test_lookback_average_is_the_mean_of_the_last_L_daily_rolling_R_day_correlations():
    a, b = _pair(120, seed=3)
    value, used = lookback_average_correlation(a, b, rolling=5, lookback=20)
    daily = [np.corrcoef(a.iloc[i - 4:i + 1], b.iloc[i - 4:i + 1])[0, 1] for i in range(100, 120)]  # 20 days, 5 each
    assert used == 20
    assert value == pytest.approx(float(np.mean(daily)))


def test_changing_either_window_changes_the_result():
    a, b = _pair(200, seed=4)
    base = lookback_average_correlation(a, b, 5, 20)[0]
    assert lookback_average_correlation(a, b, 10, 20)[0] != pytest.approx(base)
    assert lookback_average_correlation(a, b, 5, 60)[0] != pytest.approx(base)


def test_it_is_not_a_single_pearson_over_the_last_R_plus_L_days():
    a, b = _pair(200, seed=5)
    averaged = lookback_average_correlation(a, b, 5, 20)[0]
    one_big_window = float(np.corrcoef(a.iloc[-24:], b.iloc[-24:])[0, 1])
    assert averaged != pytest.approx(one_big_window)


def test_a_pair_needs_20_common_days_and_averages_the_days_there_are(tmp_path, loader, create_synthetic_parquet):
    base = random_closes(20, seed=1)  # 19 diffs: one short of the floor
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    create_synthetic_parquet(tmp_path, "CLF27", closes=2 * base + 10)
    with pytest.raises(InsufficientDataError):
        calculate_correlation("CLZ26", "CLF27", 5, 20, loader)
    base = random_closes(21, seed=1)  # 20 diffs: enough, though a full 20-day lookback of 5-day values needs 24
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    create_synthetic_parquet(tmp_path, "CLF27", closes=2 * base + 10)
    assert calculate_correlation("CLZ26", "CLF27", 5, 20, loader) == pytest.approx(1.0)


def test_a_rolling_window_below_two_and_a_lookback_below_one_are_rejected():
    a, b = _pair(50, seed=6)
    with pytest.raises(ValueError, match="rolling window"):
        lookback_average_correlation(a, b, 1, 20)
    with pytest.raises(ValueError, match="lookback"):
        lookback_average_correlation(a, b, 5, 0)
    assert lookback_average_correlation(a, b, 2, 1)[1] == 1  # the smallest legal pair works


def test_a_flat_stretch_is_left_out_of_the_average_not_counted_as_zero():
    a, b = _pair(60, seed=7)
    a.iloc[-12:-4] = 1.0  # a flat run: windows wholly inside it have no variance
    value, used = lookback_average_correlation(a, b, 5, 20)
    assert 0 < used < 20 and np.isfinite(value)


def test_beta_window_is_the_correlation_span_but_at_least_30():
    assert beta_window(5, 20) == 30
    assert beta_window(10, 100) == 109


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
    matrix = build_correlation_matrix(symbols, 5, 20, loader)
    assert list(matrix.index) == symbols
    assert list(matrix.columns) == symbols
    assert all(matrix.loc[s, s] == 1.0 for s in symbols)


def test_matrix_is_symmetric(tmp_path, loader, create_synthetic_parquet):
    symbols = ["CLZ26", "CLF27", "BRNZ26"]
    for i, symbol in enumerate(symbols):
        create_synthetic_parquet(tmp_path, symbol, closes=random_closes(100, seed=i))
    matrix = build_correlation_matrix(symbols, 5, 20, loader)
    pd.testing.assert_frame_equal(matrix, matrix.T)


def test_matrix_failed_pair_is_nan_and_does_not_raise(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", closes=random_closes(100, seed=1))
    create_synthetic_parquet(tmp_path, "CLF27", closes=random_closes(100, seed=2))
    # BRNZ26 has no Parquet file and the mock adapter cannot backfill.
    matrix = build_correlation_matrix(["CLZ26", "CLF27", "BRNZ26"], 5, 20, loader)
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


# ---------- outright_weights ----------


def test_outright_weights_decomposes_legs():
    assert outright_weights([{"symbol": "CLZ26", "ratio": 1}, {"symbol": "CLF27", "ratio": -1}]) == {
        "CLZ26": 1.0, "CLF27": -1.0,
    }


def test_outright_weights_decomposes_a_nested_leg():
    """A leg whose own symbol is itself a spread/fly (a structure nested inside another
    structure's leg) still decomposes to plain outrights."""
    weights = outright_weights([{"symbol": "CLZ26-F27", "ratio": 1}])
    assert weights == {"CLZ26": 1.0, "CLF27": -1.0}


def test_outright_weights_empty_for_no_valid_legs():
    assert outright_weights([{"symbol": "", "ratio": 1}]) == {}


# ---------- structure_return_series ----------


def test_structure_return_series_combines_weighted_legs(tmp_path, loader, create_synthetic_parquet):
    base = random_closes(60, seed=1)
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    series = structure_return_series({"CLZ26": 2.0}, loader)
    assert series is not None
    expected = pd.Series(base).diff().dropna().to_numpy() * 2.0
    assert series.to_numpy() == pytest.approx(expected)


def test_structure_return_series_inner_joins_only_its_own_legs(tmp_path, loader, create_synthetic_parquet):
    """A leg with a much shorter history than another one's ONLY shrinks THIS structure's
    own series — it must never affect a comparison that doesn't involve that leg at all."""
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=200, seed=1)
    create_synthetic_parquet(tmp_path, "CLF27", n_days=25, seed=2, end_date=date(2026, 6, 30))
    series = structure_return_series({"CLZ26": 1.0, "CLF27": -1.0}, loader)
    assert series is not None and len(series) <= 24


def test_structure_return_series_none_when_no_leg_has_data(loader):
    assert structure_return_series({"BRNZ26": 1.0}, loader) is None


def test_structure_return_series_none_below_the_observation_floor(tmp_path, loader, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=10, seed=1)
    assert structure_return_series({"CLZ26": 1.0}, loader) is None


def test_structure_return_series_needs_every_leg(tmp_path, loader, create_synthetic_parquet):
    """A leg with no history used to be dropped, which silently turned the structure into a different one."""
    create_synthetic_parquet(tmp_path, "CLZ26", closes=random_closes(60, seed=1))
    assert structure_return_series({"CLZ26": 1.0, "BRNZ26": -1.0}, loader) is None  # BRNZ26 has no data
    built = structure_series({"CLZ26": 1.0, "BRNZ26": -1.0}, loader)
    assert built.series is None and built.missing == ["BRNZ26"]


# ---------- correlate_structures ----------


def test_correlate_structures_perfectly_correlated_outrights(tmp_path, loader, create_synthetic_parquet):
    base = random_closes(100, seed=1)
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    create_synthetic_parquet(tmp_path, "CLF27", closes=2 * base + 10)
    result = correlate_structures({"CLZ26": 1.0}, {"CLF27": 1.0}, 5, 20, loader)
    assert result["correlation"] == pytest.approx(1.0)
    assert result["classification"] == "highly_correlated"
    assert result["window_used"] == 20 and result["rolling"] == 5 and result["error"] is None


def test_correlate_structures_is_the_average_of_daily_rolling_correlations_for_spreads(tmp_path, loader, create_synthetic_parquet):
    """A spread-vs-spread correlation via each structure's own derived series must equal the mean of
    the last 20 daily 5-day Pearson correlations of those same two derived series."""
    a, b, c = (random_closes(200, seed=i) for i in (1, 2, 3))
    create_synthetic_parquet(tmp_path, "CLZ26", closes=a)
    create_synthetic_parquet(tmp_path, "CLF27", closes=b)
    create_synthetic_parquet(tmp_path, "CLG27", closes=c)

    weights_a = outright_weights([{"symbol": "CLZ26", "ratio": 1}, {"symbol": "CLF27", "ratio": -1}])
    weights_b = outright_weights([{"symbol": "CLF27", "ratio": 1}, {"symbol": "CLG27", "ratio": -1}])
    result = correlate_structures(weights_a, weights_b, 5, 20, loader)

    da, db, dc = (pd.Series(x).diff().dropna() for x in (a, b, c))
    spread_a, spread_b = da - db, db - dc
    expected = spread_a.rolling(5).corr(spread_b).iloc[-20:].mean()
    assert result["window_used"] == 20
    assert result["correlation"] == pytest.approx(expected)


def test_correlate_structures_missing_data_never_raises(loader):
    result = correlate_structures({"BRNZ26": 1.0}, {"CLZ26": 1.0}, 5, 20, loader)
    assert result["correlation"] is None and result["error"]


def test_correlate_structures_falls_back_to_available_lookback(tmp_path, loader, create_synthetic_parquet):
    base = random_closes(30, seed=1)  # 29 diffs -> 25 daily 5-day values, fewer than the 60-day lookback requested
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    create_synthetic_parquet(tmp_path, "CLF27", closes=2 * base + 10)
    result = correlate_structures({"CLZ26": 1.0}, {"CLF27": 1.0}, 5, 60, loader)
    assert result["correlation"] == pytest.approx(1.0)
    assert result["window_used"] == 25
    assert "Only 25 of the requested 60-day lookback" in result["error"]


def test_correlate_structures_illiquid_unrelated_leg_does_not_starve_this_pair(tmp_path, loader, create_synthetic_parquet):
    """Regression: comparing two liquid structures must not be dragged down to a handful
    of observations just because SOME OTHER structure in the portfolio has an illiquid leg
    with sparse, barely-overlapping dates — that leg never enters this pair's own series."""
    base = random_closes(200, seed=1)
    create_synthetic_parquet(tmp_path, "CLZ26", closes=base)
    create_synthetic_parquet(tmp_path, "CLF27", closes=2 * base + 5)
    # An illiquid, unrelated leg with almost no date overlap with the above two.
    create_synthetic_parquet(tmp_path, "BRNZ26", n_days=5, seed=9, end_date=date(2020, 1, 5))

    result = correlate_structures({"CLZ26": 1.0}, {"CLF27": 1.0}, 5, 20, loader)
    assert result["window_used"] == 20  # unaffected by BRNZ26 existing elsewhere in the portfolio


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


# ---------- DataLoader.ensure_cached ----------


def test_ensure_cached_skips_symbols_already_local(tmp_path, create_synthetic_parquet):
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=10, seed=1)

    class Adapter(MockHistoricalAdapter):
        def backfill_symbols(self, symbols, start, end=None):
            raise AssertionError("should not be called for an already-cached symbol")

    loader = DataLoader(str(tmp_path), Adapter())
    assert loader.ensure_cached(["CLZ26"]) == {}


def test_ensure_cached_batches_missing_symbols_in_one_call(tmp_path):
    calls = []

    class Adapter(MockHistoricalAdapter):
        def backfill_symbols(self, symbols, start, end=None):
            calls.append(list(symbols))
            return {s: 10 for s in symbols}

    loader = DataLoader(str(tmp_path), Adapter())
    result = loader.ensure_cached(["CLZ26", "CLF27", "CLZ26"])  # duplicate collapsed

    assert calls == [["CLZ26", "CLF27"]]
    assert result == {"CLZ26": "ok", "CLF27": "ok"}


def test_ensure_cached_without_adapter_support(tmp_path):
    loader = DataLoader(str(tmp_path), MockHistoricalAdapter())
    assert loader.ensure_cached(["CLZ26"]) == {"CLZ26": "no backfill support"}


def test_ensure_cached_never_raises_on_adapter_failure(tmp_path):
    class Adapter(MockHistoricalAdapter):
        def backfill_symbols(self, symbols, start, end=None):
            raise RuntimeError("no API token configured")

    loader = DataLoader(str(tmp_path), Adapter())
    result = loader.ensure_cached(["CLZ26"])
    assert result == {"CLZ26": "no API token configured"}
