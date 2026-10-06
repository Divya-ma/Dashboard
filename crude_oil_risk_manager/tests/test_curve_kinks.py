"""Kink maths: building blocks, the four methods, echo suppression, seasonality and priorities."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.curve_kinks import (
    detect, is_kink, meets_priority, neighbour_residuals, neighbour_weights, prepare_history, robust_fit,
)
from core.curve_settings import CurveParams, load_params, save_params, validate_open_time, validate_params


def smooth_history(rows=800, n=15, seed=1):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    level = 80 + np.cumsum(rng.normal(0, 0.6, rows))
    slope = rng.normal(0, 0.15, rows)
    curvature = rng.normal(0, 0.03, rows)
    values = (
        level[:, None] - slope[:, None] * x[None, :]
        + curvature[:, None] * (x[None, :] - 7) ** 2 / 10 + rng.normal(0, 0.02, (rows, n))
    )
    dates = pd.date_range("2022-01-03", periods=rows, freq="B").values
    months = ((np.arange(n) + pd.DatetimeIndex(dates).month.values[:, None]) % 12) + 1
    return values, dates, months


def test_neighbour_weights_rows_sum_to_one():
    w = neighbour_weights(15)
    assert np.allclose(w.sum(axis=1), 1.0) and np.allclose(np.diag(w), 0.0)


def test_neighbour_residual_is_exact_for_quadratics_everywhere_and_cubics_inside():
    x = np.arange(15.0)
    assert np.allclose(neighbour_residuals(1 + 2 * x - 0.3 * x**2), 0.0, atol=1e-9)
    cubic = neighbour_residuals(1 + 2 * x - 0.3 * x**2 + 0.01 * x**3)
    assert np.allclose(cubic[2:-2], 0.0, atol=1e-9)


def test_neighbour_residual_of_a_spike_and_its_echo():
    y = np.zeros(15)
    y[7] = 1.0
    r = neighbour_residuals(y)
    assert r[7] == pytest.approx(1.0) and r[6] == pytest.approx(-2 / 3) and r[8] == pytest.approx(-2 / 3)


def test_neighbour_residual_is_nan_when_a_needed_neighbour_is_missing():
    y = np.arange(15.0)
    y[7] = np.nan
    r = neighbour_residuals(y)
    assert np.isnan(r[7]) and np.isnan(r[6]) and not np.isnan(r[0])


def test_robust_fit_ignores_a_single_outlier():
    x = np.arange(15.0)
    y = 100 - 1.5 * x + 0.05 * x**2
    clean = y.copy()
    y[6] += 3.0
    fit = robust_fit(y, 3)
    assert abs(fit[6] - clean[6]) < 0.05  # the fit stays on the curve, not on the spike


def test_detects_an_injected_kink_by_several_methods_and_suppresses_echoes():
    values, dates, months = smooth_history()
    params = CurveParams()
    prepared = prepare_history(values, dates, months, params)
    live = values[-1].copy()
    live[6] += 0.35
    results = detect(live, prepared, False, params)
    kinks = [r for r in results if is_kink(r)]
    assert [r.index for r in kinks] == [6]
    assert kinks[0].n_flags == 3 and kinks[0].priority == "HIGH" and kinks[0].direction == "rich"
    assert results[5].echo_of == 6 and not is_kink(results[5])


def test_a_kink_near_the_front_does_not_make_the_front_contract_a_kink():
    """The neighbour method extrapolates at the curve end, which amplifies a kink two points away."""
    x = np.arange(15.0)
    smooth = 100 - 1.2 * x + 0.03 * x**2
    params = CurveParams()
    live = smooth.copy()
    live[2] += 0.4
    results = detect(live, None, False, params)  # no history: cross-sectional scales
    assert [r.index for r in results if is_kink(r)] == [2]
    assert results[0].echo_of == 2 and results[0].priority == "NONE"


def test_two_independent_kinks_are_both_found():
    values, dates, months = smooth_history()
    params = CurveParams()
    prepared = prepare_history(values, dates, months, params)
    live = values[-1].copy()
    live[3] += 0.4
    live[11] -= 0.5
    kinks = {r.index: r for r in detect(live, prepared, False, params) if is_kink(r)}
    assert set(kinks) == {3, 11}
    assert kinks[3].direction == "rich" and kinks[11].direction == "cheap"


def test_a_clean_curve_has_no_kinks():
    values, dates, months = smooth_history()
    params = CurveParams()
    prepared = prepare_history(values[:-1], dates[:-1], months[:-1], params)
    assert not any(is_kink(r) for r in detect(values[-1], prepared, False, params))


def test_history_method_only_runs_for_non_outright_curves():
    values, dates, months = smooth_history()
    params = CurveParams()
    prepared = prepare_history(values, dates, months, params)
    live = values[-1].copy()
    live[4] += 0.5
    outright = detect(live, prepared, False, params)[4]
    spread_like = detect(live, prepared, True, params)[4]
    assert outright.z["history"] is None and spread_like.z["history"] is not None


def test_without_history_only_fit_and_neighbour_run():
    values, _, _ = smooth_history()
    params = CurveParams()
    live = values[-1].copy()
    live[8] += 0.6
    result = detect(live, None, True, params)[8]
    assert result.z["pca"] is None and result.z["history"] is None
    assert result.z["fit"] is not None and result.z["neighbour"] is not None and result.n_flags >= 1


def test_missing_live_prices_are_skipped_not_fatal():
    values, dates, months = smooth_history()
    params = CurveParams()
    prepared = prepare_history(values, dates, months, params)
    live = values[-1].copy()
    live[3] = np.nan
    results = detect(live, prepared, False, params)
    assert results[3].value is None and results[3].n_flags == 0
    assert results[10].z["pca"] is not None


def test_priority_levels_and_the_seasonal_downgrade():
    values, dates, months = smooth_history()
    params = CurveParams()
    prepared = prepare_history(values, dates, months, params)
    live = values[-1].copy()
    live[6] += 0.35
    month_numbers = [int(m) for m in months[-1]]
    # A recurring, identical bump at this delivery month in earlier years makes the kink seasonal.
    seasonal_values = values.copy()
    years = pd.DatetimeIndex(dates).year.values
    doy = pd.DatetimeIndex(dates).dayofyear.values
    today = date(2024, 2, 20)
    in_window = (years < 2024) & (np.abs(doy - today.timetuple().tm_yday) <= 15)
    for row in np.flatnonzero(in_window):  # the bump follows the delivery month as it moves along the curve
        seasonal_values[row, int(np.flatnonzero(months[row] == month_numbers[6])[0])] += 0.35
    seasonal_prepared = prepare_history(seasonal_values, dates, months, params)
    plain = detect(live, prepared, False, params, month_numbers, today)[6]
    seasonal = detect(live, seasonal_prepared, False, params, month_numbers, today)[6]
    assert plain.priority == "HIGH" and not plain.seasonal_normal
    assert seasonal.seasonal_normal and seasonal.priority == "MEDIUM"  # one level lower, still reported
    assert meets_priority(seasonal, "MEDIUM") and not meets_priority(seasonal, "HIGH")


def test_seasonal_baseline_needs_two_earlier_years():
    values, dates, months = smooth_history(rows=300)  # about 14 months of history only
    params = CurveParams()
    prepared = prepare_history(values, dates, months, params)
    live = values[-1].copy()
    live[6] += 0.35
    result = detect(live, prepared, False, params, [int(m) for m in months[-1]], date(2023, 3, 1))[6]
    assert result.seasonal_z is None and not result.seasonal_normal


def test_a_flat_curve_cannot_produce_enormous_z_scores_thanks_to_the_scale_floor():
    flat = np.full((200, 15), 80.0)
    dates = pd.date_range("2023-01-02", periods=200, freq="B").values
    params = CurveParams()
    prepared = prepare_history(flat, dates, None, params)
    live = np.full(15, 80.0)
    live[5] += 0.004  # less than the 0.005 floor
    assert not any(is_kink(r) for r in detect(live, prepared, True, params))


def test_lower_threshold_flags_more():
    values, dates, months = smooth_history()
    live = values[-1].copy()
    live[6] += 0.12
    strict = CurveParams(z_fit=6, z_neighbour=6, z_pca=6, z_history=6)
    loose = CurveParams(z_fit=1.5, z_neighbour=1.5, z_pca=1.5, z_history=1.5)
    n_strict = sum(is_kink(r) for r in detect(live, prepare_history(values, dates, months, strict), True, strict))
    n_loose = sum(is_kink(r) for r in detect(live, prepare_history(values, dates, months, loose), True, loose))
    assert n_loose > n_strict


# ---------- settings ----------


def test_validate_params_accepts_defaults_and_rejects_bad_values():
    defaults = CurveParams().__dict__
    assert validate_params(defaults) == CurveParams()
    with pytest.raises(ValueError):
        validate_params({**defaults, "z_fit": 0.1})
    with pytest.raises(ValueError):
        validate_params({**defaults, "min_methods_high": 9})
    with pytest.raises(ValueError):
        validate_params({**defaults, "alert_min_priority": "URGENT"})
    with pytest.raises(ValueError):
        validate_params({**defaults, "lookback_days": None})


def test_params_roundtrip_through_the_settings_table(tmp_db_path):
    from db.repository import Repository
    repo = Repository(str(tmp_db_path))
    assert load_params(repo) == CurveParams()
    save_params(repo, CurveParams(z_fit=4.5, alerts_enabled=False))
    loaded = load_params(repo)
    assert loaded.z_fit == 4.5 and loaded.alerts_enabled is False
    repo.set_setting("curve_params", {"z_fit": "garbage"})
    assert load_params(repo) == CurveParams()  # bad stored values fall back to defaults


def test_validate_open_time():
    assert validate_open_time("7:05", "Europe/London") == ("07:05", "Europe/London")
    for bad in ("25:00", "abc", "", None):
        with pytest.raises(ValueError):
            validate_open_time(bad, "UTC")
    with pytest.raises(ValueError):
        validate_open_time("07:00", "Mars/Olympus")
