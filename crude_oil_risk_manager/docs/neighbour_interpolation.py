"""Neighbour interpolation: find the one point of a curve that does not fit its neighbours.

Self-contained (numpy only). Given the values of a curve at consecutive positions (for example
15 consecutive contract months), each point is predicted from the points around it by Lagrange
interpolation. The difference between the actual value and that prediction is the RESIDUAL; a
point whose residual is large compared with what is normal for it is a KINK.

    residual_k = y_k - sum_j W[k, j] * y_j          (W has a zero diagonal)

The weights W (see `neighbour_weights`):
    interior points  cubic through k-2, k-1, k+1, k+2        weights (-1/6, 2/3, 2/3, -1/6)
    next to an end   quadratic through the 3 nearest others  weights ( 1/3, 1, -1/3)
    at an end        quadratic extrapolation from 1, 2, 3    weights ( 3, -3, 1)

Every row of weights sums to 1, so a flat curve gives zero residual. The residual is exactly
zero for any cubic curve in the interior and any quadratic curve everywhere.

Functions
    robust_scale        spread of a set of numbers that outliers cannot inflate (floored)
    neighbour_weights   the weight matrix W
    neighbour_residuals residuals of one curve (or of many curves at once); NaN-aware
    implied_values      the value each point "should" have: y - residual
    neighbour_z         z-scores: residual / scale (historical per-position, or cross-sectional)
    find_kinks          the strongest-first search that masks each kink it finds and re-tests

Run `python neighbour_interpolation.py` to execute the self-test and print the worked example.
"""

from dataclasses import dataclass

import numpy as np

MAD_TO_STD = 1.4826  # makes the median absolute deviation comparable with a standard deviation


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


def robust_scale(values, floor: float = 0.0) -> float:
    """1.4826 x median absolute deviation of `values` (NaN ignored), never below `floor`.

    The median and the MAD are barely moved by a few extreme values, so a kink cannot hide
    itself by inflating the scale it is judged against. `floor` stops an almost perfectly
    smooth set of numbers from giving a tiny scale and therefore enormous z-scores.
    """
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    return float(max(MAD_TO_STD * np.nanmedian(np.abs(values - median)), floor))


def neighbour_weights(n: int) -> np.ndarray:
    """W (n x n): row k holds the weights that predict point k from the other points.

    interior  (2 <= k <= n-3): cubic Lagrange through k-2, k-1, k+1, k+2   -> -1/6, 2/3, 2/3, -1/6
    next to an end (k = 1 or n-2): quadratic through the three nearest other points
              (k-1, k+1, k+2) -> 1/3, 1, -1/3, mirrored at the far end
    at an end (k = 0 or n-1): quadratic extrapolation from the next three points -> 3, -3, 1
    Fewer than 4 points: the plain average of the two neighbours (and nothing at the ends).
    """
    w = np.zeros((n, n))
    for k in range(n):
        if 2 <= k <= n - 3:
            w[k, [k - 2, k - 1, k + 1, k + 2]] = (-1 / 6, 2 / 3, 2 / 3, -1 / 6)
        elif k == 1 and n >= 4:
            w[k, [k - 1, k + 1, k + 2]] = (1 / 3, 1, -1 / 3)
        elif k == n - 2 and n >= 4:
            w[k, [k + 1, k - 1, k - 2]] = (1 / 3, 1, -1 / 3)
        elif k == 0 and n >= 4:
            w[k, [1, 2, 3]] = (3, -3, 1)
        elif k == n - 1 and n >= 4:
            w[k, [n - 2, n - 3, n - 4]] = (3, -3, 1)
        elif 0 < k < n - 1:
            w[k, [k - 1, k + 1]] = (0.5, 0.5)
    return w


def neighbour_residuals(values) -> np.ndarray:
    """residual = value - interpolation from the neighbours, along the last axis.

    `values` may be one curve (shape n) or many curves at once (shape T x n, one curve per row).
    A residual is NaN where the point itself is missing or where any neighbour it needs is
    missing (missing prices are never filled in).
    """
    values = np.asarray(values, dtype=float)
    w = neighbour_weights(values.shape[-1])
    missing = np.isnan(values)
    filled = np.where(missing, 0.0, values)
    needs_missing = (missing.astype(float) @ (w != 0).astype(float).T) > 0
    residual = filled - filled @ w.T
    return np.where(needs_missing | missing, np.nan, residual)


def implied_values(values) -> np.ndarray:
    """The value the neighbours say each point should have: y - residual."""
    values = np.asarray(values, dtype=float)
    return values - neighbour_residuals(values)


# ----------------------------------------------------------------------
# z-scores
# ----------------------------------------------------------------------


def neighbour_z(values, history=None, floor: float = 0.005, lookback: int = 504) -> np.ndarray:
    """z-score of each point's residual.

    With `history` (a T x n matrix of PAST curves over the same positions, oldest first), each
    position is judged against its own past: z_k = residual_k / robust_scale(past residuals at k,
    last `lookback` rows, floored). Positions differ a lot (the ends are noisier than the middle),
    so this is the better scale.

    Without history, the scale is cross-sectional: the robust scale of today's residuals across
    the curve (needs at least 3 residuals), which works from day one but is noisier.
    """
    values = np.asarray(values, dtype=float)
    residual = neighbour_residuals(values)
    if history is not None:
        past = neighbour_residuals(np.asarray(history, dtype=float))[-lookback:]
        scales = np.array([robust_scale(past[:, k], floor) for k in range(past.shape[1])])
        return residual / scales
    finite = residual[np.isfinite(residual)]
    if len(finite) < 3:
        return np.full(values.shape, np.nan)
    return residual / robust_scale(finite, floor)


# ----------------------------------------------------------------------
# Finding the real kink
# ----------------------------------------------------------------------


@dataclass
class Kink:
    index: int
    value: float
    implied: float  # what the neighbours say it should be (at the moment it was found)
    residual: float
    z: float
    direction: str  # "rich" (above its neighbours) or "cheap" (below)
    echoes: list[int]  # points that only looked wrong because of this kink


def find_kinks(values, history=None, threshold: float = 3.0, floor: float = 0.005, max_kinks: int = 6) -> list[Kink]:
    """Kinks of one curve, strongest first, each attributed to the right point.

    A point that is off also shifts the prediction of the points around it: a spike of size d at
    point j gives residual d at j, -2d/3 at j-1 and j+1, +d/6 at j-2 and j+2, and at the front
    end the weight 3 turns a spike two places away into 3d. Those neighbours would be flagged as
    well, so the points are taken one at a time:

      1. flag every point with |z| > threshold;
      2. choose the flagged point that, once MASKED (treated as missing), leaves the fewest other
         points flagged (ties: the largest |z|) and record it as a kink;
      3. recompute every z-score without it (points that needed it can no longer be judged);
      4. repeat until nothing is flagged or `max_kinks` are found.

    Step 2 cannot simply take the largest |z|: at the front end a kink two places away shows up
    as 3x its size, so the echo there can have a bigger z than the kink itself. Only the real kink
    explains the others, so masking it is what clears them.

    A point flagged in step 1 that is clean once a kink is masked is an echo of it and is listed in
    that kink's `echoes`.
    """
    working = np.asarray(values, dtype=float).copy()

    def flagged_after_masking(curve, masked=None):
        trial = curve.copy()
        if masked is not None:
            trial[masked] = np.nan
        scores = neighbour_z(trial, history, floor)
        return {int(i) for i in np.flatnonzero(np.abs(scores) > threshold)}, scores

    flagged, z = flagged_after_masking(working)
    kinks: list[Kink] = []
    for _ in range(max_kinks):
        candidates = [i for i in flagged if i not in {k.index for k in kinks}]
        if not candidates:
            break
        top = min(candidates, key=lambda i: (len(flagged_after_masking(working, i)[0] - {i}), -abs(z[i])))
        residual = neighbour_residuals(working)[top]
        kinks.append(
            Kink(top, float(working[top]), float(working[top] - residual), float(residual), float(z[top]),
                 "rich" if z[top] > 0 else "cheap", [])
        )
        working[top] = np.nan
        still, z = flagged_after_masking(working)
        kinks[-1].echoes = sorted(flagged - still - {top} - {k.index for k in kinks})
        flagged = still
    return kinks


# ----------------------------------------------------------------------
# Self-test and worked example
# ----------------------------------------------------------------------


def _example_curve(kink_at: int = 6, size: float = 0.40) -> np.ndarray:
    """15 points on a smooth curve, y = 100 - 1.2 k + 0.03 k^2, with the point at `kink_at` pushed up by `size`."""
    k = np.arange(15)
    y = 100.0 - 1.2 * k + 0.03 * k**2
    y[kink_at] += size
    return y


def _noisy_history_and_curve(seed: int = 0, days: int = 300, n: int = 15):
    """`days` past curves and one new curve, all = a smooth quadratic + 0.03 of price noise."""
    rng = np.random.default_rng(seed)
    x = np.arange(float(n))
    smooth = 100 - 1.2 * x + 0.03 * x**2
    return smooth + rng.normal(0, 0.03, (days, n)), smooth + rng.normal(0, 0.03, n)


def _selftest() -> None:
    n = 15
    w = neighbour_weights(n)
    assert np.allclose(w.sum(axis=1), 1.0) and np.allclose(np.diag(w), 0.0)

    x = np.arange(float(n))
    quadratic = 100 - 1.2 * x + 0.03 * x**2
    assert np.allclose(neighbour_residuals(quadratic), 0.0, atol=1e-9)  # exact everywhere
    cubic = neighbour_residuals(1 + 2 * x - 0.3 * x**2 + 0.01 * x**3)
    assert np.allclose(cubic[2:-2], 0.0, atol=1e-9)  # exact in the interior

    spike = np.zeros(n)
    spike[7] = 1.0
    r = neighbour_residuals(spike)
    assert np.isclose(r[7], 1.0) and np.isclose(r[6], -2 / 3) and np.isclose(r[8], -2 / 3)
    assert np.isclose(r[5], 1 / 6) and np.isclose(r[9], 1 / 6)

    holes = quadratic.copy()
    holes[7] = np.nan
    r = neighbour_residuals(holes)
    assert np.isnan(r[5:10]).all()  # the missing point and the four points that need it
    assert np.isfinite(np.delete(r, np.arange(5, 10))).all()  # everything else is still judged

    curve = quadratic.copy()
    curve[2] += 0.4  # a kink two places from the front end: the front point shows 3 x 0.4 = 1.2
    assert np.isclose(neighbour_residuals(curve)[0], 1.2)
    found = find_kinks(curve)
    assert [k.index for k in found] == [2] and 0 in found[0].echoes  # the front point was only an echo

    # Two kinks need a historical scale: from today's residuals alone, 10 of the 15 are contaminated.
    history, live = _noisy_history_and_curve()
    live[3] += 0.4
    live[11] -= 0.5
    found = find_kinks(live, history)
    assert {k.index for k in found} == {3, 11}
    assert {k.index: k.direction for k in found} == {3: "rich", 11: "cheap"}
    print("self-test passed")


def _print_example(kink_at: int = 6, title: str = "interior kink") -> None:
    y = _example_curve(kink_at)
    residual = neighbour_residuals(y)
    implied = y - residual
    z = neighbour_z(y)
    print(f"\nExample ({title}): y = 100 - 1.2 k + 0.03 k^2, point k = {kink_at} pushed up by 0.40")
    print(f"{'k':>2} {'y':>9} {'implied':>9} {'residual':>9} {'z':>8}")
    for k in range(len(y)):
        print(f"{k:>2} {y[k]:>9.3f} {implied[k]:>9.3f} {residual[k]:>9.3f} {z[k]:>8.1f}")
    print(f"scale used = {robust_scale(residual[np.isfinite(residual)], 0.005):.4f}")
    for kink in find_kinks(y):
        print(f"kink at k={kink.index}: {kink.direction}, value {kink.value:.3f}, implied {kink.implied:.3f}, "
              f"z {kink.z:.1f}, echoes {kink.echoes}")


def _print_history_example() -> None:
    history, live = _noisy_history_and_curve()
    live[3] += 0.4
    live[11] -= 0.5
    residual = neighbour_residuals(live)
    z = neighbour_z(live, history)
    scales = residual / z
    print("\nHistorical example: 300 past curves (smooth + 0.03 noise); today k = 3 is +0.40 and k = 11 is -0.50")
    print(f"{'k':>2} {'y':>9} {'residual':>9} {'scale':>7} {'z':>7}")
    for k in range(len(live)):
        print(f"{k:>2} {live[k]:>9.3f} {residual[k]:>9.3f} {abs(scales[k]):>7.3f} {z[k]:>7.1f}")
    for kink in find_kinks(live, history):
        print(f"kink at k={kink.index}: {kink.direction}, value {kink.value:.3f}, implied {kink.implied:.3f}, "
              f"z {kink.z:.1f}, echoes {kink.echoes}")


if __name__ == "__main__":
    _selftest()
    _print_example(6, "interior kink")
    _print_example(2, "kink two places from the front end")
    _print_history_example()
