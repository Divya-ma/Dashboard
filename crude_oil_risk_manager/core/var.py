"""Value-at-Risk: two methods that the VaR & Scenarios page and the New Structure builder both run.

HISTORICAL SIMULATION (`historical_var`, `portfolio_pnl_var`). The book's P&L for each of the last N days is
sum(price change x dollar weight) over its symbols; VaR is the loss at the (1 - confidence) percentile of those days,
always a POSITIVE number. It uses what really happened, including its fat tails, and needs no assumption about volatility.
Weaknesses: it assumes the past repeats, it is limited by the shortest-history symbol (a date only counts when every symbol
has a price), and it uses price differences, so absolute dollar moves drive it.

MONTE CARLO (`monte_carlo_var`). Draws many correlated one-day moves and prices the book on them:
  move_i = sigma_i x (correlated standard draw) + mean_i,      P&L = sum(dollar weight_i x move_i)
The correlation is the app-wide one (core.correlation: Pearson, the rolling window averaged over the lookback), repaired
into a valid correlation matrix before it is used. sigma_i, the size of one symbol's daily move, is either

  history   its own standard deviation of daily price changes over the last N days (the same N and the same prices the
            historical method uses). Default. Each symbol moves as much as it really does, so a fly is not shaken as
            hard as an outright, and the two methods agree to within sampling noise when returns are near normal.
  fixed     one number of points for every symbol (the VaR page's Std Dev box). A blunt tool: 0.5 points on a fly is
            $500 a lot a day, far more than a fly really moves, which is why this mode gave enormous VaRs.

Normal draws have a standard deviation of sigma_i; Uniform draws are scaled to the same sigma_i (history) or run from
Low to High (fixed). Both methods work on the symbols as quoted (core.positions.dollar_weights), so a fly has its own
price history.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.data_loader import DataLoader
from core.exceptions import CrudeOilRiskError, InsufficientDataError
from core.models import Structure
from core.positions import dollar_weights

_MIN_SCENARIOS = 20

# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _validate_inputs(confidence: float, horizon_days: int) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1 (exclusive), got {confidence}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")


def _var_from_scenarios(scenario_pnl: np.ndarray, confidence: float) -> tuple[float, float]:
    """Return (var, percentile_cutoff); var = max(0, -percentile), so never negative."""
    cutoff = float(np.percentile(scenario_pnl, (1.0 - confidence) * 100.0))
    return max(0.0, -cutoff), cutoff


def _combine_scenarios(pnl_by_symbol: dict[str, pd.Series]) -> pd.Series:
    """Sum per-symbol scenario P&L on dates where ALL symbols have data (inner join)."""
    frame = pd.concat(list(pnl_by_symbol.values()), axis=1, join="inner", keys=list(pnl_by_symbol))
    return frame.sum(axis=1)


@dataclass
class PortfolioVarResult:
    """VaR of a pre-built portfolio PnL series, or why there is none."""

    pnl: pd.Series | None = None
    var: dict[float, float] = field(default_factory=dict)  # confidence -> VaR (positive loss)
    cutoff: dict[float, float] = field(default_factory=dict)  # confidence -> percentile PnL (negative in a loss)
    observations: int = 0
    requested: int = 0
    skipped: dict[str, str] = field(default_factory=dict)  # symbol -> why it was left out
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def var_from_pnl_series(pnl: pd.Series, confidences: tuple[float, ...] = (0.95, 0.99)) -> tuple[dict, dict]:
    """(var, cutoff) by confidence for a daily PnL series, using the engine's percentile rule.

    VaR is the negative of the (1 - confidence) percentile, floored at 0. Historical
    simulation only; the series is used as given (no scaling, no smoothing).
    """
    values = pnl.to_numpy(dtype=float)
    if values.size == 0:
        raise InsufficientDataError("The portfolio PnL series is empty.")
    var, cutoff = {}, {}
    for confidence in confidences:
        _validate_inputs(confidence, 1)
        var[confidence], cutoff[confidence] = _var_from_scenarios(values, confidence)
    return var, cutoff


# ----------------------------------------------------------------------
# Historical simulation
# ----------------------------------------------------------------------

# Below this many daily observations the result is still shown, with a warning.
RELIABLE_OBSERVATIONS = 10
_SERIES_MIN_DIFF_ROWS = 2


def pnl_series(weights: dict[str, float], data_loader: DataLoader) -> tuple[pd.Series | None, dict[str, str]]:
    """Daily P&L series of a book given as {symbol: dollars per 1.0-point move}, and the symbols left out.

    pnl[t] = sum over symbols of price_diff[t] x weight, inner-joined on date. Symbols without usable price history are
    skipped (reported in the second dict) rather than zero-filled; this understates the book. None (with an empty dict) if
    there are no weights; None (with `skipped` filled) if no symbol has data or the symbols share no trading day.
    """
    if not weights:
        return None, {}
    series: dict[str, pd.Series] = {}
    skipped: dict[str, str] = {}
    for symbol, weight in weights.items():
        try:
            if data_loader.get_available_date_range(symbol) is None:
                raise CrudeOilRiskError("no price data")
            series[symbol] = data_loader.load_price_differences(symbol, min_rows=_SERIES_MIN_DIFF_ROWS) * weight
        except CrudeOilRiskError as exc:
            skipped[symbol] = str(exc)
    if not series:
        return None, skipped
    combined = _combine_scenarios(series).dropna().sort_index()
    return (combined if not combined.empty else None), skipped


def portfolio_pnl_series(structures: list[Structure], data_loader: DataLoader) -> tuple[pd.Series | None, dict[str, str]]:
    """The open structures' combined daily P&L series (see pnl_series), by the legs as quoted."""
    return pnl_series(dollar_weights(structures), data_loader)


def historical_var(
    weights: dict[str, float], lookback_days: int, data_loader: DataLoader, confidences: tuple[float, ...] = (0.95, 0.99),
) -> PortfolioVarResult:
    """1-day historical VaR of a book ({symbol: dollars per point}) over the last `lookback_days` common days.

    Symbols without price history are skipped and reported (this understates VaR); nothing is backfilled or zero-filled.
    Fewer common days than requested uses them all; fewer than RELIABLE_OBSERVATIONS still returns a result with a warning.
    """
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
    result = PortfolioVarResult(requested=lookback_days)
    if not weights:
        result.error = "No open structures — nothing to analyze"
        return result
    combined, result.skipped = pnl_series(weights, data_loader)
    if result.skipped:
        result.warnings.append(
            f"Skipped (no usable price data): {', '.join(result.skipped)}. VaR excludes these legs and is understated."
        )
    if combined is None:
        result.error = (
            "No price data for any open leg — cannot compute VaR" if len(result.skipped) == len(weights)
            else "The open legs share no common trading days — cannot compute VaR"
        )
        return result
    result.pnl = combined.iloc[-lookback_days:]
    result.observations = len(result.pnl)
    result.var, result.cutoff = var_from_pnl_series(result.pnl, confidences)
    if result.observations < RELIABLE_OBSERVATIONS:
        result.warnings.append("Insufficient data for reliable VaR")
    return result


def portfolio_pnl_var(
    structures: list[Structure], lookback_days: int, data_loader: DataLoader, confidences: tuple[float, ...] = (0.95, 0.99),
) -> PortfolioVarResult:
    """1-day historical VaR of the open structures (see historical_var)."""
    return historical_var(dollar_weights(structures), lookback_days, data_loader, confidences)


# ----------------------------------------------------------------------
# Monte Carlo
# ----------------------------------------------------------------------

DISTRIBUTION_NORMAL = "normal"
DISTRIBUTION_UNIFORM = "uniform"
VOL_HISTORY = "history"
VOL_FIXED = "fixed"

DEFAULT_MC_SIMULATIONS = 10_000
_MAX_MC_SIMULATIONS = 200_000
_MIN_MC_SIMULATIONS = 100
DEFAULT_MC_LOOKBACK = 60  # days of price changes the volatility of each symbol is measured over (history mode)
MIN_VOL_ROWS = 10
_UNIFORM_LIMIT = float(np.sqrt(3.0))  # a Uniform(-sqrt3, +sqrt3) draw has mean 0 and standard deviation 1

# The VaR page's Monte Carlo controls. The page saves what it is showing under KEY_VAR_MC and the Structure Builder reads it
# back, so the two always run the same Monte Carlo.
KEY_VAR_MC = "var_mc_params"
DEFAULT_MC_MEAN = 0.0
DEFAULT_MC_STD = 0.5  # points, used in fixed mode (and for a symbol with too little history in history mode)
DEFAULT_MC_LOW = -1.0
DEFAULT_MC_HIGH = 1.0


@dataclass
class McSettings:
    distribution: str = DISTRIBUTION_NORMAL
    dist_params: dict = field(default_factory=lambda: {"mean": DEFAULT_MC_MEAN, "std": DEFAULT_MC_STD})
    n_simulations: int = DEFAULT_MC_SIMULATIONS
    vol_mode: str = VOL_HISTORY
    lookback_days: int = DEFAULT_MC_LOOKBACK


def load_mc_settings(repository) -> McSettings:
    """The Monte Carlo controls as last shown on the VaR page, or the page's defaults."""
    raw = {}
    try:
        saved = repository.get_setting(KEY_VAR_MC, None) if repository is not None else None
        raw = saved if isinstance(saved, dict) else {}
    except Exception:  # noqa: BLE001 - a settings read failure must not stop a VaR estimate
        raw = {}

    def number(name: str, default: float) -> float:
        try:
            return float(raw.get(name, default))
        except (TypeError, ValueError):
            return default

    distribution = raw.get("distribution") if raw.get("distribution") in (DISTRIBUTION_NORMAL, DISTRIBUTION_UNIFORM) else DISTRIBUTION_NORMAL
    mean = number("mean", DEFAULT_MC_MEAN)
    params = (
        {"mean": mean, "std": max(number("std", DEFAULT_MC_STD), 0.0)}
        if distribution == DISTRIBUTION_NORMAL
        else {"low": number("low", DEFAULT_MC_LOW), "high": number("high", DEFAULT_MC_HIGH)}
    )
    simulations = min(max(int(number("n_simulations", DEFAULT_MC_SIMULATIONS)), _MIN_MC_SIMULATIONS), _MAX_MC_SIMULATIONS)
    vol_mode = raw.get("vol_mode") if raw.get("vol_mode") in (VOL_HISTORY, VOL_FIXED) else VOL_HISTORY
    lookback = max(int(number("lookback_days", DEFAULT_MC_LOOKBACK)), MIN_VOL_ROWS)
    return McSettings(distribution, params, simulations, vol_mode, lookback)


def save_mc_settings(repository, distribution: str, mean, std, low, high, n_simulations, vol_mode: str = VOL_HISTORY, lookback_days: int = DEFAULT_MC_LOOKBACK) -> None:
    """Remember the VaR page's Monte Carlo controls (written only when they changed)."""
    values = {"distribution": distribution, "mean": mean, "std": std, "low": low, "high": high, "n_simulations": n_simulations,
              "vol_mode": vol_mode, "lookback_days": lookback_days}
    try:
        if repository.get_setting(KEY_VAR_MC, None) != values:
            repository.set_setting(KEY_VAR_MC, values)
    except Exception:  # noqa: BLE001 - remembering the controls is a convenience
        pass


def _nearest_correlation_psd(matrix: np.ndarray) -> np.ndarray:
    """Repair a correlation matrix into a valid (symmetric, PSD, unit-diagonal) one.

    Pairwise-estimated correlation matrices (each cell from its own best-available date
    range, as build_correlation_matrix produces) are not guaranteed positive semi-definite,
    and missing pairs are NaN. NaNs are treated as uncorrelated (0.0), the matrix is
    symmetrized, negative eigenvalues are clipped to a small positive floor, and the result
    is rescaled back to unit diagonal so it stays a valid correlation matrix for Cholesky.
    """
    clean = np.nan_to_num(matrix, nan=0.0)
    np.fill_diagonal(clean, 1.0)
    clean = (clean + clean.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(clean)
    eigvals = np.clip(eigvals, 1e-10, None)
    repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _validate_distribution(distribution: str, dist_params: dict) -> None:
    if distribution == DISTRIBUTION_NORMAL:
        if dist_params.get("std", 0) < 0:
            raise ValueError(f"std must be >= 0, got {dist_params.get('std')}")
    elif distribution == DISTRIBUTION_UNIFORM:
        if dist_params.get("low", 0) > dist_params.get("high", 0):
            raise ValueError(
                f"low ({dist_params.get('low')}) must be <= high ({dist_params.get('high')})"
            )
    else:
        raise ValueError(f"Unknown distribution '{distribution}', expected '{DISTRIBUTION_NORMAL}' or '{DISTRIBUTION_UNIFORM}'")


def historical_volatilities(symbols: list[str], data_loader: DataLoader, lookback_days: int) -> tuple[dict[str, float], dict[str, str]]:
    """({symbol: standard deviation of its daily price changes over the last `lookback_days`}, {symbol: why none})."""
    sigma: dict[str, float] = {}
    missing: dict[str, str] = {}
    for symbol in symbols:
        try:
            changes = data_loader.load_price_differences(symbol, min_rows=MIN_VOL_ROWS).tail(lookback_days)
        except CrudeOilRiskError as exc:
            missing[symbol] = str(exc)
            continue
        sigma[symbol] = float(changes.std(ddof=1)) if len(changes) > 1 else 0.0
    return sigma, missing


def simulate_correlated_pnl(
    correlation_matrix: pd.DataFrame,
    dollar_weights: dict[str, float],
    distribution: str,
    dist_params: dict,
    n_simulations: int = DEFAULT_MC_SIMULATIONS,
    seed: int | None = None,
    sigma: dict[str, float] | None = None,
) -> np.ndarray:
    """Simulated portfolio P&L for `n_simulations` correlated one-day moves.

    z are independent standard draws (Normal(0, 1), or Uniform scaled to standard deviation 1), one column per symbol;
    L = cholesky(nearest-PSD(C)) makes them correlated; each symbol's move is  (z @ L.T) x sigma_i + mean_i.
    With `sigma` ({symbol: points}) each symbol moves by its own size; without it every symbol moves by the distribution's
    (the fixed mode): Normal(mean, std) or Uniform(low, high). `dollar_weights` is dollar P&L per 1.0-point move for each symbol.
    Symbols with a zero weight contribute nothing but are still simulated (so the requested correlation structure is
    respected even for hedge legs with a net-zero weight).
    """
    if not dollar_weights:
        return np.zeros(0, dtype=float)
    if not _MIN_MC_SIMULATIONS <= n_simulations <= _MAX_MC_SIMULATIONS:
        raise ValueError(f"n_simulations must be between {_MIN_MC_SIMULATIONS} and {_MAX_MC_SIMULATIONS}, got {n_simulations}")
    _validate_distribution(distribution, dist_params)

    symbols = list(dollar_weights)
    matrix = correlation_matrix.reindex(index=symbols, columns=symbols).to_numpy(dtype=float)
    cholesky_factor = np.linalg.cholesky(_nearest_correlation_psd(matrix))

    rng = np.random.default_rng(seed)
    n = len(symbols)
    if distribution == DISTRIBUTION_NORMAL:
        draws = rng.normal(0.0, 1.0, size=(n_simulations, n))
        fixed_sigma, fixed_mean = float(dist_params.get("std", 1.0)), float(dist_params.get("mean", 0.0))
    else:
        draws = rng.uniform(-_UNIFORM_LIMIT, _UNIFORM_LIMIT, size=(n_simulations, n))
        low, high = float(dist_params.get("low", -1.0)), float(dist_params.get("high", 1.0))
        fixed_sigma, fixed_mean = (high - low) / np.sqrt(12.0), (high + low) / 2.0
    if sigma is None:
        scale, shift = np.full(n, fixed_sigma), np.full(n, fixed_mean)
    else:
        scale = np.array([sigma.get(s, fixed_sigma) for s in symbols], dtype=float)
        shift = np.full(n, fixed_mean if distribution == DISTRIBUTION_NORMAL else 0.0)

    moves = (draws @ cholesky_factor.T) * scale + shift
    return moves @ np.array([dollar_weights[s] for s in symbols])


@dataclass
class MonteCarloVarResult:
    """Monte Carlo VaR of a set of correlated positions, or why there is none."""

    pnl: pd.Series | None = None
    var: dict[float, float] = field(default_factory=dict)
    cutoff: dict[float, float] = field(default_factory=dict)
    n_simulations: int = 0
    distribution: str = DISTRIBUTION_NORMAL
    dist_params: dict = field(default_factory=dict)
    vol_mode: str = VOL_HISTORY
    vol_lookback: int = 0
    sigma: dict[str, float] = field(default_factory=dict)  # points per symbol (history mode)
    symbols_included: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def observations(self) -> int:
        """Alias so callers that render historical- and Monte-Carlo-VaR results the same way
        (e.g. ui.layouts.var_tab.build_var_cards) can read a common "how many scenarios" field."""
        return 0 if self.pnl is None else len(self.pnl)

    @property
    def requested(self) -> int:
        return self.n_simulations


def monte_carlo_var(
    dollar_weights: dict[str, float],
    data_loader: DataLoader,
    distribution: str,
    dist_params: dict,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    rolling: int = 5,
    lookback: int = 20,
    n_simulations: int = DEFAULT_MC_SIMULATIONS,
    seed: int | None = None,
    vol_mode: str = VOL_HISTORY,
    vol_lookback: int = DEFAULT_MC_LOOKBACK,
) -> MonteCarloVarResult:
    """Correlation-matrix Monte Carlo VaR for a book of dollar-weighted positions (see the module docstring).

    The correlation matrix is core.correlation.build_correlation_matrix (the average over the last `lookback` days of the
    daily `rolling`-day correlation). A pair with no usable history is left uncorrelated rather than failing the run, and a
    warning says so. In history mode each symbol's volatility is measured over its last `vol_lookback` daily changes; a
    symbol with too little history falls back to the fixed size, with a warning. A Monte Carlo run never silently drops a
    position the way historical VaR does for missing price history.
    """
    from core.correlation import build_correlation_matrix  # local import: avoids a cycle at module load

    result = MonteCarloVarResult(distribution=distribution, dist_params=dict(dist_params), n_simulations=n_simulations,
                                 vol_mode=vol_mode, vol_lookback=vol_lookback)
    if not dollar_weights:
        result.error = "No positions — nothing to simulate"
        return result
    for confidence in confidence_levels:
        _validate_inputs(confidence, 1)

    symbols = sorted(dollar_weights)
    result.symbols_included = symbols
    if len(symbols) == 1:
        matrix = pd.DataFrame([[1.0]], index=symbols, columns=symbols)
    else:
        matrix = build_correlation_matrix(symbols, rolling, lookback, data_loader)
        if matrix.isna().to_numpy().any():
            result.warnings.append(
                "Some symbol pairs had insufficient shared history; those pairs are treated as "
                "uncorrelated (0.0), which may understate diversification benefit or risk."
            )

    sigma = None
    if vol_mode == VOL_HISTORY:
        sigma, missing = historical_volatilities(symbols, data_loader, vol_lookback)
        result.sigma = dict(sigma)
        if missing:
            result.warnings.append(
                f"No usable history to measure the volatility of {', '.join(missing)}; the fixed size of the draws is used for "
                f"{'it' if len(missing) == 1 else 'them'}."
            )

    try:
        pnl = simulate_correlated_pnl(matrix, dollar_weights, distribution, dist_params, n_simulations, seed, sigma)
    except ValueError as exc:
        result.error = str(exc)
        return result

    result.pnl = pd.Series(pnl)
    for confidence in confidence_levels:
        result.var[confidence], result.cutoff[confidence] = _var_from_scenarios(pnl, confidence)
    return result


# ----------------------------------------------------------------------
# Beta vs portfolio / position sizing
# ----------------------------------------------------------------------


def portfolio_outright_pnl_series(
    structures: list[Structure], data_loader: DataLoader, settlements
) -> tuple[pd.Series | None, dict[str, str], list[str]]:
    """The open structures' daily P&L series from their outrights' settlements.

    portfolio[t] = S[t] - S[t-1] with S[t] = sum(outright dollar weight * settlement[t]) on the dates where every
    outright has a settlement (the price is combined first, then differenced, as in core.correlation.structure_series).
    An outright with no usable history is left out and reported in the second value (this understates the portfolio);
    the third value lists outrights priced from a proxy (BZ from ICE Brent). Returns (None, skipped, proxied) when
    there is nothing to build.
    """
    from core.correlation import _leg_closes  # local import: avoids a cycle at module load

    weights = dollar_weights(structures, decompose=True)
    levels: dict[str, pd.Series] = {}
    skipped: dict[str, str] = {}
    proxied: list[str] = []
    for symbol, weight in weights.items():
        closes, via_proxy = _leg_closes(symbol, data_loader, settlements)
        if closes is None or closes.empty:
            skipped[symbol] = "no price history"
            continue
        levels[symbol] = closes * weight
        if via_proxy:
            proxied.append(symbol)
    if not levels:
        return None, skipped, proxied
    price = pd.concat(levels.values(), axis=1, join="inner").dropna().sum(axis=1).sort_index()
    diffs = price.diff().dropna()
    return (diffs if not diffs.empty else None), skipped, proxied


def beta_vs_portfolio(
    candidate_series: pd.Series,
    candidate_lots: float,
    reference_structures: list[Structure],
    data_loader: DataLoader,
    window: int,
    settlements=None,
) -> dict:
    """Beta of the reference structures' combined P&L on a candidate's own P&L series, and
    the lot count at which the candidate would match or hedge it (regression-based position
    sizing).

    portfolio_pnl = alpha + beta * candidate_pnl + eps, with candidate_pnl already scaled to
    `candidate_lots` (a structure's structure_difference_series, or an outright decomposition
    times lots). beta is dollars-of-portfolio-move per dollar-of-candidate-move at
    candidate_lots; core.regression.position_sizing_from_beta turns that into
    lots_to_match/lots_to_hedge. `reference_structures` is the book to regress against — pass
    every open structure for a brand-new candidate, or every OTHER open structure to see how
    one already-open structure relates to the rest (self-regression would inflate beta).

    With `settlements` (core.settlement_prices.SettlementPrices) the book is built from its OUTRIGHTS' daily
    settlements (portfolio_outright_pnl_series), which have a long, gap-free history. Without it the book is built from
    the leg symbols' own candles (portfolio_pnl_series), where every leg must trade on the same days, so one thinly
    traded leg can leave almost no common days.

    Returns {"error": str} if the reference book has no open traded position or no usable price data,
    or if the regression itself is underdetermined (see core.regression.beta_from_series).
    Never raises.
    """
    from core.regression import beta_from_series, position_sizing_from_beta  # local import: avoids a cycle at module load

    if not dollar_weights(reference_structures):
        return {"error": "The portfolio has no open traded positions yet (a shell structure carries no lots), so there is nothing to size against."}
    proxied: list[str] = []
    if settlements is not None:
        portfolio_series, skipped, proxied = portfolio_outright_pnl_series(reference_structures, data_loader, settlements)
        skipped_text = dict(skipped)
    else:
        portfolio_series, skipped_text = portfolio_pnl_series(reference_structures, data_loader)
    if portfolio_series is None:
        detail = f" (no price history for {', '.join(skipped_text)})" if skipped_text else ""
        return {"error": f"No portfolio price data available to regress against{detail}."}
    try:
        beta_result = beta_from_series(portfolio_series, candidate_series, window)
    except InsufficientDataError as exc:
        return {"error": str(exc)}
    sized = position_sizing_from_beta(beta_result, candidate_lots)
    sized["skipped_from_portfolio"] = skipped_text
    sized["proxied"] = proxied
    sized["error"] = None
    return sized
