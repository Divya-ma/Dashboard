"""Portfolio stop risk: how much the whole book could lose if the stops are hit, allowing for correlation.

        risk  =  sqrt( r · C · rᵀ )

  r   one value per open structure: the loss in dollars if the structure goes from its entry price to its STOP, the stop shown on
      its Home packet (the one typed with the latest entry): |entry - stop| x lots x multiplier, read the way stops are entered and
      alerted everywhere (for a buy the stop is below the entry, for a sell above it), floored at zero (a stop already moved past
      the entry locks in a profit and risks nothing). Its entry price is that of the lots still open (first in, first out).
  C   the correlation matrix of the structures: the Correlation page's "Portfolio positions" matrix, i.e. of each position's daily
      P&L changes AS HELD (a sold structure is flipped), averaged as everywhere (rolling window, then lookback, from Settings).
      Because C is of the positions' own P&L and r is a loss, every r is positive and two positions that offset each other
      (a long and a short in correlated structures) reduce the risk. If every structure is fully correlated (all 1) the result is
      the plain sum of the stops; with no correlation it is the root of the sum of their squares.

A structure with no stop has no r and is left out (and named). Where two positions share too little price history for a
correlation, they are assumed to be fully correlated (1): the figure can only err on the cautious side, and the count of such
pairs is reported. If the matrix is not a valid correlation matrix (pairwise averages need not be) it is repaired the way the VaR page
does before the root is taken.
"""

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

from core.correlation import compute_watchlist_correlation
from core.models import Structure
from core.packets import latest_level, structure_lots, structure_side
from core.structure_view import structure_entry_price
from core.var import _nearest_correlation_psd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StopRisk:
    structure_id: str
    name: str
    side: str
    lots: float
    entry: float | None
    stop: float | None
    risk: float | None  # dollars; None when there is no stop (or no entry)


def structure_stop_risk(structure: Structure, trades: list) -> StopRisk:
    """The loss in dollars if `structure` goes from its entry price to its latest stop (None without a stop)."""
    lots = structure_lots(structure)
    entry = structure_entry_price(structure) if structure.legs and lots > 0 else None
    stop = latest_level(trades, "stop_loss_price")
    risk = None
    if entry is not None and stop is not None and lots > 0:
        adverse = (entry - stop) * (1.0 if structure.legs[0].direction == "buy" else -1.0)  # points lost if the stop is hit
        risk = max(0.0, adverse) * lots * structure.legs[0].contract.multiplier
    return StopRisk(structure.structure_id, structure.name, structure_side(structure), lots, entry, stop, risk)


def combine(risks: np.ndarray, correlation: np.ndarray) -> float:
    """sqrt(r · C · rᵀ), with C repaired to a valid correlation matrix if it is not one."""
    risks = np.asarray(risks, dtype=float)
    matrix = np.asarray(correlation, dtype=float)
    if len(risks) > 1 and np.linalg.eigvalsh((matrix + matrix.T) / 2).min() < -1e-9:
        matrix = _nearest_correlation_psd(matrix)
    return float(np.sqrt(max(risks @ matrix @ risks, 0.0)))


@dataclass
class PortfolioStopRisk:
    value: float | None  # the figure, dollars; None if no open structure has a stop
    sum_of_stops: float = 0.0  # what it would be if everything were fully correlated
    counted: int = 0  # structures in the figure
    open_structures: int = 0
    without_stop: list[str] = field(default_factory=list)
    assumed_pairs: int = 0  # pairs treated as fully correlated for lack of price history
    rolling: int = 0
    lookback: int = 0
    error: str | None = None

    @property
    def diversification(self) -> float | None:
        """Share of the summed stops that correlation takes away (0 = none, as if every pair were fully correlated)."""
        return 1.0 - self.value / self.sum_of_stops if self.value is not None and self.sum_of_stops > 0 else None

    def to_dict(self) -> dict:
        return {**asdict(self), "diversification": self.diversification}


class PortfolioRisk:
    """Works the figure out for the open structures; the correlation matrix is kept until the book's legs, the stored
    settlements or the Settings windows change, so the 5-second Home refresh costs one matrix product."""

    def __init__(self, data_loader_getter, version_getter=lambda: None):
        self._loader, self._version = data_loader_getter, version_getter
        self._cache: dict[tuple, dict | None] = {}

    def _correlations(self, structures: list[Structure], rolling: int, lookback: int) -> dict[tuple[str, str], float]:
        """{(id, id): correlation} of the pairs that have one (both orders)."""
        key = (
            tuple(sorted((s.structure_id, tuple((leg.contract.symbol, leg.ratio, leg.direction) for leg in s.legs if leg.is_traded)) for s in structures)),
            self._version(), rolling, lookback,
        )
        if key not in self._cache:
            if len(self._cache) > 50:
                self._cache.clear()
            items = [{"type": "structure", "key": s.structure_id, "label": s.structure_id} for s in structures]
            result = compute_watchlist_correlation(items, {s.structure_id: s for s in structures}, rolling, lookback, self._loader())
            pairs: dict[tuple[str, str], float] = {}
            if result.matrix is not None:
                matrix = result.matrix
                for a in matrix.index:
                    for b in matrix.columns:
                        value = matrix.loc[a, b]
                        if a != b and np.isfinite(value):
                            pairs[(a, b)] = float(value)
            self._cache[key] = pairs
        return self._cache[key]

    def compute(self, structures: list[Structure], trades_by_structure: dict, rolling: int, lookback: int) -> PortfolioStopRisk:
        open_structures = [s for s in structures if structure_lots(s) > 0]
        result = PortfolioStopRisk(None, open_structures=len(open_structures), rolling=rolling, lookback=lookback)
        risks = [structure_stop_risk(s, trades_by_structure.get(s.structure_id, [])) for s in open_structures]
        result.without_stop = [r.name for r in risks if not r.risk and r.stop is None]
        counted = [(s, r) for s, r in zip(open_structures, risks) if r.risk]  # a stop that risks nothing adds nothing
        result.counted = len(counted)
        if not counted:
            return result
        vector = np.array([r.risk for _, r in counted])
        result.sum_of_stops = float(vector.sum())
        if len(counted) == 1:
            result.value = float(vector[0])
            return result
        try:
            pairs = self._correlations([s for s, _ in counted], rolling, lookback)
        except Exception as exc:  # noqa: BLE001 - never break the Home refresh; fall back to the cautious sum
            logger.exception("Portfolio stop risk: correlation failed")
            pairs, result.error = {}, f"correlation unavailable ({exc.__class__.__name__}); the stops are summed"
        ids = [s.structure_id for s, _ in counted]
        matrix = np.ones((len(ids), len(ids)))
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                if i < j:
                    value = pairs.get((a, b), pairs.get((b, a)))
                    if value is None:
                        result.assumed_pairs += 1
                    else:
                        matrix[i, j] = matrix[j, i] = value
        result.value = combine(vector, matrix)
        return result
