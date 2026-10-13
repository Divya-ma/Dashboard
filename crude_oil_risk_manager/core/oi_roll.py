"""Approximate where OI that left a contract went: the roll attribution.

Exchange OI cannot say whether a new position in a contract is a new trader or someone rolling
from another month, so this is an estimate, built from one day's change in OI per contract:

  donors     contracts whose OI fell      (total decrease D)
  receivers  contracts whose OI rose      (total increase U)
  rolled     R = min(D, U): OI cannot have rolled into more than the receivers gained,
             nor out of more than the donors lost.
  Each donor's rolled-out amount is its share of D times R; it is spread over the receivers in
  proportion to their increases ("the increase of all the contracts will be in a ratio with each
  other; divide the decreasing OI into those ratios").
  What is left is NOT a roll: in a donor, the rest of its fall is OI that exited (liquidation or
  expiry); in a receiver, the rest of its rise is fresh OI (new positions).

A contract whose OI is missing on either day is left out of that day. Pure maths: no I/O.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RollResult:
    rolled_out: pd.Series  # donors: OI that moved to other contracts (positive numbers)
    exited: pd.Series  # donors: the rest of the fall
    rolled_in: pd.Series  # receivers: OI that came from other contracts
    fresh: pd.Series  # receivers: the rest of the rise
    flows: pd.DataFrame  # donor (rows) x receiver (columns): OI estimated to have moved
    total_decrease: float
    total_increase: float

    @property
    def rolled(self) -> float:
        return float(self.rolled_out.sum())


def attribute_rolls(delta: pd.Series) -> RollResult:
    """Roll attribution from one day's OI changes by contract (NaN entries are ignored)."""
    delta = delta.dropna().astype(float)
    donors = delta[delta < 0]
    receivers = delta[delta > 0]
    decrease = float(-donors.sum())
    increase = float(receivers.sum())
    rolled = min(decrease, increase)
    if rolled <= 0:
        zero_d, zero_r = donors * 0.0, receivers * 0.0
        return RollResult(zero_d, -donors, zero_r, receivers.copy(), pd.DataFrame(0.0, index=donors.index, columns=receivers.index), decrease, increase)
    rolled_out = -donors * (rolled / decrease)
    share = receivers / increase
    flows = pd.DataFrame(np.outer(rolled_out.to_numpy(), share.to_numpy()), index=donors.index, columns=receivers.index)
    rolled_in = receivers * (rolled / increase)
    return RollResult(
        rolled_out=rolled_out, exited=-donors - rolled_out, rolled_in=rolled_in,
        fresh=receivers - rolled_in, flows=flows, total_decrease=decrease, total_increase=increase,
    )


def donor_flows(delta: pd.Series, donor: str) -> tuple[float, float, pd.Series]:
    """For one donor contract on one day: (its fall, the part of it that rolled, where that went).

    The Series is indexed by receiving contract. Returns (0.0, 0.0, empty) if the donor's OI did not fall."""
    delta = delta.dropna().astype(float)
    if donor not in delta.index or delta[donor] >= 0:
        return 0.0, 0.0, pd.Series(dtype=float)
    result = attribute_rolls(delta)
    fall = float(-delta[donor])
    return fall, float(result.rolled_out.get(donor, 0.0)), result.flows.loc[donor]


def daily_roll_matrix(delta: pd.DataFrame, next_of: dict[str, str]) -> dict[str, pd.DataFrame]:
    """attribute_rolls for every day (row) at once, for the parts the front-contract check needs.

    `delta` is the change in OI, dates x contracts (NaN = missing, left out of that day). Returns, on the
    same grid, `fall` (a contract's fall in OI that day), `rolled_out` (the part of it estimated to have
    rolled into other contracts) and `to_next` (the part of that which went into the contract named in
    `next_of`, the next delivery month). Equal to attribute_rolls day by day, in a few vectorised steps."""
    d = delta.astype(float)
    known = d.notna()
    fall = (-d).clip(lower=0).where(known, 0.0)
    rise = d.clip(lower=0).where(known, 0.0)
    decrease, increase = fall.sum(axis=1), rise.sum(axis=1)
    rolled = np.minimum(decrease, increase)
    rolled_out = fall.mul((rolled / decrease).where(decrease > 0, 0.0), axis=0)
    share = rise.div(increase.where(increase > 0), axis=0).fillna(0.0)  # each receiver's share of the day's rises
    to_next = pd.DataFrame(0.0, index=d.index, columns=d.columns)
    for contract, following in next_of.items():
        if contract in d.columns and following in d.columns:
            to_next[contract] = rolled_out[contract] * share[following]
    return {"fall": fall, "rolled_out": rolled_out, "to_next": to_next}
