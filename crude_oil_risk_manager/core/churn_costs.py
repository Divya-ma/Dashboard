"""Transaction costs and rebates of churning, per product and per structure.

Every figure is in dollars per lot per SIDE (one entry or one exit). The cost of a side is the
transaction cost (TC) minus the rebate when rebates are switched on; by default they are OFF, so a side costs the
full TC and a churn has to earn its profit after paying all of it. A structure is charged on its NET POSITION in each contract month, not
as one lot: a spread is two lots of outrights, a fly four (+1 -2 +1), a dfly eight, and a month that two
structures both hold in opposite directions nets down. So

    cost per side = sum over product and month of |net lots in that month| x (TC - rebate of the product)

and a round trip is two sides. Quoted CL/BZ structures on NYMEX pay a higher rebate (CL 0.87, BZ 0.81), used
when `combo` is set.

TICKS. A target is set in ticks; `tick_value` is the dollar value of one tick of one lot (CL and Brent:
0.01 x 1,000 barrels = $10; Gasoil: 0.25 x 100 tonnes = $25, which is an ASSUMED contract size, change it in
the table if it is different). The rates are kept in the settings table and edited on the Churn page.
"""

from dataclasses import dataclass

from core.curve_calendar import FAMILY_WEIGHTS
from core.user_settings import parse_number

KEY_CHURN_COSTS = "churn_costs"
KEY_CHURN_COMBO = "churn_combo_rebates"
KEY_CHURN_USE_REBATES = "churn_use_rebates"  # off by default: churning is judged on TC alone

STRUCTURE_LOTS = {"outright": 1, "spread": 2, "fly": 4, "dfly": 8}  # gross lots when every month appears once
STRUCTURE_WEIGHTS = FAMILY_WEIGHTS  # one definition of the structures: core.curve_calendar


@dataclass(frozen=True)
class ProductCost:
    product: str
    tick_size: float
    tick_value: float  # dollars per tick per lot
    tc: float  # per side per lot
    rebate: float  # per side per lot

    @property
    def net_per_side(self) -> float:
        return self.tc - self.rebate


DEFAULT_COSTS = {
    "CL": ProductCost("CL", 0.01, 10.0, 0.89, 0.20),
    "BZ": ProductCost("BZ", 0.01, 10.0, 0.89, 0.77),
    "BRN": ProductCost("BRN", 0.01, 10.0, 0.95, 0.25),
    "WBS": ProductCost("WBS", 0.01, 10.0, 1.02, 0.70),
    "G": ProductCost("G", 0.25, 25.0, 1.05, 0.25),
}
DEFAULT_COMBO_REBATES = {"CL": 0.87, "BZ": 0.81}  # quoted CL/BZ structures on NYMEX


def validate_costs(raw: dict) -> dict[str, ProductCost]:
    """{product: {tick_size, tick_value, tc, rebate}} -> ProductCost, or ValueError with a message safe to show."""
    costs = {}
    for product, default in DEFAULT_COSTS.items():
        row = raw.get(product) or {}
        values = {}
        for name, label in (("tick_size", "tick size"), ("tick_value", "tick value"), ("tc", "TC"), ("rebate", "rebate")):
            value = row.get(name)
            values[name] = getattr(default, name) if value in (None, "") else parse_number(value, f"{product} {label}", minimum=0.0, maximum=100000.0)
        if values["tick_size"] <= 0 or values["tick_value"] <= 0:
            raise ValueError(f"{product}: tick size and tick value must be above 0.")
        costs[product] = ProductCost(product, **values)
    return costs


def validate_combo(raw: dict) -> dict[str, float]:
    out = {}
    for product, default in DEFAULT_COMBO_REBATES.items():
        value = (raw or {}).get(product)
        out[product] = default if value in (None, "") else parse_number(value, f"{product} combo rebate", minimum=0.0, maximum=1000.0)
    return out


def load_costs(repository) -> tuple[dict[str, ProductCost], dict[str, float]]:
    """Saved rates (defaults where unset or unusable)."""
    try:
        costs = validate_costs(repository.get_setting(KEY_CHURN_COSTS, None) or {})
    except (ValueError, TypeError):
        costs = dict(DEFAULT_COSTS)
    try:
        combo = validate_combo(repository.get_setting(KEY_CHURN_COMBO, None) or {})
    except (ValueError, TypeError):
        combo = dict(DEFAULT_COMBO_REBATES)
    return costs, combo


def load_use_rebates(repository) -> bool:
    """Whether rebates are subtracted from the TC (default no: the target has to cover the whole TC)."""
    return bool(repository.get_setting(KEY_CHURN_USE_REBATES, False))


def without_rebates(costs: dict[str, ProductCost]) -> dict[str, ProductCost]:
    """The same table with every rebate set to zero, so a side costs the full TC."""
    return {p: ProductCost(c.product, c.tick_size, c.tick_value, c.tc, 0.0) for p, c in costs.items()}


def save_costs(repository, costs: dict[str, ProductCost], combo: dict[str, float], use_rebates: bool | None = None) -> None:
    repository.set_setting(KEY_CHURN_COSTS, {p: {"tick_size": c.tick_size, "tick_value": c.tick_value, "tc": c.tc, "rebate": c.rebate} for p, c in costs.items()})
    repository.set_setting(KEY_CHURN_COMBO, dict(combo))
    if use_rebates is not None:
        repository.set_setting(KEY_CHURN_USE_REBATES, bool(use_rebates))


def net_lots(legs: list[tuple[str, str, float]]) -> dict[tuple[str, str], float]:
    """Net lots per (product, month) from legs given as (product, month key, signed lots); offsetting legs net down."""
    net: dict[tuple[str, str], float] = {}
    for product, month, lots in legs:
        net[(product, month)] = net.get((product, month), 0.0) + lots
    return net


def side_cost(net: dict[tuple[str, str], float], costs: dict[str, ProductCost], combo: bool = False,
              combo_rebates: dict[str, float] | None = None) -> float:
    """Dollars per side for a net position (see the module docstring). `combo` swaps in the CL/BZ structure rebates."""
    combo_rebates = combo_rebates or DEFAULT_COMBO_REBATES
    total = 0.0
    for (product, _), lots in net.items():
        spec = costs[product]
        rebate = combo_rebates.get(product, spec.rebate) if combo else spec.rebate
        total += abs(lots) * (spec.tc - rebate)
    return total


def structure_side_cost(product: str, family: str, costs: dict[str, ProductCost]) -> float:
    """Dollars per side of one lot of a consecutive-month structure of one product (every month held once)."""
    months = [(product, str(i), w) for i, w in enumerate(STRUCTURE_WEIGHTS[family])]
    return side_cost(net_lots(months), costs)


def round_trip_cost(product: str, family: str, costs: dict[str, ProductCost]) -> float:
    return 2.0 * structure_side_cost(product, family, costs)


def breakeven_ticks(product: str, family: str, costs: dict[str, ProductCost]) -> float:
    """Ticks of price move per lot that just pay for the round trip."""
    return round_trip_cost(product, family, costs) / costs[product].tick_value
