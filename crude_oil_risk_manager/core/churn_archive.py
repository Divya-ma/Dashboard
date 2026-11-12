"""Your real churn trades against the Churn page's simulation.

Every structure in the book (open or closed) whose legs are one quoted instrument the Churn page simulates is matched to
its row, on the side you traded it:

  one leg                        CLK27-M27-N27                      -> that outright, spread or fly
  two legs of one product, +1 -1 CLQ27-U27-V27 minus CLU27-V27-X27  -> the d-fly of those two flies
  a CL leg minus the BZ leg      CLK27-M27-N27 minus BZK27-M27-N27  -> the CL-BZ box of those months

A ROUND TRIP is one exit event (a partial or full exit at one price): its lots, its booked P&L and the transaction cost of
both sides (the exit's cost twice: the entry of those lots cost the same). Exits booked without a P&L (a dissolve) are left
out. The side is the direction of the structure's first trade: bought first = long, sold first = short.

`real_trips` gives one row per contract and side: round trips, lots, days traded, the share that made money after costs,
the average move captured in ticks and the average net $ per lot. `compare` puts the simulation's figures beside them.
Contracts the simulation has no candles for (BZ and WBS alone, expired months) are counted and named, not dropped silently.

`fit_settings` CALIBRATES the two assumptions the trades can speak to, the bid-ask spread paid per curve and the stop. The
simulation is run on each traded contract and side AT YOUR OWN TARGET (the average ticks of your winning round trips there)
for a grid of spreads and stops, and the setting whose simulated ticks per lot (before costs, so the cost table plays no
part) come closest to what you really captured is the fit. A curve needs MIN_FIT_TRIPS round trips to be fitted, and a
setting is only changed when that moves the simulation at least KEEP_WITHIN ticks closer to you: your trades are few, so
a smaller gain is noise. The simulation closes every trade by the end of your hours; round trips you held OVERNIGHT are
counted per curve, because no spread or stop makes an intraday simulation match a trade held for days.
"""

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from core.models.trade import TradeEventType

MONTHS = r"[FGHJKMNQUVXZ]\d{2}(?:-[FGHJKMNQUVXZ]\d{2})*"
_SYMBOL = re.compile(rf"^([A-Z]+?)({MONTHS})$")
PRODUCT_OF_PREFIX = {"CL": "CL", "BRN": "BRN", "CO": "BRN", "G": "G", "CLBZZX": "CLBZ"}
EXITS = (TradeEventType.PARTIAL_EXIT, TradeEventType.STRUCTURE_PARTIAL_EXIT, TradeEventType.FULL_EXIT)
ENTRIES = (TradeEventType.TRADE, TradeEventType.ADD)
COLUMNS = ["product", "key", "direction", "name", "trips", "lots", "days", "win_rate", "ticks", "net_lot", "gross", "cost", "last", "win_ticks", "overnight"]
FIT_SPREADS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)  # ticks tried for a curve's bid-ask spread
FIT_STOPS = (1.0, 2.0, 3.0, 5.0)  # targets tried for the stop
MIN_FIT_TRIPS = 10  # round trips a curve needs before its spread is fitted
KEEP_WITHIN = 0.1  # ticks per lot: a setting is changed only when the fit gets closer to your results by more than this


def split_symbol(symbol: str) -> tuple[str, str] | None:
    """("CL", "K27-M27-N27") of "CLK27-M27-N27"; None for anything that is not a product code and month codes."""
    match = _SYMBOL.match(symbol or "")
    return (match.group(1), match.group(2)) if match else None


def row_key(product: str, symbol: str) -> tuple[str, str] | None:
    """The key of a Churn row: (product, months), a d-fly's months as "first fly/second fly"."""
    parts = [split_symbol(s) for s in symbol.split("/")]
    if any(p is None for p in parts):
        return None
    return product, "/".join(months for _, months in parts)


def structure_key(legs) -> tuple[str, str] | None:
    """The Churn key of a structure's legs (see the module docstring); None when it is not one simulated instrument."""
    ordered = sorted(legs, key=lambda leg: getattr(leg, "leg_order", 0) or 0)
    parts = [split_symbol(leg.contract.symbol) for leg in ordered]
    if not parts or any(p is None for p in parts):
        return None
    if len(parts) == 1:
        product = PRODUCT_OF_PREFIX.get(parts[0][0])
        return (product, parts[0][1]) if product else None
    if len(parts) != 2 or (ordered[0].ratio, ordered[1].ratio) not in ((1, -1), (-1, 1)):
        return None
    (first, months), (second, other) = parts
    if {first, second} == {"CL", "BZ"} and months == other:
        return "CLBZ", months
    if first == second and first in PRODUCT_OF_PREFIX and months != other:
        return PRODUCT_OF_PREFIX[first], f"{months}/{other}"
    return None


def _london_day(stamp) -> pd.Timestamp:
    stamp = pd.Timestamp(stamp)
    return (stamp.tz_convert("Europe/London") if stamp.tzinfo is not None else stamp).normalize()


def real_trips(structures, trades_of, tick_values: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
    """(one row per contract and side, names of structures that could not be matched). `trades_of(structure_id)` returns the
    structure's trades; `tick_values` is $ per tick per lot by Churn product."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    names: dict[tuple, str] = {}
    unmatched: list[str] = []
    for structure in structures:
        trades = sorted(trades_of(structure.structure_id), key=lambda t: t.timestamp)
        entries = [t for t in trades if t.event_type in ENTRIES]
        exits = [t for t in trades if t.event_type in EXITS and t.realized_pnl is not None and t.lots > 0]
        if not entries or not exits:
            continue
        key = structure_key(structure.legs)
        if key is None or key[0] not in tick_values:
            unmatched.append(structure.name)
            continue
        side = "long" if entries[0].direction == "buy" else "short"
        events: dict[tuple, dict] = {}
        for t in exits:  # an exit stored once per leg is one round trip
            opened = max((x.timestamp for x in entries if x.timestamp <= t.timestamp), default=entries[0].timestamp)  # the last entry before this exit
            event = events.setdefault((t.timestamp, t.event_type), {"lots": 0.0, "pnl": 0.0, "cost": 0.0, "ts": t.timestamp, "overnight": _london_day(opened) < _london_day(t.timestamp)})
            event["lots"] = max(event["lots"], float(t.lots))
            event["pnl"] += float(t.realized_pnl)
            event["cost"] += 2.0 * float(getattr(t, "transaction_cost", 0.0) or 0.0)
        groups[(*key, side)] += list(events.values())
        names[(*key, side)] = structure.name
    rows = []
    for (product, months, side), events in groups.items():
        lots = sum(e["lots"] for e in events)
        gross, cost = sum(e["pnl"] for e in events), sum(e["cost"] for e in events)
        winners = [e["pnl"] / (e["lots"] * tick_values[product]) for e in events if e["pnl"] > 0 and e["lots"]]
        rows.append({
            "product": product, "key": months, "direction": side, "name": names[(product, months, side)], "trips": len(events), "lots": lots,
            "days": len({_london_day(e["ts"]) for e in events}), "win_rate": float(np.mean([e["pnl"] - e["cost"] > 0 for e in events])),
            "ticks": gross / (lots * tick_values[product]) if lots else np.nan, "net_lot": (gross - cost) / lots if lots else np.nan,
            "gross": gross, "cost": cost, "last": max(e["ts"] for e in events), "win_ticks": float(np.mean(winners)) if winners else np.nan,
            "overnight": float(np.mean([e["overnight"] for e in events])),
        })
    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.sort_values("trips", ascending=False).reset_index(drop=True), sorted(set(unmatched))


def compare(real: pd.DataFrame, scored: pd.DataFrame | None) -> pd.DataFrame:
    """`real` with the simulation's figures of the same contract and side: label, target, hit rate, net $ per trade, score.
    A contract the simulation does not hold keeps blanks there."""
    out = real.copy()
    for column in ("label", "sim_target", "sim_hit", "sim_net_trade", "sim_score"):
        out[column] = np.nan if column != "label" else ""
    if scored is None or scored.empty or out.empty:
        return out
    lookup = {}
    for r in scored.itertuples():
        key = row_key(r.product, r.symbol)
        if key is not None:
            lookup[(*key, r.direction)] = r
    for i, r in zip(out.index, out.itertuples()):
        match = lookup.get((r.product, r.key, r.direction))
        if match is not None:
            out.loc[i, ["label", "sim_target", "sim_hit", "sim_net_trade", "sim_score"]] = [match.label, match.target, match.hit_prob, match.net_trade, match.score]
    return out


def fit_settings(real: pd.DataFrame, family_of, simulate, spreads: dict[str, float], stop: float) -> dict:
    """The spread per curve and the stop that bring the simulation closest to `real` (see the module docstring).

    `family_of(product, key)` names the curve of a contract (None = not simulated); `simulate(product, key, direction, target,
    spread, stop)` gives the simulated ticks per lot per trade before costs and the hit rate, or None. `spreads` and `stop`
    are the settings in use. Returns {"families": one dict per traded curve, "stop_now", "stop_fit", "changes": {setting: value}}."""
    grid_stops = sorted({*FIT_STOPS, float(stop)})
    sides: dict[str, list[dict]] = defaultdict(list)
    for r in real.itertuples():
        family = family_of(r.product, r.key)
        if family is None or family not in spreads or not r.lots:
            continue
        grid = sorted({*FIT_SPREADS, float(spreads[family])})
        target = max(1, int(round(r.win_ticks))) if np.isfinite(r.win_ticks) else 1
        runs = {(p, s): simulate(r.product, r.key, r.direction, target, p, s) for p in grid for s in grid_stops}
        if any(run is None for run in runs.values()):
            continue
        sides[family].append({"trips": int(r.trips), "lots": float(r.lots), "ticks": float(r.ticks), "overnight": float(r.overnight), "target": target, "runs": runs, "wins": float(r.win_rate)})

    def pooled(family: str, spread: float, at_stop: float, part: int = 0) -> float:
        rows = sides[family]
        return sum(x["lots"] * x["runs"][(spread, at_stop)][part] for x in rows) / sum(x["lots"] for x in rows)

    facts = {}
    for family, rows in sides.items():
        lots, trips = sum(x["lots"] for x in rows), sum(x["trips"] for x in rows)
        facts[family] = {"family": family, "trips": trips, "lots": lots, "ticks": sum(x["lots"] * x["ticks"] for x in rows) / lots,
                         "overnight": sum(x["trips"] * x["overnight"] for x in rows) / trips, "target": sum(x["lots"] * x["target"] for x in rows) / lots,
                         "win_rate": sum(x["trips"] * x["wins"] for x in rows) / trips, "spread_now": float(spreads[family]), "fitted": trips >= MIN_FIT_TRIPS}
    grids = {family: sorted({*FIT_SPREADS, float(spreads[family])}) for family in facts}

    def gap(family: str, spread: float, at_stop: float) -> float:
        return abs(pooled(family, spread, at_stop) - facts[family]["ticks"])

    fitted = [f for f, fact in facts.items() if fact["fitted"]]
    weight = sum(facts[f]["lots"] for f in fitted)

    def miss(at_stop: float) -> float:  # the fitted curves' gap at their best spread, with this stop
        return sum(facts[f]["lots"] * min(gap(f, p, at_stop) for p in grids[f]) for f in fitted) / weight

    stop_fit = float(stop)
    if fitted:
        best = min(grid_stops, key=miss)
        stop_fit = best if miss(stop_fit) - miss(best) > KEEP_WITHIN else stop_fit
    changes = {}
    for family, fact in facts.items():
        now = fact["spread_now"]
        fact["sim_now"] = pooled(family, now, float(stop))
        fact["spread_fit"] = now
        if fact["fitted"]:
            best = min(grids[family], key=lambda p: gap(family, p, stop_fit))
            fact["spread_fit"] = best if gap(family, now, stop_fit) - gap(family, best, stop_fit) > KEEP_WITHIN else now
            if fact["spread_fit"] != now:
                changes[f"spread_{family}"] = fact["spread_fit"]
        fact["sim_fit"], fact["hit_fit"] = pooled(family, fact["spread_fit"], stop_fit), pooled(family, fact["spread_fit"], stop_fit, 1)
    if stop_fit != float(stop):
        changes["stop_mult"] = stop_fit
    return {"families": sorted(facts.values(), key=lambda f: -f["trips"]), "stop_now": float(stop), "stop_fit": stop_fit, "changes": changes}
