"""Price an exchange-quoted structure that /fairvalue/ does not return, from simpler contracts.

Far-month spreads, flies and condors are often missing from /fairvalue/ (BRNV28-X28-Z28, say) although
the outrights and the 1-month spreads around them are priced. Each way to build one is a list of
(component symbol, weight); the first way whose components are all priced counts, and the result is
timed at its OLDEST component.

The components must have been priced within CONSISTENT_SECONDS of each other. Subtracting an outright
priced now from one priced hours ago gives a spread that is wrong by the market's move in between, which
for a structure worth cents is a large error. A way that mixes times is skipped, and no price is better
than a misleading one (the caller then reports the structure as missing).

    spread  A-B        = A - B
    fly     A-B-C      = (A-B) - (B-C)   or   A - 2B + C
    condor  A-B-C+D    = (A-B) - (C-D)   or   A - B - C + D

Symbols here are INTERNAL (CLZ26, BRNV28-X28-Z28); the caller translates them for the API. No I/O.
"""

from core.structure_utils import DEFAULT_RATIOS, parse_symbol

CONSISTENT_SECONDS = 1_800.0


def derivation_options(symbol: str) -> list[list[tuple[str, float]]]:
    """Ways to build `symbol` from other symbols, simplest pieces last: [[(component, weight), ...], ...].

    Empty for an outright, a symbol that does not parse, or one with no default ratios."""
    try:
        legs = parse_symbol(symbol)
    except ValueError:
        return []
    ratios = DEFAULT_RATIOS.get(len(legs))
    if ratios is None or len(legs) < 2:
        return []

    def spread(first: int, second: int) -> str:
        return f"{legs[first]['outright']}-{legs[second]['month']}{legs[second]['year']}"

    options: list[list[tuple[str, float]]] = []
    if len(legs) == 3:
        options.append([(spread(0, 1), 1.0), (spread(1, 2), -1.0)])
    elif len(legs) == 4:
        options.append([(spread(0, 1), 1.0), (spread(2, 3), -1.0)])
    outrights: dict[str, float] = {}
    for leg, ratio in zip(legs, ratios):
        outrights[leg["outright"]] = outrights.get(leg["outright"], 0.0) + ratio
    options.append(list(outrights.items()))
    return options


def components_needed(symbol: str) -> list[str]:
    """Every symbol that `derivation_options` could use, without repeats."""
    return list(dict.fromkeys(component for option in derivation_options(symbol) for component, _ in option))


def derive_price(
    symbol: str, prices: dict[str, tuple[float, float]], consistent: float = CONSISTENT_SECONDS,
) -> tuple[float, float, str] | None:
    """(value, time of the oldest component, how) from `prices` ({symbol: (price, unix seconds)}), or None."""
    for option in derivation_options(symbol):
        if not all(component in prices for component, _ in option):
            continue
        times = [prices[component][1] for component, _ in option]
        if max(times) - min(times) > consistent:
            continue
        value = sum(weight * prices[component][0] for component, weight in option)
        how = " ".join(f"{'+' if weight > 0 else '-'}{abs(weight):g}x{component}" for component, weight in option)
        return value, min(times), how
    return None
