"""Symbol parsing and outright decomposition for exchange-quoted structure symbols.

Users type INTERNAL symbols (CLZ26, BRNZ26, CLZ26-F27, CLZ26-F27-G27); the API
code translation (BRN -> CO, ...) is delegated to SymbolTranslator, which is
also the source of the leg identification (`parse_api_symbol`). Nothing here
touches the network or the database.

Note on recursion: a single symbol string cannot nest other symbols, so
`decompose_to_outrights` is one flat step. A structure whose legs are
themselves structure symbols (a spread as one leg of a custom structure) is
handled by `net_outright_equivalent`, which decomposes each leg and sums.
"""

import re
import uuid
from datetime import datetime, timezone

from adapters.base import PRODUCT_TO_API_CODE, SymbolTranslator
from core.models import Structure, StructureStatus

# Default leg ratios by number of legs in an exchange-quoted symbol.
DEFAULT_RATIOS: dict[int, list[int]] = {
    1: [1],
    2: [1, -1],
    3: [1, -2, 1],
    4: [1, -1, -1, 1],
}

STRUCTURE_TEMPLATES: dict[str, dict] = {
    "outright": {
        "label": "Outright", "icon": "📈", "description": "Single contract position", "legs": 1, "ratios": [1],
    },
    "spread": {
        "label": "Spread", "icon": "↕️", "description": "Two-leg calendar spread", "legs": 2, "ratios": [1, -1],
    },
    "fly": {
        "label": "Fly", "icon": "🦋", "description": "Three-leg butterfly", "legs": 3, "ratios": [1, -2, 1],
    },
    "custom": {
        "label": "Custom", "icon": "⚙️", "description": "Custom ratio structure", "legs": None, "ratios": [1],
    },
}

# validate_structure_legs marks blocking problems with this prefix; everything else is a warning.
ERROR_PREFIX = "Error: "

_TENOR_TOKEN = r"[FGHJKMNQUVXZ]\d{2}"
_TENORS_RE = re.compile(rf"{_TENOR_TOKEN}(?:[-+]{_TENOR_TOKEN})*")


def normalize_symbol(symbol: str | None) -> str:
    return (symbol or "").strip().upper().replace(" ", "")


def parse_symbol(symbol: str) -> list[dict]:
    """Legs of an internal symbol: [{"product", "letter", "year", "outright"}, ...].

    "outright" is the internal outright symbol of that leg (e.g. "CLF27").
    Raises ValueError with a readable message if the symbol is malformed.
    """
    sym = normalize_symbol(symbol)
    if not sym:
        raise ValueError("symbol is empty")
    try:
        product, rest = SymbolTranslator._match_product_prefix(sym, PRODUCT_TO_API_CODE)
    except ValueError:
        raise ValueError(f"{sym!r}: product code not recognized (use e.g. CL, BRN, BZ, WBS, G)") from None
    if not _TENORS_RE.fullmatch(rest):
        raise ValueError(f"{sym!r}: expected a month letter and 2-digit year, e.g. {product}Z26 or {product}Z26-F27")

    legs = SymbolTranslator.parse_api_symbol(SymbolTranslator.internal_to_api(sym))
    return [
        {**leg, "outright": f"{leg['product']}{leg['month']}{leg['year']}", "letter": leg["month"]}
        for leg in legs
    ]


def get_structure_type_from_symbol(symbol: str) -> str:
    """'outright' | 'spread' | 'fly' | 'condor' | 'custom', from the '-' / '+' separators."""
    minus, plus = symbol.count("-"), symbol.count("+")
    if minus == 0 and plus == 0:
        return "outright"
    if plus == 0 and minus == 1:
        return "spread"
    if plus == 0 and minus == 2:
        return "fly"
    if plus == 1 and minus == 2:  # front-t1-t2+t3, the format SymbolTranslator emits for condors
        return "condor"
    return "custom"


def decompose_to_outrights(
    structure_symbol: str, lots: float, ratio_override: list[int] | None = None
) -> dict[str, float]:
    """Net lots per outright contract for `lots` of an exchange-quoted structure symbol.

    CLZ26-F27-G27 with lots=10 -> {"CLZ26": 10.0, "CLF27": -20.0, "CLG27": 10.0}.
    Ratios default by leg count (see DEFAULT_RATIOS) unless `ratio_override` is given.
    Raises ValueError for a malformed symbol or a ratio list of the wrong length.
    """
    legs = parse_symbol(structure_symbol)
    ratios = ratio_override if ratio_override is not None else DEFAULT_RATIOS.get(len(legs))
    if ratios is None:
        raise ValueError(f"{structure_symbol!r}: no default ratios for a {len(legs)}-leg symbol")
    if len(ratios) != len(legs):
        raise ValueError(f"{structure_symbol!r} has {len(legs)} legs but {len(ratios)} ratios were given")

    net: dict[str, float] = {}
    for leg, ratio in zip(legs, ratios):
        net[leg["outright"]] = net.get(leg["outright"], 0.0) + ratio * lots
    return net


def net_outright_equivalent(legs: list[dict], lots: float = 1.0) -> tuple[dict[str, float], list[str]]:
    """Sum the outright decomposition of every leg ({"symbol", "ratio"}).

    Returns (net_lots_by_outright, ignored_symbols); incomplete or invalid legs
    are skipped and reported instead of raising, so a live preview shows partial results.
    """
    net: dict[str, float] = {}
    ignored: list[str] = []
    for leg in legs:
        symbol, ratio = normalize_symbol(leg.get("symbol")), leg.get("ratio")
        if not symbol:
            continue
        try:
            if isinstance(ratio, bool) or ratio in (None, 0):
                raise ValueError("no ratio")
            parts = decompose_to_outrights(symbol, lots * float(ratio))
        except (ValueError, TypeError):
            ignored.append(symbol)
            continue
        for outright, leg_lots in parts.items():
            net[outright] = net.get(outright, 0.0) + leg_lots
    return net, ignored


def compose_structure_symbol(leg_symbols: list[str], ratios: list[int] | None = None) -> str | None:
    """Exchange-quoted symbol for outright legs, or None if the legs don't form one.

    One leg -> that symbol itself (it may already be a spread/fly symbol). 2-4 legs
    must be same-product outrights whose ratios are the default pattern (either
    direction), e.g. [CLZ26, CLF27] with [1, -1] -> "CLZ26-F27".
    """
    symbols = [normalize_symbol(s) for s in leg_symbols]
    if not symbols or any(not s for s in symbols):
        return None
    if len(symbols) == 1:
        try:
            parse_symbol(symbols[0])
        except ValueError:
            return None
        return symbols[0]
    if len(symbols) > 4:
        return None
    try:
        parsed = [parse_symbol(s) for s in symbols]
    except ValueError:
        return None
    if any(len(p) != 1 for p in parsed) or len({p[0]["product"] for p in parsed}) != 1:
        return None
    default = DEFAULT_RATIOS[len(symbols)]
    if ratios is not None and list(ratios) not in (default, [-r for r in default]):
        return None

    tokens = [f"{p[0]['letter']}{p[0]['year']}" for p in parsed]
    head = symbols[0][: len(symbols[0]) - len(tokens[0])]
    if len(tokens) == 4:
        return f"{head}{tokens[0]}-{tokens[1]}-{tokens[2]}+{tokens[3]}"
    return head + tokens[0] + "".join(f"-{t}" for t in tokens[1:])


def validate_structure_legs(legs: list[dict]) -> list[str]:
    """Messages about a leg list ({"symbol", "ratio"} dicts); empty list = nothing to report.

    Blocking problems start with ERROR_PREFIX (no legs, missing/invalid symbol,
    zero or non-integer ratio); duplicate symbols and cross-product legs are
    warnings only. Symbol checks are format-only, never an API call.
    """
    messages: list[str] = []
    if not legs:
        return [f"{ERROR_PREFIX}A structure needs at least one leg."]

    parsed: dict[int, list[dict]] = {}
    for n, leg in enumerate(legs, start=1):
        symbol = normalize_symbol(leg.get("symbol"))
        if not symbol:
            messages.append(f"{ERROR_PREFIX}Leg {n}: symbol is required.")
        else:
            try:
                parsed[n] = parse_symbol(symbol)
            except ValueError as exc:
                messages.append(f"{ERROR_PREFIX}Leg {n}: {exc}.")

        ratio = leg.get("ratio")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio != int(ratio) or ratio == 0:
            messages.append(f"{ERROR_PREFIX}Leg {n}: ratio must be a non-zero whole number.")

    seen: dict[str, int] = {}
    for n, leg in enumerate(legs, start=1):
        symbol = normalize_symbol(leg.get("symbol"))
        if symbol and n in parsed:
            if symbol in seen:
                messages.append(f"Duplicate symbol {symbol} in legs {seen[symbol]} and {n}; check this is intended.")
            else:
                seen[symbol] = n

    products = {p[0]["product"] for p in parsed.values()}
    if len(products) > 1:
        messages.append(f"Cross-product structure ({', '.join(sorted(products))}); check the multiplier suits every leg.")
    return messages


def split_validation_messages(messages: list[str]) -> tuple[list[str], list[str]]:
    """(errors, warnings) from validate_structure_legs output; the error prefix is stripped."""
    errors = [m[len(ERROR_PREFIX):] for m in messages if m.startswith(ERROR_PREFIX)]
    warnings = [m for m in messages if not m.startswith(ERROR_PREFIX)]
    return errors, warnings


def clone_structure_as_shell(source: Structure, new_name: str | None = None) -> Structure:
    """A fresh SHELL structure with the same legs (symbols, ratios, contracts) as `source`.

    Meant for reusing a closed structure: new structure and leg ids, no lots, no entry
    prices, direction reset to "buy", no close trigger/time. The name defaults to
    "<source name> (reuse)" (cut to the 100-character limit). The caller saves it.
    """
    now = datetime.now(timezone.utc)
    legs = [
        leg.model_copy(
            update={
                "leg_id": str(uuid.uuid4()),
                "lots": 0.0,
                "entry_price": None,
                "average_entry_price": None,
                "direction": "buy",
                "is_naked": False,
            }
        )
        for leg in source.legs
    ]
    return Structure(
        name=(new_name or "").strip()[:100] or f"{source.name[:92]} (reuse)",
        structure_type=source.structure_type,
        products=list(source.products),
        legs=legs,
        status=StructureStatus.SHELL,
        created_at=now,
        last_modified_at=now,
        notes=f"Reused from structure {source.structure_id} closed on {source.closed_at}",
        close_trigger=None,
        closed_at=None,
    )
