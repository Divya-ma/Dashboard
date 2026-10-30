"""CL-BZ boxes: the difference between a CL structure and a BZ (NYMEX Brent) structure, one instrument.

Two families are quoted on the exchange, each with its own API code, so each has its own INTERNAL product code:

    internal   API        meaning
    CLBZ       WTCLCO     outbox. CLBZF27 (API WTCLCOF27) = CL Jan27 - BZ Jan27;
                          CLBZF27-G27 (API WTCLCOF27-G27) = CL Jan27 - BZ Feb27 (two months = CL month, then BZ month)
    CLBZS      CLBZZX     spreadbox (and its fly and condor). CLBZSF27-G27 (API CLBZZXF27-G27) =
                          CL Jan27 spread - BZ Jan27 spread = outbox Jan27 - outbox Feb27

A box is built from boxes of fewer months the way its CL and BZ legs are built from outrights:

    spreadbox  = outbox(a) - outbox(b)                     1 : -1
    flybox     = spreadbox(a-b) - spreadbox(b-c)           1 : -1      (= CL fly - BZ fly)
    dflybox    = flybox(a-b-c) - flybox(b-c-d)             1 : -1      (= spreadboxes in 1 : -2 : 1)

Every function here is pure: no I/O, nothing imported from the rest of the app.
"""

OUTBOX = "CLBZ"
SPREADBOX = "CLBZS"
BOX_PRODUCTS = (OUTBOX, SPREADBOX)
CL_LEG, BZ_LEG = "CL", "BZ"  # the products the box is made of (internal codes)

BOX_API_CODES = {OUTBOX: "WTCLCO", SPREADBOX: "CLBZZX"}

# month-token weights of a box with n tokens (the same patterns as the plain structures: spread, fly, condor)
_SPREADBOX_RATIOS = {2: (1, -1), 3: (1, -2, 1), 4: (1, -1, -1, 1)}


def is_box_product(product: str | None) -> bool:
    return product in BOX_PRODUCTS


def is_box_symbol(symbol: str | None) -> bool:
    """True for an internal box symbol (CLBZF27, CLBZSF27-G27), false for anything else (CLF27, CLZ26-F27)."""
    text = (symbol or "").strip().upper()
    return any(text.startswith(code) and _is_month_token(text[len(code):len(code) + 3]) for code in sorted(BOX_PRODUCTS, key=len, reverse=True))


def _is_month_token(text: str) -> bool:
    return len(text) == 3 and text[0] in "FGHJKMNQUVXZ" and text[1:].isdigit()


def box_legs(product: str, tokens: list[str]) -> list[tuple[str, float]]:
    """The CL and BZ outrights a box is made of, as [(internal outright symbol, weight), ...] for one lot bought.

    `tokens` are month tokens ("F27", "G27"). Raises ValueError for a token count the product does not quote."""
    if product == OUTBOX:
        if len(tokens) == 1:
            return [(f"{CL_LEG}{tokens[0]}", 1.0), (f"{BZ_LEG}{tokens[0]}", -1.0)]
        if len(tokens) == 2:
            return [(f"{CL_LEG}{tokens[0]}", 1.0), (f"{BZ_LEG}{tokens[1]}", -1.0)]
        raise ValueError(f"an outbox has one month (same month) or two (CL month, BZ month), got {len(tokens)}")
    if product == SPREADBOX:
        ratios = _SPREADBOX_RATIOS.get(len(tokens))
        if ratios is None:
            raise ValueError(f"a spreadbox has 2 months, a flybox 3 and a condor box 4, got {len(tokens)}")
        legs: dict[str, float] = {}
        for token, ratio in zip(tokens, ratios):
            legs[f"{CL_LEG}{token}"] = legs.get(f"{CL_LEG}{token}", 0.0) + ratio
            legs[f"{BZ_LEG}{token}"] = legs.get(f"{BZ_LEG}{token}", 0.0) - ratio
        return list(legs.items())
    raise ValueError(f"{product!r} is not a box product")


def join_tokens(tokens: list[str]) -> str:
    """Month tokens as an exchange symbol tail: F27-G27-H27, and F27-G27-H27+J27 for four (the condor form)."""
    return "-".join(tokens[:3]) + (f"+{tokens[3]}" if len(tokens) == 4 else "")


def box_symbol(product: str, tokens: list[str]) -> str:
    return product + join_tokens(tokens)


def box_derivation_options(product: str, tokens: list[str]) -> list[list[tuple[str, float]]]:
    """Ways to price one box from other symbols, simplest pieces last (see core.derived_prices).

    A spreadbox first from its pieces (the user's own method), then from the CL structure minus the BZ structure of
    the same months, and last from the CL and BZ outrights; an outbox only from its two outrights."""
    options: list[list[tuple[str, float]]] = []
    n = len(tokens)
    if product == SPREADBOX:
        if n == 2:
            options.append([(box_symbol(OUTBOX, tokens[0:1]), 1.0), (box_symbol(OUTBOX, tokens[1:2]), -1.0)])
        elif n == 3:
            options.append([(box_symbol(SPREADBOX, tokens[0:2]), 1.0), (box_symbol(SPREADBOX, tokens[1:3]), -1.0)])
        elif n == 4:
            options.append([(box_symbol(SPREADBOX, tokens[0:2]), 1.0), (box_symbol(SPREADBOX, tokens[2:4]), -1.0)])
        options.append([(CL_LEG + join_tokens(tokens), 1.0), (BZ_LEG + join_tokens(tokens), -1.0)])
    options.append(box_legs(product, tokens))
    return options


# ----------------------------------------------------------------------
# API-form helpers (the curve pages work in API symbols: WTCLCOF27, CLBZZXF27-G27)
# ----------------------------------------------------------------------

OUTBOX_API = BOX_API_CODES[OUTBOX]
BZ_API = "BZZ"


def outbox_components(api_symbol: str) -> tuple[str, str] | None:
    """(CL outright, BZ outright) in API form for a same-month outbox such as "WTCLCOF27", else None."""
    if api_symbol.startswith(OUTBOX_API) and len(api_symbol) == len(OUTBOX_API) + 3 and _is_month_token(api_symbol[len(OUTBOX_API):]):
        token = api_symbol[len(OUTBOX_API):]
        return f"{CL_LEG}{token}", f"{BZ_API}{token}"
    return None


def price_with_components(prices: dict[str, float], api_symbol: str) -> float | None:
    """`api_symbol`'s value in `prices`, or for an outbox the CL outright minus the BZ outright when only those are there."""
    if api_symbol in prices:
        return prices[api_symbol]
    parts = outbox_components(api_symbol)
    if parts is not None and all(part in prices for part in parts):
        return prices[parts[0]] - prices[parts[1]]
    return None


def outbox_weights(product: str, tokens: list[str]) -> dict[str, float] | None:
    """A box as a combination of same-month OUTBOXES (internal symbols), the instruments whose settlements are stored.

    spreadbox F27-G27 = CLBZF27 - CLBZG27; a flybox is 1 : -2 : 1 of three outboxes, and so on. A same-month outbox is itself
    and a cross-month outbox (CL Jan - BZ Feb) cannot be written this way: None."""
    if product == OUTBOX:
        return {box_symbol(OUTBOX, tokens): 1.0} if len(tokens) == 1 else None
    ratios = _SPREADBOX_RATIOS.get(len(tokens))
    if product != SPREADBOX or ratios is None:
        return None
    return {box_symbol(OUTBOX, [token]): float(ratio) for token, ratio in zip(tokens, ratios)}
