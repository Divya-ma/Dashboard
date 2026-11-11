"""Money as dollars and cents.

P&L, costs and rebates are floats, so a sum such as 5.76 - 5.76 can come out as 1e-12 instead of 0 and a "Flat" trade then reads as a
"Win". Anything that is classified (win / loss / flat) or shown as a total goes through `cents` first: two decimals, and never -0.0.
Prices and lots are not money and are left alone.
"""


def cents(amount) -> float:
    """The amount rounded to a cent (None counts as 0). -0.0 becomes 0.0."""
    return round(float(amount or 0.0), 2) + 0.0
