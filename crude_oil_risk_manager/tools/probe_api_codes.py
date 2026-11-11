"""Read-only probe of the QH API codes that were never confirmed: the CL-BZ box codes and a cross-product spread symbol.

Run it with a valid token (the one saved in the dashboard works) and read the table it prints:

    python -m tools.probe_api_codes --from-app                 # the token saved in Settings (this Windows user)
    python -m tools.probe_api_codes --token <bearer token>     # or pass one (or set QH_API_TOKEN)

Each candidate is asked for three daily candles (GET /ohlc/, 1D). "OK" means rows came back, "empty" means the API accepted the
code but had no data, "error" shows why it was refused. Nothing is written anywhere: when a candidate comes back OK, put that
format into core/box.py (box codes) or adapters/base.py SymbolTranslator (cross-product spreads) - this script only finds out which.
About 20 requests, spread over the endpoint's rate limit, so it takes a minute or two.
"""

import argparse
import os
import sys
from datetime import date

from adapters.base import MONTH_TO_LETTER


def _token(month: int, year: int) -> str:
    return f"{MONTH_TO_LETTER[month]}{year % 100:02d}"


def _after(month: int, year: int, step: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + step
    return index % 12 + 1, index // 12


def candidate_codes(month: int, year: int) -> list[tuple[str, str]]:
    """(group, code) pairs to try, built around one delivery month (the box codes core/box.py uses, and the spread guesses)."""
    a, b, c = _token(month, year), _token(*_after(month, year, 1)), _token(*_after(month, year, 2))
    return [
        ("known good", f"CL{a}"),
        ("known good", f"CO{a}"),
        ("box outbox", f"WTCLCO{a}"),
        ("box outbox", f"WTCLCO{a}-{b}"),
        ("box spreadbox", f"CLBZZX{a}-{b}"),
        ("box flybox", f"CLBZZX{a}-{b}-{c}"),
        ("box generic", "WTCLCO1"),
        ("box generic", "CLBZZX1-2"),
        ("box generic", "CLBZZX1-2-3"),
        ("cross-product spread", f"CL{a}-CO{a}"),
        ("cross-product spread", f"CO{a}-CL{a}"),
        ("cross-product spread", f"CL{a}/CO{a}"),
        ("cross-product spread", f"CL{a}_CO{a}"),
        ("cross-product spread", f"CL{a}CO{a}"),
    ]


def probe(api, codes: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """(group, code, outcome) for each candidate; never raises for a refused code."""
    results = []
    for group, code in codes:
        try:
            payload = api.ohlc("1D", instruments=[code], count=3)
            rows = sum(len(v) for v in payload.values()) if isinstance(payload, dict) else len(payload or [])
            outcome = f"OK ({rows} rows)" if rows else "empty"
        except Exception as exc:  # noqa: BLE001 - the reason is the answer
            outcome = f"error: {str(exc)[:90]}"
        results.append((group, code, outcome))
    return results


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Probe unconfirmed QH API codes (read-only)")
    parser.add_argument("--token", default=os.environ.get("QH_API_TOKEN", ""))
    parser.add_argument("--from-app", action="store_true", help="use the token saved in the dashboard's database")
    parser.add_argument("--db", default="db/crude_oil_risk.db")
    args = parser.parse_args(argv)
    token = args.token
    if args.from_app:
        from core.user_settings import KEY_API_TOKEN
        from db.repository import Repository

        token = Repository(args.db).get_setting(KEY_API_TOKEN, "")
    if not token:
        print("No token: use --token, QH_API_TOKEN or --from-app")
        return 2
    from adapters.qh_api import make_api

    today = date.today()
    month, year = _after(today.month, today.year, 6)
    for group, code, outcome in probe(make_api(lambda: token), candidate_codes(month, year)):
        print(f"{group:22} {code:22} {outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
