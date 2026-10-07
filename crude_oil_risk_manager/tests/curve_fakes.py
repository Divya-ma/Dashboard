"""Shared fakes for the Curve Kinks tests: a stand-in QH API, an alert recorder and price builders."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from adapters.base import APIError
from core.curve_calendar import FAMILY_WEIGHTS, FLY, OUTRIGHT, SPREAD, build_structures

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)


class FakeApi:
    """Answers generic_chart_data / get_all like the real API, deterministically."""

    def __init__(
        self, today: date = NOW.date(), fail_generic: bool = False, batch_bug_codes: set[str] | None = None,
        code_errors: dict[str, str] | None = None, max_rows_per_call: int | None = None,
    ):
        self.today = today
        self.fail_generic = fail_generic
        # Reproduces the real API: a batch of several groups that holds one of these codes fails as a
        # whole (an error entry, no rows, for every group in it), while the code alone is fine.
        self.batch_bug_codes = batch_bug_codes or set()
        # Also like the real API: a call asking for more than this many days in total fails as a whole.
        self.max_rows_per_call = max_rows_per_call
        self.code_errors = code_errors or {}
        self.generic_calls: list[list[dict]] = []
        self.settlement_calls: list[dict] = []

    def generic_chart_data(self, groups):
        self.generic_calls.append(groups)
        if self.fail_generic:
            raise APIError("generic endpoint down")
        out = {}
        batch_fails = len(groups) > 1 and (
            any(g["product"] in self.batch_bug_codes for g in groups)
            or (self.max_rows_per_call is not None and sum(g["count"] for g in groups) > self.max_rows_per_call)
        )
        for group in groups:
            code, count = group["product"], group["count"]
            if batch_fails or code in self.code_errors:
                message = self.code_errors.get(code, "Error fetching generic data: ValueError: Length mismatch")
                out[f"{code}_1D"] = {"status": "SUCCESS", "df": [], "isVolumeAvailable": False, "error": message}
                continue
            days = pd.bdate_range(end=pd.Timestamp(self.today), periods=count)  # includes today's unfinished bar
            rng = np.random.default_rng(sum(map(ord, code)))
            api = "".join(ch for ch in code if ch.isalpha())
            legs = code.count("-") + 1
            closes = self._closes(code, api, days, rng)
            contract = f"{api}Z26" + "".join(f"-{m}27" for m in ("F", "G", "H")[: legs - 1])
            out[f"{code}_1D"] = {
                "status": "SUCCESS", "isVolumeAvailable": True,
                "df": [
                    {"Date": int(d.timestamp() * 1000), "Open": c, "High": c, "Low": c, "Close": float(c),
                     "Volume": 1, "Product": contract}
                    for d, c in zip(days, closes)
                ],
            }
        return out

    @staticmethod
    def _closes(code, api, days, rng):
        """History consistent with make_prices: the same smooth curve, a shared market level and small noise."""
        positions = [int(p) for p in code[len(api):].split("-")]
        weights = FAMILY_WEIGHTS[{1: OUTRIGHT, 2: SPREAD, 3: FLY}[len(positions)]]
        base = 100.0 if api == "CO" else 95.0
        level = np.array([6 * np.sin(d.toordinal() / 45) for d in days])
        values = np.zeros(len(days))
        for weight, position in zip(weights, positions):
            k = position - 1
            values += weight * (base - 1.2 * k + 0.03 * k * k + level)
        return values + rng.normal(0, 0.02, len(days))

    def get_all(self, name, page_size=1000, max_pages=10, **params):
        self.settlement_calls.append({"name": name, **params})
        """Settlements on the same smooth curve as make_prices, a little below it; today's row is a decoy."""
        position = {
            s.symbol: (product, s.position - 1)
            for product in ("BRN", "CL") for s in build_structures(product, self.today)[OUTRIGHT]
        }
        rows = []
        for symbol in params["qhcode"].split(","):
            product, k = position[symbol]
            base = (100.0 if product == "BRN" else 95.0) - 1.2 * k + 0.03 * k * k
            for shift, day in ((-0.5, "2026-10-02"), (-0.2, "2026-10-05"), (9.0, "2026-10-06")):
                rows.append({"datetime": f"{day}T00:00:00", "close": base + shift, "qhcode": symbol, "gd_code": f"G{k + 1}"})
        return rows


class FakeAlerts:
    def __init__(self):
        self.sent: list[tuple[str, str, str]] = []

    def send_alert(self, level, title, body, structure_id=None, **kwargs):
        self.sent.append((level.value, title, body))

    def titles(self) -> list[str]:
        return [title for _, title, _ in self.sent]


class MutableClock:
    def __init__(self, now: datetime = NOW):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


def smooth_outrights(count: int = 15, base: float = 100.0) -> list[float]:
    return [base - 1.2 * k + 0.03 * k * k for k in range(count)]


def make_prices(products, today, now, kinks=None, skip=(), n=15) -> dict:
    """{symbol: (price, ts)} for the whole ladder of each product, consistent with smooth outrights.

    `kinks` maps an outright symbol to a price bump; spreads/flies follow from the bumped outrights.
    `skip` symbols are left out (a missing live price).
    """
    kinks = kinks or {}
    prices = {}
    for product in products:
        ladder = build_structures(product, today, n)
        base = smooth_outrights(n, 100.0 if product == "BRN" else 95.0)
        by_symbol = {s.symbol: base[i] + kinks.get(s.symbol, 0.0) for i, s in enumerate(ladder[OUTRIGHT])}
        for family in (OUTRIGHT, SPREAD, FLY):
            for structure in ladder[family]:
                value = sum(w * by_symbol[leg] for w, leg in zip(FAMILY_WEIGHTS[family], structure.legs))
                if structure.symbol not in skip:
                    prices[structure.symbol] = (value, now.timestamp())
    return prices
