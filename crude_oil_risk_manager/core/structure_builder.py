"""Business logic behind the New Structure builder modal.

Builds and saves shell structures, computes the leg-row edits for the modal and
runs the (expensive, button-triggered) correlation check against the portfolio.
The UI only collects inputs and renders what these functions return.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from adapters.base import LETTER_TO_MONTH
from core.correlation import classify_correlation, correlate_structures, outright_weights
from core.data_loader import DataLoader
from core.exceptions import CrudeOilRiskError
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from core.structure_utils import (
    STRUCTURE_TEMPLATES,
    compose_structure_symbol,
    normalize_symbol,
    parse_symbol,
    split_validation_messages,
    validate_structure_legs,
)
from core.user_settings import parse_number

logger = logging.getLogger(__name__)

# Same look-back DataLoader uses when it backfills a symbol on demand.
BACKFILL_DAYS = 365 * 3

EVENT_TEMPLATE = "template"
EVENT_ADD = "add"
EVENT_REMOVE = "remove"


class StructureBuildError(CrudeOilRiskError):
    """Blocking problems found while building a structure; `errors` are safe to show the user."""

    def __init__(self, errors: list[str], warnings: list[str] | None = None):
        super().__init__("; ".join(errors))
        self.errors = errors
        self.warnings = warnings or []


@dataclass
class BuiltStructure:
    structure: Structure
    new_contracts: list[Contract]
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Leg rows
# ----------------------------------------------------------------------


def next_leg_rows(
    template: str | None, event: str, current_rows: list[dict], remove_index: int | None = None
) -> list[dict]:
    """Leg rows ({"symbol", "ratio", "tc"}) after a template pick, "+ Add Leg" or a remove click.

    "tc" is the leg's transaction cost per lot for a single side ($); it defaults to 0.0.
    """
    spec = STRUCTURE_TEMPLATES.get(template or "")
    if spec is None:
        return []
    if event == EVENT_TEMPLATE:
        return [{"symbol": None, "ratio": ratio, "tc": 0.0} for ratio in spec["ratios"]]
    if spec["legs"] is not None:  # only Custom has an editable number of legs
        return current_rows
    if event == EVENT_ADD:
        return [*current_rows, {"symbol": None, "ratio": 1, "tc": 0.0}]
    if event == EVENT_REMOVE and remove_index is not None and len(current_rows) > 1:
        return [row for i, row in enumerate(current_rows) if i != remove_index]
    return current_rows


# ----------------------------------------------------------------------
# Building / saving
# ----------------------------------------------------------------------


def _contract_for_symbol(symbol: str, multiplier: float, tick_size: float, tick_value: float) -> Contract:
    """New Contract for an instrument the app has not seen; month/year come from the front leg."""
    front = parse_symbol(symbol)[0]
    return Contract(
        product=front["product"],
        contract_month=LETTER_TO_MONTH[front["letter"]],
        contract_year=2000 + int(front["year"]),
        symbol=symbol,
        multiplier=multiplier,
        tick_size=tick_size,
        tick_value=tick_value,
    )


def _parse_tc(value, leg_number: int, errors: list[str]) -> float:
    """Transaction cost per lot (one side): blank/None -> 0.0; must be >= 0."""
    if value is None or value == "":
        return 0.0
    try:
        tc = float(value)
    except (TypeError, ValueError):
        errors.append(f"Leg {leg_number} transaction cost must be a number.")
        return 0.0
    if tc < 0:
        errors.append(f"Leg {leg_number} transaction cost cannot be negative.")
        return 0.0
    return tc


def build_shell_structure(
    name, template, symbols, ratios, multiplier, tick_size, tick_value, notes, get_contract, tcs=None
) -> BuiltStructure:
    """Validate the builder inputs and build a SHELL structure (no lots, no entry prices).

    `get_contract(symbol)` returns the saved Contract or None. `tcs` is the per-leg
    transaction cost ($/lot, one side); missing/blank entries default to 0.0. Raises
    StructureBuildError listing every blocking problem at once.
    """
    errors: list[str] = []
    spec = STRUCTURE_TEMPLATES.get(template or "")
    if spec is None:
        errors.append("Choose a structure template.")
    name = (name or "").strip()
    if not name:
        errors.append("Structure name is required.")

    rows = [{"symbol": normalize_symbol(s), "ratio": r} for s, r in zip(symbols or [], ratios or [])]
    tc_values = [_parse_tc(t, i + 1, errors) for i, t in enumerate((tcs or [])[: len(rows)])]
    tc_values += [0.0] * (len(rows) - len(tc_values))
    messages = validate_structure_legs(rows)
    leg_errors, warnings = split_validation_messages(messages)
    errors.extend(leg_errors)
    if spec is not None and spec["legs"] is not None and len(rows) != spec["legs"]:
        errors.append(f"A {spec['label']} needs exactly {spec['legs']} leg(s).")
    if spec is not None and spec["legs"] is not None and len(rows) == spec["legs"] and not leg_errors:
        if [int(r["ratio"]) for r in rows] != spec["ratios"]:
            warnings.append(f"Ratios differ from the standard {spec['label']} pattern {spec['ratios']}.")

    numbers = {}
    for label, value, key in (
        ("Multiplier", multiplier, "multiplier"),
        ("Tick size", tick_size, "tick_size"),
        ("Tick value", tick_value, "tick_value"),
    ):
        try:
            numbers[key] = parse_number(value, label, minimum=0)
            if numbers[key] <= 0:
                raise ValueError(f"{label} must be greater than 0")
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise StructureBuildError(errors, warnings)

    contracts: dict[str, Contract] = {}
    new_contracts: list[Contract] = []
    try:
        for row in rows:
            symbol = row["symbol"]
            if symbol in contracts:
                continue
            existing = get_contract(symbol)
            if existing is not None:
                if existing.multiplier != numbers["multiplier"]:
                    warnings.append(
                        f"{symbol} is already saved with multiplier {existing.multiplier:g}; "
                        f"its saved specs are kept."
                    )
                contracts[symbol] = existing
            else:
                contracts[symbol] = _contract_for_symbol(symbol, **numbers)
                new_contracts.append(contracts[symbol])

        legs = [
            Leg(contract=contracts[row["symbol"]], ratio=int(row["ratio"]), transaction_cost_per_lot=tc)
            for row, tc in zip(rows, tc_values)
        ]
        structure = Structure(
            name=name,
            structure_type=StructureType(template),
            products=sorted({leg.contract.product for leg in legs}),
            legs=legs,
            status=StructureStatus.SHELL,
            notes=(notes or "").strip(),
        )
    except ValidationError as exc:
        raise StructureBuildError([f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()], warnings) from exc
    return BuiltStructure(structure=structure, new_contracts=new_contracts, warnings=warnings)


def save_shell_structure(repository, built: BuiltStructure) -> None:
    """Persist new contracts first (legs reference them), then the structure."""
    for contract in built.new_contracts:
        repository.save_contract(contract)
    repository.save_structure(built.structure)


def backfill_symbols_async(historical_adapter, symbols: list[str]) -> threading.Thread | None:
    """Backfill daily history for new symbols on a daemon thread; failures are only logged."""
    if not symbols or historical_adapter is None:
        return None

    def run() -> None:
        start = datetime.now(timezone.utc) - timedelta(days=BACKFILL_DAYS)
        for symbol in symbols:
            try:
                rows = historical_adapter.backfill_symbol(symbol, start)
                logger.info("Backfilled %s: %d rows", symbol, rows)
            except Exception:  # noqa: BLE001 - a failed backfill must never affect the saved structure
                logger.exception("Backfill failed for %s", symbol)

    thread = threading.Thread(target=run, name="builder-backfill", daemon=True)
    thread.start()
    return thread


# ----------------------------------------------------------------------
# Correlation vs portfolio
# ----------------------------------------------------------------------


# Statuses the Structure Builder's correlation check compares a candidate against, and
# the same set the background historical sync keeps warm via active_composite_symbols.
ACTIVE_STRUCTURE_STATUSES = [StructureStatus.SHELL, StructureStatus.OPEN, StructureStatus.PARTIALLY_CLOSED]


def active_composite_symbols(repository) -> list[str]:
    """Exchange-quoted composite symbols of every shell/open/partially-closed structure.

    Feeds the background historical sync (adapters.historical.vendor.VendorHistoricalAdapter.
    run_morning_sync's `extra_symbols`) so a structure's composite symbol is already cached
    locally by the time someone runs a correlation check against it, instead of only ever
    being fetched lazily — one API call at a time — the first time a check happens to touch
    it. Structures with no single exchange-quoted symbol (custom, cross-product, >4 legs)
    are skipped, same as check_portfolio_correlation.
    """
    structures = repository.get_all_structures(status_filter=ACTIVE_STRUCTURE_STATUSES)
    seen: list[str] = []
    for structure in structures:
        symbol = structure_symbol(structure)
        if symbol and symbol not in seen:
            seen.append(symbol)
    return seen


def structure_symbol(structure: Structure) -> str | None:
    """Exchange-quoted symbol of an existing structure, or None if it has none."""
    return compose_structure_symbol(
        [leg.contract.symbol for leg in structure.legs], [leg.ratio for leg in structure.legs]
    )


def _leg_dicts(structure: Structure) -> list[dict]:
    return [{"symbol": leg.contract.symbol, "ratio": leg.ratio} for leg in structure.legs]


def check_portfolio_correlation(
    candidate_legs: list[dict],
    portfolio: list[Structure],
    window: int,
    data_loader: DataLoader,
) -> list[dict]:
    """One row per existing active structure, plus a 'vs Whole Portfolio' summary row,
    comparing the prospective structure (`candidate_legs`, i.e. what's typed in the Legs
    step: [{"symbol", "ratio"}, ...]) against everything already active in the portfolio.

    Correlation is computed from each structure's OUTRIGHT decomposition (see
    core.correlation.outright_weights / structure_return_series / correlate_structures),
    never from a "composed" exchange-quoted symbol's own price history — that composed
    symbol frequently doesn't exist at all (custom ratios, cross-product legs, a structure
    nested inside another one's leg all make compose_structure_symbol return None), which
    used to mean most real portfolios showed "No active structures ... to compare against"
    on every check. Only the underlying outright contracts (almost always already cached)
    need price data, and each pair is aligned on its OWN legs' common dates only — an
    illiquid leg on some other structure never starves a comparison that doesn't involve it.

    Missing data never raises; a row carries correlation=None with an "error" reason
    instead. The 'vs Whole Portfolio' row is the signed average of the pairwise rows that
    did compute (a candidate that moves with the book on average reads positive here, one
    that hedges it reads negative) — it is left out entirely if nothing else computed.
    """
    candidate_weights = outright_weights(candidate_legs)
    if not candidate_weights:
        return []

    entries = [(structure, outright_weights(_leg_dicts(structure))) for structure in portfolio]
    entries = [(structure, weights) for structure, weights in entries if weights]
    if not entries:
        return []

    all_symbols = set(candidate_weights)
    for _, weights in entries:
        all_symbols.update(weights)

    ensure_cached = getattr(data_loader, "ensure_cached", None)
    if ensure_cached is not None:
        try:
            ensure_cached(list(all_symbols))
        except Exception:  # noqa: BLE001 - a failed pre-warm must never block the correlation check
            logger.exception("Bulk pre-warm before correlation check failed")

    rows = []
    for structure, weights in entries:
        try:
            result = correlate_structures(candidate_weights, weights, window, data_loader)
        except Exception:  # noqa: BLE001 - a data/adapter failure on one structure must never crash the whole check
            logger.exception("Correlation check failed for %s", structure.name)
            result = {"correlation": None, "classification": "insufficient_data", "window_used": 0, "error": "Could not compute this correlation; see the server log."}
        rows.append(
            {
                "candidate": "New structure",
                "existing_symbol": structure_symbol(structure) or ", ".join(sorted(weights)),
                "existing_structure": structure.name,
                "correlation": result["correlation"],
                "classification": result["classification"],
                "window_used": result["window_used"],
                "error": result["error"],
            }
        )

    valid = [r["correlation"] for r in rows if r["correlation"] is not None]
    if valid:
        average = sum(valid) / len(valid)
        rows.append(
            {
                "candidate": "New structure",
                "existing_symbol": "PORTFOLIO",
                "existing_structure": f"Whole Portfolio (avg. of {len(valid)} structure(s))",
                "correlation": average,
                "classification": classify_correlation(average),
                "window_used": min(r["window_used"] for r in rows if r["correlation"] is not None),
                "error": None,
            }
        )

    return rows
