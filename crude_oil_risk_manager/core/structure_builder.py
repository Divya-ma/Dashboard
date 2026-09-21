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
from core.correlation import classify_correlation, get_correlation_with_portfolio
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
    """Leg rows ({"symbol", "ratio"}) after a template pick, "+ Add Leg" or a remove click."""
    spec = STRUCTURE_TEMPLATES.get(template or "")
    if spec is None:
        return []
    if event == EVENT_TEMPLATE:
        return [{"symbol": None, "ratio": ratio} for ratio in spec["ratios"]]
    if spec["legs"] is not None:  # only Custom has an editable number of legs
        return current_rows
    if event == EVENT_ADD:
        return [*current_rows, {"symbol": None, "ratio": 1}]
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


def build_shell_structure(
    name, template, symbols, ratios, multiplier, tick_size, tick_value, notes, get_contract
) -> BuiltStructure:
    """Validate the builder inputs and build a SHELL structure (no lots, no entry prices).

    `get_contract(symbol)` returns the saved Contract or None. Raises
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

        legs = [Leg(contract=contracts[row["symbol"]], ratio=int(row["ratio"])) for row in rows]
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


def structure_symbol(structure: Structure) -> str | None:
    """Exchange-quoted symbol of an existing structure, or None if it has none."""
    return compose_structure_symbol(
        [leg.contract.symbol for leg in structure.legs], [leg.ratio for leg in structure.legs]
    )


def correlation_candidates(template: str | None, symbols: list[str], ratios: list[int]) -> list[str]:
    """Symbols to test: the structure's own exchange-quoted symbol when it has one, else each valid leg."""
    if template in ("outright", "spread", "fly"):
        composite = compose_structure_symbol(symbols, ratios)
        if composite:
            return [composite]
    valid = []
    for symbol in symbols:
        try:
            parse_symbol(symbol)
        except ValueError:
            continue
        if normalize_symbol(symbol) not in valid:
            valid.append(normalize_symbol(symbol))
    return valid


def check_portfolio_correlation(
    candidates: list[str],
    portfolio: list[Structure],
    window: int,
    data_loader: DataLoader,
) -> list[dict]:
    """One row per (candidate, existing structure): correlation, classification, message.

    Structures without an exchange-quoted symbol (custom, cross-product) are
    left out. Missing data never raises; the row carries correlation=None.
    """
    names_by_symbol: dict[str, list[str]] = {}
    for structure in portfolio:
        symbol = structure_symbol(structure)
        if symbol:
            names_by_symbol.setdefault(symbol, []).append(structure.name)
    if not names_by_symbol:
        return []

    rows = []
    for candidate in candidates:
        try:
            results = get_correlation_with_portfolio(candidate, list(names_by_symbol), window, data_loader)
        except (CrudeOilRiskError, ValueError) as exc:
            logger.warning("Correlation check failed for %s: %s", candidate, exc)
            results = {s: {"correlation": None} for s in names_by_symbol}
        for existing, entry in results.items():
            correlation = entry["correlation"]
            rows.append(
                {
                    "candidate": candidate,
                    "existing_symbol": existing,
                    "existing_structure": ", ".join(names_by_symbol[existing]),
                    "correlation": correlation,
                    "classification": classify_correlation(correlation),
                }
            )
    return rows
