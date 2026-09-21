"""Tests for core.structure_utils (decomposition, symbol type inference, leg validation)."""

from core.structure_utils import (
    ERROR_PREFIX,
    decompose_to_outrights,
    get_structure_type_from_symbol,
    validate_structure_legs,
)


def test_decompose_outright():
    assert decompose_to_outrights("CLZ26", 5) == {"CLZ26": 5.0}


def test_decompose_spread():
    assert decompose_to_outrights("CLZ26-F27", 10) == {"CLZ26": 10.0, "CLF27": -10.0}


def test_decompose_fly():
    assert decompose_to_outrights("CLZ26-F27-G27", 10) == {"CLZ26": 10.0, "CLF27": -20.0, "CLG27": 10.0}


def test_decompose_ratio_override():
    assert decompose_to_outrights("CLZ26-F27-G27", 3, ratio_override=[2, -1, -1]) == {
        "CLZ26": 6.0, "CLF27": -3.0, "CLG27": -3.0,
    }


def test_structure_type_from_symbol():
    assert get_structure_type_from_symbol("CLZ26") == "outright"
    assert get_structure_type_from_symbol("CLZ26-F27") == "spread"
    assert get_structure_type_from_symbol("CLZ26-F27-G27") == "fly"
    assert get_structure_type_from_symbol("CLZ26-F27-G27+H27") == "condor"
    assert get_structure_type_from_symbol("CLZ26-F27+G27") == "custom"


def test_validate_empty_legs_is_an_error():
    messages = validate_structure_legs([])
    assert len(messages) == 1 and messages[0].startswith(ERROR_PREFIX)


def test_validate_duplicate_symbols_is_a_warning():
    messages = validate_structure_legs([{"symbol": "CLZ26", "ratio": 1}, {"symbol": "CLZ26", "ratio": -1}])
    assert len(messages) == 1
    assert "Duplicate" in messages[0] and not messages[0].startswith(ERROR_PREFIX)


def test_validate_zero_ratio_is_an_error():
    messages = validate_structure_legs([{"symbol": "CLZ26", "ratio": 0}])
    assert len(messages) == 1 and messages[0].startswith(ERROR_PREFIX) and "ratio" in messages[0]


# ---------- clone_structure_as_shell ----------

from datetime import datetime, timezone

from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from core.structure_utils import clone_structure_as_shell


def closed_source(name="Dec-Jan spread"):
    legs = [
        Leg(
            contract=Contract(product="CL", contract_month=m, contract_year=2026 + (m == 1), symbol=s,
                              multiplier=1000, tick_size=0.01, tick_value=10),
            ratio=r, lots=10, entry_price=75.0, average_entry_price=75.0, direction="sell",
        )
        for s, m, r in [("CLZ26", 12, 1), ("CLF27", 1, -1)]
    ]
    return Structure(
        name=name, structure_type=StructureType.SPREAD, products=["CL"], legs=legs, status=StructureStatus.CLOSED,
        close_trigger="manual", closed_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def test_clone_gets_new_structure_and_leg_ids():
    source = closed_source()
    clone = clone_structure_as_shell(source)
    assert clone.structure_id != source.structure_id
    assert not {l.leg_id for l in clone.legs} & {l.leg_id for l in source.legs}


def test_clone_legs_are_reset_but_keep_symbols_ratios_and_contracts():
    source = closed_source()
    clone = clone_structure_as_shell(source)
    assert all(l.lots == 0.0 and l.entry_price is None and l.average_entry_price is None for l in clone.legs)
    assert all(l.direction == "buy" for l in clone.legs)
    assert [(l.contract.symbol, l.ratio, l.contract.multiplier) for l in clone.legs] == [("CLZ26", 1, 1000), ("CLF27", -1, 1000)]
    assert all(l.direction == "sell" and l.lots == 10 for l in source.legs)  # the source is not modified


def test_clone_status_and_bookkeeping_fields():
    source = closed_source()
    clone = clone_structure_as_shell(source)
    assert clone.status == StructureStatus.SHELL
    assert clone.close_trigger is None and clone.closed_at is None
    assert source.structure_id in clone.notes and "closed on 2026-09-01" in clone.notes
    assert clone.created_at > source.created_at or clone.created_at == clone.last_modified_at


def test_clone_name_defaults_to_reuse_suffix_and_accepts_override():
    source = closed_source()
    assert clone_structure_as_shell(source).name == "Dec-Jan spread (reuse)"
    assert clone_structure_as_shell(source, "Brand new").name == "Brand new"
    assert clone_structure_as_shell(source, "   ").name == "Dec-Jan spread (reuse)"
    assert len(clone_structure_as_shell(closed_source("x" * 100)).name) <= 100
