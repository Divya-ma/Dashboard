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
