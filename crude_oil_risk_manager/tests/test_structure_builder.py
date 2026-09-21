"""Tests for core.structure_utils, core.structure_builder, the builder layout and its callbacks."""

import pytest
from dash.exceptions import PreventUpdate

from core.exceptions import DataNotAvailableError
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from core import structure_builder as sb
from core.structure_utils import (
    ERROR_PREFIX,
    compose_structure_symbol,
    decompose_to_outrights,
    get_structure_type_from_symbol,
    net_outright_equivalent,
    parse_symbol,
    split_validation_messages,
    validate_structure_legs,
)
from db.repository import Repository
from ui.callbacks import structure_builder_callbacks as bc
from ui.callbacks import structures_callbacks as sc
from ui.container import Container
from ui.layouts.structure_builder import new_structure_modal, render_step_indicator


# ---------- decomposition ----------


def test_decompose_outright_spread_fly_condor():
    assert decompose_to_outrights("CLZ26", 10) == {"CLZ26": 10.0}
    assert decompose_to_outrights("CLZ26-F27", 5) == {"CLZ26": 5.0, "CLF27": -5.0}
    assert decompose_to_outrights("CLZ26-F27-G27", 10) == {"CLZ26": 10.0, "CLF27": -20.0, "CLG27": 10.0}
    assert decompose_to_outrights("CLZ26-F27-G27+H27", 1) == {"CLZ26": 1.0, "CLF27": -1.0, "CLG27": -1.0, "CLH27": 1.0}


def test_decompose_other_products_and_override():
    assert decompose_to_outrights("BRNZ26-F27", 2) == {"BRNZ26": 2.0, "BRNF27": -2.0}
    assert decompose_to_outrights("BZZ26", 1) == {"BZZ26": 1.0}
    assert decompose_to_outrights("GZ26", 1) == {"GZ26": 1.0}
    assert decompose_to_outrights("CLZ26-F27", 1, ratio_override=[1, -2]) == {"CLZ26": 1.0, "CLF27": -2.0}


@pytest.mark.parametrize("symbol", ["", "XYZ26", "CL", "CLZ2", "CLA26", "CLZ26--F27", "CLZ26-"])
def test_parse_symbol_rejects_malformed(symbol):
    with pytest.raises(ValueError):
        parse_symbol(symbol)


def test_decompose_rejects_bad_ratio_count_and_too_many_legs():
    with pytest.raises(ValueError):
        decompose_to_outrights("CLZ26-F27", 1, ratio_override=[1])
    with pytest.raises(ValueError):
        decompose_to_outrights("CLZ26-F27-G27-H27-J27", 1)


def test_structure_type_from_symbol():
    assert get_structure_type_from_symbol("CLZ26") == "outright"
    assert get_structure_type_from_symbol("CLZ26-F27") == "spread"
    assert get_structure_type_from_symbol("CLZ26-F27-G27") == "fly"
    assert get_structure_type_from_symbol("CLZ26-F27-G27+H27") == "condor"
    assert get_structure_type_from_symbol("CLZ26-F27+G27") == "custom"


def test_net_outright_equivalent_sums_legs_and_skips_invalid():
    legs = [
        {"symbol": "clz26-f27", "ratio": 1},
        {"symbol": "CLF27", "ratio": 2},
        {"symbol": "garbage", "ratio": 1},
        {"symbol": "", "ratio": 1},
        {"symbol": "CLG27", "ratio": None},
    ]
    net, ignored = net_outright_equivalent(legs)
    assert net == {"CLZ26": 1.0, "CLF27": 1.0}
    assert ignored == ["GARBAGE", "CLG27"]


def test_compose_structure_symbol():
    assert compose_structure_symbol(["CLZ26", "CLF27"], [1, -1]) == "CLZ26-F27"
    assert compose_structure_symbol(["CLZ26", "CLF27"], [-1, 1]) == "CLZ26-F27"
    assert compose_structure_symbol(["CLZ26", "CLF27", "CLG27"], [1, -2, 1]) == "CLZ26-F27-G27"
    assert compose_structure_symbol(["CLZ26", "CLF27", "CLG27", "CLH27"], [1, -1, -1, 1]) == "CLZ26-F27-G27+H27"
    assert compose_structure_symbol(["CLZ26", "CLF27"], [1, -2]) is None
    assert compose_structure_symbol(["CLZ26", "BRNF27"], [1, -1]) is None
    assert compose_structure_symbol(["CLZ26-F27", "CLG27"], [1, -1]) is None
    assert compose_structure_symbol(["CLZ26-F27"]) == "CLZ26-F27"
    assert compose_structure_symbol([]) is None


# ---------- validation ----------


def test_validate_structure_legs():
    assert validate_structure_legs([{"symbol": "CLZ26", "ratio": 1}, {"symbol": "CLF27", "ratio": -1}]) == []
    assert validate_structure_legs([])[0].startswith(ERROR_PREFIX)

    messages = validate_structure_legs([{"symbol": "", "ratio": 1}, {"symbol": "nope", "ratio": 0}, {"symbol": "CLZ26", "ratio": 1.5}])
    errors, warnings = split_validation_messages(messages)
    assert len(errors) == 4 and warnings == []
    assert "Leg 1" in errors[0]

    warnings_only = validate_structure_legs(
        [{"symbol": "CLZ26", "ratio": 1}, {"symbol": "clz26", "ratio": -1}, {"symbol": "BRNZ26", "ratio": 1}]
    )
    errors, warnings = split_validation_messages(warnings_only)
    assert errors == [] and any("Duplicate" in w for w in warnings) and any("Cross-product" in w for w in warnings)


# ---------- leg rows ----------


def test_next_leg_rows():
    assert sb.next_leg_rows(None, "template", []) == []
    assert [r["ratio"] for r in sb.next_leg_rows("fly", "template", [])] == [1, -2, 1]
    fixed = sb.next_leg_rows("spread", "template", [])
    assert sb.next_leg_rows("spread", "add", fixed) == fixed  # only Custom can grow
    custom = sb.next_leg_rows("custom", "template", [])
    grown = sb.next_leg_rows("custom", "add", [{"symbol": "CLZ26", "ratio": 1}, *custom])
    assert len(grown) == 3
    assert sb.next_leg_rows("custom", "remove", grown, remove_index=0)[0]["symbol"] is None
    assert sb.next_leg_rows("custom", "remove", custom, remove_index=0) == custom  # keep at least one leg


# ---------- build / save ----------


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


def spread_inputs(**overrides):
    args = dict(
        name="CL Dec-Jan", template="spread", symbols=["clz26", "CLF27"], ratios=[1, -1],
        multiplier=1000, tick_size=0.01, tick_value=10, notes="n",
    )
    args.update(overrides)
    return args


def test_build_shell_structure_creates_shell_and_new_contracts(repo):
    built = sb.build_shell_structure(**spread_inputs(), get_contract=repo.get_contract)
    s = built.structure
    assert s.status == StructureStatus.SHELL and s.structure_type == StructureType.SPREAD
    assert [l.contract.symbol for l in s.legs] == ["CLZ26", "CLF27"]
    assert all(l.lots == 0 and l.entry_price is None for l in s.legs)
    assert s.products == ["CL"] and s.notes == "n" and built.warnings == []
    front = built.new_contracts[1]
    assert (front.contract_month, front.contract_year, front.multiplier) == (1, 2027, 1000)


def test_build_supports_spread_symbol_legs_and_reuses_saved_contracts(repo):
    repo.save_contract(Contract(product="CL", contract_month=12, contract_year=2026, symbol="CLZ26",
                                multiplier=500, tick_size=0.01, tick_value=5))
    built = sb.build_shell_structure(**spread_inputs(template="custom", symbols=["CLZ26", "CLZ26-F27"], ratios=[1, 2]),
                                     get_contract=repo.get_contract)
    assert [c.symbol for c in built.new_contracts] == ["CLZ26-F27"]
    assert any("already saved with multiplier 500" in w for w in built.warnings)
    assert built.structure.legs[0].contract.multiplier == 500


def test_build_collects_all_errors(repo):
    with pytest.raises(sb.StructureBuildError) as exc:
        sb.build_shell_structure(**spread_inputs(name=" ", symbols=["CLZ26", "bad"], ratios=[1, 0], multiplier=None, tick_size=-1),
                                 get_contract=repo.get_contract)
    text = " ".join(exc.value.errors)
    for expected in ("name is required", "Leg 2", "Multiplier", "Tick size"):
        assert expected in text


def test_build_template_leg_count_and_ratio_pattern(repo):
    with pytest.raises(sb.StructureBuildError, match="exactly 2"):
        sb.build_shell_structure(**spread_inputs(symbols=["CLZ26"], ratios=[1]), get_contract=repo.get_contract)
    built = sb.build_shell_structure(**spread_inputs(ratios=[1, -2]), get_contract=repo.get_contract)
    assert any("standard Spread pattern" in w for w in built.warnings)
    with pytest.raises(sb.StructureBuildError, match="template"):
        sb.build_shell_structure(**spread_inputs(template=None), get_contract=repo.get_contract)


def test_save_shell_structure_persists_contracts_and_structure(repo):
    built = sb.build_shell_structure(**spread_inputs(), get_contract=repo.get_contract)
    sb.save_shell_structure(repo, built)
    saved = repo.get_structure(built.structure.structure_id)
    assert saved.status == StructureStatus.SHELL and len(saved.legs) == 2
    assert repo.get_contract("CLF27") is not None


def test_backfill_symbols_async_calls_adapter_and_survives_errors():
    calls = []

    class Adapter:
        def backfill_symbol(self, symbol, start):
            calls.append(symbol)
            if symbol == "BAD":
                raise RuntimeError("boom")
            return 5

    thread = sb.backfill_symbols_async(Adapter(), ["BAD", "CLZ26"])
    thread.join(timeout=5)
    assert calls == ["BAD", "CLZ26"]
    assert sb.backfill_symbols_async(Adapter(), []) is None


# ---------- correlation ----------


def structure(name, symbols, ratios, status=StructureStatus.OPEN):
    legs = [
        Leg(contract=Contract(product="CL", contract_month=12, contract_year=2026, symbol=s,
                              multiplier=1000, tick_size=0.01, tick_value=10), ratio=r, lots=1, entry_price=1.0)
        for s, r in zip(symbols, ratios)
    ]
    return Structure(name=name, structure_type=StructureType.CUSTOM, products=["CL"], legs=legs, status=status)


def test_correlation_candidates():
    assert sb.correlation_candidates("spread", ["CLZ26", "CLF27"], [1, -1]) == ["CLZ26-F27"]
    assert sb.correlation_candidates("custom", ["CLZ26", "CLF27", "bad", "CLZ26"], [1, 1, 1, 1]) == ["CLZ26", "CLF27"]
    assert sb.correlation_candidates("spread", ["CLZ26", "CLF27"], [1, -2]) == ["CLZ26", "CLF27"]


def test_check_portfolio_correlation(monkeypatch):
    calls = []

    def fake(candidate, portfolio_symbols, window, loader):
        calls.append((candidate, portfolio_symbols, window))
        return {s: {"correlation": 0.9 if s == "CLZ26" else None} for s in portfolio_symbols}

    monkeypatch.setattr(sb, "get_correlation_with_portfolio", fake)
    portfolio = [structure("outright", ["CLZ26"], [1]), structure("odd", ["CLZ26", "CLF27"], [1, -3])]
    rows = sb.check_portfolio_correlation(["CLG27"], portfolio, 60, loader := object())
    assert calls == [("CLG27", ["CLZ26"], 60)]  # the 1:-3 structure has no exchange symbol
    assert rows[0]["classification"] == "highly_correlated" and rows[0]["existing_structure"] == "outright"
    assert sb.check_portfolio_correlation(["CLG27"], [], 60, loader) == []


def test_check_portfolio_correlation_survives_data_errors(monkeypatch):
    def boom(*args):
        raise DataNotAvailableError("no data")

    monkeypatch.setattr(sb, "get_correlation_with_portfolio", boom)
    rows = sb.check_portfolio_correlation(["CLG27"], [structure("a", ["CLZ26"], [1])], 60, None)
    assert rows[0]["correlation"] is None and rows[0]["classification"] == "insufficient_data"


# ---------- layout ----------


def find(node, component_id):
    if getattr(node, "id", None) == component_id:
        return node
    children = getattr(node, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and not isinstance(child, str):
            found = find(child, component_id)
            if found is not None:
                return found
    return None


def test_modal_layout():
    modal = new_structure_modal()
    assert modal.id == "modal-new-structure" and modal.size == "xl" and modal.centered and modal.backdrop == "static"
    for component_id in [
        "builder-step-indicator", "builder-step-1", "builder-step-2", "builder-step-3", "builder-step-4",
        "template-card-outright", "template-card-spread", "template-card-fly", "template-card-custom",
        "btn-add-leg", "builder-exposure-preview", "builder-correlation-panel", "btn-refresh-correlation",
        "builder-name", "builder-multiplier", "builder-tick-size", "builder-tick-value", "builder-notes",
        "builder-validation-messages", "btn-save-structure", "btn-builder-cancel", "btn-builder-back", "btn-builder-next",
    ]:
        assert find(modal, component_id) is not None, component_id
    assert "Template" in str(render_step_indicator(1, None)) and "Spread" in str(render_step_indicator(2, "spread"))


# ---------- callbacks ----------


@pytest.fixture(autouse=True)
def env(repo, monkeypatch):
    fresh = Container(repository=repo)
    for module in (bc, sc):
        monkeypatch.setattr(module, "container", fresh)
    return fresh


def trigger(monkeypatch, triggered_id, value=1):
    class Ctx:
        pass

    ctx = Ctx()
    ctx.triggered_id = triggered_id
    ctx.triggered = [{"prop_id": "x.n_clicks", "value": value}]
    monkeypatch.setattr(bc, "callback_context", ctx)


def test_toggle_builder_modal(monkeypatch):
    trigger(monkeypatch, "btn-new-structure")
    opened = bc.toggle_builder_modal(1, None, False)
    assert opened[:4] == (True, 1, None, []) and len(opened) == 11
    trigger(monkeypatch, "btn-builder-cancel")
    assert bc.toggle_builder_modal(1, 1, True)[0] is False
    trigger(monkeypatch, "btn-new-structure")
    with pytest.raises(PreventUpdate):
        bc.toggle_builder_modal(None, None, False)


def test_select_template_and_highlight(monkeypatch):
    trigger(monkeypatch, "template-card-fly")
    assert bc.select_template(0, 0, 1, 0) == "fly"
    trigger(monkeypatch, "template-card-bogus")
    with pytest.raises(PreventUpdate):
        bc.select_template(1)
    assert bc.highlight_template("spread") == (
        "template-card", "template-card selected", "template-card", "template-card")


def test_navigate_steps_requires_template(monkeypatch):
    trigger(monkeypatch, "btn-builder-next")
    with pytest.raises(PreventUpdate):
        bc.navigate_steps(1, None, 1, None)
    assert bc.navigate_steps(1, None, 1, "spread") == 2
    assert bc.navigate_steps(1, None, 4, "spread") == 4
    trigger(monkeypatch, "btn-builder-back")
    assert bc.navigate_steps(1, 1, 3, "spread") == 2 and bc.navigate_steps(1, 1, 1, "spread") == 1


def test_step_visibility():
    *styles, back, nxt, disabled, save = bc.render_step_visibility(1, None)
    assert styles == [{}, {"display": "none"}, {"display": "none"}, {"display": "none"}]
    assert back == {"display": "none"} and disabled is True and save == {"display": "none"}
    *styles, back, nxt, disabled, save = bc.render_step_visibility(4, "spread")
    assert styles[3] == {} and nxt == {"display": "none"} and save == {} and disabled is False


def test_render_leg_rows_for_template_add_and_remove(monkeypatch):
    trigger(monkeypatch, "store-builder-template")
    children, rows, add_style = bc.render_leg_rows("fly", None, [], [], [], [])
    assert len(children) == 3 and [r["ratio"] for r in rows] == [1, -2, 1] and add_style == {"display": "none"}

    trigger(monkeypatch, "btn-add-leg")
    children, rows, add_style = bc.render_leg_rows("custom", 1, [], ["CLZ26"], [1], [])
    assert len(rows) == 2 and rows[0]["symbol"] == "CLZ26" and add_style == {}

    trigger(monkeypatch, {"type": "leg-remove", "index": 0})
    children, rows, _ = bc.render_leg_rows("custom", 1, [1, None], ["CLZ26", "CLF27"], [1, 1], [])
    assert [r["symbol"] for r in rows] == ["CLF27"]

    trigger(monkeypatch, {"type": "leg-remove", "index": 0}, value=None)  # newly created button, not a click
    with pytest.raises(PreventUpdate):
        bc.render_leg_rows("custom", 1, [None], ["CLZ26"], [1], [])


def test_fill_symbol_from_saved():
    assert bc.fill_symbol_from_saved("CLZ26") == ("CLZ26", None)
    with pytest.raises(PreventUpdate):
        bc.fill_symbol_from_saved(None)


def test_exposure_preview_and_validation_messages():
    assert "CLF27" in str(bc.update_exposure_preview(["CLZ26-F27"], [1], []))
    assert "Enter leg symbols" in str(bc.update_exposure_preview([None], [1], []))
    assert bc.update_validation_messages(2, [], [""], [1]) == []
    assert "symbol is required" in str(bc.update_validation_messages(3, [], [""], [1]))


def test_check_correlation_button_only(env):
    with pytest.raises(PreventUpdate):
        bc.check_correlation(None, ["CLZ26"], [1], "outright")
    panel, rows = bc.check_correlation(1, [""], [1], "outright")
    assert "at least one valid" in str(panel) and rows == []
    panel, rows = bc.check_correlation(1, ["CLZ26"], [1], "outright")
    assert "No active structures" in str(panel) and rows == []


def save_args(**overrides):
    args = dict(
        n_clicks=1, name="Saved", template="spread", symbols=["CLZ26", "CLF27"], ratios=[1, -1],
        multiplier=1000, tick_size=0.01, tick_value=10, notes=None,
        status_filter="active", product_filter="all", sort_by="name", portfolio_pnl={}, live_prices={},
    )
    args.update(overrides)
    return args


def test_save_structure_success_closes_modal_and_returns_rows(repo, env):
    is_open, rows, messages, toast_open, toast = bc.save_structure(**save_args())
    assert is_open is False and toast_open is True and messages == []
    assert [r["name"] for r in rows] == ["Saved"] and rows[0]["status"] == "SHELL"
    assert "Backfilling" in toast and repo.get_contract("CLF27") is not None


def test_save_structure_errors_keep_modal_open(repo):
    is_open, rows, messages, toast_open, toast = bc.save_structure(**save_args(name="", symbols=["CLZ26", "bad"]))
    assert is_open is bc.no_update and rows is bc.no_update and toast_open is bc.no_update
    assert "name is required" in str(messages) and "Leg 2" in str(messages)
    assert repo.get_all_structures() == []
    with pytest.raises(PreventUpdate):
        bc.save_structure(**save_args(n_clicks=None))
