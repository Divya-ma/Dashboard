"""Tests for the watchlist correlation engine, the Correlation tab layout and its callbacks."""

import numpy as np
import pandas as pd
import pytest
from dash.exceptions import PreventUpdate

from core.correlation import (
    MIN_OBSERVATIONS,
    build_correlation_matrix_from_series,
    compute_watchlist_correlation,
    normalize_instrument_symbol,
    structure_difference_series,
    watchlist_item_warning,
)
from core.data_loader import DataLoader
from core.exceptions import CrudeOilRiskError, InsufficientDataError
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from db.repository import Repository
from ui.callbacks import correlation_callbacks as cc
from ui.container import Container
from ui.layouts.correlation_tab import (
    PLACEHOLDER_TEXT,
    build_heatmap_figure,
    correlation_layout,
    empty_figure,
)
from ui.layouts.shell import COLORS


class NoBackfillAdapter:
    """A missing symbol must never trigger an API backfill from the heatmap tab."""

    def backfill_symbol(self, *args, **kwargs):
        raise AssertionError("the correlation tab must not backfill")


def random_closes(n, seed, start=75.0):
    return start + np.cumsum(np.random.default_rng(seed).normal(0.0, 1.5, n))


@pytest.fixture
def loader(tmp_path):
    return DataLoader(str(tmp_path), NoBackfillAdapter())


@pytest.fixture
def make_data(tmp_path, create_synthetic_parquet):
    def _make(**closes_by_symbol):
        for symbol, closes in closes_by_symbol.items():
            create_synthetic_parquet(tmp_path, symbol, closes=np.asarray(closes, dtype=float))

    return _make


def diff_series(values, name="s"):
    index = pd.date_range("2026-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(np.asarray(values, dtype=float), index=index, name=name)


def contract(symbol):
    return Contract(product="CL", contract_month=12, contract_year=2026, symbol=symbol,
                    multiplier=1000, tick_size=0.01, tick_value=10)


def structure(name, legs, status=StructureStatus.OPEN):
    """legs: [(symbol, ratio, direction, lots)]"""
    built = [Leg(contract=contract(s), ratio=r, direction=d, lots=lots, entry_price=1.0) for s, r, d, lots in legs]
    return Structure(name=name, structure_type=StructureType.CUSTOM, products=["CL"], legs=built, status=status)


# ---------- symbol normalisation ----------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("clz25", "CLZ25"),
        (" CLZ25-CLH26 ", "CLZ25-H26"),
        ("CLZ25-H26", "CLZ25-H26"),
        ("brnz26-brnf27-brng27", "BRNZ26-F27-G27"),
        ("GZ26-G27", "GZ26-G27"),  # G27 is a tenor, not the G product
        ("", ""),
        ("nonsense", "NONSENSE"),
    ],
)
def test_normalize_instrument_symbol(raw, expected):
    assert normalize_instrument_symbol(raw) == expected


# ---------- build_correlation_matrix_from_series ----------


def test_matrix_from_series_perfect_and_negative_correlation():
    base = np.random.default_rng(1).normal(size=40)
    matrix, n = build_correlation_matrix_from_series(
        {"A": diff_series(base), "B": diff_series(2 * base), "C": diff_series(-base)}, window=30
    )
    assert n == 30
    assert list(matrix.index) == ["A", "B", "C"]
    assert np.allclose(np.diag(matrix.to_numpy()), 1.0)
    assert matrix.loc["A", "B"] == pytest.approx(1.0) and matrix.loc["A", "C"] == pytest.approx(-1.0)
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)


def test_matrix_uses_all_common_days_when_lookback_is_longer():
    a, b = diff_series(np.random.default_rng(1).normal(size=12)), diff_series(np.random.default_rng(2).normal(size=12))
    _, n = build_correlation_matrix_from_series({"A": a, "B": b}, window=90)
    assert n == 12


def test_matrix_aligns_on_common_dates_and_takes_the_latest_window():
    long = diff_series(np.arange(50.0) % 7)
    short = diff_series(np.arange(50.0) % 5).iloc[20:]  # starts later
    _, n = build_correlation_matrix_from_series({"A": long, "B": short}, window=90)
    assert n == 30


def test_matrix_zero_variance_series_is_nan_not_an_error():
    flat = diff_series(np.zeros(20))
    moving = diff_series(np.random.default_rng(3).normal(size=20))
    matrix, _ = build_correlation_matrix_from_series({"flat": flat, "moving": moving}, window=20)
    assert np.isnan(matrix.loc["flat", "moving"]) and np.isnan(matrix.loc["flat", "flat"])
    assert matrix.loc["moving", "moving"] == 1.0


def test_matrix_needs_two_series_and_enough_common_days():
    with pytest.raises(InsufficientDataError, match="at least 2"):
        build_correlation_matrix_from_series({"A": diff_series(range(30))}, window=30)
    with pytest.raises(InsufficientDataError, match="common days"):
        build_correlation_matrix_from_series(
            {"A": diff_series(range(MIN_OBSERVATIONS - 1)), "B": diff_series(range(MIN_OBSERVATIONS - 1))}, window=30
        )
    with pytest.raises(ValueError):
        build_correlation_matrix_from_series({"A": diff_series(range(9)), "B": diff_series(range(9))}, window=1)


# ---------- structure series ----------


def test_structure_series_is_signed_weighted_sum_of_leg_differences(make_data, loader):
    a, b = random_closes(60, 1), random_closes(60, 2)
    make_data(CLZ26=a, CLF27=b)
    s = structure("spread", [("CLZ26", 1, "buy", 2.0), ("CLF27", -1, "buy", 2.0)])
    series = structure_difference_series(s, loader)
    expected = (np.diff(a) - np.diff(b)) * 2.0
    assert np.allclose(series.to_numpy(), expected)
    assert series.name == "spread"


def test_structure_series_direction_flips_sign_and_ratio_scales(make_data, loader):
    a, b, c = random_closes(40, 1), random_closes(40, 2), random_closes(40, 3)
    make_data(CLZ26=a, CLF27=b, CLG27=c)
    fly = structure("fly", [("CLZ26", 1, "sell", 1.0), ("CLF27", -2, "sell", 1.0), ("CLG27", 1, "sell", 1.0)])
    series = structure_difference_series(fly, loader)
    expected = -(np.diff(a) - 2 * np.diff(b) + np.diff(c))
    assert np.allclose(series.to_numpy(), expected)


def test_structure_series_missing_leg_raises_without_backfilling(make_data, loader):
    make_data(CLZ26=random_closes(40, 1))
    with pytest.raises(CrudeOilRiskError, match="CLF27"):
        structure_difference_series(structure("s", [("CLZ26", 1, "buy", 1), ("CLF27", -1, "buy", 1)]), loader)


# ---------- watchlist warnings and compute ----------


def test_watchlist_item_warnings(make_data, loader):
    make_data(CLZ26=random_closes(40, 1))
    ok = {"type": "instrument", "key": "CLZ26", "label": "CLZ26"}
    missing = {"type": "instrument", "key": "CLF27", "label": "CLF27"}
    bad = {"type": "instrument", "key": "NOPE", "label": "NOPE"}
    half = structure("half", [("CLZ26", 1, "buy", 1), ("CLF27", -1, "buy", 1)])
    by_id = {half.structure_id: half}
    assert watchlist_item_warning(ok, {}, loader) is None
    assert "no price data" in watchlist_item_warning(missing, {}, loader)
    assert "no price data" in watchlist_item_warning(bad, {}, loader)
    struct_item = {"type": "structure", "key": half.structure_id, "label": "half"}
    assert "missing price data for CLF27" in watchlist_item_warning(struct_item, by_id, loader)
    assert "no longer open" in watchlist_item_warning(struct_item, {}, loader)


def instrument(symbol):
    return {"type": "instrument", "key": symbol, "label": symbol}


def test_compute_requires_two_items(loader):
    assert compute_watchlist_correlation([instrument("CLZ26")], {}, 30, loader).error == "Add at least 2 items to compute"


def test_compute_all_missing_and_single_valid_errors(make_data, loader):
    result = compute_watchlist_correlation([instrument("CLZ26"), instrument("CLF27")], {}, 30, loader)
    assert result.error == "No valid data to compute — check symbols" and set(result.skipped) == {"CLZ26", "CLF27"}
    make_data(CLZ26=random_closes(40, 1))
    result = compute_watchlist_correlation([instrument("CLZ26"), instrument("CLF27")], {}, 30, loader)
    assert result.error == "Need at least 2 valid series" and list(result.skipped) == ["CLF27"]


def test_compute_mixed_instruments_and_structures_skips_invalid(make_data, loader):
    a, b, c = random_closes(80, 1), random_closes(80, 2), random_closes(80, 3)
    make_data(CLZ26=a, CLF27=b, CLG27=c)
    spread = structure("Z-F spread", [("CLZ26", 1, "buy", 1), ("CLF27", -1, "buy", 1)])
    broken = structure("Broken", [("CLZ26", 1, "buy", 1), ("CLH27", -1, "buy", 1)])
    items = [instrument("CLG27"), instrument("CLZ26"),
             {"type": "structure", "key": spread.structure_id, "label": "Z-F spread"},
             {"type": "structure", "key": broken.structure_id, "label": "Broken"},
             instrument("MISSING26")]
    result = compute_watchlist_correlation(
        items, {spread.structure_id: spread, broken.structure_id: broken}, 30, loader
    )
    assert result.error is None and result.observations == 30
    assert list(result.matrix.index) == ["CLG27", "CLZ26", "Z-F spread"]
    assert set(result.skipped) == {"Broken", "MISSING26"}
    expected = np.corrcoef(np.diff(a)[-30:], np.diff(a)[-30:] - np.diff(b)[-30:])[0, 1]
    assert result.matrix.loc["CLZ26", "Z-F spread"] == pytest.approx(expected)


def test_compute_reports_short_history(make_data, loader):
    make_data(CLZ26=random_closes(15, 1), CLF27=random_closes(15, 2))
    result = compute_watchlist_correlation([instrument("CLZ26"), instrument("CLF27")], {}, 90, loader)
    assert result.error is None and result.observations == 14 and result.requested == 90


# ---------- layout and figure ----------


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


def test_layout_controls_and_defaults():
    layout = correlation_layout()
    for component_id in ["corr-mode", "corr-instrument-input", "corr-structure-select", "corr-add-btn",
                         "corr-watchlist-list", "corr-lookback", "corr-compute-btn", "corr-status",
                         "corr-heatmap", "corr-watchlist", "corr-instrument-group", "corr-structure-group"]:
        assert find(layout, component_id) is not None, component_id
    assert find(layout, "corr-lookback").value == "30"
    assert [o["value"] for o in find(layout, "corr-lookback").options] == ["10", "20", "30", "60", "90"]
    assert find(layout, "corr-mode").value == "instrument"
    assert find(layout, "corr-structure-group").style == {"display": "none"}
    assert find(layout, "corr-heatmap").figure.layout.annotations[0].text == PLACEHOLDER_TEXT


def test_heatmap_figure_scale_annotations_and_title():
    matrix = pd.DataFrame([[1.0, 0.5, np.nan], [0.5, 1.0, -0.25], [np.nan, -0.25, 1.0]],
                          index=["A", "B", "C"], columns=["A", "B", "C"])
    figure = build_heatmap_figure(matrix, 30)
    trace = figure.data[0]
    assert figure.layout.title.text == "Correlation Matrix — 30d"
    assert (trace.zmin, trace.zmax) == (-1, 1)
    assert [c[1] for c in trace.colorscale] == [COLORS["ACCENT_RED"], COLORS["TEXT_PRIMARY"], COLORS["ACCENT_GREEN"]]
    assert trace.text[0][1] == "0.50" and trace.text[0][2] == "n/a" and trace.text[1][2] == "-0.25"
    assert list(trace.x) == ["A", "B", "C"] and trace.z[0][2] is None
    assert figure.layout.yaxis.autorange == "reversed"
    assert empty_figure("boom", True).layout.annotations[0].font.color == COLORS["ACCENT_RED"]


# ---------- callbacks ----------


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture
def env(repo, loader, monkeypatch):
    fresh = Container(repository=repo, data_loader=loader)
    monkeypatch.setattr(cc, "container", fresh)
    return fresh


def save(repo, s):
    for leg in s.legs:
        repo.save_contract(leg.contract)
    repo.save_structure(s)
    return s.structure_id


def trigger(monkeypatch, triggered_id, value=1):
    class Ctx:
        pass

    ctx = Ctx()
    ctx.triggered_id = triggered_id
    ctx.triggered = [{"prop_id": "x.n_clicks", "value": value}]
    monkeypatch.setattr(cc, "callback_context", ctx)


def test_toggle_mode_shows_the_right_input():
    assert cc.toggle_mode("instrument") == ({}, {"display": "none"})
    assert cc.toggle_mode("structure") == ({"display": "none"}, {})


def test_structure_dropdown_lists_open_structures_only(env, repo):
    save(repo, structure("Open one", [("CLZ26", 1, "buy", 2), ("CLF27", -1, "buy", 2)]))
    save(repo, structure("Shell", [("CLZ26", 1, "buy", 0)], StructureStatus.SHELL))
    save(repo, structure("Done", [("CLZ26", 1, "buy", 0)], StructureStatus.CLOSED))
    options = cc.populate_structure_options("/correlation")
    assert len(options) == 1
    assert options[0]["label"] == "Open one — +1 CLZ26 / -1 CLF27"
    with pytest.raises(PreventUpdate):
        cc.populate_structure_options("/structures")


def test_add_instrument_validates_normalizes_and_ignores_duplicates(env):
    assert cc.add_item(1, "instrument", "  ", None, []) [1] == "Enter an exchange symbol first."
    items, message, cleared = cc.add_item(1, "instrument", "clz25-clh26", None, [])
    assert items == [{"type": "instrument", "key": "CLZ25-H26", "label": "CLZ25-H26"}] and message == "" and cleared == ""
    again, message, _ = cc.add_item(1, "instrument", "CLZ25-H26", None, items)
    assert again is cc.no_update and message == ""  # duplicate: silently ignored
    with pytest.raises(PreventUpdate):
        cc.add_item(None, "instrument", "CLZ26", None, [])


def test_add_structure_validates_and_disambiguates_names(env, repo):
    sid = save(repo, structure("Twin", [("CLZ26", 1, "buy", 2)]))
    other = save(repo, structure("Twin", [("CLF27", 1, "buy", 2)]))
    assert cc.add_item(1, "structure", "", None, [])[1] == "Select a structure first."
    assert "no longer open" in cc.add_item(1, "structure", "", "missing", [])[1]
    first, _, _ = cc.add_item(1, "structure", "", sid, [])
    assert first == [{"type": "structure", "key": sid, "label": "Twin"}]
    second, _, _ = cc.add_item(1, "structure", "", other, first)
    assert second[1]["label"] == f"Twin ({other[:4]})"
    assert cc.add_item(1, "structure", "", sid, first)[0] is cc.no_update


def test_remove_item(monkeypatch):
    items = [instrument("A"), instrument("B"), instrument("C")]
    trigger(monkeypatch, {"type": "corr-remove", "index": 1})
    assert [i["key"] for i in cc.remove_item([None, 1, None], items)] == ["A", "C"]
    trigger(monkeypatch, {"type": "corr-remove", "index": 1}, value=None)  # a freshly rendered button
    with pytest.raises(PreventUpdate):
        cc.remove_item([None, None, None], items)
    trigger(monkeypatch, {"type": "corr-remove", "index": 9})
    with pytest.raises(PreventUpdate):
        cc.remove_item([1], items)


def test_watchlist_rows_show_warning_and_remove_buttons(env, make_data):
    make_data(CLZ26=random_closes(40, 1))
    rendered = str(cc.render_watchlist_rows([instrument("CLZ26"), instrument("CLF27")]))
    assert rendered.count("corr-remove") == 2
    assert "no price data found for this symbol — skipped" in rendered and rendered.count("skipped") == 1
    assert "No items yet" in str(cc.render_watchlist_rows([]))


def test_compute_heatmap_success_status_and_figure(env, make_data):
    make_data(CLZ26=random_closes(80, 1), CLF27=random_closes(80, 2))
    figure, status = cc.compute_heatmap(1, [instrument("CLZ26"), instrument("CLF27")], "30")
    assert figure.layout.title.text == "Correlation Matrix — 30d"
    assert "Last computed" in status.children and "30 days" in status.children and "Only" not in status.children


def test_compute_heatmap_notes_short_history_and_skipped_items(env, make_data):
    make_data(CLZ26=random_closes(20, 1), CLF27=random_closes(20, 2))
    figure, status = cc.compute_heatmap(1, [instrument("CLZ26"), instrument("CLF27"), instrument("NOPE26")], "90")
    assert "Only 19 common days available (requested 90d)" in status.children
    assert "Skipped: NOPE26" in status.children


def test_compute_heatmap_inline_errors(env, make_data):
    figure, status = cc.compute_heatmap(1, [instrument("CLZ26")], "30")
    assert figure.layout.annotations[0].text == "Add at least 2 items to compute"
    assert status.style["color"] == COLORS["ACCENT_RED"] and "Add at least 2 items to compute" in status.children
    figure, status = cc.compute_heatmap(1, [instrument("A26"), instrument("B26")], "30")
    assert "No valid data to compute — check symbols" in status.children
    make_data(CLZ26=random_closes(40, 1))
    figure, status = cc.compute_heatmap(1, [instrument("CLZ26"), instrument("B26")], "30")
    assert "Need at least 2 valid series" in status.children
    with pytest.raises(PreventUpdate):
        cc.compute_heatmap(None, [], "30")
