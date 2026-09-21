"""Tests for ui.callbacks.home_callbacks (plain functions, no browser) and the Home layout."""

import pytest
from dash.exceptions import PreventUpdate

from db.repository import Repository
from ui.callbacks import home_callbacks as hc
from ui.container import Container
from ui.layouts.home import MAX_TABLE_ROWS, home_layout
from ui.layouts.shell import COLORS


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


@pytest.fixture(autouse=True)
def env(repo, monkeypatch):
    fresh = Container(repository=repo)
    monkeypatch.setattr(hc, "container", fresh)
    return fresh


def per_structure_entry(name, total, status="open", products=("CL",), days_held=0):
    return {
        "name": name, "products": list(products), "status": status, "days_held": days_held,
        "unrealized": total, "realized": 0.0, "total": total,
        "is_stale": False, "missing_prices": [],
    }


def payload(**overrides):
    base = {
        "total_pnl": 8410.0,
        "total_unrealized": 8410.0,
        "total_realized_all_time": 12000.0,
        "todays_realized_pnl": 500.0,
        "open_structure_count": 3,
        "open_leg_count": 4,
        "net_lots_by_product": {"CL": 10.0, "BRN": -4.0},
        "largest_winner": {"structure_id": "a", "pnl": 9800.0},
        "largest_loser": {"structure_id": "c", "pnl": -4040.0},
        "per_structure": {
            "a": per_structure_entry("CL outright", 9800.0),
            "b": per_structure_entry("CL spread", 2650.0),
            "c": per_structure_entry("BRN short", -4040.0, status="partially_closed", products=("BRN",)),
        },
        "stale_symbols_in_use": [],
        "missing_price_symbols": [],
    }
    base.update(overrides)
    return base


# ---------- format helpers ----------


@pytest.mark.parametrize(
    "value,expected",
    [(1234.4, "+$1,234"), (0, "+$0"), (-1234.6, "-$1,235"), (1_000_000, "+$1,000,000"), (-5, "-$5")],
)
def test_format_pnl(value, expected):
    assert hc.format_pnl(value) == expected


def test_pnl_color():
    assert hc.pnl_color(10) == COLORS["ACCENT_GREEN"]
    assert hc.pnl_color(0) == COLORS["ACCENT_GREEN"]
    assert hc.pnl_color(-0.01) == COLORS["ACCENT_RED"]


# ---------- update_home_metrics ----------
# Output order: total realized (text, style), today's realized (text, style), unrealized card,
# open structures, open legs, net lots, margin, winner, loser, banner style, banner children.


def test_metrics_empty_store_shows_placeholders_and_hides_banner():
    result = hc.update_home_metrics({})
    assert len(result) == 13
    assert result[0] == "—" and result[2] == "—"
    assert result[4:11] == ("—",) * 7
    assert result[11] == {"display": "none"}


def test_realized_bar_shows_all_time_and_today_with_color():
    result = hc.update_home_metrics(payload())
    assert result[0] == "+$12,000" and result[1]["color"] == COLORS["ACCENT_GREEN"]
    assert result[2] == "+$500" and result[3]["color"] == COLORS["ACCENT_GREEN"]
    assert result[1]["fontSize"] == "2rem" and result[1]["fontWeight"] == "bold"


def test_realized_bar_negative_is_red_and_zero_is_neutral():
    result = hc.update_home_metrics(payload(total_realized_all_time=-2500.0, todays_realized_pnl=0.0))
    assert result[0] == "-$2,500" and result[1]["color"] == COLORS["ACCENT_RED"]
    assert result[2] == "+$0" and result[3]["color"] == COLORS["TEXT_PRIMARY"]


def test_realized_bar_missing_values_show_dash():
    result = hc.update_home_metrics(payload(total_realized_all_time=None, todays_realized_pnl=None))
    assert result[0] == "—" and result[2] == "—"


def test_unrealized_card_shows_only_unrealized_with_color():
    result = hc.update_home_metrics(payload(total_unrealized=-1800.0, total_pnl=99999.0))
    assert result[4].children == "-$1,800"
    assert result[4].style["color"] == COLORS["ACCENT_RED"]


def test_metrics_counts():
    result = hc.update_home_metrics(payload())
    assert result[5] == "3" and result[6] == "4"


def test_metrics_net_lots_compact_breakdown_by_product():
    assert hc.update_home_metrics(payload())[7] == "BRN: -4 | CL: +10"


def test_metrics_net_lots_empty_shows_dash():
    assert hc.update_home_metrics(payload(net_lots_by_product={}))[7] == "—"


def test_metrics_winner_and_loser_show_structure_name_and_pnl():
    result = hc.update_home_metrics(payload())
    winner, loser = result[9], result[10]
    assert winner[0].children == "CL outright"
    assert winner[1].children == "+$9,800"
    assert loser[0].children == "BRN short"
    assert loser[1].children == "-$4,040"


def test_metrics_no_winner_shows_dash():
    result = hc.update_home_metrics(payload(largest_winner=None, largest_loser=None))
    assert result[9] == "—" and result[10] == "—"


def test_margin_card_shows_configured_limit_and_says_usage_is_not_tracked(repo):
    repo.set_setting("margin_limit", 2_500_000.0)
    margin = hc.update_home_metrics(payload())[8]
    assert margin[0] == "—"
    assert "$2,500,000" in margin[1].children
    assert "not tracked" in margin[1].children


def test_stale_banner_hidden_when_nothing_stale():
    result = hc.update_home_metrics(payload())
    assert result[11] == {"display": "none"}


def test_stale_banner_lists_stale_symbols():
    result = hc.update_home_metrics(payload(stale_symbols_in_use=["CLZ26", "BRNZ26"]))
    style, children = result[11], result[12]
    assert style["display"] == "block"
    assert "Stale data detected for: CLZ26, BRNZ26" in children[0].children
    assert style["border"].endswith(COLORS["ACCENT_YELLOW"])


def test_stale_banner_also_reports_symbols_with_no_price():
    result = hc.update_home_metrics(payload(missing_price_symbols=["CLF27"]))
    assert result[11]["display"] == "block"
    assert "No live price for: CLF27" in result[12][0].children


# ---------- update_home_table ----------


def test_table_empty_store_gives_no_rows():
    assert hc.update_home_table({}) == []


def test_table_rows_sorted_by_total_pnl_descending_with_ids():
    rows = hc.update_home_table(payload())
    assert [r["structure_id"] for r in rows] == ["a", "b", "c"]
    assert rows[0] == {
        "structure_id": "a", "name": "CL outright", "products": "CL",
        "unrealized": 9800.0, "realized": 0.0, "total": 9800.0,
        "status": "Open", "days_held": 0,
    }
    assert rows[2]["status"] == "Partially Closed"


def test_table_joins_multiple_products():
    data = payload(per_structure={"x": per_structure_entry("Cross", 1.0, products=("BRN", "CL"))})
    assert hc.update_home_table(data)[0]["products"] == "BRN, CL"


def test_table_limited_to_max_rows():
    many = {f"s{i}": per_structure_entry(f"S{i}", float(i)) for i in range(MAX_TABLE_ROWS + 5)}
    rows = hc.update_home_table(payload(per_structure=many))
    assert len(rows) == MAX_TABLE_ROWS
    assert rows[0]["total"] == float(MAX_TABLE_ROWS + 4)


# ---------- navigate_to_structure ----------


def test_navigate_uses_row_id_from_click():
    assert hc.navigate_to_structure({"rowId": "abc-123", "colId": "name"}) == "/structures/abc-123"


@pytest.mark.parametrize("clicked", [None, {}, {"rowId": None}])
def test_navigate_without_row_id_does_nothing(clicked):
    with pytest.raises(PreventUpdate):
        hc.navigate_to_structure(clicked)


# ---------- layout ----------


def test_home_layout_contains_all_component_ids():
    text = str(home_layout())
    for component_id in [
        "home-realized-pnl-bar", "home-total-realized-pnl", "home-today-realized-pnl",
        "home-card-unrealized-pnl", "home-card-open-structures", "home-card-open-legs",
        "home-card-margin-used", "home-card-net-lots",
        "home-winner-card", "home-loser-card", "home-stale-banner", "home-structures-table",
    ]:
        assert component_id in text, component_id
    assert "home-card-total-pnl" not in text and "home-card-today-pnl" not in text
