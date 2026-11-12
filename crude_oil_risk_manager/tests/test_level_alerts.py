"""Price-level alerts: build and validate, price, fire once, store, and the Price Alerts page."""

from datetime import datetime, timezone

import pytest

from core.alerts import AlertManager
from core.level_alerts import ABOVE, ACTIVE, AUTO, BELOW, TRIGGERED, LevelAlertError, build_level_alert, check_level_alerts, default_name, is_hit, level_price
from core.models import AlertLevel, Contract, Leg, Structure, StructureStatus, StructureType
from core.structure_view import structure_live_price
from db.repository import Repository
from ui.callbacks import level_alerts_callbacks as lc
from ui.callbacks import shell_callbacks as shell
from ui.container import Container
from ui.layouts.level_alerts_tab import alerts_table

PRICES = {"BRNJ27": 80.0, "BRNK27": 79.5, "BRNM27": 79.0}


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


def build(repo=None, **kw):
    args = dict(name="", template="spread", symbols=["BRNJ27", "BRNK27"], ratios=[1, -1], level=0.8, direction=AUTO, prices=PRICES,
                get_contract=(repo.get_contract if repo else (lambda s: None)))
    args.update(kw)
    return build_level_alert(args["name"], args["template"], args["symbols"], args["ratios"], args["level"], args["direction"], args["prices"], args["get_contract"])


# ---------------- price and direction ----------------


def test_the_structure_price_is_the_same_composite_the_structures_page_shows():
    legs = [{"symbol": "BRNJ27", "ratio": 1}, {"symbol": "BRNK27", "ratio": -2}, {"symbol": "BRNM27", "ratio": 1}]
    assert level_price(legs, PRICES) == pytest.approx(80.0 - 2 * 79.5 + 79.0)
    assert level_price(legs, {"BRNJ27": 80.0}) is None  # a leg without a price: no price, never a partial number
    structure = Structure(name="x", structure_type=StructureType.FLY, products=["BRN"], status=StructureStatus.SHELL, legs=[
        Leg(contract=Contract(product="BRN", contract_month=m, contract_year=2027, symbol=s, multiplier=1000, tick_size=0.01, tick_value=10), ratio=r)
        for s, m, r in (("BRNJ27", 4, -1), ("BRNK27", 5, 2), ("BRNM27", 6, -1))])
    negative = [{"symbol": s, "ratio": r} for s, r in (("BRNJ27", -1), ("BRNK27", 2), ("BRNM27", -1))]
    assert level_price(negative, PRICES) == pytest.approx(structure_live_price(structure, PRICES))  # a short-first structure keeps the same sign rule


def test_a_level_fires_at_or_beyond_it_in_the_chosen_direction():
    assert is_hit(ABOVE, 1.0, 1.0) and is_hit(ABOVE, 1.0, 1.2) and not is_hit(ABOVE, 1.0, 0.9)
    assert is_hit(BELOW, 1.0, 1.0) and is_hit(BELOW, 1.0, 0.8) and not is_hit(BELOW, 1.0, 1.1)


def test_auto_picks_the_direction_from_where_the_price_is_now():
    assert build(level=0.8)[0].direction == ABOVE  # the spread is 0.5 now
    assert build(level=0.2)[0].direction == BELOW
    assert build(level=0.5)[0].direction == ABOVE  # equal counts as above
    alert, _, _ = build(level=0.8)
    assert alert.created_price == pytest.approx(0.5) and alert.status == ACTIVE and alert.name == "BRNJ27-K27"


def test_a_price_that_is_not_known_needs_a_chosen_direction():
    with pytest.raises(LevelAlertError, match="no live price"):
        build(prices={})
    alert, _, _ = build(prices={}, direction=BELOW, level=-1)
    assert alert.direction == BELOW and alert.created_price is None and alert.level == -1  # spreads and flies trade below zero


def test_a_level_that_is_already_reached_is_allowed_with_a_warning():
    alert, _, warnings = build(direction=BELOW, level=0.9)  # the spread is 0.5, already below 0.9
    assert alert.direction == BELOW and any("fires on the next price check" in w for w in warnings)


def test_bad_forms_say_everything_that_is_wrong_at_once():
    with pytest.raises(LevelAlertError) as caught:
        build(symbols=["BRNJ27", "NOPE"], ratios=[1, 0], level="abc", direction="sideways")
    text = " ".join(caught.value.errors)
    assert "price level as a number" in text and "when the alert should fire" in text and "Leg 2" in text
    with pytest.raises(LevelAlertError, match="exactly 2 leg"):
        build(symbols=["BRNJ27"], ratios=[1])
    with pytest.raises(LevelAlertError, match="limited to 100"):
        build(name="x" * 101)
    assert build(name="  My spread ")[0].name == "My spread"


def test_new_contracts_are_returned_so_the_live_poll_will_price_them(repo):
    _, new_contracts, _ = build(repo)
    assert sorted(c.symbol for c in new_contracts) == ["BRNJ27", "BRNK27"]
    for contract in new_contracts:
        repo.save_contract(contract)
    assert build(repo)[1] == []  # known now


def test_the_default_name_is_the_exchange_symbol_or_the_legs():
    assert default_name([{"symbol": "BRNJ27", "ratio": 1}, {"symbol": "BRNK27", "ratio": -2}, {"symbol": "BRNM27", "ratio": 1}]) in ("BRNJ27-K27-M27", "+1x BRNJ27 / -2x BRNK27 / +1x BRNM27")
    assert "BRNJ27" in default_name([{"symbol": "BRNJ27", "ratio": 3}])


# ---------------- storage ----------------


def test_alerts_are_stored_listed_marked_triggered_once_and_deleted(repo):
    alert, contracts, _ = build(repo)
    for contract in contracts:
        repo.save_contract(contract)
    repo.save_level_alert(alert)
    (stored,) = repo.get_level_alerts()
    assert stored.name == alert.name and stored.legs == alert.legs and stored.level == 0.8 and stored.status == ACTIVE
    assert repo.get_level_alerts(TRIGGERED) == [] and len(repo.get_level_alerts(ACTIVE)) == 1
    assert repo.mark_level_alert_triggered(alert.alert_id, 0.85) is True
    assert repo.mark_level_alert_triggered(alert.alert_id, 0.9) is False  # only one caller wins: the alert is sent once
    (done,) = repo.get_level_alerts(TRIGGERED)
    assert done.triggered_price == 0.85 and isinstance(done.triggered_at, datetime)
    assert repo.delete_level_alert(alert.alert_id) is True and repo.delete_level_alert(alert.alert_id) is False
    assert repo.get_level_alerts() == []
    actions = [row["action"] for row in repo.get_audit_log(10, entity_type="level_alert")]
    assert actions == ["level_alert.delete", "level_alert.create"]


# ---------------- firing ----------------


class Recorder:
    def __init__(self):
        self.sent = []

    def send_alert(self, level, title, body, structure_id=None):
        self.sent.append((level, title, body))
        return (level, title, body)


def stored(repo, **kw):
    alert, contracts, _ = build(repo, **kw)
    for contract in contracts:
        repo.save_contract(contract)
    repo.save_level_alert(alert)
    return alert


def test_an_alert_fires_once_when_the_price_reaches_the_level(repo):
    stored(repo, level=0.8)  # fires at or above 0.8
    manager = Recorder()
    assert check_level_alerts(repo, manager, {"BRNJ27": 80.2, "BRNK27": 79.5}) == []  # 0.7: not yet
    (sent,) = check_level_alerts(repo, manager, {"BRNJ27": 80.4, "BRNK27": 79.5})  # 0.9
    assert sent[0] == AlertLevel.WARNING and "BRNJ27-K27" in sent[1] and "0.8" in sent[2]
    assert check_level_alerts(repo, manager, {"BRNJ27": 81.0, "BRNK27": 79.5}) == []  # still beyond the level: not sent again
    assert len(manager.sent) == 1 and repo.get_level_alerts()[0].status == TRIGGERED


def test_a_falling_alert_fires_below_and_a_missing_price_is_skipped(repo):
    stored(repo, level=0.2)  # fires at or below 0.2
    manager = Recorder()
    assert check_level_alerts(repo, manager, {"BRNJ27": 80.0}) == []  # one leg has no price
    assert len(check_level_alerts(repo, manager, {"BRNJ27": 80.0, "BRNK27": 79.9})) == 1  # the spread is 0.1, at or under 0.2
    assert len(manager.sent) == 1


def test_a_broken_alert_table_never_stops_the_price_checks():
    class Broken:
        def get_level_alerts(self, status=None):
            raise RuntimeError("database is locked")

    assert check_level_alerts(Broken(), Recorder(), PRICES) == []


def test_the_live_price_callback_raises_a_persistent_toast_for_a_level_alert(repo, monkeypatch):
    stored(repo, level=0.8)
    monkeypatch.setattr(shell, "container", Container(repository=repo, alert_manager=AlertManager(repo)))
    (toast,) = shell.check_price_alerts({"BRNJ27": {"price": 80.4}, "BRNK27": {"price": 79.5}}, None)
    assert toast.icon == "warning" and toast.duration == 0 and "Price level hit" in toast.header
    with pytest.raises(shell.PreventUpdate):  # nothing new on the next price
        shell.check_price_alerts({"BRNJ27": {"price": 80.5}, "BRNK27": {"price": 79.5}}, [toast.to_plotly_json()])


# ---------------- the page ----------------


@pytest.fixture
def page(repo, monkeypatch):
    fresh = Container(repository=repo, alert_manager=AlertManager(repo))
    monkeypatch.setattr(lc, "container", fresh)
    return fresh


LIVE = {"BRNJ27": {"price": 80.0}, "BRNK27": {"price": 79.5}}


def test_the_template_fills_the_leg_rows_and_a_custom_structure_can_add_and_remove_legs(page, monkeypatch):
    class Ctx:
        triggered_id = "la-template"
        triggered = [{"prop_id": "la-template.value", "value": "fly"}]

    ctx = Ctx()
    monkeypatch.setattr(lc, "callback_context", ctx)
    rows = lc.update_leg_rows("fly", None, [], [], [])
    assert len(rows) == 3 and "la-remove" not in str(rows)  # a fixed template has no remove buttons
    assert [r.children[2].children.value for r in rows] == [1, -2, 1]

    ctx.triggered_id, ctx.triggered = "la-add-leg", [{"prop_id": "la-add-leg.n_clicks", "value": 1}]
    assert len(lc.update_leg_rows("fly", 1, [], ["BRNJ27", "BRNK27", "BRNM27"], [1, -2, 1])) == 3  # a fly stays three legs
    added = lc.update_leg_rows("custom", 1, [], ["BRNJ27"], [1])
    assert len(added) == 2 and "la-remove" in str(added) and added[0].children[1].children.value == "BRNJ27"  # typed values are kept

    ctx.triggered_id, ctx.triggered = {"type": "la-remove", "index": 0}, [{"prop_id": "x.n_clicks", "value": 1}]
    left = lc.update_leg_rows("custom", 1, [1, None], ["BRNJ27", "BRNK27"], [1, -1])
    assert len(left) == 1 and left[0].children[1].children.value == "BRNK27"
    ctx.triggered = [{"prop_id": "x.n_clicks", "value": None}]
    with pytest.raises(lc.PreventUpdate):  # a remove button that was only just drawn
        lc.update_leg_rows("custom", 1, [None, None], ["BRNJ27", "BRNK27"], [1, -1])
    ctx.triggered_id, ctx.triggered = "la-add-leg", [{"prop_id": "la-add-leg.n_clicks", "value": None}]
    with pytest.raises(lc.PreventUpdate):
        lc.update_leg_rows("custom", None, [], [], [])


def test_creating_an_alert_saves_it_and_its_contracts_and_clears_the_form(page, repo):
    status, counter, level, name = lc.create_alert(1, "spread", ["BRNJ27", "BRNK27"], [1, -1], 0.8, AUTO, "", LIVE, 4)
    assert counter == 5 and level is None and name is None and "created" in str(status)
    (alert,) = repo.get_level_alerts()
    assert alert.direction == ABOVE and alert.level == 0.8
    assert {"BRNJ27", "BRNK27"} <= {c.symbol for c in repo.get_all_contracts()}  # the live poll covers every saved contract
    refused = lc.create_alert(1, "spread", ["BRNJ27", "BRNK27"], [1, -1], "oops", AUTO, "", LIVE, 5)
    assert "price level as a number" in str(refused[0]) and refused[1:] == (lc.no_update,) * 3
    assert len(repo.get_level_alerts()) == 1
    with pytest.raises(lc.PreventUpdate):
        lc.create_alert(None, "spread", [], [], 1, AUTO, "", LIVE, 0)


def test_the_table_lists_active_alerts_closest_first_then_triggered_ones_with_a_delete_button(page, repo):
    stored(repo, level=0.8, name="far")
    stored(repo, level=0.55, name="near")
    done = stored(repo, level=0.9, name="done")
    repo.mark_level_alert_triggered(done.alert_id, 0.95)
    table = alerts_table(repo.get_level_alerts(), {"BRNJ27": 80.0, "BRNK27": 79.5})
    text = str(table)
    assert text.index("'near'") < text.index("'far'") < text.index("'done'")
    assert text.count("la-delete") == 3 and "Triggered" in text and "to go" in text
    assert "No price alerts yet" in str(alerts_table([], {}))
    assert "to go" not in str(alerts_table(repo.get_level_alerts(), {}))  # no live price: no distance


def test_deleting_removes_the_alert_and_ignores_a_new_button(page, repo, monkeypatch):
    alert = stored(repo)

    class Ctx:
        triggered_id = {"type": "la-delete", "index": alert.alert_id}
        triggered = [{"prop_id": "x.n_clicks", "value": None}]

    monkeypatch.setattr(lc, "callback_context", Ctx())
    with pytest.raises(lc.PreventUpdate):
        lc.delete_alert([None], 3)  # a button that was just drawn, not clicked
    assert len(repo.get_level_alerts()) == 1
    Ctx.triggered = [{"prop_id": "x.n_clicks", "value": 1}]
    assert lc.delete_alert([1], 3) == 4 and repo.get_level_alerts() == []
    assert "No price alerts yet" in str(lc.render_table(LIVE, 4))


def test_the_page_is_in_the_menu_and_the_router():
    from ui.layouts.shell import NAV_ITEMS

    assert shell._ROUTES["/level-alerts"] == ("level_alerts_tab", "level_alerts_layout")
    assert any(path == "/level-alerts" for path, _ in NAV_ITEMS)
