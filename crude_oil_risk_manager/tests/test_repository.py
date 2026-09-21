"""Tests for db.repository.Repository — the SQL boundary of the application."""

import pytest
from sqlalchemy import text

from core.models import (
    Alert,
    AlertLevel,
    Contract,
    Leg,
    PnLRecord,
    Structure,
    StructureStatus,
    StructureType,
    Trade,
    TradeEventType,
)
from db.repository import Repository


def make_contract(**overrides) -> Contract:
    defaults = dict(
        product="CL",
        contract_month=12,
        contract_year=2025,
        symbol="CLZ25",
        multiplier=1000.0,
        tick_size=0.01,
        tick_value=10.0,
    )
    defaults.update(overrides)
    return Contract(**defaults)


def make_leg(**overrides) -> Leg:
    defaults = dict(contract=make_contract(), ratio=1)
    defaults.update(overrides)
    return Leg(**defaults)


@pytest.fixture
def repo(tmp_db_path):
    return Repository(str(tmp_db_path))


def test_initialize_db_creates_all_tables(repo):
    expected_tables = {
        "contracts",
        "structures",
        "legs",
        "trades",
        "pnl_records",
        "alerts",
        "settings",
    }
    with repo._engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type = 'table'")
        ).all()
    table_names = {row[0] for row in rows}
    assert expected_tables.issubset(table_names)


def test_save_and_get_contract_roundtrip(repo):
    contract = make_contract()
    repo.save_contract(contract)
    fetched = repo.get_contract("CLZ25")
    assert fetched is not None
    assert fetched == contract


def test_get_contract_not_found_returns_none(repo):
    assert repo.get_contract("NOPE") is None


def test_save_structure_with_two_legs_persists(repo):
    leg1 = make_leg(contract=make_contract(symbol="CLZ25"), ratio=1, lots=5.0, entry_price=75.0)
    leg2 = make_leg(
        contract=make_contract(symbol="CLF26", contract_month=1, contract_year=2026),
        ratio=-1,
        lots=5.0,
        entry_price=76.0,
    )
    structure = Structure(
        name="CL Dec25-Jan26 Spread",
        structure_type=StructureType.SPREAD,
        products=["CL"],
        legs=[leg1, leg2],
        status=StructureStatus.OPEN,
    )
    repo.save_structure(structure)

    with repo._engine.connect() as conn:
        structure_count = conn.execute(
            text("SELECT COUNT(*) FROM structures WHERE structure_id = :id"),
            {"id": structure.structure_id},
        ).scalar_one()
        leg_count = conn.execute(
            text("SELECT COUNT(*) FROM legs WHERE structure_id = :id"),
            {"id": structure.structure_id},
        ).scalar_one()
    assert structure_count == 1
    assert leg_count == 2


def test_get_structure_reconstructs_full_structure(repo):
    leg1 = make_leg(contract=make_contract(symbol="CLZ25"), ratio=1, lots=5.0, entry_price=75.0)
    leg2 = make_leg(
        contract=make_contract(symbol="CLF26", contract_month=1, contract_year=2026),
        ratio=-1,
        lots=5.0,
        entry_price=76.0,
    )
    structure = Structure(
        name="CL Dec25-Jan26 Spread",
        structure_type=StructureType.SPREAD,
        products=["CL"],
        legs=[leg1, leg2],
        status=StructureStatus.OPEN,
    )
    repo.save_structure(structure)

    fetched = repo.get_structure(structure.structure_id)
    assert fetched is not None
    assert fetched.structure_id == structure.structure_id
    assert fetched.name == "CL Dec25-Jan26 Spread"
    assert fetched.leg_count == 2
    fetched_symbols = {leg.contract.symbol for leg in fetched.legs}
    assert fetched_symbols == {"CLZ25", "CLF26"}
    for leg in fetched.legs:
        assert isinstance(leg.contract, Contract)


def test_get_structure_not_found_returns_none(repo):
    assert repo.get_structure("nonexistent") is None


def test_get_all_structures_filters_by_status(repo):
    shell_structure = Structure(
        name="Shell",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[make_leg()],
        status=StructureStatus.SHELL,
    )
    open_structure = Structure(
        name="Open",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[make_leg(contract=make_contract(symbol="CLF26", contract_month=1, contract_year=2026))],
        status=StructureStatus.OPEN,
    )
    repo.save_structure(shell_structure)
    repo.save_structure(open_structure)

    open_only = repo.get_all_structures(status_filter=[StructureStatus.OPEN])
    assert len(open_only) == 1
    assert open_only[0].structure_id == open_structure.structure_id

    all_structures = repo.get_all_structures()
    assert len(all_structures) == 2


def test_update_structure_status_sets_closed_at(repo):
    structure = Structure(
        name="To Close",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[make_leg()],
        status=StructureStatus.OPEN,
    )
    repo.save_structure(structure)
    repo.update_structure_status(structure.structure_id, StructureStatus.CLOSED, close_trigger="manual")

    fetched = repo.get_structure(structure.structure_id)
    assert fetched.status == StructureStatus.CLOSED
    assert fetched.close_trigger == "manual"
    assert fetched.closed_at is not None


def test_update_structure_legs_appends_audit_note(repo):
    leg = make_leg()
    structure = Structure(
        name="Legs Update",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[leg],
        status=StructureStatus.OPEN,
    )
    repo.save_structure(structure)

    updated_leg = leg.model_copy(update={"lots": 20.0, "entry_price": 80.0})
    repo.update_structure_legs(structure.structure_id, [updated_leg])

    fetched = repo.get_structure(structure.structure_id)
    assert fetched.legs[0].lots == 20.0
    assert "legs updated" in fetched.notes


def test_save_trade_then_get_trades_for_structure(repo):
    leg = make_leg()
    structure = Structure(
        name="Trade Test",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[leg],
        status=StructureStatus.OPEN,
    )
    repo.save_structure(structure)

    trade = Trade(
        structure_id=structure.structure_id,
        leg_id=leg.leg_id,
        event_type=TradeEventType.TRADE,
        lots=10.0,
        price=75.5,
        direction="buy",
    )
    repo.save_trade(trade)

    trades = repo.get_trades_for_structure(structure.structure_id)
    assert len(trades) == 1
    assert trades[0].trade_id == trade.trade_id
    assert trades[0].price == 75.5


def test_get_trades_for_leg(repo):
    leg = make_leg()
    structure = Structure(
        name="Leg Trade Test",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[leg],
        status=StructureStatus.OPEN,
    )
    repo.save_structure(structure)
    trade = Trade(
        structure_id=structure.structure_id,
        leg_id=leg.leg_id,
        event_type=TradeEventType.TRADE,
        lots=10.0,
        price=75.5,
        direction="buy",
    )
    repo.save_trade(trade)

    trades = repo.get_trades_for_leg(leg.leg_id)
    assert len(trades) == 1
    assert trades[0].leg_id == leg.leg_id


def test_save_pnl_record_then_get_latest_pnl(repo):
    structure = Structure(
        name="PnL Test",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[make_leg()],
        status=StructureStatus.OPEN,
    )
    repo.save_structure(structure)

    record = PnLRecord(
        structure_id=structure.structure_id,
        unrealized_pnl=100.0,
        realized_pnl=50.0,
        total_pnl=150.0,
        last_price_used={"CLZ25": 76.0},
    )
    repo.save_pnl_record(record)

    latest = repo.get_latest_pnl(structure.structure_id)
    assert latest is not None
    assert latest.record_id == record.record_id
    assert latest.total_pnl == 150.0
    assert latest.last_price_used == {"CLZ25": 76.0}


def test_get_latest_pnl_none_when_no_records(repo):
    assert repo.get_latest_pnl("nonexistent") is None


def test_save_alert_then_get_unacknowledged_alerts(repo):
    alert = Alert(level=AlertLevel.WARNING, title="Loss limit", body="Structure exceeded loss limit.")
    repo.save_alert(alert)

    unacknowledged = repo.get_unacknowledged_alerts()
    assert any(a.alert_id == alert.alert_id for a in unacknowledged)


def test_acknowledge_alert_removes_from_unacknowledged(repo):
    alert = Alert(level=AlertLevel.CRITICAL, title="Stop hit", body="Portfolio stop hit.")
    repo.save_alert(alert)
    repo.acknowledge_alert(alert.alert_id)

    unacknowledged = repo.get_unacknowledged_alerts()
    assert all(a.alert_id != alert.alert_id for a in unacknowledged)

    history = repo.get_alert_history()
    assert any(a.alert_id == alert.alert_id and a.acknowledged for a in history)


def test_set_setting_get_setting_roundtrip_various_types(repo):
    repo.set_setting("max_lots", 100)
    repo.set_setting("var_confidence", 0.95)
    repo.set_setting("default_currency", "USD")
    repo.set_setting("thresholds", {"pnl_stop": -50000.0, "structure_max_loss": -10000.0})

    assert repo.get_setting("max_lots") == 100
    assert repo.get_setting("var_confidence") == 0.95
    assert repo.get_setting("default_currency") == "USD"
    assert repo.get_setting("thresholds") == {"pnl_stop": -50000.0, "structure_max_loss": -10000.0}


def test_get_setting_default_when_missing(repo):
    assert repo.get_setting("nonexistent_key", default="fallback") == "fallback"


def test_get_all_settings(repo):
    repo.set_setting("a", 1)
    repo.set_setting("b", "two")
    all_settings = repo.get_all_settings()
    assert all_settings == {"a": 1, "b": "two"}


def test_delete_contract_raises_when_referenced(repo):
    leg = make_leg(contract=make_contract(symbol="CLZ25"))
    structure = Structure(
        name="Referencing Structure",
        structure_type=StructureType.OUTRIGHT,
        products=["CL"],
        legs=[leg],
        status=StructureStatus.OPEN,
    )
    repo.save_structure(structure)

    with pytest.raises(ValueError):
        repo.delete_contract("CLZ25")


def test_delete_contract_succeeds_when_unreferenced(repo):
    contract = make_contract(symbol="CLZ25")
    repo.save_contract(contract)
    repo.delete_contract("CLZ25")
    assert repo.get_contract("CLZ25") is None


def test_get_first_pnl_since_returns_earliest_record_at_or_after_cutoff(repo):
    from datetime import datetime, timedelta, timezone

    structure = Structure(
        name="PnL history", structure_type=StructureType.OUTRIGHT, products=["CL"],
        legs=[make_leg()], status=StructureStatus.OPEN,
    )
    repo.save_structure(structure)
    midnight = datetime(2026, 9, 21, tzinfo=timezone.utc)

    def record(total, when):
        return PnLRecord(
            structure_id=structure.structure_id, unrealized_pnl=total, realized_pnl=0.0,
            total_pnl=total, timestamp=when,
        )

    repo.save_pnl_record(record(1.0, midnight - timedelta(hours=1)))  # yesterday
    repo.save_pnl_record(record(2.0, midnight + timedelta(hours=3)))
    repo.save_pnl_record(record(3.0, midnight + timedelta(hours=1)))  # earliest today
    repo.save_pnl_record(record(4.0, midnight + timedelta(hours=5)))

    first = repo.get_first_pnl_since(structure.structure_id, midnight)
    assert first is not None and first.total_pnl == 3.0
    assert repo.get_first_pnl_since(structure.structure_id, midnight + timedelta(days=1)) is None
    assert repo.get_first_pnl_since("no-such-structure", midnight) is None


# ---------- schema migration and new columns ----------

import sqlite3


def _old_database(path):
    """A database created before legs.direction and trades.stop_loss_price / target_price existed."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE contracts (contract_id TEXT PRIMARY KEY, product TEXT NOT NULL, contract_month INTEGER NOT NULL,
            contract_year INTEGER NOT NULL, symbol TEXT NOT NULL UNIQUE, multiplier REAL NOT NULL, tick_size REAL NOT NULL,
            tick_value REAL NOT NULL, currency TEXT NOT NULL DEFAULT 'USD', expiry_date TEXT, first_notice_date TEXT,
            created_at TEXT NOT NULL);
        CREATE TABLE structures (structure_id TEXT PRIMARY KEY, name TEXT NOT NULL, structure_type TEXT NOT NULL,
            products TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'shell', created_at TEXT NOT NULL,
            last_modified_at TEXT NOT NULL, notes TEXT DEFAULT '', close_trigger TEXT, closed_at TEXT,
            has_duplicate_tenors INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE legs (leg_id TEXT PRIMARY KEY, structure_id TEXT NOT NULL, contract_id TEXT NOT NULL,
            ratio INTEGER NOT NULL, lots REAL NOT NULL DEFAULT 0.0, entry_price REAL, average_entry_price REAL,
            is_naked INTEGER NOT NULL DEFAULT 0, leg_order INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE trades (trade_id TEXT PRIMARY KEY, structure_id TEXT NOT NULL, leg_id TEXT NOT NULL,
            event_type TEXT NOT NULL, lots REAL NOT NULL, price REAL NOT NULL, direction TEXT NOT NULL,
            timestamp TEXT NOT NULL, realized_pnl REAL, notes TEXT DEFAULT '');
        INSERT INTO contracts VALUES ('CLZ26','CL',12,2026,'CLZ26',1000,0.01,10,'USD',NULL,NULL,'2026-01-01T00:00:00+00:00');
        INSERT INTO structures VALUES ('sold','Sold','outright','["CL"]','open','2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00','',NULL,NULL,0);
        INSERT INTO structures VALUES ('bought','Bought','outright','["CL"]','open','2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00','',NULL,NULL,0);
        INSERT INTO legs VALUES ('l-sold','sold','CLZ26',1,10,0.45,0.45,0,0);
        INSERT INTO legs VALUES ('l-bought','bought','CLZ26',1,10,0.45,0.45,0,0);
        INSERT INTO trades VALUES ('t1','sold','l-sold','trade',10,0.45,'sell','2026-01-02T00:00:00+00:00',NULL,'');
        INSERT INTO trades VALUES ('t2','bought','l-bought','trade',10,0.45,'buy','2026-01-02T00:00:00+00:00',NULL,'');
        """
    )
    conn.commit()
    conn.close()


def test_migration_adds_columns_and_backfills_direction_from_first_trade(tmp_path):
    path = str(tmp_path / "old.db")
    _old_database(path)
    repo = Repository(path)
    assert repo.get_structure("sold").legs[0].direction == "sell"
    assert repo.get_structure("bought").legs[0].direction == "buy"
    trade = repo.get_trades_for_structure("sold")[0]
    assert trade.stop_loss_price is None and trade.target_price is None


def test_migration_is_idempotent_and_does_not_overwrite_direction(tmp_path):
    path = str(tmp_path / "old.db")
    _old_database(path)
    Repository(path)
    again = Repository(path)  # second start: ALTERs fail silently, nothing is re-backfilled
    leg = again.get_structure("bought").legs[0]
    again.update_structure_legs("bought", [leg.model_copy(update={"direction": "sell"})])
    assert Repository(path).get_structure("bought").legs[0].direction == "sell"


def test_direction_and_alert_levels_round_trip(repo):
    from core.models import Contract, Leg, Structure, StructureStatus, StructureType, Trade, TradeEventType

    contract = Contract(product="CL", contract_month=12, contract_year=2026, symbol="CLZ26",
                        multiplier=1000, tick_size=0.01, tick_value=10)
    leg = Leg(contract=contract, ratio=1, lots=5, entry_price=75.0, direction="sell")
    structure = Structure(name="S", structure_type=StructureType.OUTRIGHT, products=["CL"], legs=[leg],
                          status=StructureStatus.OPEN)
    repo.save_contract(contract)
    repo.save_structure(structure)
    repo.save_trade(Trade(structure_id=structure.structure_id, leg_id=leg.leg_id, event_type=TradeEventType.TRADE,
                          lots=5, price=75.0, direction="sell", stop_loss_price=77.0, target_price=70.0))
    assert repo.get_structure(structure.structure_id).legs[0].direction == "sell"
    trade = repo.get_trades_for_structure(structure.structure_id)[0]
    assert (trade.stop_loss_price, trade.target_price) == (77.0, 70.0)


def test_leg_direction_validator():
    import pytest as _pytest

    from core.models import Contract, Leg

    contract = Contract(product="CL", contract_month=12, contract_year=2026, symbol="CLZ26",
                        multiplier=1000, tick_size=0.01, tick_value=10)
    assert Leg(contract=contract, ratio=1).direction == "buy"
    with _pytest.raises(ValueError):
        Leg(contract=contract, ratio=1, direction="hold")
