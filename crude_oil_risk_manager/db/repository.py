"""Repository layer: the single SQL boundary for the crude oil risk manager.

This module is the only place in the application that reads from or writes
to the SQLite database. All other modules interact with the database
exclusively through the Repository class, using core.models types as
input/output — raw SQLite rows never leave this module.

Uses SQLAlchemy Core (text() queries + engine/connection management) only;
no ORM mapping.
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine import Connection, Engine

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

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


class Repository:
    """The single SQL boundary: all database access goes through this class."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._engine: Engine = create_engine(f"sqlite:///{db_path}")
        self.initialize_db()

    def initialize_db(self) -> None:
        """Read and execute db/schema.sql. Safe to call repeatedly (IF NOT EXISTS)."""
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
        with self._engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))
        self._migrate_columns()

    def _migrate_columns(self) -> None:
        """Add columns introduced after the first release to databases created without them.

        SQLite has no ADD COLUMN IF NOT EXISTS, so each ALTER is attempted and the
        "duplicate column" error is ignored. When legs.direction is added to an existing
        database it is back-filled from each structure's first entry trade, so positions
        entered as 'sell' before direction was stored keep the right PnL sign.
        """
        for statement in (
            "ALTER TABLE trades ADD COLUMN stop_loss_price REAL",
            "ALTER TABLE trades ADD COLUMN target_price REAL",
        ):
            self._try_alter(statement)
        if self._try_alter("ALTER TABLE legs ADD COLUMN direction TEXT NOT NULL DEFAULT 'buy'"):
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE legs SET direction = COALESCE((
                            SELECT t.direction FROM trades t
                            WHERE t.structure_id = legs.structure_id AND t.event_type = 'trade'
                            ORDER BY t.timestamp ASC LIMIT 1
                        ), 'buy')
                        """
                    )
                )

    def _try_alter(self, statement: str) -> bool:
        """Run an ALTER TABLE; False (ignored) if the column already exists."""
        try:
            with self._engine.begin() as conn:
                conn.execute(text(statement))
            return True
        except OperationalError:
            return False

    # ------------------------------------------------------------------
    # Contract methods
    # ------------------------------------------------------------------

    def save_contract(self, contract: Contract) -> None:
        """INSERT OR REPLACE into contracts."""
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT OR REPLACE INTO contracts (
                        contract_id, product, contract_month, contract_year,
                        symbol, multiplier, tick_size, tick_value, currency,
                        expiry_date, first_notice_date, created_at
                    ) VALUES (
                        :contract_id, :product, :contract_month, :contract_year,
                        :symbol, :multiplier, :tick_size, :tick_value, :currency,
                        :expiry_date, :first_notice_date, :created_at
                    )
                    """
                ),
                {
                    "contract_id": contract.symbol,
                    "product": contract.product,
                    "contract_month": contract.contract_month,
                    "contract_year": contract.contract_year,
                    "symbol": contract.symbol,
                    "multiplier": contract.multiplier,
                    "tick_size": contract.tick_size,
                    "tick_value": contract.tick_value,
                    "currency": contract.currency,
                    "expiry_date": contract.expiry_date.isoformat() if contract.expiry_date else None,
                    "first_notice_date": (
                        contract.first_notice_date.isoformat() if contract.first_notice_date else None
                    ),
                    "created_at": _utcnow_iso(),
                },
            )

    def get_contract(self, symbol: str) -> Contract | None:
        """Fetch a contract by symbol; return None if not found."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM contracts WHERE symbol = :symbol"), {"symbol": symbol}
            ).mappings().first()
        if row is None:
            return None
        return self._row_to_contract(row)

    def get_all_contracts(self) -> list[Contract]:
        """Return all contracts ordered by product, contract_year, contract_month."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT * FROM contracts ORDER BY product, contract_year, contract_month"
                )
            ).mappings().all()
        return [self._row_to_contract(row) for row in rows]

    def delete_contract(self, symbol: str) -> None:
        """Delete a contract by symbol. Raises ValueError if referenced by a leg."""
        with self._engine.begin() as conn:
            referenced = conn.execute(
                text("SELECT COUNT(*) FROM legs WHERE contract_id = :contract_id"),
                {"contract_id": symbol},
            ).scalar_one()
            if referenced:
                raise ValueError(
                    f"cannot delete contract {symbol!r}: referenced by {referenced} leg(s)"
                )
            conn.execute(text("DELETE FROM contracts WHERE symbol = :symbol"), {"symbol": symbol})

    @staticmethod
    def _row_to_contract(row) -> Contract:
        return Contract(
            product=row["product"],
            contract_month=row["contract_month"],
            contract_year=row["contract_year"],
            symbol=row["symbol"],
            multiplier=row["multiplier"],
            tick_size=row["tick_size"],
            tick_value=row["tick_value"],
            currency=row["currency"],
            expiry_date=_parse_date(row["expiry_date"]),
            first_notice_date=_parse_date(row["first_notice_date"]),
        )

    # ------------------------------------------------------------------
    # Structure methods
    # ------------------------------------------------------------------

    def save_structure(self, structure: Structure) -> None:
        """Upsert the structure row and all its legs in a single transaction."""
        with self._engine.begin() as conn:
            self._upsert_structure_row(conn, structure)
            for order, leg in enumerate(structure.legs):
                self._upsert_leg_row(conn, structure.structure_id, leg, order)

    @staticmethod
    def _upsert_structure_row(conn: Connection, structure: Structure) -> None:
        conn.execute(
            text(
                """
                INSERT OR REPLACE INTO structures (
                    structure_id, name, structure_type, products, status,
                    created_at, last_modified_at, notes, close_trigger,
                    closed_at, has_duplicate_tenors
                ) VALUES (
                    :structure_id, :name, :structure_type, :products, :status,
                    :created_at, :last_modified_at, :notes, :close_trigger,
                    :closed_at, :has_duplicate_tenors
                )
                """
            ),
            {
                "structure_id": structure.structure_id,
                "name": structure.name,
                "structure_type": structure.structure_type.value,
                "products": json.dumps(structure.products),
                "status": structure.status.value,
                "created_at": structure.created_at.isoformat(),
                "last_modified_at": structure.last_modified_at.isoformat(),
                "notes": structure.notes,
                "close_trigger": structure.close_trigger,
                "closed_at": structure.closed_at.isoformat() if structure.closed_at else None,
                "has_duplicate_tenors": int(structure.has_duplicate_tenors),
            },
        )

    def _upsert_leg_row(self, conn: Connection, structure_id: str, leg: Leg, order: int) -> None:
        # Ensure the leg's embedded contract exists so the FK is satisfiable.
        conn.execute(
            text(
                """
                INSERT OR REPLACE INTO contracts (
                    contract_id, product, contract_month, contract_year,
                    symbol, multiplier, tick_size, tick_value, currency,
                    expiry_date, first_notice_date, created_at
                ) VALUES (
                    :contract_id, :product, :contract_month, :contract_year,
                    :symbol, :multiplier, :tick_size, :tick_value, :currency,
                    :expiry_date, :first_notice_date,
                    COALESCE(
                        (SELECT created_at FROM contracts WHERE contract_id = :contract_id),
                        :created_at
                    )
                )
                """
            ),
            {
                "contract_id": leg.contract.symbol,
                "product": leg.contract.product,
                "contract_month": leg.contract.contract_month,
                "contract_year": leg.contract.contract_year,
                "symbol": leg.contract.symbol,
                "multiplier": leg.contract.multiplier,
                "tick_size": leg.contract.tick_size,
                "tick_value": leg.contract.tick_value,
                "currency": leg.contract.currency,
                "expiry_date": leg.contract.expiry_date.isoformat() if leg.contract.expiry_date else None,
                "first_notice_date": (
                    leg.contract.first_notice_date.isoformat()
                    if leg.contract.first_notice_date
                    else None
                ),
                "created_at": _utcnow_iso(),
            },
        )
        conn.execute(
            text(
                """
                INSERT OR REPLACE INTO legs (
                    leg_id, structure_id, contract_id, ratio, lots,
                    entry_price, average_entry_price, is_naked, direction, leg_order
                ) VALUES (
                    :leg_id, :structure_id, :contract_id, :ratio, :lots,
                    :entry_price, :average_entry_price, :is_naked, :direction, :leg_order
                )
                """
            ),
            {
                "leg_id": leg.leg_id,
                "structure_id": structure_id,
                "contract_id": leg.contract.symbol,
                "ratio": leg.ratio,
                "lots": leg.lots,
                "entry_price": leg.entry_price,
                "average_entry_price": leg.average_entry_price,
                "is_naked": int(leg.is_naked),
                "direction": leg.direction,
                "leg_order": order,
            },
        )

    def get_structure(self, structure_id: str) -> Structure | None:
        """Fetch a structure with all its legs and each leg's contract."""
        with self._engine.connect() as conn:
            structure_row = conn.execute(
                text("SELECT * FROM structures WHERE structure_id = :structure_id"),
                {"structure_id": structure_id},
            ).mappings().first()
            if structure_row is None:
                return None
            leg_rows = conn.execute(
                text(
                    "SELECT * FROM legs WHERE structure_id = :structure_id ORDER BY leg_order ASC"
                ),
                {"structure_id": structure_id},
            ).mappings().all()
            legs = [self._row_to_leg(conn, row) for row in leg_rows]
        return self._row_to_structure(structure_row, legs)

    def get_all_structures(
        self, status_filter: list[StructureStatus] | None = None
    ) -> list[Structure]:
        """Return all (optionally status-filtered) structures with full legs and contracts."""
        with self._engine.connect() as conn:
            if status_filter:
                placeholders = {f"status{i}": s.value for i, s in enumerate(status_filter)}
                clause = ", ".join(f":{k}" for k in placeholders)
                query = text(
                    f"SELECT * FROM structures WHERE status IN ({clause}) ORDER BY created_at DESC"
                )
                structure_rows = conn.execute(query, placeholders).mappings().all()
            else:
                structure_rows = conn.execute(
                    text("SELECT * FROM structures ORDER BY created_at DESC")
                ).mappings().all()

            structures = []
            for structure_row in structure_rows:
                leg_rows = conn.execute(
                    text(
                        "SELECT * FROM legs WHERE structure_id = :structure_id ORDER BY leg_order ASC"
                    ),
                    {"structure_id": structure_row["structure_id"]},
                ).mappings().all()
                legs = [self._row_to_leg(conn, row) for row in leg_rows]
                structures.append(self._row_to_structure(structure_row, legs))
        return structures

    def update_structure_status(
        self,
        structure_id: str,
        status: StructureStatus,
        close_trigger: str | None = None,
    ) -> None:
        """Update a structure's status and last_modified_at; set closed_at if CLOSED."""
        now = _utcnow_iso()
        closed_at = now if status == StructureStatus.CLOSED else None
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE structures
                    SET status = :status,
                        last_modified_at = :last_modified_at,
                        close_trigger = COALESCE(:close_trigger, close_trigger),
                        closed_at = COALESCE(:closed_at, closed_at)
                    WHERE structure_id = :structure_id
                    """
                ),
                {
                    "status": status.value,
                    "last_modified_at": now,
                    "close_trigger": close_trigger,
                    "closed_at": closed_at,
                    "structure_id": structure_id,
                },
            )

    def update_structure_legs(
        self, structure_id: str, legs: list[Leg], audit_note: str = "legs updated"
    ) -> None:
        """Upsert all legs for a structure and append a timestamped audit line to its notes."""
        now = _utcnow_iso()
        with self._engine.begin() as conn:
            for order, leg in enumerate(legs):
                self._upsert_leg_row(conn, structure_id, leg, order)
            conn.execute(
                text(
                    """
                    UPDATE structures
                    SET notes = notes || :audit_note,
                        last_modified_at = :last_modified_at
                    WHERE structure_id = :structure_id
                    """
                ),
                {
                    "audit_note": f"\n[{now}] {audit_note}",
                    "last_modified_at": now,
                    "structure_id": structure_id,
                },
            )

    def _row_to_leg(self, conn: Connection, row) -> Leg:
        contract_row = conn.execute(
            text("SELECT * FROM contracts WHERE contract_id = :contract_id"),
            {"contract_id": row["contract_id"]},
        ).mappings().first()
        contract = self._row_to_contract(contract_row)
        return Leg(
            leg_id=row["leg_id"],
            contract=contract,
            ratio=row["ratio"],
            lots=row["lots"],
            entry_price=row["entry_price"],
            average_entry_price=row["average_entry_price"],
            is_naked=bool(row["is_naked"]),
            direction=row["direction"] or "buy",
        )

    @staticmethod
    def _row_to_structure(row, legs: list[Leg]) -> Structure:
        return Structure(
            structure_id=row["structure_id"],
            name=row["name"],
            structure_type=StructureType(row["structure_type"]),
            products=json.loads(row["products"]),
            legs=legs,
            status=StructureStatus(row["status"]),
            created_at=_parse_dt(row["created_at"]),
            last_modified_at=_parse_dt(row["last_modified_at"]),
            notes=row["notes"] or "",
            close_trigger=row["close_trigger"],
            closed_at=_parse_dt(row["closed_at"]),
            has_duplicate_tenors=bool(row["has_duplicate_tenors"]),
        )

    # ------------------------------------------------------------------
    # Trade methods
    # ------------------------------------------------------------------

    def save_trade(self, trade: Trade) -> None:
        """INSERT into trades."""
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO trades (
                        trade_id, structure_id, leg_id, event_type, lots,
                        price, direction, timestamp, realized_pnl, notes,
                        stop_loss_price, target_price
                    ) VALUES (
                        :trade_id, :structure_id, :leg_id, :event_type, :lots,
                        :price, :direction, :timestamp, :realized_pnl, :notes,
                        :stop_loss_price, :target_price
                    )
                    """
                ),
                {
                    "trade_id": trade.trade_id,
                    "structure_id": trade.structure_id,
                    "leg_id": trade.leg_id,
                    "event_type": trade.event_type.value,
                    "lots": trade.lots,
                    "price": trade.price,
                    "direction": trade.direction,
                    "timestamp": trade.timestamp.isoformat(),
                    "realized_pnl": trade.realized_pnl,
                    "notes": trade.notes,
                    "stop_loss_price": trade.stop_loss_price,
                    "target_price": trade.target_price,
                },
            )

    def get_trades_for_structure(self, structure_id: str) -> list[Trade]:
        """Return all trades for a structure, ordered by timestamp ASC."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT * FROM trades WHERE structure_id = :structure_id ORDER BY timestamp ASC"
                ),
                {"structure_id": structure_id},
            ).mappings().all()
        return [self._row_to_trade(row) for row in rows]

    def get_trades_for_leg(self, leg_id: str) -> list[Trade]:
        """Return all trades for a specific leg, ordered by timestamp ASC."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM trades WHERE leg_id = :leg_id ORDER BY timestamp ASC"),
                {"leg_id": leg_id},
            ).mappings().all()
        return [self._row_to_trade(row) for row in rows]

    @staticmethod
    def _row_to_trade(row) -> Trade:
        return Trade(
            trade_id=row["trade_id"],
            structure_id=row["structure_id"],
            leg_id=row["leg_id"],
            event_type=TradeEventType(row["event_type"]),
            lots=row["lots"],
            price=row["price"],
            direction=row["direction"],
            timestamp=_parse_dt(row["timestamp"]),
            realized_pnl=row["realized_pnl"],
            notes=row["notes"] or "",
            stop_loss_price=row["stop_loss_price"],
            target_price=row["target_price"],
        )

    # ------------------------------------------------------------------
    # PnL methods
    # ------------------------------------------------------------------

    def save_pnl_record(self, record: PnLRecord) -> None:
        """INSERT into pnl_records."""
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO pnl_records (
                        record_id, structure_id, unrealized_pnl, realized_pnl,
                        total_pnl, timestamp, last_price_used, is_stale, stale_symbols
                    ) VALUES (
                        :record_id, :structure_id, :unrealized_pnl, :realized_pnl,
                        :total_pnl, :timestamp, :last_price_used, :is_stale, :stale_symbols
                    )
                    """
                ),
                {
                    "record_id": record.record_id,
                    "structure_id": record.structure_id,
                    "unrealized_pnl": record.unrealized_pnl,
                    "realized_pnl": record.realized_pnl,
                    "total_pnl": record.total_pnl,
                    "timestamp": record.timestamp.isoformat(),
                    "last_price_used": json.dumps(record.last_price_used),
                    "is_stale": int(record.is_stale),
                    "stale_symbols": json.dumps(record.stale_symbols),
                },
            )

    def get_latest_pnl(self, structure_id: str) -> PnLRecord | None:
        """Return the most recent PnLRecord for the structure."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT * FROM pnl_records
                    WHERE structure_id = :structure_id
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """
                ),
                {"structure_id": structure_id},
            ).mappings().first()
        if row is None:
            return None
        return self._row_to_pnl_record(row)

    def get_first_pnl_since(self, structure_id: str, since: datetime) -> PnLRecord | None:
        """Return the earliest PnLRecord at or after `since` (e.g. the first of the day)."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT * FROM pnl_records
                    WHERE structure_id = :structure_id AND timestamp >= :since
                    ORDER BY timestamp ASC
                    LIMIT 1
                    """
                ),
                {"structure_id": structure_id, "since": since.isoformat()},
            ).mappings().first()
        if row is None:
            return None
        return self._row_to_pnl_record(row)

    def get_pnl_history(
        self,
        structure_id: str,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
    ) -> list[PnLRecord]:
        """Return PnL history for a structure, optionally date-filtered, ordered ASC."""
        query = "SELECT * FROM pnl_records WHERE structure_id = :structure_id"
        params: dict[str, Any] = {"structure_id": structure_id}
        if from_dt is not None:
            query += " AND timestamp >= :from_dt"
            params["from_dt"] = from_dt.isoformat()
        if to_dt is not None:
            query += " AND timestamp <= :to_dt"
            params["to_dt"] = to_dt.isoformat()
        query += " ORDER BY timestamp ASC"
        with self._engine.connect() as conn:
            rows = conn.execute(text(query), params).mappings().all()
        return [self._row_to_pnl_record(row) for row in rows]

    @staticmethod
    def _row_to_pnl_record(row) -> PnLRecord:
        return PnLRecord(
            record_id=row["record_id"],
            structure_id=row["structure_id"],
            unrealized_pnl=row["unrealized_pnl"],
            realized_pnl=row["realized_pnl"],
            total_pnl=row["total_pnl"],
            timestamp=_parse_dt(row["timestamp"]),
            last_price_used=json.loads(row["last_price_used"]),
            is_stale=bool(row["is_stale"]),
            stale_symbols=json.loads(row["stale_symbols"]),
        )

    # ------------------------------------------------------------------
    # Alert methods
    # ------------------------------------------------------------------

    def save_alert(self, alert: Alert) -> None:
        """INSERT into alerts."""
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO alerts (
                        alert_id, level, title, body, timestamp,
                        structure_id, acknowledged, channels_sent
                    ) VALUES (
                        :alert_id, :level, :title, :body, :timestamp,
                        :structure_id, :acknowledged, :channels_sent
                    )
                    """
                ),
                {
                    "alert_id": alert.alert_id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "body": alert.body,
                    "timestamp": alert.timestamp.isoformat(),
                    "structure_id": alert.structure_id,
                    "acknowledged": int(alert.acknowledged),
                    "channels_sent": json.dumps(alert.channels_sent),
                },
            )

    def get_unacknowledged_alerts(self) -> list[Alert]:
        """Return all alerts where acknowledged == 0, ordered by timestamp DESC."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY timestamp DESC"
                )
            ).mappings().all()
        return [self._row_to_alert(row) for row in rows]

    def acknowledge_alert(self, alert_id: str) -> None:
        """Set acknowledged = 1 for the given alert."""
        with self._engine.begin() as conn:
            conn.execute(
                text("UPDATE alerts SET acknowledged = 1 WHERE alert_id = :alert_id"),
                {"alert_id": alert_id},
            )

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """Return the most recent alerts (all statuses), limited to `limit` rows."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT :limit"),
                {"limit": limit},
            ).mappings().all()
        return [self._row_to_alert(row) for row in rows]

    @staticmethod
    def _row_to_alert(row) -> Alert:
        return Alert(
            alert_id=row["alert_id"],
            level=AlertLevel(row["level"]),
            title=row["title"],
            body=row["body"],
            timestamp=_parse_dt(row["timestamp"]),
            structure_id=row["structure_id"],
            acknowledged=bool(row["acknowledged"]),
            channels_sent=json.loads(row["channels_sent"]),
        )

    # ------------------------------------------------------------------
    # Settings methods
    # ------------------------------------------------------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Fetch a JSON-decoded setting value; return default if key not found."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT value FROM settings WHERE key = :key"), {"key": key}
            ).mappings().first()
        if row is None:
            return default
        return json.loads(row["value"])

    def set_setting(self, key: str, value: Any) -> None:
        """JSON-encode value and INSERT OR REPLACE into settings."""
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT OR REPLACE INTO settings (key, value, updated_at)
                    VALUES (:key, :value, :updated_at)
                    """
                ),
                {"key": key, "value": json.dumps(value), "updated_at": _utcnow_iso()},
            )

    def get_all_settings(self) -> dict[str, Any]:
        """Return all settings as a plain dict, JSON-decoded."""
        with self._engine.connect() as conn:
            rows = conn.execute(text("SELECT key, value FROM settings")).mappings().all()
        return {row["key"]: json.loads(row["value"]) for row in rows}
