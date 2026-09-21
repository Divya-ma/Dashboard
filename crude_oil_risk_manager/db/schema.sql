-- Database schema for the crude oil risk manager.
-- SQLite. All datetimes are stored as TEXT in ISO 8601 UTC format.
-- All JSON blobs are stored as TEXT (JSON-serialized).

CREATE TABLE IF NOT EXISTS contracts (
    contract_id TEXT PRIMARY KEY,          -- the symbol, e.g. "CLZ25"
    product TEXT NOT NULL,
    contract_month INTEGER NOT NULL,
    contract_year INTEGER NOT NULL,
    symbol TEXT NOT NULL UNIQUE,
    multiplier REAL NOT NULL,
    tick_size REAL NOT NULL,
    tick_value REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    expiry_date TEXT,
    first_notice_date TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS structures (
    structure_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    structure_type TEXT NOT NULL,
    products TEXT NOT NULL,                -- JSON array
    status TEXT NOT NULL DEFAULT 'shell',
    created_at TEXT NOT NULL,
    last_modified_at TEXT NOT NULL,
    notes TEXT DEFAULT '',
    close_trigger TEXT,
    closed_at TEXT,
    has_duplicate_tenors INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS legs (
    leg_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES structures(structure_id),
    contract_id TEXT NOT NULL REFERENCES contracts(contract_id),
    ratio INTEGER NOT NULL,
    lots REAL NOT NULL DEFAULT 0.0,
    entry_price REAL,
    average_entry_price REAL,
    is_naked INTEGER NOT NULL DEFAULT 0,
    direction TEXT NOT NULL DEFAULT 'buy',   -- 'buy' or 'sell': flips the PnL sign
    leg_order INTEGER NOT NULL DEFAULT 0    -- preserves user-defined ordering of legs
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES structures(structure_id),
    leg_id TEXT NOT NULL REFERENCES legs(leg_id),
    event_type TEXT NOT NULL,
    lots REAL NOT NULL,
    price REAL NOT NULL,
    direction TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    realized_pnl REAL,
    notes TEXT DEFAULT '',
    stop_loss_price REAL,
    target_price REAL
);

CREATE TABLE IF NOT EXISTS pnl_records (
    record_id TEXT PRIMARY KEY,
    structure_id TEXT NOT NULL REFERENCES structures(structure_id),
    unrealized_pnl REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    total_pnl REAL NOT NULL,
    timestamp TEXT NOT NULL,
    last_price_used TEXT NOT NULL,          -- JSON: {symbol: price}
    is_stale INTEGER NOT NULL DEFAULT 0,
    stale_symbols TEXT NOT NULL DEFAULT '[]' -- JSON array
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    level TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    structure_id TEXT,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    channels_sent TEXT NOT NULL DEFAULT '[]' -- JSON array
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,                    -- JSON-encoded
    updated_at TEXT NOT NULL
);

-- Indexes: foreign keys and common query patterns

CREATE INDEX IF NOT EXISTS idx_legs_structure_id ON legs(structure_id);

CREATE INDEX IF NOT EXISTS idx_trades_structure_id ON trades(structure_id);
CREATE INDEX IF NOT EXISTS idx_trades_leg_id ON trades(leg_id);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);

CREATE INDEX IF NOT EXISTS idx_pnl_records_structure_id ON pnl_records(structure_id);
CREATE INDEX IF NOT EXISTS idx_pnl_records_timestamp ON pnl_records(timestamp);

CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);
CREATE INDEX IF NOT EXISTS idx_alerts_structure_id ON alerts(structure_id);
