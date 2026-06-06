"""
TradingDatabase — SQLite-backed persistent storage for trading data.

Replaces the JSONL append-only log pattern with indexed, transactional storage.
Single file: workspace/trading.db with WAL mode for concurrent safety.

Design:
- Uses Python's built-in sqlite3 (no new dependencies)
- WAL mode enabled: PRAGMA journal_mode=WAL
- All writes go through a context manager for safe transactions
- JSONL files kept as backup during transition (controlled by LEGACY_JSONL_BACKUP)
"""

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("./workspace/trading.db")


class TradingDatabase:
    """Thread-safe SQLite-backed trading database.

    Usage:
        db = TradingDatabase()
        with db.transaction() as conn:
            conn.execute("INSERT INTO trades ...")
    """

    _instances: Dict[str, "TradingDatabase"] = {}
    _lock = threading.Lock()

    VALID_TABLES = frozenset({
        "trades", "experiments", "pipeline_results",
        "oos_results", "validation_trades", "api_cache", "_migrations"
    })

    def __new__(cls, db_path=None, legacy_backup=True):
        path = Path(db_path) if isinstance(db_path, str) else (db_path or DB_PATH)
        key = str(path)
        with cls._lock:
            if key not in cls._instances:
                instance = super().__new__(cls)
                instance.db_path = path
                instance.legacy_backup = legacy_backup
                if str(instance.db_path) != ":memory:":
                    instance.db_path.parent.mkdir(parents=True, exist_ok=True)
                instance._initialized = False
                cls._instances[key] = instance
            return cls._instances[key]

    def __init__(self, db_path=None, legacy_backup=True) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._init_schema()

    # ── Schema ──

    SCHEMA_TRADES = """
    CREATE TABLE IF NOT EXISTS trades (
        id TEXT PRIMARY KEY,
        strategy_id TEXT NOT NULL,
        pair TEXT NOT NULL,
        side TEXT NOT NULL,
        entry_price REAL,
        exit_price REAL,
        quantity REAL,
        pnl REAL,
        pnl_pct REAL,
        fee_total REAL,
        slippage_total REAL,
        regime TEXT,
        timestamp_open INTEGER NOT NULL,
        timestamp_close INTEGER,
        metadata TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id, timestamp_open);
    CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp_open DESC);
    """

    SCHEMA_EXPERIMENTS = """
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        strategy_id TEXT NOT NULL,
        strategy_type TEXT NOT NULL,
        params TEXT NOT NULL,
        metrics TEXT NOT NULL,
        regime TEXT,
        research_window_end TEXT,
        created_at INTEGER NOT NULL,
        status TEXT DEFAULT 'pending',
        verdict TEXT DEFAULT 'discarded'
    );
    CREATE INDEX IF NOT EXISTS idx_experiments_strategy ON experiments(strategy_id);
    CREATE INDEX IF NOT EXISTS idx_experiments_regime ON experiments(regime, status);
    """

    SCHEMA_OOS_RESULTS = """
    CREATE TABLE IF NOT EXISTS oos_results (
        id TEXT PRIMARY KEY,
        strategy_id TEXT NOT NULL,
        strategy_type TEXT NOT NULL,
        sharpe REAL,
        net_sharpe REAL,
        win_rate REAL,
        max_drawdown REAL,
        trade_count INTEGER,
        passed INTEGER NOT NULL,
        recommendation TEXT,
        validated_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_oos_strategy ON oos_results(strategy_id);
    """

    SCHEMA_PIPELINE_RESULTS = """
    CREATE TABLE IF NOT EXISTS pipeline_results (
        id TEXT PRIMARY KEY,
        strategy_id TEXT NOT NULL,
        strategy_type TEXT NOT NULL,
        gate TEXT NOT NULL,
        passed INTEGER NOT NULL,
        details TEXT,
        created_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_pipeline_strategy ON pipeline_results(strategy_id);
    """

    SCHEMA_VALIDATION_TRADES = """
    CREATE TABLE IF NOT EXISTS validation_trades (
        id TEXT PRIMARY KEY,
        strategy_id TEXT NOT NULL,
        pair TEXT NOT NULL,
        pnl REAL,
        position_size REAL,
        timestamp INTEGER NOT NULL,
        metadata TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_validation_strategy ON validation_trades(strategy_id);
    """

    SCHEMA_MIGRATIONS = """
    CREATE TABLE IF NOT EXISTS _migrations (
        name TEXT PRIMARY KEY,
        applied_at INTEGER NOT NULL
    );
    """

    SCHEMA_API_CACHE = """
    CREATE TABLE IF NOT EXISTS api_cache (
        cache_key TEXT PRIMARY KEY,
        data TEXT NOT NULL,
        source TEXT NOT NULL,
        cached_at INTEGER NOT NULL,
        ttl_seconds INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_api_cache_source ON api_cache(source, cached_at);
    """

    def _init_schema(self):
        """Create all tables and indexes if they don't exist."""
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            for schema in [
                self.SCHEMA_MIGRATIONS,
                self.SCHEMA_API_CACHE,
                self.SCHEMA_TRADES,
                self.SCHEMA_EXPERIMENTS,
                self.SCHEMA_OOS_RESULTS,
                self.SCHEMA_PIPELINE_RESULTS,
                self.SCHEMA_VALIDATION_TRADES,
            ]:
                conn.executescript(schema)
            logger.info("Database schema initialised at %s", self.db_path)

    # ── Connection management ──

    @contextmanager
    def _connect(self):
        """Get a raw connection (internal use)."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator:
        """Context manager for safe transactions.

        Usage:
            with db.transaction() as conn:
                conn.execute("INSERT INTO ...", ...)
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── CRUD: Trades ──

    def insert_trade(self, trade: dict) -> str:
        """Insert a trade record."""
        import uuid
        tid = trade.get("id", uuid.uuid4().hex)
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO trades
                (id, strategy_id, pair, side, entry_price, exit_price,
                 quantity, pnl, pnl_pct, fee_total, slippage_total,
                 regime, timestamp_open, timestamp_close, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tid,
                    trade.get("strategy_id", ""),
                    trade.get("pair", ""),
                    trade.get("side", ""),
                    trade.get("entry_price"),
                    trade.get("exit_price"),
                    trade.get("quantity"),
                    trade.get("pnl"),
                    trade.get("pnl_pct"),
                    trade.get("fee_total", 0),
                    trade.get("slippage_total", 0),
                    trade.get("regime", ""),
                    int(trade.get("timestamp_open", 0)),
                    int(trade.get("timestamp_close", 0)) if trade.get("timestamp_close") else None,
                    json.dumps(trade.get("metadata", {})),
                ),
            )
        return tid

    def query_trades(
        self,
        strategy_id: Optional[str] = None,
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query trades with optional filters."""
        with self.transaction() as conn:
            parts = ["SELECT * FROM trades WHERE 1=1"]
            params = []
            if strategy_id:
                parts.append("AND strategy_id = ?")
                params.append(strategy_id)
            if since:
                parts.append("AND timestamp_open >= ?")
                params.append(since)
            parts.append("ORDER BY timestamp_open DESC LIMIT ?")
            params.append(limit)
            rows = conn.execute(" ".join(parts), params).fetchall()
            return [dict(r) for r in rows]

    # ── CRUD: Experiments ──

    def insert_experiment(self, experiment: dict) -> str:
        """Insert an experiment record."""
        import uuid
        eid = experiment.get("id", uuid.uuid4().hex)
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO experiments
                (id, strategy_id, strategy_type, params, metrics, regime,
                 research_window_end, created_at, status, verdict)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    eid,
                    experiment.get("strategy_id", ""),
                    experiment.get("strategy_type", ""),
                    json.dumps(experiment.get("params", {})),
                    json.dumps(experiment.get("metrics", {})),
                    experiment.get("regime", ""),
                    experiment.get("research_window_end", ""),
                    int(datetime.utcnow().timestamp()),
                    experiment.get("status", "pending"),
                    experiment.get("verdict", "discarded"),
                ),
            )
        return eid

    def query_experiments(
        self,
        strategy_type: Optional[str] = None,
        regime: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query experiments with optional filters."""
        with self.transaction() as conn:
            parts = ["SELECT * FROM experiments WHERE 1=1"]
            params = []
            if strategy_type:
                parts.append("AND strategy_type = ?")
                params.append(strategy_type)
            if regime:
                parts.append("AND regime = ?")
                params.append(regime)
            parts.append("ORDER BY created_at DESC LIMIT ?")
            params.append(limit)
            rows = conn.execute(" ".join(parts), params).fetchall()
            return [dict(r) for r in rows]

    # ── CRUD: OOS Results ──

    def insert_oos_result(self, result: dict) -> str:
        """Insert an OOS validation result."""
        import uuid
        rid = result.get("id", uuid.uuid4().hex)
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO oos_results
                (id, strategy_id, strategy_type, sharpe, net_sharpe,
                 win_rate, max_drawdown, trade_count, passed,
                 recommendation, validated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rid,
                    result.get("strategy_id", ""),
                    result.get("strategy_type", ""),
                    result.get("sharpe", 0),
                    result.get("net_sharpe", 0),
                    result.get("win_rate", 0),
                    result.get("max_drawdown", 0),
                    result.get("trade_count", 0),
                    1 if result.get("passed") else 0,
                    result.get("recommendation", ""),
                    int(datetime.utcnow().timestamp()),
                ),
            )
        return rid

    def query_oos_results(
        self,
        strategy_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query OOS results."""
        with self.transaction() as conn:
            if strategy_id:
                rows = conn.execute(
                    "SELECT * FROM oos_results WHERE strategy_id = ? ORDER BY validated_at DESC LIMIT ?",
                    (strategy_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM oos_results ORDER BY validated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ── CRUD: Pipeline Results ──

    def insert_pipeline_result(self, result: dict) -> str:
        """Insert a pipeline result."""
        import uuid
        pid = result.get("id", uuid.uuid4().hex)
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO pipeline_results
                (id, strategy_id, strategy_type, gate, passed, details, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    pid,
                    result.get("strategy_id", ""),
                    result.get("strategy_type", ""),
                    result.get("gate", ""),
                    1 if result.get("passed") else 0,
                    json.dumps(result.get("details", {})),
                    int(datetime.utcnow().timestamp()),
                ),
            )
        return pid

    def query_pipeline_results(
        self,
        strategy_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query pipeline results."""
        with self.transaction() as conn:
            if strategy_id:
                rows = conn.execute(
                    "SELECT * FROM pipeline_results WHERE strategy_id = ? ORDER BY created_at DESC LIMIT ?",
                    (strategy_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM pipeline_results ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ── CRUD: Validation Trades ──

    def insert_validation_trade(self, trade: dict) -> str:
        """Insert a validation trade record."""
        import uuid
        vid = trade.get("id", uuid.uuid4().hex)
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO validation_trades
                (id, strategy_id, pair, pnl, position_size, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    vid,
                    trade.get("strategy_id", ""),
                    trade.get("pair", ""),
                    trade.get("pnl"),
                    trade.get("position_size", 0),
                    int(trade.get("timestamp", datetime.utcnow().timestamp())),
                    json.dumps(trade.get("metadata", {})),
                ),
            )
        return vid

    def query_validation_trades(
        self,
        strategy_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query validation trades."""
        with self.transaction() as conn:
            if strategy_id:
                rows = conn.execute(
                    "SELECT * FROM validation_trades WHERE strategy_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (strategy_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM validation_trades ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ── Migration from JSONL ──

    def migrate_jsonl(self) -> int:
        """Migrate existing JSONL files into SQLite. Safe to run multiple times."""
        from config import settings

        migrated = 0
        migration_name = "jsonl_to_sqlite"

        # Check if already applied
        with self.transaction() as conn:
            already = conn.execute(
                "SELECT name FROM _migrations WHERE name = ?", (migration_name,)
            ).fetchone()
            if already:
                logger.info("Migration '%s' already applied, skipping", migration_name)
                return 0

        workspace = Path("./workspace")

        # Migrate trades from audit_log.jsonl
        audit_path = workspace / "audit_log.jsonl"
        if audit_path.exists():
            count = 0
            with open(audit_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        self.insert_trade({
                            "strategy_id": entry.get("strategy_name", entry.get("strategy_id", "unknown")),
                            "pair": entry.get("pair", ""),
                            "side": entry.get("side", ""),
                            "entry_price": entry.get("entry_price"),
                            "exit_price": entry.get("exit_price"),
                            "quantity": 0,
                            "pnl": entry.get("pnl_usdt", 0),
                            "pnl_pct": entry.get("pnl_pct", 0),
                            "fee_total": 0,
                            "regime": entry.get("regime", ""),
                            "timestamp_open": int(
                                datetime.fromisoformat(entry["timestamp"]).timestamp()
                                if isinstance(entry.get("timestamp"), str)
                                else entry.get("timestamp", 0)
                            ),
                            "metadata": {"status": entry.get("status", ""), "risk_verdict": entry.get("risk_verdict", "")},
                        })
                        count += 1
                    except Exception as exc:
                        logger.debug("Skipping audit entry during migration: %s", exc)
            logger.info("Migrated %d trades from audit_log.jsonl", count)
            migrated += count

        # Migrate experiments
        exp_path = workspace / "experiments.jsonl"
        if exp_path.exists():
            count = 0
            with open(exp_path) as f:
                for line in f:
                    try:
                        exp = json.loads(line.strip())
                        self.insert_experiment({
                            "strategy_id": f"{exp.get('strategy_type', 'unknown')}_{exp.get('iteration', 0)}",
                            "strategy_type": exp.get("strategy_type", ""),
                            "params": exp.get("params", {}),
                            "metrics": {
                                "sharpe": exp.get("sharpe", 0),
                                "win_rate": exp.get("win_rate", 0),
                                "max_drawdown": exp.get("max_drawdown", 0),
                                "total_trades": exp.get("total_trades", 0),
                                "profit_factor": exp.get("profit_factor", 1.0),
                            },
                            "regime": exp.get("regime", ""),
                            "status": "completed",
                            "verdict": exp.get("verdict", "discarded"),
                        })
                        count += 1
                    except Exception as exc:
                        logger.debug("Skipping experiment entry during migration: %s", exc)
            logger.info("Migrated %d experiments from experiments.jsonl", count)
            migrated += count

        # Migrate pipeline results
        pipe_path = workspace / "pipeline_results.jsonl"
        if pipe_path.exists():
            count = 0
            with open(pipe_path) as f:
                for line in f:
                    try:
                        pr = json.loads(line.strip())
                        self.insert_pipeline_result({
                            "strategy_id": pr.get("strategy_id", ""),
                            "strategy_type": pr.get("strategy_type", ""),
                            "gate": pr.get("failed_at_name", str(pr.get("passed_gates", 0))),
                            "passed": pr.get("failed_at") is None,
                            "details": {
                                "passed_gates": pr.get("passed_gates", 0),
                                "total_gates": pr.get("total_gates", 11),
                                "reason": pr.get("reason", ""),
                                "oos_passed": pr.get("oos_passed"),
                            },
                        })
                        count += 1
                    except Exception as exc:
                        logger.debug("Skipping pipeline entry during migration: %s", exc)
            logger.info("Migrated %d pipeline results from pipeline_results.jsonl", count)
            migrated += count

        # Migrate OOS results
        oos_path = workspace / "oos_results.jsonl"
        if oos_path.exists():
            count = 0
            with open(oos_path) as f:
                for line in f:
                    try:
                        oos = json.loads(line.strip())
                        self.insert_oos_result({
                            "strategy_id": oos.get("strategy_id", ""),
                            "strategy_type": oos.get("strategy_type", ""),
                            "sharpe": oos.get("oos_sharpe", 0),
                            "net_sharpe": oos.get("net_sharpe", 0),
                            "win_rate": oos.get("oos_win_rate", 0),
                            "max_drawdown": oos.get("oos_max_drawdown", 0),
                            "trade_count": oos.get("oos_trades", 0),
                            "passed": oos.get("passed", False),
                            "recommendation": oos.get("recommendation", ""),
                        })
                        count += 1
                    except Exception as exc:
                        logger.debug("Skipping OOS entry during migration: %s", exc)
            logger.info("Migrated %d OOS results from oos_results.jsonl", count)
            migrated += count

        # Migrate validation trades
        val_path = workspace / "validation_trades.jsonl"
        if val_path.exists():
            count = 0
            with open(val_path) as f:
                for line in f:
                    try:
                        vt = json.loads(line.strip())
                        self.insert_validation_trade({
                            "strategy_id": vt.get("strategy_id", "unknown"),
                            "pair": vt.get("pair", ""),
                            "pnl": vt.get("pnl", vt.get("pnl_pct", 0)),
                            "position_size": vt.get("position_size", 0),
                            "timestamp": int(
                                datetime.fromisoformat(vt["logged_at"]).timestamp()
                                if isinstance(vt.get("logged_at"), str)
                                else datetime.utcnow().timestamp()
                            ),
                            "metadata": vt,
                        })
                        count += 1
                    except Exception as exc:
                        logger.debug("Skipping validation trade during migration: %s", exc)
            logger.info("Migrated %d validation trades from validation_trades.jsonl", count)
            migrated += count

        # Mark migration as applied
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO _migrations (name, applied_at) VALUES (?, ?)",
                (migration_name, int(datetime.utcnow().timestamp())),
            )

        logger.info("JSONL migration complete: %d records migrated", migrated)
        return migrated

    # ── Utility ──

    def integrity_check(self) -> str:
        """Run PRAGMA integrity_check."""
        with self.transaction() as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            return str(result[0])

    @property
    def is_wal_mode(self) -> bool:
        """Check if WAL mode is active."""
        with self.transaction() as conn:
            result = conn.execute("PRAGMA journal_mode").fetchone()
            return result[0].upper() == "WAL"

    def table_count(self, table_name: str) -> int:
        """Count rows in a table. Raises ValueError for invalid table names."""
        if table_name not in self.VALID_TABLES:
            raise ValueError(f"Invalid table name: {table_name!r}")
        with self.transaction() as conn:
            result = conn.execute(
                f"SELECT COUNT(*) FROM [{table_name}]"
            ).fetchone()
            return result[0]

    def clear_all(self) -> None:
        """Clear all data (for testing)."""
        with self.transaction() as conn:
            for table in self.VALID_TABLES - {"_migrations"}:
                conn.execute(f"DELETE FROM [{table}]")

    # ── API Cache ──

    def get_cached(self, cache_key: str) -> Optional[dict]:
        """Return cached data dict, or None if missing/expired."""
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT data, cached_at, ttl_seconds FROM api_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        if time.time() - row["cached_at"] > row["ttl_seconds"]:
            with self.transaction() as conn:
                conn.execute(
                    "DELETE FROM api_cache WHERE cache_key = ?", (cache_key,)
                )
            return None
        return json.loads(row["data"])

    def set_cached(
        self, cache_key: str, data: dict, source: str, ttl_seconds: int
    ) -> None:
        """Insert or replace a cached entry."""
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO api_cache
                   (cache_key, data, source, cached_at, ttl_seconds)
                   VALUES (?, ?, ?, ?, ?)""",
                (cache_key, json.dumps(data), source, int(time.time()), ttl_seconds),
            )
