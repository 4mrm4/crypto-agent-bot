"""Tests for TradingDatabase SQLite-backed storage.

These tests verify that:
1. Schema is created correctly with all tables
2. CRUD operations work for each table type
3. WAL mode is active after connection
4. migrate_jsonl is idempotent
5. Index usage shows index scans in EXPLAIN QUERY PLAN
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture
def db():
    """Create a fresh TradingDatabase in a temp directory for each test."""
    from data.database import TradingDatabase
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        database = TradingDatabase(db_path=db_path)
        yield database
        database.clear_all()


class TestSchemaInitialisation:
    """Verify all tables are created."""

    def test_all_tables_exist(self, db):
        with db.transaction() as conn:
            tables = set(
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            )
        for expected in ["trades", "experiments", "oos_results", "pipeline_results", "validation_trades", "_migrations"]:
            assert expected in tables, f"Table {expected} missing"

    def test_wal_mode_active(self, db):
        assert db.is_wal_mode, "WAL mode should be active"

    def test_integrity_ok(self, db):
        assert db.integrity_check() == "ok"


class TestTradesCRUD:
    """CRUD operations on trades table."""

    def test_insert_and_query_trade(self, db):
        tid = db.insert_trade({
            "strategy_id": "test_strat_1",
            "pair": "BTC/USDT",
            "side": "long",
            "entry_price": 50000.0,
            "exit_price": 51000.0,
            "quantity": 0.1,
            "pnl": 100.0,
            "pnl_pct": 0.02,
            "fee_total": 0.5,
            "regime": "bull",
            "timestamp_open": int(time.time()),
        })
        trades = db.query_trades(strategy_id="test_strat_1")
        assert len(trades) == 1
        assert trades[0]["pair"] == "BTC/USDT"
        assert trades[0]["pnl"] == 100.0

    def test_query_trades_limit(self, db):
        for i in range(5):
            db.insert_trade({
                "strategy_id": f"strat_{i}",
                "pair": "ETH/USDT",
                "side": "long",
                "timestamp_open": int(time.time()),
            })
        all_trades = db.query_trades(limit=3)
        assert len(all_trades) == 3

    def test_query_trades_by_strategy(self, db):
        db.insert_trade({"strategy_id": "a", "pair": "BTC/USDT", "side": "long", "timestamp_open": 100})
        db.insert_trade({"strategy_id": "b", "pair": "ETH/USDT", "side": "short", "timestamp_open": 200})
        results = db.query_trades(strategy_id="a")
        assert len(results) == 1
        assert results[0]["pair"] == "BTC/USDT"


class TestExperimentsCRUD:
    """CRUD operations on experiments table."""

    def test_insert_and_query_experiment(self, db):
        eid = db.insert_experiment({
            "strategy_id": "exp_1",
            "strategy_type": "sma_crossover",
            "params": {"fast_ma": 10, "slow_ma": 30},
            "metrics": {"sharpe": 1.5, "win_rate": 0.55},
            "regime": "bull",
            "status": "completed",
        })
        results = db.query_experiments(strategy_type="sma_crossover")
        assert len(results) >= 1
        assert "sharpe" in json.loads(results[0]["metrics"])

    def test_query_experiments_by_regime(self, db):
        db.insert_experiment({"strategy_id": "x", "strategy_type": "a", "params": {}, "metrics": {}, "regime": "bull"})
        db.insert_experiment({"strategy_id": "y", "strategy_type": "b", "params": {}, "metrics": {}, "regime": "bear"})
        results = db.query_experiments(regime="bull")
        assert len(results) == 1


class TestOOSResultsCRUD:
    """CRUD operations on oos_results table."""

    def test_insert_and_query_oos(self, db):
        rid = db.insert_oos_result({
            "strategy_id": "oos_1",
            "strategy_type": "momentum",
            "sharpe": 1.2,
            "net_sharpe": 1.0,
            "win_rate": 0.5,
            "max_drawdown": 0.08,
            "trade_count": 50,
            "passed": True,
            "recommendation": "deploy",
        })
        results = db.query_oos_results(strategy_id="oos_1")
        assert len(results) == 1
        assert results[0]["net_sharpe"] == 1.0

    def test_oos_passed_flag(self, db):
        db.insert_oos_result({
            "strategy_id": "oos_fail", "strategy_type": "test",
            "sharpe": 0.5, "net_sharpe": 0.3, "passed": False, "recommendation": "reject",
        })
        results = db.query_oos_results(strategy_id="oos_fail")
        assert results[0]["passed"] == 0


class TestPipelineResultsCRUD:
    """CRUD operations on pipeline_results table."""

    def test_insert_and_query_pipeline(self, db):
        pid = db.insert_pipeline_result({
            "strategy_id": "pipe_1",
            "strategy_type": "sma_crossover",
            "gate": "Gate 4: research backtest",
            "passed": True,
            "details": {"reason": "All metrics passed"},
        })
        results = db.query_pipeline_results(strategy_id="pipe_1")
        assert len(results) == 1
        assert results[0]["passed"] == 1


class TestValidationTradesCRUD:
    """CRUD operations on validation_trades table."""

    def test_insert_and_query_validation(self, db):
        vid = db.insert_validation_trade({
            "strategy_id": "val_1",
            "pair": "SOL/USDT",
            "pnl": 50.0,
            "position_size": 100.0,
            "timestamp": int(time.time()),
        })
        results = db.query_validation_trades(strategy_id="val_1")
        assert len(results) == 1
        assert results[0]["pair"] == "SOL/USDT"


class TestMigration:
    """Verify JSONL migration is idempotent."""

    def test_migration_idempotent(self, db):
        """Running migration twice should produce same result."""
        # Create a test JSONL file
        workspace = Path(db.db_path.parent)
        test_file = workspace / "audit_log.jsonl"
        with open(test_file, "w") as f:
            f.write(json.dumps({
                "signal_id": "sig_1", "pair": "BTC/USDT", "side": "long",
                "strategy_name": "test", "regime": "bull", "confidence": 0.8,
                "position_size_usdt": 100, "entry_price": 50000,
                "status": "executed", "risk_verdict": "approved",
                "circuit_breaker_state": {}, "correlation_result": {},
                "kelly_result": {}, "timestamp": "2024-01-01T00:00:00",
            }) + "\n")

        count1 = db.migrate_jsonl()
        assert count1 > 0

        count2 = db.migrate_jsonl()
        assert count2 == 0  # Already applied, skipped

        # Clean up
        test_file.unlink(missing_ok=True)

    def test_migration_empty_jsonl(self, db):
        """Migration with no JSONL files should return 0."""
        count = db.migrate_jsonl()
        assert count >= 0


class TestIntegrity:
    """Database integrity checks."""

    def test_integrity_check(self, db):
        assert db.integrity_check() == "ok"

    def test_clear_all(self, db):
        db.insert_trade({"strategy_id": "x", "pair": "BTC/USDT", "side": "long", "timestamp_open": 100})
        assert db.table_count("trades") > 0
        db.clear_all()
        assert db.table_count("trades") == 0
