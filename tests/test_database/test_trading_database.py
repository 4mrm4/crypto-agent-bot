"""Tests for TradingDatabase SQL injection fixes and table_count/clear_all."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.database import TradingDatabase


# ── Fixtures ──


@pytest.fixture
def db(tmp_path):
    """Return a TradingDatabase instance backed by a temp file (avoids :memory: singleton issues)."""
    db_path = str(tmp_path / "test.db")
    return TradingDatabase(db_path=db_path)


# ── table_count ──


class TestTableCount:
    """table_count() validates table names and returns correct counts."""

    def test_valid_table_returns_count(self, db):
        """A valid table name returns an integer row count."""
        count = db.table_count("trades")
        assert isinstance(count, int)
        assert count >= 0

    def test_invalid_table_raises_value_error(self, db):
        """An invalid table name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid table name"):
            db.table_count("nonexistent")

    def test_sql_injection_raises_value_error(self, db):
        """A SQL injection string is rejected before reaching the database."""
        with pytest.raises(ValueError, match="Invalid table name"):
            db.table_count("trades; DROP TABLE trades; --")

    def test_sql_injection_union_raises_value_error(self, db):
        """A UNION-based SQL injection string is rejected."""
        with pytest.raises(ValueError, match="Invalid table name"):
            db.table_count(
                "trades UNION SELECT sql FROM sqlite_master --"
            )

    def test_all_valid_tables_work(self, db):
        """Every table in VALID_TABLES returns a count without error."""
        for table in TradingDatabase.VALID_TABLES:
            count = db.table_count(table)
            assert isinstance(count, int)

    def test_multiple_calls_consistent(self, db):
        """Calling table_count twice on the same table returns the same value."""
        assert db.table_count("trades") == db.table_count("trades")


# ── clear_all ──


class TestClearAll:
    """clear_all() deletes rows from all tables except _migrations."""

    def test_clear_all_does_not_raise(self, db):
        """clear_all() runs without error on a fresh database."""
        db.clear_all()  # should not raise

    def test_clear_all_resets_counts(self, db):
        """After clear_all, all tables have 0 rows."""
        db.clear_all()
        for table in TradingDatabase.VALID_TABLES:
            assert db.table_count(table) == 0, f"{table} not empty after clear_all"

    def test_clear_all_skips_migrations(self, db):
        """_migrations table is not cleared by clear_all."""
        count_before = db.table_count("_migrations")
        db.clear_all()
        count_after = db.table_count("_migrations")
        assert count_after >= count_before  # _migrations was not cleared

    def test_injection_in_table_name_blocked(self, db):
        """SQL injection in table name is blocked by VALID_TABLES guard."""
        # This tests that table_count can't be injected; clear_all
        # uses VALID_TABLES so it's inherently safe.
        with pytest.raises(ValueError, match="Invalid table name"):
            db.table_count("api_cache; DELETE FROM trades; --")
