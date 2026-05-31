"""Tests for walk_forward_validate in backtesting/engine.py"""
from unittest.mock import patch
from backtesting.engine import BacktestEngine


def _mock_backtest_result(sharpe=0.8, win_rate=0.45, trades=8):
    return {
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "max_drawdown": 0.04,
        "total_trades": trades,
    }


def test_walk_forward_returns_correct_structure():
    engine = BacktestEngine()
    with patch.object(engine, "run_backtest", return_value=_mock_backtest_result()):
        result = engine.walk_forward_validate(
            strategy_params={"fast_ma": 10, "slow_ma": 30},
            strategy_type="sma_crossover",
            start_date="20260427",
            end_date="20260527",
            windows=3,
        )
    assert "windows" in result
    assert "consistency_score" in result
    assert "avg_sharpe" in result
    assert "is_robust" in result


def test_walk_forward_consistency_all_pass():
    engine = BacktestEngine()
    with patch.object(engine, "run_backtest", return_value=_mock_backtest_result(sharpe=1.0, trades=10)):
        result = engine.walk_forward_validate(
            strategy_params={},
            start_date="20260427",
            end_date="20260527",
            windows=3,
        )
    assert result["consistency_score"] == 1.0
    assert result["is_robust"] is True


def test_walk_forward_consistency_all_fail():
    engine = BacktestEngine()
    with patch.object(engine, "run_backtest", return_value=_mock_backtest_result(sharpe=-0.5, trades=2)):
        result = engine.walk_forward_validate(
            strategy_params={},
            start_date="20260427",
            end_date="20260527",
            windows=3,
        )
    assert result["consistency_score"] == 0.0
    assert result["is_robust"] is False


def test_walk_forward_rejects_tiny_window():
    engine = BacktestEngine()
    result = engine.walk_forward_validate(
        strategy_params={},
        start_date="20260527",
        end_date="20260530",  # Only 3 days
        windows=3,
    )
    assert "error" in result


if __name__ == "__main__":
    test_walk_forward_returns_correct_structure()
    test_walk_forward_consistency_all_pass()
    test_walk_forward_consistency_all_fail()
    test_walk_forward_rejects_tiny_window()
    print("All walk_forward tests passed.")
