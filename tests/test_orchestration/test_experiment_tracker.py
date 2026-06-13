"""Tests for orchestration/experiment_tracker.py"""
import os
import tempfile
from orchestration.experiment_tracker import ExperimentTracker, Experiment


def _make_exp(strategy_type="sma_crossover", sharpe=0.5, trades=10,
              verdict="kept", regime="ranging", **kwargs):
    return Experiment(
        strategy_type=strategy_type,
        params={"fast_ma": 10, "slow_ma": 30, **kwargs},
        timerange="20260427-20260530",
        regime=regime,
        sentiment_score=0.1,
        sharpe=sharpe,
        win_rate=0.45,
        max_drawdown=0.05,
        total_trades=trades,
        verdict=verdict,
        iteration=1,
    )


def test_record_and_persist():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        t = ExperimentTracker(path)
        t.record(_make_exp())
        # Reload from disk
        t2 = ExperimentTracker(path)
        assert len(t2._experiments) == 1
    finally:
        os.unlink(path)


def test_get_best_filters_by_trades():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        t = ExperimentTracker(path)
        t.record(_make_exp(sharpe=2.0, trades=3))  # Too few trades
        t.record(_make_exp(sharpe=0.8, trades=10))  # Valid
        best = t.get_best(k=5)
        assert len(best) == 1
        assert best[0].sharpe == 0.8
    finally:
        os.unlink(path)


def test_get_best_filters_by_regime():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        t = ExperimentTracker(path)
        t.record(_make_exp(sharpe=1.5, regime="trending"))
        t.record(_make_exp(sharpe=0.9, regime="ranging"))
        ranging_best = t.get_best(regime="ranging", k=5)
        assert len(ranging_best) == 1
        assert ranging_best[0].sharpe == 0.9
    finally:
        os.unlink(path)


def test_suggest_next_params_enforces_fast_slow_order():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        t = ExperimentTracker(path)
        t.record(_make_exp(fast_ma=20, slow_ma=50, sharpe=1.2))
        result = t.suggest_next_params("sma_crossover", {"fast_ma": 10, "slow_ma": 30})
        assert result["fast_ma"] < result["slow_ma"], \
            f"fast_ma={result['fast_ma']} should be < slow_ma={result['slow_ma']}"
    finally:
        os.unlink(path)


def test_suggest_next_params_stays_in_range():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        t = ExperimentTracker(path)
        t.record(_make_exp(fast_ma=3, slow_ma=10, sharpe=1.0))
        result = t.suggest_next_params("sma_crossover", {"fast_ma": 40, "slow_ma": 180})
        assert 3 <= result["fast_ma"] <= 50
        assert 10 <= result["slow_ma"] <= 200
    finally:
        os.unlink(path)


def test_summary_format():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        t = ExperimentTracker(path)
        t.record(_make_exp(verdict="kept"))
        t.record(_make_exp(verdict="discarded", sharpe=0.1))
        summary = t.summary()
        assert "Total experiments: 2" in summary
        assert "Kept: 1" in summary
        assert "Discarded: 1" in summary
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_record_and_persist()
    test_get_best_filters_by_trades()
    test_get_best_filters_by_regime()
    test_suggest_next_params_enforces_fast_slow_order()
    test_suggest_next_params_stays_in_range()
    test_summary_format()
    print("All ExperimentTracker tests passed.")
