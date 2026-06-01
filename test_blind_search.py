"""Tests for BlindParameterSearch."""

import json

from backtesting.blind_search import BlindParameterSearch


class TestBlindSearchDefaults:
    def test_generate_default_variants(self):
        bps = BlindParameterSearch()
        variants = bps._generate_default_variants("sma_crossover", n=10)
        assert len(variants) == 10
        for v in variants:
            assert isinstance(v, dict)

    def test_compute_aggregate_stats_empty(self):
        bps = BlindParameterSearch()
        stats = bps.compute_aggregate_stats([])
        assert stats["n_total"] == 0
        assert stats["median_sharpe"] == 0.0

    def test_compute_aggregate_stats_single(self):
        bps = BlindParameterSearch()
        results = [{"sharpe_ratio": 1.5, "win_rate": 0.5, "max_drawdown": 0.1,
                     "total_trades": 30, "params": {"fast_ma": 10}}]
        stats = bps.compute_aggregate_stats(results)
        assert stats["n_valid"] == 1

    def test_select_best_for_wfv_empty(self):
        bps = BlindParameterSearch()
        best = bps.select_best_for_wfv([])
        assert best is None

    def test_select_best_for_wfv_single(self):
        bps = BlindParameterSearch()
        results = [{"sharpe_ratio": 1.5, "win_rate": 0.5, "max_drawdown": 0.1,
                     "total_trades": 30, "params": {"fast_ma": 10}}]
        best = bps.select_best_for_wfv(results)
        assert best is not None
        assert "params" in best
        assert "metrics" in best

    def test_select_best_for_wfv_picks_highest(self):
        bps = BlindParameterSearch()
        results = [
            {"sharpe_ratio": 0.5, "win_rate": 0.4, "max_drawdown": 0.1,
             "total_trades": 30, "params": {"fast_ma": 5}},
            {"sharpe_ratio": 2.0, "win_rate": 0.6, "max_drawdown": 0.05,
             "total_trades": 50, "params": {"fast_ma": 10}},
            {"sharpe_ratio": 1.0, "win_rate": 0.3, "max_drawdown": 0.2,
             "total_trades": 20, "params": {"fast_ma": 20}},
        ]
        best = bps.select_best_for_wfv(results)
        assert best["params"]["fast_ma"] == 10

    def test_aggregate_stats_includes_range(self):
        bps = BlindParameterSearch()
        results = [
            {"sharpe_ratio": 1.0, "win_rate": 0.5, "max_drawdown": 0.1,
             "total_trades": 30, "params": {"fast_ma": 10}},
            {"sharpe_ratio": 2.0, "win_rate": 0.6, "max_drawdown": 0.05,
             "total_trades": 40, "params": {"fast_ma": 20}},
        ]
        stats = bps.compute_aggregate_stats(results)
        assert "sharpe_range" in stats
        assert len(stats["sharpe_range"]) == 2
