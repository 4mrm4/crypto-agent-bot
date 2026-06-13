"""Tests for data/strategy_concepts.py"""

from data.strategy_concepts import (
    STRATEGY_CONCEPTS,
    get_concepts_for_regime,
    get_concept_by_name,
    get_all_concept_names,
)

VALID_REGIMES = {"strong_uptrend", "strong_downtrend", "ranging", "volatile", "weak_trend"}
FREQTRADE_TYPES = {"sma_crossover", "rsi_oversold", "breakout", "macd_crossover",
                   "volatility_squeeze", "sentiment_driven", "multi_timeframe",
                   "mean_reversion", "momentum"}
VALID_CATEGORIES = {"trend_following", "mean_reversion", "breakout", "momentum",
                    "volatility", "sentiment"}


class TestStrategyConceptsData:
    def test_has_ten_concepts(self):
        assert len(STRATEGY_CONCEPTS) == 10

    def test_all_concepts_have_required_fields(self):
        required = {"name", "category", "description", "indicators", "entry",
                    "exit", "best_regime", "freqtrade_type", "suggested_params"}
        for c in STRATEGY_CONCEPTS:
            missing = required - set(c.keys())
            assert not missing, f"Concept '{c.get('name')}' missing: {missing}"

    def test_all_names_unique(self):
        names = [c["name"] for c in STRATEGY_CONCEPTS]
        assert len(names) == len(set(names))

    def test_all_best_regimes_valid(self):
        for c in STRATEGY_CONCEPTS:
            assert c["best_regime"] in VALID_REGIMES, \
                f"{c['name']} has invalid regime: {c['best_regime']}"

    def test_all_freqtrade_types_valid(self):
        for c in STRATEGY_CONCEPTS:
            assert c["freqtrade_type"] in FREQTRADE_TYPES

    def test_all_categories_valid(self):
        for c in STRATEGY_CONCEPTS:
            assert c["category"] in VALID_CATEGORIES

    def test_suggested_params_is_dict(self):
        for c in STRATEGY_CONCEPTS:
            assert isinstance(c["suggested_params"], dict)

    def test_all_names_are_strings(self):
        for c in STRATEGY_CONCEPTS:
            assert isinstance(c["name"], str) and len(c["name"]) > 0

    def test_all_descriptions_nonempty(self):
        for c in STRATEGY_CONCEPTS:
            assert len(c["description"]) > 10

    def test_indicators_is_list(self):
        for c in STRATEGY_CONCEPTS:
            assert isinstance(c["indicators"], list)
            assert len(c["indicators"]) > 0

    def test_golden_cross_params(self):
        gc = get_concept_by_name("Golden Cross")
        assert gc["suggested_params"] == {"fast_ma": 50, "slow_ma": 200}

    def test_rsi_divergence_best_regime(self):
        rd = get_concept_by_name("RSI Divergence")
        assert rd["best_regime"] == "ranging"

    def test_momentum_with_volume_category(self):
        mv = get_concept_by_name("Momentum with Volume")
        assert mv["category"] == "momentum"


class TestGetConceptsForRegime:
    def test_uptrend_returns_concepts(self):
        concepts = get_concepts_for_regime("strong_uptrend")
        assert len(concepts) >= 2
        for c in concepts:
            assert c["best_regime"] == "strong_uptrend"

    def test_ranging_returns_concepts(self):
        concepts = get_concepts_for_regime("ranging")
        assert len(concepts) >= 2
        for c in concepts:
            assert c["best_regime"] == "ranging"

    def test_volatile_returns_concepts(self):
        concepts = get_concepts_for_regime("volatile")
        assert len(concepts) >= 2

    def test_unknown_regime_returns_empty(self):
        concepts = get_concepts_for_regime("nonexistent_regime")
        assert concepts == []


class TestGetConceptByName:
    def test_exact_match(self):
        c = get_concept_by_name("Golden Cross")
        assert c["name"] == "Golden Cross"
        assert c["freqtrade_type"] == "sma_crossover"

    def test_case_insensitive(self):
        c = get_concept_by_name("golden cross")
        assert c["name"] == "Golden Cross"

    def test_nonexistent_returns_empty(self):
        c = get_concept_by_name("nonexistent")
        assert c == {}

    def test_empty_string(self):
        c = get_concept_by_name("")
        assert c == {}


class TestGetAllConceptNames:
    def test_returns_ten_names(self):
        names = get_all_concept_names()
        assert len(names) == 10

    def test_returns_strings(self):
        names = get_all_concept_names()
        assert all(isinstance(n, str) for n in names)

    def test_includes_golden_cross(self):
        names = get_all_concept_names()
        assert "Golden Cross" in names

    def test_no_duplicates(self):
        names = get_all_concept_names()
        assert len(names) == len(set(names))
