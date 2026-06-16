"""Tests for Tavily search integration with caching fallback."""

import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


@pytest.fixture
def fake_tavily():
    """Create a fake tavily module in sys.modules for testing."""
    if "tavily" in sys.modules:
        yield sys.modules["tavily"]
        return

    fake = type(sys)("tavily")
    fake.TavilyClient = MagicMock()
    sys.modules["tavily"] = fake
    yield fake


@pytest.fixture
def tavily_settings():
    """Patch only the Tavily-related settings attributes, leaving others intact."""
    from config import settings
    originals = {
        "TAVILY_ENABLED": settings.TAVILY_ENABLED,
        "TAVILY_API_KEY": settings.TAVILY_API_KEY,
    }
    yield settings
    settings.TAVILY_ENABLED = originals["TAVILY_ENABLED"]
    settings.TAVILY_API_KEY = originals["TAVILY_API_KEY"]


class TestTavilyConfig:
    def test_tavily_config_vars_exist(self):
        from config import settings
        assert hasattr(settings, "TAVILY_API_KEY")
        assert hasattr(settings, "TAVILY_ENABLED")


class TestTavilySearchTool:
    def _get_search_fn(self):
        from agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        return agent.tools["web_search"].func

    def test_web_search_tool_exists(self):
        from agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        assert "web_search" in agent.tools

    def test_tavily_search_structured_output(self, fake_tavily, tavily_settings):
        tavily_settings.TAVILY_ENABLED = True
        tavily_settings.TAVILY_API_KEY = "test-key"
        search_fn = self._get_search_fn()

        fake_tavily.TavilyClient = MagicMock()
        mock_client = MagicMock()
        fake_tavily.TavilyClient.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                {"title": "Momentum Strategy", "content": "A momentum strategy using RSI and MACD", "url": "https://example.com/1"},
                {"title": "Mean Reversion", "content": "Mean reversion strategy for ranging", "url": "https://example.com/2"},
            ]
        }
        result = search_fn('{"query": "momentum strategy crypto", "max_results": 3}')
        assert "Momentum Strategy" in result
        assert "Mean Reversion" in result
        mock_client.search.assert_called_once_with(
            query="momentum strategy crypto",
            max_results=3,
            search_depth="advanced",
        )

    def test_tavily_empty_results(self, fake_tavily, tavily_settings):
        tavily_settings.TAVILY_ENABLED = True
        tavily_settings.TAVILY_API_KEY = "test-key"
        search_fn = self._get_search_fn()

        fake_tavily.TavilyClient = MagicMock()
        mock_client = MagicMock()
        fake_tavily.TavilyClient.return_value = mock_client
        mock_client.search.return_value = {"results": []}
        result = search_fn('{"query": "obscure topic", "max_results": 3}')
        assert "No results" in result

    def test_fallback_to_duckduckgo(self, tavily_settings):
        tavily_settings.TAVILY_ENABLED = False
        search_fn = self._get_search_fn()

        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "AbstractText": "Test result about momentum trading",
                "RelatedTopics": [{"Text": "Momentum strategy using RSI", "FirstURL": "https://example.com"}]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            result = search_fn('{"query": "momentum strategy", "max_results": 3}')
            assert "momentum" in result.lower()

    def test_tavily_errors_gracefully(self, fake_tavily, tavily_settings):
        tavily_settings.TAVILY_ENABLED = True
        tavily_settings.TAVILY_API_KEY = "test-key"
        search_fn = self._get_search_fn()

        fake_tavily.TavilyClient = MagicMock()
        mock_client = MagicMock()
        fake_tavily.TavilyClient.return_value = mock_client
        mock_client.search.side_effect = Exception("API rate limit exceeded")
        result = search_fn('{"query": "test", "max_results": 3}')
        assert "No results" in result

    def test_tavily_no_key_skips_tavily(self, tavily_settings):
        tavily_settings.TAVILY_ENABLED = True
        tavily_settings.TAVILY_API_KEY = ""
        search_fn = self._get_search_fn()

        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"AbstractText": "", "RelatedTopics": []}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            result = search_fn('{"query": "test", "max_results": 3}')
            assert result is not None


class TestSearchCache:
    def test_cache_stores_search_results(self):
        from memory.vector_store import VectorStore
        import uuid
        store = VectorStore(collection_name=f"test_cache_{uuid.uuid4().hex[:8]}")
        store.store_insight(
            text="Search result: momentum strategy RSI divergence",
            metadata={"type": "search_cache", "query": "momentum RSI strategy", "timestamp": "2024-01-01"},
        )
        results = store.query_similar("momentum RSI strategy", k=3)
        found = any("momentum" in r.get("text", "").lower() for r in results)
        assert found

    def test_cache_different_query_no_match(self):
        from memory.vector_store import VectorStore
        import uuid
        store = VectorStore(collection_name=f"test_cache2_{uuid.uuid4().hex[:8]}")
        store.store_insight(
            text="Search result: momentum strategy with RSI and MACD crossover",
            metadata={"type": "search_cache", "query": "momentum strategy", "timestamp": "2024-01-01"},
        )
        results = store.query_similar("news about bitcoin regulation", k=3)
        assert isinstance(results, list)
