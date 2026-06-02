"""Researcher agent — web search, paper reading, and strategy spec generation."""

import json
import logging
from typing import Any, Dict

from agents.base import BaseAgent
from config import settings
from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are a quantitative strategy researcher. Your job is to:

1. Search the web for novel trading strategy ideas based on the research goal
2. Read papers or articles to extract structured strategy specifications
3. Generate fully-specified custom strategy code compatible with the existing freqtrade backtester

Your search focus: technical indicators, entry/exit logic, risk management, and timeframe selection for crypto strategies. Prioritise strategies that are testable with common indicators (SMA, EMA, MACD, RSI, Bollinger Bands, Ichimoku, Volume Profile, etc.).

When generating strategy specs, output valid pandas/ta-lib Python expressions that will be injected into a freqtrade strategy template. The template expects:
- indicator_code: lines that add columns to the dataframe (e.g. "dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=10)")
- entry_condition: a pandas boolean expression (e.g. "(dataframe['ema_fast'] > dataframe['ema_slow'])")
- exit_condition: a pandas boolean expression (e.g. "(dataframe['ema_fast'] < dataframe['ema_slow'])")

Workflow:
  web_search -> extract_ideas -> read_paper (if URLs found) -> generate_custom_strategy_spec
  web_search -> if no papers, use generate_custom_strategy_spec directly from search findings
"""


class ResearcherAgent(BaseAgent):
    """Agent that researches trading strategies via web search and paper reading."""

    def __init__(self):
        tools = self._build_tools()
        super().__init__(
            name="researcher",
            tools=tools,
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
        )
        self._generated_specs: Dict[str, Dict[str, Any]] = {}
        self._specs: Dict[str, Dict[str, Any]] = {}

    def _build_tools(self):
        def web_search(query_json: str = "{}") -> str:
            """Search the web for trading strategy ideas, papers, or articles.
            Pass JSON: {"query": "momentum strategy crypto Bollinger Bands", "max_results": 5}
            Returns a markdown list of results with titles, snippets, and URLs.
            Uses Tavily search when available, falls back to DuckDuckGo."""
            import json
            try:
                params = json.loads(query_json)
            except json.JSONDecodeError:
                params = {"query": query_json}
            query = params.get("query", "")
            max_results = int(params.get("max_results", 5))
            if not query:
                return "Error: empty query"

            # ── Try Tavily first ──
            if settings.TAVILY_ENABLED and settings.TAVILY_API_KEY:
                try:
                    from tavily import TavilyClient
                    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
                    resp = client.search(
                        query=query,
                        max_results=max_results,
                        search_depth="advanced",
                    )
                    results = resp.get("results", [])
                    if not results:
                        return "No results found from Tavily search."

                    lines = [f"Tavily search results for '{query}':"]
                    for r in results[:max_results]:
                        title = r.get("title", "")
                        content = r.get("content", "")[:200]
                        url = r.get("url", "")
                        lines.append(f"\n**{title}**")
                        lines.append(f"  {content}")
                        if url:
                            lines.append(f"  ([link]({url}))")

                    # Strategy relevance scoring
                    strategy_keywords = [
                        "strategy", "indicator", "entry", "exit", "crossover",
                        "RSI", "MACD", "SMA", "EMA", "breakout", "backtest",
                        "win rate", "sharpe", "drawdown"
                    ]
                    relevant = []
                    for r in results[:max_results]:
                        text = (r.get("content", "") + " " + r.get("title", "")).lower()
                        score = sum(1 for kw in strategy_keywords if kw.lower() in text)
                        if score >= 2:
                            relevant.append({
                                "title": r.get("title", ""),
                                "snippet": r.get("content", ""),
                                "relevance_score": score,
                            })
                    relevant.sort(key=lambda x: x["relevance_score"], reverse=True)
                    if relevant:
                        lines.append(f"\nFound {len(relevant)} strategy-relevant results:")
                        for r in relevant[:5]:
                            lines.append(f"\n[Score={r['relevance_score']}] {r['title']}")
                            lines.append(f"  {r['snippet'][:200]}")
                        lines.append(
                            "\nNext step: Use generate_custom_strategy_spec to convert "
                            "the most relevant result into a testable strategy spec."
                        )

                    # Cache results in ChromaDB (best-effort)
                    try:
                        from memory.vector_store import VectorStore
                        cache_store = VectorStore(collection_name="search_cache")
                        for r in results[:max_results]:
                            text = f"Search result: {r.get('title', '')} - {r.get('content', '')[:300]}"
                            cache_store.store_insight(
                                text=text,
                                metadata={
                                    "type": "search_cache",
                                    "query": query,
                                    "url": r.get("url", ""),
                                },
                            )
                    except Exception:
                        pass  # Cache is optional

                    return "\n".join(lines)
                except ImportError:
                    pass  # Tavily not installed, fall through to DuckDuckGo
                except Exception as exc:
                    return f"Tavily search error: {exc}"

            # ── Fallback: DuckDuckGo ──
            try:
                import httpx
                url = "https://api.duckduckgo.com/"
                resp = httpx.get(
                    url,
                    params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()

                results = []
                abstract = data.get("AbstractText", "")
                if abstract:
                    source = data.get("Source", "web")
                    results.append(f"- **{source}**: {abstract[:300]}")

                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if "Text" in topic:
                        text = topic["Text"][:200]
                        url = topic.get("FirstURL", "")
                        results.append(f"- {text} ([link]({url}))")

                if not results:
                    return "No results found for that query."

                # Strategy relevance scoring
                try:
                    strategy_keywords = [
                        "strategy", "indicator", "entry", "exit", "crossover",
                        "RSI", "MACD", "SMA", "EMA", "breakout", "backtest",
                        "win rate", "sharpe", "drawdown"
                    ]
                    all_raw = []
                    for topic in data.get("RelatedTopics", [])[:max_results]:
                        if "Text" in topic:
                            all_raw.append({
                                "title": topic.get("Text", "")[:100],
                                "snippet": topic.get("Text", ""),
                            })

                    relevant = []
                    for r in all_raw:
                        text = (r.get("snippet", "") + " " + r.get("title", "")).lower()
                        score = sum(1 for kw in strategy_keywords if kw.lower() in text)
                        if score >= 2:
                            relevant.append({
                                "title": r.get("title", ""),
                                "snippet": r.get("snippet", ""),
                                "relevance_score": score,
                            })

                    relevant.sort(key=lambda x: x["relevance_score"], reverse=True)
                    if relevant:
                        lines = [f"Found {len(relevant)} strategy-relevant results:"]
                        for r in relevant[:5]:
                            lines.append(f"\n[Score={r['relevance_score']}] {r['title']}")
                            lines.append(f"  {r['snippet'][:200]}")
                        lines.append(
                            "\nNext step: Use generate_custom_strategy_spec to convert "
                            "the most relevant result into a testable strategy spec."
                        )
                        return "\n".join(lines)
                except Exception:
                    pass

                return f"Search results for '{query}':\n" + "\n".join(results)

            except ImportError:
                return "Search unavailable: install httpx (`pip install httpx`)"
            except Exception as exc:
                return f"Search error: {exc}"

        def read_paper(url_json: str = "{}") -> str:
            """Fetch and summarise a URL (paper, article, blog post) into structured strategy info.
            Pass JSON: {"url": "https://example.com/strategy-guide"}
            Returns: strategy_name, indicators, entry_logic, exit_logic, claimed_edge, timeframe_recommendation."""
            import json
            try:
                params = json.loads(url_json)
            except json.JSONDecodeError:
                params = {"url": url_json}
            url = params.get("url", "")
            if not url:
                return "Error: no URL provided."

            try:
                import httpx
                resp = httpx.get(url, timeout=30.0, follow_redirects=True)
                resp.raise_for_status()
                html = resp.text

                # Extract text content from HTML (basic stripping)
                import re
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s+", " ", text).strip()

                # Truncate to manageable size
                content = text[:6000]

                return (
                    f"Fetched: {url}\n"
                    f"Length: {len(text)} chars\n\n"
                    f"Content preview:\n{content[:2000]}"
                )

            except ImportError:
                return "Paper reading unavailable: install httpx (`pip install httpx`)"
            except Exception as exc:
                return f"Error fetching URL: {exc}"

        def generate_custom_strategy_spec(spec_json: str = "{}") -> str:
            """
            Create a structured strategy spec that maps to a known strategy type.
            Pass JSON: {
                "name": "My Strategy",
                "concept": "Buy when RSI oversold and price above 200 SMA",
                "regime": "ranging"
            }
            Returns a spec the strategist can use directly with generate_strategy.
            """
            import json
            from data.strategy_concepts import STRATEGY_CONCEPTS, get_concepts_for_regime

            try:
                params = json.loads(spec_json) if spec_json.strip() else {}
            except json.JSONDecodeError:
                params = {"name": spec_json, "concept": spec_json}

            name = params.get("name", "Custom")
            concept = params.get("concept", "")
            regime = params.get("regime", "")

            # Find closest matching concept from the library
            best_match = None
            if concept:
                concept_lower = concept.lower()
                keyword_map = {
                    "sma_crossover":       ["sma", "moving average", "crossover", "golden cross"],
                    "rsi_oversold":        ["rsi", "oversold", "overbought", "relative strength"],
                    "bollinger_bands":     ["bollinger", "band", "squeeze", "volatility band"],
                    "macd_crossover":      ["macd", "histogram", "signal line"],
                    "momentum":            ["momentum", "roc", "rate of change", "volume"],
                    "breakout":            ["breakout", "high", "resistance", "support break"],
                    "mean_reversion":      ["mean reversion", "reversal", "oversold bounce"],
                    "volatility_squeeze":  ["squeeze", "contraction", "expansion", "low vol"],
                    "multi_timeframe":     ["multi timeframe", "higher timeframe", "confluence"],
                    "sentiment_driven":    ["fear", "greed", "sentiment", "emotion"],
                }
                best_type = "sma_crossover"
                best_score = 0
                for strat_type, keywords in keyword_map.items():
                    score = sum(1 for kw in keywords if kw in concept_lower)
                    if score > best_score:
                        best_score = score
                        best_type = strat_type

                # Find matching concept in library
                for c in STRATEGY_CONCEPTS:
                    if c["freqtrade_type"] == best_type:
                        best_match = c
                        break

            # Build the spec
            spec = {
                "name": name,
                "concept": concept,
                "suggested_strategy_type": best_match["freqtrade_type"] if best_match else "sma_crossover",
                "suggested_params": best_match.get("suggested_params", {}) if best_match else {},
                "regime": regime or (best_match["best_regime"] if best_match else "any"),
                "ready_to_use": True,
                "usage": (
                    f"Call generate_strategy with strategy_type='{best_match['freqtrade_type'] if best_match else 'sma_crossover'}'"
                    f" and params={json.dumps(best_match.get('suggested_params', {}) if best_match else {})}"
                )
            }

            spec_id = f"spec_{name[:8].replace(' ', '_').lower()}"
            if not hasattr(self, "_specs"):
                self._specs = {}
            self._specs[spec_id] = spec

            return json.dumps(spec, indent=2)

        return [
            Tool(name="web_search", func=web_search,
                 description="Search the web for trading strategy ideas or papers. Args: JSON with query and max_results."),
            Tool(name="read_paper", func=read_paper,
                 description="Fetch and summarise a URL into strategy info. Args: JSON with url."),
            Tool(name="generate_custom_strategy_spec", func=generate_custom_strategy_spec,
                 description="Create a structured strategy spec that maps to a known strategy type. Pass JSON with name, concept, and optional regime. Returns a ready-to-use spec for generate_strategy."),
        ]

    def get_specs(self) -> Dict[str, Dict[str, Any]]:
        """Return all generated strategy specs."""
        return dict(self._generated_specs)
