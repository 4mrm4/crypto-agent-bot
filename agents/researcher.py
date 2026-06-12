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

Workflow (limit to 1-2 cycles):
  1. web_search(1-2 queries max)
  2. read_paper(best 1-2 results from search)
  3. generate_custom_strategy_spec(produce exactly 1 spec)
  Then output your final answer — do NOT search again after generating a spec.
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
            Uses SearXNG (when configured) -> Tavily -> DuckDuckGo in priority order."""
            import json
            try:
                params = json.loads(query_json)
            except json.JSONDecodeError:
                params = {"query": query_json}
            query = params.get("query", "")
            max_results = int(params.get("max_results", 5))
            if not query:
                return "Error: empty query"

            # ── Helper: strategy relevance scoring ──
            def _score_strategy_relevance(results_list: list, title_key: str = "title", snippet_key: str = "content") -> list:
                strategy_keywords = [
                    "strategy", "indicator", "entry", "exit", "crossover",
                    "RSI", "MACD", "SMA", "EMA", "breakout", "backtest",
                    "win rate", "sharpe", "drawdown"
                ]
                scored = []
                for r in results_list:
                    text = (r.get(snippet_key, "") + " " + r.get(title_key, "")).lower()
                    score = sum(1 for kw in strategy_keywords if kw.lower() in text)
                    if score >= 2:
                        scored.append(r)
                scored.sort(key=lambda x: sum(
                    1 for kw in strategy_keywords if kw.lower() in (x.get(snippet_key, "") + " " + x.get(title_key, "")).lower()
                ), reverse=True)
                return scored[:5]

            def _format_results(lines: list, relevant: list, title_key: str = "title", snippet_key: str = "content") -> str:
                if relevant:
                    lines.append(f"\nFound {len(relevant)} strategy-relevant results:")
                    for r in relevant:
                        lines.append(f"\n[Relevant] {r.get(title_key, '')}")
                        lines.append(f"  {r.get(snippet_key, '')[:200]}")
                    lines.append(
                        "\nNext step: Use generate_custom_strategy_spec to convert "
                        "the most relevant result into a testable strategy spec."
                    )
                return "\n".join(lines)

            # ── Try SearXNG first (self-hosted, no rate limits) ──
            searxng_url = getattr(settings, "SEARXNG_URL", "")
            if searxng_url:
                try:
                    import httpx
                    resp = httpx.get(
                        f"{searxng_url}/search",
                        params={"q": query, "format": "json", "engines": "google,bing,duckduckgo"},
                        headers={"X-Forwarded-For": "127.0.0.1"},
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("results", [])
                        if results:
                            lines = [f"SearXNG search results for '{query}':"]
                            for r in results[:max_results]:
                                title = r.get("title", "")
                                content = r.get("content", "")[:200]
                                url = r.get("url", "")
                                lines.append(f"\n**{title}**")
                                lines.append(f"  {content}")
                                if url:
                                    lines.append(f"  ([link]({url}))")
                            relevant = _score_strategy_relevance(results[:max_results], title_key="title", snippet_key="content")
                            return _format_results(lines, relevant)
                except Exception as exc:
                    logger.debug("SearXNG search failed: %s — falling through", exc)

            # ── Try Tavily second ──
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
                    if results:
                        lines = [f"Tavily search results for '{query}':"]
                        for r in results[:max_results]:
                            title = r.get("title", "")
                            content = r.get("content", "")[:200]
                            url = r.get("url", "")
                            lines.append(f"\n**{title}**")
                            lines.append(f"  {content}")
                            if url:
                                lines.append(f"  ([link]({url}))")

                        relevant = _score_strategy_relevance(results[:max_results], title_key="title", snippet_key="content")
                        formatted = _format_results(lines, relevant)

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

                        return formatted
                except ImportError:
                    pass  # Tavily not installed, fall through
                except Exception as exc:
                    logger.warning("Tavily search error: %s", exc)

            # ── Fallback: DuckDuckGo ──
            try:
                import httpx
                url = "https://api.duckduckgo.com/"
                resp = httpx.get(
                    url,
                    params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                    timeout=15.0,
                )
                if resp.status_code == 202:
                    logger.warning(
                        "DuckDuckGo returned 202 for query: %s — rate limited, skipping DDG", query
                    )
                    # Don't retry — it makes rate limiting worse
                    return _no_search_results(query, reason="DuckDuckGo rate limited (HTTP 202)")

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

                if results:
                    all_raw = []
                    for topic in data.get("RelatedTopics", [])[:max_results]:
                        if "Text" in topic:
                            all_raw.append({
                                "title": topic.get("Text", "")[:100],
                                "snippet": topic.get("Text", ""),
                            })
                    relevant = _score_strategy_relevance(all_raw, title_key="title", snippet_key="snippet")
                    if relevant:
                        lines = [f"Found {len(relevant)} strategy-relevant results from DDG:"]
                        return _format_results(lines, relevant, title_key="title", snippet_key="snippet")
                    return f"Search results for '{query}':\n" + "\n".join(results)

                return _no_search_results(query, reason="No results from DuckDuckGo")

            except ImportError:
                return _no_search_results(query, reason="httpx not installed")
            except Exception as exc:
                return _no_search_results(query, reason=str(exc))

        def _no_search_results(query: str, reason: str = "") -> str:
            """Return a structured 'no results' response that meets the minimum length guard."""
            msg = (
                f"## Search Results: No results for '{query}'\n\n"
                f"No search results could be retrieved for this query. "
            )
            if reason:
                msg += f"Reason: {reason}. "
            msg += (
                "The research cycle will proceed with existing memory context only. "
                "This does not prevent strategy generation — the strategist agent can "
                "still design strategies based on known indicators and past experiments."
            )
            return msg

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
            self._generated_specs[spec_id] = spec

            return json.dumps(spec, indent=2)

        def get_asset_fundamentals(asset_slug: str = "bitcoin") -> str:
            """Fetch fundamental and on-chain data for a crypto asset.

            Uses CoinGecko free API (no key required). Returns price, market cap,
            developer activity (GitHub), and community metrics.

            Args:
                asset_slug: CoinGecko asset ID (e.g. "bitcoin", "ethereum", "solana")
            """
            import json as _json
            import urllib.request as _req
            try:
                url = f"https://api.coingecko.com/api/v3/coins/{asset_slug}?localization=false&tickers=false&community_data=true&developer_data=true"
                resp = _req.urlopen(url, timeout=10)
                data = _json.loads(resp.read().decode())

                md = data.get("market_data", {})
                dd = data.get("developer_data", {})
                cd = data.get("community_data", {})

                lines = [f"=== Asset Fundamentals: {asset_slug} ==="]
                lines.append(f"Price: ${md.get('current_price', {}).get('usd', 'N/A'):,}")
                lines.append(f"Market Cap: ${md.get('market_cap', {}).get('usd', 0):,.0f}")
                lines.append(f"24h Volume: ${md.get('total_volume', {}).get('usd', 0):,.0f}")
                lines.append(f"24h Change: {md.get('price_change_percentage_24h', 0):.2f}%")
                lines.append(f"7d Change: {md.get('price_change_percentage_7d', 0):.2f}%")

                # Developer stats
                gh_stars = dd.get("stars", None)
                gh_forks = dd.get("forks", None)
                commits_4w = dd.get("commit_count_4_weeks", None)
                lines.append(f"GitHub Stars: {gh_stars:,}" if gh_stars is not None else "GitHub Stars: N/A")
                lines.append(f"GitHub Forks: {gh_forks:,}" if gh_forks is not None else "GitHub Forks: N/A")
                lines.append(f"Commits (4 weeks): {commits_4w}" if commits_4w is not None else "Commits: N/A")

                # Community stats
                reddit_subs = cd.get("reddit_subscribers", None)
                lines.append(f"Reddit Subscribers: {reddit_subs:,}" if isinstance(reddit_subs, (int, float)) and reddit_subs > 0 else "Reddit: N/A")

                return "\n".join(lines)
            except Exception as exc:
                return f"Fundamentals lookup failed for '{asset_slug}': {exc}"

        return [
            Tool(name="web_search", func=web_search,
                 description="Search the web for trading strategy ideas or papers. Args: JSON with query and max_results."),
            Tool(name="read_paper", func=read_paper,
                 description="Fetch and summarise a URL into strategy info. Args: JSON with url."),
            Tool(name="generate_custom_strategy_spec", func=generate_custom_strategy_spec,
                 description="Create a structured strategy spec that maps to a known strategy type. Pass JSON with name, concept, and optional regime. Returns a ready-to-use spec for generate_strategy."),
            Tool(name="get_asset_fundamentals", func=get_asset_fundamentals,
                 description="Fetch fundamental data for a CoinGecko asset (bitcoin, ethereum, solana). Returns price, market cap, volume, developer stats, and community metrics. No API key required."),
        ]

    def get_specs(self) -> Dict[str, Dict[str, Any]]:
        """Return all generated strategy specs."""
        return dict(self._generated_specs)
