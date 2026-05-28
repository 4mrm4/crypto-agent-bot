"""Researcher agent — web search, paper reading, and strategy spec generation."""

import json
import logging
from typing import Any, Dict

from agents.base import BaseAgent
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

    def _build_tools(self):
        def web_search(query_json: str = "{}") -> str:
            """Search the web for trading strategy ideas, papers, or articles.
            Pass JSON: {"query": "momentum strategy crypto Bollinger Bands", "max_results": 5}
            Returns a markdown list of results with titles, snippets, and URLs."""
            import json
            try:
                params = json.loads(query_json)
            except json.JSONDecodeError:
                params = {"query": query_json}
            query = params.get("query", "")
            max_results = int(params.get("max_results", 5))
            if not query:
                return "Error: empty query"

            try:
                import httpx
                # Use DuckDuckGo instant answer API (no API key needed)
                url = "https://api.duckduckgo.com/"
                resp = httpx.get(
                    url,
                    params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()

                results = []
                # Abstract text
                abstract = data.get("AbstractText", "")
                if abstract:
                    source = data.get("Source", "web")
                    results.append(f"- **{source}**: {abstract[:300]}")

                # Related topics
                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if "Text" in topic:
                        text = topic["Text"][:200]
                        url = topic.get("FirstURL", "")
                        results.append(f"- {text} ([link]({url}))")

                if not results:
                    return "No results found for that query."

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
            """Generate a fully-specified custom strategy spec for the freqtrade backtester.
            Pass JSON with:
            {
                "name": "ema_trend_follow",
                "indicator_code": "dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=10)\\ndataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=30)",
                "entry_condition": "(dataframe['ema_fast'] > dataframe['ema_slow']) & (dataframe['volume'] > 1000)",
                "exit_condition": "(dataframe['ema_fast'] < dataframe['ema_slow'])",
                "timeframe": "1h",
                "description": "EMA trend following with volume confirmation"
            }

            Returns the spec ID and a summary. The spec can be passed to the strategist's
            generate_strategy tool with strategy_type='custom' and these code blocks."""
            import json, uuid
            try:
                spec = json.loads(spec_json)
            except json.JSONDecodeError:
                spec = {}

            required = ["indicator_code", "entry_condition", "exit_condition"]
            missing = [k for k in required if not spec.get(k)]
            if missing:
                return f"Error: missing required fields: {', '.join(missing)}"

            spec_id = uuid.uuid4().hex[:8]
            spec.setdefault("name", f"custom_{spec_id}")
            spec.setdefault("timeframe", "1h")
            spec.setdefault("description", "Custom strategy")
            self._generated_specs[spec_id] = spec

            return (
                f"Strategy spec [{spec_id}] created: {spec['name']}\n"
                f"  Description: {spec['description']}\n"
                f"  Timeframe: {spec['timeframe']}\n"
                f"  Indicator code: {spec['indicator_code'][:100]}...\n"
                f"  Entry: {spec['entry_condition'][:100]}...\n"
                f"  Exit: {spec['exit_condition'][:100]}...\n\n"
                f"To backtest this, call strategist's generate_strategy with:\n"
                f'  {{"strategy_type": "custom", '
                f'"indicator_code": "{spec["indicator_code"]}", '
                f'"entry_condition": "{spec["entry_condition"]}", '
                f'"exit_condition": "{spec["exit_condition"]}"}}'
            )

        return [
            Tool(name="web_search", func=web_search,
                 description="Search the web for trading strategy ideas or papers. Args: JSON with query and max_results."),
            Tool(name="read_paper", func=read_paper,
                 description="Fetch and summarise a URL into strategy info. Args: JSON with url."),
            Tool(name="generate_custom_strategy_spec", func=generate_custom_strategy_spec,
                 description="Generate a fully-specified custom strategy for freqtrade. Args: JSON with indicator_code, entry_condition, exit_condition."),
        ]

    def get_specs(self) -> Dict[str, Dict[str, Any]]:
        """Return all generated strategy specs."""
        return dict(self._generated_specs)
