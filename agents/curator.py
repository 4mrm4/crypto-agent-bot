"""Curator agent — retrieves relevant past insights for agent context."""

import logging
from typing import Optional

from langchain_core.tools import Tool

from agents.base import BaseAgent
from memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

CURATOR_SYSTEM_PROMPT = """You are the Memory Curator. Your job is to:
1. Search the vector memory for insights related to the current goal
2. Summarise what was learned from previous similar research
3. Provide context to help other agents avoid repeating failed approaches
4. Always check what strategies were tried before and how they performed

Use search_memory to find relevant past insights. Then synthesise a brief context summary.

IMPORTANT: Use ONLY plain ASCII text. No emoji, no Unicode symbols."""


class CuratorAgent(BaseAgent):
    """Agent that retrieves and summarises past insights from vector memory."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self._vector_store = vector_store or VectorStore()
        tools = self._build_tools()
        super().__init__(
            name="curator",
            tools=tools,
            system_prompt=CURATOR_SYSTEM_PROMPT,
        )

    def _build_tools(self):
        def search_memory_fn(query_json: str = '{"query":""}') -> str:
            """Search vector memory for relevant past insights.
            Args: JSON like {"query": "SMA crossover BTC", "k": 5}"""
            import json
            try:
                params = json.loads(query_json) if query_json.strip() else {}
            except json.JSONDecodeError:
                params = {"query": query_json}

            query = params.get("query", "")
            k = int(params.get("k", 5))
            if not query:
                return "Error: empty query"

            results = self._vector_store.query_similar(query, k=k)
            if not results:
                return "No relevant memories found."

            lines = [f"Found {len(results)} relevant past insights:"]
            for i, r in enumerate(results):
                text = r["text"][:200]
                meta = r.get("metadata", {})
                lines.append(f"\n[{i+1}] {text}")
                if meta:
                    lines.append(f"    (goal: {meta.get('goal_id', 'N/A')})")
            return "\n".join(lines)

        def store_insight_fn(kwargs_json: str = '{"text":"","metadata":{}}') -> str:
            """Store a new insight into vector memory.
            Args: JSON like {"text": "SMA 10/30 had Sharpe -2.66", "metadata": {"goal_id": "...", "agent": "strategist"}}"""
            import json
            try:
                params = json.loads(kwargs_json) if kwargs_json.strip() else {}
            except json.JSONDecodeError:
                return f"Error: invalid JSON"

            text = params.get("text", "")
            metadata = params.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if not text:
                return "Error: empty text"

            self._vector_store.store_insight(text, metadata=metadata)
            return f"Stored insight ({len(text)} chars)"
        from langchain_core.tools import Tool

        return [
            Tool(name="search_memory", func=search_memory_fn,
                 description="Search past agent insights. Args: JSON with 'query' and 'k'."),
            Tool(name="store_insight", func=store_insight_fn,
                 description="Store a new insight into long-term memory. Args: JSON with 'text' and 'metadata'."),
        ]

    def inject_context(self, goal: str, k: int = 3) -> str:
        """Retrieve relevant memories and format them as prompt context."""
        results = self._vector_store.query_similar(goal, k=k)
        if not results:
            return ""
        parts = ["[MEMORY CONTEXT - Past relevant research:]"]
        for r in results:
            parts.append(f"- {r['text'][:300]}")
            meta = r.get("metadata", {})
            if meta.get("goal_id"):
                parts[-1] += f" (from goal {meta['goal_id']})"
        return "\n".join(parts)

    def store_goal_result(self, goal_id: str, description: str, result: dict):
        """Store a completed goal's results into memory."""
        self._vector_store.store_insight(
            f"Goal: {description}\nBoard: {result.get('board_summary', '')}\nStrategies: {len(result.get('strategies', []))}",
            metadata={"goal_id": goal_id, "type": "goal_result"},
        )

    def store_result(self, goal: str, output: str, metadata: dict = None):
        """Store a research result into vector memory."""
        self._vector_store.store_insight(
            f"Goal: {goal}\nResult: {output[:500]}",
            metadata=metadata or {},
        )