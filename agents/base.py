"""Base agent class used by all specialised agents in the system."""

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agents.token_tracker import TokenTracker
from config import settings

logger = logging.getLogger(__name__)


class TokenUsageHandler(BaseCallbackHandler):
    """Captures token usage from every LLM call into the global TokenTracker."""

    def on_llm_end(self, response, **kwargs):
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage", {})
            if usage:
                TokenTracker.get().add_usage(
                    prompt=usage.get("prompt_tokens", 0),
                    completion=usage.get("completion_tokens", 0),
                )
        except Exception:
            pass


class BaseAgent:
    """Lightweight wrapper around a LangGraph ReAct agent with tool access."""

    def __init__(
        self,
        name: str,
        tools: List[Tool],
        llm: Optional[ChatOpenAI] = None,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.name = name
        self.tools = {t.name: t for t in tools}
        self.tool_list = tools
        self.llm = llm or ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL or None,
            callbacks=[TokenUsageHandler()],
        )
        self.system_prompt = system_prompt

        # Build the compiled LangGraph ReAct agent
        self._agent = create_react_agent(
            model=self.llm,
            tools=self.tool_list,
            prompt=self.system_prompt,
        )

    def run(self, input_text: str) -> Dict[str, Any]:
        """Invoke the agent and return the structured result."""
        logger.info("[%s] Running with input: %.120s", self.name, input_text)
        result = self._agent.invoke(
            {"messages": [("user", input_text)]},
            config={"recursion_limit": 50},
        )
        raw_content = result.get("messages", [])[-1].content if result.get("messages") else ""
        # Handle content blocks (LangChain list format) or plain string
        if isinstance(raw_content, list):
            output = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in raw_content
            )
        else:
            output = str(raw_content)
        # Sanitise non-ASCII characters for Windows console
        output = output.encode("ascii", errors="replace").decode("ascii")
        steps = [
            {"thought": m.content, "tool": m.name if hasattr(m, "name") else ""}
            for m in result.get("messages", [])
            if hasattr(m, "additional_kwargs") and m.additional_kwargs.get("tool_calls")
        ]
        return {"output": output, "intermediate_steps": steps}

    def get_tool(self, name: str) -> Optional[Tool]:
        """Retrieve a registered tool by name."""
        return self.tools.get(name)