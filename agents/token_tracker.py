"""Thread-safe global token usage tracker.

All agents share this singleton so we can emit cumulative token
usage counts to the Web UI in real-time.
"""

import threading
from typing import Dict


class TokenTracker:
    """Singleton that accumulates token counts across all LLM calls."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._call_lock = threading.Lock()

    @classmethod
    def get(cls) -> "TokenTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def add_usage(self, prompt: int = 0, completion: int = 0):
        with self._call_lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def get_usage(self) -> Dict[str, int]:
        with self._call_lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def reset(self):
        with self._call_lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
