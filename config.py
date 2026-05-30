"""Central configuration module loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from .env with sensible defaults."""

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EXCHANGE_ID: str = os.getenv("EXCHANGE_ID", "binance")
    SYMBOL: str = os.getenv("SYMBOL", "BTC/USDT")
    TIMEFRAME: str = os.getenv("TIMEFRAME", "1h")
    DATA_LIMIT: int = int(os.getenv("DATA_LIMIT", "500"))
    BACKTEST_CONFIG_PATH: str = os.getenv(
        "BACKTEST_CONFIG_PATH", "./ft_userdata/config.json"
    )
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    WORKSPACE_REGISTRY_PATH: str = os.getenv(
        "WORKSPACE_REGISTRY_PATH", "./workspace_registry.json"
    )
    PAPER_INITIAL_BALANCE: float = float(os.getenv("PAPER_INITIAL_BALANCE", "10000.0"))

    # Freqtrade subprocess timeout in seconds
    BACKTEST_TIMEOUT: int = int(os.getenv("BACKTEST_TIMEOUT", "300"))

    # LangChain model
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")

    # Optional API keys
    CRYPTOPANIC_API_KEY: str = os.getenv("CRYPTOPANIC_API_KEY", "")
    WHALE_ALERT_API_KEY: str = os.getenv("WHALE_ALERT_API_KEY", "")

    # Feature flags
    ENABLE_SENTIMENT: bool = os.getenv("ENABLE_SENTIMENT", "true").lower() == "true"
    ENABLE_PATTERNS: bool = os.getenv("ENABLE_PATTERNS", "true").lower() == "true"
    ENABLE_ONCHAIN: bool = os.getenv("ENABLE_ONCHAIN", "false").lower() == "true"


settings = Settings()