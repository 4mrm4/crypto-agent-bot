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
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

    # Optional API keys
    CRYPTOPANIC_API_KEY: str = os.getenv("CRYPTOPANIC_API_KEY", "")
    COINGECKO_API_KEY: str = os.getenv("COINGECKO_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    WHALE_ALERT_API_KEY: str = os.getenv("WHALE_ALERT_API_KEY", "")

    # Feature flags
    ENABLE_SENTIMENT: bool = os.getenv("ENABLE_SENTIMENT", "true").lower() == "true"
    ENABLE_PATTERNS: bool = os.getenv("ENABLE_PATTERNS", "true").lower() == "true"
    ENABLE_ONCHAIN: bool = os.getenv("ENABLE_ONCHAIN", "false").lower() == "true"

    # Autonomous mode
    AUTONOMOUS_INTERVAL_MINUTES: int = int(os.getenv("AUTONOMOUS_INTERVAL_MINUTES", "30"))
    DECAY_THRESHOLD: float = float(os.getenv("DECAY_THRESHOLD", "0.20"))
    COVERAGE_GAP_SHARPE: float = float(os.getenv("COVERAGE_GAP_SHARPE", "0.80"))
    MAX_DAYS_WITHOUT_REGIME_RESEARCH: int = int(os.getenv("MAX_DAYS_WITHOUT_REGIME_RESEARCH", "7"))

    # Live execution
    EXECUTION_MODE: str = os.getenv("EXECUTION_MODE", "paper")
    LIVE_MAX_POSITION_USDT: float = float(os.getenv("LIVE_MAX_POSITION_USDT", "100.0"))
    TWAP_THRESHOLD_USDT: float = float(os.getenv("TWAP_THRESHOLD_USDT", "500.0"))
    STOP_LOSS_DEFAULT: float = float(os.getenv("STOP_LOSS_DEFAULT", "0.04"))
    TAKE_PROFIT_DEFAULT: float = float(os.getenv("TAKE_PROFIT_DEFAULT", "0.08"))

    # Anti-overfitting and data integrity
    BACKTEST_OPTIMISM_FACTOR: float = float(os.getenv("BACKTEST_OPTIMISM_FACTOR", "0.55"))
    LIVE_START_DATE: str = os.getenv("LIVE_START_DATE", "")
    VALIDATION_MODE_DAYS: int = int(os.getenv("VALIDATION_MODE_DAYS", "90"))

    # Multi-exchange
    EXCHANGES: str = os.getenv("EXCHANGES", "binance")
    FUTURES_ENABLED: bool = os.getenv("FUTURES_ENABLED", "false").lower() == "true"

    # Transaction cost model
    MAKER_FEE: float = float(os.getenv("MAKER_FEE", "0.001"))
    TAKER_FEE: float = float(os.getenv("TAKER_FEE", "0.00075"))
    SLIPPAGE_PCT: float = float(os.getenv("SLIPPAGE_PCT", "0.0005"))
    SLIPPAGE_MODEL: str = os.getenv("SLIPPAGE_MODEL", "fixed")

    # Tavily search
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    TAVILY_ENABLED: bool = os.getenv("TAVILY_ENABLED", "true").lower() == "true"

    # Legacy JSONL backup (keep for 30 days after SQLite migration, then disable)
    LEGACY_JSONL_BACKUP: bool = os.getenv("LEGACY_JSONL_BACKUP", "true").lower() == "true"

    # Fast pre-filter (SignalFactory + FastMetrics)
    VECTORBT_PREFILTER_ENABLED: bool = os.getenv("VECTORBT_PREFILTER_ENABLED", "true").lower() == "true"
    VECTORBT_PREFILTER_MIN_SHARPE: float = float(os.getenv("VECTORBT_PREFILTER_MIN_SHARPE", "0.5"))
    VECTORBT_PREFILTER_MIN_WIN_RATE: float = float(os.getenv("VECTORBT_PREFILTER_MIN_WIN_RATE", "0.40"))
    VECTORBT_PREFILTER_MIN_TRADES: int = int(os.getenv("VECTORBT_PREFILTER_MIN_TRADES", "3"))

    # Redis (optional — StateBroker uses in-memory by default)
    REDIS_URL: str = os.getenv("REDIS_URL", "")


settings = Settings()