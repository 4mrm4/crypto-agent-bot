"""Tests for LiveExecutor exchange API key usage.

Verifies that the exchange is initialised with EXCHANGE_API_KEY and
EXCHANGE_SECRET from settings, NOT OPENAI_API_KEY.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_settings():
    """Patch config.settings so exchange keys are predictable.

    We patch at the config level because LiveExecutor.__init__ re-imports
    ``from config import settings`` inside the method body, bypassing any
    patch on ``execution.live_executor.settings``.
    """
    with patch("config.settings") as mock:
        mock.EXCHANGE_API_KEY = "test_exchange_key_123"
        mock.EXCHANGE_SECRET = "test_secret_456"
        mock.OPENAI_API_KEY = "sk-should-not-be-used"
        mock.LLM_MODEL = "deepseek-chat"
        mock.LLM_TEMPERATURE = 0.3
        mock.OPENAI_BASE_URL = "https://api.deepseek.com/v1"
        mock.EXECUTION_MODE = "live"
        mock.LIVE_START_DATE = ""
        mock.STOP_LOSS_DEFAULT = 0.04
        mock.TAKE_PROFIT_DEFAULT = 0.08
        mock.TWAP_THRESHOLD_USDT = 500
        mock.LIVE_MAX_POSITION_USDT = 100
        mock.PAPER_INITIAL_BALANCE = 10000.0
        yield mock


@pytest.fixture
def mock_ccxt_exchange():
    """Return a MagicMock that stands in for a ccxt Exchange class."""
    exchange_instance = MagicMock()
    exchange_instance.load_markets = MagicMock()

    exchange_class = MagicMock(return_value=exchange_instance)
    with patch("ccxt.kraken", exchange_class):
        yield exchange_class, exchange_instance


class TestLiveExecutorExchangeKeys:
    """Suite: exchange credentials come from exchange-specific settings."""

    def test_uses_exchange_api_key_not_openai_key(
        self, mock_settings, mock_ccxt_exchange
    ):
        """The Exchange API key passed to CCXT must be EXCHANGE_API_KEY,
        never OPENAI_API_KEY."""
        exchange_class, exchange_instance = mock_ccxt_exchange

        from execution.live_executor import LiveExecutor

        executor = LiveExecutor(
            exchange_id="kraken",
            paper_mode=False,
        )
        _ = executor.exchange  # trigger lazy init

        call_kwargs = exchange_class.call_args[0][0]
        assert call_kwargs["apiKey"] == "test_exchange_key_123", (
            f"Expected 'test_exchange_key_123' but got {call_kwargs['apiKey']!r}"
        )
        assert call_kwargs["apiKey"] != mock_settings.OPENAI_API_KEY, (
            "Exchange apiKey must NOT be OPENAI_API_KEY"
        )

    def test_secret_passed_correctly(
        self, mock_settings, mock_ccxt_exchange
    ):
        """The Exchange secret must be passed to CCXT verbatim."""
        exchange_class, exchange_instance = mock_ccxt_exchange

        from execution.live_executor import LiveExecutor

        executor = LiveExecutor(
            exchange_id="kraken",
            paper_mode=False,
        )
        _ = executor.exchange  # trigger lazy init

        call_kwargs = exchange_class.call_args[0][0]
        assert call_kwargs["secret"] == "test_secret_456", (
            f"Expected 'test_secret_456' but got {call_kwargs['secret']!r}"
        )

    def test_still_uses_paper_mode_by_default(
        self, mock_settings, mock_ccxt_exchange
    ):
        """Default paper_mode=True means exchange is never initialised, so
        no API keys are required."""
        exchange_class, exchange_instance = mock_ccxt_exchange

        from execution.live_executor import LiveExecutor

        executor = LiveExecutor(exchange_id="kraken")  # paper_mode defaults True
        result = executor.exchange  # paper mode returns None

        assert result is None, "Paper mode should return None for exchange property"
        exchange_class.assert_not_called(), (
            "CCXT exchange class should never be constructed in paper mode"
        )
