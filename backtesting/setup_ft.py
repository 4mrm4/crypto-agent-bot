"""Scaffold a minimal Freqtrade user data directory for backtesting.

Run this once before running backtests:

    python backtesting/setup_ft.py
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FT_DIR = Path("./ft_userdata")


def main():
    FT_DIR.mkdir(parents=True, exist_ok=True)
    subdirs = ["data", "strategies", "backtest_results", "logs", "notebooks"]
    for d in subdirs:
        (FT_DIR / d).mkdir(parents=True, exist_ok=True)

    # Write a minimal config.json
    config_path = FT_DIR / "config.json"
    if not config_path.exists():
        config = {
            "max_open_trades": 3,
            "stake_currency": "USDT",
            "stake_amount": 100,
            "dry_run": True,
            "dry_run_wallet": 10000,
            "trading_mode": "spot",
            "exchange": {
                "name": "binance",
                "pair_whitelist": ["BTC/USDT"],
                "ccxt_config": {"enableRateLimit": True},
            },
            "pairlists": [{"method": "StaticPairList"}],
            "timeframe": "1h",
            "fiat_display_currency": "USD",
            "dataformat_ohlcv": "json",
            "telegram_enabled": False,
            "api_server": {
                "enabled": False,
                "listen_ip_address": "127.0.0.1",
                "listen_port": 8080,
                "username": "admin",
                "password": "changeme",
                "jwt_secret_key": "somethingRandomSomethingRandom123",
            },
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Created %s", config_path)
    else:
        logger.info("%s already exists, skipping.", config_path)

    # Write a placeholder strategy so Freqtrade doesn't complain
    strategy_path = FT_DIR / "strategies" / "SampleStrategy.py"
    if not strategy_path.exists():
        strategy_path.write_text(
            '''from freqtrade.strategy import IStrategy

class SampleStrategy(IStrategy):
    timeframe = "1h"
    minimal_roi = {"0": 0.01}
    stoploss = -0.05

    def populate_indicators(self, dataframe, metadata):
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        return dataframe
''',
            encoding="utf-8",
        )
        logger.info("Created %s", strategy_path)

    logger.info("Freqtrade user data directory ready at %s", FT_DIR.resolve())
    logger.info("Run 'python backtesting/setup_ft.py' again if you need to re-scaffold.")


if __name__ == "__main__":
    main()