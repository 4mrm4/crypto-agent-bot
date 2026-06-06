"""Auto-download historical data on startup if not already available."""
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

from config import settings

logger = logging.getLogger(__name__)

DEFAULT_PAIRS = ["BTC/USDT", "ETH/USDT"]
DEFAULT_TIMEFRAMES = ["1h", "4h"]
# 2 years back from today
DEFAULT_START = (datetime.utcnow() - timedelta(days=730)).strftime("%Y%m%d")


def ensure_data_available(
    ft_userdata_dir: str = "./ft_userdata",
    pairs: list = None,
    timeframes: list = None,
    start_date: str = None,
) -> bool:
    """
    Check if historical data exists and is sufficiently recent.
    If not, download it automatically.
    Returns True if data is ready, False if download failed.
    """
    pairs = pairs or DEFAULT_PAIRS
    timeframes = timeframes or DEFAULT_TIMEFRAMES
    start_date = start_date or DEFAULT_START

    ft_path = Path(ft_userdata_dir).resolve()
    data_path = ft_path / "data" / settings.EXCHANGE_ID

    # Check if we have at least 30 days of 1h BTC data
    btc_files = []
    if data_path.exists():
        # Freqtrade stores as BTC_USDT-1h.json or BTC_USDT-1h-trades.json.gz
        for pattern in ["BTC_USDT-1h.json", "BTC_USDT-1h*.json",
                        "BTC_USDT-1h*.feather", "BTC_USDT-1h*.gz"]:
            found = list(data_path.glob(pattern))
            if found:
                btc_files.extend(found)
                break  # found in this pattern, stop

    # Also check parent data dir (Freqtrade sometimes stores without exchange subdir)
    if not btc_files:
        parent_data = ft_path / "data"
        if parent_data.exists():
            for pattern in ["BTC_USDT-1h.json", "BTC_USDT-1h*.json"]:
                found = list(parent_data.glob(pattern))
                if found:
                    btc_files.extend(found)
                    break

    if btc_files:
        largest = max(btc_files, key=lambda f: f.stat().st_size)
        size_kb = largest.stat().st_size / 1024
        # 1h data: ~1KB per candle. 30 days = 720 candles ≈ 720KB minimum
        if size_kb > 500:
            logger.info(
                "Historical data OK: %s (%.0fKB, ~%d candles estimated)",
                largest.name, size_kb, int(size_kb)
            )
            return True
        else:
            logger.warning(
                "Data file too small (%.0fKB) — likely empty or corrupted. Re-downloading.",
                size_kb
            )

    logger.info("Historical data not found or insufficient — downloading...")
    logger.info("Pairs: %s, Timeframes: %s, From: %s", pairs, timeframes, start_date)

    # Find freqtrade
    candidates = [
        Path(ft_userdata_dir).parent / "venv" / "Scripts" / "freqtrade.exe",
        Path(ft_userdata_dir).parent / "venv" / "Scripts" / "freqtrade",
    ]
    ft_cmd = "freqtrade"
    for c in candidates:
        if c.exists():
            ft_cmd = str(c)
            break

    cmd = [
        ft_cmd,
        "download-data",
        "--userdir", str(ft_path),
        "--exchange", settings.EXCHANGE_ID,
        "-p", *pairs,
        "--timerange", f"{start_date}-",
        "--timeframes", *timeframes,
        "--datadir", str(data_path.parent),
        "--prepend",
    ]

    try:
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            logger.error("Data download failed:\n%s", result.stderr[-500:])
            return False
        logger.info("Data download complete.")
        return True
    except subprocess.TimeoutExpired:
        logger.error("Data download timed out after 10 minutes.")
        return False
    except Exception as exc:
        logger.error("Data download error: %s", exc)
        return False
