"""Strategy template snippets injected into temporary Freqtrade strategy files.

Contains the string templates for all strategy types (SMA crossover, MACD,
RSI, Bollinger Bands, combined signals, momentum, breakout, mean reversion,
volatility squeeze, sentiment-driven, multi-timeframe) along with the
STRATEGY_REGISTRY dict that maps strategy type names to their code snippets
and default parameters.
"""

from typing import Any, Dict


STRATEGY_TEMPLATE = '''"""
Auto-generated strategy by crypto_agent_bot.
Do not edit manually — generated on $timestamp.
"""
from freqtrade.strategy import IStrategy, IntParameter
import pandas as pd
import talib.abstract as ta


class $strategy_name(IStrategy):
    # --- User-defined parameters (set by agent) ---
    timeframe = "$timeframe"
    minimal_roi = $minimal_roi
    stoploss = $stoploss
    trailing_stop = $trailing_stop
    startup_candle_count = $startup_candle_count
    process_only_new_candles = True
    use_exit_signal = True
    can_short = False

    # --- Indicator parameters ---
$indicator_params_block
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Coerce string-typed columns upfront (covers PyArrow backend which stores
        # strings as pd.ArrowDtype(pa.string()), not caught by simple 'string' check)
        import pandas.api.types as ptypes
        for col in dataframe.columns:
            if ptypes.is_string_dtype(dataframe[col]):
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        $indicator_code
        # Second pass: catch any new columns created by indicator code
        for col in dataframe.columns:
            if ptypes.is_string_dtype(dataframe[col]):
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                $entry_condition
            ),
            "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                $exit_condition
            ),
            "exit_long"] = 1
        return dataframe
'''


# ── Default SMA crossover indicator/entry/exit snippets ──

SMA_CROSSOVER_INDICATOR = """
        dataframe['fast_ma'] = ta.SMA(dataframe, timeperiod=self.fast_ma.value)
        dataframe['slow_ma'] = ta.SMA(dataframe, timeperiod=self.slow_ma.value)
"""

SMA_CROSSOVER_ENTRY = """
        (dataframe['fast_ma'].shift(1) <= dataframe['slow_ma'].shift(1)) &
        (dataframe['fast_ma'] > dataframe['slow_ma'])
"""

SMA_CROSSOVER_EXIT = """
        (dataframe['fast_ma'].shift(1) >= dataframe['slow_ma'].shift(1)) &
        (dataframe['fast_ma'] < dataframe['slow_ma'])
"""

# ── MACD Crossover snippets ──

MACD_CROSSOVER_INDICATOR = """
        macd_data = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value,
        )
        dataframe['macd'] = macd_data['macd'].astype(float)
        dataframe['macdsignal'] = macd_data['macdsignal'].astype(float)
        dataframe['macd_hist'] = (dataframe['macd'] - dataframe['macdsignal']).astype(float)
"""

MACD_CROSSOVER_ENTRY = """
        (dataframe['macd_hist'].shift(1) <= 0) & (dataframe['macd_hist'] > 0)
"""

MACD_CROSSOVER_EXIT = """
        (dataframe['macd_hist'].shift(1) >= 0) & (dataframe['macd_hist'] < 0)
"""

# ── RSI Oversold/Overbought snippets ──

RSI_INDICATOR = """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
"""

RSI_OVERSOLD_ENTRY = """
        (dataframe['rsi'] < self.rsi_buy_threshold.value) & (dataframe['rsi'].shift(1) >= self.rsi_buy_threshold.value)
"""

RSI_OVERSOLD_EXIT = """
        (dataframe['rsi'] > self.rsi_sell_threshold.value) & (dataframe['rsi'].shift(1) <= self.rsi_sell_threshold.value)
"""

# ── Bollinger Bands snippets ──

BB_INDICATOR = """
        upper, middle, lower = ta.BBANDS(
            dataframe['close'].astype(float),
            timeperiod=self.bb_period.value,
            nbdevup=2.0,
            nbdevdn=2.0,
        )
        dataframe['bb_upper'] = upper.astype(float)
        dataframe['bb_middle'] = middle.astype(float)
        dataframe['bb_lower'] = lower.astype(float)
"""

BB_ENTRY = """
        (dataframe['close'] < dataframe['bb_lower']) & (dataframe['close'].shift(1) >= dataframe['bb_lower'].shift(1))
"""

BB_EXIT = """
        (dataframe['close'] > dataframe['bb_upper']) & (dataframe['close'].shift(1) <= dataframe['bb_upper'].shift(1))
"""

# ── Combined SMA + RSI filter snippets ──

SMA_RSI_INDICATOR = """
        dataframe['fast_ma'] = ta.SMA(dataframe, timeperiod=self.fast_ma.value)
        dataframe['slow_ma'] = ta.SMA(dataframe, timeperiod=self.slow_ma.value)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
"""

SMA_RSI_ENTRY = """
        (dataframe['fast_ma'].shift(1) <= dataframe['slow_ma'].shift(1)) &
        (dataframe['fast_ma'] > dataframe['slow_ma']) &
        (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70)
"""

SMA_RSI_EXIT = """
        (dataframe['fast_ma'].shift(1) >= dataframe['slow_ma'].shift(1)) &
        (dataframe['fast_ma'] < dataframe['slow_ma'])
"""

# ── Strategy registry: maps type to its snippets and defaults ──

STRATEGY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "sma_crossover": {
        "indicator_code": SMA_CROSSOVER_INDICATOR,
        "entry_condition": SMA_CROSSOVER_ENTRY,
        "exit_condition": SMA_CROSSOVER_EXIT,
        "indicator_params_block": """
    fast_ma = IntParameter(5, 50, default=$fast_ma, space="buy")
    slow_ma = IntParameter(20, 200, default=$slow_ma, space="buy")
""",
        "default_params": {"fast_ma": 10, "slow_ma": 30, "startup_candle_count": 30},
    },
    "macd_crossover": {
        "indicator_code": MACD_CROSSOVER_INDICATOR,
        "entry_condition": MACD_CROSSOVER_ENTRY,
        "exit_condition": MACD_CROSSOVER_EXIT,
        "indicator_params_block": """
    macd_fast = IntParameter(8, 20, default=12, space="buy")
    macd_slow = IntParameter(20, 40, default=26, space="buy")
    macd_signal = IntParameter(6, 14, default=9, space="buy")
""",
        "default_params": {"startup_candle_count": 33},
    },
    "rsi_oversold": {
        "indicator_code": RSI_INDICATOR,
        "entry_condition": RSI_OVERSOLD_ENTRY,
        "exit_condition": RSI_OVERSOLD_EXIT,
        "indicator_params_block": """
    rsi_period = IntParameter(10, 21, default=14, space="buy")
    rsi_buy_threshold = IntParameter(25, 35, default=30, space="buy")
    rsi_sell_threshold = IntParameter(65, 80, default=70, space="sell")
""",
        "default_params": {"startup_candle_count": 20},
    },
    "bollinger_bands": {
        "indicator_code": BB_INDICATOR,
        "entry_condition": BB_ENTRY,
        "exit_condition": BB_EXIT,
        "indicator_params_block": """
    bb_period = IntParameter(15, 30, default=20, space="buy")
""",
        "default_params": {"startup_candle_count": 26},
    },
    "combined_sma_rsi": {
        "indicator_code": SMA_RSI_INDICATOR,
        "entry_condition": SMA_RSI_ENTRY,
        "exit_condition": SMA_RSI_EXIT,
        "indicator_params_block": """
    fast_ma = IntParameter(5, 50, default=$fast_ma, space="buy")
    slow_ma = IntParameter(20, 200, default=$slow_ma, space="buy")
""",
        "default_params": {"fast_ma": 10, "slow_ma": 30, "startup_candle_count": 30},
    },
    "custom": {
        "indicator_code": "",
        "entry_condition": "",
        "exit_condition": "",
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 20},
    },

    "momentum": {
        "indicator_code": """
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=10)
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
    """,
        "entry_condition": """
        (dataframe['roc'] > 2.0) &
        (dataframe['volume'] > dataframe['volume_ma'] * 1.5) &
        (dataframe['rsi'] > 50) & (dataframe['rsi'] < 75)
    """,
        "exit_condition": """
        (dataframe['roc'] < 0) | (dataframe['rsi'] > 75)
    """,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 25},
    },

    "breakout": {
        "indicator_code": """
        dataframe['highest_high'] = dataframe['high'].rolling(20).max().shift(1)
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
    """,
        "entry_condition": """
        (dataframe['close'] > dataframe['highest_high']) &
        (dataframe['volume'] > dataframe['volume_ma'] * 1.3)
    """,
        "exit_condition": """
        (dataframe['close'] < dataframe['highest_high'] - dataframe['atr'] * 2)
    """,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 25},
    },

    "mean_reversion": {
        "indicator_code": """
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bb_upper.astype(float)
        dataframe['bb_middle'] = bb_middle.astype(float)
        dataframe['bb_lower'] = bb_lower.astype(float)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['distance_from_mean'] = (dataframe['close'] - dataframe['bb_middle']) / dataframe['bb_middle']
    """,
        "entry_condition": """
        (dataframe['close'] < dataframe['bb_lower']) &
        (dataframe['rsi'] < 35) &
        (dataframe['distance_from_mean'] < -0.02)
    """,
        "exit_condition": """
        (dataframe['close'] > dataframe['bb_middle']) | (dataframe['rsi'] > 60)
    """,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 25},
    },

    "volatility_squeeze": {
        "indicator_code": """
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bb_upper.astype(float)
        dataframe['bb_middle'] = bb_middle.astype(float)
        dataframe['bb_lower'] = bb_lower.astype(float)
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        dataframe['bb_width_min'] = dataframe['bb_width'].rolling(120).min()
        dataframe['macd'], dataframe['macdsignal'], _ = [
            x.astype(float) for x in ta.MACD(dataframe['close'].astype(float))]
    """,
        "entry_condition": """
        (dataframe['bb_width'] <= dataframe['bb_width_min'] * 1.05) &
        (dataframe['macd'] > dataframe['macdsignal'])
    """,
        "exit_condition": """
        (dataframe['bb_width'] > dataframe['bb_width_min'] * 3) |
        (dataframe['macd'] < dataframe['macdsignal'])
    """,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 130},
    },

    "sentiment_driven": {
        "indicator_code": """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
    """,
        "entry_condition": """
        (dataframe['rsi'] < 40) &
        (dataframe['close'] > dataframe['sma50'])
    """,
        "exit_condition": """
        (dataframe['rsi'] > 65) | (dataframe['close'] < dataframe['sma50'])
    """,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 55},
    },

    "multi_timeframe": {
        "indicator_code": """
        # Primary timeframe indicators
        dataframe['fast_sma'] = ta.SMA(dataframe, timeperiod=self.fast_ma.value)
        dataframe['slow_sma'] = ta.SMA(dataframe, timeperiod=self.slow_ma.value)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        # Higher timeframe proxies (longer SMAs on same data)
        dataframe['sma80'] = ta.SMA(dataframe, timeperiod=self.higher_tf_fast.value)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=self.higher_tf_slow.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
    """,
        "entry_condition": """
        # Short-term signal: fast SMA crosses above slow SMA
        (dataframe['fast_sma'].shift(1) <= dataframe['slow_sma'].shift(1)) &
        (dataframe['fast_sma'] > dataframe['slow_sma']) &
        # Long-term confirmation: price above 200 SMA (higher timeframe proxy)
        (dataframe['close'] > dataframe['sma200']) &
        # Trend strength: ADX above threshold
        (dataframe['adx'] > self.adx_threshold.value) &
        # RSI not overbought/oversold
        (dataframe['rsi'] > self.rsi_oversold.value) & (dataframe['rsi'] < self.rsi_overbought.value)
    """,
        "exit_condition": """
        (dataframe['fast_sma'].shift(1) >= dataframe['slow_sma'].shift(1)) &
        (dataframe['fast_sma'] < dataframe['slow_sma']) |
        (dataframe['close'] < dataframe['sma200'])
    """,
        "indicator_params_block": """
    fast_ma = IntParameter(5, 50, default=$fast_ma, space="buy")
    slow_ma = IntParameter(20, 100, default=$slow_ma, space="buy")
    adx_period = IntParameter(7, 21, default=$adx_period, space="buy")
    adx_threshold = IntParameter(15, 40, default=$adx_threshold, space="buy")
    rsi_period = IntParameter(10, 21, default=$rsi_period, space="buy")
    rsi_oversold = IntParameter(30, 50, default=$rsi_oversold, space="buy")
    rsi_overbought = IntParameter(60, 80, default=$rsi_overbought, space="sell")
    higher_tf_fast = IntParameter(60, 120, default=$higher_tf_fast, space="buy")
    higher_tf_slow = IntParameter(150, 250, default=$higher_tf_slow, space="buy")
""",
        "default_params": {"fast_ma": 10, "slow_ma": 30, "adx_period": 14, "adx_threshold": 20, "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 70, "higher_tf_fast": 80, "higher_tf_slow": 200, "startup_candle_count": 205},
    },
}
