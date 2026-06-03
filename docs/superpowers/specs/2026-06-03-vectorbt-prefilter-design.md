# vectorbt Fast Pre-filter (SignalFactory + FastMetrics)

## Motivation

The bot backtests every strategy variant via Freqtrade subprocess (30-60s per run).
Most generated strategies are obviously worthless (negative Sharpe, 1 trade, 0% win rate)
but still consume the full backtest budget. A fast vectorized pre-filter eliminates
those in <1 second **before** the Freqtrade call.

## Design Constraints

- **Zero agent changes**: the pre-filter is invisible to strategist, orchestrator, etc.
  They call `run_backtest()` as before and get back a result or a `pre_filter_rejected` signal.
- **Identical indicator logic**: each SignalFactory function uses the **same TA-Lib calls**
  with the **same parameters** as the corresponding Freqtrade strategy template in
  `backtesting/engine.py`. If they diverge, the pre-filter becomes misleading.
- **Deliberately loose thresholds**: the filter catches extreme cases (Sharpe < 0, 0% WR, < 3 trades).
  It is NOT a substitute for the OOS gate or walk-forward validation.
- **TA-Lib returns numpy arrays**: each TA-Lib result is wrapped in `pd.Series()` immediately
  so that `.shift(1)` and boolean masking work identically to the template's pandas DataFrame.
- **ASCII only** throughout.

## Architecture

```
StrategistAgent
    │  generate_strategy(params)
    ▼
BacktestEngine.run_backtest(strategy_type, params)
    │
    ├── [NEW] _run_prefilter()          < 1 second
    │       │
    │       ├── SignalFactory.generate(df, type, params) → signals Series
    │       └── FastMetrics.compute(df, signals) → metrics dict
    │
    ├── fail ──► return pre_filter_rejected result (early return)
    │
    └── pass ──► [existing] generate Freqtrade strategy file, subprocess, parse results
```

## Module: `backtesting/signal_factory.py`

### SignalFactory

Static-method class with a `REGISTRY` dict mapping strategy type → generator function.

```python
class SignalFactory:
    REGISTRY = { ... }  # 11 entries

    @staticmethod
    def generate(df: pd.DataFrame, strategy_type: str, params: dict) -> pd.Series:
        fn = SignalFactory.REGISTRY[strategy_type]
        return fn(df, params)
```

Each `_signal_*` function follows the same pattern:

1. Compute indicators using **TA-Lib with same parameters as the template**
2. Wrap each TA-Lib result in `pd.Series(array)` immediately
3. Build entry condition (same boolean logic as template entry_condition)
4. Build exit condition (same boolean logic as template exit_condition)
5. Combine into signal series: `1` on entry, `-1` on exit, `0` otherwise
6. Forward-fill positions: once entered, stay in until exit fires

#### Strategy type → function mapping (all 11)

| Type | Indicators (TA-Lib) | Entry | Exit |
|------|-------------------|-------|------|
| `sma_crossover` | SMA(fast), SMA(slow) | fast cross above slow | fast cross below slow |
| `macd_crossover` | MACD(fast,slow,signal) | hist crosses 0↑ | hist crosses 0↓ |
| `rsi_oversold` | RSI(period) | RSI < buy_threshold & prev >= buy_threshold | RSI > sell_threshold & prev <= sell_threshold |
| `bollinger_bands` | BBANDS(period, 2std) | close < lower & prev >= lower | close > upper & prev <= upper |
| `combined_sma_rsi` | SMA(fast), SMA(slow), RSI(14) | fast cross above slow + RSI 30-70 | fast cross below slow |
| `momentum` | ROC(10), SMA(volume,20), RSI(14) | ROC > 2%, volume > 1.5×MA, RSI 50-75 | ROC < 0 or RSI > 75 |
| `breakout` | rolling(20).max(), SMA(volume,20), ATR(14) | close > 20d high, volume > 1.3×MA | close < high - 2×ATR |
| `mean_reversion` | BBANDS(20,2std), RSI(14) | close < lower, RSI < 35, dist < -2% | close > middle or RSI > 60 |
| `volatility_squeeze` | BBANDS(20,2std), MACD | BB width near 120d min + MACD > signal | BB width > 3×min or MACD < signal |
| `sentiment_driven` | RSI(14), SMA(50) | RSI < 40 & close > SMA50 | RSI > 65 or close < SMA50 |
| `multi_timeframe` | SMA(20,50,80,200), RSI(14), ADX(14) | SMA20×50 cross, close > SMA200, ADX > 20, RSI 40-70 | SMA20×50 cross down or close < SMA200 |

### FastMetrics

```python
class FastMetrics:
    @staticmethod
    def compute(df: pd.DataFrame, signals: pd.Series, portfolio_value: float = 10000) -> dict:
```

Returns:
- `sharpe_ratio`: annualized from trade return series (using daily risk-free ≈ 0)
- `win_rate`: fraction of closed trades with positive return
- `max_drawdown`: peak-to-trough of cumulative equity
- `total_trades`: number of completed entry→exit cycles
- `total_return_pct`: net return as % of portfolio
- `passed`: boolean — all thresholds met

## Integration: `BacktestEngine.run_backtest()`

A single block at the top of `run_backtest()`:

```python
if settings.VECTORBT_PREFILTER_ENABLED:
    result = self._run_prefilter(strategy_type, strategy_params, ...)
    if not result["passed"]:
        logger.info("Pre-filter rejected %s: Sharpe=%.2f WR=%.1f%% trades=%d",
                     strategy_type, result["sharpe_ratio"],
                     result["win_rate"] * 100, result["total_trades"])
        return result  # early return, no Freqtrade call
```

`_run_prefilter()`:
1. Fetches OHLCV for timerange (or uses provided df)
2. Calls `SignalFactory.generate(df, type, params)` → signals
3. Calls `FastMetrics.compute(df, signals)` → metrics
4. Returns dict with `passed: bool` + metrics

The result dict has the same shape as a normal backtest result
so consumers don't need special handling:
```python
{
    "pre_filter_rejected": True,
    "sharpe_ratio": 0.12,
    "win_rate": 0.33,
    "total_trades": 2,
    "max_drawdown": 0.08,
    ...
}
```

## Configuration (`config.py`)

```python
VECTORBT_PREFILTER_ENABLED: bool = True
VECTORBT_PREFILTER_MIN_SHARPE: float = 0.5
VECTORBT_PREFILTER_MIN_WIN_RATE: float = 0.40
VECTORBT_PREFILTER_MIN_TRADES: int = 3
```

Thresholds are deliberately loose — the filter catches only
extreme-edge cases. Not a replacement for proper validation.

## Error Handling

- Unknown strategy type: log warning, pass through (don't block)
- No data / empty df: pass through (data issue, not strategy issue)
- TA-Lib exception on any single indicator: log, pass through
- FastMetrics on < 2 trades: returns neutral metrics (passes through)

Fail-open philosophy: the pre-filter is an optimization, not a safety gate.
If anything goes wrong, let the strategy through to Freqtrade.

## Testing

All functions are pure (df + params → metrics). Test file: `test_signal_factory.py`:

- Each of 11 strategy types: signal shape, entry/exit alignment, known-answer tests
- FastMetrics: known return series → known Sharpe, WR, DD
- Integration: `BacktestEngine._run_prefilter()` with mocked fetch
- Edge cases: empty df, unknown type, params missing keys, all-hold signals
- Threshold compliance: borderline strategies that should/shouldn't pass
- TA-Lib ndarray wrapping: verify `.shift(1)` works after pd.Series() wrapping
