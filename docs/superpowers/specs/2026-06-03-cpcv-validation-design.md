# CPCV Validation — Combinatorial Purged Cross-Validation

## Motivation

Replace Gate 6 of the deployment pipeline (walk-forward validation) with
Combinatorial Purged Cross-Validation, a more rigorous robustness test
from Marcos Lopez de Prado's Advances in Financial ML.

Walk-forward tests 3 sequential train/test splits. CPCV tests all
C(n, k) combinatorial combinations of k test folds from n total folds,
producing a distribution of outcomes rather than a single path estimate.

## Architecture

```
CPCVSplitter (pure numpy)         CPCVValidator                 DeploymentPipeline
  - generates fold indices          - runs CPCV on strategy      - Gate 6 = CPCV
  - combinatorial combos             - uses SignalFactory         - replaces walk_forward_validate
  - purges label overlaps            - aggregates per-path stats  - walk_forward_validate
  - embargo after test window        - returns CPCVResult           stays in BacktestEngine
```

## Module: `backtesting/cpcv_validator.py`

### CPCVSplitter

```python
class CPCVSplitter:
    def __init__(self, n_folds=6, k_test=2, embargo_days=5):
        # n_folds total, k_test test folds per path
        # embargo_days gap after each test fold
```

- Generates all `C(n_folds, k_test)` combinations of test folds
- For each combination: maps fold indices back to sample positions
- Purges training samples whose (prediction) label spans overlap the test window
- Removes training samples within `embargo_days` after each test fold boundary

### CPCVValidator

```python
@dataclass
class CPCVResult:
    sharpe_distribution: List[float]
    sharpe_avg: float
    sharpe_std: float
    sharpe_lower_quartile: float   # Q1 — robust metric
    sharpe_upper_quartile: float   # Q3
    median_sharpe: float
    pct_positive_sharpe: float
    n_paths: int
    n_paths_valid: int             # paths with >= MIN_TRADES
    passed: bool
    reason: str

class CPCVValidator:
    MIN_TRADES_PER_PATH = 5
    MAX_NAN_RATIO = 0.30           # fail if > 30% of paths produce NaN Sharpe

    def validate(self, df, strategy_type, params, n_folds=6) -> CPCVResult:
        # 1. Split via CPCVSplitter
        # 2. For each path: SignalFactory.generate(train) → signals on test → FastMetrics
        # 3. Paths with < 5 trades → NaN Sharpe (excluded from distribution)
        # 4. If > 30% paths are NaN → fail
        # 5. Aggregate: mean, std, quartiles, % positive
        # 6. Pass if: median Sharpe > 0 AND Q1 > -0.5 AND > 50% paths positive
```

### Edge Cases

| Case | Handling |
|------|----------|
| Path with 0-4 trades | Sharpe = NaN, excluded from distribution |
| > 30% of paths NaN | Fail with "insufficient trades" reason |
| All paths NaN | Fail |
| Single fold only | Degenerate to single test split (not combinatorial) |
| n_folds = k_test | All folds = test, no training data → raise |

## Thresholds (deliberately modest — noise filter, not quality gate)

```python
MIN_MEDIAN_SHARPE = 0.0        # better than coin flip
MIN_LOWER_QUARTILE = -0.5      # worst path isn't catastrophic
MIN_PCT_POSITIVE = 0.50         # majority of paths have positive Sharpe
```

CPCV is a noise filter. Strict quality enforcement is Gate 10 (OOS validation).

## Pipeline Integration

In `DeploymentPipeline.run_full_pipeline()`:

```python
# Gate 6 (replaced walk-forward):
cpcv = CPCVValidator()
research_df = engine.fetch_data(timerange=DATA_SPLIT.research_timerange(), ...)
cpcv_result = cpcv.validate(research_df, strategy_type, best_params)
if not cpcv_result.passed:
    return PipelineResult(failed_at=6, reason=f"CPCV: {cpcv_result.reason}")
```

`walk_forward_validate()` remains on BacktestEngine for the autonomous research loop.

## Dependencies

- numpy (index manipulation, combinatorial generation)
- pandas (time series alignment)
- No external ML libraries
- Uses `backtesting.signal_factory.SignalFactory` and `FastMetrics`

## Testing

File: `test_cpcv_validator.py`

- CPCVSplitter: fold counts, combo generation, purge/embargo, edge cases
- CPCVResult: dataclass fields, pass/fail logic
- CPCVValidator: synthetic returns, NaN path handling, threshold compliance
- Pipeline integration: gate replacement returns correct failed_at
