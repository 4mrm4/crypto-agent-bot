# crypto_agent_bot

Modular crypto trading bot with 7 LangGraph ReAct agents, Freqtrade backtesting, ChromaDB strategy memory, and a real-time Web UI.

**Latest updates (June 2026 — v10: Batch 2 + CPCV + Pre-filter + State Broker + Agent Split):**

### Core Infrastructure
- **DeepSeek Chat API** — migrated from OpenRouter to direct DeepSeek API (`api.deepseek.com/v1`, model `deepseek-chat`)
- **Live token usage** — real-time token counter in Web UI showing prompt/completion/total per run
- **EventBus WebSocket streaming** — `monkey_patch_hermes` for auto-research mode, real-time agent activity pushed to UI
- **Auto data download** — checks BTC/USDT 1h data ≥500KB on startup, auto-downloads latest 2 years if missing

### Anti-Overfitting System (v8, continued)

### Transaction Cost Realism (`backtesting/engine.py`)
- **TransactionCostModel** — dataclass with maker fee (0.1%), taker fee (0.075%), slippage (0.05%) derived from config
- **Fee injection** — `--fee` flag passed to every Freqtrade subprocess call; fee baked into `config.fee`
- **Net-of-costs metrics** — `net_sharpe_ratio` computed in parsed results using cost drag estimation
- **OOSValidator pass criteria** — now uses `net_sharpe` (after cost model), not gross Sharpe. Strategies must clear 0.8 after costs
- **PerformanceMonitor thresholds** — tightened to 20-40% Sharpe degradation (costs already modelled)
- **BACKTEST_OPTIMISM_FACTOR** — no longer hardcoded at 0.55; imported from `config.settings`
- Config vars: `MAKER_FEE`, `TAKER_FEE`, `SLIPPAGE_PCT`, `SLIPPAGE_MODEL`

### Web Search Upgrade (`agents/researcher.py`)
- **Tavily primary search** — when `TAVILY_API_KEY` is set and `TAVILY_ENABLED=true`, uses Tavily API (advanced search depth)
- **DuckDuckGo fallback** — falls back automatically when Tavily is uninstalled, disabled, or errors
- **Result caching** — search results stored in ChromaDB (`search_cache` collection) to avoid redundant API calls
- Config vars: `TAVILY_API_KEY`, `TAVILY_ENABLED`

### SQLite Database (`data/database.py`)
- **TradingDatabase** — SQLite-backed persistent storage replacing JSONL as primary data store
- **5 tables**: `trades`, `experiments`, `oos_results`, `pipeline_results`, `validation_trades` — all with indexed columns
- **WAL mode** — concurrent read/write safe, no corruption on crash
- **Context manager** — `with db.transaction() as conn:` for safe, atomic transactions
- **Migration** — `migrate_jsonl()` reads existing JSONL files and populates SQLite (idempotent)
- **Dual write** — all 5 modules write to both SQLite and JSONL; JSONL writes disabled via `LEGACY_JSONL_BACKUP=false`
- Config var: `LEGACY_JSONL_BACKUP`
- Single file: `workspace/trading.db`
- **Hard data holdout** (`backtesting/data_split.py`) — frozen `DataSplitConfig` singleton defines research window (2017–2023) and holdout (2024–2026). All backtests raise `ValueError` on holdout overlap. Walk-forward validation stays strictly within research bounds.
- **Blind parameter search** (`backtesting/blind_search.py`) — 5-phase protocol: LLM defines search space blind → batch backtest → aggregate stats only → directional guidance → quantitative selection. No individual variant results leak to the LLM.
- **Out-of-sample validation** (`backtesting/oos_validator.py`) — `OOSValidator` runs on holdout data only. Results written to `oos_results.jsonl`, never to ChromaDB. Four thresholds: Sharpe≥0.8, WR≥0.42, DD≤0.15, trades≥10.
- **Synthetic data sanity** (`backtesting/synthetic_validator.py`) — random walk checker (max Sharpe 0.3 on noise) + Monte Carlo permutation test (p<0.05 for statistical significance).
- **Conservative Kelly sizing** (`agents/risk_manager.py`) — `PositionSizingTier` enum (VALIDATION 2% → CAUTIOUS 5% → NORMAL 10%). Bayesian Kelly using Beta(2,2) posterior with 90% CI lower bound as win rate. `kelly_position_size_conservative()` applies degradation haircut + Bayesian CI lower bound.
- **Validation mode** (`execution/validation_mode.py`) — 90-day conservative execution: 2% position cap, tight circuit breakers (-1.5% daily / -4% weekly), separate audit log. Requires Sharp≥0.6 + 50 trades for graduation.
- **11-gate deployment pipeline** (`orchestration/deployment_pipeline.py`) — 9 automated gates + 2 manual OOS gates. Gate 6 replaced with CPCV (combinatorial purged cross-validation). Tracks strategy state from `explored` → `promising` → `validated` → `pending_oos` → `deployable`.
- **Performance monitoring** (`monitoring/performance_monitor.py`) — statistical significance testing (≥30 trades required), expected degradation ranges, regime mismatch detection with 3-day suspension threshold.
- **ChromaDB contamination guard** (`memory/vector_store.py`, `agents/curator.py`) — strategies tagged with `discovered_on_window` metadata. Cross-window exclusion queries prevent data leakage between research cycles.

### Regime-Conditioned Gating (`execution/signal_scanner.py`)
- **RegimeTransitionTracker** — tracks regime changes with timestamps, 30-minute cooldown on transitions
- **Validated regimes** — each strategy lists which regimes it was validated in; gating blocks mismatched signals
- **Confidence gating** — low regime confidence raises signal floor from 0.6 to 0.75

### 5-Level Strategy Decay (`orchestration/strategy_manager.py`)
- HEALTHY (>=0.90), WARNING (>=0.75), DECAYING (>=0.50), CRITICAL (<0.50), RETIRED
- **Auto-recovery** — DECAYING->WARNING after 3+ improvements; CRITICAL->DECAYING on single improvement
- **Critical counter** — auto-retire after 5 consecutive critical evaluations

### Portfolio VaR (`risk/portfolio_var.py`)
- Variance-covariance VaR (95%/99%), historical simulation, marginal VaR per position
- Correlation matrix with high-correlation warnings (>0.7)
- Exposure limits: 50% total portfolio cap, 15% single position cap

### Telegram Alerter (`monitoring/telegram_alerter.py`)
- Inline approve/reject buttons on trade signals with 5-minute timeout
- Critical alerts forwarded from anomaly detector, circuit breaker, strategy retirement
- Event bus subscriber — automatic forwarding of anomaly_detected, circuit_breaker_halt events
- No-op without TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID

### Fast Pre-filter (`backtesting/signal_factory.py`)
- **SignalFactory** — 11 vectorized signal functions mirroring Freqtrade templates (same TA-Lib, same params)
- **FastMetrics** — vectorized Sharpe, win rate, max DD, trade count (<1s per strategy)
- Runs inside BacktestEngine before Freqtrade subprocess — zero agent changes
- Loose thresholds (Sharpe>=0.5, WR>=40%, trades>=3) — noise filter, not quality gate

### CPCV Validation (`backtesting/cpcv_validator.py`)
- **CPCVSplitter** — generates all C(n_folds, k_test) combinatorial train/test splits with purge + embargo
- **CPCVValidator** — evaluates strategy across all combinatorial paths via SignalFactory
- Paths with <5 trades = NaN (excluded); >30% NaN = validation fails
- Gate 6 in deployment pipeline; walk_forward_validate preserved for research loop

### State Broker (`state/state_broker.py`)
- Key-value store with TTL expiry + pub/sub for real-time event distribution
- In-memory by default, optional Redis backend via REDIS_URL
- Integrated: LiveExecutor (position state), SignalScanner (signal state), Hermes (system heartbeat)

### Strategist Agent Split
- **StrategistAgent** — 4 tools: strategy design only (generate, concepts, params, research)
- **BacktesterAgent** (new) — 7 tools: backtesting execution (run, hyperopt, WFV, blind search, compare, config, data)
- **IterationTrackerAgent** (new) — 4 tools: strategy memory (best, history, store result, store insight)
- 7 agents total, wired via HermesOrchestrator with keyword-based routing

### Smart Backtesting
- **Metrics parsing** — handles multiple Freqtrade output schemas with multi-field-name fallback; debug JSON dump on each run
- **Hyperopt support** — 6 strategy types have `IntParameter` declarations (`sma_crossover`, `combined_sma_rsi`, `macd_crossover`, `rsi_oversold`, `bollinger_bands`, `multi_timeframe`); `--spaces buy sell roi stoploss` flag
- **Walk-forward validation** — splits timerange into N windows, auto-detects data range, enforces 14-day minimum window
- **ExperimentTracker** (`orchestration/experiment_tracker.py`) — JSONL-backed experiment store with composite scoring (`Sharpe×0.5 + WR×0.3 + (1-DD×10)×0.2`); `suggest_next_params()` with type safety and valid range clamping
- **Type coercion guard** — every generated strategy's `populate_indicators()` auto-coerces string columns to numeric to prevent Arrow backend crashes

### Autonomous Research
- **`--auto-research`** mode — runs the full pipeline autonomously: regime detection → sentiment → web search → backtest → iterate → converge
- **Auto-research from Web UI** — submit goals starting with `Auto-research:` through the UI modal
- **Realistic convergence targets** — Sharpe ≥ 0.8, WR ≥ 40%, DD ≤ 15%, trades ≥ 5
- **Strategy concept library** (`data/strategy_concepts.py`) — 10 structured concepts with regime mapping, injected into LLM system prompt

### Researchers & Agents
- **Concept-aware strategy specs** — `generate_custom_strategy_spec` auto-maps concepts to strategy types using keyword matching
- **Strategy-relevant web search** — Tavily search (primary) with DuckDuckGo fallback, results scored by trading keyword relevance, cached in ChromaDB
- **Strategy memory** — ChromaDB stores backtest results (type, params, metrics, regime); past winners inform future LLM runs

### Web UI
- **6 metric cards** — Sharpe, Win Rate, Drawdown, Trades, Market Sentiment, **Token Usage** (live-updating)
- **Agent timeline** — real-time agent activity feed
- **Hypothesis display** — iteration counter with critique overlay
- **SVG Sharpe chart** — no external charting dependencies
- **Auto-scrolling event log** — WebSocket event stream with heartbeats

### Data Sources
- **Market data**: CCXT (Binance), OHLCV via `fetch_ohlcv()`
- **Sentiment**: Fear & Greed Index (alternative.me), CoinGecko news (requires `COINGECKO_API_KEY`), CryptoPanic (requires `CRYPTOPANIC_API_KEY`)
- **Patterns**: 13 TA-Lib candlestick patterns (hammer, engulfing, morning star, etc.)
- **Regime**: ADX/ATR/SMA200 classification → strong_uptrend/downtrend, ranging, volatile, weak_trend
- **On-chain**: Whale Alert (requires `WHALE_ALERT_API_KEY`), CoinGecko volume proxy — gated by `ENABLE_ONCHAIN` flag

## Architecture

```mermaid
graph TD
    MAIN[main.py] --> UI[FastAPI Server]
    MAIN --> AUTO[AutonomousResearchLoop]
    MAIN --> HERMES[HermesOrchestrator]
    MAIN --> WS[VibeWorkspace CLI]

    HERMES --> GRAPH[LangGraph State Graph]
    GRAPH --> ANALYST[Analyst Agent]
    GRAPH --> STRATEGIST[Strategist Agent]
    GRAPH --> BT[Backtester Agent]
    GRAPH --> IT[IterationTracker Agent]
    GRAPH --> CURATOR[MemoryCurator Agent]
    GRAPH --> RISK[RiskManager Agent]
    GRAPH --> RESEARCHER[Researcher Agent]

    ANALYST --> FETCHER[MarketDataFetcher / CCXT]
    STRATEGIST --> ENGINE[BacktestEngine / Freqtrade]
    STRATEGIST --> TRACKER[ExperimentTracker]
    CURATOR --> MEMORY[VectorStore / ChromaDB]
    RESEARCHER --> WEB[Web Search / Tavily + DDG]
    RESEARCHER --> CONCEPTS[Strategy Concepts]
    HERMES -.-> UIWS[Web UI / WebSocket]

    ENGINE --> FREQ[Freqtrade Subprocess]
    ENGINE --> DATA[DataSplitConfig / Holdout Guard]
    ENGINE --> BLIND[BlindParameterSearch]
    ENGINE --> OOS[OOSValidator]
    ENGINE --> SYNTH[SyntheticValidator]

    AUTO --> REGIME[MarketRegimeDetector]
    AUTO --> SENT[SentimentFetcher]
    AUTO --> HERMES

    SC[SignalScanner] --> PE[LiveExecutor]
    SC --> REGIME
    PE --> EXCH[Exchange / CCXT]
    PE --> RM[RiskManager Agent]
    RM --> KELLY[Kelly Criterion]
    RM --> CB[CircuitBreaker]
    AD[AnomalyDetector] --> CB
    WS2[MarketDataStream] --> EXCH
    MEF[MultiExchangeFetcher] --> EXCH

    DP[DeploymentPipeline] --> ENGINE
    DP --> CPCV[CPCVValidator]
    DP --> OOS
    DP --> SYNTH
    ENGINE --> SF[SignalFactory Pre-filter]
    VM[ValidationMode] --> PE
    PM[PerformanceMonitor] --> RM
    AD --> TA[TelegramAlerter]
    PE --> SB[StateBroker]
    SC --> SB
    HERMES --> SB

    subgraph "AutoResearch Outer Loop"
        HYP[Generate Hypothesis] --> RESEARCH[Web Research]
        RESEARCH --> BACKTEST[Backtest Strategies]
        BACKTEST --> CRIT[Critique Results]
        CRIT --> CONV{Converged?}
        CONV -->|No| HYP
        CONV -->|Yes| DONE[Finish]
    end
```

## Components

| Component | File | Role |
|---|---|---|
| **MarketDataFetcher** | `data/fetcher.py` | Live OHLCV + price via CCXT |
| **BacktestEngine** | `backtesting/engine.py` | Strategy backtesting via Freqtrade, 11 strategy types + hyperopt + walk-forward |
| **AutoSetupData** | `backtesting/setup_data.py` | Auto-downloads historical data on startup |
| **VectorStore** | `memory/vector_store.py` | ChromaDB vector store with strategy memory (Sharpe-filtered, regime-aware) |
| **OnChainFetcher** | `data/onchain.py` | Whale Alert + CoinGecko volume proxy (gated) |
| **StrategyConcepts** | `data/strategy_concepts.py` | 10 structured concepts with regime mappings |
| **AnalystAgent** | `agents/analyst.py` | Market analysis with live data tools |
| **StrategistAgent** | `agents/strategist.py` | 4 tools: strategy design (generate, concepts, params, research) |
| **BacktesterAgent** | `agents/backtester.py` | 7 tools: backtesting execution (run, hyperopt, WFV, blind search, compare, config, data) |
| **IterationTrackerAgent** | `agents/iteration_tracker.py` | 4 tools: strategy memory (best, history, store result, store insight) |
| **ResearcherAgent** | `agents/researcher.py` | Web search (Tavily + DDG fallback), paper reading, concept-mapped strategy specs |
| **RiskManagerAgent** | `agents/risk_manager.py` | Risk metrics + go/no-go verdict |
| **MemoryCuratorAgent** | `agents/curator.py` | Cross-session memory retrieval from ChromaDB |
| **HermesOrchestrator** | `orchestration/hermes.py` | Multi-agent LangGraph orchestration + outer research loop |
| **LangGraph Graph** | `orchestration/graph.py` | State-machine orchestration with 4 cycles |
| **SignalFactory** | `backtesting/signal_factory.py` | 11 fast vectorized signal generators matching Freqtrade templates |
| **CPCVValidator** | `backtesting/cpcv_validator.py` | Combinatorial Purged Cross-Validation (C(n,k) paths) |
| **StateBroker** | `state/state_broker.py` | Shared key-value + pub/sub (in-memory or Redis) |
| **PortfolioVaR** | `risk/portfolio_var.py` | Covariance & historical VaR (95%/99%), marginal VaR |
| **TelegramAlerter** | `monitoring/telegram_alerter.py` | Human-in-the-loop alerts with inline approve/reject |
| **AutoResearch** | `orchestration/auto_research.py` | Autonomous pipeline: regime → sentiment → research → backtest → converge |
| **ExperimentTracker** | `orchestration/experiment_tracker.py` | JSONL-backed experiment store with composite scoring |
| **ResearchIteration** | `orchestration/research.py` | Hypothesis/critique dataclass with convergence check |
| **VibeWorkspace** | `workspace/vibe.py` | Rich CLI research workspace |
| **PaperTrader** | `execution/paper_trader.py` | Dry-run live simulation |
| **FastAPI Server** | `api/server.py` | WebSocket streaming + REST endpoints |
| **EventBus** | `api/event_bus.py` | Async event streaming for Web UI |
| **Web UI** | `ui/index.html` | Single-file React dashboard with 6 metric cards |
| **TokenTracker** | `agents/token_tracker.py` | Thread-safe token usage accumulator with live UI counter |
| **SentimentFetcher** | `data/sentiment.py` | Fear & Greed + CoinGecko news (with API key support) |
| **PatternDetector** | `data/patterns.py` | 13 candlestick patterns via TA-Lib |
| **MarketRegimeDetector** | `data/regime.py` | ADX/ATR/SMA200 regime classification |
| **DataSplitConfig** | `backtesting/data_split.py` | Frozen singleton defining research/holdout split |
| **BlindParameterSearch** | `backtesting/blind_search.py` | Blind 5-phase parameter search |
| **OOSValidator** | `backtesting/oos_validator.py` | Holdout-only strategy validation |
| **SyntheticValidator** | `backtesting/synthetic_validator.py` | Random walk + permutation sanity checks |
| **DeploymentPipeline** | `orchestration/deployment_pipeline.py` | 11-gate strategy deployment gauntlet |
| **PerformanceMonitor** | `monitoring/performance_monitor.py` | Live vs backtest degradation tracking |
| **AnomalyDetector** | `monitoring/anomaly_detector.py` | 7-checks: rapid drawdown, stuck positions, API errors |
| **MarketDataStream** | `data/stream.py` | Live Binance WebSocket feeds |
| **MultiExchangeFetcher** | `data/fetcher.py` | Binance/Bybit best-price routing |
| **AutonomousResearchLoop** | `orchestration/autonomous_loop.py` | Self-directing research engine |
| **LiveExecutor** | `execution/live_executor.py` | Bridges approved strategies to exchange orders |
| **TradeSignal** | `execution/trade_signal.py` | Structured trade proposal with full provenance |
| **SignalScanner** | `execution/signal_scanner.py` | Scans pairs × approved strategies every 60s |
| **AuditLog** | `execution/audit_log.py` | Append-only JSONL log of every trade decision |
| **StrategyManager** | `orchestration/strategy_manager.py` | Strategy decay detection + auto-retire |
| **ValidationMode** | `execution/validation_mode.py` | 90-day conservative execution with tight CB |
| **PerformanceMonitor** | `monitoring/performance_monitor.py` | Live vs backtest degradation tracking |
| **TradingDatabase** | `data/database.py` | SQLite-backed storage (5 tables, WAL mode) replacing JSONL as primary store |

## Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
source venv/Scripts/activate   # Git Bash
venv\Scripts\activate          # Windows CMD

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env with your LLM API key (DeepSeek or OpenAI-compatible)
# The system uses DEEPSEEK by default: api.deepseek.com/v1, model deepseek-chat

# 5. Scaffold Freqtrade user data
python backtesting/setup_ft.py

# 6. Historical data downloads automatically on first run
```

## Usage

```bash
# Web UI dashboard
python main.py --ui

# Web UI with demo goal auto-started
python main.py --demo

# Autonomous research mode (CLI)
python main.py --auto-research "BTC momentum strategies 2026"

# Interactive workspace
python main.py run

# One-shot research goal
python main.py new-goal "Find optimal SMA crossover for BTC/USDT"
```

## v7 — Autonomous Trading System

> **Upgraded June 2026**: AutonomousResearchLoop + LiveExecutor + SignalScanner + full risk management

### New Architecture

```mermaid
graph TD
    AL[AutonomousResearchLoop] -->|"self-generates goals"| HERMES
    AL -->|"regime"| REGIME[MarketRegimeDetector]
    AL -->|"decay"| SM[StrategyManager]
    SC[SignalScanner] -->|"scans pairs"| PE[LiveExecutor]
    SC -->|"regime filter"| REGIME
    PE -->|"paper/live"| EXCH[Exchange/CCXT]
    PE -->|"audit"| ALG[AuditLog]
    PE -->|"risk gate"| RM[RiskManagerAgent]
    RM -->|"kelly"| KELLY[Kelly Criterion]
    RM -->|"circuit breaker"| CB[CircuitBreaker]
    AD[AnomalyDetector] -->|"7 checks"| CB
    WS[MarketDataStream] -->|"WebSocket"| EXCH
    MEF[MultiExchangeFetcher] -->|"fallback"| BYBIT[Bybit/OKX]
```

### New CLI Commands

```bash
# Fully autonomous mode (self-directing, no human goals needed)
python main.py --autonomous

# Autonomous mode with web UI dashboard
python main.py --autonomous --ui

# Legacy modes still work
python main.py --ui
python main.py --auto-research "BTC momentum strategies"
```

### Key New Components

| Component | File | Role |
|-----------|------|------|
| **AutonomousResearchLoop** | `orchestration/autonomous_loop.py` | Self-directing research engine |
| **RegimeSnapshot** | `data/regime.py` | Rich regime classification with confidence + strategy map |
| **RiskManagerAgent** | `agents/risk_manager.py` | Kelly sizing, correlation gate, circuit breaker, pre-trade approval |
| **CircuitBreaker** | `agents/risk_manager.py` | Global halt switch — stops all trading on critical conditions |
| **LiveExecutor** | `execution/live_executor.py` | Bridges approved strategies to exchange orders |
| **TradeSignal** | `execution/trade_signal.py` | Structured trade proposal with full provenance |
| **AuditLog** | `execution/audit_log.py` | Append-only JSONL log of every trade decision |
| **SignalScanner** | `execution/signal_scanner.py` | Scans all pairs × approved strategies every 60s |
| **MarketDataStream** | `data/stream.py` | Live Binance WebSocket feeds |
| **AnomalyDetector** | `monitoring/anomaly_detector.py` | 7 anomaly checks: rapid drawdown, stuck positions, API errors |
| **StrategyManager** | `orchestration/strategy_manager.py` | Strategy decay detection + auto-retire |
| **MultiExchangeFetcher** | `data/fetcher.py` | Best-price routing + Binance/Bybit fallback |
| **ResearchGoal** | `orchestration/research.py` | Structured goal dataclass |

### Safety Features

| Feature | Description | Threshold |
|---------|-------------|-----------|
| **Kelly Sizing** | Fractional Kelly (25% default) caps position size | Max 10% of portfolio per trade |
| **Correlation Gate** | Rejects over-concentrated positions | Max correlation 0.7 |
| **Circuit Breaker** | Halts all trading on critical conditions | Daily -3%, Weekly -8% |
| **Rapid Drawdown** | >2% in <10 minutes triggers halt | Automatic |
| **Signal Flood** | >10 signals in 60 seconds (likely bug) | Auto-halt |
| **Strategy Decay** | Auto-retires strategies with score < 0.5 | Live/backtest ratio |
| **Paper Mode** | Always starts in paper mode | `EXECUTION_MODE=paper` |

### New .env Variables

```ini
AUTONOMOUS_INTERVAL_MINUTES=30
DECAY_THRESHOLD=0.20
COVERAGE_GAP_SHARPE=0.80
MAX_DAYS_WITHOUT_REGIME_RESEARCH=7
EXECUTION_MODE=paper
LIVE_MAX_POSITION_USDT=100
TWAP_THRESHOLD_USDT=500
STOP_LOSS_DEFAULT=0.04
TAKE_PROFIT_DEFAULT=0.08
EXCHANGES=binance,bybit
FUTURES_ENABLED=false
```

### Auto-Research from Web UI
Submit a goal in the modal starting with `Auto-research:`:
```
Auto-research: BTC momentum with volume confirmation and ADX filter
```

The system will run the full pipeline autonomously — regime detection, sentiment analysis, web research, strategy generation, backtesting, and iteration until convergence.

## Web UI

```bash
python main.py --ui
```

Opens at `http://127.0.0.1:8765`. Features:
- **6 metric cards**: Sharpe Ratio, Win Rate, Max Drawdown, Total Trades, Market Sentiment, **Token Usage** (live)
- **Agent timeline**: real-time activity feed showing which agent is running
- **Hypothesis display**: current hypothesis with iteration counter
- **SVG Sharpe chart**: iteration history plotted as a line chart
- **Auto-scrolling event log**: complete WebSocket event stream
- **Run New Goal modal**: configure cycles, iterations, and goal text

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | LLM API key (DeepSeek or OpenAI-compatible) |
| `OPENAI_BASE_URL` | `https://api.deepseek.com/v1` | API base URL |
| `LLM_MODEL` | `deepseek-chat` | Model name |
| `EXCHANGE_ID` | `binance` | CCXT exchange |
| `SYMBOL` | `BTC/USDT` | Trading pair |
| `TIMEFRAME` | `1h` | Candle interval |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector store directory |
| `ENABLE_SENTIMENT` | `true` | Enable Fear & Greed / news sentiment |
| `ENABLE_PATTERNS` | `true` | Enable candlestick pattern detection |
| `ENABLE_ONCHAIN` | `false` | Enable on-chain data (requires Whale Alert key) |
| `CRYPTOPANIC_API_KEY` | — | Optional CryptoPanic news key |
| `COINGECKO_API_KEY` | — | Optional CoinGecko Pro API key (for news) |
| `HF_TOKEN` | — | Optional HuggingFace token (suppresses model download warning) |
| `WHALE_ALERT_API_KEY` | — | Optional Whale Alert key |
| `MAKER_FEE` | `0.001` | Maker fee rate (0.1%) for backtest cost model |
| `TAKER_FEE` | `0.00075` | Taker fee rate (0.075%) for backtest cost model |
| `SLIPPAGE_PCT` | `0.0005` | Slippage estimate (0.05%) per trade leg |
| `SLIPPAGE_MODEL` | `fixed` | Slippage model type (`fixed` or `volume_scaled`) |
| `TAVILY_API_KEY` | — | Optional Tavily API key for web search |
| `TAVILY_ENABLED` | `true` | Enable Tavily search (falls back to DuckDuckGo) |
| `LEGACY_JSONL_BACKUP` | `true` | Keep JSONL file writes alongside SQLite |

## Strategy Types

The strategist supports 11 strategy types:

| Type | Description | Indicators |
|---|---|---|
| `sma_crossover` | SMA fast/slow crossover | `ta.SMA` |
| `macd_crossover` | MACD signal line / histogram crossover | `ta.MACD` (parameterized) |
| `rsi_oversold` | RSI oversold/overbought | `ta.RSI` (parameterized thresholds) |
| `bollinger_bands` | Bollinger Bands lower/upper touch | `ta.BBANDS` (parameterized period) |
| `combined_sma_rsi` | SMA crossover + RSI filter | `ta.SMA`, `ta.RSI` |
| `momentum` | ROC + volume confirmation | `ta.ROC`, `ta.SMA`, `ta.RSI` |
| `breakout` | N-period high breakout with volume spike | rolling max, `ta.SMA`, `ta.ATR` |
| `mean_reversion` | BB + RSI oversold for ranging markets | `ta.BBANDS`, `ta.RSI` |
| `volatility_squeeze` | BB width contraction then MACD expansion | `ta.BBANDS`, `ta.MACD` |
| `sentiment_driven` | RSI + SMA when fear/greed < 30 | `ta.RSI`, `ta.SMA` |
| `multi_timeframe` | SMA20/50 crossover + SMA200 + ADX | `ta.SMA`, `ta.ADX` |

## Strategist Tools (13 total)

`generate_strategy`, `set_backtest_config`, `run_backtest`, `download_data`, `compare_strategies`, `interpret_metrics`, `suggest_next_params`, `get_best_strategy`, `get_iteration_history`, `get_research_history`, `run_hyperopt`, `walk_forward_validate`

## Running Tests

```bash
# Existing tests (v6 + v7)
python -m pytest test_phase2.py test_sentiment.py test_patterns.py test_regime.py -v
python test_experiment_tracker.py
python test_walk_forward.py

# Anti-overfitting tests (87 tests)
python -m pytest test_data_split.py test_blind_search.py test_oos_validator.py -v
python -m pytest test_deployment_pipeline.py test_kelly_conservative.py -v
python -m pytest test_synthetic_validator.py test_validation_mode.py -v
python -m pytest test_performance_monitor.py -v

# v9 — Transaction costs, Tavily, SQLite (38 tests)
python -m pytest test_transaction_costs.py test_tavily_search.py test_database.py -v

# All tests
python -m pytest test_data_split.py test_blind_search.py test_oos_validator.py \
  test_deployment_pipeline.py test_kelly_conservative.py \
  test_synthetic_validator.py test_validation_mode.py \
  test_performance_monitor.py test_transaction_costs.py \
  test_tavily_search.py test_database.py \
  test_phase2.py test_sentiment.py test_patterns.py test_regime.py -v
```

## How It Works

1. **Define a research goal** via the CLI or Web UI (or use `Auto-research:` prefix for full autonomous mode)
2. **AutoResearch** runs the outer loop: hypothesis → research → backtest → critique → repeat until convergence
3. **Market context** (sentiment, patterns, regime) is injected before each research cycle
4. **4-cycle LangGraph** processes: analysis → enhanced analysis → strategy generation/backtesting → risk assessment
5. **Strategist** uses ChromaDB strategy memory to learn from past winners; runs hyperopt / walk-forward validation
6. **ExperimentTracker** records every backtest with composite scoring for data-driven parameter suggestions
7. **Results** are stored in ChromaDB, workspace registry, and streamed live to the Web UI via WebSocket
8. **Convergence** is checked against realistic targets (Sharpe ≥ 0.8, WR ≥ 40%, DD ≤ 15%, trades ≥ 5)
