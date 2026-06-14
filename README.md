# crypto_agent_bot

Autonomous crypto trading research system using **LangGraph ReAct agents**, **Freqtrade** backtesting, **ChromaDB** strategy memory, and a real-time **WebSocket-streamed dashboard**.

---

## Quick Start

```bash
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env   # edit with your API key
python backtesting/setup_ft.py
python main.py --demo  # full pipeline demo
```

## Modes

| Command | What it does |
|---------|-------------|
| `python main.py --ui` | Web dashboard at `http://127.0.0.1:8765` |
| `python main.py --autonomous` | Headless self-directing research loop (30 min interval) |
| `python main.py --autonomous --ui` | Autonomous + dashboard |
| `python main.py --auto-research "BTC momentum"` | One-shot research goal |
| `python main.py --demo` | UI + demo goal auto-started |
| `python main.py run` | Interactive CLI workspace |
| `python -m pytest -q` | 792 tests, 56 test files |

---

## Architecture

### Six LangGraph ReAct Agents

| Agent | Module | Role |
|-------|--------|------|
| **Strategist** | `agents/strategist.py` | Designs strategies from 11 predefined types + LLM reasoning |
| **Backtester** | `agents/backtester.py` | Runs Freqtrade via subprocess, bypasses LLM for `backtest` commands |
| **Curator** | `agents/curator.py` | Evaluates results, stores winners to ChromaDB |
| **Iteration Tracker** | `agents/iteration_tracker.py` | Convergence detection, strategy decay monitoring |
| **Risk Manager** | `agents/risk_manager.py` | Kelly Criterion sizing, circuit breaker checks |
| **Researcher** | `agents/researcher.py` | Web search (SearXNG → Tavily → DDG), content scraping |

### Orchestrator (`orchestration/hermes.py`)

Runs a Kanban **TaskBoard** with a **LangGraph state graph**. The cycle:

```
Strategist generates → Backtester tests → Curator evaluates → IterationTracker checks convergence
                                                                  ↓
                                                   Converged? → Yes → strategy deployed
                                                   No → Strategist retries with past context
```

The **AutonomousResearchLoop** (`orchestration/autonomous_loop.py`) wraps this in a self-directing outer loop: monitor market regime → detect strategy decay → spawn research goal → run → repeat (every 30 min).

### Strategy Generation Lifecycle

1. **Strategist** picks from 11 types (`sma_crossover`, `macd_crossover`, `rsi_oversold`, `bollinger_bands`, `combined_sma_rsi`, `momentum`, `breakout`, `mean_reversion`, `volatility_squeeze`, `sentiment_driven`, `multi_timeframe`), guided by market regime (trending/ranging/volatile) and past winners from ChromaDB
2. **BacktestEngine** runs a vectorized pre-filter (`SignalFactory`, pandas, <1s) then a full Freqtrade subprocess
3. **Anti-overfitting pipeline**: CPCVValidator (Combinatorial Purged CV) → OOSValidator → WalkForwardValidator → BlindSearch → SyntheticValidator
4. **Strategy evaluated** against: Sharpe ≥ 0.8, win rate ≥ 40%, max drawdown < 15%, trades ≥ 5

### Execution

- **Default**: paper trading (`execution/paper_trader.py`)
- **Live**: Kraken Pro via CCXT (`execution/live_executor.py`) with graduated validation (paper → small live → full)
- **Risk**: Kelly Criterion (3 profiles), two-layer Circuit Breaker (drawdown + anomaly), portfolio VaR
- **Quality gating**: RandomForest classifier scores trade quality from backtest metrics

### Web UI

Single-file React app (`ui/index.html`) — **Babel standalone, no build step** (ADR-007). FastAPI server at port 8765 with 18 REST endpoints + WebSocket at `/ws/autonomous` (19 event types). Components: candlestick chart, 4 SVG gauges, Sharpe history chart, agent feed, iteration table, terminal log panel.

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — all modes |
| `config.py` | Settings from `.env` |
| `orchestration/hermes.py` | Main orchestrator |
| `orchestration/autonomous_loop.py` | Self-directing research loop |
| `orchestration/graph.py` | LangGraph state graph |
| `backtesting/engine.py` | Freqtrade subprocess wrapper |
| `backtesting/signal_factory.py` | Vectorized pre-filter |
| `backtesting/cpcv_validator.py` | Anti-overfitting validator |
| `execution/live_executor.py` | Kraken Pro order routing |
| `execution/paper_trader.py` | Paper trading simulation |
| `api/server.py` | FastAPI + WebSocket server |
| `api/event_bus.py` | Async event pub/sub |
| `data/fetcher.py` | CCXT market data |
| `data/database.py` | SQLite (singleton, WAL mode) |
| `memory/vector_store.py` | ChromaDB strategy memory |
| `state/state_broker.py` | Inter-agent state sharing |
| `state/circuit_breaker.py` | Trading safety halt |
| `risk/kelly.py` | Position sizing |
| `docs/adr/` | 10 Architecture Decision Records |
| `CONTEXT.md` | Domain context and ubiquitous language |
| `graphify-out/graph.html` | Interactive knowledge graph (3,410 nodes) |

---

## Data Sources

| Source | Module | Purpose |
|--------|--------|---------|
| CCXT | `data/fetcher.py` | Primary OHLCV + order book (Binance/Bybit/Kraken) |
| CoinCap v3 | `data/coincap_fetcher.py` | Backup price feed (disabled by default) |
| CryptoPanic | `data/sentiment.py` | News sentiment |
| Santiment | `data/santiment_fetcher.py` | Social volume + dev activity (throttled 60s) |
| Fear & Greed | `data/fear_greed.py` | Market sentiment index |
| CoinGecko | `agents/researcher.py` | Asset fundamentals (no key needed) |
| SearXNG | `agents/researcher.py` | Self-hosted web search (bypasses DDG rate limits) |
| Tavily | `agents/researcher.py` | Web search fallback |

---

## Configuration (.env)

| Variable | Default | Notes |
|----------|---------|-------|
| `OPENAI_API_KEY` | — | Required. Works with DeepSeek, OpenAI, OpenRouter |
| `OPENAI_BASE_URL` | `https://api.deepseek.com/v1` | API base URL |
| `LLM_MODEL` | `deepseek-chat` | Model name |
| `EXCHANGE_ID` | `binance` | CCXT exchange ID |
| `SYMBOL` | `BTC/USDT` | Trading pair |
| `TIMEFRAME` | `1h` | Candlestick interval |
| `EXECUTION_MODE` | `paper` | `paper` or `live` |
| `AUTONOMOUS_INTERVAL_MINUTES` | `30` | Research loop interval |
| `TAVILY_API_KEY` | — | Optional Tavily web search key |
| `COINCAP_API_KEY` | — | Optional CoinCap key |
| `SANTIMENT_API_KEY` | — | Optional Santiment key |
| `SEARXNG_URL` | `http://localhost:4000` | Self-hosted SearXNG |

---

## Storage

| System | Module | Tables/Collections |
|--------|--------|-------------------|
| SQLite | `data/database.py` | strategies, backtests, trades, goals, metrics (WAL mode, singleton) |
| ChromaDB | `memory/vector_store.py` | strategy memory, search cache, agent memory |
| StateBroker | `state/state_broker.py` | In-memory key-value + pub/sub (optional Redis) |

---

## Architecture Decisions (ADRs)

| ADR | Decision |
|-----|----------|
| 001 | Multi-agent LangGraph architecture |
| 002 | SignalFactory must match Freqtrade TA-Lib calls (prevent signal mismatch) |
| 003 | SQLite singleton via `__new__` (prevent connection conflicts) |
| 004 | EventBus + StateBroker for inter-agent communication |
| 005 | Kelly Criterion with 3 risk profiles |
| 006 | CPCV for anti-overfitting (11-gate deployment pipeline) |
| 007 | Babel standalone, no build step |
| 008 | Two-layer circuit breaker (drawdown + anomaly) |
| 009 | Windows async/encoding compatibility |
| 010 | Kraken Pro as primary execution exchange |

---

## Knowledge Graph

`graphify-out/graph.html` — interactive graph of the codebase (3,410 nodes, 5,997 edges, 262 communities). Open in a browser to explore. Regenerate with `/graphify .` (requires OpenClaude).
