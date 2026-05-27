"""RiskManagerAgent — evaluates strategy risk and produces go/no-go recommendations."""

import json
import logging
from typing import Optional
import numpy as np
from langchain_core.tools import Tool
from agents.base import BaseAgent
from data.fetcher import MarketDataFetcher

logger = logging.getLogger(__name__)

RISK_MANAGER_PROMPT = """You are a risk management specialist. Your job:
1. Assess strategy risk using the provided tools
2. Compute value-at-risk from recent market data
3. Check position sizing against max drawdown
4. Assess current market volatility
5. Produce a clear go/no-go recommendation

Be conservative. Always favour capital preservation.
IMPORTANT: Use ONLY plain ASCII text. No emoji, no Unicode symbols."""


def _normalize(symbol: str) -> str:
    s = symbol.strip().upper()
    if "/" not in s:
        s = s + "/USDT"
    return s


class RiskManagerAgent(BaseAgent):
    """Evaluates trading strategies for risk and produces a risk report."""

    def __init__(self, fetcher: Optional[MarketDataFetcher] = None):
        self._fetcher = fetcher or MarketDataFetcher()
        tools = self._build_tools()
        super().__init__(name="risk_manager", tools=tools, system_prompt=RISK_MANAGER_PROMPT)

    def _build_tools(self):
        def _safe_fetch(symbol, limit):
            try:
                return self._fetcher.fetch_ohlcv(symbol, timeframe="1h", limit=limit)
            except Exception:
                logger.warning("Failed to fetch %s, falling back to BTC/USDT", symbol)
                return self._fetcher.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=limit)

        def compute_var_fn(symbol_str: str = "BTC/USDT") -> str:
            symbol = _normalize(symbol_str)
            df = _safe_fetch(symbol, 200)
            returns = df["close"].pct_change().dropna()
            if len(returns) == 0:
                return "VaR: N/A (no data)"
            var = float(np.percentile(returns, 5))
            return f"VaR (95%): {var:.4f} ({var*100:.2f}% max expected hourly loss)"

        def assess_volatility_fn(symbol_str: str = "BTC/USDT") -> str:
            symbol = _normalize(symbol_str)
            df = _safe_fetch(symbol, 168)
            returns = df["close"].pct_change().dropna()
            if len(returns) == 0:
                return "Volatility: N/A (no data)"
            vol = float(returns.std())
            ann = vol * (365 * 24) ** 0.5
            verdict = "HIGH" if ann > 0.8 else "MODERATE" if ann > 0.4 else "LOW"
            return f"Vol: {vol:.4f} (ann: {ann:.2f}) - {verdict}"

        def assess_drawdown_fn(dd_str: str = "0.05") -> str:
            """Compare max drawdown vs risk tolerance.
            Pass JSON: {"max_drawdown":0.03,"tolerance":0.05} or just the drawdown value."""
            try:
                p = json.loads(dd_str) if dd_str.strip().startswith("{") else {"max_drawdown": float(dd_str)}
            except (json.JSONDecodeError, ValueError):
                p = {}
            dd = float(p.get("max_drawdown", 0.05))
            tol = float(p.get("tolerance", 0.05))
            v = "ACCEPTABLE" if dd <= tol else f"EXCEEDS tolerance by {dd/tol:.1f}x"
            return f"Drawdown: {dd:.2%} vs tolerance {tol:.2%} - {v}"

        def risk_report_fn(metrics_json: str = "{}") -> str:
            """Generate a risk report. Pass JSON with: sharpe, win_rate, max_drawdown, tolerance."""
            try:
                m = json.loads(metrics_json)
            except json.JSONDecodeError:
                return "Error: pass JSON with sharpe, win_rate, max_drawdown"
            sharpe = float(m.get("sharpe", m.get("sharpe_ratio", 0)))
            wr = float(m.get("win_rate", m.get("winrate", 0)))
            dd = float(m.get("max_drawdown", m.get("drawdown", 0.05)))
            tol = float(m.get("tolerance", 0.05))
            symbol = m.get("symbol", "BTC/USDT")

            issues = []
            if sharpe < 1: issues.append(f"Low Sharpe ({sharpe:.2f})")
            if dd > tol: issues.append(f"Drawdown ({dd:.2%}) exceeds tolerance ({tol:.2%})")
            if wr < 0.4: issues.append(f"Low win rate ({wr:.0%})")

            verdict = "NO-GO" if len(issues) >= 2 else "CONDITIONAL" if issues else "GO"
            lines = [
                f"Risk Report for {symbol}",
                f"  Sharpe: {sharpe:.2f}",
                f"  Win rate: {wr:.0%}",
                f"  Max drawdown: {dd:.2%}",
                f"  Tolerance: {tol:.2%}",
            ]
            for i in issues:
                lines.append(f"  Issue: {i}")
            lines.append(f"  Recommendation: {verdict}")
            return "\n".join(lines)

        return [
            Tool(name="compute_value_at_risk", func=compute_var_fn, description="Compute VaR. Args: symbol string"),
            Tool(name="assess_volatility", func=assess_volatility_fn, description="Assess volatility. Args: symbol string"),
            Tool(name="assess_drawdown", func=assess_drawdown_fn, description="Compare drawdown vs tolerance. Args: decimal or JSON"),
            Tool(name="risk_report", func=risk_report_fn, description="Generate risk report. Args: JSON with sharpe, win_rate, max_drawdown, tolerance"),
        ]