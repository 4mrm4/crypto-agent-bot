"""
TradeQualityScorer — lightweight ML-based trade quality filter.

Learns from historical backtest experiments which strategy+regime+metric
combinations produce positive outcomes. Scores each incoming trade signal
and adjusts position sizing via a quality multiplier.

Cold-start safe: returns quality=1.0 (no-op) when insufficient training data.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

MODEL_PATH = Path("./workspace/quality_scorer.pkl")

# Minimum samples before the model activates (cold-start threshold)
MIN_TRAINING_SAMPLES = 30

# Quality score thresholds for sizing multiplier
HIGH_QUALITY_THRESHOLD = 0.70   # >= 0.70 -> full multiplier (1.0)
BLOCK_THRESHOLD = 0.40          # < 0.40 -> block trade (multiplier 0.0)

# Retrain after this many new trades
RETRAIN_INTERVAL_TRADES = 25

# Known strategy types and regimes for deterministic one-hot encoding
KNOWN_STRATEGY_TYPES = [
    "momentum", "mean_reversion", "rsi_oversold", "sma_crossover",
    "macd_crossover", "bollinger_bands", "breakout", "volatility_squeeze",
    "trend_following", "grid", "custom",
]
KNOWN_REGIMES = [
    "strong_uptrend", "uptrend", "ranging", "volatile",
    "downtrend", "strong_downtrend", "unknown",
]


@dataclass
class QualityPrediction:
    """Result of a single quality prediction."""
    quality_score: float
    quality_multiplier: float
    n_training_samples: int
    model_trained: bool
    blocked: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "quality_score": self.quality_score,
            "quality_multiplier": self.quality_multiplier,
            "n_training_samples": self.n_training_samples,
            "model_trained": self.model_trained,
            "blocked": self.blocked,
            "reason": self.reason,
        }


class TradeQualityScorer:
    """ML-based trade quality scorer with cold-start passthrough.

    Usage:
        scorer = TradeQualityScorer()
        scorer.train()  # load historical data and fit
        prediction = scorer.predict_quality(signal, backtest_metrics)
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self._model_path = model_path
        self._model: Optional[RandomForestClassifier] = None
        self._feature_names: List[str] = []
        self._n_samples = 0
        self._trained_at: Optional[str] = None
        self._trades_since_retrain = 0
        self._load_model()

    # -- Public API --

    def train(self, force: bool = False) -> bool:
        """Load training data and fit the model.

        Returns True if training occurred, False if insufficient data.
        Pass force=True to override MIN_TRAINING_SAMPLES check.
        """
        X, y, feature_names = self._load_training_data()
        if len(y) < (0 if force else MIN_TRAINING_SAMPLES):
            logger.info(
                "ML scorer: %d samples < %d minimum — staying in cold-start mode",
                len(y), MIN_TRAINING_SAMPLES,
            )
            return False

        logger.info("ML scorer: training on %d samples, %d features", len(y), X.shape[1])

        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        )
        self._model.fit(X, y)
        self._feature_names = feature_names
        self._n_samples = len(y)
        self._trained_at = datetime.utcnow().isoformat()
        self._trades_since_retrain = 0

        self._save_model()
        logger.info("ML scorer: training complete (accuracy=%.3f)", self._model.score(X, y))
        return True

    def predict_quality(
        self,
        signal: Any,
        backtest_metrics: Optional[Dict[str, float]] = None,
    ) -> QualityPrediction:
        """Score a trade signal's quality.

        Args:
            signal: Object with fields: strategy_type, regime, confidence, sharpe,
                    win_rate, max_drawdown (or TradeSignal-like duck-typed object).
            backtest_metrics: Optional dict with additional metrics
                              (profit_factor, total_trades, oos_passed).

        Returns:
            QualityPrediction with score, multiplier, and blocking decision.
        """
        if self._model is None or self._n_samples < MIN_TRAINING_SAMPLES:
            return QualityPrediction(
                quality_score=1.0,
                quality_multiplier=1.0,
                n_training_samples=self._n_samples,
                model_trained=self._model is not None,
                blocked=False,
                reason="cold_start",
            )

        features = self._build_feature_vector(signal, backtest_metrics or {})
        proba = self._model.predict_proba(features.reshape(1, -1))
        # Probability of class 1 (positive outcome)
        quality_score = float(proba[0, 1]) if proba.shape[1] > 1 else 0.5

        multiplier = self._score_to_multiplier(quality_score)
        blocked = multiplier == 0.0

        return QualityPrediction(
            quality_score=round(quality_score, 4),
            quality_multiplier=multiplier,
            n_training_samples=self._n_samples,
            model_trained=True,
            blocked=blocked,
            reason="blocked" if blocked else "allowed",
        )

    def record_trade_executed(self) -> None:
        """Increment the retrain counter. Call after each trade execution."""
        self._trades_since_retrain += 1
        if self._trades_since_retrain >= RETRAIN_INTERVAL_TRADES:
            logger.info("ML scorer: %d trades since last retrain, triggering retrain", self._trades_since_retrain)
            self.train()

    def is_trained(self) -> bool:
        return self._model is not None and self._n_samples >= MIN_TRAINING_SAMPLES

    # -- Feature Engineering --

    def _load_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load experiments from SQLite and build feature matrix + labels.

        Returns (X, y, feature_names).
        """
        try:
            from data.database import TradingDatabase
            db = TradingDatabase()
            experiments = db.query_experiments_with_verdict(
                verdicts=["deployed", "promoted", "adopted", "discarded"],
                limit=5000,
            )
        except Exception as exc:
            logger.warning("ML scorer: cannot query experiments: %s", exc)
            experiments = []

        rows = []
        for exp in experiments:
            metrics_raw = exp.get("metrics", "{}")
            if isinstance(metrics_raw, str):
                try:
                    metrics = json.loads(metrics_raw)
                except (json.JSONDecodeError, TypeError):
                    metrics = {}
            else:
                metrics = metrics_raw if isinstance(metrics_raw, dict) else {}

            verdict = exp.get("verdict", "discarded")
            label = 1 if verdict in ("deployed", "promoted", "adopted") else 0

            rows.append({
                "strategy_type": exp.get("strategy_type", "custom"),
                "regime": exp.get("regime", "unknown"),
                "sharpe": float(metrics.get("sharpe", 0)),
                "win_rate": float(metrics.get("win_rate", 0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0)),
                "profit_factor": float(metrics.get("profit_factor", 1.0)),
                "total_trades": int(metrics.get("total_trades", 0)),
                "oos_passed": 1 if exp.get("walk_forward_passed") else 0,
                "label": label,
            })

        if not rows:
            return np.empty((0, 1)), np.array([]), []

        feature_names, feature_rows = self._build_feature_matrix(rows)
        y = np.array([r["label"] for r in rows])
        return np.array(feature_rows), y, feature_names

    def _build_feature_matrix(self, rows: List[dict]) -> Tuple[List[str], List[List[float]]]:
        """Build a feature matrix from training rows. Returns (feature_names, matrix_rows)."""
        known_stypes = KNOWN_STRATEGY_TYPES
        known_regimes = KNOWN_REGIMES

        feature_names = []
        # One-hot: strategy_type
        for st in known_stypes:
            feature_names.append(f"strategy_type__{st}")
        # One-hot: regime
        for r in known_regimes:
            feature_names.append(f"regime__{r}")
        # One-hot: strategy_type x regime interaction
        for st in known_stypes:
            for r in known_regimes:
                feature_names.append(f"st_x_reg__{st}__{r}")
        # Numeric features
        feature_names += ["sharpe", "win_rate", "max_drawdown_neg", "profit_factor", "log_total_trades", "oos_passed"]

        matrix = []
        for row in rows:
            vec = []
            st = row.get("strategy_type", "custom")
            reg = row.get("regime", "unknown")

            # strategy_type one-hot
            for known_st in known_stypes:
                vec.append(1.0 if st == known_st else 0.0)

            # regime one-hot
            for known_r in known_regimes:
                vec.append(1.0 if reg == known_r else 0.0)

            # interaction one-hot
            for known_st in known_stypes:
                for known_r in known_regimes:
                    vec.append(1.0 if st == known_st and reg == known_r else 0.0)

            # numeric features
            vec.append(float(row.get("sharpe", 0)))
            vec.append(float(row.get("win_rate", 0)))
            vec.append(-float(row.get("max_drawdown", 0)))  # negate: lower dd is better
            vec.append(min(float(row.get("profit_factor", 1.0)), 10.0))  # cap at 10x
            vec.append(np.log1p(float(row.get("total_trades", 0))))  # log-transform
            vec.append(float(row.get("oos_passed", 0)))

            matrix.append(vec)

        return feature_names, matrix

    def _build_feature_vector(self, signal: Any, backtest_metrics: Dict[str, float]) -> np.ndarray:
        """Build a single feature vector from a signal for prediction.

        Handles both object attribute access and dict-like access.
        """
        def _get(obj, key: str, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        st = _get(signal, "strategy_type", "custom") or "custom"
        reg = _get(signal, "regime", "unknown") or "unknown"

        vec = []
        # strategy_type one-hot
        for known_st in KNOWN_STRATEGY_TYPES:
            vec.append(1.0 if st == known_st else 0.0)
        # regime one-hot
        for known_r in KNOWN_REGIMES:
            vec.append(1.0 if reg == known_r else 0.0)
        # interaction one-hot
        for known_st in KNOWN_STRATEGY_TYPES:
            for known_r in KNOWN_REGIMES:
                vec.append(1.0 if st == known_st and reg == known_r else 0.0)
        # numeric
        vec.append(float(backtest_metrics.get("sharpe", _get(signal, "sharpe", 0))))
        vec.append(float(backtest_metrics.get("win_rate", _get(signal, "win_rate", 0))))
        vec.append(-float(backtest_metrics.get("max_drawdown", _get(signal, "max_drawdown", 0))))
        vec.append(min(float(backtest_metrics.get("profit_factor", 1.0)), 10.0))
        vec.append(np.log1p(float(backtest_metrics.get("total_trades", 0))))
        vec.append(float(backtest_metrics.get("oos_passed", 0)))

        return np.array(vec, dtype=np.float64)

    @staticmethod
    def _score_to_multiplier(score: float) -> float:
        """Map raw quality score to position sizing multiplier."""
        if score >= HIGH_QUALITY_THRESHOLD:
            return 1.0
        if score < BLOCK_THRESHOLD:
            return 0.0
        return round(score, 4)

    # -- Persistence --

    def _save_model(self) -> None:
        """Persist model to disk."""
        data = {
            "model": self._model,
            "feature_names": self._feature_names,
            "n_samples": self._n_samples,
            "trained_at": self._trained_at,
        }
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("ML scorer: model saved to %s", self._model_path)

    def _load_model(self) -> bool:
        """Load model from disk if available. Returns True if loaded."""
        if not self._model_path.exists():
            return False
        try:
            with open(self._model_path, "rb") as f:
                data = pickle.load(f)
            self._model = data.get("model")
            self._feature_names = data.get("feature_names", [])
            self._n_samples = data.get("n_samples", 0)
            self._trained_at = data.get("trained_at")
            logger.info(
                "ML scorer: loaded model (%d samples, trained %s)",
                self._n_samples, self._trained_at,
            )
            return True
        except Exception as exc:
            logger.warning("ML scorer: failed to load model: %s", exc)
            return False
