"""Vector memory store using ChromaDB for persistent RAG-backed agent memory."""

import logging
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

# Lazy import so ChromaDB is only loaded when actually used
_embedding_function = None


def _get_embedding_function():
    """Return a sentence-transformer embedding function (no API key needed)."""
    global _embedding_function
    if _embedding_function is None:
        try:
            from chromadb.utils import embedding_functions
            _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("Loaded local embedding model: all-MiniLM-L6-v2")
        except Exception as exc:
            logger.warning("Could not load sentence-transformers, using default: %s", exc)
            _embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_function


class VectorStore:
    """ChromaDB-based persistent vector store for agent memories."""

    def __init__(self, collection_name: str = "agent_memory", persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_DB_PATH
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._init_db()

    def _init_db(self):
        """Initialise ChromaDB client and collection (persistent by default)."""
        import chromadb
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        embedding_fn = _get_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_fn,
        )
        logger.info(
            "ChromaDB ready: %s (%d docs)",
            self.collection_name,
            self._collection.count(),
        )

    @property
    def collection(self):
        if self._collection is None:
            self._init_db()
        return self._collection

    def store_insight(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        """Store a piece of insight text into vector memory."""
        metadata = metadata or {}
        doc_id = doc_id or str(hash(text))[:16]
        self.collection.upsert(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id],
        )
        logger.debug("Stored insight (%s): %.60s", doc_id, text[:60])

    def query_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query the top-k most similar stored insights. Returns empty list if collection is empty."""
        if self.count() == 0:
            return []
        n = min(k, self.count())
        if n == 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=n)
        entries = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                entries.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })
        return entries

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        """Delete all documents in the collection."""
        self.collection.delete(where={})
        logger.info("Cleared collection: %s", self.collection_name)

    def store_strategy_result(
        self,
        strategy_type: str,
        params: dict,
        metrics: dict,
        regime: str = "",
        sentiment_score: float = 0.0,
        timerange: str = "",
    ) -> None:
        """Store a completed backtest result for future retrieval."""
        import json, uuid
        text = (
            f"Strategy type={strategy_type} params={json.dumps(params)} | "
            f"Sharpe={metrics.get('sharpe_ratio', 0):.2f} "
            f"WR={metrics.get('win_rate', 0):.0%} "
            f"DD={abs(metrics.get('max_drawdown', 0)):.2%} "
            f"Trades={metrics.get('total_trades', 0)} | "
            f"Regime={regime} Sentiment={sentiment_score:.2f} "
            f"Timerange={timerange}"
        )
        metadata = {
            "type": "strategy_result",
            "strategy_type": strategy_type,
            "sharpe": str(round(metrics.get("sharpe_ratio", 0), 4)),
            "win_rate": str(round(metrics.get("win_rate", 0), 4)),
            "max_drawdown": str(round(abs(metrics.get("max_drawdown", 0)), 4)),
            "total_trades": str(metrics.get("total_trades", 0)),
            "regime": regime,
            "timerange": timerange,
        }
        doc_id = f"strategy_{uuid.uuid4().hex[:12]}"
        self._collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id],
        )

    def get_best_strategies(
        self,
        regime: str = "",
        min_sharpe: float = 0.5,
        k: int = 5,
    ) -> list:
        """
        Retrieve top performing past strategies from memory.
        Filters by regime if provided, returns up to k results sorted by Sharpe.
        """
        query = f"best strategy sharpe win rate regime {regime}" if regime else "best strategy high sharpe win rate"
        results = self.query_similar(query, k=k * 3)  # over-fetch then filter

        strategy_results = [
            r for r in results
            if r["metadata"].get("type") == "strategy_result"
        ]

        # Filter by min Sharpe
        filtered = []
        for r in strategy_results:
            try:
                sharpe = float(r["metadata"].get("sharpe", 0))
                if sharpe >= min_sharpe:
                    filtered.append((sharpe, r))
            except (ValueError, TypeError):
                pass

        # Filter by regime if specified
        if regime:
            regime_filtered = [
                (s, r) for s, r in filtered
                if r["metadata"].get("regime", "") == regime
            ]
            if regime_filtered:
                filtered = regime_filtered

        # Sort by Sharpe descending and return top k
        filtered.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in filtered[:k]]