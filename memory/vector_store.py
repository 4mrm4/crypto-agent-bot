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