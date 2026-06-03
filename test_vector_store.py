"""Tests for memory/vector_store.py"""

from unittest.mock import MagicMock, patch

from memory.vector_store import VectorStore


@patch("chromadb.PersistentClient")
@patch("memory.vector_store._get_embedding_function")
def test_vector_store_init(mock_emb, mock_client):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client.return_value.get_or_create_collection.return_value = mock_collection

    vs = VectorStore()
    assert vs._client is not None
    assert vs._collection is not None


@patch("chromadb.PersistentClient")
@patch("memory.vector_store._get_embedding_function")
def test_query_similar_empty(mock_emb, mock_client):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client.return_value.get_or_create_collection.return_value = mock_collection

    vs = VectorStore()
    results = vs.query_similar("test", k=5)
    assert results == []


@patch("chromadb.PersistentClient")
@patch("memory.vector_store._get_embedding_function")
def test_store_insight(mock_emb, mock_client):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client.return_value.get_or_create_collection.return_value = mock_collection

    vs = VectorStore()
    vs.store_insight("test text", {"key": "val"})
    mock_collection.upsert.assert_called_once()


@patch("chromadb.PersistentClient")
@patch("memory.vector_store._get_embedding_function")
def test_store_insight_non_dict_metadata(mock_emb, mock_client):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client.return_value.get_or_create_collection.return_value = mock_collection

    vs = VectorStore()
    vs.store_insight("text", metadata="bad")
    # no crash = success


@patch("chromadb.PersistentClient")
@patch("memory.vector_store._get_embedding_function")
def test_store_insight_none_metadata(mock_emb, mock_client):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client.return_value.get_or_create_collection.return_value = mock_collection

    vs = VectorStore()
    vs.store_insight("text", metadata=None)


@patch("chromadb.PersistentClient")
@patch("memory.vector_store._get_embedding_function")
def test_get_best_strategies_empty(mock_emb, mock_client):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client.return_value.get_or_create_collection.return_value = mock_collection

    vs = VectorStore()
    results = vs.get_best_strategies(min_sharpe=0.0, k=5)
    assert isinstance(results, list)


@patch("chromadb.PersistentClient")
@patch("memory.vector_store._get_embedding_function")
def test_count(mock_emb, mock_client):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 42
    mock_client.return_value.get_or_create_collection.return_value = mock_collection

    vs = VectorStore()
    assert vs.count() == 42


@patch("chromadb.PersistentClient")
@patch("memory.vector_store._get_embedding_function")
def test_clear(mock_emb, mock_client):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 5
    mock_client.return_value.get_or_create_collection.return_value = mock_collection

    vs = VectorStore()
    vs.clear()
    mock_collection.delete.assert_called_once()


@patch("chromadb.PersistentClient")
@patch("memory.vector_store._get_embedding_function")
def test_query_similar_with_data(mock_emb, mock_client):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"k": "v1"}, {"k": "v2"}]],
        "distances": [[0.1, 0.2]],
    }
    mock_client.return_value.get_or_create_collection.return_value = mock_collection

    vs = VectorStore()
    results = vs.query_similar("test query", k=2)
    assert len(results) == 2
    assert results[0]["text"] == "doc1"


@patch("chromadb.PersistentClient")
@patch("memory.vector_store._get_embedding_function")
def test_store_strategy_result(mock_emb, mock_client):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client.return_value.get_or_create_collection.return_value = mock_collection

    vs = VectorStore()
    vs.store_strategy_result("momentum", {"fast": 10}, {"sharpe": 1.5}, regime="uptrend")
    mock_collection.add.assert_called_once()
