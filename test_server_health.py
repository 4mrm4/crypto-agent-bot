"""Tests for /api/data/health endpoint."""
import pytest
from fastapi.testclient import TestClient
from api.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint_returns_json(client):
    resp = client.get("/api/data/health")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    # Returns a message and sources dict when empty
    if "sources" in data:
        assert isinstance(data["sources"], dict)
    else:
        # Has source-specific keys
        for key in data:
            assert "source" in data[key]
            assert "consecutive_failures" in data[key]
            assert "is_healthy" in data[key]
