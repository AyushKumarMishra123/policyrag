"""
Unit tests for the /health diagnostic endpoint.
"""

from fastapi.testclient import TestClient


def test_health_check_endpoint(test_client: TestClient):
    """
    Validates that the /health endpoint responds with HTTP 200 and
    contains all required subsystem diagnostic fields.
    """
    response = test_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] in ["ok", "degraded"]
    assert "api_key_configured" in data
    assert "handbook_present" in data
    assert "vector_db_ready" in data
    assert "vector_db_count" in data
    assert "model" in data
    assert "embedding_model" in data
    assert "version" in data


def test_health_check_does_not_crash_when_unconfigured(monkeypatch, test_client: TestClient):
    """
    Validates requirement: Missing API keys or uninitialized vector databases
    must report 'degraded' status without crashing the server.
    """
    from app import config

    # Simulate completely empty configuration
    monkeypatch.setattr(config.settings, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config.settings, "GOOGLE_API_KEY", None)
    monkeypatch.setattr(config.settings, "CHROMA_PERSIST_DIR", "non_existent_chroma_dir")

    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["api_key_configured"] is False
    assert data["vector_db_ready"] is False
