"""
Unit tests for request payload validation across API endpoints.
"""

from fastapi.testclient import TestClient


def test_ask_empty_question_fails_validation(test_client: TestClient):
    """Empty question string should trigger 422 validation error."""
    response = test_client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_whitespace_question_fails_validation(test_client: TestClient):
    """Whitespace-only question string should trigger 422 validation error."""
    response = test_client.post("/ask", json={"question": "    "})
    assert response.status_code == 422


def test_ask_single_character_fails_validation(test_client: TestClient):
    """Question shorter than 2 characters should trigger 422 validation error."""
    response = test_client.post("/ask", json={"question": "?"})
    assert response.status_code == 422


def test_ask_missing_payload_fails_validation(test_client: TestClient):
    """Missing JSON payload body should trigger 422 validation error."""
    response = test_client.post("/ask", json={})
    assert response.status_code == 422


def test_chat_endpoint_alias_validation(test_client: TestClient):
    """Validates /chat alias performs identical validation to /ask."""
    response = test_client.post("/chat", json={"question": "   "})
    assert response.status_code == 422


def test_ingest_invalid_file_extension(monkeypatch, test_client: TestClient):
    """Uploading non-PDF file should trigger 400 Bad Request."""
    from app import config
    monkeypatch.setattr(config.settings, "GEMINI_API_KEY", "dummy_key")

    files = {"file": ("test_doc.txt", b"Invalid content", "text/plain")}
    response = test_client.post("/ingest", files=files)
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]
