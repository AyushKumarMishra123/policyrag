"""
Unit tests for frontend client logic, API integration, error parsing, and health checks.
"""

from unittest.mock import MagicMock, patch
import requests

from frontend import fetch_backend_health, send_policy_query


def test_send_policy_query_success():
    """
    Validates that a successful 200 response from backend is correctly parsed
    by the frontend client into payload data.
    """
    mock_payload = {
        "query": "I have 72% attendance. How many attendance marks will I get?",
        "answer": "You will receive 4 marks.",
        "sources": ["Mirai_SoT_Policy_Handbook_2026.pdf"],
        "page_numbers": [3],
        "grounded": True,
        "insufficient_evidence": False,
        "citations": [
            {
                "source": "Mirai_SoT_Policy_Handbook_2026.pdf",
                "page": 3,
                "snippet": "60% - 74.99% -> 4 marks Awarded.",
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("requests.post", return_value=mock_resp):
        data, err = send_policy_query("I have 72% attendance. How many marks?")
        assert err is None
        assert data["answer"] == "You will receive 4 marks."
        assert data["page_numbers"] == [3]
        assert data["grounded"] is True


def test_send_policy_query_connection_error():
    """
    Validates that backend connection refusal generates a user-friendly error message.
    """
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        data, err = send_policy_query("What is the leave policy?")
        assert data is None
        assert "Connection Failed" in err
        assert "backend.py" in err


def test_send_policy_query_timeout():
    """
    Validates that backend timeouts generate a distinct timeout error message.
    """
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Read timed out")):
        data, err = send_policy_query("What is the leave policy?")
        assert data is None
        assert "Request Timed Out" in err


def test_send_policy_query_503_unconfigured():
    """
    Validates that a 503 response from unconfigured backend explains configuration steps.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.json.return_value = {"detail": "Google Gemini API key is not configured."}

    with patch("requests.post", return_value=mock_resp):
        data, err = send_policy_query("What is the leave policy?")
        assert data is None
        assert "503" in err
        assert "GEMINI_API_KEY" in err


def test_send_policy_query_empty_string():
    """
    Validates that an empty or whitespace string is rejected prior to sending network request.
    """
    data, err = send_policy_query("   ")
    assert data is None
    assert "enter a valid policy question" in err


def test_fetch_backend_health_success():
    """
    Validates frontend health checker successfully parses a healthy backend response.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "ok",
        "api_key_configured": True,
        "handbook_present": True,
        "vector_db_ready": True,
        "vector_db_count": 48,
        "model": "gemini-3.8-flash",
        "embedding_model": "models/text-embedding-004",
        "version": "1.0.0",
    }

    with patch("requests.get", return_value=mock_resp):
        health, err = fetch_backend_health()
        assert err is None
        assert health["status"] == "ok"
        assert health["vector_db_count"] == 48


def test_fetch_backend_health_offline():
    """
    Validates frontend health checker reports error gracefully when backend is down.
    """
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError()):
        health, err = fetch_backend_health()
        assert health is None
        assert "Cannot connect to backend" in err
