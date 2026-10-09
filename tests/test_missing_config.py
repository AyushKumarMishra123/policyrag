"""
Unit tests verifying graceful handling of missing API keys, handbooks, and vector databases.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.ingestion import extract_documents_from_pdf, get_embedding_function
from app.rag_chain import get_llm


def test_missing_api_key_ask_endpoint(monkeypatch, test_client: TestClient):
    """
    When Gemini API key is missing, /ask should return 503 with a helpful message.
    """
    from app import config
    monkeypatch.setattr(config.settings, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config.settings, "GOOGLE_API_KEY", None)

    response = test_client.post("/ask", json={"question": "What is the attendance policy?"})
    assert response.status_code == 503
    assert "Gemini API key is not configured" in response.json()["detail"]


def test_missing_api_key_ingest_endpoint(monkeypatch, test_client: TestClient):
    """
    When Gemini API key is missing, /ingest should return 503 explaining embeddings require a key.
    """
    from app import config
    monkeypatch.setattr(config.settings, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config.settings, "GOOGLE_API_KEY", None)

    response = test_client.post("/ingest")
    assert response.status_code == 503
    assert "Gemini API key is missing" in response.json()["detail"]


def test_missing_api_key_helper_functions(monkeypatch):
    """
    Helper functions get_llm and get_embedding_function should raise descriptive ValueErrors
    when invoked without credentials.
    """
    from app import config
    monkeypatch.setattr(config.settings, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config.settings, "GOOGLE_API_KEY", None)

    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        get_llm()

    with pytest.raises(ValueError, match="Gemini API key is not configured"):
        get_embedding_function()


def test_missing_handbook_file():
    """
    extract_documents_from_pdf should raise FileNotFoundError when file is missing.
    """
    non_existent = Path("non_existent_folder/missing_handbook.pdf")
    with pytest.raises(FileNotFoundError):
        extract_documents_from_pdf(non_existent)


def test_missing_vector_database_ask_endpoint(monkeypatch, test_client: TestClient):
    """
    When API key is present but vector DB directory does not exist,
    /ask should return 503 asking user to ingest documents.
    """
    from app import config
    monkeypatch.setattr(config.settings, "GEMINI_API_KEY", "test_key_xyz")
    monkeypatch.setattr(config.settings, "CHROMA_PERSIST_DIR", "non_existent_chroma_path")

    response = test_client.post("/ask", json={"question": "What is the attendance policy?"})
    assert response.status_code == 503
    assert "ChromaDB vector store is not initialized" in response.json()["detail"]
