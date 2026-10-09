"""
Unit tests for system error handling, timeout handling, and retriever resilience.
"""

import asyncio
from unittest.mock import patch
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.retrieval import MultiQueryRetrieverLCEL
from tests.conftest import MockRetriever


def test_ask_endpoint_timeout_handling(monkeypatch, test_client: TestClient):
    """
    Validates that if RAG execution exceeds timeout, the backend responds with HTTP 504.
    """
    from app import config
    import backend

    monkeypatch.setattr(config.settings, "GEMINI_API_KEY", "dummy_key")
    monkeypatch.setattr(config.Settings, "vector_db_exists", property(lambda self: True))
    monkeypatch.setattr(config.settings, "REQUEST_TIMEOUT_SECONDS", 0.05)

    class SlowChain:
        def query(self, question: str):
            import time
            time.sleep(0.2)
            return None

    monkeypatch.setattr(backend, "get_or_initialize_rag_chain", lambda: SlowChain())
    monkeypatch.setattr(backend, "rag_chain_instance", SlowChain())

    response = test_client.post("/ask", json={"question": "What is the policy?"})
    assert response.status_code == 504
    assert "exceeded timeout limit" in response.json()["detail"]


def test_multi_query_retriever_fallback_on_error():
    """
    Validates that if query expansion LLM fails, MultiQueryRetrieverLCEL
    gracefully falls back to the original query without raising exceptions.
    """
    docs = [
        Document(
            page_content="Campus policy excerpt.",
            metadata={"source": "Mirai_Handbook.pdf", "page": 1, "chunk_id": "c1"},
        )
    ]
    base_retriever = MockRetriever(docs=docs)

    # Retriever with no LLM (or broken generator)
    mq_retriever = MultiQueryRetrieverLCEL(base_retriever=base_retriever, llm=None, k=2)

    results = mq_retriever.invoke("Casual student inquiry")
    assert len(results) == 1
    assert results[0].page_content == "Campus policy excerpt."


def test_corrupted_pdf_error_reporting(tmp_path):
    """
    Validates that a corrupted non-PDF file disguised as a PDF raises ValueError.
    """
    from app.ingestion import extract_documents_from_pdf
    bad_pdf = tmp_path / "corrupted.pdf"
    bad_pdf.write_bytes(b"This is not a real PDF structure %PDF-invalid")

    import pytest
    with pytest.raises(ValueError, match="Failed to read PDF file"):
        extract_documents_from_pdf(bad_pdf)
