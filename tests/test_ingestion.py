"""
Unit tests for the document ingestion pipeline, metadata preservation,
semantic chunking, and deduplication.
"""

from pathlib import Path
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.ingestion import (
    extract_documents_from_pdf,
    ingest_handbook,
    split_documents,
)
from app.utils import compute_chunk_id


def test_source_metadata_preservation():
    """
    Validates that PDF extraction preserves page numbers (1-indexed)
    and source filename in document metadata.
    """
    handbook_path = settings.resolved_handbook_path
    assert handbook_path.is_file(), f"Handbook PDF not found at {handbook_path}"

    docs = extract_documents_from_pdf(handbook_path)
    assert len(docs) == 16, f"Expected 16 extracted pages, got {len(docs)}"

    for i, doc in enumerate(docs):
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert doc.metadata["page"] == i + 1
        assert len(doc.page_content.strip()) > 0


def test_chunking_and_chunk_id_determinism():
    """
    Validates that semantic chunking creates overlapping chunks
    and stable, deterministic hash IDs.
    """
    sample_doc = Document(
        page_content="Mirai School of Technology Academic Policy. Section 1: Attendance.\n\n" * 20,
        metadata={"source": "Mirai_Handbook.pdf", "page": 3},
    )

    chunks, chunk_ids = split_documents([sample_doc], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert len(chunks) == len(chunk_ids)

    # Validate deterministic ID generation
    for chunk, c_id in zip(chunks, chunk_ids):
        expected_id = compute_chunk_id("Mirai_Handbook.pdf", 3, chunk.page_content)
        assert c_id == expected_id
        assert chunk.metadata["chunk_id"] == expected_id


def test_empty_pdf_rejection(tmp_path: Path):
    """
    Validates that a 0-byte empty file raises a descriptive ValueError.
    """
    empty_file = tmp_path / "empty_policy.pdf"
    empty_file.write_bytes(b"")

    with pytest.raises(ValueError, match="0 bytes"):
        extract_documents_from_pdf(empty_file)


def test_ingestion_with_mock_embeddings_and_deduplication(tmp_path: Path, mock_embeddings):
    """
    Validates end-to-end ingestion into ChromaDB using mock embeddings:
    indexes chunks, verifies chunk count, and asserts idempotency on re-ingestion.
    """
    persist_dir = tmp_path / "test_chroma"
    handbook_path = settings.resolved_handbook_path

    # Initial ingestion
    original_persist = settings.CHROMA_PERSIST_DIR
    settings.CHROMA_PERSIST_DIR = str(persist_dir)

    try:
        res1 = ingest_handbook(
            pdf_path=handbook_path,
            rebuild=True,
            embedding_model=mock_embeddings,
        )
        assert res1.status == "success"
        assert res1.chunks_indexed > 0

        first_count = res1.chunks_indexed

        # Re-ingestion without rebuild: chunk count should remain identical due to chunk ID deduplication
        res2 = ingest_handbook(
            pdf_path=handbook_path,
            rebuild=False,
            embedding_model=mock_embeddings,
        )
        assert res2.status == "success"
        assert res2.chunks_indexed == first_count

    finally:
        settings.CHROMA_PERSIST_DIR = original_persist
