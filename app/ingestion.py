"""
Document ingestion pipeline for extracting, chunking, embedding, and indexing
the Mirai Student Policy Handbook into ChromaDB.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pypdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.schemas import IngestResponse
from app.utils import clean_text, compute_chunk_id


def get_embedding_function(api_key: Optional[str] = None) -> Embeddings:
    """
    Initializes and returns the Google Gemini embeddings client.
    Includes automated fallback to supported embedding models (e.g. gemini-embedding-001)
    if the primary configured model is not supported in the active API environment.
    """
    key = (api_key or settings.effective_api_key).strip()
    if not key:
        raise ValueError(
            "Gemini API key is not configured. Set GEMINI_API_KEY in .env or environment variables."
        )

    primary_model = settings.EMBEDDING_MODEL
    try:
        emb = GoogleGenerativeAIEmbeddings(model=primary_model, google_api_key=key)
        emb.embed_query("ping")
        return emb
    except Exception as exc:
        err_msg = str(exc).lower()
        if "not found" in err_msg or "not supported" in err_msg:
            candidates = ["models/gemini-embedding-001", "models/gemini-embedding-2", "models/text-embedding-004"]
            for candidate in candidates:
                if candidate == primary_model:
                    continue
                try:
                    fallback_emb = GoogleGenerativeAIEmbeddings(model=candidate, google_api_key=key)
                    fallback_emb.embed_query("ping")
                    return fallback_emb
                except Exception:
                    continue
        return GoogleGenerativeAIEmbeddings(model=primary_model, google_api_key=key)


def extract_documents_from_pdf(pdf_path: str | Path) -> List[Document]:
    """
    Extracts text page-by-page from a PDF file using pypdf,
    preserving 1-indexed page numbers and source filename in metadata.
    """
    target = Path(pdf_path)
    if not target.is_file():
        raise FileNotFoundError(f"Source PDF file not found at: {target.resolve()}")

    if target.stat().st_size == 0:
        raise ValueError(f"Source PDF file is empty (0 bytes): {target.name}")

    try:
        reader = pypdf.PdfReader(str(target))
    except Exception as exc:
        raise ValueError(f"Failed to read PDF file '{target.name}': {exc}") from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"PDF file '{target.name}' contains 0 pages.")

    documents: List[Document] = []
    source_name = target.name

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = ""
        # Attempt layout extraction first for optimal table/structure preservation
        try:
            raw_text = page.extract_text(extraction_mode="layout") or ""
        except Exception:
            raw_text = page.extract_text() or ""

        cleaned = clean_text(raw_text)
        if cleaned:
            documents.append(
                Document(
                    page_content=cleaned,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "total_pages": total_pages,
                    },
                )
            )

    if not documents:
        raise ValueError(
            f"No readable text could be extracted from PDF '{target.name}'. "
            "The document may be scanned or empty."
        )

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Tuple[List[Document], List[str]]:
    """
    Splits extracted documents into semantic overlapping chunks and computes
    deterministic chunk IDs to prevent duplicate embeddings.
    """
    c_size = chunk_size or settings.CHUNK_SIZE
    c_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[Document] = []
    chunk_ids: List[str] = []

    for doc in documents:
        page_chunks = text_splitter.split_documents([doc])
        for chunk in page_chunks:
            source = chunk.metadata.get("source", "Handbook")
            page = int(chunk.metadata.get("page", 1))
            c_id = compute_chunk_id(source, page, chunk.page_content)

            chunk.metadata["chunk_id"] = c_id
            chunks.append(chunk)
            chunk_ids.append(c_id)

    return chunks, chunk_ids


def ingest_handbook(
    pdf_path: Optional[str | Path] = None,
    rebuild: bool = False,
    embedding_model: Optional[Embeddings] = None,
) -> IngestResponse:
    """
    End-to-end ingestion pipeline:
    1. Ensures target directories exist.
    2. Extracts text with page preservation.
    3. Splits into semantic chunks with unique IDs.
    4. Indexes into ChromaDB with Gemini embeddings (or provided mock embeddings).
    5. Deduplicates chunks automatically.
    """
    # Auto-create essential directories
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    resolved_path = Path(pdf_path) if pdf_path else settings.resolved_handbook_path

    # Extract pages
    extracted_docs = extract_documents_from_pdf(resolved_path)
    total_pages = len(set(doc.metadata.get("page", 1) for doc in extracted_docs))

    # Split into chunks
    chunks, chunk_ids = split_documents(extracted_docs)

    # Obtain embedding model
    embed_fn = embedding_model or get_embedding_function()

    # Initialize Chroma vectorstore
    vectorstore = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    # Safe rebuild if requested
    if rebuild:
        try:
            vectorstore.delete_collection()
        except Exception:
            pass
        vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embed_fn,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )

    # Ingest chunks with explicit IDs to guarantee idempotency and avoid duplicates
    vectorstore.add_documents(documents=chunks, ids=chunk_ids)

    indexed_count = vectorstore._collection.count()

    return IngestResponse(
        status="success",
        filename=resolved_path.name,
        chunks_indexed=indexed_count,
        total_pages=total_pages,
        message=(
            f"Successfully ingested {len(chunks)} chunks across {total_pages} pages "
            f"from '{resolved_path.name}'. Total index count: {indexed_count}."
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Mirai Policy Handbook into ChromaDB.")
    parser.add_argument("--path", type=str, default=None, help="Path to handbook PDF file")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector database from scratch")
    args = parser.parse_args()

    print(f"Starting ingestion (rebuild={args.rebuild})...")
    try:
        result = ingest_handbook(pdf_path=args.path, rebuild=args.rebuild)
        print(f"Ingestion Succeeded: {result.message}")
    except Exception as e:
        print(f"Ingestion Failed: {e}")
        exit(1)
