"""
Utility functions for text normalization, hashing, citations, and evidence detection.
"""

import hashlib
import re
from typing import List, Tuple, Set
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """
    Cleans raw extracted PDF text by normalizing whitespace, standardizing line breaks,
    and removing excessive spacing while maintaining tables and indentation.
    """
    if not text:
        return ""

    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Clean trailing whitespaces per line
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned = "\n".join(lines)

    # Collapse 3 or more consecutive newlines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def compute_chunk_id(source: str, page: int, chunk_content: str) -> str:
    """
    Generates a deterministic SHA-256 hash ID for a document chunk
    to prevent duplicate chunks across multiple ingestion runs.
    """
    normalized_content = " ".join(chunk_content.split())
    hash_payload = f"{source}::p{page}::{normalized_content}"
    return hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()[:24]


def extract_citations_and_pages(documents: List[Document]) -> Tuple[List[str], List[int]]:
    """
    Extracts unique source filenames and sorted 1-indexed page numbers
    from a list of LangChain Document objects.
    """
    sources: Set[str] = set()
    pages: Set[int] = set()

    for doc in documents:
        src = doc.metadata.get("source", "Mirai_SoT_Policy_Handbook_2026.pdf")
        if src:
            # Keep only basename if full path was stored
            sources.add(src.split("/")[-1])

        page = doc.metadata.get("page")
        if page is not None:
            try:
                pages.add(int(page))
            except (ValueError, TypeError):
                pass

    return sorted(list(sources)), sorted(list(pages))


def format_docs_for_prompt(documents: List[Document]) -> str:
    """
    Formats retrieved documents into a structured reference context with
    explicit page number annotations for the LLM.
    """
    if not documents:
        return "No relevant handbook sections were found in the database."

    formatted_chunks = []
    for i, doc in enumerate(documents, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        source_name = doc.metadata.get("source", "Handbook").split("/")[-1]
        header = f"--- [EXCERPT {i} | Source: {source_name} | Page {page_num}] ---"
        formatted_chunks.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(formatted_chunks)


def is_insufficient_evidence_response(answer: str) -> bool:
    """
    Heuristic to determine if the generated answer indicates that the handbook
    lacks sufficient evidence to answer or declines to answer.
    """
    if not answer:
        return True

    lower = answer.lower()
    negative_phrases = [
        "cannot be found within the retrieved",
        "cannot be found in the provided",
        "does not mention",
        "does not specify",
        "does not provide information",
        "does not contain any information",
        "does not contain sufficient information",
        "not mentioned in the policy handbook",
        "not mentioned in the provided",
        "no information provided",
        "insufficient evidence",
        "i could not find any policy",
        "i cannot answer this based on the handbook",
        "handbook does not state",
        "no monetary fine is specified",
        "no specific fine is mentioned",
        "no fine amount is mentioned",
        "politely decline",
    ]

    return any(phrase in lower for phrase in negative_phrases)
