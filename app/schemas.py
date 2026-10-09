"""
Pydantic data models and schemas for Mirai API and RAG pipelines.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    """Payload for submitting a policy question."""
    question: str = Field(
        ...,
        description="The student's policy inquiry (e.g. attendance, leaves, clubs).",
        examples=["I have 72% attendance. How many attendance marks will I get?"],
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 2:
            raise ValueError("Question must contain at least 2 non-whitespace characters.")
        return stripped


class SourceCitation(BaseModel):
    """Detailed metadata for a cited document excerpt."""
    source: str = Field(..., description="Filename or identifier of the source document")
    page: int = Field(..., description="1-indexed page number of the cited handbook page")
    snippet: str = Field(..., description="Relevant excerpt supporting the answer")


class QuestionResponse(BaseModel):
    """Structured response from the Mirai policy advisor."""
    query: str = Field(..., description="Original student query")
    answer: str = Field(..., description="Fact-checked policy answer or insufficiency statement")
    sources: List[str] = Field(default_factory=list, description="Unique source document names")
    page_numbers: List[int] = Field(default_factory=list, description="Unique 1-indexed page numbers")
    grounded: bool = Field(..., description="True if answer is directly grounded in retrieved handbook evidence")
    insufficient_evidence: bool = Field(..., description="True if evidence was absent or inadequate to answer")
    citations: List[SourceCitation] = Field(default_factory=list, description="Granular citations with snippets")


class IngestResponse(BaseModel):
    """Status report from document ingestion."""
    status: str = Field(..., description="Ingestion outcome status (e.g., success, skipped, error)")
    filename: str = Field(..., description="Name of the processed handbook PDF")
    chunks_indexed: int = Field(..., description="Total semantic chunks stored in vector store")
    total_pages: int = Field(..., description="Total pages extracted from the source PDF")
    message: str = Field(..., description="Human-readable summary of ingestion execution")


class HealthResponse(BaseModel):
    """System health check and component diagnostics."""
    status: str = Field(..., description="Overall service status ('ok' or 'degraded')")
    api_key_configured: bool = Field(..., description="Whether Gemini API key is configured")
    handbook_present: bool = Field(..., description="Whether the source policy handbook PDF exists")
    vector_db_ready: bool = Field(..., description="Whether Chroma vector database is initialized and populated")
    vector_db_count: int = Field(..., description="Number of vectors currently indexed in ChromaDB")
    model: str = Field(..., description="Target Gemini generation model")
    embedding_model: str = Field(..., description="Target Gemini embedding model")
    version: str = Field(..., description="Application release version")


class ErrorResponse(BaseModel):
    """Standardized error payload."""
    detail: str = Field(..., description="Human-readable error description")
    error_type: Optional[str] = Field(default=None, description="Classification of the error")
