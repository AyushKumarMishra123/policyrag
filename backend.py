"""
FastAPI Backend Application for Mirai Autonomous Student Policy Advisor.
Exposes endpoints for system health diagnostics, document ingestion, and policy Q&A.
"""

import asyncio
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import settings
from app.ingestion import ingest_handbook
from app.rag_chain import MiraiPolicyChain, create_mirai_rag_chain
from app.retrieval import get_vectorstore
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    QuestionRequest,
    QuestionResponse,
)


# Global RAG chain instance holder
rag_chain_instance: Optional[MiraiPolicyChain] = None


def get_or_initialize_rag_chain() -> Optional[MiraiPolicyChain]:
    """
    Safely retrieves or lazily initializes the RAG chain.
    Returns None if requirements (API key or vector database) are unfulfilled.
    """
    global rag_chain_instance
    if rag_chain_instance is not None:
        return rag_chain_instance

    if not settings.is_api_key_configured:
        return None

    if not settings.vector_db_exists:
        return None

    try:
        rag_chain_instance = create_mirai_rag_chain()
        return rag_chain_instance
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown management.
    Ensures background initialization doesn't block startup.
    """
    # Attempt lazy initialization on startup
    get_or_initialize_rag_chain()
    yield


app = FastAPI(
    title="Mirai Autonomous Student Policy Advisor API",
    description=(
        "Production-grade FastAPI service powering the Mirai Student Policy Advisor. "
        "Strictly fact-grounded Q&A over the official Mirai School of Technology Handbook."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local Streamlit frontend and external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check & component diagnostics",
    tags=["Diagnostics"],
)
async def health_check():
    """
    Returns the real-time operational status of all subsystem components:
    Gemini API credentials, source handbook PDF presence, and ChromaDB vector store readiness.
    """
    vector_ready = False
    vector_count = 0

    if settings.vector_db_exists:
        try:
            if settings.is_api_key_configured:
                vs = get_vectorstore()
                vector_count = vs._collection.count()
                vector_ready = vector_count > 0
            else:
                # Vector directory exists on disk, but API key missing to connect embedding function
                vector_ready = True
        except Exception:
            vector_ready = False

    is_healthy = settings.is_api_key_configured and settings.handbook_exists and vector_ready

    return HealthResponse(
        status="ok" if is_healthy else "degraded",
        api_key_configured=settings.is_api_key_configured,
        handbook_present=settings.handbook_exists,
        vector_db_ready=vector_ready,
        vector_db_count=vector_count,
        model=settings.GEMINI_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        version=__version__,
    )


@app.post(
    "/ask",
    response_model=QuestionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query payload"},
        503: {"model": ErrorResponse, "description": "Vector database or API key unconfigured"},
        504: {"model": ErrorResponse, "description": "Request processing timeout"},
    },
    summary="Ask a student policy question",
    tags=["Policy Q&A"],
)
async def ask_question(request: QuestionRequest):
    """
    Processes a student inquiry through the Mirai RAG pipeline.
    Retrieves official handbook evidence, applies multi-query expansion,
    and returns a fact-grounded response with page citations.
    """
    if not settings.is_api_key_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Google Gemini API key is not configured. Please supply GEMINI_API_KEY "
                "in your .env file or server environment."
            ),
        )

    if not settings.vector_db_exists:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "ChromaDB vector store is not initialized or contains no documents. "
                "Please trigger document ingestion via POST /ingest first."
            ),
        )

    chain = get_or_initialize_rag_chain()
    if not chain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to initialize RAG pipeline. Check vector store and API configuration.",
        )

    try:
        # Enforce timeout for inference & retrieval
        response = await asyncio.wait_for(
            asyncio.to_thread(chain.query, request.question),
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
        return response
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Inference request exceeded timeout limit of {settings.REQUEST_TIMEOUT_SECONDS}s.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing policy RAG chain: {str(exc)}",
        )


@app.post(
    "/chat",
    response_model=QuestionResponse,
    summary="Chat endpoint alias (Problem Statement compliant)",
    tags=["Policy Q&A"],
)
async def chat_endpoint(request: QuestionRequest):
    """Alias for /ask fulfilling the project problem statement specification."""
    return await ask_question(request)


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest handbook PDF into vector database",
    tags=["Ingestion"],
)
async def ingest_document(
    file: Optional[UploadFile] = File(
        default=None,
        description="Optional PDF file upload. If omitted, ingests the configured handbook.",
    ),
    rebuild: bool = Query(
        default=False,
        description="Rebuild index completely by purging existing collection.",
    ),
):
    """
    Ingests, chunks, embeds, and indexes a policy handbook into ChromaDB.
    Supports either direct file upload or default handbook processing.
    """
    if not settings.is_api_key_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API key is missing. Embeddings cannot be generated without GEMINI_API_KEY.",
        )

    target_pdf_path: Path

    if file is not None:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type '{file.filename}'. Only PDF files are supported.",
            )

        os.makedirs("data", exist_ok=True)
        target_pdf_path = Path("data") / file.filename

        try:
            with open(target_pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save uploaded file: {e}",
            )
    else:
        target_pdf_path = settings.resolved_handbook_path
        if not target_pdf_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Default handbook PDF not found at '{target_pdf_path}'. Upload a PDF instead.",
            )

    try:
        result = await asyncio.to_thread(
            ingest_handbook,
            pdf_path=target_pdf_path,
            rebuild=rebuild,
        )

        # Reset global RAG chain instance to reload fresh vectorstore
        global rag_chain_instance
        rag_chain_instance = None
        get_or_initialize_rag_chain()

        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True,
    )
