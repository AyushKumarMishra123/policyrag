"""
Configuration management for Mirai Student Policy Advisor using Pydantic Settings.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys (Supports either GEMINI_API_KEY or GOOGLE_API_KEY)
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API Key")
    GOOGLE_API_KEY: Optional[str] = Field(default=None, description="Alternative Google API Key")

    # Model Configuration
    GEMINI_MODEL: str = Field(
        default="gemini-3.8-flash",
        description="Generative LLM model name (e.g. gemini-3.8-flash)",
    )
    EMBEDDING_MODEL: str = Field(
        default="models/text-embedding-004",
        description="Embeddings model (e.g. models/text-embedding-004)",
    )
    TEMPERATURE: float = Field(
        default=0.0,
        description="Sampling temperature (0.0 for strict deterministic factuality)",
    )

    # Handbook & Storage Paths
    HANDBOOK_PATH: str = Field(
        default="data/Mirai_SoT_Policy_Handbook_2026.pdf",
        description="Path to the source handbook PDF",
    )
    CHROMA_PERSIST_DIR: str = Field(
        default="chroma_db",
        description="Local directory for ChromaDB vector storage",
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="mirai_policy_handbook",
        description="ChromaDB collection identifier",
    )

    # Chunking & Retrieval Parameters
    CHUNK_SIZE: int = Field(default=1000, description="Semantic chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=200, description="Semantic chunk overlap in characters")
    RETRIEVER_K: int = Field(default=4, description="Top-k chunks to retrieve")
    USE_MULTI_QUERY: bool = Field(default=True, description="Enable multi-query query expansion")

    # Server & Client Configuration
    BACKEND_HOST: str = Field(default="0.0.0.0", description="Backend binding host")
    BACKEND_PORT: int = Field(default=8000, description="Backend port")
    BACKEND_URL: str = Field(default="http://127.0.0.1:8000", description="Backend base URL")
    REQUEST_TIMEOUT_SECONDS: float = Field(default=30.0, description="Request timeout in seconds")

    @property
    def effective_api_key(self) -> str:
        """Returns the configured API key if available, else empty string."""
        return (self.GEMINI_API_KEY or self.GOOGLE_API_KEY or "").strip()

    @property
    def is_api_key_configured(self) -> bool:
        """Returns True if a non-empty API key is present."""
        return bool(self.effective_api_key)

    @property
    def resolved_handbook_path(self) -> Path:
        """
        Locates the handbook PDF, looking at configured path first,
        then fallback candidates in the repository.
        """
        primary = Path(self.HANDBOOK_PATH)
        if primary.is_file():
            return primary

        # Common fallback locations
        fallbacks = [
            Path("data/Mirai_SoT_Policy_Handbook_2026.pdf"),
            Path("data/Mirai_SoT_Policy_Handbook_2026 (1).pdf"),
            Path("Mirai_SoT_Policy_Handbook_2026 (1).pdf"),
            Path("Mirai_SoT_Policy_Handbook_2026.pdf"),
        ]
        for candidate in fallbacks:
            if candidate.is_file():
                return candidate
        return primary

    @property
    def handbook_exists(self) -> bool:
        """Checks if the source handbook PDF is available on disk."""
        return self.resolved_handbook_path.is_file()

    @property
    def vector_db_exists(self) -> bool:
        """Checks if Chroma persistence directory exists and contains data."""
        p = Path(self.CHROMA_PERSIST_DIR)
        if not p.is_dir():
            return False
        # Chroma stores chroma.sqlite3 and subdirectories
        return any(p.iterdir())


# Global settings instance
settings = Settings()
