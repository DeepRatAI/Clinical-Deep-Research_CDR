"""
CDR Configuration

Centralized configuration using Pydantic Settings.
All configuration is loaded from environment variables with sensible defaults.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="", extra="ignore"
    )

    # HuggingFace (PRIMARY PROVIDER)
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    hf_endpoint_url: str | None = Field(default=None, alias="HF_ENDPOINT_URL")
    hf_model: str = Field(default="meta-llama/Meta-Llama-3.1-70B-Instruct", alias="HF_MODEL")
    hf_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    hf_max_tokens: int = Field(default=4096, ge=1)

    # Groq (FREE TIER - generous limits)
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # Gemini (Google AI Studio - 1M+ tokens/day FREE)
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")

    # Cerebras (1M tokens/day FREE)
    cerebras_api_key: str | None = Field(default=None, alias="CEREBRAS_API_KEY")

    # OpenRouter (400+ models, free tier)
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")

    # Cloudflare Workers AI
    cloudflare_api_key: str | None = Field(default=None, alias="CLOUDFLARE_API_KEY")
    cloudflare_account_id: str | None = Field(default=None, alias="CLOUDFLARE_ACCOUNT_ID")

    # OpenAI (FALLBACK)
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=4096, ge=1)

    # Anthropic (FALLBACK)
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", alias="ANTHROPIC_MODEL")

    # Default provider: Gemini by default (1M+ tokens/day FREE)
    default_provider: Literal[
        "gemini",
        "cerebras",
        "cloudflare",
        "openrouter",
        "huggingface",
        "groq",
        "openai",
        "anthropic",
    ] = Field(default="gemini", alias="LLM_DEFAULT_PROVIDER")


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="", extra="ignore"
    )

    # NCBI/PubMed
    ncbi_api_key: str | None = Field(default=None, alias="NCBI_API_KEY")
    ncbi_email: str = Field(default="cdr@example.com", alias="NCBI_EMAIL")
    pubmed_max_results: int = Field(default=100, ge=1, le=1000)

    # ClinicalTrials.gov
    clinical_trials_max_results: int = Field(default=50, ge=1, le=500)

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=32, ge=1)

    # Reranker
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    reranker_top_k: int = Field(default=30, ge=1)

    # BM25
    bm25_top_k: int = Field(default=50, ge=1)

    # Dense retrieval
    dense_top_k: int = Field(default=50, ge=1)


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="", extra="ignore"
    )

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="cdr_embeddings", alias="QDRANT_COLLECTION")
    qdrant_vector_size: int = Field(default=384, ge=1)  # MiniLM default


class StorageSettings(BaseSettings):
    """Storage configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="", extra="ignore"
    )

    artifact_path: Path = Field(default=Path("/data/cdr/artifacts"), alias="CDR_ARTIFACT_PATH")
    db_path: Path = Field(default=Path("/data/cdr/cdr.db"), alias="CDR_DB_PATH")

    @field_validator("artifact_path", "db_path", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Resolve path and expand user."""
        return Path(v).expanduser().resolve()


class VerificationSettings(BaseSettings):
    """Verification thresholds."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="", extra="ignore"
    )

    citation_coverage_threshold: float = Field(
        default=0.90, ge=0.0, le=1.0, alias="CDR_CITATION_COVERAGE_THRESHOLD"
    )
    min_studies_for_claim: int = Field(default=1, ge=1)
    max_iterations: int = Field(default=3, ge=1, le=10)


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="", extra="ignore"
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_format: Literal["json", "text"] = Field(default="json", alias="LOG_FORMAT")


class FeatureFlags(BaseSettings):
    """Feature flags for optional functionality."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="", extra="ignore"
    )

    enable_hitl: bool = Field(default=False, alias="CDR_ENABLE_HITL")
    strict_mode: bool = Field(default=True, alias="CDR_STRICT_MODE")
    debug: bool = Field(default=False, alias="CDR_DEBUG")
    max_records_per_source: int = Field(default=100, alias="CDR_MAX_RECORDS_PER_SOURCE")


class Settings(BaseSettings):
    """
    Main CDR settings aggregator.

    Usage:
        from cdr.config import get_settings
        settings = get_settings()
        print(settings.llm.openai_api_key)
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Sub-settings (composed)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.storage.artifact_path.mkdir(parents=True, exist_ok=True)
        self.storage.db_path.parent.mkdir(parents=True, exist_ok=True)


# Singleton pattern for settings
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
