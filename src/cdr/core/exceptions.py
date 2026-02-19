"""
CDR Custom Exceptions

This module defines all custom exceptions used throughout the CDR system.
Exceptions are organized by layer/responsibility.
"""

from typing import Any


class CDRError(Exception):
    """Base exception for all CDR errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================


class ConfigurationError(CDRError):
    """Error in system configuration."""

    pass


class MissingAPIKeyError(ConfigurationError):
    """Required API key is missing."""

    def __init__(self, key_name: str):
        super().__init__(f"Missing required API key: {key_name}", {"key_name": key_name})


# =============================================================================
# VALIDATION ERRORS
# =============================================================================


class ValidationError(CDRError):
    """Error in data validation."""

    pass


class PICOValidationError(ValidationError):
    """PICO structure is invalid or incomplete."""

    def __init__(self, message: str, missing_fields: list[str] | None = None):
        super().__init__(message, {"missing_fields": missing_fields or []})


class SchemaValidationError(ValidationError):
    """Schema validation failed."""

    pass


# =============================================================================
# RETRIEVAL ERRORS
# =============================================================================


class RetrievalError(CDRError):
    """Base error for retrieval operations."""

    pass


class PubMedError(RetrievalError):
    """Error communicating with PubMed API."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message, {"status_code": status_code})


class ClinicalTrialsError(RetrievalError):
    """Error communicating with ClinicalTrials.gov API."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message, {"status_code": status_code})


class VectorStoreError(RetrievalError):
    """Error with vector store operations."""

    pass


class EmbeddingError(RetrievalError):
    """Error generating embeddings."""

    pass


class CacheError(RetrievalError):
    """Error with cache operations."""

    pass


# =============================================================================
# PARSING ERRORS
# =============================================================================


class ParsingError(CDRError):
    """Base error for document parsing."""

    pass


class PDFParsingError(ParsingError):
    """Error parsing PDF document."""

    def __init__(self, message: str, file_path: str | None = None):
        super().__init__(message, {"file_path": file_path})


class HTMLParsingError(ParsingError):
    """Error parsing HTML content."""

    pass


class SnippetExtractionError(ParsingError):
    """Error extracting snippets from document."""

    pass


# =============================================================================
# EXTRACTION ERRORS
# =============================================================================


class ExtractionError(CDRError):
    """Base error for structured extraction."""

    pass


class StudyCardExtractionError(ExtractionError):
    """Error extracting StudyCard from evidence."""

    def __init__(self, message: str, record_id: str | None = None):
        super().__init__(message, {"record_id": record_id})


class RoB2AssessmentError(ExtractionError):
    """Error in RoB2 assessment."""

    pass


# =============================================================================
# LLM ERRORS
# =============================================================================


class LLMError(CDRError):
    """Base error for LLM operations."""

    pass


class LLMProviderError(LLMError):
    """Error from LLM provider."""

    def __init__(
        self, message: str, provider: str, status_code: int | None = None, retryable: bool = False
    ):
        super().__init__(
            message, {"provider": provider, "status_code": status_code, "retryable": retryable}
        )
        self.retryable = retryable


class LLMRateLimitError(LLMProviderError):
    """Rate limit exceeded."""

    def __init__(self, provider: str, retry_after: float | None = None):
        super().__init__(f"Rate limit exceeded for {provider}", provider=provider, retryable=True)
        self.retry_after = retry_after


class LLMContextLengthError(LLMError):
    """Input exceeds model context length."""

    def __init__(self, message: str, max_tokens: int, actual_tokens: int):
        super().__init__(message, {"max_tokens": max_tokens, "actual_tokens": actual_tokens})


# =============================================================================
# VERIFICATION ERRORS
# =============================================================================


class VerificationError(CDRError):
    """Base error for verification operations."""

    pass


class CitationCoverageError(VerificationError):
    """Citation coverage below threshold."""

    def __init__(self, coverage: float, threshold: float, uncovered_claims: list[str]):
        super().__init__(
            f"Citation coverage {coverage:.2%} below threshold {threshold:.2%}",
            {"coverage": coverage, "threshold": threshold, "uncovered_claims": uncovered_claims},
        )


class EntailmentError(VerificationError):
    """Claim not entailed by supporting evidence."""

    def __init__(self, claim_id: str, snippet_id: str):
        super().__init__(
            f"Claim {claim_id} not entailed by snippet {snippet_id}",
            {"claim_id": claim_id, "snippet_id": snippet_id},
        )


class VerificationGateError(VerificationError):
    """Verification gate blocked publication."""

    def __init__(self, failed_checks: list[str]):
        super().__init__(
            f"Verification gate failed: {', '.join(failed_checks)}",
            {"failed_checks": failed_checks},
        )


# =============================================================================
# SAFETY ERRORS
# =============================================================================


class SafetyError(CDRError):
    """Safety-related errors."""

    pass


class SafetyViolationError(SafetyError):
    """Request violates safety guidelines."""

    def __init__(self, reason: str):
        super().__init__(f"Safety violation: {reason}", {"reason": reason})


class ScopeViolationError(SafetyError):
    """Request outside allowed scope."""

    def __init__(self, reason: str):
        super().__init__(f"Scope violation: {reason}", {"reason": reason})


# =============================================================================
# ORCHESTRATION ERRORS
# =============================================================================


class OrchestrationError(CDRError):
    """Base error for workflow orchestration."""

    pass


class GraphExecutionError(OrchestrationError):
    """Error executing graph node."""

    def __init__(self, node: str, message: str, state_snapshot: dict[str, Any] | None = None):
        super().__init__(
            f"Error in node '{node}': {message}", {"node": node, "state_snapshot": state_snapshot}
        )


class CheckpointError(OrchestrationError):
    """Error with state checkpointing."""

    pass


class MaxIterationsError(OrchestrationError):
    """Maximum iterations exceeded."""

    def __init__(self, max_iterations: int, loop_type: str):
        super().__init__(
            f"Maximum iterations ({max_iterations}) exceeded for {loop_type}",
            {"max_iterations": max_iterations, "loop_type": loop_type},
        )


# =============================================================================
# STORAGE ERRORS
# =============================================================================


class StorageError(CDRError):
    """Base error for storage operations."""

    pass


class ArtifactNotFoundError(StorageError):
    """Requested artifact not found."""

    def __init__(self, run_id: str, artifact_type: str):
        super().__init__(
            f"Artifact '{artifact_type}' not found for run {run_id}",
            {"run_id": run_id, "artifact_type": artifact_type},
        )


class RunNotFoundError(StorageError):
    """Requested run not found."""

    def __init__(self, run_id: str):
        super().__init__(f"Run not found: {run_id}", {"run_id": run_id})
