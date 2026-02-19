"""
CDR Verification Layer

Citation verification and entailment checking.
"""

from cdr.verification.verifier import (
    Verifier,
    CitationChecker,
    CitationCheckResult,
    batch_verify,
    BatchVerificationResult,
)

__all__ = [
    "Verifier",
    "CitationChecker",
    "CitationCheckResult",
    "batch_verify",
    "BatchVerificationResult",
]
