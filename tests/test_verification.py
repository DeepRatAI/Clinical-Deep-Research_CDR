"""
Tests for CDR Verification Layer.

Tests for citation verification and entailment checking.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from cdr.core.enums import GRADECertainty, VerificationStatus
from cdr.core.schemas import (
    EvidenceClaim,
    Snippet,
    SourceRef,
    VerificationCheck,
    VerificationResult,
)


# =============================================================================
# VERIFIER TESTS
# =============================================================================


class TestVerifier:
    """Tests for Verifier class."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM for verification."""
        mock = MagicMock()
        mock.complete.return_value = MagicMock(
            content="""{
                "entailment": "ENTAILS",
                "confidence": 0.9,
                "supporting_quote": "The study showed significant results",
                "reasoning": "The source directly supports the claim",
                "caveats": []
            }"""
        )
        return mock

    @pytest.fixture
    def sample_claim(self):
        """Create sample claim for testing."""
        return EvidenceClaim(
            claim_id="claim_001",
            statement="Treatment reduces mortality by 30%",
            certainty=GRADECertainty.MODERATE,
            supporting_study_ids=["pmid:12345"],
            snippets=[
                Snippet(
                    text="Mortality was reduced by 30% in the treatment group",
                    source_ref=SourceRef(
                        record_id="pmid:12345",
                        location="Results, Table 2",
                    ),
                )
            ],
        )

    def test_normalize_text(self, mock_llm):
        """Test text normalization."""
        from cdr.verification.verifier import Verifier

        verifier = Verifier(mock_llm)

        text = "  This   is   a   TEST  text.  "
        normalized = verifier._normalize_text(text)

        assert normalized == "this is a test text"

    def test_find_snippet_in_source_exact(self, mock_llm):
        """Test finding exact snippet match."""
        from cdr.verification.verifier import Verifier

        verifier = Verifier(mock_llm)

        snippet = "mortality was reduced"
        source = "The study showed that mortality was reduced by 30%."

        assert verifier._find_snippet_in_source(snippet, source) is True

    def test_find_snippet_in_source_fuzzy(self, mock_llm):
        """Test fuzzy snippet matching."""
        from cdr.verification.verifier import Verifier

        verifier = Verifier(mock_llm)

        # Similar but not exact
        snippet = "mortality reduced significantly"
        source = "The mortality was significantly reduced in patients."

        # Should match with fuzzy threshold
        result = verifier._find_snippet_in_source(snippet, source, fuzzy_threshold=0.6)
        assert result is True

    def test_find_snippet_not_found(self, mock_llm):
        """Test when snippet is not in source."""
        from cdr.verification.verifier import Verifier

        verifier = Verifier(mock_llm)

        snippet = "completely different text"
        source = "The study showed mortality reduction."

        result = verifier._find_snippet_in_source(snippet, source)
        assert result is False

    def test_aggregate_checks_all_verified(self, mock_llm):
        """Test aggregation when all checks pass."""
        from cdr.verification.verifier import Verifier

        verifier = Verifier(mock_llm)

        checks = [
            VerificationCheck(
                claim_id="c1",
                source_ref=SourceRef(record_id="p1"),
                status=VerificationStatus.VERIFIED,
                confidence=0.9,
            ),
            VerificationCheck(
                claim_id="c1",
                source_ref=SourceRef(record_id="p2"),
                status=VerificationStatus.VERIFIED,
                confidence=0.85,
            ),
        ]

        result = verifier._aggregate_checks("c1", checks)

        assert result.overall_status == VerificationStatus.VERIFIED
        assert result.overall_confidence == pytest.approx(0.875, rel=0.01)

    def test_aggregate_checks_with_failure(self, mock_llm):
        """Test aggregation when some checks fail."""
        from cdr.verification.verifier import Verifier

        verifier = Verifier(mock_llm)

        checks = [
            VerificationCheck(
                claim_id="c1",
                source_ref=SourceRef(record_id="p1"),
                status=VerificationStatus.VERIFIED,
                confidence=0.9,
            ),
            VerificationCheck(
                claim_id="c1",
                source_ref=SourceRef(record_id="p2"),
                status=VerificationStatus.CONTRADICTED,
                confidence=0.1,
            ),
        ]

        result = verifier._aggregate_checks("c1", checks)

        # Should be PARTIAL because one contradicted
        assert result.overall_status == VerificationStatus.PARTIAL

    def test_aggregate_checks_empty(self, mock_llm):
        """Test aggregation with no checks."""
        from cdr.verification.verifier import Verifier

        verifier = Verifier(mock_llm)

        result = verifier._aggregate_checks("c1", [])

        assert result.overall_status == VerificationStatus.UNVERIFIABLE


# =============================================================================
# CITATION CHECKER TESTS
# =============================================================================


class TestCitationChecker:
    """Tests for CitationChecker."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        mock = MagicMock()
        mock.complete.return_value = MagicMock(
            content="""{
                "accurate": true,
                "issues": [],
                "severity": "none",
                "corrected_text": null
            }"""
        )
        return mock

    def test_parse_citation_result_accurate(self, mock_llm):
        """Test parsing accurate citation result."""
        from cdr.verification.verifier import CitationChecker

        checker = CitationChecker(mock_llm)
        result = checker._parse_citation_result(
            '{"accurate": true, "issues": [], "severity": "none"}'
        )

        assert result.accurate is True
        assert len(result.issues) == 0

    def test_parse_citation_result_with_issues(self, mock_llm):
        """Test parsing citation with issues."""
        from cdr.verification.verifier import CitationChecker

        checker = CitationChecker(mock_llm)
        result = checker._parse_citation_result(
            """{
                "accurate": false,
                "issues": ["Effect size misreported", "CI not included"],
                "severity": "major",
                "corrected_text": "The effect was 0.5 (95% CI: 0.3-0.7)"
            }"""
        )

        assert result.accurate is False
        assert len(result.issues) == 2
        assert result.severity == "major"
        assert result.corrected_text is not None


# =============================================================================
# BATCH VERIFICATION TESTS
# =============================================================================


class TestBatchVerification:
    """Tests for batch verification functions."""

    @pytest.fixture
    def sample_verification_results(self):
        """Create sample verification results."""
        return {
            "claim_001": VerificationResult(
                claim_id="claim_001",
                checks=[],
                overall_status=VerificationStatus.VERIFIED,
                overall_confidence=0.9,
            ),
            "claim_002": VerificationResult(
                claim_id="claim_002",
                checks=[],
                overall_status=VerificationStatus.PARTIAL,
                overall_confidence=0.6,
            ),
            "claim_003": VerificationResult(
                claim_id="claim_003",
                checks=[],
                overall_status=VerificationStatus.CONTRADICTED,
                overall_confidence=0.2,
            ),
        }

    def test_batch_verification_result_counts(self, sample_verification_results):
        """Test batch result counting."""
        from cdr.verification.verifier import BatchVerificationResult

        result = BatchVerificationResult(
            results=sample_verification_results,
            verified_count=1,
            partial_count=1,
            contradicted_count=1,
            unverifiable_count=0,
            total_count=3,
            pass_rate=0.67,
            overall_pass=True,
        )

        assert result.verified_count == 1
        assert result.contradicted_count == 1
        assert result.total_count == 3

    def test_batch_contradicted_claims(self, sample_verification_results):
        """Test getting contradicted claims."""
        from cdr.verification.verifier import BatchVerificationResult

        result = BatchVerificationResult(
            results=sample_verification_results,
            verified_count=1,
            partial_count=1,
            contradicted_count=1,
            unverifiable_count=0,
            total_count=3,
            pass_rate=0.67,
            overall_pass=True,
        )

        contradicted = result.contradicted_claims()

        assert len(contradicted) == 1
        assert "claim_003" in contradicted

    def test_batch_needs_review(self, sample_verification_results):
        """Test getting claims needing review."""
        from cdr.verification.verifier import BatchVerificationResult

        result = BatchVerificationResult(
            results=sample_verification_results,
            verified_count=1,
            partial_count=1,
            contradicted_count=1,
            unverifiable_count=0,
            total_count=3,
            pass_rate=0.67,
            overall_pass=True,
        )

        needs_review = result.needs_review()

        assert "claim_002" in needs_review


# =============================================================================
# ENTAILMENT TESTS
# =============================================================================


class TestEntailmentParsing:
    """Tests for entailment response parsing."""

    def test_parse_entailment_valid_json(self):
        """Test parsing valid entailment response."""
        from cdr.verification.verifier import Verifier

        mock_llm = MagicMock()
        verifier = Verifier(mock_llm)

        response = """{
            "entailment": "ENTAILS",
            "confidence": 0.85,
            "supporting_quote": "The data shows...",
            "reasoning": "Direct support",
            "caveats": ["Limited sample"]
        }"""

        result = verifier._parse_entailment_response(response)

        assert result["entailment"] == "ENTAILS"
        assert result["confidence"] == 0.85

    def test_parse_entailment_with_code_block(self):
        """Test parsing response wrapped in code block."""
        from cdr.verification.verifier import Verifier

        mock_llm = MagicMock()
        verifier = Verifier(mock_llm)

        response = """```json
{
    "entailment": "PARTIAL",
    "confidence": 0.6,
    "reasoning": "Partial support"
}
```"""

        result = verifier._parse_entailment_response(response)

        assert result["entailment"] == "PARTIAL"

    def test_parse_entailment_invalid_json(self):
        """Test parsing invalid JSON fallback."""
        from cdr.verification.verifier import Verifier

        mock_llm = MagicMock()
        verifier = Verifier(mock_llm)

        response = "This is not valid JSON"

        result = verifier._parse_entailment_response(response)

        assert result["entailment"] == "NEUTRAL"
        assert result["confidence"] == 0.5


# =============================================================================
# VERIFICATION STATUS TESTS
# =============================================================================


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.PARTIAL.value == "partial"
        assert VerificationStatus.CONTRADICTED.value == "contradicted"
        assert VerificationStatus.UNVERIFIABLE.value == "unverifiable"

    def test_status_from_string(self):
        """Test creating status from string."""
        assert VerificationStatus("verified") == VerificationStatus.VERIFIED
        assert VerificationStatus("contradicted") == VerificationStatus.CONTRADICTED

        with pytest.raises(ValueError):
            VerificationStatus("invalid_status")
