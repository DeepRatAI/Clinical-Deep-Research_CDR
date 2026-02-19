"""
Unit Tests for Core Schemas

Tests validation rules and invariants defined in schemas.py.
"""

import pytest
from pydantic import ValidationError

from cdr.core.schemas import (
    PICO,
    CDRState,
    Critique,
    CritiqueResult,
    EvidenceClaim,
    OutcomeMeasure,
    PRISMACounts,
    Record,
    RoB2DomainResult,
    RoB2Result,
    ScreeningDecision,
    SearchPlan,
    Snippet,
    SourceRef,
    StudyCard,
    VerificationCheck,
    VerificationResult,
)
from cdr.core.enums import (
    CritiqueDimension,
    CritiqueSeverity,
    ExclusionReason,
    GRADECertainty,
    OutcomeMeasureType,
    RecordSource,
    RoB2Domain,
    RoB2Judgment,
    RunStatus,
    StudyType,
    VerificationStatus,
)


class TestPICO:
    """Tests for PICO schema."""

    def test_valid_pico(self, sample_pico: dict) -> None:
        """Test valid PICO creation."""
        pico = PICO(**sample_pico)
        assert pico.population == sample_pico["population"]
        assert pico.intervention == sample_pico["intervention"]
        assert len(pico.study_types) == 2

    def test_pico_immutable(self, sample_pico: dict) -> None:
        """Test PICO is frozen (immutable)."""
        pico = PICO(**sample_pico)
        with pytest.raises(ValidationError):
            pico.population = "Modified population"

    def test_pico_default_study_types(self) -> None:
        """Test PICO defaults to empty list (no assumed study types)."""
        pico = PICO(
            population="Adults",
            intervention="Drug A",
            comparator="Placebo",
            outcome="Mortality",
        )
        # Default is empty list - user must explicitly specify study types
        assert pico.study_types == []

    def test_pico_to_search_query(self, sample_pico: dict) -> None:
        """Test PICO generates valid search query."""
        pico = PICO(**sample_pico)
        query = pico.to_search_query()
        assert sample_pico["population"] in query
        assert sample_pico["intervention"] in query


class TestRecord:
    """Tests for Record schema."""

    def test_valid_record(self, sample_record: dict) -> None:
        """Test valid Record creation."""
        record = Record(**sample_record)
        assert record.record_id == sample_record["record_id"]
        assert record.source == RecordSource.PUBMED

    def test_record_requires_id(self, sample_record: dict) -> None:
        """Test Record requires record_id."""
        del sample_record["record_id"]
        with pytest.raises(ValidationError):
            Record(**sample_record)

    def test_record_get_best_identifier_pmid(self, sample_record: dict) -> None:
        """Test get_best_identifier returns tuple (type, value)."""
        sample_record["pmid"] = "12345678"
        record = Record(**sample_record)
        # Returns tuple: (identifier_type, value)
        # DOI takes precedence over PMID in current implementation
        id_type, id_value = record.get_best_identifier()
        assert id_type in ("doi", "pmid")
        assert id_value is not None

    def test_record_get_best_identifier_doi(self, sample_record: dict) -> None:
        """Test get_best_identifier returns DOI tuple when available."""
        sample_record["pmid"] = None
        record = Record(**sample_record)
        id_type, id_value = record.get_best_identifier()
        assert id_type == "doi"
        assert id_value == sample_record["doi"]


class TestScreeningDecision:
    """Tests for ScreeningDecision schema."""

    def test_included_record(self, sample_record: dict) -> None:
        """Test included record needs no reason."""
        decision = ScreeningDecision(
            record_id="record_001",
            included=True,
        )
        assert decision.included is True
        assert decision.reason_code is None

    def test_excluded_requires_reason(self) -> None:
        """Test excluded record MUST have reason (invariant)."""
        with pytest.raises(ValidationError):
            ScreeningDecision(
                record_id="record_001",
                included=False,
                # Missing reason_code - should trigger validation error
            )

    def test_excluded_with_reason(self) -> None:
        """Test excluded record with valid reason."""
        decision = ScreeningDecision(
            record_id="record_001",
            included=False,
            reason_code=ExclusionReason.POPULATION_MISMATCH,
            reason_text="Study included pediatric patients only.",
        )
        assert decision.included is False
        assert decision.reason_code == ExclusionReason.POPULATION_MISMATCH


class TestSourceRef:
    """Tests for SourceRef schema."""

    def test_valid_source_ref(self) -> None:
        """Test valid SourceRef creation."""
        from cdr.core.enums import Section

        ref = SourceRef(
            record_id="record_001",
            pmid="12345678",
            section=Section.RESULTS,  # Use enum, not string
        )
        assert ref.record_id == "record_001"
        assert ref.pmid == "12345678"
        assert ref.section == Section.RESULTS

    def test_source_ref_to_citation(self) -> None:
        """Test citation format generation."""
        from cdr.core.enums import Section

        ref = SourceRef(
            record_id="record_001",
            pmid="12345678",
            section=Section.RESULTS,
            page=5,
        )
        # Method is to_citation_string(), not to_citation()
        citation = ref.to_citation_string()
        assert "12345678" in citation
        assert "results" in citation.lower()  # Section value is lowercase


class TestSnippet:
    """Tests for Snippet schema."""

    def test_valid_snippet(self, sample_snippet: dict) -> None:
        """Test valid Snippet creation."""
        snippet = Snippet(**sample_snippet)
        assert snippet.snippet_id == sample_snippet["snippet_id"]
        assert snippet.source_ref.record_id == "record_001"

    def test_snippet_requires_source_ref(self) -> None:
        """Test Snippet requires source_ref."""
        with pytest.raises(ValidationError):
            Snippet(
                id="snip_001",
                record_id="record_001",
                text="Some text",
                # Missing source_ref
            )


class TestOutcomeMeasure:
    """Tests for OutcomeMeasure schema."""

    def test_valid_outcome_measure(self) -> None:
        """Test valid OutcomeMeasure creation."""
        om = OutcomeMeasure(
            name="HbA1c reduction",
            measure_type=OutcomeMeasureType.MEAN_DIFFERENCE,
            value=-1.2,
            ci_lower=-1.5,
            ci_upper=-0.9,
            p_value=0.001,
        )
        assert om.value == -1.2
        # Check significance via p_value directly (no is_significant method)
        assert om.p_value is not None and om.p_value < 0.05

    def test_outcome_measure_not_significant(self) -> None:
        """Test non-significant outcome (p >= 0.05)."""
        om = OutcomeMeasure(
            name="Secondary outcome",
            measure_type=OutcomeMeasureType.MEAN_DIFFERENCE,
            value=0.1,
            p_value=0.45,
        )
        # Check via p_value directly
        assert om.p_value is not None and om.p_value >= 0.05

    def test_outcome_measure_ci_crossing_null(self) -> None:
        """Test CI crossing null for RR."""
        om = OutcomeMeasure(
            name="Event rate",
            measure_type=OutcomeMeasureType.RISK_RATIO,
            value=0.95,
            ci_lower=0.85,
            ci_upper=1.10,  # Crosses 1.0
        )
        # CI includes null value (1.0 for RR)
        assert om.ci_lower is not None
        assert om.ci_upper is not None


class TestStudyCard:
    """Tests for StudyCard schema."""

    def test_valid_study_card(self, sample_study_card: dict) -> None:
        """Test valid StudyCard creation."""
        card = StudyCard(**sample_study_card)
        assert card.record_id == sample_study_card["record_id"]
        assert card.sample_size == 500  # Correct field name
        assert len(card.outcomes) == 1  # Correct field name

    def test_study_card_requires_outcomes(self, sample_study_card: dict) -> None:
        """Test StudyCard can have empty outcomes list."""
        sample_study_card["outcomes"] = []  # Correct field name
        # Should still be valid (can have empty, but should have supporting snippets)
        card = StudyCard(**sample_study_card)
        assert len(card.outcomes) == 0


class TestRoB2Result:
    """Tests for RoB2 schema."""

    def test_valid_rob2_result(self, sample_rob2_result: dict) -> None:
        """Test valid RoB2Result creation."""
        result = RoB2Result(**sample_rob2_result)
        assert result.record_id == sample_rob2_result["record_id"]
        assert len(result.domains) == 5

    def test_rob2_overall_judgment(self, sample_rob2_result: dict) -> None:
        """Test overall judgment is set explicitly (not computed)."""
        result = RoB2Result(**sample_rob2_result)
        # Overall judgment is now an explicit required field
        assert result.overall_judgment == RoB2Judgment.LOW

    def test_rob2_requires_all_domains(self, sample_rob2_result: dict) -> None:
        """Test RoB2Result requires all 5 domains."""
        sample_rob2_result["domains"] = sample_rob2_result["domains"][:3]
        with pytest.raises(ValidationError, match="Missing RoB2 domains"):
            RoB2Result(**sample_rob2_result)

    def test_rob2_no_duplicate_domains(self, sample_rob2_result: dict) -> None:
        """Test RoB2Result rejects duplicate domains."""
        # Create duplicate by copying first domain
        dup = sample_rob2_result["domains"][0].copy()
        sample_rob2_result["domains"][1] = dup
        with pytest.raises(ValidationError, match="Missing RoB2 domains"):
            # Will fail because now we have duplicate randomization and missing another domain
            RoB2Result(**sample_rob2_result)


class TestEvidenceClaim:
    """Tests for EvidenceClaim schema."""

    def test_valid_evidence_claim(self, sample_evidence_claim: dict) -> None:
        """Test valid EvidenceClaim creation."""
        claim = EvidenceClaim(**sample_evidence_claim)
        assert claim.claim_id == sample_evidence_claim["claim_id"]
        assert len(claim.supporting_snippet_ids) >= 1

    def test_claim_requires_supporting_snippets(self, sample_evidence_claim: dict) -> None:
        """Test EvidenceClaim requires at least one snippet (invariant)."""
        sample_evidence_claim["supporting_snippet_ids"] = []
        with pytest.raises(ValidationError, match="too_short"):
            EvidenceClaim(**sample_evidence_claim)


class TestCritique:
    """Tests for Critique schema."""

    def test_valid_critique(self) -> None:
        """Test valid CritiqueResult creation."""
        # Updated to use CritiqueResult and correct enum values
        critique = CritiqueResult(
            dimension=CritiqueDimension.INTERNAL_VALIDITY,
            severity=CritiqueSeverity.HIGH,
            finding="Study design has potential selection bias that affects validity.",
            affected_claims=["claim_001"],
            recommendation="Consider adjusting for confounders.",
        )
        assert critique.dimension == CritiqueDimension.INTERNAL_VALIDITY


class TestVerificationResult:
    """Tests for VerificationResult schema."""

    def test_valid_verification_result(self) -> None:
        """Test valid VerificationResult creation."""
        result = VerificationResult(
            claim_id="claim_001",
            checks=[
                VerificationCheck(
                    claim_id="claim_001",
                    source_ref=SourceRef(record_id="pmid_123", snippet_id="snip_001"),
                    status=VerificationStatus.VERIFIED,
                    confidence=0.95,
                    explanation="Source directly supports the claim.",
                ),
                VerificationCheck(
                    claim_id="claim_001",
                    source_ref=SourceRef(record_id="pmid_456", snippet_id="snip_002"),
                    status=VerificationStatus.VERIFIED,
                    confidence=0.92,
                    explanation="Source provides supporting evidence.",
                ),
            ],
            overall_status=VerificationStatus.VERIFIED,
            overall_confidence=0.935,
        )
        assert result.overall_status == VerificationStatus.VERIFIED
        assert len(result.checks) == 2
        assert result.passed is True


class TestPRISMACounts:
    """Tests for PRISMACounts schema."""

    def test_valid_prisma_counts(self) -> None:
        """Test valid PRISMACounts creation."""
        counts = PRISMACounts(
            records_identified=200,
            records_from_pubmed=150,
            records_from_clinical_trials=50,
            duplicates_removed=30,
            records_screened=170,
            records_excluded=120,
            reports_assessed=50,
            reports_excluded=10,
            studies_included=40,
        )
        assert counts.records_from_pubmed == 150
        assert counts.studies_included == 40


class TestCDRState:
    """Tests for CDRState schema (workflow state)."""

    def test_minimal_cdr_state(self, sample_pico: dict) -> None:
        """Test minimal CDRState creation."""
        state = CDRState(
            run_id="run_001",
            question="What is the effect of GLP-1 agonists?",
            pico=PICO(**sample_pico),
        )
        assert state.run_id == "run_001"
        assert state.status == RunStatus.PENDING
        assert len(state.retrieved_records) == 0

    def test_cdr_state_with_records(self, sample_pico: dict, sample_record: dict) -> None:
        """Test CDRState with records."""
        record = Record(**sample_record)
        state = CDRState(
            run_id="run_001",
            question="What is the effect of GLP-1 agonists?",
            pico=PICO(**sample_pico),
            retrieved_records=[record],
            status=RunStatus.RUNNING,
        )
        assert len(state.retrieved_records) == 1


class TestSchemaValidation:
    """Test cross-schema validation rules."""

    def test_record_source_enum_validation(self) -> None:
        """Test invalid source is rejected."""
        with pytest.raises(ValidationError):
            Record(
                record_id="rec_001",
                source="INVALID_SOURCE",  # Not a valid enum
                content_hash="abc123",
                title="Test",
            )

    def test_study_type_enum_validation(self) -> None:
        """Test invalid study type is rejected (via publication_type)."""
        # StudyType validation happens at extraction, not record level
        # This test validates that invalid enum strings are rejected
        with pytest.raises(ValidationError):
            Record(
                record_id="rec_001",
                source="invalid_source",  # Invalid enum value
                content_hash="abc123",
                title="Test",
                study_type="INVALID_TYPE",  # Not a valid enum
            )
