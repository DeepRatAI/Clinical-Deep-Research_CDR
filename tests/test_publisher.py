"""
Tests for CDR Publisher Layer.

MEDIUM-5 fix: Tests for Publisher alignment with current schemas.
Refs: CDR_Integral_Audit_2026-01-20.md MEDIUM-5
"""

import pytest
from unittest.mock import MagicMock

from cdr.core.enums import RoB2Domain, RoB2Judgment, GRADECertainty
from cdr.core.schemas import (
    CDRState,
    PRISMACounts,
    RoB2DomainResult,
    RoB2Result,
    EvidenceClaim,
    PICO,
)
from cdr.publisher.publisher import Publisher


class TestPublisherSchemaAlignment:
    """Tests for Publisher alignment with current schemas."""

    @pytest.fixture
    def sample_prisma_counts(self):
        """Create sample PRISMACounts with current schema fields."""
        return PRISMACounts(
            records_identified=500,
            records_from_pubmed=400,
            records_from_clinical_trials=100,
            duplicates_removed=50,
            records_screened=450,
            records_excluded=350,
            reports_sought=100,
            reports_not_retrieved=10,
            reports_assessed=90,
            reports_excluded=40,
            studies_included=50,
        )

    @pytest.fixture
    def sample_rob2_results(self):
        """Create sample RoB2 results as list (not dict)."""
        return [
            RoB2Result(
                record_id="rec_001",
                overall_judgment=RoB2Judgment.LOW,
                overall_rationale="Well-conducted RCT",
                domains=[
                    RoB2DomainResult(
                        domain=RoB2Domain.RANDOMIZATION,
                        judgment=RoB2Judgment.LOW,
                        rationale="Adequate randomization",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.DEVIATIONS,
                        judgment=RoB2Judgment.LOW,
                        rationale="ITT analysis",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.MISSING_DATA,
                        judgment=RoB2Judgment.LOW,
                        rationale="Low attrition",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.MEASUREMENT,
                        judgment=RoB2Judgment.LOW,
                        rationale="Blinded assessment",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.SELECTION,
                        judgment=RoB2Judgment.LOW,
                        rationale="Pre-registered protocol",
                    ),
                ],
            ),
            RoB2Result(
                record_id="rec_002",
                overall_judgment=RoB2Judgment.SOME_CONCERNS,
                overall_rationale="Some methodological issues",
                domains=[
                    RoB2DomainResult(
                        domain=RoB2Domain.RANDOMIZATION,
                        judgment=RoB2Judgment.SOME_CONCERNS,
                        rationale="Unclear allocation concealment",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.DEVIATIONS,
                        judgment=RoB2Judgment.LOW,
                        rationale="ITT analysis",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.MISSING_DATA,
                        judgment=RoB2Judgment.LOW,
                        rationale="Complete data",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.MEASUREMENT,
                        judgment=RoB2Judgment.LOW,
                        rationale="Objective outcome",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.SELECTION,
                        judgment=RoB2Judgment.LOW,
                        rationale="Protocol available",
                    ),
                ],
            ),
        ]

    def test_publisher_initializes(self):
        """Test Publisher can be instantiated."""
        publisher = Publisher(output_dir="./test_reports")
        # output_dir is converted to Path
        assert str(publisher.output_dir) == "test_reports"

    def test_build_rob2_summary_accepts_list(self, sample_rob2_results):
        """Test _build_rob2_summary accepts list of RoB2Result."""
        publisher = Publisher()

        summary = publisher._build_rob2_summary(sample_rob2_results)

        assert "rec_001" in summary
        assert "rec_002" in summary
        assert "low" in summary
        assert "some_concerns" in summary

    def test_build_rob2_summary_empty_list(self):
        """Test _build_rob2_summary handles empty list."""
        publisher = Publisher()

        summary = publisher._build_rob2_summary([])

        assert "No risk of bias assessment" in summary

    def test_build_appendix_rob2_accepts_list(self, sample_rob2_results):
        """Test _build_appendix_rob2 accepts list of RoB2Result."""
        publisher = Publisher()

        appendix = publisher._build_appendix_rob2(sample_rob2_results)

        assert "rec_001" in appendix
        assert "rec_002" in appendix
        assert "randomization_process" in appendix
        assert "low" in appendix

    def test_build_prisma_text_uses_correct_fields(self, sample_prisma_counts):
        """Test _build_prisma_text uses correct PRISMACounts fields."""
        publisher = Publisher()

        prisma_text = publisher._build_prisma_text(sample_prisma_counts)

        # Should use new field names
        assert "500" in prisma_text  # records_identified
        assert "450" in prisma_text  # records_screened
        assert "90" in prisma_text  # reports_assessed
        assert "50" in prisma_text  # studies_included

    def test_prisma_counts_schema_has_correct_fields(self):
        """Test PRISMACounts has expected field names."""
        # Verify field names match what Publisher expects
        counts = PRISMACounts(
            records_identified=100,
            records_screened=80,
            reports_assessed=60,
            studies_included=40,
        )

        assert hasattr(counts, "records_identified")
        assert hasattr(counts, "records_screened")
        assert hasattr(counts, "reports_assessed")
        assert hasattr(counts, "studies_included")
        # These old fields should NOT exist
        assert not hasattr(counts, "identified")
        assert not hasattr(counts, "screened")
        assert not hasattr(counts, "full_text_assessed")
        assert not hasattr(counts, "included")


class TestPublisherFormatting:
    """Tests for Publisher formatting helpers."""

    def test_build_grade_table(self):
        """Test GRADE table building."""
        publisher = Publisher()

        claims = [
            EvidenceClaim(
                claim_id="claim_001",
                claim_text="Treatment X reduces symptoms by 50%",
                certainty=GRADECertainty.MODERATE,
                supporting_snippet_ids=["snip_001", "snip_002"],
            ),
        ]

        table = publisher._build_grade_table(claims)

        assert "claim_001" in table
        assert "moderate" in table
        assert "2" in table  # 2 supporting snippets

    def test_build_grade_table_empty(self):
        """Test GRADE table with no claims."""
        publisher = Publisher()

        table = publisher._build_grade_table([])

        assert "No claims synthesized" in table

    def test_format_search_strategy(self):
        """Test search strategy formatting."""
        from cdr.core.schemas import SearchPlan
        from cdr.core.enums import RunStatus

        pico = PICO(
            population="Adults with heart disease",
            intervention="Aspirin",
            comparator="Placebo",
            outcome="Heart attack prevention",
        )

        state = CDRState(
            run_id="test_run",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            pico=pico,
            search_plan=SearchPlan(
                pico=pico,
                pubmed_query="aspirin AND heart attack AND randomized controlled trial",
                ct_gov_query="aspirin AND heart attack",
            ),
        )

        publisher = Publisher()
        strategy = publisher._format_search_strategy(state)

        assert "PubMed" in strategy
        assert "ClinicalTrials.gov" in strategy
        assert "aspirin" in strategy

    def test_build_key_findings(self):
        """Test key findings building."""
        publisher = Publisher()

        claims = [
            EvidenceClaim(
                claim_id="c1",
                claim_text="Finding one with strong evidence",
                certainty=GRADECertainty.HIGH,
                supporting_snippet_ids=["s1"],
            ),
            EvidenceClaim(
                claim_id="c2",
                claim_text="Finding two with moderate evidence",
                certainty=GRADECertainty.MODERATE,
                supporting_snippet_ids=["s2"],
            ),
        ]

        findings = publisher._build_key_findings(claims)

        assert "HIGH" in findings
        assert "MODERATE" in findings
        assert "Finding one" in findings
