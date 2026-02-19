"""
Tests for Publisher schema synchronization (MEDIUM-5).

Verifies that Publisher correctly uses current schema definitions
and includes new fields like executed_searches and exclusion_reasons.

Refs: CDR_Integral_Audit_2026-01-20.md MEDIUM-5, PRISMA 2020, PRISMA-S (BMJ 2021)
"""

import json
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from cdr.core.enums import ExclusionReason, GRADECertainty, RoB2Judgment, RoB2Domain, StudyType
from cdr.core.schemas import (
    CDRState,
    EvidenceClaim,
    ExecutedSearch,
    OutcomeMeasure,
    PICO,
    PRISMACounts,
    RoB2DomainResult,
    RoB2Result,
    SearchPlan,
    StudyCard,
)
from cdr.publisher.publisher import Publisher


@pytest.fixture
def minimal_pico():
    """Create minimal PICO for testing."""
    return PICO(
        population="Adults with condition",
        intervention="Treatment X",
        comparator="Placebo",
        outcome="Recovery rate",
    )


@pytest.fixture
def mock_synthesis_result():
    """Create mock synthesis result."""
    mock = Mock()
    mock.claims = [
        EvidenceClaim(
            claim_id="claim-1",
            claim_text="Treatment X significantly improves recovery rate in adults",
            certainty=GRADECertainty.MODERATE,
            supporting_snippet_ids=["snip-1"],
        )
    ]
    mock.overall_narrative = "Overall, evidence suggests..."
    mock.heterogeneity_assessment = "Low heterogeneity"
    mock.high_certainty_claims = []
    mock.low_certainty_claims = []
    return mock


class TestPRISMACountsSync:
    """Tests for PRISMACounts field synchronization."""

    def test_publisher_uses_records_identified(self):
        """Publisher correctly uses records_identified field."""
        prisma = PRISMACounts(
            records_identified=100,
            duplicates_removed=10,
            records_screened=90,
            reports_assessed=50,
            studies_included=20,
        )

        publisher = Publisher()
        prisma_text = publisher._build_prisma_text(prisma)

        assert "100" in prisma_text  # records_identified
        assert "90" in prisma_text  # records_screened
        assert "20" in prisma_text  # studies_included

    def test_exclusion_reasons_included_in_prisma(self):
        """Exclusion reasons are included in PRISMA text."""
        prisma = PRISMACounts(
            records_identified=100,
            records_screened=90,
            records_excluded=40,
            exclusion_reasons={
                "wrong_population": 15,
                "wrong_outcome": 10,
                "wrong_study_type": 8,
                "no_comparison": 7,
            },
        )

        publisher = Publisher()
        prisma_text = publisher._build_prisma_text(prisma)

        assert "Exclusion reasons" in prisma_text
        assert "wrong_population: 15" in prisma_text
        assert "wrong_outcome: 10" in prisma_text

    def test_exclusion_reasons_sorted_by_count(self):
        """Exclusion reasons are sorted by count (descending)."""
        prisma = PRISMACounts(
            records_identified=50,
            exclusion_reasons={
                "reason_a": 5,
                "reason_b": 20,
                "reason_c": 10,
            },
        )

        publisher = Publisher()
        prisma_text = publisher._build_prisma_text(prisma)

        # Find positions
        pos_b = prisma_text.find("reason_b")
        pos_c = prisma_text.find("reason_c")
        pos_a = prisma_text.find("reason_a")

        # reason_b (20) should appear before reason_c (10) before reason_a (5)
        assert pos_b < pos_c < pos_a

    def test_empty_exclusion_reasons_omitted(self):
        """Empty exclusion_reasons dict doesn't add section."""
        prisma = PRISMACounts(
            records_identified=50,
            exclusion_reasons={},  # Empty
        )

        publisher = Publisher()
        prisma_text = publisher._build_prisma_text(prisma)

        assert "Exclusion reasons" not in prisma_text


class TestExecutedSearchesSync:
    """Tests for ExecutedSearch field synchronization."""

    def test_executed_searches_in_search_strategy(self, minimal_pico):
        """Executed searches are included in search strategy."""
        state = CDRState(
            question="Test question",
            run_id="test-run",
            search_plan=SearchPlan(
                pico=minimal_pico,
                pubmed_query="hypertension treatment",
                ct_gov_query="hypertension",
            ),
            executed_searches=[
                ExecutedSearch(
                    database="PubMed",
                    query_planned="hypertension treatment",
                    query_executed="hypertension treatment",
                    executed_at=datetime(2026, 1, 20, 10, 30, tzinfo=timezone.utc),
                    results_count=150,
                    results_fetched=100,
                    notes=None,
                ),
                ExecutedSearch(
                    database="ClinicalTrials.gov",
                    query_planned="hypertension treatment drugs",
                    query_executed="hypertension treatment",
                    executed_at=datetime(2026, 1, 20, 10, 31, tzinfo=timezone.utc),
                    results_count=25,
                    results_fetched=25,
                    notes="Query truncated to 6 words",
                ),
            ],
        )

        publisher = Publisher()
        strategy_text = publisher._format_search_strategy(state)

        assert "Executed searches" in strategy_text
        assert "PubMed" in strategy_text
        assert "ClinicalTrials.gov" in strategy_text
        assert "Results found: 150" in strategy_text
        assert "Query truncated" in strategy_text

    def test_executed_searches_in_json_report(self, mock_synthesis_result, minimal_pico):
        """Executed searches are included in JSON report."""
        state = CDRState(
            question="Test question",
            run_id="test-run",
            executed_searches=[
                ExecutedSearch(
                    database="PubMed",
                    query_planned="diabetes treatment",
                    query_executed="diabetes treatment",
                    executed_at=datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc),
                    results_count=200,
                    results_fetched=100,
                    notes=None,
                ),
            ],
            pico=minimal_pico,
            prisma_counts=PRISMACounts(records_identified=200),
        )

        publisher = Publisher()
        json_output = publisher._generate_json(state, mock_synthesis_result, None, None)

        data = json.loads(json_output)

        assert "executed_searches" in data
        assert len(data["executed_searches"]) == 1
        assert data["executed_searches"][0]["database"] == "PubMed"
        assert data["executed_searches"][0]["results_count"] == 200
        assert data["executed_searches"][0]["query_executed"] == "diabetes treatment"

    def test_json_version_updated(self, mock_synthesis_result, minimal_pico):
        """JSON report version updated for PRISMA-S support."""
        state = CDRState(
            question="Test",
            run_id="test-run",
            pico=minimal_pico,
            prisma_counts=PRISMACounts(),
        )

        publisher = Publisher()
        json_output = publisher._generate_json(state, mock_synthesis_result, None, None)

        data = json.loads(json_output)
        assert data["meta"]["version"] == "1.1"

    def test_empty_executed_searches(self, mock_synthesis_result, minimal_pico):
        """Empty executed_searches produces empty array in JSON."""
        state = CDRState(
            question="Test",
            run_id="test-run",
            executed_searches=[],  # Empty
            pico=minimal_pico,
            prisma_counts=PRISMACounts(),
        )

        publisher = Publisher()
        json_output = publisher._generate_json(state, mock_synthesis_result, None, None)

        data = json.loads(json_output)
        assert data["executed_searches"] == []


class TestRoB2ResultsSync:
    """Tests for RoB2Result list handling."""

    @pytest.fixture
    def valid_rob2_domains(self):
        """Create valid RoB2 domains for all required areas."""
        return [
            RoB2DomainResult(
                domain=RoB2Domain.RANDOMIZATION,
                judgment=RoB2Judgment.LOW,
                rationale="Computer-generated sequence with adequate concealment",
            ),
            RoB2DomainResult(
                domain=RoB2Domain.DEVIATIONS,
                judgment=RoB2Judgment.LOW,
                rationale="Participants and investigators blinded throughout trial",
            ),
            RoB2DomainResult(
                domain=RoB2Domain.MISSING_DATA,
                judgment=RoB2Judgment.LOW,
                rationale="Less than 5% attrition in both groups",
            ),
            RoB2DomainResult(
                domain=RoB2Domain.MEASUREMENT,
                judgment=RoB2Judgment.LOW,
                rationale="Standard validated measurement tools used by trained staff",
            ),
            RoB2DomainResult(
                domain=RoB2Domain.SELECTION,
                judgment=RoB2Judgment.LOW,
                rationale="Pre-registered protocol available, all outcomes reported",
            ),
        ]

    def test_rob2_results_as_list(self, mock_synthesis_result, minimal_pico, valid_rob2_domains):
        """RoB2 results are correctly handled as list."""
        state = CDRState(
            question="Test",
            run_id="test-run",
            rob2_results=[
                RoB2Result(
                    record_id="study-1",
                    overall_judgment=RoB2Judgment.LOW,
                    overall_rationale="Low risk overall",
                    domains=valid_rob2_domains,
                ),
            ],
            pico=minimal_pico,
            prisma_counts=PRISMACounts(),
        )

        publisher = Publisher()
        json_output = publisher._generate_json(state, mock_synthesis_result, None, None)

        data = json.loads(json_output)

        assert "rob2_results" in data
        assert len(data["rob2_results"]) == 1
        assert data["rob2_results"][0]["record_id"] == "study-1"
        assert data["rob2_results"][0]["overall_judgment"] == "low"

    def test_rob2_summary_handles_list(self, valid_rob2_domains):
        """RoB2 summary correctly processes list of results."""
        rob2_results = [
            RoB2Result(
                record_id="study-1",
                overall_judgment=RoB2Judgment.LOW,
                overall_rationale="Low risk of bias - all domains assessed as low risk",
                domains=valid_rob2_domains,
            ),
        ]

        publisher = Publisher()
        summary = publisher._build_rob2_summary(rob2_results)

        # Should contain some indication of the result
        assert len(summary) > 0
        assert "study-1" in summary or "low" in summary.lower() or "1" in summary


class TestStudyCardsSync:
    """Tests for StudyCard field synchronization."""

    def test_study_cards_outcomes_in_json(self, mock_synthesis_result, minimal_pico):
        """StudyCard outcomes are correctly serialized in JSON."""
        state = CDRState(
            question="Test",
            run_id="test-run",
            study_cards=[
                StudyCard(
                    record_id="study-1",
                    study_type=StudyType.RCT,
                    sample_size=100,
                    outcomes=[
                        OutcomeMeasure(
                            name="Blood pressure",
                            value=-5.2,
                        ),
                    ],
                    supporting_snippet_ids=["snip-1"],
                ),
            ],
            pico=minimal_pico,
            prisma_counts=PRISMACounts(),
        )

        publisher = Publisher()
        json_output = publisher._generate_json(state, mock_synthesis_result, None, None)

        data = json.loads(json_output)

        assert len(data["included_studies"]) == 1
        assert data["included_studies"][0]["record_id"] == "study-1"
        assert data["included_studies"][0]["sample_size"] == 100
        assert len(data["included_studies"][0]["outcomes"]) == 1
        assert data["included_studies"][0]["outcomes"][0]["name"] == "Blood pressure"
        assert data["included_studies"][0]["outcomes"][0]["value"] == -5.2


class TestEvidenceClaimsSync:
    """Tests for EvidenceClaim field synchronization."""

    def test_claims_supporting_snippets_in_json(self, minimal_pico):
        """Claims supporting_snippet_ids are included in JSON."""
        state = CDRState(
            question="Test",
            run_id="test-run",
            pico=minimal_pico,
            prisma_counts=PRISMACounts(),
        )

        mock_synthesis = Mock()
        mock_synthesis.claims = [
            EvidenceClaim(
                claim_id="claim-1",
                claim_text="This is a test claim with sufficient length to pass validation",
                certainty=GRADECertainty.HIGH,
                supporting_snippet_ids=["snip-1", "snip-2", "snip-3"],
            )
        ]
        mock_synthesis.overall_narrative = "Narrative"
        mock_synthesis.heterogeneity_assessment = "Low"
        mock_synthesis.high_certainty_claims = []
        mock_synthesis.low_certainty_claims = []

        publisher = Publisher()
        json_output = publisher._generate_json(state, mock_synthesis, None, None)

        data = json.loads(json_output)

        assert "claims" in data
        assert len(data["claims"]) == 1
        assert data["claims"][0]["supporting_snippets"] == ["snip-1", "snip-2", "snip-3"]
        assert data["claims"][0]["certainty"] == "high"
