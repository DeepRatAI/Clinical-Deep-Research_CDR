"""
Tests for CDR Evaluation Module

Tests metrics, golden set, and evaluation reports.
"""

import pytest
from datetime import datetime

from cdr.evaluation import (
    CDRMetricsEvaluator,
    EvaluationReport,
    MetricResult,
    MetricStatus,
    evaluate_cdr_output,
    GOLDEN_SET,
    get_golden_set,
    get_question_by_id,
    validate_against_golden_set,
    GoldenSetQuestion,
    ExpectedEvidenceLevel,
)
from cdr.core.schemas import (
    EvidenceClaim,
    Snippet,
    SourceRef,
    VerificationResult,
    VerificationCheck,
)
from cdr.core.enums import VerificationStatus
from cdr.composition.schemas import ComposedHypothesis, ThreatAnalysis


# =============================================================================
# Fixtures - Using correct schema architecture
# =============================================================================


@pytest.fixture
def sample_source_refs():
    """Create sample SourceRef objects."""
    return {
        "snip-1": SourceRef(record_id="rec-1", pmid="12345678"),
        "snip-2": SourceRef(record_id="rec-2", pmid="23456789"),
        "snip-3": SourceRef(record_id="rec-3", pmid="34567890"),
        "snip-4": SourceRef(record_id="rec-4", pmid="45678901"),
    }


@pytest.fixture
def sample_snippets(sample_source_refs):
    """Create sample snippets with proper SourceRef objects."""
    return [
        Snippet(
            snippet_id="snip-1",
            text="Aspirin significantly reduced MACE (HR 0.80, 95% CI 0.72-0.89) in randomized trials.",
            source_ref=sample_source_refs["snip-1"],
        ),
        Snippet(
            snippet_id="snip-2",
            text="Secondary prevention with aspirin showed consistent benefit across multiple populations.",
            source_ref=sample_source_refs["snip-2"],
        ),
        Snippet(
            snippet_id="snip-3",
            text="Risk reduction of 18-22% observed across trials for major cardiovascular events.",
            source_ref=sample_source_refs["snip-3"],
        ),
        Snippet(
            snippet_id="snip-4",
            text="GI bleeding increased (OR 1.5, 95% CI 1.2-1.8) with aspirin therapy.",
            source_ref=sample_source_refs["snip-4"],
        ),
    ]


@pytest.fixture
def sample_claims():
    """Create sample evidence claims (no verification_status - that's separate)."""
    return [
        EvidenceClaim(
            claim_id="claim-1",
            claim_text="Aspirin reduces cardiovascular events in secondary prevention populations.",
            certainty="high",
            supporting_snippet_ids=["snip-1", "snip-2"],
        ),
        EvidenceClaim(
            claim_id="claim-2",
            claim_text="The effect size is approximately 20% relative risk reduction for MACE.",
            certainty="moderate",
            supporting_snippet_ids=["snip-3"],
        ),
        EvidenceClaim(
            claim_id="claim-3",
            claim_text="Bleeding risk increases significantly with chronic aspirin use.",
            certainty="high",
            supporting_snippet_ids=["snip-4"],
        ),
    ]


@pytest.fixture
def sample_verification_results(sample_source_refs):
    """Create sample verification results (separate from claims per architecture)."""
    return {
        "claim-1": VerificationResult(
            claim_id="claim-1",
            checks=[
                VerificationCheck(
                    claim_id="claim-1",
                    source_ref=sample_source_refs["snip-1"],
                    status=VerificationStatus.VERIFIED,
                    confidence=0.95,
                )
            ],
            overall_status=VerificationStatus.VERIFIED,
            overall_confidence=0.95,
        ),
        "claim-2": VerificationResult(
            claim_id="claim-2",
            checks=[
                VerificationCheck(
                    claim_id="claim-2",
                    source_ref=sample_source_refs["snip-3"],
                    status=VerificationStatus.VERIFIED,
                    confidence=0.85,
                )
            ],
            overall_status=VerificationStatus.VERIFIED,
            overall_confidence=0.85,
        ),
        "claim-3": VerificationResult(
            claim_id="claim-3",
            checks=[
                VerificationCheck(
                    claim_id="claim-3",
                    source_ref=sample_source_refs["snip-4"],
                    status=VerificationStatus.PARTIAL,
                    confidence=0.70,
                )
            ],
            overall_status=VerificationStatus.PARTIAL,
            overall_confidence=0.70,
        ),
    }


@pytest.fixture
def sample_hypotheses():
    """Create sample composed hypotheses for testing."""
    return [
        ComposedHypothesis(
            hypothesis_id="hyp-1",
            hypothesis_text="If aspirin reduces platelet aggregation AND platelet aggregation contributes to plaque rupture, THEN aspirin may reduce acute coronary events via plaque stabilization",
            source_claim_ids=["claim-1", "claim-2"],
            strength="moderate",
            confidence_score=0.7,
            threat_analysis=ThreatAnalysis(
                rival_hypotheses=["Aspirin effect may be via other mechanisms"],
                uncontrolled_confounders=["concurrent statin use"],
            ),
        ),
    ]


# =============================================================================
# MetricResult Tests
# =============================================================================


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = MetricResult(
            name="test_metric",
            value=0.95,
            threshold=0.80,
            status=MetricStatus.PASS,
            details="95% coverage achieved",
        )
        d = result.to_dict()
        assert d["name"] == "test_metric"
        assert d["value"] == 0.95
        assert d["threshold"] == 0.80
        assert d["status"] == "pass"
        assert d["details"] == "95% coverage achieved"


# =============================================================================
# EvaluationReport Tests
# =============================================================================


class TestEvaluationReport:
    """Tests for EvaluationReport dataclass."""

    def test_add_metric_pass(self):
        """Adding passing metric keeps overall_pass True."""
        report = EvaluationReport(run_id="test-run", dod_level=3)
        report.add_metric(
            MetricResult(
                name="test",
                value=0.9,
                threshold=0.8,
                status=MetricStatus.PASS,
            )
        )
        assert report.overall_pass is True

    def test_add_metric_fail(self):
        """Adding failing metric sets overall_pass False."""
        report = EvaluationReport(run_id="test-run", dod_level=3)
        report.add_metric(
            MetricResult(
                name="test",
                value=0.5,
                threshold=0.8,
                status=MetricStatus.FAIL,
            )
        )
        assert report.overall_pass is False

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = EvaluationReport(
            run_id="test-run",
            dod_level=3,
            summary="All passed",
        )
        d = report.to_dict()
        assert d["run_id"] == "test-run"
        assert d["dod_level"] == 3
        assert d["overall_pass"] is True
        assert "metrics" in d


# =============================================================================
# CDRMetricsEvaluator Tests
# =============================================================================


class TestCDRMetricsEvaluator:
    """Tests for CDRMetricsEvaluator class."""

    def test_evaluate_snippet_coverage_all_covered(self, sample_claims):
        """All claims with snippets = 100% coverage."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_snippet_coverage(sample_claims)
        assert result.value == 1.0
        assert result.status == MetricStatus.PASS

    def test_evaluate_snippet_coverage_empty(self):
        """No claims = N/A."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_snippet_coverage([])
        assert result.status == MetricStatus.NOT_APPLICABLE

    def test_evaluate_verification_coverage_high(self, sample_claims, sample_verification_results):
        """High verification coverage passes (PARTIAL counts as passed)."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_verification_coverage(
            sample_claims, sample_verification_results
        )
        # 3 verified out of 3 = 100% (PARTIAL counts as passed per VerificationResult.passed)
        assert result.value == pytest.approx(1.0, rel=0.01)

    def test_evaluate_verification_coverage_all_verified(self, sample_source_refs):
        """100% verified passes."""
        claims = [
            EvidenceClaim(
                claim_id="c1",
                claim_text="This is a verified claim with enough characters to pass validation.",
                certainty="high",
                supporting_snippet_ids=["s1"],
            ),
        ]
        verification_results = {
            "c1": VerificationResult(
                claim_id="c1",
                checks=[
                    VerificationCheck(
                        claim_id="c1",
                        source_ref=sample_source_refs["snip-1"],
                        status=VerificationStatus.VERIFIED,
                        confidence=0.95,
                    )
                ],
                overall_status=VerificationStatus.VERIFIED,
                overall_confidence=0.95,
            ),
        }
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_verification_coverage(claims, verification_results)
        assert result.value == 1.0
        assert result.status == MetricStatus.PASS

    def test_evaluate_verification_coverage_none_results(self, sample_claims):
        """No verification results = 0% coverage (WARN at this threshold)."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_verification_coverage(sample_claims, None)
        assert result.value == 0.0
        # 0% is below threshold but metrics may return WARN depending on implementation
        assert result.status in (MetricStatus.FAIL, MetricStatus.WARN)

    def test_evaluate_claims_evidence_ratio_good(self, sample_claims, sample_snippets):
        """Good ratio (claims ≤ 1.2 * snippets) passes."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_claims_evidence_ratio(sample_claims, sample_snippets)
        # 3 claims / 4 snippets = 0.75
        assert result.value == 0.75
        assert result.status == MetricStatus.PASS

    def test_evaluate_claims_evidence_ratio_bad(self, sample_snippets, sample_source_refs):
        """Too many claims vs snippets fails."""
        # 6 claims for 4 snippets = 1.5 ratio
        claims = [
            EvidenceClaim(
                claim_id=f"c{i}",
                claim_text=f"This is claim number {i} with sufficient length to pass validation.",
                certainty="moderate",
                supporting_snippet_ids=["snip-1"],
            )
            for i in range(6)
        ]
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_claims_evidence_ratio(claims, sample_snippets)
        assert result.value == 1.5
        assert result.status == MetricStatus.FAIL

    def test_evaluate_composition_rate_with_hypotheses(self, sample_hypotheses):
        """Having hypotheses passes composition rate."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_composition_rate(sample_hypotheses)
        assert result.value == 1.0
        assert result.status == MetricStatus.PASS

    def test_evaluate_composition_rate_no_hypotheses(self):
        """No hypotheses warns (soft expectation)."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_composition_rate([])
        assert result.value == 0.0
        assert result.status == MetricStatus.WARN

    def test_evaluate_context_precision(self, sample_claims, sample_snippets):
        """Context precision measures snippet usage."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_context_precision(sample_snippets, sample_claims)
        # All 4 snippets are referenced
        assert result.value == 1.0
        assert result.status == MetricStatus.PASS

    def test_evaluate_answer_faithfulness(self, sample_claims, sample_verification_results):
        """Faithfulness measures grounded claims (VERIFIED and PARTIAL both count)."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_answer_faithfulness(sample_claims, sample_verification_results)
        # All 3 claims pass (PARTIAL counts as grounded per VerificationResult.passed)
        # So 3/3 = 1.0
        assert result.value == pytest.approx(1.0, rel=0.01)
        assert result.status == MetricStatus.PASS  # Above 0.8 threshold

    def test_evaluate_answer_faithfulness_no_verification(self, sample_claims):
        """Faithfulness without verification = based on snippets only."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_answer_faithfulness(sample_claims, None)
        # All claims have snippets, so 100%
        assert result.value == 1.0

    def test_evaluate_citation_accuracy_all_valid(self, sample_claims, sample_snippets):
        """All citations reference existing snippets."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_citation_accuracy(sample_claims, sample_snippets)
        # All snippet IDs (snip-1 through snip-4) exist
        assert result.value == 1.0
        assert result.status == MetricStatus.PASS

    def test_evaluate_citation_accuracy_invalid_refs(self, sample_snippets):
        """Some citations reference non-existent snippets."""
        claims = [
            EvidenceClaim(
                claim_id="c1",
                claim_text="This claim references both valid and invalid snippets.",
                certainty="high",
                supporting_snippet_ids=["snip-1", "snip-999"],  # snip-999 doesn't exist
            ),
        ]
        evaluator = CDRMetricsEvaluator(dod_level=3)
        result = evaluator.evaluate_citation_accuracy(claims, sample_snippets)
        assert result.value == 0.5
        assert result.status == MetricStatus.FAIL

    def test_evaluate_run_full(
        self,
        sample_claims,
        sample_snippets,
        sample_hypotheses,
        sample_verification_results,
    ):
        """Full evaluation returns comprehensive report."""
        report = evaluate_cdr_output(
            run_id="test-run",
            claims=sample_claims,
            snippets=sample_snippets,
            hypotheses=sample_hypotheses,
            verification_results=sample_verification_results,
            dod_level=3,
        )
        assert report.run_id == "test-run"
        assert report.dod_level == 3
        assert len(report.metrics) >= 5  # At least 5 metrics evaluated
        assert "DoD 3" in report.summary


# =============================================================================
# Golden Set Tests
# =============================================================================


class TestGoldenSet:
    """Tests for Golden Set functionality."""

    def test_golden_set_has_5_questions(self):
        """Golden set contains 5 benchmark questions."""
        assert len(GOLDEN_SET) == 5

    def test_get_golden_set(self):
        """get_golden_set returns all questions."""
        questions = get_golden_set()
        assert len(questions) == 5
        assert all(isinstance(q, GoldenSetQuestion) for q in questions)

    def test_get_question_by_id_exists(self):
        """Getting existing question by ID works."""
        q = get_question_by_id("GS-001")
        assert q is not None
        assert q.id == "GS-001"
        assert "aspirin" in q.question.lower()

    def test_get_question_by_id_not_exists(self):
        """Getting non-existent question returns None."""
        q = get_question_by_id("GS-999")
        assert q is None

    def test_golden_set_question_structure(self):
        """All questions have required fields."""
        for q in GOLDEN_SET:
            assert q.id.startswith("GS-")
            assert len(q.question) > 10
            assert len(q.population) > 5
            assert len(q.intervention) > 5
            assert len(q.outcome) > 5
            assert q.expected_evidence_level in ExpectedEvidenceLevel
            assert q.expected_min_studies > 0

    def test_golden_set_composition_questions(self):
        """At least 2 questions expect composition."""
        composition_qs = [q for q in GOLDEN_SET if q.composition_expected]
        assert len(composition_qs) >= 2

    def test_validate_against_golden_set_pass(self):
        """Validation passes for good results."""
        result = validate_against_golden_set(
            question_id="GS-001",
            claims_count=10,
            snippets_count=15,
            verification_coverage=0.95,
            studies_found=12,
            hypotheses_count=0,
        )
        assert result["valid"] is True
        assert result["checks"]["min_studies"]["passed"] is True
        assert result["checks"]["verification_coverage"]["passed"] is True

    def test_validate_against_golden_set_fail_studies(self):
        """Validation fails for insufficient studies."""
        result = validate_against_golden_set(
            question_id="GS-001",
            claims_count=5,
            snippets_count=5,
            verification_coverage=0.9,
            studies_found=2,  # Expected: 10
        )
        assert result["valid"] is False
        assert result["checks"]["min_studies"]["passed"] is False

    def test_validate_against_golden_set_unknown_id(self):
        """Validation returns error for unknown question."""
        result = validate_against_golden_set(
            question_id="GS-999",
            claims_count=5,
            snippets_count=5,
            verification_coverage=0.9,
            studies_found=10,
        )
        assert result["valid"] is False
        assert "error" in result


# =============================================================================
# DoD Level Tests
# =============================================================================


class TestDoDLevels:
    """Tests for different DoD level thresholds."""

    def test_dod2_thresholds(self):
        """DoD 2 has lower thresholds."""
        evaluator = CDRMetricsEvaluator(dod_level=2)
        assert evaluator.thresholds.get("verification_coverage") == 0.80

    def test_dod3_thresholds(self):
        """DoD 3 has higher thresholds."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        assert evaluator.thresholds.get("verification_coverage") == 0.95

    def test_dod3_includes_composition_rate(self):
        """DoD 3 evaluates composition rate."""
        evaluator = CDRMetricsEvaluator(dod_level=3)
        assert "composition_emitted_rate" in evaluator.thresholds

    def test_dod2_excludes_composition_rate(self):
        """DoD 2 doesn't require composition."""
        evaluator = CDRMetricsEvaluator(dod_level=2)
        assert "composition_emitted_rate" not in evaluator.thresholds
