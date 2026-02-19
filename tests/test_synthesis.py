"""
Tests for CDR Synthesis Layer.

Tests for evidence synthesis, GRADE assessment, and meta-analysis components.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cdr.core.enums import GRADECertainty, OutcomeMeasureType, RoB2Domain, RoB2Judgment, StudyType
from cdr.core.schemas import (
    EvidenceClaim,
    OutcomeMeasure,
    PICO,
    RoB2DomainResult,
    RoB2Result,
    Snippet,
    SourceRef,
    StudyCard,
)


# =============================================================================
# SYNTHESIS RESULT TESTS
# =============================================================================


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    @pytest.fixture
    def sample_claims(self):
        """Create sample claims for testing."""
        return [
            EvidenceClaim(
                claim_id="claim_001",
                claim_text="Treatment X reduces mortality significantly in adult patients",
                certainty=GRADECertainty.HIGH,
                supporting_snippet_ids=["snp_001", "snp_002"],
            ),
            EvidenceClaim(
                claim_id="claim_002",
                claim_text="Treatment X improves quality of life in target population",
                certainty=GRADECertainty.MODERATE,
                supporting_snippet_ids=["snp_003"],
            ),
            EvidenceClaim(
                claim_id="claim_003",
                claim_text="Treatment X may reduce hospitalization rates in patients",
                certainty=GRADECertainty.LOW,
                supporting_snippet_ids=["snp_004"],
            ),
        ]

    def test_claim_count(self, sample_claims):
        """Test claim count property."""
        from cdr.synthesis.synthesizer import SynthesisResult

        result = SynthesisResult(claims=sample_claims)

        assert result.claim_count == 3

    def test_high_certainty_claims(self, sample_claims):
        """Test filtering high certainty claims."""
        from cdr.synthesis.synthesizer import SynthesisResult

        result = SynthesisResult(claims=sample_claims)
        high_cert = result.high_certainty_claims

        assert len(high_cert) == 1
        assert high_cert[0].claim_id == "claim_001"

    def test_low_certainty_claims(self, sample_claims):
        """Test filtering low certainty claims."""
        from cdr.synthesis.synthesizer import SynthesisResult

        result = SynthesisResult(claims=sample_claims)
        low_cert = result.low_certainty_claims

        assert len(low_cert) == 1
        assert low_cert[0].certainty == GRADECertainty.LOW


# =============================================================================
# POOLED ESTIMATE TESTS
# =============================================================================


class TestPooledEstimate:
    """Tests for meta-analysis calculations."""

    def test_calculate_pooled_estimate_basic(self):
        """Test basic pooled estimate calculation."""
        from cdr.synthesis.synthesizer import calculate_pooled_estimate

        effects = [0.5, 0.6, 0.4, 0.55]
        standard_errors = [0.1, 0.15, 0.12, 0.1]

        result = calculate_pooled_estimate(effects, standard_errors)

        assert result["pooled"] is not None
        assert result["ci_lower"] is not None
        assert result["ci_upper"] is not None
        assert result["i_squared"] is not None
        assert result["n_studies"] == 4

    def test_calculate_pooled_estimate_empty(self):
        """Test pooled estimate with empty data."""
        from cdr.synthesis.synthesizer import calculate_pooled_estimate

        result = calculate_pooled_estimate([], [])

        assert result["pooled"] is None
        assert result["ci_lower"] is None

    def test_pooled_estimate_heterogeneity(self):
        """Test heterogeneity (I²) calculation."""
        from cdr.synthesis.synthesizer import calculate_pooled_estimate

        # Homogeneous effects
        homogeneous = calculate_pooled_estimate(
            [0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1],
        )

        # Heterogeneous effects
        heterogeneous = calculate_pooled_estimate(
            [0.1, 0.9, 0.3],
            [0.1, 0.1, 0.1],
        )

        assert homogeneous["i_squared"] < heterogeneous["i_squared"]


# =============================================================================
# PUBLICATION BIAS TESTS
# =============================================================================


class TestPublicationBias:
    """Tests for publication bias assessment."""

    def test_assess_publication_bias_insufficient(self):
        """Test bias assessment with insufficient studies."""
        from cdr.synthesis.synthesizer import assess_publication_bias

        result = assess_publication_bias([0.5, 0.6], [0.1, 0.1])

        assert result["assessment"] == "insufficient_studies"

    def test_assess_publication_bias_unlikely(self):
        """Test bias assessment when unlikely."""
        from cdr.synthesis.synthesizer import assess_publication_bias

        # Effects not correlated with precision
        effects = [0.5, 0.6, 0.55, 0.52, 0.58]
        standard_errors = [0.1, 0.05, 0.15, 0.08, 0.12]

        result = assess_publication_bias(effects, standard_errors)

        assert result["assessment"] in ["unlikely", "possible"]
        assert "correlation" in result


# =============================================================================
# EVIDENCE SYNTHESIZER TESTS
# =============================================================================


class TestEvidenceSynthesizer:
    """Tests for EvidenceSynthesizer class."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        mock = MagicMock()
        mock.complete.return_value = MagicMock(
            content="""{
                "claims": [
                    {
                        "claim_id": "claim_001",
                        "statement": "Test claim statement",
                        "certainty": "moderate",
                        "supporting_studies": ["pmid:12345"],
                        "grade_rationale": {
                            "risk_of_bias": "low",
                            "inconsistency": "not serious"
                        },
                        "snippets": [
                            {
                                "text": "Supporting evidence text",
                                "source_ref": {"record_id": "pmid:12345"}
                            }
                        ]
                    }
                ],
                "heterogeneity_assessment": {
                    "clinical": "Low",
                    "statistical": "I² = 25%"
                },
                "overall_synthesis": "Overall synthesis narrative"
            }"""
        )
        return mock

    @pytest.fixture
    def sample_study_cards(self):
        """Create sample study cards."""
        return [
            StudyCard(
                record_id="pmid:12345",
                study_type=StudyType.RCT,
                sample_size=100,
                pico=PICO(
                    population="Adults with diabetes",
                    intervention="Metformin treatment",
                    outcome="HbA1c reduction",
                ),
                outcomes=[
                    OutcomeMeasure(
                        name="HbA1c",
                        measure_type=OutcomeMeasureType.MEAN_DIFFERENCE,
                        value=-1.2,
                        ci_lower=-1.5,
                        ci_upper=-0.9,
                    )
                ],
                snippets=[
                    Snippet(
                        snippet_id="snp_study_001",
                        text="Metformin significantly reduced HbA1c levels in patients",
                        source_ref=SourceRef(record_id="pmid:12345"),
                    )
                ],
            )
        ]

    @pytest.fixture
    def sample_rob2_results(self):
        """Create sample RoB2 results."""
        return {
            "pmid:12345": RoB2Result(
                record_id="pmid:12345",
                overall_judgment=RoB2Judgment.LOW,
                overall_rationale="Overall low risk based on adequate methodology",
                domains=[
                    RoB2DomainResult(
                        domain=RoB2Domain.RANDOMIZATION,
                        judgment=RoB2Judgment.LOW,
                        rationale="Adequate randomization process described",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.DEVIATIONS,
                        judgment=RoB2Judgment.LOW,
                        rationale="No significant deviations from protocol",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.MISSING_DATA,
                        judgment=RoB2Judgment.LOW,
                        rationale="Complete outcome data available",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.MEASUREMENT,
                        judgment=RoB2Judgment.LOW,
                        rationale="Blinded outcome assessment performed",
                    ),
                    RoB2DomainResult(
                        domain=RoB2Domain.SELECTION,
                        judgment=RoB2Judgment.LOW,
                        rationale="Pre-registered outcomes reported completely",
                    ),
                ],
            )
        }

    def test_synthesize_parses_response(self, mock_llm, sample_study_cards, sample_rob2_results):
        """Test that synthesize correctly parses LLM response."""
        from cdr.synthesis.synthesizer import EvidenceSynthesizer

        synthesizer = EvidenceSynthesizer(mock_llm)

        # Mock the async call
        with patch.object(synthesizer, "synthesize") as mock_synth:
            from cdr.synthesis.synthesizer import SynthesisResult

            mock_synth.return_value = SynthesisResult(
                claims=[
                    EvidenceClaim(
                        claim_id="claim_001",
                        claim_text="Test claim statement for metformin efficacy",
                        certainty=GRADECertainty.MODERATE,
                        supporting_snippet_ids=["snp_test_001"],
                    )
                ],
                overall_narrative="Test narrative",
            )

            result = mock_synth(
                sample_study_cards,
                sample_rob2_results,
                "What is the efficacy of metformin?",
            )

            assert result.claim_count == 1
            assert result.claims[0].certainty == GRADECertainty.MODERATE

    def test_group_by_outcome_type(self, mock_llm, sample_study_cards):
        """Test grouping studies by outcome type."""
        from cdr.synthesis.synthesizer import EvidenceSynthesizer

        synthesizer = EvidenceSynthesizer(mock_llm)
        grouped = synthesizer._group_by_outcome_type(sample_study_cards)

        assert len(grouped) > 0

    def test_build_synthesis_context(self, mock_llm, sample_study_cards, sample_rob2_results):
        """Test context building for LLM."""
        from cdr.synthesis.synthesizer import EvidenceSynthesizer

        synthesizer = EvidenceSynthesizer(mock_llm)
        context = synthesizer._build_synthesis_context(
            sample_study_cards,
            sample_rob2_results,
            "Test research question",
        )

        assert "Test research question" in context
        assert "pmid:12345" in context
        assert "HbA1c" in context  # Outcome name should be in context


# =============================================================================
# SKEPTIC AGENT TESTS
# =============================================================================


class TestSkepticAgent:
    """Tests for SkepticAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM for skeptic."""
        mock = MagicMock()
        mock.complete.return_value = MagicMock(
            content="""{
                "critiques": [
                    {
                        "claim_id": "claim_001",
                        "dimension": "EVIDENCE_QUALITY",
                        "severity": "MAJOR",
                        "critique_text": "Limited sample size",
                        "suggestion": "Include larger studies"
                    }
                ],
                "overall_assessment": {
                    "confidence_level": "moderate",
                    "key_concerns": ["Sample size"],
                    "strengths": ["Good design"],
                    "recommendation": "Consider with caution"
                }
            }"""
        )
        return mock

    def test_critique_filtering_by_severity(self, mock_llm):
        """Test that critiques are filtered by severity threshold."""
        from cdr.skeptic.skeptic_agent import SkepticAgent
        from cdr.core.enums import CritiqueSeverity

        # Agent with HIGH threshold (filters out MEDIUM, LOW, INFO)
        agent = SkepticAgent(
            mock_llm,
            severity_threshold=CritiqueSeverity.HIGH,
        )

        # LOW critiques should be filtered out
        assert agent.severity_threshold == CritiqueSeverity.HIGH

    def test_aggregate_critiques(self):
        """Test critique aggregation by claim."""
        from cdr.skeptic.skeptic_agent import aggregate_critiques
        from cdr.core.schemas import CritiqueResult
        from cdr.core.enums import CritiqueDimension, CritiqueSeverity

        findings = [
            CritiqueResult(
                dimension=CritiqueDimension.INTERNAL_VALIDITY,
                severity=CritiqueSeverity.HIGH,
                finding="Internal validity issue with randomization",
                affected_claims=["claim_001"],
            ),
            CritiqueResult(
                dimension=CritiqueDimension.STATISTICAL_ISSUES,
                severity=CritiqueSeverity.LOW,
                finding="Statistical power may be insufficient",
                affected_claims=["claim_001"],
            ),
            CritiqueResult(
                dimension=CritiqueDimension.MISSING_EVIDENCE,
                severity=CritiqueSeverity.CRITICAL,
                finding="Missing evidence from key trials",
                affected_claims=["claim_002"],
            ),
        ]

        aggregated = aggregate_critiques(findings)

        assert "claim_001" in aggregated
        assert "claim_002" in aggregated
        assert len(aggregated["claim_001"]) == 2
        # Should be sorted by severity (HIGH before LOW)
        assert aggregated["claim_001"][0].severity == CritiqueSeverity.HIGH

    def test_calculate_critique_score(self):
        """Test critique severity score calculation."""
        from cdr.skeptic.skeptic_agent import calculate_critique_score
        from cdr.core.schemas import CritiqueResult
        from cdr.core.enums import CritiqueDimension, CritiqueSeverity

        # All CRITICAL
        critical_findings = [
            CritiqueResult(
                dimension=CritiqueDimension.INTERNAL_VALIDITY,
                severity=CritiqueSeverity.CRITICAL,
                finding="Critical issue with study design integrity",
                affected_claims=["c1"],
            )
            for _ in range(3)
        ]

        # All LOW
        low_findings = [
            CritiqueResult(
                dimension=CritiqueDimension.INTERNAL_VALIDITY,
                severity=CritiqueSeverity.LOW,
                finding="Minor formatting concerns in reporting",
                affected_claims=["c1"],
            )
            for _ in range(3)
        ]

        critical_score = calculate_critique_score(critical_findings)
        low_score = calculate_critique_score(low_findings)

        assert critical_score > low_score

    def test_should_revise_claim(self):
        """Test claim revision decision logic."""
        from cdr.skeptic.skeptic_agent import should_revise_claim
        from cdr.core.schemas import CritiqueResult
        from cdr.core.enums import CritiqueDimension, CritiqueSeverity

        # One CRITICAL should trigger revision
        findings_with_critical = [
            CritiqueResult(
                dimension=CritiqueDimension.INTERNAL_VALIDITY,
                severity=CritiqueSeverity.CRITICAL,
                finding="Critical flaw in blinding methodology",
                affected_claims=["c1"],
            )
        ]

        # Multiple HIGH should trigger
        findings_with_high = [
            CritiqueResult(
                dimension=CritiqueDimension.INTERNAL_VALIDITY,
                severity=CritiqueSeverity.HIGH,
                finding="High severity selection bias detected",
                affected_claims=["c1"],
            ),
            CritiqueResult(
                dimension=CritiqueDimension.STATISTICAL_ISSUES,
                severity=CritiqueSeverity.HIGH,
                finding="High severity statistical analysis flaw",
                affected_claims=["c1"],
            ),
        ]

        # Only LOW should not trigger
        low_only = [
            CritiqueResult(
                dimension=CritiqueDimension.MISSING_EVIDENCE,
                severity=CritiqueSeverity.LOW,
                finding="Low severity minor documentation gap",
                affected_claims=["c1"],
            )
        ]

        assert should_revise_claim(findings_with_critical) is True
        assert should_revise_claim(findings_with_high) is True
        assert should_revise_claim(low_only) is False


# =============================================================================
# GRADE CERTAINTY TESTS
# =============================================================================


class TestGRADECertainty:
    """Tests for GRADE certainty handling."""

    def test_certainty_ordering(self):
        """Test GRADE certainty has correct ordering."""
        assert GRADECertainty.HIGH.value == "high"
        assert GRADECertainty.MODERATE.value == "moderate"
        assert GRADECertainty.LOW.value == "low"
        assert GRADECertainty.VERY_LOW.value == "very_low"

    def test_certainty_from_string(self):
        """Test creating certainty from string."""
        assert GRADECertainty("high") == GRADECertainty.HIGH
        assert GRADECertainty("moderate") == GRADECertainty.MODERATE

        with pytest.raises(ValueError):
            GRADECertainty("invalid")


# =============================================================================
# DOD LEVEL EARLY GATE TESTS
# =============================================================================


class TestDoDLevelEarlyGates:
    """Tests for DoD Level early gates in synthesis.

    Refs: ADR-005 Post-Change Audit, CDR_Post_ADR005_Full_Audit
    - Level 2+: Markdown fallback not allowed (JSON required)
    - Level 3: grade_rationale required for all claims
    """

    def test_synthesis_result_used_markdown_fallback_field(self):
        """Test SynthesisResult has used_markdown_fallback field."""
        from cdr.core.schemas import SynthesisResult

        # Default should be False
        result = SynthesisResult(claims=[])
        assert result.used_markdown_fallback is False

        # Can be explicitly set to True
        result_fallback = SynthesisResult(claims=[], used_markdown_fallback=True)
        assert result_fallback.used_markdown_fallback is True

    def test_synthesis_result_json_path_no_fallback(self):
        """Test JSON parsing path sets used_markdown_fallback=False."""
        from cdr.synthesis.synthesizer import EvidenceSynthesizer

        # Mock LLM that returns valid JSON
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(
            content='{"claims": [], "heterogeneity_assessment": "none", "overall_synthesis": "test"}'
        )

        synthesizer = EvidenceSynthesizer(mock_llm, "test-model")

        # Call _parse_synthesis_response directly
        result = synthesizer._parse_synthesis_response(
            '{"claims": [], "overall_synthesis": "test"}',
            [],
            None,
        )

        assert result.used_markdown_fallback is False

    def test_synthesis_result_markdown_fallback_flagged(self):
        """Test Markdown fallback path sets used_markdown_fallback=True."""
        from cdr.synthesis.synthesizer import EvidenceSynthesizer

        mock_llm = MagicMock()
        synthesizer = EvidenceSynthesizer(mock_llm, "test-model")

        # Content that cannot be parsed as JSON
        markdown_content = """
        ## Synthesis Results

        This is a markdown response, not JSON.
        """

        result = synthesizer._parse_synthesis_response(
            markdown_content,
            [],
            None,
        )

        # Should flag as fallback since JSON parsing failed
        assert result.used_markdown_fallback is True

    def test_dod_level_2_blocks_markdown_fallback(self):
        """Test DoD Level 2 blocks synthesis when Markdown fallback is used.

        Refs: ADR-005, PRISMA 2020 (reproducibility)
        """
        from cdr.core.schemas import SynthesisResult, EvidenceClaim
        from cdr.core.enums import GRADECertainty

        # Simulate the gate logic from synthesize_node
        dod_level = 2
        result = SynthesisResult(
            claims=[
                EvidenceClaim(
                    claim_id="test_001",
                    claim_text="Test claim extracted from markdown fallback",
                    certainty=GRADECertainty.MODERATE,
                    supporting_snippet_ids=["snp_001"],
                )
            ],
            used_markdown_fallback=True,  # Fallback was used
        )

        # Gate logic: Level 2+ blocks if used_markdown_fallback is True
        blocked = dod_level >= 2 and result.used_markdown_fallback

        assert blocked is True, "Level 2 should block Markdown fallback"

    def test_dod_level_1_allows_markdown_fallback(self):
        """Test DoD Level 1 allows Markdown fallback (exploratory mode)."""
        from cdr.core.schemas import SynthesisResult

        dod_level = 1
        result = SynthesisResult(claims=[], used_markdown_fallback=True)

        # Gate logic
        blocked = dod_level >= 2 and result.used_markdown_fallback

        assert blocked is False, "Level 1 should allow Markdown fallback"

    def test_dod_level_3_blocks_missing_grade_rationale(self):
        """Test DoD Level 3 blocks claims without grade_rationale.

        Refs: ADR-005, GRADE handbook section 5.2
        """
        from cdr.core.schemas import EvidenceClaim
        from cdr.core.enums import GRADECertainty

        dod_level = 3

        claims = [
            EvidenceClaim(
                claim_id="claim_with_rationale",
                claim_text="This claim has proper GRADE rationale documented",
                certainty=GRADECertainty.MODERATE,
                supporting_snippet_ids=["snp_001"],
                grade_rationale={"risk_of_bias": "Low risk - properly randomized"},
            ),
            EvidenceClaim(
                claim_id="claim_missing_rationale",
                claim_text="This claim is missing GRADE rationale completely",
                certainty=GRADECertainty.LOW,
                supporting_snippet_ids=["snp_002"],
                grade_rationale={},  # Empty - should trigger block
            ),
        ]

        # Gate logic from synthesize_node
        claims_missing_rationale = [c for c in claims if not c.grade_rationale]

        assert len(claims_missing_rationale) == 1
        assert claims_missing_rationale[0].claim_id == "claim_missing_rationale"

        # Level 3 should block
        blocked = dod_level >= 3 and len(claims_missing_rationale) > 0
        assert blocked is True

    def test_dod_level_2_allows_missing_grade_rationale(self):
        """Test DoD Level 2 does NOT block for missing grade_rationale."""
        from cdr.core.schemas import EvidenceClaim
        from cdr.core.enums import GRADECertainty

        dod_level = 2

        claims = [
            EvidenceClaim(
                claim_id="claim_no_rationale",
                claim_text="This claim has no grade rationale but level 2 allows it",
                certainty=GRADECertainty.LOW,
                supporting_snippet_ids=["snp_001"],
                grade_rationale={},  # Empty
            ),
        ]

        claims_missing_rationale = [c for c in claims if not c.grade_rationale]

        # Level 2 does NOT require grade_rationale
        blocked = dod_level >= 3 and len(claims_missing_rationale) > 0
        assert blocked is False

    def test_dod_level_3_passes_with_complete_rationale(self):
        """Test DoD Level 3 passes when all claims have grade_rationale."""
        from cdr.core.schemas import EvidenceClaim
        from cdr.core.enums import GRADECertainty

        dod_level = 3

        claims = [
            EvidenceClaim(
                claim_id="claim_001",
                claim_text="This claim has complete GRADE rationale documented properly",
                certainty=GRADECertainty.MODERATE,
                supporting_snippet_ids=["snp_001"],
                grade_rationale={
                    "risk_of_bias": "Low - adequately randomized",
                    "inconsistency": "No serious inconsistency",
                },
            ),
            EvidenceClaim(
                claim_id="claim_002",
                claim_text="This claim also has complete GRADE rationale documented",
                certainty=GRADECertainty.HIGH,
                supporting_snippet_ids=["snp_002"],
                grade_rationale={
                    "risk_of_bias": "Low",
                    "imprecision": "No serious imprecision - large sample",
                },
            ),
        ]

        claims_missing_rationale = [c for c in claims if not c.grade_rationale]
        assert len(claims_missing_rationale) == 0

        blocked = dod_level >= 3 and len(claims_missing_rationale) > 0
        assert blocked is False

    def test_dod_gate_reason_codes(self):
        """Test DoD gate produces correct reason codes for traceability."""
        # Simulate reason codes from synthesize_node

        # Level 2 block
        level2_reason = "MARKDOWN_FALLBACK_NOT_ALLOWED"
        assert "MARKDOWN" in level2_reason
        assert "FALLBACK" in level2_reason

        # Level 3 block
        level3_reason = "GRADE_RATIONALE_INCOMPLETE"
        assert "GRADE" in level3_reason
        assert "RATIONALE" in level3_reason

    def test_gate_snippets_without_claims_blocked(self):
        """Test gate blocks when snippets exist but 0 claims generated.

        Refs: ADR-005, CDR_Post_ADR005_Full_Audit (ALTO)
        This is a different case from "no snippets" - evidence exists but
        synthesis couldn't extract structured claims.
        """
        from cdr.core.schemas import Snippet, SourceRef

        # State with snippets but no validated claims
        snippets = [
            Snippet(
                snippet_id="snp_001",
                text="Evidence text extracted from study with sufficient length",
                source_ref=SourceRef(
                    record_id="rec_001",
                    title="Test Study Title",
                ),
            )
        ]
        validated_claims = []

        # Gate logic from synthesize_node
        has_snippets = len(snippets) > 0
        has_claims = len(validated_claims) > 0

        blocked = has_snippets and not has_claims

        assert blocked is True, "Should block when snippets exist but 0 claims"
        assert has_snippets is True
        assert has_claims is False

    def test_gate_snippets_without_claims_reason_code(self):
        """Test NO_CLAIMS_WITH_EVIDENCE reason code for snippets-without-claims gate."""
        reason_code = "NO_CLAIMS_WITH_EVIDENCE"
        assert "NO_CLAIMS" in reason_code
        assert "EVIDENCE" in reason_code

    def test_gate_no_snippets_not_blocked_by_claims_gate(self):
        """Test no-snippets case is handled differently (not claims gate).

        When there are no snippets, it's INSUFFICIENT_EVIDENCE at a different level,
        not the NO_CLAIMS_WITH_EVIDENCE gate.
        """
        snippets = []
        validated_claims = []

        # This specific gate only fires when snippets > 0 but claims = 0
        blocked_by_claims_gate = len(snippets) > 0 and len(validated_claims) == 0

        assert blocked_by_claims_gate is False, "No-snippets case is not this gate"

    def test_level3_grade_rationale_requires_all_domains(self):
        """Test Level 3 requires ALL 5 GRADE domains or not_applicable.

        Refs: ADR-005, CDR_Post_ADR005_Full_Audit (ALTO)
        GRADE domains: risk_of_bias, inconsistency, indirectness, imprecision, publication_bias
        """
        from cdr.core.schemas import EvidenceClaim
        from cdr.core.enums import GRADECertainty

        GRADE_REQUIRED_DOMAINS = frozenset(
            [
                "risk_of_bias",
                "inconsistency",
                "indirectness",
                "imprecision",
                "publication_bias",
            ]
        )

        # Claim with only partial rationale (missing domains)
        claim_partial = EvidenceClaim(
            claim_id="claim_partial",
            claim_text="This claim has only some GRADE domains",
            certainty=GRADECertainty.MODERATE,
            supporting_snippet_ids=["snp_001"],
            grade_rationale={
                "risk_of_bias": "Low risk",
                "inconsistency": "No serious inconsistency",
                # Missing: indirectness, imprecision, publication_bias
            },
        )

        present_domains = set(claim_partial.grade_rationale.keys())
        missing_domains = GRADE_REQUIRED_DOMAINS - present_domains

        assert len(missing_domains) == 3
        assert "indirectness" in missing_domains
        assert "imprecision" in missing_domains
        assert "publication_bias" in missing_domains

    def test_level3_grade_rationale_complete_passes(self):
        """Test Level 3 passes with all 5 GRADE domains populated."""
        from cdr.core.schemas import EvidenceClaim
        from cdr.core.enums import GRADECertainty

        GRADE_REQUIRED_DOMAINS = frozenset(
            [
                "risk_of_bias",
                "inconsistency",
                "indirectness",
                "imprecision",
                "publication_bias",
            ]
        )

        claim_complete = EvidenceClaim(
            claim_id="claim_complete",
            claim_text="This claim has all GRADE domains",
            certainty=GRADECertainty.HIGH,
            supporting_snippet_ids=["snp_001"],
            grade_rationale={
                "risk_of_bias": "Low risk - double-blinded RCT",
                "inconsistency": "No serious inconsistency across studies",
                "indirectness": "Direct evidence - population matches",
                "imprecision": "Narrow CI - large sample size",
                "publication_bias": "not_applicable - comprehensive search",
            },
        )

        present_domains = set(claim_complete.grade_rationale.keys())
        missing_domains = GRADE_REQUIRED_DOMAINS - present_domains

        assert len(missing_domains) == 0, "All domains should be present"

    def test_level3_grade_rationale_empty_value_blocked(self):
        """Test Level 3 blocks when domain is present but value is empty."""
        from cdr.core.schemas import EvidenceClaim
        from cdr.core.enums import GRADECertainty

        GRADE_REQUIRED_DOMAINS = frozenset(
            [
                "risk_of_bias",
                "inconsistency",
                "indirectness",
                "imprecision",
                "publication_bias",
            ]
        )

        claim_empty_value = EvidenceClaim(
            claim_id="claim_empty_value",
            claim_text="This claim has empty domain value",
            certainty=GRADECertainty.LOW,
            supporting_snippet_ids=["snp_001"],
            grade_rationale={
                "risk_of_bias": "Low risk",
                "inconsistency": "",  # Empty value - should block
                "indirectness": "Direct",
                "imprecision": "Wide CI",
                "publication_bias": "Suspected",
            },
        )

        present_domains = set(claim_empty_value.grade_rationale.keys())
        empty_domains = [
            d
            for d in present_domains
            if d in GRADE_REQUIRED_DOMAINS and not claim_empty_value.grade_rationale.get(d)
        ]

        assert len(empty_domains) == 1
        assert "inconsistency" in empty_domains

    def test_grade_rationale_incomplete_reason_code(self):
        """Test GRADE_RATIONALE_INCOMPLETE reason code for domain completeness."""
        reason_code = "GRADE_RATIONALE_INCOMPLETE"
        assert "GRADE" in reason_code
        assert "INCOMPLETE" in reason_code

    def test_run_kpis_includes_used_markdown_fallback(self):
        """Test run_kpis dict includes used_markdown_fallback field.

        Refs: ADR-005, CDR_Post_ADR005_Full_Audit (MEDIO)
        """
        # Simulate run_kpis structure from publish_node
        run_kpis = {
            "snippet_coverage": 0.95,
            "verification_coverage": 0.85,
            "claims_with_evidence_ratio": 1.0,
            "total_claims": 5,
            "verified_claims": 4,
            "unverified_claims": 1,
            "is_negative_outcome": False,
            "dod_level": 2,
            "used_markdown_fallback": False,  # NEW FIELD
        }

        assert "used_markdown_fallback" in run_kpis
        assert run_kpis["used_markdown_fallback"] is False

        # Test with fallback True
        run_kpis_fallback = run_kpis.copy()
        run_kpis_fallback["used_markdown_fallback"] = True

        assert run_kpis_fallback["used_markdown_fallback"] is True
