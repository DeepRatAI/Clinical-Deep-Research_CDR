"""
Composition Quality Integration Tests with Realistic LLM Mock.

Validates that CompositionEngine produces hypotheses that pass ALL 9
quality gates from scripts/validate_golden_set_composition.py.

Uses a realistic mock LLM provider that returns well-structured JSON
responses (mimicking real HuggingFace/OpenAI output), not trivial stubs.

Refs:
- CDR_Integral_Audit_2026-01-20.md HIGH-1 (compositional inference)
- scripts/validate_golden_set_composition.py (quality gates)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from cdr.composition import CompositionEngine
from cdr.composition.schemas import (
    ComposedHypothesis,
    HypothesisStrength,
    MechanisticRelation,
    ProposedStudyDesign,
    ThreatAnalysis,
)
from cdr.core.schemas import EvidenceClaim, PICO


# =============================================================================
# REALISTIC LLM MOCK
# =============================================================================


# Mimics what a real LLM (e.g. Llama-3.1-70B-Instruct) returns for
# GLP-1 / Metformin / cardiovascular evidence composition.

REALISTIC_RELATION_RESPONSE_CLAIM_1 = json.dumps(
    {
        "relations": [
            {
                "source_concept": "GLP-1 receptor agonists",
                "target_concept": "HbA1c reduction",
                "mechanism": "incretin-mediated insulin secretion and glucagon suppression",
                "relation_type": "causal",
                "confidence": 0.9,
            },
            {
                "source_concept": "GLP-1 receptor agonists",
                "target_concept": "weight loss",
                "mechanism": "delayed gastric emptying and central appetite suppression",
                "relation_type": "causal",
                "confidence": 0.85,
            },
        ]
    }
)

REALISTIC_RELATION_RESPONSE_CLAIM_2 = json.dumps(
    {
        "relations": [
            {
                "source_concept": "sustained HbA1c reduction",
                "target_concept": "reduced cardiovascular events",
                "mechanism": "decreased oxidative stress and advanced glycation end-products in vascular endothelium",
                "relation_type": "causal",
                "confidence": 0.75,
            },
            {
                "source_concept": "HbA1c reduction",
                "target_concept": "decreased microvascular complications",
                "mechanism": "reduced retinal and renal capillary basement membrane thickening",
                "relation_type": "mechanistic",
                "confidence": 0.8,
            },
        ]
    }
)

REALISTIC_RELATION_RESPONSE_CLAIM_3 = json.dumps(
    {
        "relations": [
            {
                "source_concept": "GLP-1 receptor agonists",
                "target_concept": "reduced major adverse cardiovascular events",
                "mechanism": "direct anti-inflammatory effects on vascular endothelium and reduced atherosclerotic plaque progression",
                "relation_type": "causal",
                "confidence": 0.85,
            },
        ]
    }
)

REALISTIC_COMPOSITION_RESPONSE = json.dumps(
    {
        "hypothesis_text": (
            "If GLP-1 receptor agonists are administered to patients with type 2 diabetes, "
            "then cardiovascular events will be reduced by 15-26%, because GLP-1 agonists "
            "both improve glycemic control (reducing HbA1c by 1.0-1.5%) and exert direct "
            "anti-inflammatory effects on vascular endothelium, independently lowering "
            "atherosclerotic plaque progression (A + B implies C)"
        ),
        "mechanistic_chain": [
            {
                "source": "GLP-1 receptor agonists",
                "target": "HbA1c reduction",
                "mechanism": "incretin-mediated insulin secretion and glucagon suppression",
            },
            {
                "source": "HbA1c reduction",
                "target": "reduced cardiovascular events",
                "mechanism": "decreased oxidative stress and advanced glycation end-products",
            },
            {
                "source": "GLP-1 receptor agonists",
                "target": "atherosclerotic plaque reduction",
                "mechanism": "direct anti-inflammatory effects on vascular endothelium",
            },
        ],
        "strength": "strong",
        "confidence": 0.82,
        "rival_hypotheses": [
            "Weight loss from GLP-1 agonists, not glycemic control, is the primary driver of CV benefit",
            "Blood pressure reduction by GLP-1 agonists accounts for the observed CV risk reduction",
            "Survivor bias: patients tolerating GLP-1 agonists represent a healthier subpopulation",
        ],
        "uncontrolled_confounders": [
            "Concomitant statin and antihypertensive use",
            "Socioeconomic status and healthcare access differences",
        ],
        "evidence_gaps": [
            "No head-to-head RCT comparing GLP-1 agonist CV effects to matched glycemic control without GLP-1",
            "Limited evidence in non-Western populations",
        ],
        "mcid_estimate": {
            "value": 0.85,
            "unit": "hazard ratio",
            "rationale": "Based on LEADER trial precedent; HR 0.85 = 15% RRR is clinically meaningful for MACE",
        },
        "reasoning": (
            "Claim A establishes GLP-1 agonists reduce HbA1c via incretin pathway. "
            "Claim B (LEADER/SUSTAIN-6) shows 13-26% MACE reduction. "
            "The composition hypothesizes a dual pathway: glycemic and anti-inflammatory, "
            "which together could explain the magnitude of CV benefit beyond what glycemic "
            "control alone predicts."
        ),
    }
)

REALISTIC_TEST_DESIGN_RESPONSE = json.dumps(
    {
        "proposed_population": (
            "Adults 40-75 years with T2DM, HbA1c 7.0-10.0%, on stable metformin ≥3 months, "
            "no prior major cardiovascular event, eGFR ≥30 mL/min"
        ),
        "proposed_intervention": "Semaglutide 1mg subcutaneous weekly for 104 weeks",
        "proposed_comparator": "Placebo injection weekly with metformin continuation",
        "proposed_outcome": "Time to first MACE (composite: cardiovascular death, non-fatal MI, non-fatal stroke)",
        "mcid_value": 0.85,
        "mcid_rationale": (
            "Hazard ratio of 0.85 (15% relative risk reduction) based on LEADER trial precedent "
            "and FDA cardiovascular outcome trial guidelines for anti-diabetic agents"
        ),
        "recommended_design": "RCT",
        "minimum_sample_size": 9340,
        "follow_up_duration": "3.8 years median (event-driven, minimum 2 years per patient)",
        "critical_measurements": [
            "HbA1c quarterly",
            "MACE adjudication by independent blinded committee",
            "hs-CRP, IL-6 (inflammatory biomarkers) at baseline, 6, 12, 24 months",
            "Body weight, blood pressure, lipid panel quarterly",
            "Adverse events including pancreatitis, retinopathy",
            "eGFR and UACR semi-annually",
        ],
        "blinding_requirements": "double-blind (matched placebo injection devices)",
    }
)


@dataclass
class _MockLLMResponse:
    content: str
    usage: dict | None = None


def _create_realistic_provider(responses: list[str]):
    """Create a mock provider that returns realistic JSON responses in sequence."""
    provider = MagicMock()
    call_count = {"n": 0}

    def _complete(**kwargs):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return _MockLLMResponse(content=responses[idx])

    provider.complete.side_effect = _complete
    return provider


# =============================================================================
# GOLDEN SET CLAIMS (realistic medical evidence)
# =============================================================================

GS002_CLAIMS = [
    EvidenceClaim(
        claim_id="gs002-claim-1",
        claim_text=(
            "GLP-1 receptor agonists (semaglutide, liraglutide) reduce HbA1c "
            "by 1.0-1.5% in patients with type 2 diabetes based on multiple RCTs."
        ),
        certainty="high",
        supporting_snippet_ids=["snip-1"],
    ),
    EvidenceClaim(
        claim_id="gs002-claim-2",
        claim_text=(
            "Sustained HbA1c reduction below 7% decreases the incidence of "
            "cardiovascular events and microvascular complications in T2DM patients."
        ),
        certainty="high",
        supporting_snippet_ids=["snip-2"],
    ),
    EvidenceClaim(
        claim_id="gs002-claim-3",
        claim_text=(
            "The LEADER and SUSTAIN-6 trials demonstrated that GLP-1 agonists "
            "reduce major adverse cardiovascular events by 13-26% in T2DM patients."
        ),
        certainty="high",
        supporting_snippet_ids=["snip-3"],
    ),
]

GS002_PICO = PICO(
    population="Adults with type 2 diabetes mellitus",
    intervention="GLP-1 receptor agonists",
    comparator="Placebo or standard care",
    outcome="HbA1c reduction and cardiovascular events",
)


# =============================================================================
# QUALITY GATE CHECKER (mirrors validate_golden_set_composition.py)
# =============================================================================


def validate_hypothesis(hyp: ComposedHypothesis) -> dict[str, bool]:
    """Validate all 9 quality gates on a composed hypothesis."""
    return {
        "has_if_then_structure": (
            "If" in hyp.hypothesis_text and "then" in hyp.hypothesis_text.lower()
        ),
        "has_mechanistic_chain": len(hyp.mechanistic_chain or []) > 0,
        "has_rival_hypotheses": bool(hyp.threat_analysis and hyp.threat_analysis.rival_hypotheses),
        "has_confounders": bool(
            hyp.threat_analysis and hyp.threat_analysis.uncontrolled_confounders
        ),
        "has_evidence_gaps": bool(hyp.threat_analysis and hyp.threat_analysis.evidence_gaps),
        "has_proposed_test": hyp.proposed_test is not None,
        "has_mcid": (hyp.proposed_test is not None and hyp.proposed_test.mcid_value is not None),
        "source_claims_multiple": len(hyp.source_claim_ids) >= 2,
        "confidence_valid": 0.0 <= hyp.confidence_score <= 1.0,
    }


# =============================================================================
# TESTS
# =============================================================================


class TestCompositionQualityGates:
    """Validate all 9 quality gates pass with realistic LLM mock."""

    @pytest.fixture
    def engine(self):
        """Engine with realistic mock provider that sequences through
        relation-extraction → composition → test-design responses."""
        responses = [
            REALISTIC_RELATION_RESPONSE_CLAIM_1,
            REALISTIC_RELATION_RESPONSE_CLAIM_2,
            REALISTIC_RELATION_RESPONSE_CLAIM_3,
            REALISTIC_COMPOSITION_RESPONSE,
            REALISTIC_TEST_DESIGN_RESPONSE,
            # Duplicates in case engine tries more pairs
            REALISTIC_COMPOSITION_RESPONSE,
            REALISTIC_TEST_DESIGN_RESPONSE,
        ]
        provider = _create_realistic_provider(responses)
        return CompositionEngine(provider=provider, model="test-model")

    def test_engine_produces_hypotheses(self, engine):
        """Engine generates at least one hypothesis from 3 claims."""
        hypotheses = engine.run(
            GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=2, include_test_designs=True
        )
        assert len(hypotheses) >= 1, "Expected at least one hypothesis"

    def test_all_9_quality_gates_pass(self, engine):
        """Every generated hypothesis passes all 9 quality gates."""
        hypotheses = engine.run(
            GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=2, include_test_designs=True
        )
        for hyp in hypotheses:
            checks = validate_hypothesis(hyp)
            failed = [k for k, v in checks.items() if not v]
            assert not failed, f"Hypothesis {hyp.hypothesis_id} failed gates: {failed}"

    def test_quality_gate_if_then_structure(self, engine):
        """Hypothesis text starts with 'If' and contains 'then'."""
        hypotheses = engine.run(GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1)
        for hyp in hypotheses:
            assert hyp.hypothesis_text.startswith("If")
            assert "then" in hyp.hypothesis_text.lower()

    def test_quality_gate_mechanistic_chain(self, engine):
        """Mechanistic chain has at least one link."""
        hypotheses = engine.run(GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1)
        for hyp in hypotheses:
            assert len(hyp.mechanistic_chain) >= 1
            # Each link must have a mechanism
            for link in hyp.mechanistic_chain:
                assert link.mechanism is not None

    def test_quality_gate_rival_hypotheses(self, engine):
        """Threat analysis includes at least 2 rival hypotheses."""
        hypotheses = engine.run(GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1)
        for hyp in hypotheses:
            assert hyp.threat_analysis is not None
            assert len(hyp.threat_analysis.rival_hypotheses) >= 2

    def test_quality_gate_confounders(self, engine):
        """Threat analysis identifies at least 1 uncontrolled confounder."""
        hypotheses = engine.run(GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1)
        for hyp in hypotheses:
            assert hyp.threat_analysis is not None
            assert len(hyp.threat_analysis.uncontrolled_confounders) >= 1

    def test_quality_gate_evidence_gaps(self, engine):
        """Threat analysis identifies at least 1 evidence gap."""
        hypotheses = engine.run(GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1)
        for hyp in hypotheses:
            assert hyp.threat_analysis is not None
            assert len(hyp.threat_analysis.evidence_gaps) >= 1

    def test_quality_gate_proposed_test(self, engine):
        """Hypothesis has a proposed test design when requested."""
        hypotheses = engine.run(
            GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1, include_test_designs=True
        )
        for hyp in hypotheses:
            assert hyp.proposed_test is not None
            assert hyp.proposed_test.proposed_population
            assert hyp.proposed_test.proposed_intervention
            assert hyp.proposed_test.proposed_outcome

    def test_quality_gate_mcid(self, engine):
        """Proposed test includes MCID value and rationale."""
        hypotheses = engine.run(
            GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1, include_test_designs=True
        )
        for hyp in hypotheses:
            assert hyp.proposed_test is not None
            assert hyp.proposed_test.mcid_value is not None
            assert hyp.proposed_test.mcid_rationale is not None

    def test_quality_gate_source_claims_multiple(self, engine):
        """Hypothesis references at least 2 source claims."""
        hypotheses = engine.run(GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1)
        for hyp in hypotheses:
            assert len(hyp.source_claim_ids) >= 2

    def test_quality_gate_confidence_valid(self, engine):
        """Confidence score is within [0.0, 1.0]."""
        hypotheses = engine.run(GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1)
        for hyp in hypotheses:
            assert 0.0 <= hyp.confidence_score <= 1.0

    def test_success_rate_above_80(self, engine):
        """Overall quality check pass rate ≥ 80% (matches CI gate)."""
        hypotheses = engine.run(
            GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=2, include_test_designs=True
        )
        total_checks = 0
        passed_checks = 0
        for hyp in hypotheses:
            checks = validate_hypothesis(hyp)
            total_checks += len(checks)
            passed_checks += sum(checks.values())

        if total_checks > 0:
            success_rate = (passed_checks / total_checks) * 100
            assert success_rate >= 80.0, f"Quality gate pass rate {success_rate:.1f}% < 80%"


class TestCompositionEdgeCases:
    """Edge cases for composition quality."""

    def test_single_claim_returns_empty(self):
        """Engine returns empty list for < 2 claims."""
        provider = _create_realistic_provider([])
        engine = CompositionEngine(provider=provider)
        result = engine.run([GS002_CLAIMS[0]], pico=GS002_PICO, max_hypotheses=1)
        assert result == []

    def test_no_provider_returns_empty(self):
        """Engine returns empty list when provider is None."""
        engine = CompositionEngine(provider=None)
        result = engine.run(GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1)
        assert result == []

    def test_provider_error_handled_gracefully(self):
        """Engine handles provider errors without crashing."""
        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("API timeout")
        engine = CompositionEngine(provider=provider)
        result = engine.run(GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1)
        assert result == []

    def test_malformed_json_handled(self):
        """Engine handles malformed JSON responses from LLM."""
        provider = _create_realistic_provider(
            [
                "This is not JSON at all",
                "Still not JSON",
                "Nope",
            ]
        )
        engine = CompositionEngine(provider=provider)
        result = engine.run(GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1)
        assert result == []

    def test_partial_json_handled(self):
        """Engine handles incomplete JSON responses."""
        partial = json.dumps({"relations": []})  # empty relations
        provider = _create_realistic_provider([partial, partial, partial])
        engine = CompositionEngine(provider=provider)
        result = engine.run(GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1)
        # No relations extracted → no composable pairs → empty
        assert result == []


class TestCompositionWithTestDesign:
    """Validate test design generation quality."""

    @pytest.fixture
    def engine_with_design(self):
        responses = [
            REALISTIC_RELATION_RESPONSE_CLAIM_1,
            REALISTIC_RELATION_RESPONSE_CLAIM_2,
            REALISTIC_RELATION_RESPONSE_CLAIM_3,
            REALISTIC_COMPOSITION_RESPONSE,
            REALISTIC_TEST_DESIGN_RESPONSE,
            REALISTIC_COMPOSITION_RESPONSE,
            REALISTIC_TEST_DESIGN_RESPONSE,
        ]
        provider = _create_realistic_provider(responses)
        return CompositionEngine(provider=provider, model="test-model")

    def test_test_design_is_rct(self, engine_with_design):
        """Proposed test design recommends RCT for strong evidence."""
        hypotheses = engine_with_design.run(
            GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1, include_test_designs=True
        )
        for hyp in hypotheses:
            assert hyp.proposed_test is not None
            assert hyp.proposed_test.recommended_design == "RCT"

    def test_test_design_has_sample_size(self, engine_with_design):
        """Proposed test calculates minimum sample size."""
        hypotheses = engine_with_design.run(
            GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1, include_test_designs=True
        )
        for hyp in hypotheses:
            assert hyp.proposed_test is not None
            assert hyp.proposed_test.minimum_sample_size is not None
            assert hyp.proposed_test.minimum_sample_size >= 100

    def test_test_design_has_critical_measurements(self, engine_with_design):
        """Proposed test lists critical measurements."""
        hypotheses = engine_with_design.run(
            GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1, include_test_designs=True
        )
        for hyp in hypotheses:
            assert hyp.proposed_test is not None
            assert len(hyp.proposed_test.critical_measurements) >= 2

    def test_test_design_has_blinding(self, engine_with_design):
        """Proposed test specifies blinding requirements."""
        hypotheses = engine_with_design.run(
            GS002_CLAIMS, pico=GS002_PICO, max_hypotheses=1, include_test_designs=True
        )
        for hyp in hypotheses:
            assert hyp.proposed_test is not None
            assert hyp.proposed_test.blinding_requirements is not None
