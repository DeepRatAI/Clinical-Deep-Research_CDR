"""
Compositional Inference Schemas

Defines data structures for compositional hypothesis generation.

HIGH-1 fix: Schemas for A+B⇒C compositional inference.
Refs: CDR_Integral_Audit_2026-01-20.md HIGH-1

Scientific foundations:
- Causal representation learning: https://arxiv.org/abs/2102.11107
- Target trial emulation: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4866534/
- GRADE certainty: https://gradepro.org/handbook/
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RelationType(str, Enum):
    """Types of mechanistic relations extracted from evidence."""

    CAUSAL = "causal"  # A causes B (strong RCT evidence)
    ASSOCIATIVE = "associative"  # A associated with B (observational)
    MECHANISTIC = "mechanistic"  # A leads to B via mechanism M
    TEMPORAL = "temporal"  # A precedes B
    MODULATING = "modulating"  # A modifies effect of B
    INHIBITORY = "inhibitory"  # A inhibits/reduces B
    UNKNOWN = "unknown"


class HypothesisStrength(str, Enum):
    """Confidence level in composed hypothesis."""

    STRONG = "strong"  # Multiple RCTs, consistent mechanisms
    MODERATE = "moderate"  # Mix of evidence types, plausible
    WEAK = "weak"  # Limited evidence, speculative
    EXPLORATORY = "exploratory"  # Hypothesis-generating only


class MechanisticRelation(BaseModel):
    """
    A mechanistic relation extracted from evidence.

    Represents a directed relationship: source → target via mechanism.

    Example:
        - Source: "GLP-1 agonists"
        - Target: "reduced HbA1c"
        - Mechanism: "increased insulin secretion and decreased glucagon"
        - Relation Type: CAUSAL (from RCT evidence)
    """

    model_config = ConfigDict(frozen=True)

    relation_id: str = Field(..., description="Unique identifier")
    source_concept: str = Field(..., min_length=3, description="Source entity/intervention")
    target_concept: str = Field(..., min_length=3, description="Target outcome/effect")
    mechanism: str | None = Field(default=None, description="Mechanistic pathway if known")
    relation_type: RelationType = Field(default=RelationType.ASSOCIATIVE)

    # Evidence linkage
    supporting_claim_ids: list[str] = Field(
        default_factory=list, description="Claim IDs supporting this relation"
    )
    supporting_snippet_ids: list[str] = Field(
        default_factory=list, description="Snippet IDs with direct evidence"
    )

    # Confidence
    evidence_strength: HypothesisStrength = Field(default=HypothesisStrength.WEAK)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)


class ThreatAnalysis(BaseModel):
    """
    Analysis of threats to validity for a composed hypothesis.

    Identifies rival hypotheses, confounders, and limitations.
    """

    model_config = ConfigDict(frozen=True)

    # Rival hypotheses that could explain the same observation
    rival_hypotheses: list[str] = Field(
        default_factory=list,
        description="Alternative explanations for observed relationship",
    )

    # Known confounders not controlled in evidence
    uncontrolled_confounders: list[str] = Field(
        default_factory=list,
        description="Potential confounding variables",
    )

    # Generalizability limitations
    generalizability_concerns: list[str] = Field(
        default_factory=list,
        description="Limitations in applying to broader populations",
    )

    # Missing evidence gaps
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="Critical evidence missing to confirm hypothesis",
    )

    # Overall threat severity (impacts hypothesis strength)
    overall_threat_level: Literal["low", "moderate", "high", "critical"] = Field(default="moderate")


class ProposedStudyDesign(BaseModel):
    """
    Proposed study design to validate a composed hypothesis.

    Follows target trial emulation principles where applicable.
    """

    model_config = ConfigDict(frozen=True)

    # PICO for proposed test
    proposed_population: str = Field(..., min_length=10)
    proposed_intervention: str = Field(..., min_length=5)
    proposed_comparator: str = Field(..., min_length=5)
    proposed_outcome: str = Field(..., min_length=5)

    # MCID (Minimal Clinically Important Difference)
    mcid_value: float | None = Field(default=None, description="Proposed MCID for primary outcome")
    mcid_rationale: str | None = Field(default=None, description="Justification for MCID value")

    # Study design recommendation
    recommended_design: str = Field(
        default="RCT",
        description="Recommended study design (RCT, pragmatic trial, observational)",
    )
    minimum_sample_size: int | None = Field(
        default=None, ge=10, description="Estimated minimum N for adequate power"
    )
    follow_up_duration: str | None = Field(default=None, description="Recommended follow-up period")

    # Key methodological requirements
    critical_measurements: list[str] = Field(
        default_factory=list, description="Essential measurements/biomarkers"
    )
    blinding_requirements: str | None = Field(
        default=None, description="Blinding requirements if applicable"
    )


class ComposedHypothesis(BaseModel):
    """
    A novel hypothesis composed from multiple evidence streams.

    This is the primary output of compositional inference (A+B⇒C).

    Example:
        Evidence A: "GLP-1 agonists reduce HbA1c in T2DM patients"
        Evidence B: "Sustained HbA1c reduction decreases cardiovascular events"
        Composed C: "GLP-1 agonists may reduce cardiovascular events in T2DM
                     via sustained glycemic control"
    """

    model_config = ConfigDict(frozen=True)

    hypothesis_id: str = Field(..., description="Unique identifier")

    # The composed hypothesis statement
    hypothesis_text: str = Field(
        ...,
        min_length=30,
        max_length=1000,
        description="The novel hypothesis statement",
    )

    # Component evidence (A + B → C)
    source_claim_ids: list[str] = Field(
        ...,
        min_length=2,
        description="Claim IDs that form the basis of this hypothesis",
    )

    # Mechanistic chain
    mechanistic_chain: list[MechanisticRelation] = Field(
        default_factory=list,
        description="Ordered chain of mechanistic relations",
    )

    # Strength and confidence
    strength: HypothesisStrength = Field(default=HypothesisStrength.WEAK)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)

    # Scientific rigor components
    threat_analysis: ThreatAnalysis | None = Field(
        default=None, description="Analysis of threats to validity"
    )
    proposed_test: ProposedStudyDesign | None = Field(
        default=None, description="Proposed validation study design"
    )

    # Metadata
    reasoning_trace: str | None = Field(
        default=None, description="LLM reasoning chain for transparency"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("source_claim_ids")
    @classmethod
    def validate_minimum_claims(cls, v: list[str]) -> list[str]:
        """Composed hypothesis requires at least 2 source claims."""
        if len(v) < 2:
            raise ValueError("Composed hypothesis requires at least 2 source claims (A+B)")
        return v

    def is_testable(self) -> bool:
        """Check if hypothesis has a proposed test design."""
        return self.proposed_test is not None

    def has_mechanism(self) -> bool:
        """Check if hypothesis has mechanistic chain."""
        return bool(self.mechanistic_chain)
