"""
CDR Compositional Inference Module

Implements the A+B⇒C compositional inference capability for generating
testable hypotheses from combined evidence.

HIGH-1 fix: Compositional inference not implemented beyond comments.
Refs: CDR_Integral_Audit_2026-01-20.md HIGH-1

Key concepts:
- Compositional Hypothesis: Novel hypothesis C derived from evidence A and B
- Mechanistic Relation: Causal/mechanistic link extracted from evidence
- Threat Analysis: Identification of rival hypotheses and confounders
- Test Design: Proposed validation methodology with MCID

Scientific foundations:
- Causal representation learning: https://arxiv.org/abs/2102.11107
- Target trial emulation: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4866534/
"""

from cdr.composition.composer import CompositionEngine
from cdr.composition.schemas import (
    ComposedHypothesis,
    HypothesisStrength,
    MechanisticRelation,
    RelationType,
    ProposedStudyDesign,
    ThreatAnalysis,
)

__all__ = [
    "CompositionEngine",
    "ComposedHypothesis",
    "HypothesisStrength",
    "MechanisticRelation",
    "RelationType",
    "ProposedStudyDesign",
    "ThreatAnalysis",
]
