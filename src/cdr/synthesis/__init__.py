"""
CDR Synthesis Layer

Evidence synthesis and GRADE assessment.
"""

from cdr.synthesis.synthesizer import (
    EvidenceSynthesizer,
    SynthesisResult,
    calculate_pooled_estimate,
    assess_publication_bias,
)

__all__ = [
    "EvidenceSynthesizer",
    "SynthesisResult",
    "calculate_pooled_estimate",
    "assess_publication_bias",
]
