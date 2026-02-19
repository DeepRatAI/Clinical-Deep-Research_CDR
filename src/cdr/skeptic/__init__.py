"""
CDR Skeptic Layer

Adversarial verification agent.
"""

from cdr.skeptic.skeptic_agent import (
    SkepticAgent,
    CounterArgument,
    aggregate_critiques,
    calculate_critique_score,
    should_revise_claim,
)

__all__ = [
    "SkepticAgent",
    "CounterArgument",
    "aggregate_critiques",
    "calculate_critique_score",
    "should_revise_claim",
]
