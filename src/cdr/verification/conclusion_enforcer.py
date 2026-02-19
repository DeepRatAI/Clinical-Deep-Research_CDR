"""
Conclusion Enforcer

CRITICAL: Conclusions must obey the run status.
If status == UNPUBLISHABLE, conclusion cannot affirm effects.

Refs:
- DoD3 Contract: "Conclusions are a function of Gate Report"
- PRISMA 2020: Transparent reporting of limitations
- GRADE: Certainty must match evidence quality
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# =============================================================================
# STRONG LANGUAGE PATTERNS (that affirm effects)
# =============================================================================

# These patterns indicate affirmative conclusions about efficacy/safety
STRONG_AFFIRMATIVE_PATTERNS = [
    r"\breduces?\b.*\brisk\b",
    r"\bprevents?\b",
    r"\bimproves?\b.*\boutcomes?\b",
    r"\beffective\b",
    r"\bsuperior\b",
    r"\bshould be used\b",
    r"\bis recommended\b",
    r"\bhas been shown to\b",
    r"\bdemonstrates? (that|efficacy|benefit)\b",
    r"\bprobably\s+(reduces?|prevents?|improves?)\b",
    r"\bmay\s+(reduce|prevent|improve)\b.*\bbeneficial\b",
    r"\bsignificantly?\s+(reduces?|decreases?|lowers?)\b",
    r"\bcertainty.*(moderate|high)\b",
    r"\b(moderate|high)\s+certainty\b",
]

# Compiled patterns for efficiency
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in STRONG_AFFIRMATIVE_PATTERNS]


@dataclass
class DegradedConclusion:
    """Result of conclusion degradation."""

    original: str
    degraded: str
    was_degraded: bool
    reasons: List[str] = field(default_factory=list)
    gate_summary: str | None = None


def degrade_conclusion_for_unpublishable(
    original_conclusion: str,
    status_reason: str,
    gate_report: dict | None = None,
    blocker_count: int = 0,
) -> DegradedConclusion:
    """
    Degrade a conclusion when the run is UNPUBLISHABLE.

    RULE: If overall_status == unpublishable, conclusion cannot contain
    language like "probably reduces / reduces / improves" (even with low/moderate).
    Must output degraded mode explaining WHY the conclusion cannot be drawn.

    Args:
        original_conclusion: The original conclusion text
        status_reason: Why the run is unpublishable
        gate_report: Optional gate report with violation details
        blocker_count: Number of blocker violations

    Returns:
        DegradedConclusion with transformed text
    """
    if not original_conclusion:
        return DegradedConclusion(
            original="",
            degraded="No conclusion could be generated.",
            was_degraded=False,
            reasons=["No original conclusion provided"],
        )

    # Check if conclusion uses strong affirmative language
    matches_strong = []
    for pattern in COMPILED_PATTERNS:
        if pattern.search(original_conclusion):
            matches_strong.append(pattern.pattern)

    if not matches_strong:
        # Conclusion doesn't affirm effects - add warning header only
        degraded = (
            f"⚠️ **THIS RUN IS UNPUBLISHABLE** (Reason: {status_reason})\n\n"
            f"The following synthesis was generated but does not meet publication standards:\n\n"
            f"{original_conclusion}"
        )
        return DegradedConclusion(
            original=original_conclusion,
            degraded=degraded,
            was_degraded=True,
            reasons=["Added unpublishable warning header"],
        )

    # Conclusion affirms effects - must be fully degraded
    # Build explanation from gate report
    explanation_parts = []

    if blocker_count > 0:
        explanation_parts.append(
            f"{blocker_count} blocker violations were detected in the evidence chain"
        )

    if gate_report:
        # Extract specific failure reasons
        gate_results = gate_report.get("gate_results", {})

        for gate_name, result in gate_results.items():
            if isinstance(result, dict) and not result.get("passed", True):
                violation_count = result.get("metadata", {}).get("violations", 0)
                if violation_count > 0:
                    gate_label = gate_name.replace("_", " ").title()
                    explanation_parts.append(f"{gate_label}: {violation_count} violations")

    if "claims_without_valid_snippets" in status_reason:
        explanation_parts.append("Claims lack valid supporting evidence snippets")

    if "critique_blockers" in status_reason:
        explanation_parts.append(
            "Critical methodological issues were identified by the skeptic agent"
        )

    if "dod3_gate_failures" in status_reason:
        explanation_parts.append("Evidence chain failed DoD3 gate validation")

    # Build degraded conclusion
    explanation = "; ".join(explanation_parts) if explanation_parts else status_reason

    degraded = (
        f"⚠️ **CONCLUSION CANNOT BE DRAWN**\n\n"
        f"**Status:** UNPUBLISHABLE\n"
        f"**Reason:** {explanation}\n\n"
        f"---\n\n"
        f"**What this means:** The evidence retrieved does not meet the scientific standards "
        f"required for publication. This may be due to:\n"
        f"- PICO mismatch between query and available evidence\n"
        f"- Study type violations (e.g., observational evidence for therapeutic questions)\n"
        f"- Insufficient high-quality RCT evidence\n"
        f"- Evidence chain traceability failures\n\n"
        f"**Original synthesis (FOR REFERENCE ONLY - NOT VALIDATED):**\n"
        f"_{original_conclusion}_"
    )

    return DegradedConclusion(
        original=original_conclusion,
        degraded=degraded,
        was_degraded=True,
        reasons=matches_strong,
        gate_summary=explanation,
    )


def check_conclusion_obeys_status(
    conclusion: str,
    status: str,
) -> Tuple[bool, List[str]]:
    """
    Check if a conclusion appropriately obeys its run status.

    Args:
        conclusion: The conclusion text
        status: The run status ("publishable", "unpublishable", "partially_publishable", etc.)

    Returns:
        (is_compliant, violations)
    """
    violations = []

    if status in ("unpublishable", "insufficient_evidence"):
        # Check for strong affirmative language
        for pattern in COMPILED_PATTERNS:
            match = pattern.search(conclusion)
            if match:
                violations.append(
                    f"Conclusion contains affirmative language '{match.group()}' "
                    f"but status is {status}"
                )

    return len(violations) == 0, violations


def degrade_conclusion_for_partial(
    original_conclusion: str,
    status_reason: str,
    valid_claim_count: int,
    total_claim_count: int,
    gate_report: dict | None = None,
) -> DegradedConclusion:
    """
    Degrade a conclusion when the run is PARTIALLY_PUBLISHABLE.

    FIX 7: For partial publishability, we keep the valid parts but clearly mark
    which sections are supported vs unsupported.

    Args:
        original_conclusion: The original conclusion text
        status_reason: Why the run is partially publishable
        valid_claim_count: Number of claims with valid evidence
        total_claim_count: Total number of claims
        gate_report: Optional gate report with violation details

    Returns:
        DegradedConclusion with transformed text
    """
    if not original_conclusion:
        return DegradedConclusion(
            original="",
            degraded="No conclusion could be generated.",
            was_degraded=False,
            reasons=["No original conclusion provided"],
        )

    # For partially publishable, add a clear header explaining the mixed status
    degraded = (
        f"⚠️ **PARTIALLY PUBLISHABLE** ({valid_claim_count}/{total_claim_count} claims validated)\n\n"
        f"**Note:** This synthesis includes both validated and unvalidated components. "
        f"Only claims with surviving evidence after DoD3 gate enforcement should be cited.\n\n"
        f"**Reason:** {status_reason}\n\n"
        f"---\n\n"
        f"{original_conclusion}\n\n"
        f"---\n\n"
        f"**⚠️ AUDIT NOTE:** Review the Gate Report to identify which specific claims "
        f"have valid supporting evidence and which were excluded."
    )

    return DegradedConclusion(
        original=original_conclusion,
        degraded=degraded,
        was_degraded=True,
        reasons=[f"Partial validation: {valid_claim_count}/{total_claim_count} claims valid"],
        gate_summary=status_reason,
    )
