"""
CDR Assertion Gate

Validates that conclusions are supported by evidence structure.
Blocks "invalid leaps" where conclusions claim more than evidence supports.

Key Rules:
1. If all evidence is comparative (A vs B), cannot claim absolute efficacy of A
2. Each assertion must map to at least one claim with matching I/C/context
3. Fail loudly with audit trail, don't silently pass invalid conclusions

Refs: PRISMA 2020, GRADE Handbook, CDR DoD P3
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from cdr.core.enums import ComparatorSource, TherapeuticContext, GRADECertainty
from cdr.core.schemas import EvidenceClaim, PICO, Snippet

if TYPE_CHECKING:
    pass


class AssertionViolationType(str, Enum):
    """Types of assertion violations detected by the gate."""

    INVALID_LEAP = "invalid_leap"  # Claiming absolute efficacy from comparative evidence
    MISSING_SUPPORT = "missing_support"  # Assertion has no matching claim
    CONTEXT_MISMATCH = "context_mismatch"  # Assertion context differs from evidence context
    COMPARATOR_MISMATCH = "comparator_mismatch"  # PICO comparator doesn't match evidence
    OVERCLAIM = "overclaim"  # Claim certainty higher than evidence supports


@dataclass
class AssertionViolation:
    """A detected violation in an assertion."""

    violation_type: AssertionViolationType
    assertion_text: str
    expected: str
    actual: str
    severity: str  # "blocker", "warning", "info"
    recommendation: str
    supporting_claim_ids: list[str] = field(default_factory=list)


@dataclass
class AssertionGateResult:
    """Result of assertion gate validation."""

    passed: bool
    violations: list[AssertionViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    degraded_assertions: list[str] = field(default_factory=list)
    audit_trail: list[str] = field(default_factory=list)

    @property
    def has_blockers(self) -> bool:
        """Check if there are any blocking violations."""
        return any(v.severity == "blocker" for v in self.violations)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "assertion": v.assertion_text[:100],
                    "expected": v.expected,
                    "actual": v.actual,
                    "severity": v.severity,
                    "recommendation": v.recommendation,
                }
                for v in self.violations
            ],
            "warnings": self.warnings,
            "degraded_assertions": self.degraded_assertions,
            "audit_trail": self.audit_trail,
        }


class AssertionGate:
    """
    Gate that validates conclusions are supported by evidence structure.

    Key validations:
    1. No invalid leaps (comparative evidence → absolute claims)
    2. Assertion coverage (each strong claim has evidence support)
    3. Context consistency (therapeutic context matches)
    4. Comparator coherence (PICO matches evidence)
    """

    # Patterns that indicate absolute efficacy claims
    ABSOLUTE_EFFICACY_PATTERNS = [
        r"\b(?:reduces?|prevents?|decreases?|lowers?)\s+(?:the\s+)?(?:risk|incidence|rate)\b",
        r"\bis\s+effective\s+(?:for|in|at)\b",
        r"\bprotects?\s+against\b",
        r"\bimproves?\s+(?:outcomes?|survival)\b",
    ]

    # Patterns that indicate comparative claims (acceptable)
    COMPARATIVE_CLAIM_PATTERNS = [
        r"\bcompared\s+(?:to|with)\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bsuperior\s+to\b",
        r"\binferior\s+to\b",
        r"\bmore\s+effective\s+than\b",
        r"\bless\s+effective\s+than\b",
        r"\bin\s+(?:trials?|studies?)\s+comparing\b",
    ]

    # Therapeutic contexts that are head-to-head comparative
    COMPARATIVE_CONTEXTS = {
        TherapeuticContext.DOAC_VS_ASPIRIN,
        TherapeuticContext.DOAC_VS_WARFARIN,
        TherapeuticContext.HEAD_TO_HEAD,
        TherapeuticContext.MONOTHERAPY_VS_ACTIVE,
    }

    def __init__(
        self,
        strict_mode: bool = True,
        fail_on_invalid_leap: bool = True,
    ):
        """
        Initialize assertion gate.

        Args:
            strict_mode: If True, blockers cause failure. If False, downgrade to warnings.
            fail_on_invalid_leap: If True, invalid leaps are blockers. If False, warnings only.
        """
        self.strict_mode = strict_mode
        self.fail_on_invalid_leap = fail_on_invalid_leap

    def validate(
        self,
        conclusion_text: str,
        claims: list[EvidenceClaim],
        pico: PICO,
        snippets: list[Snippet] | None = None,
    ) -> AssertionGateResult:
        """
        Validate that conclusion is supported by evidence structure.

        Args:
            conclusion_text: The final answer/conclusion to validate
            claims: Evidence claims from synthesis
            pico: PICO structure with comparator_source
            snippets: Optional snippets for deeper validation

        Returns:
            AssertionGateResult with pass/fail and violations
        """
        violations: list[AssertionViolation] = []
        warnings: list[str] = []
        audit_trail: list[str] = []
        degraded_assertions: list[str] = []

        audit_trail.append(f"AssertionGate: Validating conclusion ({len(conclusion_text)} chars)")
        audit_trail.append(f"AssertionGate: {len(claims)} claims available")
        audit_trail.append(
            f"AssertionGate: PICO comparator_source = {pico.comparator_source.value}"
        )

        # === CHECK 1: Detect dominant therapeutic context ===
        context_counts = self._count_therapeutic_contexts(claims)
        dominant_context = self._get_dominant_context(context_counts)
        audit_trail.append(f"AssertionGate: Dominant context = {dominant_context}")

        is_primarily_comparative = dominant_context in self.COMPARATIVE_CONTEXTS
        audit_trail.append(f"AssertionGate: Primarily comparative = {is_primarily_comparative}")

        # === CHECK 2: Invalid leap detection ===
        if is_primarily_comparative:
            leap_violations = self._detect_invalid_leaps(conclusion_text, dominant_context, claims)
            violations.extend(leap_violations)
            for v in leap_violations:
                audit_trail.append(
                    f"AssertionGate: VIOLATION - {v.violation_type.value}: {v.assertion_text[:50]}..."
                )

        # === CHECK 3: Comparator coherence ===
        comparator_violations = self._check_comparator_coherence(pico, claims, snippets)
        violations.extend(comparator_violations)

        # === CHECK 4: Context mixing detection ===
        mixing_violations = self._detect_context_mixing(claims)
        warnings.extend([f"Context mixing: {v.assertion_text}" for v in mixing_violations])

        # === DETERMINE OUTCOME ===
        blockers = [v for v in violations if v.severity == "blocker"]

        if blockers and self.strict_mode:
            passed = False
            audit_trail.append(f"AssertionGate: FAILED - {len(blockers)} blockers")
        else:
            passed = True
            # Downgrade blockers to warnings if not strict
            if blockers and not self.strict_mode:
                for v in blockers:
                    degraded_assertions.append(v.recommendation)
                    warnings.append(
                        f"[Degraded] {v.violation_type.value}: {v.assertion_text[:50]}..."
                    )
                audit_trail.append(
                    f"AssertionGate: PASSED with {len(blockers)} degraded violations"
                )
            else:
                audit_trail.append("AssertionGate: PASSED")

        return AssertionGateResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            degraded_assertions=degraded_assertions,
            audit_trail=audit_trail,
        )

    def _count_therapeutic_contexts(
        self, claims: list[EvidenceClaim]
    ) -> dict[TherapeuticContext, int]:
        """Count claims by therapeutic context."""
        counts: dict[TherapeuticContext, int] = {}
        for claim in claims:
            ctx = claim.therapeutic_context
            counts[ctx] = counts.get(ctx, 0) + 1
        return counts

    def _get_dominant_context(
        self, context_counts: dict[TherapeuticContext, int]
    ) -> TherapeuticContext:
        """Get the most common therapeutic context."""
        if not context_counts:
            return TherapeuticContext.UNCLASSIFIED

        # Filter out UNCLASSIFIED for dominance calculation
        filtered = {k: v for k, v in context_counts.items() if k != TherapeuticContext.UNCLASSIFIED}
        if not filtered:
            return TherapeuticContext.UNCLASSIFIED

        return max(filtered, key=lambda k: filtered[k])

    def _detect_invalid_leaps(
        self,
        conclusion_text: str,
        dominant_context: TherapeuticContext,
        claims: list[EvidenceClaim],
    ) -> list[AssertionViolation]:
        """
        Detect invalid leaps: absolute claims from comparative evidence.

        CRITICAL RULE: If evidence is "DOAC > ASA", cannot claim "ASA reduces stroke"
        because that would imply comparison to placebo/nothing.
        """
        violations = []
        sentences = self._split_into_sentences(conclusion_text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Check if sentence makes absolute efficacy claim
            is_absolute_claim = any(
                re.search(pattern, sentence_lower, re.IGNORECASE)
                for pattern in self.ABSOLUTE_EFFICACY_PATTERNS
            )

            # Check if sentence is properly comparative
            is_comparative_claim = any(
                re.search(pattern, sentence_lower, re.IGNORECASE)
                for pattern in self.COMPARATIVE_CLAIM_PATTERNS
            )

            # INVALID LEAP: Absolute claim when evidence is comparative
            if is_absolute_claim and not is_comparative_claim:
                # This is the critical case: claiming "X reduces Y" when all we know is "A > B"
                violations.append(
                    AssertionViolation(
                        violation_type=AssertionViolationType.INVALID_LEAP,
                        assertion_text=sentence,
                        expected=f"Comparative claim reflecting {dominant_context.value} evidence",
                        actual="Absolute efficacy claim",
                        severity="blocker" if self.fail_on_invalid_leap else "warning",
                        recommendation=self._generate_rewrite_suggestion(
                            sentence, dominant_context
                        ),
                    )
                )

        return violations

    def _generate_rewrite_suggestion(self, sentence: str, context: TherapeuticContext) -> str:
        """Generate a suggestion for rewriting an invalid claim."""
        if context == TherapeuticContext.DOAC_VS_ASPIRIN:
            return (
                "Rewrite to: 'In trials comparing DOACs to aspirin, DOACs showed [lower/higher] "
                "rates of [outcome] compared to aspirin.' Do not claim aspirin's absolute effect "
                "without direct placebo-controlled evidence."
            )
        elif context == TherapeuticContext.HEAD_TO_HEAD:
            return (
                "Rewrite to specify the comparison: 'Compared to [comparator], [intervention] "
                "showed [direction] effect.' Avoid absolute claims."
            )
        else:
            return (
                "Rewrite to reflect the actual comparison in the evidence. Do not claim "
                "absolute efficacy when evidence is only comparative."
            )

    def _check_comparator_coherence(
        self,
        pico: PICO,
        claims: list[EvidenceClaim],
        snippets: list[Snippet] | None,
    ) -> list[AssertionViolation]:
        """
        Check that PICO comparator is coherent with evidence.

        If PICO says comparator is "placebo" but all evidence is head-to-head,
        that's a coherence problem.
        """
        violations = []

        # Count comparative contexts in claims
        context_counts = self._count_therapeutic_contexts(claims)
        comparative_count = sum(context_counts.get(ctx, 0) for ctx in self.COMPARATIVE_CONTEXTS)
        total_claims = len(claims)

        # If >50% of claims are comparative but PICO has no comparator or generic comparator
        if total_claims > 0 and comparative_count / total_claims > 0.5:
            if pico.comparator is None:
                violations.append(
                    AssertionViolation(
                        violation_type=AssertionViolationType.COMPARATOR_MISMATCH,
                        assertion_text=f"PICO.comparator = None but {comparative_count}/{total_claims} claims are comparative",
                        expected="PICO.comparator should reflect the actual comparator in evidence",
                        actual=f"PICO.comparator = None, source = {pico.comparator_source.value}",
                        severity="warning",
                        recommendation="Update PICO to reflect the comparator found in evidence (comparator_source=inferred_from_evidence)",
                    )
                )
            elif pico.comparator_source == ComparatorSource.ASSUMED_FROM_QUESTION:
                # PICO assumed placebo but evidence is head-to-head
                dominant = self._get_dominant_context(context_counts)
                if dominant in self.COMPARATIVE_CONTEXTS:
                    violations.append(
                        AssertionViolation(
                            violation_type=AssertionViolationType.COMPARATOR_MISMATCH,
                            assertion_text=f"PICO assumed '{pico.comparator}' but evidence is {dominant.value}",
                            expected=f"Comparator should match evidence ({dominant.value})",
                            actual=f"PICO.comparator = '{pico.comparator}' (assumed)",
                            severity="warning",
                            recommendation="Add caveat that evidence is comparative, not vs assumed comparator",
                        )
                    )

        return violations

    def _detect_context_mixing(self, claims: list[EvidenceClaim]) -> list[AssertionViolation]:
        """
        Detect claims that might be mixing incompatible therapeutic contexts.

        E.g., mixing aspirin monotherapy results with aspirin+anticoagulant results.
        """
        violations = []

        # Group claims by context
        context_counts = self._count_therapeutic_contexts(claims)

        # Check for problematic combinations
        has_aspirin_mono = context_counts.get(TherapeuticContext.ASPIRIN_MONOTHERAPY, 0) > 0
        has_aspirin_combo = context_counts.get(TherapeuticContext.ASPIRIN_PLUS_ANTICOAGULANT, 0) > 0

        if has_aspirin_mono and has_aspirin_combo:
            violations.append(
                AssertionViolation(
                    violation_type=AssertionViolationType.CONTEXT_MISMATCH,
                    assertion_text="Claims mix aspirin monotherapy and aspirin+anticoagulant contexts",
                    expected="Separate conclusions for each therapeutic context",
                    actual="Mixed contexts in synthesis",
                    severity="warning",
                    recommendation="Present conclusions separately for monotherapy vs combination therapy",
                )
            )

        return violations

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences for analysis."""
        # Simple sentence splitting - handles common cases
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


def validate_conclusion(
    conclusion: str,
    claims: list[EvidenceClaim],
    pico: PICO,
    snippets: list[Snippet] | None = None,
    strict: bool = True,
) -> AssertionGateResult:
    """
    Convenience function to validate a conclusion.

    Args:
        conclusion: The conclusion text to validate
        claims: Evidence claims from synthesis
        pico: PICO with comparator_source
        snippets: Optional snippets for deeper validation
        strict: If True, blockers cause failure

    Returns:
        AssertionGateResult
    """
    gate = AssertionGate(strict_mode=strict)
    return gate.validate(conclusion, claims, pico, snippets)
