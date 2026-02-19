"""
CDR Skeptic Layer

Adversarial verification agent that challenges claims and identifies weaknesses.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cdr.core.enums import CritiqueDimension, CritiqueSeverity, GRADECertainty
from cdr.core.schemas import (
    Critique,
    CritiqueResult,
    EvidenceClaim,
    Snippet,
    SourceRef,
)
from cdr.observability.tracer import tracer

if TYPE_CHECKING:
    from cdr.llm.base import BaseLLMProvider
    from cdr.synthesis.synthesizer import SynthesisResult


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

# CRITICAL: Dimension values MUST match enums.py CritiqueDimension exactly
# Valid dimensions: internal_validity, external_validity, statistical_issues,
#                   missing_evidence, conflicting_evidence, overstatement,
#                   confounders, search_bias

SKEPTIC_SYSTEM_PROMPT = """You are a rigorous scientific skeptic and methodologist.

Your role is to CHALLENGE evidence claims by identifying weaknesses, gaps, and
potential errors. You must be adversarial but fair - your critiques must be
substantive and actionable.

For each claim, evaluate across these dimensions (use EXACTLY these values):

1. internal_validity
   - Are source studies well-designed?
   - Is there selection bias in included studies?
   - Is the claim logically valid based on evidence?

2. external_validity
   - Do study populations match target population?
   - Are settings transferable?
   - Are there important subgroups not addressed?

3. statistical_issues
   - Are statistical methods appropriate?
   - Is heterogeneity addressed?
   - Are confidence intervals acceptable?

4. missing_evidence
   - Are there important outcomes not measured?
   - Is follow-up duration adequate?
   - Are adverse effects addressed?

5. conflicting_evidence
   - Are there studies that contradict the claim?
   - Is the conflict acknowledged and explained?

6. overstatement
   - Does the claim overstate what evidence supports?
   - Is causation claimed from correlation?

7. confounders
   - Are important confounders addressed?
   - Could unmeasured confounders explain results?

8. search_bias
   - Is there evidence of publication bias?
   - Was the search comprehensive?

Severity levels (use EXACTLY these values):
- critical: Invalidates the claim - must be addressed
- high: Significantly weakens the claim - should be addressed
- medium: Weakness that should be acknowledged
- low: Minor observation
- info: Observation for completeness

Output JSON:
{
    "critiques": [
        {
            "claim_id": "claim_001",
            "dimension": "internal_validity",
            "severity": "high",
            "finding": "Detailed critique explaining the issue (min 10 chars)",
            "affected_claims": ["claim_001"],
            "recommendation": "Specific recommendation to address"
        }
    ],
    "blockers": ["List of critical issues that block publication"],
    "recommendations": ["List of general recommendations"],
    "overall_assessment": "Summary of the overall assessment"
}

Rules:
- Use EXACT dimension and severity values shown above (lowercase)
- Be specific - vague critiques are not useful
- Provide actionable recommendations
- Acknowledge strengths, not just weaknesses
- Prioritize by severity
- Do NOT invent issues - only critique based on evidence"""


COUNTER_ARGUMENT_PROMPT = """You are a debate opponent in scientific discourse.

Given a claim and its supporting evidence, generate the strongest possible
counter-arguments. These should be arguments a skeptical reviewer or opponent
might raise.

Focus on:
1. Alternative explanations for findings
2. Methodological concerns
3. Generalizability limitations
4. Missing evidence that would strengthen/weaken the claim
5. Potential confounders not addressed

Output JSON with counter-arguments and their strength (strong/moderate/weak)."""


DEVIL_ADVOCATE_PROMPT = """You are playing devil's advocate.

Take the OPPOSITE position of the given claim and argue for it convincingly.
This helps stress-test the original conclusion.

Your counter-position should:
1. Be evidence-based where possible
2. Highlight genuine uncertainties
3. Identify what evidence would change the conclusion
4. Note legitimate alternative interpretations

This is an intellectual exercise to ensure robustness, not actual disagreement."""


# =============================================================================
# SKEPTIC AGENT
# =============================================================================


class SkepticAgent:
    """Adversarial agent that challenges claims and identifies weaknesses."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        model: str = "gpt-4o",
        severity_threshold: CritiqueSeverity = CritiqueSeverity.LOW,
    ) -> None:
        """Initialize skeptic agent.

        Args:
            llm_provider: LLM provider for critique generation
            model: Model to use
            severity_threshold: Minimum severity to include in output
        """
        self.llm = llm_provider
        self.model = model
        self.severity_threshold = severity_threshold

        # Severity ordering for filtering (per CritiqueSeverity enum)
        self._severity_order = {
            CritiqueSeverity.CRITICAL: 5,
            CritiqueSeverity.HIGH: 4,
            CritiqueSeverity.MEDIUM: 3,
            CritiqueSeverity.LOW: 2,
            CritiqueSeverity.INFO: 1,
        }

    def critique(
        self,
        synthesis_result: SynthesisResult,
        research_question: str,
        study_summaries: list[dict] | None = None,
    ) -> Critique:
        """Generate critiques for synthesized claims.

        Args:
            synthesis_result: Result from evidence synthesis
            research_question: Original research question
            study_summaries: Optional additional context about studies

        Returns:
            Critique with findings organized by severity

        Raises:
            Exception: If critique generation fails
        """
        with tracer.start_span("skeptic.critique") as span:
            span.set_attribute("claim_count", len(synthesis_result.claims))

            context = self._build_critique_context(
                synthesis_result, research_question, study_summaries
            )

            messages = [
                {"role": "system", "content": SKEPTIC_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ]

            # Use sync complete() - NOT async
            # The provider's complete() method handles the actual call
            response = self.llm.complete(
                messages=messages,
                model=self.model,
                temperature=0.4,
                max_tokens=4000,
            )

            result = self._parse_critique_response(response.content)

            # Filter findings by severity threshold
            # Note: Critique is frozen, so we create a new one with filtered findings
            filtered_findings = [
                f
                for f in result.findings
                if self._severity_order.get(f.severity, 0)
                >= self._severity_order.get(self.severity_threshold, 0)
            ]

            # Create new Critique with filtered findings
            result = Critique(
                findings=filtered_findings,
                blockers=result.blockers,
                recommendations=result.recommendations,
                overall_assessment=result.overall_assessment,
            )

            span.set_attribute("critique_count", len(result.findings))
            span.set_attribute("critical_count", len(result.blockers))

            return result

    def _build_critique_context(
        self,
        synthesis_result: SynthesisResult,
        research_question: str,
        study_summaries: list[dict] | None,
    ) -> str:
        """Build context for critique generation."""
        lines = [
            f"RESEARCH QUESTION: {research_question}",
            "",
            "SYNTHESIZED CLAIMS TO CRITIQUE:",
            "",
        ]

        for claim in synthesis_result.claims:
            lines.append(f"--- Claim: {claim.claim_id} ---")
            # Use claim_text per EvidenceClaim schema
            lines.append(f"Statement: {claim.claim_text}")
            lines.append(f"GRADE Certainty: {claim.certainty.value}")
            # Use supporting_snippet_ids per EvidenceClaim schema
            lines.append(f"Supporting Snippets: {', '.join(claim.supporting_snippet_ids)}")

            if claim.limitations:
                lines.append("Limitations:")
                for limitation in claim.limitations:
                    lines.append(f"  - {limitation}")

            if claim.conflicting_snippet_ids:
                lines.append(f"Conflicting Snippets: {', '.join(claim.conflicting_snippet_ids)}")

            lines.append("")

        lines.append(f"HETEROGENEITY: {synthesis_result.heterogeneity_assessment}")
        lines.append(f"OVERALL SYNTHESIS: {synthesis_result.overall_narrative}")

        if study_summaries:
            lines.append("")
            lines.append("ADDITIONAL STUDY CONTEXT:")
            for summary in study_summaries[:5]:
                lines.append(f"  - {summary}")

        lines.append("")
        lines.append("Please provide rigorous critiques of these claims.")

        return "\n".join(lines)

    def _parse_critique_response(self, content: str) -> Critique:
        """Parse LLM response into Critique.

        Returns:
            Critique object containing CritiqueResult findings
        """
        import re
        
        # Clean JSON - handle markdown code blocks
        content = content.strip()
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
        content = content.strip()

        # Try to find JSON object in the response
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'\{[^{}]*("critiques"|"findings")[^{}]*\[.*?\]\s*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # If parsing still fails, create a structured response from the text
                    return Critique(
                        findings=[],
                        blockers=[],  # Don't block on parsing failure
                        recommendations=["LLM response was not valid JSON - manual review may be needed"],
                        overall_assessment=content[:500] if content else "No assessment available",
                    )
            else:
                # No JSON found - extract what we can from text
                return Critique(
                    findings=[],
                    blockers=[],  # Don't block on parsing failure
                    recommendations=["LLM response was not valid JSON"],
                    overall_assessment=content[:500] if content else "No assessment available",
                )

        # Parse individual findings into CritiqueResult objects
        findings = []
        blockers = []

        for crit_data in data.get("critiques", []):
            try:
                dimension = CritiqueDimension(crit_data.get("dimension", "MISSING_EVIDENCE"))
            except ValueError:
                dimension = CritiqueDimension.MISSING_EVIDENCE

            # Map LLM severity to CritiqueSeverity enum values
            severity_str = crit_data.get("severity", "medium").lower()
            severity_map = {
                "critical": CritiqueSeverity.CRITICAL,
                "major": CritiqueSeverity.HIGH,  # Map MAJOR -> HIGH
                "high": CritiqueSeverity.HIGH,
                "medium": CritiqueSeverity.MEDIUM,
                "minor": CritiqueSeverity.LOW,  # Map MINOR -> LOW
                "low": CritiqueSeverity.LOW,
                "note": CritiqueSeverity.INFO,  # Map NOTE -> INFO
                "info": CritiqueSeverity.INFO,
            }
            severity = severity_map.get(severity_str, CritiqueSeverity.MEDIUM)

            # Build finding text: prioritize "finding" (per prompt), fallback to "critique_text"
            finding_text = crit_data.get("finding", "") or crit_data.get("critique_text", "")
            if len(finding_text) < 10:
                finding_text = f"[{dimension.value}] {finding_text or 'No details provided'}"

            # Get affected claims - try "affected_claims" first (per prompt), then "claim_id"
            affected_claims = crit_data.get("affected_claims", [])
            if not affected_claims:
                claim_id = crit_data.get("claim_id")
                if claim_id:
                    affected_claims = [claim_id]

            # Get recommendation: try "recommendation" first (per prompt), then "suggestion"
            recommendation = crit_data.get("recommendation") or crit_data.get("suggestion")

            findings.append(
                CritiqueResult(
                    dimension=dimension,
                    severity=severity,
                    finding=finding_text,
                    affected_claims=affected_claims,
                    recommendation=recommendation,
                )
            )

            # Track blockers (critical severity)
            if severity == CritiqueSeverity.CRITICAL:
                blockers.append(finding_text)

        overall = data.get("overall_assessment", {})
        # Handle case where overall_assessment is a string instead of dict
        if isinstance(overall, str):
            overall_text = overall
            overall = {"recommendation": overall}
        else:
            overall_text = overall.get("recommendation", "") if isinstance(overall, dict) else ""

        recommendations = []
        if isinstance(overall, dict) and overall.get("recommendation"):
            recommendations.append(overall.get("recommendation"))

        return Critique(
            findings=findings,
            blockers=blockers,
            recommendations=recommendations,
            overall_assessment=overall_text
            if isinstance(overall, str)
            else overall.get("recommendation", ""),
        )

    def generate_counter_arguments(
        self,
        claim: EvidenceClaim,
    ) -> list["CounterArgument"]:
        """Generate counter-arguments for a specific claim.

        Args:
            claim: The claim to challenge

        Returns:
            List of counter-arguments
        """
        with tracer.start_span("skeptic.counter_arguments") as span:
            span.set_attribute("claim_id", claim.claim_id)

            # Use claim_text and supporting_snippet_ids per EvidenceClaim schema
            context = f"""
CLAIM: {claim.claim_text}

CERTAINTY: {claim.certainty.value}

SUPPORTING SNIPPETS: {", ".join(claim.supporting_snippet_ids)}

Generate strong counter-arguments to this claim."""

            messages = [
                {"role": "system", "content": COUNTER_ARGUMENT_PROMPT},
                {"role": "user", "content": context},
            ]

            # Use sync complete() - NOT async
            response = self.llm.complete(
                messages=messages,
                model=self.model,
                temperature=0.5,
                max_tokens=2000,
            )

            return self._parse_counter_arguments(response.content)

    def _parse_counter_arguments(self, content: str) -> list["CounterArgument"]:
        """Parse counter-arguments from LLM response."""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        try:
            data = json.loads(content)
            args = data.get("counter_arguments", [])
        except json.JSONDecodeError:
            return []

        result = []
        for arg in args:
            strength_map = {"strong": 3, "moderate": 2, "weak": 1}
            result.append(
                CounterArgument(
                    argument=arg.get("argument", ""),
                    strength=arg.get("strength", "moderate"),
                    evidence_needed=arg.get("evidence_needed"),
                    rebuttal_hint=arg.get("rebuttal_hint"),
                )
            )

        return result

    def devils_advocate(
        self,
        claim: EvidenceClaim,
    ) -> str:
        """Generate devil's advocate position for a claim.

        Args:
            claim: The claim to oppose

        Returns:
            Devil's advocate argument text
        """
        with tracer.start_span("skeptic.devils_advocate") as span:
            # Use claim_text per EvidenceClaim schema
            context = f"""
CLAIM TO OPPOSE: {claim.claim_text}

Take the opposite position and argue convincingly against this claim.
"""

            messages = [
                {"role": "system", "content": DEVIL_ADVOCATE_PROMPT},
                {"role": "user", "content": context},
            ]

            # Use sync complete() - NOT async
            response = self.llm.complete(
                messages=messages,
                model=self.model,
                temperature=0.6,
                max_tokens=1500,
            )

            return response.content


# =============================================================================
# DATA CLASSES
# =============================================================================

from dataclasses import dataclass


@dataclass
class CounterArgument:
    """A counter-argument to a claim."""

    argument: str
    strength: str  # strong, moderate, weak
    evidence_needed: str | None = None
    rebuttal_hint: str | None = None


# =============================================================================
# CRITIQUE AGGREGATION
# =============================================================================


def aggregate_critiques(
    findings: list[CritiqueResult],
) -> dict[str, list[CritiqueResult]]:
    """Aggregate critique findings by affected claim ID.

    Args:
        findings: List of CritiqueResult findings

    Returns:
        Dict mapping claim_id to list of findings affecting that claim
    """
    aggregated: dict[str, list[CritiqueResult]] = {}

    for finding in findings:
        for claim_id in finding.affected_claims:
            if claim_id not in aggregated:
                aggregated[claim_id] = []
            aggregated[claim_id].append(finding)

    # Sort by severity within each claim (using correct enum values)
    severity_order = {
        CritiqueSeverity.CRITICAL: 5,
        CritiqueSeverity.HIGH: 4,
        CritiqueSeverity.MEDIUM: 3,
        CritiqueSeverity.LOW: 2,
        CritiqueSeverity.INFO: 1,
    }

    for claim_id in aggregated:
        aggregated[claim_id].sort(
            key=lambda f: severity_order.get(f.severity, 0),
            reverse=True,
        )

    return aggregated


def calculate_critique_score(findings: list[CritiqueResult]) -> float:
    """Calculate overall critique severity score.

    Higher score = more severe critiques.

    Args:
        findings: List of CritiqueResult findings

    Returns:
        Score from 0-100
    """
    if not findings:
        return 0.0

    severity_weights = {
        CritiqueSeverity.CRITICAL: 40,
        CritiqueSeverity.HIGH: 25,
        CritiqueSeverity.MEDIUM: 15,
        CritiqueSeverity.LOW: 10,
        CritiqueSeverity.INFO: 2,
    }

    total_weight = sum(severity_weights.get(f.severity, 0) for f in findings)
    max_possible = len(findings) * 40  # If all were CRITICAL

    return min(100.0, (total_weight / max_possible) * 100) if max_possible > 0 else 0.0


def should_revise_claim(
    findings: list[CritiqueResult],
    threshold_critical: int = 1,
    threshold_high: int = 2,
) -> bool:
    """Determine if a claim should be revised based on critique findings.

    Args:
        findings: CritiqueResult findings for a claim
        threshold_critical: Number of critical findings that trigger revision
        threshold_high: Number of high severity findings that trigger revision

    Returns:
        True if claim should be revised
    """
    critical_count = sum(1 for f in findings if f.severity == CritiqueSeverity.CRITICAL)
    high_count = sum(1 for f in findings if f.severity == CritiqueSeverity.HIGH)

    return critical_count >= threshold_critical or high_count >= threshold_high
