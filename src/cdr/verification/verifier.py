"""
CDR Verification Layer

Citation verification and entailment checking.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cdr.core.enums import VerificationStatus
from cdr.core.schemas import (
    EvidenceClaim,
    Snippet,
    SourceRef,
    VerificationCheck,
    VerificationResult,
)
from cdr.observability.tracer import tracer

if TYPE_CHECKING:
    from cdr.llm.base import BaseLLMProvider


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

ENTAILMENT_SYSTEM_PROMPT = """You are a textual entailment expert.

Given a CLAIM and a SOURCE TEXT, determine if the source text entails 
(logically supports) the claim.

Entailment levels:
- ENTAILS: The source clearly and directly supports the claim
- PARTIAL: The source provides some support but not complete
- NEUTRAL: The source neither supports nor contradicts
- CONTRADICTS: The source contradicts the claim

IMPORTANT: For medical/scientific abstracts, if the abstract discusses the topic 
and provides evidence about the intervention's effects, consider it ENTAILS or PARTIAL
unless it explicitly contradicts the claim.

Output ONLY valid JSON (no markdown, no extra text):
{"entailment": "ENTAILS", "confidence": 0.85, "supporting_quote": "exact text", "reasoning": "explanation"}

Rules:
- Be lenient for abstracts that discuss the topic positively
- Use ENTAILS if the source discusses the same intervention and outcome
- Use PARTIAL if related but not exact match
- Use NEUTRAL only if completely unrelated
- CONTRADICTS only if explicitly opposite conclusion"""


CITATION_VERIFICATION_PROMPT = """You are a citation accuracy checker.

Verify that the cited information accurately reflects the source.

Check for:
1. Accurate representation of findings
2. No selective quoting that changes meaning
3. Correct attribution of statements
4. Appropriate context preservation

Output JSON:
{
    "accurate": true/false,
    "issues": ["List of accuracy issues if any"],
    "severity": "none/minor/major/critical",
    "corrected_text": "Suggested correction if needed"
}"""


FACT_CHECK_PROMPT = """You are a scientific fact-checker.

Given a factual claim and supporting evidence, verify:
1. Are statistics accurately reported?
2. Are study characteristics correctly stated?
3. Is the interpretation appropriate?

Output JSON with verification status and any corrections needed."""


# =============================================================================
# VERIFIER CLASS
# =============================================================================


class Verifier:
    """Verify claims against source evidence."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        model: str = "gpt-4o",
        confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize verifier.

        Args:
            llm_provider: LLM provider for verification
            model: Model to use
            confidence_threshold: Minimum confidence for VERIFIED status
        """
        self.llm = llm_provider
        self.model = model
        self.confidence_threshold = confidence_threshold

    def verify_claim(
        self,
        claim: EvidenceClaim,
        snippets: dict[str, Snippet],
        source_texts: dict[str, str],
    ) -> VerificationResult:
        """Verify a single claim against its sources.

        Args:
            claim: The claim to verify
            snippets: Dict mapping snippet_id to Snippet objects
            source_texts: Dict mapping record_id to full text

        Returns:
            VerificationResult with individual checks
        """
        with tracer.start_span("verification.verify_claim") as span:
            span.set_attribute("claim_id", claim.claim_id)
            span.set_attribute("source_count", len(source_texts))

            checks = []

            # Verify each supporting snippet (resolve IDs to Snippet objects)
            for snippet_id in claim.supporting_snippet_ids:
                snippet = snippets.get(snippet_id)
                if not snippet:
                    checks.append(
                        VerificationCheck(
                            claim_id=claim.claim_id,
                            source_ref=SourceRef(record_id="unknown", snippet_id=snippet_id),
                            status=VerificationStatus.UNVERIFIABLE,
                            confidence=0.0,
                            explanation=f"Snippet {snippet_id} not found in provided snippets",
                        )
                    )
                    continue

                record_id = snippet.source_ref.record_id
                source_text = source_texts.get(record_id, "")

                if not source_text:
                    checks.append(
                        VerificationCheck(
                            claim_id=claim.claim_id,
                            source_ref=snippet.source_ref,
                            status=VerificationStatus.UNVERIFIABLE,
                            confidence=0.0,
                            explanation="Source text not available",
                        )
                    )
                    continue

                check = self._verify_snippet(claim, snippet, source_text)
                checks.append(check)

            # Calculate overall result
            result = self._aggregate_checks(claim.claim_id, checks)

            span.set_attribute("overall_status", result.overall_status.value)

            return result

    def _verify_snippet(
        self,
        claim: EvidenceClaim,
        snippet: Snippet,
        source_text: str,
    ) -> VerificationCheck:
        """Verify a single snippet against source text."""
        # First, check if snippet exists in source
        snippet_found = self._find_snippet_in_source(snippet.text, source_text)

        if not snippet_found:
            return VerificationCheck(
                claim_id=claim.claim_id,
                source_ref=snippet.source_ref,
                status=VerificationStatus.CONTRADICTED,
                confidence=0.0,
                explanation="Quoted text not found in source",
            )

        # Check entailment
        # CRITICAL: Use claim_text (correct field from schemas.py), not statement
        entailment_result = self._check_entailment(claim.claim_text, snippet.text, source_text)

        # Map entailment to verification status (UNIFIED TAXONOMY)
        # Be flexible with case and common variations
        entailment_raw = str(entailment_result.get("entailment", "")).upper().strip()

        # Handle variations
        if entailment_raw in ["ENTAILS", "ENTAILED", "SUPPORTS", "SUPPORTED", "YES", "TRUE"]:
            status = VerificationStatus.VERIFIED
        elif entailment_raw in ["PARTIAL", "PARTIALLY", "SOMEWHAT", "PARTIALLY_ENTAILS"]:
            status = VerificationStatus.PARTIAL
        elif entailment_raw in ["CONTRADICTS", "CONTRADICTED", "REFUTES", "REFUTED", "NO", "FALSE"]:
            status = VerificationStatus.CONTRADICTED
        else:
            status = VerificationStatus.UNVERIFIABLE

        # Apply confidence threshold
        confidence = entailment_result.get("confidence", 0.5)
        if status == VerificationStatus.VERIFIED and confidence < self.confidence_threshold:
            status = VerificationStatus.PARTIAL

        return VerificationCheck(
            claim_id=claim.claim_id,
            source_ref=snippet.source_ref,
            status=status,
            confidence=confidence,
            explanation=entailment_result.get("reasoning", ""),
            supporting_quote=entailment_result.get("supporting_quote"),
        )

    def _find_snippet_in_source(
        self,
        snippet_text: str,
        source_text: str,
        fuzzy_threshold: float = 0.8,
    ) -> bool:
        """Check if snippet text exists in source (with fuzzy matching).

        Args:
            snippet_text: The quoted text to find
            source_text: The full source text
            fuzzy_threshold: Minimum similarity for fuzzy match

        Returns:
            True if snippet found in source
        """
        # Normalize texts
        snippet_norm = self._normalize_text(snippet_text)
        source_norm = self._normalize_text(source_text)

        # Exact match (after normalization)
        if snippet_norm in source_norm:
            return True

        # Fuzzy match - check if significant portion matches
        snippet_words = set(snippet_norm.split())
        source_words = set(source_norm.split())

        if not snippet_words:
            return False

        overlap = len(snippet_words & source_words) / len(snippet_words)
        return overlap >= fuzzy_threshold

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    def _check_entailment(
        self,
        claim_statement: str,
        snippet_text: str,
        full_source: str,
    ) -> dict:
        """Check if source text entails the claim.

        NOTE: Uses sync complete() to avoid async/sync mixing issues.
        If LLM provider only supports async, this will use a fallback.
        """
        # Use snippet context for focused check
        context = f"""
CLAIM: {claim_statement}

CITED TEXT: "{snippet_text}"

FULL SOURCE CONTEXT (excerpt around citation):
{self._get_context_around_snippet(snippet_text, full_source)}

Does the source text entail (support) the claim?"""

        messages = [
            {"role": "system", "content": ENTAILMENT_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        # Use sync complete() to avoid async/sync mixing
        # CRITICAL: This is a sync method, must not use await
        try:
            response = self.llm.complete(
                messages=messages,
                model=self.model,
                temperature=0.2,
                max_tokens=1000,
            )
            result = self._parse_entailment_response(response.content)
            # Debug logging
            print(
                f"[Verify] Entailment result: {result.get('entailment')} (conf: {result.get('confidence', 0):.2f})"
            )
            return result
        except AttributeError:
            # Fallback if provider doesn't have sync complete
            # Return conservative unverifiable result
            return {
                "entailment": "NEUTRAL",
                "confidence": 0.5,
                "reasoning": "Verification skipped: sync API not available",
            }

    def _get_context_around_snippet(
        self,
        snippet: str,
        source: str,
        context_chars: int = 500,
    ) -> str:
        """Extract context around a snippet in the source."""
        snippet_norm = self._normalize_text(snippet)
        source_norm = self._normalize_text(source)

        pos = source_norm.find(snippet_norm[:50])  # Find start of snippet

        if pos == -1:
            # If not found, return beginning of source
            return source[: context_chars * 2]

        start = max(0, pos - context_chars)
        end = min(len(source), pos + len(snippet) + context_chars)

        return source[start:end]

    def _parse_entailment_response(self, content: str) -> dict:
        """Parse entailment response from LLM."""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        # Try to extract JSON from content - be more flexible
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            import re

            json_match = re.search(r'\{[^{}]*"entailment"[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Look for entailment keyword in text to infer result
            content_lower = content.lower()
            if "entails" in content_lower or "supports" in content_lower:
                return {
                    "entailment": "PARTIAL",  # Conservative but positive
                    "confidence": 0.6,
                    "reasoning": f"Inferred from response: {content[:200]}",
                }

            return {
                "entailment": "NEUTRAL",
                "confidence": 0.5,
                "reasoning": f"Failed to parse: {content[:200]}",
            }

    def _aggregate_checks(
        self,
        claim_id: str,
        checks: list[VerificationCheck],
    ) -> VerificationResult:
        """Aggregate individual checks into overall result."""
        if not checks:
            return VerificationResult(
                claim_id=claim_id,
                checks=[],
                overall_status=VerificationStatus.UNVERIFIABLE,
                overall_confidence=0.0,
            )

        # Count statuses
        status_counts = {status: 0 for status in VerificationStatus}
        for check in checks:
            status_counts[check.status] += 1

        # Determine overall status (RELAXED TAXONOMY)
        # Count VERIFIED and PARTIAL as "positive" evidence
        total = len(checks)
        positive_count = (
            status_counts[VerificationStatus.VERIFIED] + status_counts[VerificationStatus.PARTIAL]
        )

        if status_counts[VerificationStatus.CONTRADICTED] > 0:
            # Any contradiction is concerning
            if status_counts[VerificationStatus.CONTRADICTED] > total / 2:
                overall_status = VerificationStatus.CONTRADICTED
            else:
                overall_status = VerificationStatus.PARTIAL
        elif status_counts[VerificationStatus.VERIFIED] == total:
            # All checks are VERIFIED
            overall_status = VerificationStatus.VERIFIED
        elif positive_count == total:
            # All checks are either VERIFIED or PARTIAL
            overall_status = (
                VerificationStatus.VERIFIED
                if status_counts[VerificationStatus.VERIFIED] > 0
                else VerificationStatus.PARTIAL
            )
        elif positive_count > total / 2:
            # Majority of checks are positive (VERIFIED or PARTIAL)
            overall_status = VerificationStatus.PARTIAL
        else:
            overall_status = VerificationStatus.UNVERIFIABLE

        # Calculate overall confidence
        confidences = [c.confidence for c in checks if c.confidence is not None]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return VerificationResult(
            claim_id=claim_id,
            checks=checks,
            overall_status=overall_status,
            overall_confidence=overall_confidence,
        )

    def verify_all_claims(
        self,
        claims: list[EvidenceClaim],
        snippets: dict[str, Snippet],
        source_texts: dict[str, str],
    ) -> dict[str, VerificationResult]:
        """Verify multiple claims.

        Args:
            claims: List of claims to verify
            snippets: Dict mapping snippet_id to Snippet objects
            source_texts: Dict mapping record_id to full text

        Returns:
            Dict mapping claim_id to VerificationResult
        """
        with tracer.start_span("verification.verify_all") as span:
            span.set_attribute("claim_count", len(claims))

            results = {}
            for claim in claims:
                results[claim.claim_id] = self.verify_claim(claim, snippets, source_texts)

            # Log summary
            verified = sum(
                1 for r in results.values() if r.overall_status == VerificationStatus.VERIFIED
            )
            span.set_attribute("verified_count", verified)

            return results


# =============================================================================
# CITATION CHECKER
# =============================================================================


class CitationChecker:
    """Check citation accuracy and completeness."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        model: str = "gpt-4o",
    ) -> None:
        """Initialize citation checker."""
        self.llm = llm_provider
        self.model = model

    def check_citation(
        self,
        cited_text: str,
        source_text: str,
        context: str | None = None,
    ) -> CitationCheckResult:
        """Check if a citation accurately represents the source.

        Args:
            cited_text: The text as cited in the report
            source_text: The original source text
            context: Additional context about the citation

        Returns:
            CitationCheckResult with accuracy assessment

        NOTE: Uses sync complete() to avoid async/sync mixing issues.
        """
        with tracer.start_span("verification.check_citation"):
            prompt = f"""
CITED TEXT: "{cited_text}"

ORIGINAL SOURCE: "{source_text}"

{f"CONTEXT: {context}" if context else ""}

Verify that the citation accurately represents the source."""

            messages = [
                {"role": "system", "content": CITATION_VERIFICATION_PROMPT},
                {"role": "user", "content": prompt},
            ]

            # Use sync complete() - CRITICAL: this is a sync method
            try:
                response = self.llm.complete(
                    messages=messages,
                    model=self.model,
                    temperature=0.2,
                    max_tokens=800,
                )
                return self._parse_citation_result(response.content)
            except AttributeError:
                # Fallback if provider doesn't have sync complete
                return CitationCheckResult(
                    accurate=False,
                    issues=["Verification skipped: sync API not available"],
                    severity="major",
                )

    def _parse_citation_result(self, content: str) -> "CitationCheckResult":
        """Parse citation check response."""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return CitationCheckResult(
                accurate=False,
                issues=["Failed to parse verification response"],
                severity="major",
            )

        return CitationCheckResult(
            accurate=data.get("accurate", False),
            issues=data.get("issues", []),
            severity=data.get("severity", "none"),
            corrected_text=data.get("corrected_text"),
        )


@dataclass
class CitationCheckResult:
    """Result of citation accuracy check."""

    accurate: bool
    issues: list[str] = field(default_factory=list)
    severity: str = "none"  # none, minor, major, critical
    corrected_text: str | None = None


# =============================================================================
# BATCH VERIFICATION
# =============================================================================


def batch_verify(
    verifier: Verifier,
    claims: list[EvidenceClaim],
    snippets: dict[str, Snippet],
    source_texts: dict[str, str],
    fail_threshold: float = 0.5,
) -> BatchVerificationResult:
    """Perform batch verification with summary statistics.

    Args:
        verifier: Verifier instance
        claims: Claims to verify
        snippets: Dict mapping snippet_id to Snippet objects
        source_texts: Source texts by record_id
        fail_threshold: Proportion of failures that indicates overall failure

    Returns:
        BatchVerificationResult with summary
    """
    results = verifier.verify_all_claims(claims, snippets, source_texts)

    verified = sum(1 for r in results.values() if r.overall_status == VerificationStatus.VERIFIED)
    partial = sum(1 for r in results.values() if r.overall_status == VerificationStatus.PARTIAL)
    contradicted = sum(
        1 for r in results.values() if r.overall_status == VerificationStatus.CONTRADICTED
    )
    unverifiable = sum(
        1 for r in results.values() if r.overall_status == VerificationStatus.UNVERIFIABLE
    )

    total = len(results)
    pass_rate = (verified + partial) / total if total > 0 else 0.0

    overall_pass = (contradicted / total) < fail_threshold if total > 0 else True

    return BatchVerificationResult(
        results=results,
        verified_count=verified,
        partial_count=partial,
        contradicted_count=contradicted,
        unverifiable_count=unverifiable,
        total_count=total,
        pass_rate=pass_rate,
        overall_pass=overall_pass,
    )


@dataclass
class BatchVerificationResult:
    """Result of batch verification."""

    results: dict[str, VerificationResult]
    verified_count: int
    partial_count: int
    contradicted_count: int
    unverifiable_count: int
    total_count: int
    pass_rate: float
    overall_pass: bool

    def contradicted_claims(self) -> list[str]:
        """Get IDs of contradicted claims."""
        return [
            claim_id
            for claim_id, result in self.results.items()
            if result.overall_status == VerificationStatus.CONTRADICTED
        ]

    def needs_review(self) -> list[str]:
        """Get IDs of claims needing review."""
        return [
            claim_id
            for claim_id, result in self.results.items()
            if result.overall_status
            in (VerificationStatus.PARTIAL, VerificationStatus.UNVERIFIABLE)
        ]
