"""
DoD3 Enforcement Layer - Hard Exclusion and Claim Degradation

This module implements the enforcement layer that CLOSES THE LOOP:
- Records/snippets marked out-of-scope CANNOT support claims
- Claims that would rely on excluded evidence are DEGRADED
- Hypotheses without structural support are SUPPRESSED

CRITICAL: This is the last line of defense before output generation.

Refs:
- DoD3 Contract: CDR_Agent_Guidance_and_Development_Protocol.md
- PRISMA 2020: Eligibility criteria must be enforced
- GRADE Handbook: Indirectness leads to downgrade
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cdr.core.schemas import PICO, EvidenceClaim, Record, Snippet
    from cdr.composition.schemas import ComposedHypothesis

logger = logging.getLogger(__name__)


# =============================================================================
# ENFORCEMENT RESULT TYPES
# =============================================================================


class ExclusionReason(str, Enum):
    """Reason for excluding evidence from claim support."""

    POPULATION_OUT_OF_SCOPE = "population_out_of_scope"
    POPULATION_EXCLUDED = "population_excluded"  # Explicit exclusion (without AF)
    COMPARATOR_MISMATCH = "comparator_mismatch"
    COMPARATOR_JUMPED = "comparator_jumped"  # PICO=placebo, evidence=head-to-head
    STUDY_TYPE_MISMATCH = "study_type_mismatch"
    CONTEXT_PURITY_VIOLATION = "context_purity_violation"
    DUPLICATE = "duplicate"
    INSUFFICIENT_MATCH_SCORE = "insufficient_match_score"


@dataclass
class ExcludedEvidence:
    """Record of excluded evidence with full audit trail."""

    evidence_type: str  # "record" or "snippet"
    evidence_id: str
    pmid: str | None
    reason: ExclusionReason
    detail: str
    pico_component: str | None = None  # P, I, C, O
    match_score: float | None = None
    excluded_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "evidence_type": self.evidence_type,
            "evidence_id": self.evidence_id,
            "pmid": self.pmid,
            "reason": self.reason.value,
            "detail": self.detail,
            "pico_component": self.pico_component,
            "match_score": self.match_score,
            "excluded_at": self.excluded_at.isoformat(),
        }


@dataclass
class DegradedClaim:
    """A claim that has been degraded due to DoD3 violations."""

    claim_id: str
    original_text: str
    degraded_text: str | None
    reason: str
    removed_snippet_ids: list[str]
    remaining_snippet_ids: list[str]
    is_orphan: bool = False  # True if no valid evidence remains

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "original_text": self.original_text,
            "degraded_text": self.degraded_text,
            "reason": self.reason,
            "removed_snippet_ids": self.removed_snippet_ids,
            "remaining_snippet_ids": self.remaining_snippet_ids,
            "is_orphan": self.is_orphan,
        }


@dataclass
class SuppressedHypothesis:
    """A hypothesis that has been suppressed due to lack of structural support."""

    hypothesis_id: str
    hypothesis_text: str
    reason: str
    source_claim_ids: list[str]
    orphan_claim_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_text": self.hypothesis_text,
            "reason": self.reason,
            "source_claim_ids": self.source_claim_ids,
            "orphan_claim_ids": self.orphan_claim_ids,
        }


@dataclass
class EnforcementResult:
    """Complete result of DoD3 enforcement."""

    # Excluded evidence
    excluded_records: list[ExcludedEvidence] = field(default_factory=list)
    excluded_snippets: list[ExcludedEvidence] = field(default_factory=list)

    # Degraded claims
    degraded_claims: list[DegradedClaim] = field(default_factory=list)
    orphan_claims: list[str] = field(default_factory=list)  # Claims with no valid evidence

    # Suppressed hypotheses
    suppressed_hypotheses: list[SuppressedHypothesis] = field(default_factory=list)

    # Valid evidence (post-enforcement)
    valid_record_ids: set[str] = field(default_factory=set)
    valid_snippet_ids: set[str] = field(default_factory=set)
    valid_claim_ids: set[str] = field(default_factory=set)
    valid_hypothesis_ids: set[str] = field(default_factory=set)

    # Enforcement metadata
    run_id: str = ""
    enforced_at: datetime = field(default_factory=datetime.utcnow)
    enforcement_mode: str = "strict"  # "strict" or "lenient"

    @property
    def has_exclusions(self) -> bool:
        return bool(self.excluded_records or self.excluded_snippets)

    @property
    def has_degradations(self) -> bool:
        return bool(self.degraded_claims or self.orphan_claims)

    @property
    def has_suppressions(self) -> bool:
        return bool(self.suppressed_hypotheses)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "enforced_at": self.enforced_at.isoformat(),
            "enforcement_mode": self.enforcement_mode,
            "excluded_records": [e.to_dict() for e in self.excluded_records],
            "excluded_snippets": [e.to_dict() for e in self.excluded_snippets],
            "degraded_claims": [c.to_dict() for c in self.degraded_claims],
            "orphan_claims": self.orphan_claims,
            "suppressed_hypotheses": [h.to_dict() for h in self.suppressed_hypotheses],
            "valid_record_ids": sorted(self.valid_record_ids),
            "valid_snippet_ids": sorted(self.valid_snippet_ids),
            "valid_claim_ids": sorted(self.valid_claim_ids),
            "valid_hypothesis_ids": sorted(self.valid_hypothesis_ids),
            "summary": {
                "total_excluded_records": len(self.excluded_records),
                "total_excluded_snippets": len(self.excluded_snippets),
                "total_degraded_claims": len(self.degraded_claims),
                "total_orphan_claims": len(self.orphan_claims),
                "total_suppressed_hypotheses": len(self.suppressed_hypotheses),
            },
        }


# =============================================================================
# DOD3 ENFORCER
# =============================================================================


class DoD3Enforcer:
    """
    Enforces DoD3 gates by hard-excluding out-of-scope evidence.

    This class takes the gate violations and applies HARD ENFORCEMENT:
    1. Records/snippets with BLOCKER violations → Excluded section
    2. Claims that referenced excluded evidence → Degraded or orphaned
    3. Hypotheses based on orphan claims → Suppressed

    The output is a clean set of evidence that CAN legally support claims.
    """

    def __init__(
        self,
        strict: bool = True,
        min_snippets_per_claim: int = 1,
        suppress_hypotheses_on_unpublishable: bool = True,
    ):
        """
        Args:
            strict: If True, any violation → exclude. If False, only FAIL violations.
            min_snippets_per_claim: Minimum valid snippets to keep a claim
            suppress_hypotheses_on_unpublishable: Suppress hypotheses if run is unpublishable
        """
        self.strict = strict
        self.min_snippets_per_claim = min_snippets_per_claim
        self.suppress_hypotheses_on_unpublishable = suppress_hypotheses_on_unpublishable

    def enforce(
        self,
        run_id: str,
        pico: "PICO",
        records: list["Record"],
        snippets: list["Snippet"],
        claims: list["EvidenceClaim"],
        hypotheses: list[Any] | None = None,
        gate_violations: list[dict] | None = None,
        is_unpublishable: bool = False,
    ) -> EnforcementResult:
        """
        Apply DoD3 enforcement and produce clean evidence sets.

        Args:
            run_id: Run identifier
            pico: PICO frame
            records: All retrieved records
            snippets: All extracted snippets
            claims: All generated claims
            hypotheses: Composed hypotheses (optional)
            gate_violations: Pre-computed gate violations (optional)
            is_unpublishable: If True, apply stricter enforcement

        Returns:
            EnforcementResult with valid/excluded evidence
        """
        result = EnforcementResult(
            run_id=run_id,
            enforcement_mode="strict" if self.strict else "lenient",
        )

        # Build lookups
        records_by_id = {r.record_id: r for r in records}
        snippets_by_id = {s.snippet_id: s for s in snippets}
        snippet_to_record = {s.snippet_id: s.source_ref.record_id for s in snippets}

        # 1. IDENTIFY EXCLUDED RECORDS
        excluded_record_ids = set()
        if gate_violations:
            for v in gate_violations:
                if v.get("result") == "fail" and v.get("record_id"):
                    record_id = v["record_id"]
                    excluded_record_ids.add(record_id)

                    record = records_by_id.get(record_id)
                    pmid = record.pmid if record else None

                    result.excluded_records.append(
                        ExcludedEvidence(
                            evidence_type="record",
                            evidence_id=record_id,
                            pmid=pmid,
                            reason=self._map_violation_to_reason(v.get("mismatch_type", "")),
                            detail=v.get("message", "DoD3 gate failure"),
                            pico_component=v.get("pico_component"),
                            match_score=v.get("match_score"),
                        )
                    )

        # 2. IDENTIFY EXCLUDED SNIPPETS
        excluded_snippet_ids = set()
        if gate_violations:
            for v in gate_violations:
                if v.get("result") == "fail" and v.get("snippet_id"):
                    snippet_id = v["snippet_id"]
                    excluded_snippet_ids.add(snippet_id)

                    snippet = snippets_by_id.get(snippet_id)
                    pmid = snippet.source_ref.pmid if snippet and snippet.source_ref else None

                    result.excluded_snippets.append(
                        ExcludedEvidence(
                            evidence_type="snippet",
                            evidence_id=snippet_id,
                            pmid=pmid,
                            reason=self._map_violation_to_reason(v.get("mismatch_type", "")),
                            detail=v.get("message", "DoD3 gate failure"),
                            pico_component=v.get("pico_component"),
                            match_score=v.get("match_score"),
                        )
                    )

        # Snippets from excluded records are also excluded
        for snippet in snippets:
            record_id = snippet.source_ref.record_id
            if record_id in excluded_record_ids and snippet.snippet_id not in excluded_snippet_ids:
                excluded_snippet_ids.add(snippet.snippet_id)
                result.excluded_snippets.append(
                    ExcludedEvidence(
                        evidence_type="snippet",
                        evidence_id=snippet.snippet_id,
                        pmid=snippet.source_ref.pmid,
                        reason=ExclusionReason.POPULATION_OUT_OF_SCOPE,
                        detail=f"Parent record {record_id} is excluded",
                        pico_component="P",
                    )
                )

        # 3. COMPUTE VALID SETS
        result.valid_record_ids = {r.record_id for r in records} - excluded_record_ids
        result.valid_snippet_ids = {s.snippet_id for s in snippets} - excluded_snippet_ids

        # 4. DEGRADE CLAIMS THAT RELY ON EXCLUDED EVIDENCE
        for claim in claims:
            supporting = set(claim.supporting_snippet_ids)
            valid_supporting = supporting & result.valid_snippet_ids
            removed = supporting - valid_supporting

            if removed:
                # Claim has some evidence removed
                if len(valid_supporting) < self.min_snippets_per_claim:
                    # ORPHAN: Not enough valid evidence
                    result.orphan_claims.append(claim.claim_id)
                    result.degraded_claims.append(
                        DegradedClaim(
                            claim_id=claim.claim_id,
                            original_text=claim.claim_text,
                            degraded_text=None,  # Orphan, no text
                            reason=f"Insufficient valid evidence: {len(valid_supporting)} < {self.min_snippets_per_claim}",
                            removed_snippet_ids=list(removed),
                            remaining_snippet_ids=list(valid_supporting),
                            is_orphan=True,
                        )
                    )
                else:
                    # DEGRADED: Some evidence removed but still valid
                    result.valid_claim_ids.add(claim.claim_id)
                    result.degraded_claims.append(
                        DegradedClaim(
                            claim_id=claim.claim_id,
                            original_text=claim.claim_text,
                            degraded_text=claim.claim_text,  # Keep text but track degradation
                            reason=f"Evidence reduced: {len(removed)} snippets excluded",
                            removed_snippet_ids=list(removed),
                            remaining_snippet_ids=list(valid_supporting),
                            is_orphan=False,
                        )
                    )
            else:
                # All evidence is valid
                result.valid_claim_ids.add(claim.claim_id)

        # 5. SUPPRESS HYPOTHESES IF NEEDED
        if hypotheses:
            for hyp in hypotheses:
                # Handle both Pydantic objects and dicts
                hyp_id = getattr(hyp, "hypothesis_id", None) or hyp.get("hypothesis_id", "")
                hyp_text = getattr(hyp, "hypothesis_text", None) or hyp.get("hypothesis_text", "")
                source_claims = getattr(hyp, "source_claim_ids", None) or hyp.get(
                    "source_claim_ids", []
                )

                orphan_sources = [c for c in source_claims if c in result.orphan_claims]

                should_suppress = False
                reason = ""

                if self.suppress_hypotheses_on_unpublishable and is_unpublishable:
                    should_suppress = True
                    reason = "Run is unpublishable - hypotheses suppressed"
                elif orphan_sources and len(orphan_sources) == len(source_claims):
                    should_suppress = True
                    reason = f"All source claims are orphaned: {orphan_sources}"
                elif orphan_sources:
                    # Partial orphan - still suppress if strict
                    if self.strict:
                        should_suppress = True
                        reason = f"Some source claims are orphaned: {orphan_sources}"

                if should_suppress:
                    result.suppressed_hypotheses.append(
                        SuppressedHypothesis(
                            hypothesis_id=hyp_id,
                            hypothesis_text=hyp_text,
                            reason=reason,
                            source_claim_ids=source_claims,
                            orphan_claim_ids=orphan_sources,
                        )
                    )
                else:
                    result.valid_hypothesis_ids.add(hyp_id)

        return result

    def _map_violation_to_reason(self, mismatch_type: str) -> ExclusionReason:
        """Map gate violation type to exclusion reason."""
        mapping = {
            "population_excluded": ExclusionReason.POPULATION_EXCLUDED,
            "population_not_mentioned": ExclusionReason.POPULATION_OUT_OF_SCOPE,
            "population_context_mismatch": ExclusionReason.POPULATION_OUT_OF_SCOPE,
            "comparator_indirect": ExclusionReason.COMPARATOR_MISMATCH,
            "comparator_jumped": ExclusionReason.COMPARATOR_JUMPED,
            "comparator_missing": ExclusionReason.COMPARATOR_MISMATCH,
            "study_type_mismatch": ExclusionReason.STUDY_TYPE_MISMATCH,
            "context_purity_violation": ExclusionReason.CONTEXT_PURITY_VIOLATION,
            "mixed_therapy_modes": ExclusionReason.CONTEXT_PURITY_VIOLATION,
            "duplicate_evidence": ExclusionReason.DUPLICATE,
        }
        return mapping.get(mismatch_type, ExclusionReason.INSUFFICIENT_MATCH_SCORE)

    def apply_to_claims(
        self,
        claims: list["EvidenceClaim"],
        enforcement_result: EnforcementResult,
    ) -> list["EvidenceClaim"]:
        """
        Apply enforcement result to claims - filter out orphans, update snippet IDs.

        Args:
            claims: Original claims
            enforcement_result: Result from enforce()

        Returns:
            List of claims with updated supporting_snippet_ids, excluding orphans
        """
        valid_claims = []

        for claim in claims:
            if claim.claim_id in enforcement_result.orphan_claims:
                continue  # Skip orphan claims

            # Update supporting snippet IDs to only valid ones
            valid_snippets = [
                sid
                for sid in claim.supporting_snippet_ids
                if sid in enforcement_result.valid_snippet_ids
            ]

            if valid_snippets:
                # Create updated claim with filtered snippets
                # Note: We can't modify the original claim directly, so we track the change
                claim.supporting_snippet_ids = valid_snippets
                valid_claims.append(claim)

        return valid_claims

    def apply_to_hypotheses(
        self,
        hypotheses: list[Any],
        enforcement_result: EnforcementResult,
    ) -> list[Any]:
        """
        Apply enforcement result to hypotheses - filter out suppressed.

        Args:
            hypotheses: Original hypotheses
            enforcement_result: Result from enforce()

        Returns:
            List of valid hypotheses (excluding suppressed)
        """
        suppressed_ids = {h.hypothesis_id for h in enforcement_result.suppressed_hypotheses}

        valid = []
        for hyp in hypotheses:
            hyp_id = getattr(hyp, "hypothesis_id", None) or hyp.get("hypothesis_id", "")
            if hyp_id not in suppressed_ids:
                valid.append(hyp)

        return valid


# =============================================================================
# SUB-PICO DECOMPOSITION
# =============================================================================


@dataclass
class SubPICO:
    """A sub-PICO representing a specific comparison found in evidence."""

    sub_pico_id: str
    population: str
    intervention: str
    comparator: str
    outcome: str
    label: str  # e.g., "PICO-A (DOAC vs Aspirin)"
    dominant_in_evidence: bool = False
    supporting_record_ids: list[str] = field(default_factory=list)
    supporting_snippet_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sub_pico_id": self.sub_pico_id,
            "population": self.population,
            "intervention": self.intervention,
            "comparator": self.comparator,
            "outcome": self.outcome,
            "label": self.label,
            "dominant_in_evidence": self.dominant_in_evidence,
            "supporting_record_ids": self.supporting_record_ids,
            "supporting_snippet_ids": self.supporting_snippet_ids,
        }


class SubPICODecomposer:
    """
    Decomposes evidence into sub-PICOs when comparator mismatch is detected.

    When PICO asks for "aspirin vs placebo" but evidence is primarily "DOAC vs aspirin",
    this creates separate sub-PICOs so conclusions can be properly sectioned.
    """

    DRUG_CLASSES = {
        "doac": ["apixaban", "rivaroxaban", "dabigatran", "edoxaban", "doac", "noac"],
        "aspirin": ["aspirin", "asa", "acetylsalicylic acid"],
        "warfarin": ["warfarin", "vka", "coumadin"],
        "clopidogrel": ["clopidogrel", "plavix"],
        "placebo": ["placebo", "no treatment", "usual care", "control"],
    }

    def __init__(self):
        pass

    def decompose(
        self,
        pico: "PICO",
        snippets: list["Snippet"],
        records: list["Record"] | None = None,
    ) -> list[SubPICO]:
        """
        Analyze evidence and decompose into sub-PICOs.

        Args:
            pico: Original PICO
            snippets: All snippets
            records: Optional records for additional context

        Returns:
            List of SubPICOs found in evidence
        """
        import re

        # Track comparisons found
        comparisons = {}  # (drug1, drug2) -> count
        snippet_by_comparison = {}  # (drug1, drug2) -> [snippet_ids]

        for snippet in snippets:
            text = snippet.text.lower()

            # Find comparison patterns
            patterns = [
                r"(\w+)\s+(?:vs?\.?|versus|compared\s+(?:to|with))\s+(\w+)",
                r"randomized\s+to\s+(\w+)\s+or\s+(\w+)",
                r"assigned\s+to\s+(?:receive\s+)?(\w+)\s+or\s+(\w+)",
            ]

            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    drug1, drug2 = match.group(1), match.group(2)

                    # Normalize to drug class
                    class1 = self._normalize_drug(drug1)
                    class2 = self._normalize_drug(drug2)

                    if class1 and class2 and class1 != class2:
                        key = tuple(sorted([class1, class2]))
                        comparisons[key] = comparisons.get(key, 0) + 1
                        if key not in snippet_by_comparison:
                            snippet_by_comparison[key] = []
                        snippet_by_comparison[key].append(snippet.snippet_id)

        if not comparisons:
            return []

        # Create sub-PICOs for each comparison found
        sub_picos = []
        sorted_comparisons = sorted(comparisons.items(), key=lambda x: -x[1])

        for i, ((drug1, drug2), count) in enumerate(sorted_comparisons[:5]):  # Max 5 sub-PICOs
            label = f"PICO-{chr(65 + i)} ({drug1.upper()} vs {drug2.upper()})"

            sub_pico = SubPICO(
                sub_pico_id=f"sub_pico_{i + 1}",
                population=pico.population,
                intervention=drug1,
                comparator=drug2,
                outcome=pico.outcome,
                label=label,
                dominant_in_evidence=(i == 0),
                supporting_snippet_ids=snippet_by_comparison.get((drug1, drug2), [])
                or snippet_by_comparison.get((drug2, drug1), []),
            )
            sub_picos.append(sub_pico)

        return sub_picos

    def _normalize_drug(self, drug: str) -> str | None:
        """Normalize drug name to class."""
        drug = drug.lower().strip()

        for cls, drugs in self.DRUG_CLASSES.items():
            if drug in drugs or any(d in drug for d in drugs):
                return cls

        return None

    def generate_sectioned_conclusion(
        self,
        sub_picos: list[SubPICO],
        original_pico: "PICO",
        claims: list["EvidenceClaim"],
    ) -> dict:
        """
        Generate a sectioned conclusion based on sub-PICOs.

        Returns:
            Dict with sections for each sub-PICO and a summary
        """
        sections = {}

        # Check if original PICO is direct match
        original_comparator = original_pico.comparator.lower() if original_pico.comparator else ""
        has_direct_evidence = False

        for sub_pico in sub_picos:
            if (
                original_comparator in sub_pico.comparator.lower()
                or sub_pico.comparator.lower() in original_comparator
            ):
                has_direct_evidence = True

        if not has_direct_evidence and sub_picos:
            sections["caveat"] = (
                f"NOTE: The original PICO requested '{original_pico.intervention}' vs "
                f"'{original_pico.comparator}'. However, the evidence found primarily "
                f"addresses different comparisons. Conclusions are sectioned accordingly."
            )

        for sub_pico in sub_picos:
            sections[sub_pico.sub_pico_id] = {
                "label": sub_pico.label,
                "intervention": sub_pico.intervention,
                "comparator": sub_pico.comparator,
                "snippet_count": len(sub_pico.supporting_snippet_ids),
                "is_dominant": sub_pico.dominant_in_evidence,
            }

        return {
            "has_direct_evidence": has_direct_evidence,
            "sub_pico_count": len(sub_picos),
            "sections": sections,
        }


# =============================================================================
# HYPOTHESIS GATE
# =============================================================================


class HypothesisGate:
    """
    Gate for validating hypotheses before output.

    DoD3 Rule: No causal hypotheses if:
    - Run is unpublishable
    - Source claims are orphaned
    - No direct evidence for the hypothesis
    """

    def __init__(self, strict: bool = True):
        self.strict = strict

    def check_hypothesis(
        self,
        hypothesis: Any,
        valid_claim_ids: set[str],
        orphan_claim_ids: list[str],
        is_unpublishable: bool,
    ) -> tuple[bool, str]:
        """
        Check if a hypothesis should pass or be suppressed.

        Args:
            hypothesis: The hypothesis to check
            valid_claim_ids: Set of valid claim IDs
            orphan_claim_ids: List of orphaned claim IDs
            is_unpublishable: If the run is unpublishable

        Returns:
            (should_pass, reason)
        """
        hyp_id = getattr(hypothesis, "hypothesis_id", None) or hypothesis.get("hypothesis_id", "")
        source_claims = getattr(hypothesis, "source_claim_ids", None) or hypothesis.get(
            "source_claim_ids", []
        )
        confidence = getattr(hypothesis, "confidence_score", None) or hypothesis.get(
            "confidence_score", 0.5
        )

        # Rule 1: Suppress all hypotheses if unpublishable and strict
        if is_unpublishable and self.strict:
            return False, "Run is unpublishable - hypotheses suppressed"

        # Rule 2: Check if source claims are valid
        orphan_sources = [c for c in source_claims if c in orphan_claim_ids]
        valid_sources = [c for c in source_claims if c in valid_claim_ids]

        if not valid_sources:
            return False, f"All source claims are orphaned: {orphan_sources}"

        if orphan_sources and self.strict:
            return False, f"Some source claims are orphaned: {orphan_sources}"

        # Rule 3: High-confidence hypotheses need strong support
        if confidence > 0.7 and len(valid_sources) < 2:
            return (
                False,
                f"High-confidence hypothesis needs ≥2 valid source claims, has {len(valid_sources)}",
            )

        return True, "Hypothesis passes gate"


# =============================================================================
# GATE REPORT RENDERER
# =============================================================================


class GateReportRenderer:
    """
    Renders Gate Report for PDF/HTML output.

    DoD3 BLOCKER: When status != publishable, the export must include
    a Gate Report with checks, violations, and degradations.
    """

    def render_html(
        self,
        gate_report: dict,
        enforcement_result: EnforcementResult | None = None,
    ) -> str:
        """
        Render Gate Report as HTML for PDF inclusion.

        Args:
            gate_report: The gate_report dict from the run
            enforcement_result: Optional enforcement result for details

        Returns:
            HTML string
        """
        if not gate_report:
            return "<p>No Gate Report available.</p>"

        html = []

        # Header
        status = gate_report.get("overall_status", "unknown").upper()
        status_color = "#22543d" if status == "PUBLISHABLE" else "#742a2a"

        html.append(f"""
        <div class='gate-report-box' style='background: #fef3c7; border: 2px solid #d69e2e; padding: 16px; margin: 16px 0; border-radius: 8px;'>
            <h3 style='margin: 0 0 12px 0; color: #744210;'>DoD3 Gate Report</h3>
            <div style='display: flex; gap: 20px; flex-wrap: wrap;'>
                <div><strong>Status:</strong> <span style='color: {status_color}; font-weight: bold;'>{status}</span></div>
                <div><strong>Generated:</strong> {gate_report.get("generated_at", "N/A")[:19]}</div>
            </div>
        """)

        # Summary
        summary = gate_report.get("summary", {})
        html.append(f"""
            <div style='margin-top: 12px; display: flex; gap: 16px; flex-wrap: wrap;'>
                <div style='padding: 8px 16px; background: #fff; border-radius: 6px;'>
                    <strong>Total Checks:</strong> {summary.get("total_checks", 0)}
                </div>
                <div style='padding: 8px 16px; background: #c6f6d5; border-radius: 6px;'>
                    <strong>Passed:</strong> {summary.get("passed", 0)}
                </div>
                <div style='padding: 8px 16px; background: #fefcbf; border-radius: 6px;'>
                    <strong>Warned:</strong> {summary.get("warned", 0)}
                </div>
                <div style='padding: 8px 16px; background: #fed7d7; border-radius: 6px;'>
                    <strong>Failed:</strong> {summary.get("failed", 0)}
                </div>
            </div>
        """)

        # Gate Results
        gate_results = gate_report.get("gate_results", {})
        if gate_results:
            html.append("<div style='margin-top: 16px;'><strong>Gate Results:</strong></div>")
            html.append(
                "<table style='width: 100%; margin-top: 8px; border-collapse: collapse; font-size: 9pt;'>"
            )
            html.append(
                "<tr style='background: #edf2f7;'><th style='padding: 6px; text-align: left;'>Gate</th><th style='padding: 6px;'>Result</th><th style='padding: 6px;'>Details</th></tr>"
            )

            for gate_name, gate_data in gate_results.items():
                result = gate_data.get("result", "unknown")
                result_icon = "✓" if result == "pass" else ("⚠" if result == "warn" else "✗")
                result_color = (
                    "#22543d"
                    if result == "pass"
                    else ("#744210" if result == "warn" else "#742a2a")
                )

                metadata = gate_data.get("metadata", {})
                details = ", ".join(f"{k}: {v}" for k, v in metadata.items())[:100]

                html.append(f"""
                <tr>
                    <td style='padding: 6px; border-bottom: 1px solid #e2e8f0;'>{gate_name.replace("_", " ").title()}</td>
                    <td style='padding: 6px; border-bottom: 1px solid #e2e8f0; text-align: center; color: {result_color};'>{result_icon} {result.upper()}</td>
                    <td style='padding: 6px; border-bottom: 1px solid #e2e8f0; font-size: 8pt; color: #718096;'>{details}</td>
                </tr>
                """)

            html.append("</table>")

        # Blocker Violations
        blockers = gate_report.get("blocker_violations", [])
        if blockers:
            html.append(
                f"<div style='margin-top: 16px;'><strong style='color: #742a2a;'>Blocker Violations ({len(blockers)}):</strong></div>"
            )
            html.append("<div style='margin-top: 8px;'>")

            for i, v in enumerate(blockers[:10], 1):  # Max 10
                html.append(f"""
                <div style='background: #fff5f5; border-left: 3px solid #c53030; padding: 8px; margin: 4px 0; font-size: 9pt;'>
                    <strong>{i}. [{v.get("gate", "unknown")}]</strong> {v.get("mismatch_type", "unknown")}<br>
                    <span style='color: #718096;'>Record: {v.get("record_id", "N/A")} | PMID: {v.get("pmid", "N/A")}</span><br>
                    <span>{v.get("message", "")[:150]}</span>
                </div>
                """)

            if len(blockers) > 10:
                html.append(
                    f"<div style='font-size: 8pt; color: #a0aec0;'>... and {len(blockers) - 10} more violations</div>"
                )

            html.append("</div>")

        # Enforcement Summary (if available)
        if enforcement_result:
            html.append(f"""
            <div style='margin-top: 16px; padding-top: 12px; border-top: 1px dashed #d69e2e;'>
                <strong>Enforcement Applied:</strong>
                <ul style='margin: 8px 0; padding-left: 20px; font-size: 9pt;'>
                    <li>Records Excluded: {len(enforcement_result.excluded_records)}</li>
                    <li>Snippets Excluded: {len(enforcement_result.excluded_snippets)}</li>
                    <li>Claims Degraded: {len(enforcement_result.degraded_claims)}</li>
                    <li>Claims Orphaned: {len(enforcement_result.orphan_claims)}</li>
                    <li>Hypotheses Suppressed: {len(enforcement_result.suppressed_hypotheses)}</li>
                </ul>
            </div>
            """)

        html.append("</div>")

        return "\n".join(html)

    def render_markdown(
        self,
        gate_report: dict,
        enforcement_result: EnforcementResult | None = None,
    ) -> str:
        """Render Gate Report as Markdown."""
        if not gate_report:
            return "No Gate Report available."

        lines = [
            "## DoD3 Gate Report",
            "",
            f"**Status:** {gate_report.get('overall_status', 'unknown').upper()}",
            f"**Generated:** {gate_report.get('generated_at', 'N/A')[:19]}",
            "",
        ]

        summary = gate_report.get("summary", {})
        lines.extend(
            [
                "### Summary",
                "",
                f"- Total Checks: {summary.get('total_checks', 0)}",
                f"- Passed: {summary.get('passed', 0)}",
                f"- Warned: {summary.get('warned', 0)}",
                f"- Failed: {summary.get('failed', 0)}",
                "",
            ]
        )

        blockers = gate_report.get("blocker_violations", [])
        if blockers:
            lines.extend(
                [
                    "### Blocker Violations",
                    "",
                ]
            )
            for i, v in enumerate(blockers[:10], 1):
                lines.append(
                    f"{i}. **[{v.get('gate', 'unknown')}]** {v.get('mismatch_type', 'unknown')}"
                )
                lines.append(
                    f"   - Record: {v.get('record_id', 'N/A')} | PMID: {v.get('pmid', 'N/A')}"
                )
                lines.append(f"   - {v.get('message', '')[:100]}")
            lines.append("")

        return "\n".join(lines)
