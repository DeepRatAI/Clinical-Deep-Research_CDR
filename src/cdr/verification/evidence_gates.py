"""
Evidence Validation Gates

Deterministic gates that validate evidence alignment with PICO requirements.
These are NOT prompt-based - they use pattern matching and structural checks.

Gates:
1. PopulationMatchGate - Verify evidence mentions PICO population, not excluded
2. ComparatorAlignmentGate - Verify comparator in evidence matches PICO
3. StudyTypeConsistencyGate - Filter or flag mismatched study designs
4. DeduplicationGate - Remove duplicate PMIDs per claim

References:
- PRISMA 2020: Eligibility criteria enforcement
- Cochrane Handbook Section 5: Eligibility criteria
- GRADE Handbook: Indirectness domain (population, comparator mismatches)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cdr.core.schemas import PICO, EvidenceClaim, Record, Snippet

logger = logging.getLogger(__name__)


class GateResult(str, Enum):
    """Result of a gate check."""

    PASS = "pass"
    WARN = "warn"  # Flag but don't block
    FAIL = "fail"  # Block or degrade
    SKIP = "skip"  # Not applicable


class MismatchType(str, Enum):
    """Type of mismatch detected."""

    POPULATION_EXCLUDED = "population_excluded"
    POPULATION_NOT_MENTIONED = "population_not_mentioned"
    COMPARATOR_INDIRECT = "comparator_indirect"
    COMPARATOR_MISSING = "comparator_missing"
    STUDY_TYPE_MISMATCH = "study_type_mismatch"
    DUPLICATE_EVIDENCE = "duplicate_evidence"
    EXCLUSION_CRITERIA_VIOLATION = "exclusion_criteria_violation"


@dataclass
class GateViolation:
    """A single gate violation."""

    gate_name: str
    mismatch_type: MismatchType
    result: GateResult
    record_id: str | None = None
    pmid: str | None = None
    snippet_id: str | None = None
    claim_id: str | None = None
    message: str = ""
    evidence_text: str | None = None  # Relevant snippet text

    def to_dict(self) -> dict:
        return {
            "gate": self.gate_name,
            "mismatch_type": self.mismatch_type.value,
            "result": self.result.value,
            "record_id": self.record_id,
            "pmid": self.pmid,
            "snippet_id": self.snippet_id,
            "claim_id": self.claim_id,
            "message": self.message,
        }


@dataclass
class GateCheckResult:
    """Result of running a gate on a piece of evidence."""

    gate_name: str
    result: GateResult
    violations: list[GateViolation] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.result in (GateResult.PASS, GateResult.SKIP)

    @property
    def failed(self) -> bool:
        return self.result == GateResult.FAIL

    @property
    def warned(self) -> bool:
        return self.result == GateResult.WARN


# =============================================================================
# POPULATION MATCH GATE
# =============================================================================


class PopulationMatchGate:
    """
    Gate 1: Verify evidence is relevant to PICO population.

    Hard fail conditions:
    - Abstract/text explicitly EXCLUDES the PICO population
    - Exclusion criteria list PICO population

    Warn conditions:
    - PICO population not mentioned in abstract

    Pass conditions:
    - PICO population mentioned as inclusion/study population
    """

    NAME = "population_match"

    # Common exclusion patterns - EXPANDED for better capture
    # Each pattern should capture text that lists excluded conditions
    EXCLUSION_PATTERNS = [
        # Explicit exclusion criteria sections
        r"(?:exclusion\s+criteria|excluded|excluding)[:\s]+([^.]+)",
        r"patients\s+with\s+(?:the\s+)?following\s+(?:conditions?\s+)?were\s+excluded[:\s]+([^.]+)",
        r"(?:we\s+)?excluded\s+(?:patients?|subjects?|participants?)\s+(?:with|who)[:\s]+([^.]+)",
        # "without" patterns (very common in clinical trials) - HIGH PRIORITY
        r"(?:participants?|patients?|subjects?|individuals?|adults?)\s+without\s+([^,\.]+(?:,\s*[^,\.]+)*)",
        r"without\s+(?:a\s+)?(?:diagnosed|history\s+of|prior)\s+([^,\.]+(?:,\s*[^,\.]+)*)",
        r"without\s+(?:a\s+)?diagnosed\s+(?:cardiovascular\s+event|[^,]+),\s*([^,\.]+)",
        # "no history of" patterns
        r"(?:with\s+)?no\s+(?:prior\s+)?history\s+of\s+([^,\.]+)",
        r"no\s+(?:known|diagnosed|prior)\s+([^,\.]+)",
        # Negative selection (elderly without condition)
        r"(?:older|elderly)\s+(?:adults?|patients?)\s+without\s+([^,\.]+)",
        # Were not included patterns
        r"(?:patients?|subjects?)\s+(?:were\s+not\s+included|not\s+enrolled)\s+(?:if\s+they\s+had\s+)?([^.]+)",
        # Ineligible patterns
        r"(?:ineligible|not\s+eligible)\s+(?:if|when)\s+(?:they\s+had\s+)?([^.]+)",
    ]

    # Population synonyms for common conditions
    POPULATION_SYNONYMS = {
        "atrial fibrillation": [
            "atrial fibrillation",
            "af",
            "a-fib",
            "afib",
            "atrial flutter",
            "afl",
            "afib/afl",
            "nonvalvular atrial fibrillation",
            "nvaf",
            "paroxysmal atrial fibrillation",
            "persistent atrial fibrillation",
            "permanent atrial fibrillation",
        ],
        "diabetes": [
            "diabetes",
            "diabetic",
            "dm",
            "t2dm",
            "t1dm",
            "type 2 diabetes",
            "type 1 diabetes",
            "diabetes mellitus",
            "hyperglycemia",
        ],
        "stroke": [
            "stroke",
            "cerebrovascular",
            "cva",
            "ischemic stroke",
            "hemorrhagic stroke",
            "tia",
            "transient ischemic attack",
        ],
        "heart failure": [
            "heart failure",
            "hf",
            "chf",
            "cardiac failure",
            "hfref",
            "hfpef",
            "lvef",
            "reduced ejection fraction",
        ],
    }

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, FAIL when population not mentioned.
                   If False, WARN when not mentioned.
        """
        self.strict = strict

    def _get_population_terms(self, population: str) -> list[str]:
        """Get all synonyms/variations for a population term."""
        pop_lower = population.lower()
        terms = [pop_lower]

        # Add synonyms from known conditions
        for key, synonyms in self.POPULATION_SYNONYMS.items():
            if key in pop_lower or any(s in pop_lower for s in synonyms[:3]):
                terms.extend(synonyms)

        # Add individual words (for multi-word populations)
        words = pop_lower.split()
        if len(words) > 1:
            terms.extend(w for w in words if len(w) > 3)

        return list(set(terms))

    def _is_in_exclusion_criteria(
        self, text: str, population_terms: list[str]
    ) -> tuple[bool, str | None]:
        """Check if population appears in exclusion criteria."""
        text_lower = text.lower()

        for pattern in self.EXCLUSION_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                exclusion_text = match.group(1) if match.lastindex else match.group(0)
                for term in population_terms:
                    if term in exclusion_text:
                        return True, exclusion_text

        return False, None

    def _population_mentioned(self, text: str, population_terms: list[str]) -> bool:
        """Check if any population term is mentioned."""
        text_lower = text.lower()
        return any(term in text_lower for term in population_terms)

    def check_record(self, record: "Record", pico: "PICO") -> GateCheckResult:
        """Check if a record is relevant to PICO population."""
        population_terms = self._get_population_terms(pico.population)

        # Get text to analyze
        text_to_check = (record.abstract or "") + " " + record.title

        # Check 1: Is population explicitly excluded?
        is_excluded, exclusion_text = self._is_in_exclusion_criteria(
            text_to_check, population_terms
        )

        if is_excluded:
            violation = GateViolation(
                gate_name=self.NAME,
                mismatch_type=MismatchType.POPULATION_EXCLUDED,
                result=GateResult.FAIL,
                record_id=record.record_id,
                pmid=record.pmid,
                message=f"PICO population '{pico.population}' appears in exclusion criteria",
                evidence_text=exclusion_text,
            )
            logger.warning(
                f"PopulationMatchGate: FAIL - {record.pmid or record.record_id} "
                f"excludes population '{pico.population}'"
            )
            return GateCheckResult(
                gate_name=self.NAME,
                result=GateResult.FAIL,
                violations=[violation],
                metadata={"population_terms": population_terms[:5]},
            )

        # Check 2: Is population mentioned at all?
        if not self._population_mentioned(text_to_check, population_terms):
            result = GateResult.FAIL if self.strict else GateResult.WARN
            violation = GateViolation(
                gate_name=self.NAME,
                mismatch_type=MismatchType.POPULATION_NOT_MENTIONED,
                result=result,
                record_id=record.record_id,
                pmid=record.pmid,
                message=f"PICO population '{pico.population}' not mentioned in abstract/title",
            )
            logger.info(
                f"PopulationMatchGate: {result.value} - {record.pmid or record.record_id} "
                f"does not mention population '{pico.population}'"
            )
            return GateCheckResult(
                gate_name=self.NAME,
                result=result,
                violations=[violation],
                metadata={"population_terms": population_terms[:5]},
            )

        # PASS: Population mentioned, not excluded
        logger.debug(
            f"PopulationMatchGate: PASS - {record.pmid or record.record_id} "
            f"mentions population '{pico.population}'"
        )
        return GateCheckResult(
            gate_name=self.NAME,
            result=GateResult.PASS,
            metadata={"population_terms": population_terms[:5]},
        )

    def check_snippet(self, snippet: "Snippet", pico: "PICO") -> GateCheckResult:
        """Check if a snippet is relevant to PICO population."""
        population_terms = self._get_population_terms(pico.population)

        # Check exclusion patterns in snippet text
        is_excluded, exclusion_text = self._is_in_exclusion_criteria(snippet.text, population_terms)

        if is_excluded:
            violation = GateViolation(
                gate_name=self.NAME,
                mismatch_type=MismatchType.EXCLUSION_CRITERIA_VIOLATION,
                result=GateResult.FAIL,
                snippet_id=snippet.snippet_id,
                record_id=snippet.source_ref.record_id,
                pmid=snippet.source_ref.pmid,
                message=f"Snippet mentions exclusion of PICO population",
                evidence_text=exclusion_text,
            )
            return GateCheckResult(
                gate_name=self.NAME,
                result=GateResult.FAIL,
                violations=[violation],
            )

        # For snippets, we don't require population mention (might be results section)
        return GateCheckResult(gate_name=self.NAME, result=GateResult.PASS)


# =============================================================================
# COMPARATOR ALIGNMENT GATE
# =============================================================================


class ComparatorAlignmentGate:
    """
    Gate 2: Verify evidence comparator aligns with PICO comparator.

    Fail conditions:
    - PICO.comparator = "placebo" but evidence is "drug A vs drug B"
    - Claim uses evidence from wrong comparator context

    Warn conditions:
    - Comparator not explicitly stated in evidence (indirect)

    Pass conditions:
    - Evidence comparator matches PICO comparator
    """

    NAME = "comparator_alignment"

    # Comparator categories
    PLACEBO_TERMS = [
        "placebo",
        "sham",
        "no treatment",
        "usual care",
        "standard care",
        "no intervention",
        "control",
        "untreated",
    ]

    ACTIVE_COMPARATOR_PATTERNS = [
        r"(\w+)\s+(?:vs?\.?|versus|compared\s+(?:to|with))\s+(\w+)",
        r"comparing\s+(\w+)\s+(?:and|with|to)\s+(\w+)",
    ]

    # Drug class mappings for detecting mismatches
    DRUG_CLASSES = {
        "doac": ["apixaban", "rivaroxaban", "dabigatran", "edoxaban", "doac", "noac"],
        "antiplatelet": ["aspirin", "clopidogrel", "ticagrelor", "prasugrel"],
        "anticoagulant": ["warfarin", "vka", "heparin", "enoxaparin", "fondaparinux"],
    }

    def __init__(self, strict: bool = False):
        self.strict = strict

    def _is_placebo_comparator(self, comparator: str | None) -> bool:
        """Check if comparator is placebo/no treatment."""
        if not comparator:
            return False
        comp_lower = comparator.lower()
        return any(term in comp_lower for term in self.PLACEBO_TERMS)

    def _extract_comparators_from_text(self, text: str) -> list[tuple[str, str]]:
        """Extract intervention vs comparator pairs from text."""
        pairs = []
        text_lower = text.lower()

        for pattern in self.ACTIVE_COMPARATOR_PATTERNS:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if match.lastindex and match.lastindex >= 2:
                    pairs.append((match.group(1), match.group(2)))

        return pairs

    def _get_drug_class(self, drug: str) -> str | None:
        """Get drug class for a drug name."""
        drug_lower = drug.lower()
        for drug_class, drugs in self.DRUG_CLASSES.items():
            if drug_lower in drugs or any(d in drug_lower for d in drugs):
                return drug_class
        return None

    def check_evidence_comparator(
        self,
        text: str,
        pico: "PICO",
        record_id: str | None = None,
        pmid: str | None = None,
    ) -> GateCheckResult:
        """Check if evidence comparator aligns with PICO."""
        if not pico.comparator:
            # No PICO comparator specified, skip
            return GateCheckResult(gate_name=self.NAME, result=GateResult.SKIP)

        pico_is_placebo = self._is_placebo_comparator(pico.comparator)

        # Extract comparators from evidence
        comparator_pairs = self._extract_comparators_from_text(text)

        if not comparator_pairs:
            # No explicit comparison found
            violation = GateViolation(
                gate_name=self.NAME,
                mismatch_type=MismatchType.COMPARATOR_MISSING,
                result=GateResult.WARN,
                record_id=record_id,
                pmid=pmid,
                message="No explicit comparator found in evidence text",
            )
            return GateCheckResult(
                gate_name=self.NAME,
                result=GateResult.WARN,
                violations=[violation],
            )

        # Check each comparator pair
        violations = []
        for int_drug, comp_drug in comparator_pairs:
            # Check if PICO expects placebo but evidence is active comparison
            if pico_is_placebo:
                comp_class = self._get_drug_class(comp_drug)
                if comp_class:  # Active comparator detected
                    violation = GateViolation(
                        gate_name=self.NAME,
                        mismatch_type=MismatchType.COMPARATOR_INDIRECT,
                        result=GateResult.FAIL,
                        record_id=record_id,
                        pmid=pmid,
                        message=f"PICO expects '{pico.comparator}' but evidence compares {int_drug} vs {comp_drug}",
                        evidence_text=f"{int_drug} vs {comp_drug}",
                    )
                    violations.append(violation)

        if violations:
            logger.warning(
                f"ComparatorAlignmentGate: FAIL - {pmid or record_id} "
                f"has indirect comparator (expected: {pico.comparator})"
            )
            return GateCheckResult(
                gate_name=self.NAME,
                result=GateResult.FAIL,
                violations=violations,
                metadata={"comparator_pairs": comparator_pairs[:3]},
            )

        return GateCheckResult(
            gate_name=self.NAME,
            result=GateResult.PASS,
            metadata={"comparator_pairs": comparator_pairs[:3]},
        )

    def check_record(self, record: "Record", pico: "PICO") -> GateCheckResult:
        """Check record for comparator alignment."""
        text = (record.abstract or "") + " " + record.title
        return self.check_evidence_comparator(text, pico, record.record_id, record.pmid)


# =============================================================================
# STUDY TYPE CONSISTENCY GATE
# =============================================================================


class StudyTypeConsistencyGate:
    """
    Gate 3: Verify study type matches PICO requirements.

    STRICT by default when PICO explicitly specifies study types.

    Fail conditions:
    - PICO.study_types = [RCT] but evidence is observational/secondary
    - Evidence is case_report or case_series (very low quality)

    Warn conditions:
    - Study type cannot be determined

    Pass conditions:
    - Study type matches PICO requirements
    """

    NAME = "study_type_consistency"

    # Study type hierarchies
    EXPERIMENTAL = ["rct", "randomized", "controlled trial", "clinical trial", "randomised"]
    OBSERVATIONAL = [
        "cohort",
        "case-control",
        "cross-sectional",
        "observational",
        "registry",
        "prospective",
        "retrospective",
    ]
    SECONDARY = [
        "systematic review",
        "meta-analysis",
        "review",
        "umbrella",
        "narrative review",
        "scoping review",
    ]
    CASE_LEVEL = ["case report", "case series", "case study"]  # Very low quality

    # Types that should be BLOCKED when RCT required (hard exclusion)
    BLOCK_WHEN_RCT = [
        "case_report",
        "case_series",
        "narrative_review",
        "letter",
        "editorial",
        "comment",
    ]

    def __init__(self, strict: bool = False, study_type_strict: bool = True):
        """
        Args:
            strict: General strict mode
            study_type_strict: When True, FAIL (not WARN) on type mismatch. Default True.
        """
        self.strict = strict
        self.study_type_strict = study_type_strict

    def _detect_study_type(self, record: "Record") -> tuple[str | None, str | None]:
        """
        Detect study type from record metadata.
        Returns: (category, specific_type) e.g. ('observational', 'cohort')
        """
        specific_type = None

        # Check publication types first (most reliable)
        for pt in record.publication_type:
            pt_lower = pt.lower()

            # Check for very low quality types first (block these)
            for case_t in self.CASE_LEVEL:
                if case_t in pt_lower:
                    return "case_level", pt_lower.replace(" ", "_")

            if any(exp in pt_lower for exp in self.EXPERIMENTAL):
                return "experimental", "rct"
            if any(obs in pt_lower for obs in self.OBSERVATIONAL):
                specific_type = next(
                    (obs for obs in self.OBSERVATIONAL if obs in pt_lower), "observational"
                )
                return "observational", specific_type
            if any(sec in pt_lower for sec in self.SECONDARY):
                specific_type = next(
                    (sec for sec in self.SECONDARY if sec in pt_lower), "secondary"
                )
                return "secondary", specific_type.replace(" ", "_")

        # Check abstract for clues
        abstract = (record.abstract or "").lower()
        title = record.title.lower()
        text = abstract + " " + title

        # Case level (block these)
        for case_t in self.CASE_LEVEL:
            if case_t in text:
                return "case_level", case_t.replace(" ", "_")

        if "randomized" in text or "randomised" in text or "rct" in text:
            if "trial" in text:
                return "experimental", "rct"
        if any(sec in text for sec in self.SECONDARY):
            specific = next((sec for sec in self.SECONDARY if sec in text), "secondary")
            return "secondary", specific.replace(" ", "_")
        if any(obs in text for obs in self.OBSERVATIONAL):
            specific = next((obs for obs in self.OBSERVATIONAL if obs in text), "observational")
            return "observational", specific

        return None, None

    def check_record(self, record: "Record", pico: "PICO") -> GateCheckResult:
        """Check if record study type matches PICO requirements."""
        from cdr.core.enums import StudyType

        if not pico.study_types:
            # No study type restriction
            return GateCheckResult(gate_name=self.NAME, result=GateResult.SKIP)

        category, specific_type = self._detect_study_type(record)

        # ALWAYS block case reports/series regardless of PICO
        if category == "case_level":
            violation = GateViolation(
                gate_name=self.NAME,
                mismatch_type=MismatchType.STUDY_TYPE_MISMATCH,
                result=GateResult.FAIL,
                record_id=record.record_id,
                pmid=record.pmid,
                message=f"Case-level evidence ({specific_type}) excluded - very low quality for systematic review",
            )
            logger.warning(
                f"StudyTypeGate: FAIL - {record.pmid or record.record_id} "
                f"is case-level evidence ({specific_type})"
            )
            return GateCheckResult(
                gate_name=self.NAME,
                result=GateResult.FAIL,
                violations=[violation],
                metadata={"category": category, "specific_type": specific_type},
            )

        if not category:
            violation = GateViolation(
                gate_name=self.NAME,
                mismatch_type=MismatchType.STUDY_TYPE_MISMATCH,
                result=GateResult.WARN,
                record_id=record.record_id,
                pmid=record.pmid,
                message="Could not detect study type from record",
            )
            return GateCheckResult(
                gate_name=self.NAME,
                result=GateResult.WARN,
                violations=[violation],
                metadata={"category": None, "specific_type": None},
            )

        # Check if detected type matches any required type
        pico_requires_experimental = any(st in [StudyType.RCT] for st in pico.study_types)
        pico_allows_observational = any(
            st in [StudyType.COHORT, StudyType.CASE_CONTROL] for st in pico.study_types
        )

        # RCT required but not experimental
        if pico_requires_experimental and category != "experimental":
            # Use study_type_strict (default True) for type enforcement
            result = GateResult.FAIL if self.study_type_strict else GateResult.WARN

            violation = GateViolation(
                gate_name=self.NAME,
                mismatch_type=MismatchType.STUDY_TYPE_MISMATCH,
                result=result,
                record_id=record.record_id,
                pmid=record.pmid,
                message=f"PICO requires RCT but record is {category}/{specific_type}",
            )

            log_fn = logger.warning if result == GateResult.FAIL else logger.info
            log_fn(
                f"StudyTypeGate: {result.value} - {record.pmid or record.record_id} "
                f"is {category}/{specific_type} (PICO requires RCT)"
            )

            return GateCheckResult(
                gate_name=self.NAME,
                result=result,
                violations=[violation],
                metadata={"category": category, "specific_type": specific_type},
            )

        # PASS: Study type matches requirements
        return GateCheckResult(
            gate_name=self.NAME,
            result=GateResult.PASS,
            metadata={"category": category, "specific_type": specific_type},
        )


# =============================================================================
# DEDUPLICATION GATE
# =============================================================================


class DeduplicationGate:
    """
    Gate 4: Remove duplicate evidence from claims.

    Checks:
    - Same PMID appearing multiple times in a claim's supporting evidence
    - Same content appearing under different IDs
    """

    NAME = "deduplication"

    def check_claim(self, claim: "EvidenceClaim", snippets: list["Snippet"]) -> GateCheckResult:
        """Check for duplicate evidence in a claim."""
        # Get PMIDs from supporting snippets
        pmid_to_snippets: dict[str, list[str]] = {}

        for sid in claim.supporting_snippet_ids:
            # Find snippet
            snippet = next((s for s in snippets if s.snippet_id == sid), None)
            if snippet and snippet.source_ref.pmid:
                pmid = snippet.source_ref.pmid
                if pmid not in pmid_to_snippets:
                    pmid_to_snippets[pmid] = []
                pmid_to_snippets[pmid].append(sid)

        # Find duplicates
        violations = []
        duplicate_pmids = []
        for pmid, snippet_ids in pmid_to_snippets.items():
            if len(snippet_ids) > 1:
                duplicate_pmids.append(pmid)
                violation = GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.DUPLICATE_EVIDENCE,
                    result=GateResult.WARN,
                    pmid=pmid,
                    claim_id=claim.claim_id,
                    message=f"PMID {pmid} appears {len(snippet_ids)} times in claim",
                )
                violations.append(violation)

        if violations:
            logger.warning(
                f"DeduplicationGate: WARN - Claim {claim.claim_id} has duplicate PMIDs: {duplicate_pmids}"
            )
            return GateCheckResult(
                gate_name=self.NAME,
                result=GateResult.WARN,
                violations=violations,
                metadata={"duplicate_pmids": duplicate_pmids},
            )

        return GateCheckResult(
            gate_name=self.NAME,
            result=GateResult.PASS,
            metadata={"unique_pmids": list(pmid_to_snippets.keys())},
        )


# =============================================================================
# CONCLUSION DEGRADATION GATE
# =============================================================================


class ConclusionDegradationGate:
    """
    Gate 5: Validate and potentially degrade conclusions based on evidence quality.

    This gate checks:
    1. Proportion of direct vs indirect evidence
    2. Study type quality (RCT vs observational)
    3. Population match quality

    When evidence is primarily indirect or weak, suggests degradation phrases.
    """

    NAME = "conclusion_degradation"

    # Phrases that indicate strong conclusions (to be degraded if evidence is weak)
    STRONG_PHRASES = [
        "is effective",
        "prevents",
        "reduces the risk",
        "should be used",
        "is recommended",
        "has been shown to",
        "demonstrates that",
    ]

    # Degradation phrases to suggest when evidence is weak
    DEGRADATION_PHRASES = {
        "indirect_comparator": "Based on indirect comparisons (no head-to-head trials), ",
        "population_excluded": "Evidence is limited as most studies excluded the target population; ",
        "observational_only": "Based primarily on observational evidence (no RCTs available), ",
        "mixed_quality": "Evidence certainty is very low due to mixed study quality; ",
        "few_studies": "Based on limited evidence (fewer than 3 relevant studies), ",
    }

    def __init__(self, strict: bool = False):
        self.strict = strict

    def check_evidence_quality(
        self,
        gate_results: list[GateCheckResult],
        total_records: int,
        direct_evidence_count: int,
    ) -> tuple[GateResult, str | None, list[str]]:
        """
        Analyze gate results to determine evidence quality.

        Returns:
            (result, reason, suggested_degradations)
        """
        # Count failures by type
        failure_counts = {
            "population_excluded": 0,
            "comparator_indirect": 0,
            "study_type_mismatch": 0,
        }

        for gr in gate_results:
            for v in gr.violations:
                if v.mismatch_type.value in failure_counts:
                    failure_counts[v.mismatch_type.value] += 1

        suggested_degradations = []

        # Check if majority of evidence has indirect comparator
        if failure_counts["comparator_indirect"] > total_records * 0.3:
            suggested_degradations.append(self.DEGRADATION_PHRASES["indirect_comparator"])

        # Check if significant population exclusions
        if failure_counts["population_excluded"] > 5:
            suggested_degradations.append(self.DEGRADATION_PHRASES["population_excluded"])

        # Check if very few direct studies
        if direct_evidence_count < 3:
            suggested_degradations.append(self.DEGRADATION_PHRASES["few_studies"])

        # Determine result
        if len(suggested_degradations) >= 2:
            result = GateResult.FAIL if self.strict else GateResult.WARN
            reason = "Multiple evidence quality concerns detected"
        elif len(suggested_degradations) == 1:
            result = GateResult.WARN
            reason = "Evidence quality concern detected"
        else:
            result = GateResult.PASS
            reason = None

        return result, reason, suggested_degradations

    def degrade_conclusion_text(
        self,
        conclusion: str,
        degradations: list[str],
    ) -> str:
        """
        Add degradation phrases to a conclusion if it uses strong language.

        Returns the modified conclusion text.
        """
        if not degradations:
            return conclusion

        # Check if conclusion uses strong language
        conclusion_lower = conclusion.lower()
        uses_strong_language = any(phrase in conclusion_lower for phrase in self.STRONG_PHRASES)

        if not uses_strong_language:
            # Already appropriately cautious
            return conclusion

        # Prepend degradation context
        degradation_prefix = " ".join(degradations)

        # Make first letter lowercase for smooth joining
        if conclusion and conclusion[0].isupper():
            modified_conclusion = conclusion[0].lower() + conclusion[1:]
        else:
            modified_conclusion = conclusion

        return f"{degradation_prefix}{modified_conclusion}"


# =============================================================================
# MASTER EVIDENCE VALIDATOR
# =============================================================================


@dataclass
class EvidenceValidationResult:
    """Complete validation result for a piece of evidence."""

    record_id: str | None = None
    pmid: str | None = None
    overall_result: GateResult = GateResult.PASS
    gate_results: list[GateCheckResult] = field(default_factory=list)
    is_in_scope: bool = True
    degraded_reason: str | None = None

    def __post_init__(self):
        """Compute overall result from gate results."""
        if any(gr.failed for gr in self.gate_results):
            self.overall_result = GateResult.FAIL
            self.is_in_scope = False
            # Get first failure reason
            for gr in self.gate_results:
                if gr.failed and gr.violations:
                    self.degraded_reason = gr.violations[0].message
                    break
        elif any(gr.warned for gr in self.gate_results):
            self.overall_result = GateResult.WARN

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "pmid": self.pmid,
            "overall_result": self.overall_result.value,
            "is_in_scope": self.is_in_scope,
            "degraded_reason": self.degraded_reason,
            "gates": {gr.gate_name: gr.result.value for gr in self.gate_results},
        }


class EvidenceValidator:
    """
    Master validator that runs all evidence gates.

    Usage:
        validator = EvidenceValidator(strict=True)
        result = validator.validate_record(record, pico)
        if not result.is_in_scope:
            # Exclude or degrade this evidence
    """

    def __init__(self, strict: bool = False, study_type_strict: bool = True):
        """
        Args:
            strict: If True, use strict mode for all gates (more failures).
            study_type_strict: If True, FAIL on study type mismatch. Default True.
        """
        self.population_gate = PopulationMatchGate(strict=strict)
        self.comparator_gate = ComparatorAlignmentGate(strict=strict)
        self.study_type_gate = StudyTypeConsistencyGate(
            strict=strict, study_type_strict=study_type_strict
        )
        self.dedup_gate = DeduplicationGate()
        self.conclusion_gate = ConclusionDegradationGate(strict=strict)
        self.strict = strict
        self.study_type_strict = study_type_strict

        # Track gate results for conclusion degradation
        self._all_gate_results: list[GateCheckResult] = []

    def validate_record(self, record: "Record", pico: "PICO") -> EvidenceValidationResult:
        """Run all gates on a record."""
        gate_results = []

        # Gate 1: Population match
        pop_result = self.population_gate.check_record(record, pico)
        gate_results.append(pop_result)

        # Gate 2: Comparator alignment
        comp_result = self.comparator_gate.check_record(record, pico)
        gate_results.append(comp_result)

        # Gate 3: Study type consistency
        study_result = self.study_type_gate.check_record(record, pico)
        gate_results.append(study_result)

        result = EvidenceValidationResult(
            record_id=record.record_id,
            pmid=record.pmid,
            gate_results=gate_results,
        )

        # Recompute overall after all gates
        result.__post_init__()

        logger.info(
            f"EvidenceValidator: {record.pmid or record.record_id} -> "
            f"{result.overall_result.value} (in_scope={result.is_in_scope})"
        )

        return result

    def validate_snippet(self, snippet: "Snippet", pico: "PICO") -> EvidenceValidationResult:
        """Run gates on a snippet."""
        gate_results = []

        # Gate 1: Population match (for exclusion criteria in snippet)
        pop_result = self.population_gate.check_snippet(snippet, pico)
        gate_results.append(pop_result)

        # Gate 2: Comparator alignment in snippet text
        comp_result = self.comparator_gate.check_evidence_comparator(
            snippet.text,
            pico,
            snippet.source_ref.record_id,
            snippet.source_ref.pmid,
        )
        gate_results.append(comp_result)

        result = EvidenceValidationResult(
            record_id=snippet.source_ref.record_id,
            pmid=snippet.source_ref.pmid,
            gate_results=gate_results,
        )
        result.__post_init__()

        return result

    def validate_claim(
        self,
        claim: "EvidenceClaim",
        snippets: list["Snippet"],
        pico: "PICO",
    ) -> EvidenceValidationResult:
        """Validate a claim's supporting evidence."""
        gate_results = []

        # Gate 4: Deduplication
        dedup_result = self.dedup_gate.check_claim(claim, snippets)
        gate_results.append(dedup_result)

        # Validate each supporting snippet
        snippet_violations = []
        for sid in claim.supporting_snippet_ids:
            snippet = next((s for s in snippets if s.snippet_id == sid), None)
            if snippet:
                snippet_result = self.validate_snippet(snippet, pico)
                if not snippet_result.is_in_scope:
                    for gr in snippet_result.gate_results:
                        snippet_violations.extend(gr.violations)

        # If any supporting snippet fails, claim is degraded
        if snippet_violations:
            # Add a synthetic gate result for snippet failures
            snippet_gate = GateCheckResult(
                gate_name="supporting_evidence",
                result=GateResult.FAIL if self.strict else GateResult.WARN,
                violations=snippet_violations,
            )
            gate_results.append(snippet_gate)

        result = EvidenceValidationResult(
            record_id=None,
            pmid=None,
            gate_results=gate_results,
        )
        result.__post_init__()

        return result

    def degrade_conclusion_if_needed(
        self,
        conclusion: str,
        total_records: int,
        direct_evidence_count: int,
    ) -> tuple[str, bool, list[str]]:
        """
        Check if conclusion needs degradation and apply it.

        Args:
            conclusion: The conclusion text to potentially degrade
            total_records: Total records screened
            direct_evidence_count: Count of records with direct evidence

        Returns:
            (modified_conclusion, was_degraded, degradation_reasons)
        """
        _result, _reason, degradations = self.conclusion_gate.check_evidence_quality(
            self._all_gate_results,
            total_records,
            direct_evidence_count,
        )

        if degradations:
            modified = self.conclusion_gate.degrade_conclusion_text(conclusion, degradations)
            logger.warning(
                f"ConclusionDegradationGate: Degrading conclusion. Reasons: {degradations}"
            )
            return modified, True, degradations

        return conclusion, False, []

    def record_gate_result(self, gate_result: GateCheckResult) -> None:
        """Record a gate result for later conclusion degradation analysis."""
        self._all_gate_results.append(gate_result)
