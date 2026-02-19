"""
DoD3 Evidence Gates - Complete Clinical Validity Enforcement

This module implements the complete DoD3 (Definition of Done Level 3) gate system
for validating clinical evidence alignment with PICO requirements.

CRITICAL: These gates are BLOCKER level - they must PASS for a run to be publishable.

Gate System:
1. PICOMatchGate - Full P/I/C/O validation per snippet/record
2. ComparatorIntegrityGate - comparator_source tracking + sub-PICO decomposition
3. StudyTypeEnforcementGate - Strict enforcement + stratification
4. ContextPurityGate - Therapeutic context tagging + purity rules
5. AssertionCoverageGate - Assertion extraction + coverage validation
6. GateReportGenerator - Visible gate report for audit trail

References:
- PRISMA 2020: Eligibility criteria enforcement
- GRADE Handbook: Indirectness domain (population, comparator mismatches)
- Cochrane Handbook Section 5: Eligibility criteria
- DoD3 Contract: CDR_Agent_Guidance_and_Development_Protocol.md
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cdr.core.schemas import PICO, EvidenceClaim, Record, Snippet

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================


class GateResult(str, Enum):
    """Result of a gate check."""

    PASS = "pass"
    WARN = "warn"  # Flag but don't block
    FAIL = "fail"  # Block or degrade (BLOCKER)
    SKIP = "skip"  # Not applicable


class MismatchType(str, Enum):
    """Type of mismatch detected - maps to exclusion reasons."""

    # Population mismatches (P)
    POPULATION_EXCLUDED = "population_excluded"
    POPULATION_NOT_MENTIONED = "population_not_mentioned"
    POPULATION_CONTEXT_MISMATCH = "population_context_mismatch"
    # P0-04: Population dominance violations
    POPULATION_INCIDENTAL = "population_incidental"  # Population mentioned but not as primary target

    # Intervention mismatches (I)
    INTERVENTION_NOT_FOUND = "intervention_not_found"
    INTERVENTION_INCIDENTAL = "intervention_incidental"

    # Comparator mismatches (C)
    COMPARATOR_INDIRECT = "comparator_indirect"
    COMPARATOR_MISSING = "comparator_missing"
    COMPARATOR_JUMPED = "comparator_jumped"  # Claim jumps from A vs B to A vs C

    # Outcome mismatches (O)
    OUTCOME_NOT_FOUND = "outcome_not_found"
    OUTCOME_MISMATCH = "outcome_mismatch"

    # Study type
    STUDY_TYPE_MISMATCH = "study_type_mismatch"
    STUDY_TYPE_UNKNOWN = "study_type_unknown"

    # Context purity
    CONTEXT_PURITY_VIOLATION = "context_purity_violation"
    MIXED_THERAPY_MODES = "mixed_therapy_modes"

    # P0-03: Evidence anchoring violations
    EVIDENCE_NOT_DIRECT_RCT = "evidence_not_direct_rct"  # Claim uses subanalysis/secondary source
    EVIDENCE_INDIRECT_SOURCE = "evidence_indirect_source"  # From SR/MA, not direct RCT

    # Other
    DUPLICATE_EVIDENCE = "duplicate_evidence"
    ASSERTION_UNSUPPORTED = "assertion_unsupported"
    INVALID_LEAP = "invalid_leap"


class PopulationRole(str, Enum):
    """P0-04: Role of population mention in the evidence."""

    PRIMARY_TARGET = "primary_target"  # Evidence is ABOUT this population
    INCIDENTAL_MENTION = "incidental_mention"  # Population mentioned but not studied
    EXCLUSION_CRITERION = "exclusion_criterion"  # Population was excluded
    SUBGROUP = "subgroup"  # Subgroup analysis within larger trial
    NOT_MENTIONED = "not_mentioned"  # Population not found in text


@dataclass
class GateViolation:
    """A single gate violation with full audit trail."""

    gate_name: str
    mismatch_type: MismatchType
    result: GateResult
    record_id: str | None = None
    pmid: str | None = None
    snippet_id: str | None = None
    claim_id: str | None = None
    message: str = ""
    evidence_text: str | None = None
    pico_component: str | None = None  # "P", "I", "C", "O"
    match_score: float | None = None  # 0.0 to 1.0
    # P0-04: Population role classification
    population_role: str | None = None  # PopulationRole value

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
            "evidence_text": self.evidence_text[:200] if self.evidence_text else None,
            "pico_component": self.pico_component,
            "match_score": self.match_score,
            # P0-04: Population role for dominance analysis
            "population_role": self.population_role,
        }


@dataclass
class GateCheckResult:
    """Result of running a gate on evidence."""

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

    def to_dict(self) -> dict:
        return {
            "gate": self.gate_name,
            "result": self.result.value,
            "passed": self.passed,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "metadata": self.metadata,
        }


# =============================================================================
# POPULATION CONTEXT TYPES
# =============================================================================


class PopulationContext(str, Enum):
    """Population context classification for stratification."""

    CLINICAL_AF = "clinical_af"
    SUBCLINICAL_AF = "subclinical_af"
    DEVICE_DETECTED_AF = "device_detected_af"
    POST_ABLATION = "post_ablation"
    AF_PCI = "af_pci"  # AF + percutaneous coronary intervention
    AF_ICH = "af_ich"  # AF + intracranial hemorrhage
    AF_LAAC = "af_laac"  # AF + left atrial appendage closure
    AF_TAVI = "af_tavi"  # AF + transcatheter aortic valve
    ELDERLY_NO_CVD = "elderly_no_cvd"  # Elderly without cardiovascular disease
    DIABETES_PRIMARY_PREV = "diabetes_primary_prevention"
    GENERAL = "general"
    UNKNOWN = "unknown"


class TherapyMode(str, Enum):
    """Therapy mode classification for context purity."""

    ASPIRIN_MONOTHERAPY = "aspirin_monotherapy"
    ASPIRIN_VS_PLACEBO = "aspirin_vs_placebo"
    DOAC_VS_ASPIRIN = "doac_vs_aspirin"
    ASPIRIN_PLUS_ANTICOAGULANT = "aspirin_plus_anticoagulant"
    DUAL_ANTIPLATELET = "dual_antiplatelet"
    TRIPLE_THERAPY = "triple_therapy"
    WARFARIN_VS_ASPIRIN = "warfarin_vs_aspirin"
    DOAC_VS_PLACEBO = "doac_vs_placebo"
    OTHER = "other"
    UNKNOWN = "unknown"


# =============================================================================
# PICO MATCH GATE (P/I/C/O full validation)
# =============================================================================


class PICOMatchGate:
    """
    Complete PICO matching gate - validates all 4 components.

    DoD3 BLOCKER: Evidence must match PICO components to support claims.

    For each component:
    - P: Population must be target, not excluded
    - I: Intervention must be real, not incidental mention
    - C: Comparator must align with PICO (no jumping)
    - O: Outcome must be measured, not just mentioned
    """

    NAME = "pico_match"

    # =========================================================================
    # POPULATION PATTERNS
    # =========================================================================

    EXCLUSION_PATTERNS = [
        # Explicit exclusion criteria sections
        r"(?:exclusion\s+criteria|excluded|excluding)[:\s]+([^.]+)",
        r"patients\s+with\s+(?:the\s+)?following\s+(?:conditions?\s+)?were\s+excluded[:\s]+([^.]+)",
        r"(?:we\s+)?excluded\s+(?:patients?|subjects?|participants?)\s+(?:with|who)[:\s]+([^.]+)",
        # "without" patterns (HIGH PRIORITY - catches ASPREE/CHIP type studies)
        r"(?:participants?|patients?|subjects?|individuals?|adults?|persons?)\s+without\s+([^,\.;]+(?:,\s*[^,\.;]+)*)",
        r"without\s+(?:a\s+)?(?:diagnosed|history\s+of|prior|known)\s+([^,\.;]+(?:,\s*[^,\.;]+)*)",
        r"without\s+(?:a\s+)?diagnosed\s+(?:cardiovascular\s+event|[^,;]+),\s*([^,\.;]+)",
        # "no history of" patterns
        r"(?:with\s+)?no\s+(?:prior\s+)?history\s+of\s+([^,\.;]+)",
        r"no\s+(?:known|diagnosed|prior|documented)\s+([^,\.;]+)",
        # Negative selection (elderly without condition)
        r"(?:older|elderly)\s+(?:adults?|patients?|persons?)\s+without\s+([^,\.;]+)",
        # Not included patterns
        r"(?:patients?|subjects?)\s+(?:were\s+not\s+included|not\s+enrolled)\s+(?:if\s+they\s+had\s+)?([^.]+)",
        # Ineligible patterns
        r"(?:ineligible|not\s+eligible)\s+(?:if|when)\s+(?:they\s+had\s+)?([^.]+)",
        # Absence as characteristic
        r"absence\s+of\s+([^,\.;]+)",
        r"free\s+of\s+([^,\.;]+)",
    ]

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
            "subclinical atrial fibrillation",
            "device-detected atrial fibrillation",
            "ahre",  # atrial high-rate episodes
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
            "systemic embolism",
            "cardioembolic",
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

    # =========================================================================
    # INTERVENTION PATTERNS
    # =========================================================================

    INTERVENTION_SYNONYMS = {
        "aspirin": [
            "aspirin",
            "asa",
            "acetylsalicylic acid",
            "acetyl salicylic acid",
            "antiplatelet",
            "aspirin therapy",
        ],
        "warfarin": [
            "warfarin",
            "coumadin",
            "vka",
            "vitamin k antagonist",
        ],
        "doac": [
            "doac",
            "noac",
            "apixaban",
            "rivaroxaban",
            "dabigatran",
            "edoxaban",
            "direct oral anticoagulant",
            "novel oral anticoagulant",
        ],
        "clopidogrel": [
            "clopidogrel",
            "plavix",
            "thienopyridine",
        ],
    }

    # Patterns that indicate incidental mention (not as intervention)
    INCIDENTAL_PATTERNS = [
        r"history\s+of\s+(?:aspirin|asa)\s+use",
        r"prior\s+(?:aspirin|asa)\s+(?:treatment|therapy)",
        r"concomitant\s+(?:aspirin|asa)",  # Might be incidental
        r"background\s+(?:aspirin|asa)\s+therapy",
    ]

    # =========================================================================
    # OUTCOME PATTERNS
    # =========================================================================

    OUTCOME_AS_PRIMARY = [
        r"primary\s+(?:end\s*point|outcome)[:\s]+([^.]+)",
        r"primary\s+efficacy\s+(?:end\s*point|outcome)[:\s]+([^.]+)",
        r"main\s+outcome\s+measure[:\s]+([^.]+)",
    ]

    OUTCOME_SYNONYMS = {
        "stroke": [
            "stroke",
            "ischemic stroke",
            "hemorrhagic stroke",
            "cva",
            "cerebrovascular event",
            "systemic embolism",
            "sse",
            "stroke or systemic embolism",
        ],
        "bleeding": [
            "bleeding",
            "hemorrhage",
            "major bleeding",
            "intracranial hemorrhage",
            "ich",
            "gastrointestinal bleeding",
            "gi bleeding",
        ],
        "cardiovascular": [
            "cardiovascular event",
            "mace",
            "major adverse cardiovascular event",
            "myocardial infarction",
            "mi",
            "cv death",
            "cardiovascular death",
        ],
        "mortality": [
            "mortality",
            "death",
            "all-cause mortality",
            "survival",
        ],
    }

    # =========================================================================
    # COMPARATOR PATTERNS
    # =========================================================================

    PLACEBO_TERMS = [
        "placebo",
        "sham",
        "no treatment",
        "usual care",
        "standard care",
        "no intervention",
        "control",
        "untreated",
        "no aspirin",
        "placebo-controlled",
        "no antiplatelet",
    ]

    COMPARATOR_EXTRACTION = [
        r"(\w+)\s+(?:vs?\.?|versus|compared\s+(?:to|with))\s+(\w+)",
        r"comparing\s+(\w+)\s+(?:and|with|to)\s+(\w+)",
        r"randomized\s+to\s+(\w+)\s+or\s+(\w+)",
        r"assigned\s+to\s+(?:receive\s+)?(\w+)\s+or\s+(\w+)",
    ]

    DRUG_CLASSES = {
        "doac": ["apixaban", "rivaroxaban", "dabigatran", "edoxaban", "doac", "noac"],
        "antiplatelet": ["aspirin", "asa", "clopidogrel", "ticagrelor", "prasugrel"],
        "anticoagulant": ["warfarin", "vka", "heparin", "enoxaparin", "fondaparinux"],
    }

    def __init__(
        self,
        population_threshold: float = 0.3,
        intervention_threshold: float = 0.5,
        outcome_threshold: float = 0.3,
        strict: bool = True,
    ):
        """
        Args:
            population_threshold: Minimum score for population match (0-1)
            intervention_threshold: Minimum score for intervention match (0-1)
            outcome_threshold: Minimum score for outcome match (0-1)
            strict: If True, FAIL on low match. If False, WARN.
        """
        self.population_threshold = population_threshold
        self.intervention_threshold = intervention_threshold
        self.outcome_threshold = outcome_threshold
        self.strict = strict

    def _get_synonyms(self, term: str, synonym_dict: dict) -> list[str]:
        """Get all synonyms for a term."""
        term_lower = term.lower()
        terms = [term_lower]

        for key, synonyms in synonym_dict.items():
            if key in term_lower or any(s in term_lower for s in synonyms[:3]):
                terms.extend(synonyms)

        # Add individual words for multi-word terms, but exclude common stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "with",
            "in",
            "of",
            "for",
            "to",
            "by",
            "from",
            "as",
            "on",
            "at",
            "into",
            "that",
            "which",
            "who",
            "whom",
            "this",
            "these",
            "those",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "patients",
            "patient",
            "study",
            "studies",
            "treatment",
            "therapy",
            "group",
            "groups",
            "trial",
            "trials",
            "research",
            "analysis",
        }
        words = term_lower.split()
        if len(words) > 1:
            terms.extend(w for w in words if len(w) > 3 and w not in stopwords)

        return list(set(terms))

    def _check_exclusion_patterns(
        self, text: str, population_terms: list[str]
    ) -> tuple[bool, str | None]:
        """Check if population appears in exclusion context."""
        text_lower = text.lower()

        for pattern in self.EXCLUSION_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                exclusion_text = match.group(1) if match.lastindex else match.group(0)
                for term in population_terms:
                    if term in exclusion_text:
                        return True, exclusion_text[:150]

        return False, None

    def _check_population_match(
        self, text: str, pico: "PICO"
    ) -> tuple[float, bool, str | None, PopulationRole]:
        """
        Check population match with P0-04 Population Dominance.

        Returns: (score, is_excluded, exclusion_text, population_role)

        PopulationRole determines if population mention is:
        - PRIMARY_TARGET: Study focuses on this population (full credit)
        - INCIDENTAL_MENTION: Population appears but not as focus (warning)
        - EXCLUSION_CRITERION: Population explicitly excluded (hard fail)
        - SUBGROUP: Post-hoc subgroup analysis (warning)
        - NOT_MENTIONED: No population match found (fail)

        Scoring:
        - 1.0 if primary population term found (e.g., "atrial fibrillation")
        - 0.5-0.9 for synonym matches
        - 0.0-0.3 if no matches
        """
        population_terms = self._get_synonyms(pico.population, self.POPULATION_SYNONYMS)
        text_lower = text.lower()

        # P0-04: Check exclusion first (EXCLUSION_CRITERION role)
        is_excluded, exclusion_text = self._check_exclusion_patterns(text, population_terms)
        if is_excluded:
            return 0.0, True, exclusion_text, PopulationRole.EXCLUSION_CRITERION

        # P0-04: Check for subgroup/post-hoc analysis patterns
        subgroup_patterns = [
            r"subgroup\s+analysis",
            r"post.?hoc\s+(?:analysis|subgroup)",
            r"exploratory\s+(?:analysis|subgroup)",
            r"in\s+the\s+subgroup\s+of",
            r"(?:pre.?specified|pre-specified)\s+subgroup",
        ]
        for pattern in subgroup_patterns:
            if re.search(pattern, text_lower):
                # Found subgroup context - check if population is mentioned
                primary_term = pico.population.lower().strip()
                if primary_term in text_lower or any(
                    term in text_lower for term in population_terms
                ):
                    return 0.5, False, None, PopulationRole.SUBGROUP

        # P0-04: Check for incidental mention patterns
        incidental_patterns = [
            r"(?:also|including|such\s+as)\s+patients?\s+with",
            r"(?:subset|subpopulation)\s+of\s+patients",
            r"(?:among|in)\s+those\s+(?:who|with)",
            r"secondary\s+(?:population|cohort)",
        ]

        # Primary term match gets full score (PRIMARY_TARGET role)
        primary_term = pico.population.lower().strip()
        if primary_term in text_lower:
            # Check if it's an incidental mention despite primary term match
            for pattern in incidental_patterns:
                match = re.search(pattern, text_lower)
                if match and primary_term in text_lower[match.start() :]:
                    return 0.4, False, None, PopulationRole.INCIDENTAL_MENTION
            return 1.0, False, None, PopulationRole.PRIMARY_TARGET

        # Check for any synonym match
        matched_terms = [term for term in population_terms if term in text_lower]
        if matched_terms:
            # Check for incidental mention with synonyms
            for pattern in incidental_patterns:
                match = re.search(pattern, text_lower)
                if match and any(
                    term in text_lower[match.start() :] for term in matched_terms
                ):
                    return 0.3, False, None, PopulationRole.INCIDENTAL_MENTION

            # Synonym match as primary target
            max_match_len = max(len(term) for term in matched_terms)
            score = min(1.0, 0.5 + (max_match_len / 30))
            return score, False, None, PopulationRole.PRIMARY_TARGET

        # No match
        return 0.0, False, None, PopulationRole.NOT_MENTIONED

    def _check_intervention_match(self, text: str, pico: "PICO") -> tuple[float, bool]:
        """
        Check intervention match.

        Returns: (score, is_incidental)
        """
        intervention_terms = self._get_synonyms(pico.intervention, self.INTERVENTION_SYNONYMS)
        text_lower = text.lower()

        # Check for incidental mention
        for pattern in self.INCIDENTAL_PATTERNS:
            if re.search(pattern, text_lower):
                return 0.2, True  # Low score, incidental

        # Calculate match score
        matches = sum(1 for term in intervention_terms if term in text_lower)
        if not intervention_terms:
            return 0.5, False

        score = min(1.0, matches / max(2, len(intervention_terms) // 2))
        return score, False

    def _check_outcome_match(self, text: str, pico: "PICO") -> float:
        """Check outcome match."""
        outcome_terms = self._get_synonyms(pico.outcome, self.OUTCOME_SYNONYMS)
        text_lower = text.lower()

        # Bonus for primary outcome mention
        is_primary = any(re.search(p, text_lower) for p in self.OUTCOME_AS_PRIMARY)

        matches = sum(1 for term in outcome_terms if term in text_lower)
        if not outcome_terms:
            return 0.5

        score = min(1.0, matches / max(2, len(outcome_terms) // 2))
        if is_primary and score > 0:
            score = min(1.0, score + 0.2)

        return score

    def _check_comparator_match(self, text: str, pico: "PICO") -> tuple[float, str | None, bool]:
        """
        Check comparator match.

        Returns: (score, detected_comparator, is_jumped)
        """
        if not pico.comparator:
            return 1.0, None, False  # No comparator to check

        text_lower = text.lower()
        pico_is_placebo = any(t in pico.comparator.lower() for t in self.PLACEBO_TERMS)

        # Extract comparators from text
        detected_comparators = []
        for pattern in self.COMPARATOR_EXTRACTION:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if match.lastindex and match.lastindex >= 2:
                    detected_comparators.append((match.group(1), match.group(2)))

        if not detected_comparators:
            # No comparator found - WARN but don't fail
            return 0.5, None, False

        # Check if comparator matches PICO
        for int_drug, comp_drug in detected_comparators:
            # If PICO expects placebo but evidence is active comparison
            if pico_is_placebo:
                comp_class = None
                for cls, drugs in self.DRUG_CLASSES.items():
                    if comp_drug in drugs or any(d in comp_drug for d in drugs):
                        comp_class = cls
                        break

                if comp_class:  # Active comparator when placebo expected
                    return 0.0, f"{int_drug} vs {comp_drug}", True

        # Check if PICO comparator terms are present
        comparator_terms = self._get_synonyms(pico.comparator, {})
        matches = sum(1 for term in comparator_terms if term in text_lower)
        score = min(1.0, matches / max(1, len(comparator_terms)))

        return score, detected_comparators[0] if detected_comparators else None, False

    def check_snippet(self, snippet: "Snippet", pico: "PICO") -> GateCheckResult:
        """Full PICO validation on a snippet."""
        violations = []
        text = snippet.text

        # P - Population (P0-04: includes population_role)
        pop_score, is_excluded, excl_text, pop_role = self._check_population_match(text, pico)
        if is_excluded:
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.POPULATION_EXCLUDED,
                    result=GateResult.FAIL,
                    snippet_id=snippet.snippet_id,
                    record_id=snippet.source_ref.record_id,
                    pmid=snippet.source_ref.pmid,
                    message=f"PICO population '{pico.population}' appears in exclusion context",
                    evidence_text=excl_text,
                    pico_component="P",
                    match_score=0.0,
                    population_role=pop_role,
                )
            )
        elif pop_role == PopulationRole.INCIDENTAL_MENTION:
            # P0-04: Population mentioned but not as primary target
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.POPULATION_INCIDENTAL,
                    result=GateResult.WARN,
                    snippet_id=snippet.snippet_id,
                    record_id=snippet.source_ref.record_id,
                    pmid=snippet.source_ref.pmid,
                    message=f"Population '{pico.population}' mentioned incidentally, not as primary target",
                    pico_component="P",
                    match_score=pop_score,
                    population_role=pop_role,
                )
            )
        elif pop_role == PopulationRole.SUBGROUP:
            # P0-04: Population only in subgroup analysis context
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.POPULATION_INCIDENTAL,
                    result=GateResult.WARN,
                    snippet_id=snippet.snippet_id,
                    record_id=snippet.source_ref.record_id,
                    pmid=snippet.source_ref.pmid,
                    message=f"Population '{pico.population}' appears in subgroup analysis context",
                    pico_component="P",
                    match_score=pop_score,
                    population_role=pop_role,
                )
            )
        elif pop_score < self.population_threshold:
            result = GateResult.FAIL if self.strict else GateResult.WARN
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.POPULATION_NOT_MENTIONED,
                    result=result,
                    snippet_id=snippet.snippet_id,
                    record_id=snippet.source_ref.record_id,
                    pmid=snippet.source_ref.pmid,
                    message=f"Population match score {pop_score:.2f} below threshold {self.population_threshold}",
                    pico_component="P",
                    match_score=pop_score,
                    population_role=pop_role,
                )
            )

        # I - Intervention
        int_score, is_incidental = self._check_intervention_match(text, pico)
        if is_incidental:
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.INTERVENTION_INCIDENTAL,
                    result=GateResult.WARN,
                    snippet_id=snippet.snippet_id,
                    record_id=snippet.source_ref.record_id,
                    pmid=snippet.source_ref.pmid,
                    message=f"Intervention '{pico.intervention}' appears incidentally, not as study intervention",
                    pico_component="I",
                    match_score=int_score,
                )
            )
        elif int_score < self.intervention_threshold:
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.INTERVENTION_NOT_FOUND,
                    result=GateResult.WARN,
                    snippet_id=snippet.snippet_id,
                    record_id=snippet.source_ref.record_id,
                    pmid=snippet.source_ref.pmid,
                    message=f"Intervention match score {int_score:.2f} below threshold",
                    pico_component="I",
                    match_score=int_score,
                )
            )

        # C - Comparator
        comp_score, detected_comp, is_jumped = self._check_comparator_match(text, pico)
        if is_jumped:
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.COMPARATOR_JUMPED,
                    result=GateResult.FAIL,
                    snippet_id=snippet.snippet_id,
                    record_id=snippet.source_ref.record_id,
                    pmid=snippet.source_ref.pmid,
                    message=f"PICO expects '{pico.comparator}' but evidence shows '{detected_comp}'",
                    evidence_text=str(detected_comp),
                    pico_component="C",
                    match_score=0.0,
                )
            )

        # O - Outcome
        out_score = self._check_outcome_match(text, pico)
        if out_score < self.outcome_threshold:
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.OUTCOME_NOT_FOUND,
                    result=GateResult.WARN,
                    snippet_id=snippet.snippet_id,
                    record_id=snippet.source_ref.record_id,
                    pmid=snippet.source_ref.pmid,
                    message=f"Outcome match score {out_score:.2f} below threshold",
                    pico_component="O",
                    match_score=out_score,
                )
            )

        # Determine overall result
        if any(v.result == GateResult.FAIL for v in violations):
            result = GateResult.FAIL
        elif violations:
            result = GateResult.WARN
        else:
            result = GateResult.PASS

        return GateCheckResult(
            gate_name=self.NAME,
            result=result,
            violations=violations,
            metadata={
                "population_score": pop_score,
                "intervention_score": int_score,
                "outcome_score": out_score,
                "comparator_score": comp_score,
                "is_excluded": is_excluded,
            },
        )

    def check_record(self, record: "Record", pico: "PICO") -> GateCheckResult:
        """Full PICO validation on a record (abstract + title)."""
        from cdr.core.schemas import Snippet, SourceRef

        # Create synthetic snippet from record
        text = f"{record.title}\n\n{record.abstract or ''}"

        # Use internal logic directly
        violations = []

        # P - Population (P0-04: includes population_role)
        pop_score, is_excluded, excl_text, pop_role = self._check_population_match(text, pico)
        if is_excluded:
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.POPULATION_EXCLUDED,
                    result=GateResult.FAIL,
                    record_id=record.record_id,
                    pmid=record.pmid,
                    message=f"PICO population '{pico.population}' in exclusion criteria",
                    evidence_text=excl_text,
                    pico_component="P",
                    match_score=0.0,
                    population_role=pop_role,
                )
            )
        elif pop_role == PopulationRole.INCIDENTAL_MENTION:
            # P0-04: Population mentioned but not as primary target
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.POPULATION_INCIDENTAL,
                    result=GateResult.WARN,
                    record_id=record.record_id,
                    pmid=record.pmid,
                    message=f"Population '{pico.population}' mentioned incidentally, not as primary target",
                    pico_component="P",
                    match_score=pop_score,
                    population_role=pop_role,
                )
            )
        elif pop_role == PopulationRole.SUBGROUP:
            # P0-04: Population only in subgroup analysis context
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.POPULATION_INCIDENTAL,
                    result=GateResult.WARN,
                    record_id=record.record_id,
                    pmid=record.pmid,
                    message=f"Population '{pico.population}' appears in subgroup analysis context",
                    pico_component="P",
                    match_score=pop_score,
                    population_role=pop_role,
                )
            )
        elif pop_score < self.population_threshold:
            result = GateResult.FAIL if self.strict else GateResult.WARN
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.POPULATION_NOT_MENTIONED,
                    result=result,
                    record_id=record.record_id,
                    pmid=record.pmid,
                    message=f"Population match score {pop_score:.2f} below threshold",
                    pico_component="P",
                    match_score=pop_score,
                    population_role=pop_role,
                )
            )

        # C - Comparator (check for indirect comparator)
        comp_score, detected_comp, is_jumped = self._check_comparator_match(text, pico)
        if is_jumped:
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.COMPARATOR_INDIRECT,
                    result=GateResult.FAIL,
                    record_id=record.record_id,
                    pmid=record.pmid,
                    message=f"Evidence comparator '{detected_comp}' != PICO '{pico.comparator}'",
                    evidence_text=str(detected_comp),
                    pico_component="C",
                    match_score=0.0,
                )
            )

        # Determine overall result
        if any(v.result == GateResult.FAIL for v in violations):
            result = GateResult.FAIL
        elif violations:
            result = GateResult.WARN
        else:
            result = GateResult.PASS

        return GateCheckResult(
            gate_name=self.NAME,
            result=result,
            violations=violations,
            metadata={
                "population_score": pop_score,
                "comparator_score": comp_score,
                "is_excluded": is_excluded,
                "detected_comparator": str(detected_comp) if detected_comp else None,
            },
        )


# =============================================================================
# STUDY TYPE ENFORCEMENT GATE
# =============================================================================


class StudyTypeEnforcementGate:
    """
    Strict study type enforcement with stratification.

    DoD3 BLOCKER: If PICO.study_types = [RCT], only RCT/cluster RCT pass.
    Non-matching types go to separate stratum or are excluded.
    """

    NAME = "study_type_enforcement"

    # Study type detection patterns
    EXPERIMENTAL = ["rct", "randomized", "controlled trial", "clinical trial", "randomised"]
    OBSERVATIONAL = [
        "cohort",
        "case-control",
        "cross-sectional",
        "observational",
        "registry",
        "prospective",
        "retrospective",
        "real-world",
    ]
    SECONDARY = [
        "systematic review",
        "meta-analysis",
        "review",
        "umbrella",
        "narrative review",
        "scoping review",
        "pooled analysis",
    ]
    CASE_LEVEL = ["case report", "case series", "case study"]
    LETTERS_EDITORIALS = ["letter", "editorial", "comment", "correspondence"]

    # Map detected types to StudyType enum values
    TYPE_MAPPING = {
        "rct": "rct",
        "randomized controlled trial": "rct",
        "cohort": "cohort",
        "case-control": "case_control",
        "cross-sectional": "cross_sectional",
        "systematic review": "systematic_review",
        "meta-analysis": "meta_analysis",
        "case report": "case_report",
        "case series": "case_series",
    }

    def __init__(self, strict: bool = True, allow_stratification: bool = True):
        """
        Args:
            strict: If True, FAIL on type mismatch. If False, WARN.
            allow_stratification: If True, allow mixed types but stratify.
        """
        self.strict = strict
        self.allow_stratification = allow_stratification

    def _detect_study_type(self, record: "Record") -> tuple[str | None, str | None]:
        """
        Detect study type from record.

        Returns: (category, specific_type)
        """
        # Check publication types first (most reliable)
        for pt in record.publication_type:
            pt_lower = pt.lower()

            # Letters/editorials - always exclude
            if any(t in pt_lower for t in self.LETTERS_EDITORIALS):
                return "letter_editorial", pt_lower

            # Case level - very low quality
            for case_t in self.CASE_LEVEL:
                if case_t in pt_lower:
                    return "case_level", case_t.replace(" ", "_")

            # Experimental
            if any(exp in pt_lower for exp in self.EXPERIMENTAL):
                return "experimental", "rct"

            # Secondary
            if any(sec in pt_lower for sec in self.SECONDARY):
                specific = next((sec for sec in self.SECONDARY if sec in pt_lower), "secondary")
                return "secondary", specific.replace(" ", "_")

            # Observational
            if any(obs in pt_lower for obs in self.OBSERVATIONAL):
                specific = next(
                    (obs for obs in self.OBSERVATIONAL if obs in pt_lower), "observational"
                )
                return "observational", specific

        # Check title + abstract
        text = f"{record.title} {record.abstract or ''}".lower()

        # Letters/editorials
        if any(t in text for t in self.LETTERS_EDITORIALS):
            return "letter_editorial", "letter"

        # Case level
        for case_t in self.CASE_LEVEL:
            if case_t in text:
                return "case_level", case_t.replace(" ", "_")

        # RCT detection (look for randomization language)
        if "randomized" in text or "randomised" in text:
            if "trial" in text or "controlled" in text:
                return "experimental", "rct"

        # Secondary
        for sec in self.SECONDARY:
            if sec in text:
                return "secondary", sec.replace(" ", "_")

        # Observational
        for obs in self.OBSERVATIONAL:
            if obs in text:
                return "observational", obs

        return None, None

    def check_record(self, record: "Record", pico: "PICO") -> GateCheckResult:
        """Check study type compliance with PICO requirements."""
        from cdr.core.enums import StudyType

        if not pico.study_types:
            return GateCheckResult(
                gate_name=self.NAME,
                result=GateResult.SKIP,
                metadata={"reason": "No study type requirement in PICO"},
            )

        category, specific_type = self._detect_study_type(record)

        # Always block letters/editorials and case reports
        if category in ("letter_editorial", "case_level"):
            violation = GateViolation(
                gate_name=self.NAME,
                mismatch_type=MismatchType.STUDY_TYPE_MISMATCH,
                result=GateResult.FAIL,
                record_id=record.record_id,
                pmid=record.pmid,
                message=f"{category}/{specific_type} excluded - not evidence synthesis eligible",
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
                mismatch_type=MismatchType.STUDY_TYPE_UNKNOWN,
                result=GateResult.WARN,
                record_id=record.record_id,
                pmid=record.pmid,
                message="Could not detect study type",
            )
            return GateCheckResult(
                gate_name=self.NAME,
                result=GateResult.WARN,
                violations=[violation],
                metadata={"category": None, "specific_type": None},
            )

        # Check if PICO requires RCT
        pico_requires_rct = any(st in [StudyType.RCT] for st in pico.study_types)

        if pico_requires_rct and category != "experimental":
            result = GateResult.FAIL if self.strict else GateResult.WARN

            violation = GateViolation(
                gate_name=self.NAME,
                mismatch_type=MismatchType.STUDY_TYPE_MISMATCH,
                result=result,
                record_id=record.record_id,
                pmid=record.pmid,
                message=f"PICO requires RCT but record is {category}/{specific_type}",
            )

            return GateCheckResult(
                gate_name=self.NAME,
                result=result,
                violations=[violation],
                metadata={
                    "category": category,
                    "specific_type": specific_type,
                    "stratum": category if self.allow_stratification else None,
                },
            )

        return GateCheckResult(
            gate_name=self.NAME,
            result=GateResult.PASS,
            metadata={"category": category, "specific_type": specific_type},
        )


# =============================================================================
# CONTEXT PURITY GATE
# =============================================================================


class ContextPurityGate:
    """
    Therapeutic context tagging and purity enforcement.

    DoD3 BLOCKER: Claims cannot mix incompatible therapy modes.
    """

    NAME = "context_purity"

    # Context detection patterns
    CONTEXT_PATTERNS = {
        PopulationContext.POST_ABLATION: [
            r"post.?ablation",
            r"after\s+ablation",
            r"ablation\s+for\s+af",
            r"catheter\s+ablation",
            r"pulmonary\s+vein\s+isolation",
        ],
        PopulationContext.SUBCLINICAL_AF: [
            r"subclinical\s+(?:atrial\s+fibrillation|af)",
            r"device.?detected\s+(?:af|atrial)",
            r"ahre",
            r"atrial\s+high.?rate\s+episode",
        ],
        PopulationContext.AF_PCI: [
            r"af\s+(?:and|with)\s+pci",
            r"atrial\s+fibrillation\s+(?:and|with)\s+(?:pci|stent)",
            r"percutaneous\s+coronary",
            r"af.?pci",
            r"acs\s+(?:and|with)\s+af",
        ],
        PopulationContext.AF_ICH: [
            r"intracranial\s+hemorrhage",
            r"ich\s+(?:and|with)\s+af",
            r"af\s+(?:after|following)\s+ich",
        ],
        PopulationContext.AF_LAAC: [
            r"laa\s*c",
            r"left\s+atrial\s+appendage\s+closure",
            r"watchman",
            r"amulet",
        ],
        PopulationContext.ELDERLY_NO_CVD: [
            r"elderly\s+without\s+(?:cardiovascular|cv)",
            r"older\s+adults?\s+without",
            r"healthy\s+elderly",
        ],
    }

    THERAPY_PATTERNS = {
        TherapyMode.ASPIRIN_MONOTHERAPY: [
            r"aspirin\s+(?:alone|monotherapy)",
            r"aspirin\s+vs?\s+placebo",
        ],
        TherapyMode.DOAC_VS_ASPIRIN: [
            r"(?:apixaban|rivaroxaban|dabigatran|edoxaban|doac)\s+vs?\s+aspirin",
            r"(?:apixaban|rivaroxaban|dabigatran|edoxaban)\s+compared\s+(?:to|with)\s+aspirin",
        ],
        TherapyMode.ASPIRIN_PLUS_ANTICOAGULANT: [
            r"aspirin\s+(?:plus|and|\+)\s+(?:anticoagulant|doac|warfarin|apixaban|rivaroxaban|dabigatran|edoxaban)",
            r"(?:anticoagulant|doac|warfarin|apixaban|rivaroxaban|dabigatran|edoxaban)\s+(?:plus|and|\+)\s+aspirin",
            r"dual\s+(?:antithrombotic|therapy)",
            r"triple\s+therapy",
            r"tact",
        ],
        TherapyMode.WARFARIN_VS_ASPIRIN: [
            r"warfarin\s+vs?\s+aspirin",
            r"warfarin\s+compared\s+(?:to|with)\s+aspirin",
        ],
    }

    def __init__(self, strict: bool = True):
        self.strict = strict

    def detect_population_context(self, text: str) -> PopulationContext:
        """Detect population context from text."""
        text_lower = text.lower()

        for context, patterns in self.CONTEXT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return context

        return PopulationContext.GENERAL

    def detect_therapy_mode(self, text: str) -> TherapyMode:
        """Detect therapy mode from text."""
        text_lower = text.lower()

        for mode, patterns in self.THERAPY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return mode

        return TherapyMode.UNKNOWN

    def check_claim_purity(
        self,
        claim: "EvidenceClaim",
        snippets: list["Snippet"],
    ) -> GateCheckResult:
        """
        Check if claim mixes incompatible contexts.

        A claim should not mix:
        - aspirin_monotherapy with aspirin_plus_anticoagulant
        - post_ablation with general AF
        - etc.
        """
        violations = []

        # Get contexts from supporting snippets
        contexts = set()
        modes = set()

        for sid in claim.supporting_snippet_ids:
            snippet = next((s for s in snippets if s.snippet_id == sid), None)
            if snippet:
                ctx = self.detect_population_context(snippet.text)
                mode = self.detect_therapy_mode(snippet.text)
                contexts.add(ctx)
                modes.add(mode)

        # Check for incompatible mixes
        incompatible_mode_pairs = [
            (TherapyMode.ASPIRIN_MONOTHERAPY, TherapyMode.ASPIRIN_PLUS_ANTICOAGULANT),
            (TherapyMode.ASPIRIN_VS_PLACEBO, TherapyMode.DOAC_VS_ASPIRIN),
        ]

        for mode1, mode2 in incompatible_mode_pairs:
            if mode1 in modes and mode2 in modes:
                violations.append(
                    GateViolation(
                        gate_name=self.NAME,
                        mismatch_type=MismatchType.MIXED_THERAPY_MODES,
                        result=GateResult.FAIL if self.strict else GateResult.WARN,
                        claim_id=claim.claim_id,
                        message=f"Claim mixes incompatible therapy modes: {mode1.value} and {mode2.value}",
                    )
                )

        # Check for too many different contexts (suggests heterogeneity)
        specific_contexts = {
            c for c in contexts if c not in (PopulationContext.GENERAL, PopulationContext.UNKNOWN)
        }
        if len(specific_contexts) > 2:
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.CONTEXT_PURITY_VIOLATION,
                    result=GateResult.WARN,
                    claim_id=claim.claim_id,
                    message=f"Claim spans {len(specific_contexts)} different population contexts",
                )
            )

        if any(v.result == GateResult.FAIL for v in violations):
            result = GateResult.FAIL
        elif violations:
            result = GateResult.WARN
        else:
            result = GateResult.PASS

        return GateCheckResult(
            gate_name=self.NAME,
            result=result,
            violations=violations,
            metadata={
                "contexts_detected": [c.value for c in contexts],
                "modes_detected": [m.value for m in modes],
            },
        )


# =============================================================================
# ASSERTION COVERAGE GATE
# =============================================================================


@dataclass
class Assertion:
    """An atomic assertion extracted from a claim."""

    assertion_id: str
    text: str
    strength: str  # "strong", "weak", "uncertain"
    polarity: str  # "beneficial", "harm", "no_difference", "unclear"
    claim_id: str
    supporting_snippet_ids: list[str]
    pico_components_matched: dict[str, bool]  # {"P": True, "I": True, ...}

    def to_dict(self) -> dict:
        return {
            "assertion_id": self.assertion_id,
            "text": self.text,
            "strength": self.strength,
            "polarity": self.polarity,
            "claim_id": self.claim_id,
            "supporting_count": len(self.supporting_snippet_ids),
            "pico_matched": self.pico_components_matched,
        }


class AssertionCoverageGate:
    """
    Extract assertions and validate coverage.

    DoD3 BLOCKER: Strong assertions must have structural support.
    """

    NAME = "assertion_coverage"

    # Patterns for assertion strength
    STRONG_PATTERNS = [
        r"significantly\s+(?:reduces?|reduced|increased?|improved?)",
        r"demonstrated\s+(?:superiority|benefit|efficacy)",
        r"(?:is|was)\s+(?:effective|superior|beneficial)",
        r"should\s+be\s+(?:used|recommended|preferred)",
        r"prevents?\s+(?:stroke|death|events)",
        r"reduces?\s+(?:risk|incidence|mortality)",
        r"significantly\s+\w+\s+(?:risk|stroke|mortality)",  # "significantly reduces risk"
    ]

    WEAK_PATTERNS = [
        r"may\s+(?:reduce|prevent|be|have)",
        r"might\s+(?:reduce|prevent|be)",
        r"could\s+(?:reduce|prevent|be)",
        r"(?:limited|insufficient)\s+(?:evidence|efficacy)",
        r"uncertain(?:ty)?",
        r"inconclusive",
        r"limited\s+efficacy",
    ]

    BENEFICIAL_PATTERNS = [
        r"reduc(?:e|ed|es|ing)\s+(?:risk|incidence|stroke)",
        r"prevent(?:s|ed|ing)?(?:\s+stroke)?",
        r"protect(?:s|ed|ing)?",
        r"benefit(?:s|ed)?",
        r"improv(?:e|ed|es|ing)",
        r"superior",
        r"reduces\s+stroke\s+risk",
    ]

    HARM_PATTERNS = [
        r"increas(?:e|ed|es|ing)\s+(?:risk|bleeding)",
        r"(?:caus|lead)(?:e|ed|es|ing)\s+(?:harm|bleeding|death)",
        r"harmful",
        r"danger(?:ous)?",
        r"adverse",
    ]

    def __init__(self, strict: bool = True):
        self.strict = strict

    def extract_assertions(
        self,
        claim: "EvidenceClaim",
        snippets: list["Snippet"],
    ) -> list[Assertion]:
        """Extract atomic assertions from a claim."""
        assertions = []

        # Analyze claim text
        text_lower = claim.claim_text.lower()

        # Determine strength
        if any(re.search(p, text_lower) for p in self.STRONG_PATTERNS):
            strength = "strong"
        elif any(re.search(p, text_lower) for p in self.WEAK_PATTERNS):
            strength = "weak"
        else:
            strength = "uncertain"

        # Determine polarity
        if any(re.search(p, text_lower) for p in self.BENEFICIAL_PATTERNS):
            polarity = "beneficial"
        elif any(re.search(p, text_lower) for p in self.HARM_PATTERNS):
            polarity = "harm"
        else:
            polarity = "unclear"

        # Create main assertion
        assertion = Assertion(
            assertion_id=f"a_{claim.claim_id[:8]}",
            text=claim.claim_text,
            strength=strength,
            polarity=polarity,
            claim_id=claim.claim_id,
            supporting_snippet_ids=claim.supporting_snippet_ids,
            pico_components_matched={"P": False, "I": False, "C": False, "O": False},
        )

        assertions.append(assertion)
        return assertions

    def check_assertion_coverage(
        self,
        assertions: list[Assertion],
        snippets: list["Snippet"],
        pico: "PICO",
        records: list["Record"] | None = None,
    ) -> GateCheckResult:
        """
        Validate assertion coverage.

        Strong assertions must have at least 1 snippet with P/I/C/O match.

        P0-05: Also validates that supporting snippets come from study types
        allowed by PICO. If PICO=RCT, no systematic_review/meta_analysis
        can be supporting evidence for causal claims.
        """
        from cdr.core.enums import StudyType

        violations = []
        pico_gate = PICOMatchGate(strict=False)
        study_type_gate = StudyTypeEnforcementGate(strict=True)

        # P0-05: Check if PICO requires RCT only
        pico_requires_rct = pico.study_types and any(
            st in [StudyType.RCT] for st in pico.study_types
        )

        # Build record lookup for study type validation
        record_lookup: dict[str, "Record"] = {}
        if records:
            for r in records:
                record_lookup[r.record_id] = r

        for assertion in assertions:
            if assertion.strength == "strong":
                # Check each supporting snippet for PICO match
                has_full_match = False
                study_type_violations = []

                for sid in assertion.supporting_snippet_ids:
                    snippet = next((s for s in snippets if s.snippet_id == sid), None)
                    if snippet:
                        result = pico_gate.check_snippet(snippet, pico)
                        if result.passed:
                            has_full_match = True

                        # P0-05: Check study type of the record this snippet comes from
                        if pico_requires_rct and snippet.source_ref and snippet.source_ref.record_id:
                            record = record_lookup.get(snippet.source_ref.record_id)
                            if record:
                                type_result = study_type_gate.check_record(record, pico)
                                if type_result.failed:
                                    # Snippet comes from disallowed study type
                                    category = type_result.metadata.get("category", "unknown")
                                    specific = type_result.metadata.get("specific_type", "unknown")
                                    study_type_violations.append(
                                        GateViolation(
                                            gate_name=self.NAME,
                                            mismatch_type=MismatchType.STUDY_TYPE_MISMATCH,
                                            result=GateResult.FAIL,
                                            claim_id=assertion.claim_id,
                                            snippet_id=sid,
                                            record_id=snippet.source_ref.record_id,
                                            pmid=snippet.source_ref.pmid,
                                            message=(
                                                f"P0-05: PICO requires RCT but claim uses evidence from "
                                                f"{category}/{specific}. Systematic reviews and meta-analyses "
                                                f"cannot be primary supporting evidence for causal claims."
                                            ),
                                        )
                                    )

                if not has_full_match and assertion.supporting_snippet_ids:
                    violations.append(
                        GateViolation(
                            gate_name=self.NAME,
                            mismatch_type=MismatchType.ASSERTION_UNSUPPORTED,
                            result=GateResult.FAIL if self.strict else GateResult.WARN,
                            claim_id=assertion.claim_id,
                            message=f"Strong assertion lacks snippet with full PICO match",
                        )
                    )

                # P0-05: Add study type violations
                violations.extend(study_type_violations)

        if any(v.result == GateResult.FAIL for v in violations):
            result = GateResult.FAIL
        elif violations:
            result = GateResult.WARN
        else:
            result = GateResult.PASS

        return GateCheckResult(
            gate_name=self.NAME,
            result=result,
            violations=violations,
            metadata={
                "total_assertions": len(assertions),
                "strong_assertions": len([a for a in assertions if a.strength == "strong"]),
                "pico_requires_rct": pico_requires_rct,
            },
        )


# =============================================================================
# P0-03: EVIDENCE ANCHORING GATE
# =============================================================================


class EvidenceAnchoringGate:
    """
    P0-03: Evidence Anchoring Gate

    DoD3 BLOCKER: Claims about causal effects MUST be anchored to direct RCT evidence,
    not to secondary sources (systematic reviews, meta-analyses, subanalyses).

    This gate validates:
    1. Claims are supported by direct trial evidence (NCT-linked or primary publication)
    2. Claims are NOT solely anchored to sub-analyses or secondary pooled analyses
    3. When PICO requires RCT, supporting snippets come from actual RCT publications

    References:
    - GRADE Handbook: Direct vs indirect evidence
    - Cochrane Handbook: Primary vs secondary studies
    """

    NAME = "evidence_anchoring"

    # Patterns indicating secondary/indirect sources (NOT direct RCT)
    SECONDARY_SOURCE_PATTERNS = [
        r"meta.?analysis",
        r"systematic\s+review",
        r"pooled\s+analysis",
        r"umbrella\s+review",
        r"narrative\s+review",
    ]

    # Patterns indicating sub-analysis (not primary endpoint)
    SUBANALYSIS_PATTERNS = [
        r"sub.?analysis",
        r"subgroup\s+analysis",
        r"post.?hoc\s+analysis",
        r"exploratory\s+analysis",
        r"secondary\s+analysis",
        r"pre.?specified\s+subgroup",
    ]

    # Patterns indicating direct RCT (good anchors)
    DIRECT_RCT_PATTERNS = [
        r"randomized\s+(?:controlled\s+)?trial",
        r"randomised\s+(?:controlled\s+)?trial",
        r"NCT\d{8}",  # ClinicalTrials.gov identifier
        r"ISRCTN\d+",  # ISRCTN registry
        r"primary\s+(?:end\s*point|outcome)\s+(?:was|included)",
        r"intention.?to.?treat",
        r"ITT\s+(?:analysis|population)",
    ]

    def __init__(self, strict: bool = True):
        self.strict = strict

    def _is_direct_rct_source(self, text: str) -> tuple[bool, str]:
        """Check if text indicates direct RCT source."""
        text_lower = text.lower()

        # Check for direct RCT markers
        for pattern in self.DIRECT_RCT_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, "direct_rct"

        return False, "unknown"

    def _is_secondary_source(self, text: str) -> tuple[bool, str]:
        """Check if text indicates secondary source (SR/MA)."""
        text_lower = text.lower()

        for pattern in self.SECONDARY_SOURCE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, pattern.replace("\\s+", " ").replace(".?", "-")

        return False, ""

    def _is_subanalysis(self, text: str) -> tuple[bool, str]:
        """Check if text indicates subanalysis (not primary endpoint)."""
        text_lower = text.lower()

        for pattern in self.SUBANALYSIS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, pattern.replace("\\s+", " ").replace(".?", "-")

        return False, ""

    def check_claim_anchoring(
        self,
        claim: "EvidenceClaim",
        snippets: list["Snippet"],
        records: list["Record"],
        pico: "PICO",
    ) -> GateCheckResult:
        """
        Check that a claim is properly anchored to direct RCT evidence.

        A claim is properly anchored if:
        1. At least one supporting snippet comes from a direct RCT
        2. The claim is not solely based on secondary sources
        """
        from cdr.core.enums import StudyType

        violations = []

        # Get supporting snippets for this claim
        supporting_snippets = [
            s for s in snippets if s.snippet_id in claim.supporting_snippet_ids
        ]

        if not supporting_snippets:
            # No support = unsupported (handled elsewhere)
            return GateCheckResult(
                gate_name=self.NAME,
                result=GateResult.SKIP,
                metadata={"reason": "No supporting snippets to validate"},
            )

        # Build record lookup
        record_lookup = {r.record_id: r for r in records}

        # Analyze each supporting snippet
        has_direct_rct = False
        all_secondary = True
        all_subanalysis = True

        for snippet in supporting_snippets:
            # Get the record for this snippet
            record = record_lookup.get(
                snippet.source_ref.record_id if snippet.source_ref else None
            )

            # Check snippet text
            snippet_text = snippet.text
            is_direct, _ = self._is_direct_rct_source(snippet_text)
            is_secondary, secondary_type = self._is_secondary_source(snippet_text)
            is_subanalysis, subanalysis_type = self._is_subanalysis(snippet_text)

            # Also check record title/abstract if available
            if record:
                record_text = f"{record.title} {record.abstract or ''}"
                record_is_direct, _ = self._is_direct_rct_source(record_text)
                record_is_secondary, _ = self._is_secondary_source(record_text)

                is_direct = is_direct or record_is_direct
                is_secondary = is_secondary or record_is_secondary

            if is_direct:
                has_direct_rct = True
                all_secondary = False
                all_subanalysis = False
            elif not is_secondary:
                all_secondary = False
            if not is_subanalysis:
                all_subanalysis = False

        # Check if PICO requires RCT
        pico_requires_rct = pico.study_types and any(
            st in [StudyType.RCT] for st in pico.study_types
        )

        # Generate violations
        if pico_requires_rct and not has_direct_rct:
            result_type = GateResult.FAIL if self.strict else GateResult.WARN
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.EVIDENCE_NOT_DIRECT_RCT,
                    result=result_type,
                    claim_id=claim.claim_id,
                    message=(
                        "P0-03: PICO requires RCT but claim lacks direct RCT evidence. "
                        "Claims must be anchored to primary RCT publications, not only "
                        "systematic reviews, meta-analyses, or subanalyses."
                    ),
                )
            )

        if all_secondary and supporting_snippets:
            result_type = GateResult.FAIL if self.strict else GateResult.WARN
            violations.append(
                GateViolation(
                    gate_name=self.NAME,
                    mismatch_type=MismatchType.EVIDENCE_INDIRECT_SOURCE,
                    result=result_type,
                    claim_id=claim.claim_id,
                    message=(
                        "P0-03: Claim is solely anchored to secondary sources "
                        "(systematic reviews/meta-analyses). Direct RCT evidence required."
                    ),
                )
            )

        if any(v.result == GateResult.FAIL for v in violations):
            result = GateResult.FAIL
        elif violations:
            result = GateResult.WARN
        else:
            result = GateResult.PASS

        return GateCheckResult(
            gate_name=self.NAME,
            result=result,
            violations=violations,
            metadata={
                "has_direct_rct": has_direct_rct,
                "all_secondary": all_secondary,
                "all_subanalysis": all_subanalysis,
                "supporting_snippet_count": len(supporting_snippets),
            },
        )


# =============================================================================
# GATE REPORT GENERATOR
# =============================================================================


@dataclass
class GateReport:
    """Complete gate validation report for audit trail.

    P0-07: This is the SINGLE SOURCE OF TRUTH for status_reason.
    The status_reason here must be used consistently in JSON/PDF/UI.
    """

    run_id: str
    generated_at: datetime
    overall_status: str  # "publishable", "unpublishable", "partially_publishable"

    # P0-07: Single source of truth for status reason
    # This field is canonical and must be used in JSON/PDF/UI exports
    status_reason: str = ""
    status_reason_code: str = ""  # Machine-readable code (e.g., "dod3_gate_failures")

    # Gate results
    gate_results: dict[str, GateCheckResult] = field(default_factory=dict)

    # Aggregated counts
    total_checks: int = 0
    passed_checks: int = 0
    warned_checks: int = 0
    failed_checks: int = 0

    # Blocker violations (only support_integrity, not retrieval_quality)
    blocker_violations: list[GateViolation] = field(default_factory=list)

    # P0-02: Track retrieval quality issues separately (not blockers)
    retrieval_quality_violations: list[GateViolation] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Deduplicate blocker violations by record_id for accurate counting
        # A record failing multiple times (e.g., in multiple snippets) counts once
        unique_record_violations: dict[str, list[GateViolation]] = {}
        for v in self.blocker_violations:
            key = v.record_id or v.snippet_id or v.claim_id or "unknown"
            if key not in unique_record_violations:
                unique_record_violations[key] = []
            unique_record_violations[key].append(v)

        # Summary by unique record/entity
        blocker_summary = {
            "unique_records_failing": len(unique_record_violations),
            "total_violations": len(self.blocker_violations),
            "by_record": {
                record_id: {
                    "count": len(violations),
                    "types": list({v.mismatch_type.value for v in violations if v.mismatch_type}),
                }
                for record_id, violations in unique_record_violations.items()
            },
        }

        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at.isoformat(),
            "overall_status": self.overall_status,
            # P0-07: Single source of truth for status_reason
            "status_reason": self.status_reason,
            "status_reason_code": self.status_reason_code,
            "summary": {
                "total_checks": self.total_checks,
                "passed": self.passed_checks,
                "warned": self.warned_checks,
                "failed": self.failed_checks,
                # Deduplicated blocker count for accurate publishability assessment
                "unique_blocker_records": len(unique_record_violations),
                # P0-02: Separate counters for support_integrity vs retrieval_quality
                "support_integrity_blockers": len(self.blocker_violations),
                "retrieval_quality_warnings": len(self.retrieval_quality_violations),
            },
            "blocker_summary": blocker_summary,  # CRITICAL: Deduplicated view for CI/harness
            "gate_results": {k: v.to_dict() for k, v in self.gate_results.items()},
            "blocker_violations": [
                v.to_dict() for v in self.blocker_violations
            ],  # Full detail retained
            "retrieval_quality_violations": [
                v.to_dict() for v in self.retrieval_quality_violations
            ],  # P0-02: Noise that was correctly filtered
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        """Generate Markdown report for PDF."""
        lines = [
            "# Gate Report",
            "",
            f"**Run ID:** {self.run_id}",
            f"**Generated:** {self.generated_at.isoformat()}",
            f"**Status:** {self.overall_status.upper()}",
            "",
            "## Summary",
            "",
            f"- Total checks: {self.total_checks}",
            f"- Passed: {self.passed_checks}",
            f"- Warnings: {self.warned_checks}",
            f"- Failed: {self.failed_checks}",
            "",
        ]

        if self.blocker_violations:
            lines.extend(
                [
                    "## Blocker Violations",
                    "",
                ]
            )
            for v in self.blocker_violations[:10]:  # Top 10
                lines.append(f"- **{v.gate_name}**: {v.message}")
                if v.pmid:
                    lines.append(f"  - PMID: {v.pmid}")
                if v.claim_id:
                    lines.append(f"  - Claim: {v.claim_id[:8]}")
            lines.append("")

        if self.recommendations:
            lines.extend(
                [
                    "## Recommendations",
                    "",
                ]
            )
            for r in self.recommendations:
                lines.append(f"- {r}")
            lines.append("")

        return "\n".join(lines)


class GateReportGenerator:
    """Generate comprehensive gate validation reports.

    CRITICAL FIX (Fix 2): Separates retrieval quality (noise) from support integrity.

    - Violations in INCLUDED evidence (post-enforcement) → BLOCKER
    - Violations in EXCLUDED evidence (filtered by enforcer) → WARNING

    This prevents penalizing retrieval noise that was already correctly filtered.
    """

    def __init__(self):
        self.pico_gate = PICOMatchGate(strict=True)
        self.study_type_gate = StudyTypeEnforcementGate(strict=True)
        self.context_gate = ContextPurityGate(strict=True)
        self.assertion_gate = AssertionCoverageGate(strict=True)
        # P0-03: Evidence anchoring gate
        self.anchoring_gate = EvidenceAnchoringGate(strict=True)

    def validate_run(
        self,
        run_id: str,
        pico: "PICO",
        records: list["Record"],
        snippets: list["Snippet"],
        claims: list["EvidenceClaim"],
        *,  # keyword-only after this
        included_record_ids: set[str] | None = None,
        included_snippet_ids: set[str] | None = None,
    ) -> GateReport:
        """Run all gates and generate report.

        Args:
            run_id: Unique run identifier
            pico: PICO specification for validation
            records: All records (included + excluded)
            snippets: All snippets (included + excluded)
            claims: Claims to validate
            included_record_ids: Record IDs that passed enforcement (still in final answer).
                                If None, all records are considered included (legacy behavior).
            included_snippet_ids: Snippet IDs that passed enforcement (still in final answer).
                                 If None, all snippets are considered included (legacy behavior).

        Returns:
            GateReport with proper separation of support_integrity vs retrieval_quality violations
        """
        # If no enforcement context provided, all evidence is "included" (legacy behavior)
        if included_record_ids is None:
            included_record_ids = {r.record_id for r in records}
        if included_snippet_ids is None:
            included_snippet_ids = {s.snippet_id for s in snippets}

        report = GateReport(
            run_id=run_id,
            generated_at=datetime.utcnow(),
            overall_status="publishable",
        )

        # Track violations by category
        # CRITICAL: Only support_integrity_violations are BLOCKERS
        support_integrity_violations = []  # Violations in INCLUDED evidence → BLOCKER
        retrieval_quality_violations = []  # Violations in EXCLUDED evidence → WARNING
        all_violations = []  # Combined for legacy compatibility

        def classify_violation(v: GateViolation) -> None:
            """Classify violation as support_integrity or retrieval_quality.

            Support integrity = violation in evidence that IS used in final answer
            Retrieval quality = violation in evidence that was FILTERED OUT

            Only support_integrity violations block publishability.
            """
            all_violations.append(v)

            # Check if this violation is from included or excluded evidence
            is_included = False
            if v.record_id and v.record_id in included_record_ids:
                is_included = True
            elif v.snippet_id and v.snippet_id in included_snippet_ids:
                is_included = True
            elif not v.record_id and not v.snippet_id:
                # Claim-level violations (no specific record/snippet)
                # These are always support_integrity if claims exist
                is_included = True

            if is_included:
                support_integrity_violations.append(v)
            else:
                # Downgrade to WARNING level for retrieval quality issues
                retrieval_quality_violations.append(v)

        # 1. PICO Match Gate on records
        pico_violations_list = []  # Actual violation objects
        pico_passes = 0
        for record in records:
            result = self.pico_gate.check_record(record, pico)
            report.total_checks += 1
            if result.failed:
                report.failed_checks += 1
                for v in result.violations:
                    pico_violations_list.append(v)
                    classify_violation(v)
            elif result.warned:
                report.warned_checks += 1
            else:
                pico_passes += 1
                report.passed_checks += 1

        report.gate_results["pico_match_records"] = GateCheckResult(
            gate_name="pico_match_records",
            result=GateResult.FAIL if pico_violations_list else GateResult.PASS,
            violations=pico_violations_list,  # CRITICAL: Include actual violations for schema consistency
            metadata={"violation_count": len(pico_violations_list), "passes": pico_passes},
        )

        # 2. Study Type Gate
        type_violations_list = []  # Actual violation objects
        type_passes = 0
        for record in records:
            result = self.study_type_gate.check_record(record, pico)
            report.total_checks += 1
            if result.failed:
                report.failed_checks += 1
                for v in result.violations:
                    type_violations_list.append(v)
                    classify_violation(v)
            elif result.warned:
                report.warned_checks += 1
            else:
                type_passes += 1
                report.passed_checks += 1

        report.gate_results["study_type_enforcement"] = GateCheckResult(
            gate_name="study_type_enforcement",
            result=GateResult.FAIL if type_violations_list else GateResult.PASS,
            violations=type_violations_list,  # CRITICAL: Include actual violations
            metadata={"violation_count": len(type_violations_list), "passes": type_passes},
        )

        # 3. PICO Match Gate on snippets
        snippet_violations_list = []  # Actual violation objects
        for snippet in snippets:
            result = self.pico_gate.check_snippet(snippet, pico)
            report.total_checks += 1
            if result.failed:
                report.failed_checks += 1
                for v in result.violations:
                    snippet_violations_list.append(v)
                    classify_violation(v)
            elif result.warned:
                report.warned_checks += 1
            else:
                report.passed_checks += 1

        report.gate_results["pico_match_snippets"] = GateCheckResult(
            gate_name="pico_match_snippets",
            result=GateResult.FAIL if snippet_violations_list else GateResult.PASS,
            violations=snippet_violations_list,  # CRITICAL: Include actual violations
            metadata={"violation_count": len(snippet_violations_list)},
        )

        # 4. Context Purity Gate on claims
        context_violations_list = []  # Actual violation objects
        for claim in claims:
            result = self.context_gate.check_claim_purity(claim, snippets)
            report.total_checks += 1
            if result.failed:
                report.failed_checks += 1
                for v in result.violations:
                    context_violations_list.append(v)
                    classify_violation(v)
            elif result.warned:
                report.warned_checks += 1
            else:
                report.passed_checks += 1

        report.gate_results["context_purity"] = GateCheckResult(
            gate_name="context_purity",
            result=GateResult.FAIL if context_violations_list else GateResult.PASS,
            violations=context_violations_list,  # CRITICAL: Include actual violations
            metadata={"violation_count": len(context_violations_list)},
        )

        # 5. Assertion Coverage Gate (P0-05: now includes study type validation)
        all_assertions = []
        for claim in claims:
            assertions = self.assertion_gate.extract_assertions(claim, snippets)
            all_assertions.extend(assertions)

        if all_assertions:
            # P0-05: Pass records to check study types of supporting snippets
            assertion_result = self.assertion_gate.check_assertion_coverage(
                all_assertions, snippets, pico, records=records
            )
            report.total_checks += 1
            if assertion_result.failed:
                report.failed_checks += 1
                for v in assertion_result.violations:
                    classify_violation(v)
            elif assertion_result.warned:
                report.warned_checks += 1
            else:
                report.passed_checks += 1

            report.gate_results["assertion_coverage"] = assertion_result

        # 6. P0-03: Evidence Anchoring Gate - verify claims use direct RCT evidence
        anchoring_violations_list = []
        for claim in claims:
            result = self.anchoring_gate.check_claim_anchoring(
                claim, snippets, records, pico
            )
            report.total_checks += 1
            if result.failed:
                report.failed_checks += 1
                for v in result.violations:
                    anchoring_violations_list.append(v)
                    classify_violation(v)
            elif result.warned:
                report.warned_checks += 1
            else:
                report.passed_checks += 1

        report.gate_results["evidence_anchoring"] = GateCheckResult(
            gate_name="evidence_anchoring",
            result=GateResult.FAIL if anchoring_violations_list else GateResult.PASS,
            violations=anchoring_violations_list,
            metadata={"violation_count": len(anchoring_violations_list)},
        )

        # =====================================================================
        # CRITICAL FIX 2: Separate blocker (support_integrity) from warning (retrieval_quality)
        # =====================================================================

        # Only support_integrity violations (from INCLUDED evidence) are blockers
        # Retrieval quality issues (from EXCLUDED evidence) are warnings, not blockers
        report.blocker_violations = [
            v for v in support_integrity_violations if v.result == GateResult.FAIL
        ]

        # Store retrieval quality issues for audit trail (but don't block)
        retrieval_warnings = [
            v for v in retrieval_quality_violations if v.result == GateResult.FAIL
        ]

        # Add metadata to distinguish violation types
        report.gate_results["_violation_classification"] = GateCheckResult(
            gate_name="violation_classification",
            result=GateResult.PASS if not report.blocker_violations else GateResult.FAIL,
            violations=[],  # Summary only
            metadata={
                "support_integrity_blockers": len(report.blocker_violations),
                "retrieval_quality_warnings": len(retrieval_warnings),
                "total_violations": len(all_violations),
                "explanation": (
                    "support_integrity = violations in INCLUDED evidence (blocks publishability). "
                    "retrieval_quality = violations in EXCLUDED evidence (noise, already filtered)."
                ),
            },
        )

        # =====================================================================
        # P0-02 & P0-07: Set status and reason based ONLY on support_integrity violations
        # =====================================================================
        # Only support_integrity violations (from INCLUDED evidence) are blockers
        report.blocker_violations = [
            v for v in support_integrity_violations if v.result == GateResult.FAIL
        ]

        # Store retrieval quality issues for audit trail (but don't block)
        report.retrieval_quality_violations = [
            v for v in retrieval_quality_violations if v.result == GateResult.FAIL
        ]

        # P0-07: Generate canonical status_reason (single source of truth)
        if report.blocker_violations:
            report.overall_status = "unpublishable"

            # Count unique blocker records
            unique_blocker_records = set()
            for v in report.blocker_violations:
                if v.record_id:
                    unique_blocker_records.add(v.record_id)
                elif v.snippet_id:
                    unique_blocker_records.add(v.snippet_id)

            # Generate machine-readable code and human-readable reason
            report.status_reason_code = "dod3_support_integrity_failures"
            report.status_reason = (
                f"{len(report.blocker_violations)} support integrity violations in "
                f"{len(unique_blocker_records)} evidence records used by claims. "
                f"({len(report.retrieval_quality_violations)} additional retrieval noise violations "
                f"were correctly filtered and did not affect status.)"
            )

            # Generate recommendations based on violation types
            violation_types = {v.mismatch_type for v in report.blocker_violations}

            if MismatchType.POPULATION_EXCLUDED in violation_types:
                report.recommendations.append(
                    "Review claim→snippet mapping - claims are citing evidence that excludes target population"
                )

            if MismatchType.POPULATION_NOT_MENTIONED in violation_types:
                report.recommendations.append(
                    "Claims are using evidence that doesn't mention the target population"
                )

            if MismatchType.COMPARATOR_INDIRECT in violation_types:
                report.recommendations.append(
                    "Create sub-PICOs for different comparators found in evidence"
                )

            if MismatchType.STUDY_TYPE_MISMATCH in violation_types:
                report.recommendations.append(
                    "Claims cite evidence from study types not allowed by PICO (e.g., systematic review when RCT required)"
                )
        else:
            report.overall_status = "publishable"
            report.status_reason_code = "all_gates_passed"
            report.status_reason = (
                f"All {report.passed_checks} evidence checks passed. "
                f"{len(report.retrieval_quality_violations)} retrieval noise issues were "
                f"correctly filtered and did not affect publishability."
            )

        return report


# =============================================================================
# MASTER DOD3 VALIDATOR
# =============================================================================


@dataclass
class DoD3ValidationResult:
    """Complete DoD3 validation result."""

    passed: bool
    gate_report: GateReport
    excluded_records: list[str]  # record_ids excluded by gates
    excluded_snippets: list[str]  # snippet_ids excluded by gates
    degraded_claims: list[str]  # claim_ids that were degraded

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "gate_report": self.gate_report.to_dict(),
            "excluded_records": self.excluded_records,
            "excluded_snippets": self.excluded_snippets,
            "degraded_claims": self.degraded_claims,
        }


class DoD3Validator:
    """
    Master validator implementing full DoD3 contract.

    Usage:
        validator = DoD3Validator()
        result = validator.validate(run_id, pico, records, snippets, claims)

        if not result.passed:
            # Run is unpublishable
            # Use result.gate_report for audit trail
    """

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.pico_gate = PICOMatchGate(strict=strict)
        self.study_type_gate = StudyTypeEnforcementGate(strict=strict)
        self.context_gate = ContextPurityGate(strict=strict)
        self.assertion_gate = AssertionCoverageGate(strict=strict)
        self.report_generator = GateReportGenerator()

    def validate_record(self, record: "Record", pico: "PICO") -> tuple[bool, list[GateViolation]]:
        """Validate a single record against PICO."""
        violations = []

        # PICO match
        pico_result = self.pico_gate.check_record(record, pico)
        if pico_result.failed:
            violations.extend(pico_result.violations)

        # Study type
        type_result = self.study_type_gate.check_record(record, pico)
        if type_result.failed:
            violations.extend(type_result.violations)

        return len(violations) == 0, violations

    def validate_snippet(
        self, snippet: "Snippet", pico: "PICO"
    ) -> tuple[bool, list[GateViolation]]:
        """Validate a single snippet against PICO."""
        result = self.pico_gate.check_snippet(snippet, pico)
        return result.passed, result.violations

    def validate(
        self,
        run_id: str,
        pico: "PICO",
        records: list["Record"],
        snippets: list["Snippet"],
        claims: list["EvidenceClaim"],
    ) -> DoD3ValidationResult:
        """Run full DoD3 validation.

        CRITICAL FIX (P0-02): We now properly separate retrieval noise from support integrity:
        1. First, identify which records/snippets FAIL validation (excluded)
        2. Then, calculate which ones PASS (included)
        3. Pass included_record_ids/included_snippet_ids to validate_run()
        4. GateReportGenerator will classify violations:
           - Violations in INCLUDED evidence → BLOCKER (support_integrity)
           - Violations in EXCLUDED evidence → WARNING (retrieval_quality)

        This prevents penalizing the run for retrieval noise that was correctly filtered.
        """
        excluded_records = []
        excluded_snippets = []
        degraded_claims = []

        # =========================================================================
        # STEP 1: Validate records and identify exclusions
        # =========================================================================
        for record in records:
            passed, violations = self.validate_record(record, pico)
            if not passed:
                excluded_records.append(record.record_id)

        # =========================================================================
        # STEP 2: Validate snippets and identify exclusions
        # =========================================================================
        for snippet in snippets:
            passed, violations = self.validate_snippet(snippet, pico)
            if not passed:
                excluded_snippets.append(snippet.snippet_id)

        # =========================================================================
        # STEP 3: Calculate INCLUDED evidence (what PASSES validation)
        # These are the records/snippets that can legitimately support claims
        # =========================================================================
        all_record_ids = {r.record_id for r in records}
        all_snippet_ids = {s.snippet_id for s in snippets}
        excluded_record_set = set(excluded_records)
        excluded_snippet_set = set(excluded_snippets)

        # INCLUDED = records/snippets that PASS validation (not excluded)
        included_record_ids = all_record_ids - excluded_record_set
        included_snippet_ids = all_snippet_ids - excluded_snippet_set

        # =========================================================================
        # STEP 4: Check which claims actually USE excluded evidence
        # If a claim's supporting_snippet_ids includes excluded snippets,
        # those snippets create SUPPORT INTEGRITY violations (blockers)
        # =========================================================================
        claims_using_excluded_evidence = set()
        for claim in claims:
            excluded_support = [
                sid for sid in claim.supporting_snippet_ids if sid in excluded_snippet_set
            ]
            if excluded_support:
                degraded_claims.append(claim.claim_id)
                claims_using_excluded_evidence.add(claim.claim_id)
                # CRITICAL: If a claim USES excluded snippets, those snippets
                # must be counted as support_integrity violations
                # So we need to "include" them in the validation context
                for sid in excluded_support:
                    included_snippet_ids.add(sid)  # Force these to be counted as blockers

        # =========================================================================
        # STEP 5: Generate gate report with proper violation classification
        # =========================================================================
        gate_report = self.report_generator.validate_run(
            run_id=run_id,
            pico=pico,
            records=records,
            snippets=snippets,
            claims=claims,
            # CRITICAL: Pass inclusion context for proper violation classification
            included_record_ids=included_record_ids,
            included_snippet_ids=included_snippet_ids,
        )

        return DoD3ValidationResult(
            passed=gate_report.overall_status == "publishable",
            gate_report=gate_report,
            excluded_records=excluded_records,
            excluded_snippets=excluded_snippets,
            degraded_claims=degraded_claims,
        )
