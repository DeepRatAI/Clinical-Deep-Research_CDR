"""
CDR Core Enumerations

This module defines all enumerations used throughout the CDR system.
These are critical for maintaining type safety and consistent vocabulary.
"""

from enum import Enum, auto


class StudyType(str, Enum):
    """Classification of study designs."""

    RCT = "rct"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    COHORT = "cohort"
    CASE_CONTROL = "case_control"
    CROSS_SECTIONAL = "cross_sectional"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"
    GUIDELINE = "guideline"
    NARRATIVE_REVIEW = "narrative_review"
    EDITORIAL = "editorial"
    LETTER = "letter"
    OTHER = "other"
    UNKNOWN = "unknown"


class RecordSource(str, Enum):
    """Source of retrieved records."""

    PUBMED = "pubmed"
    CLINICAL_TRIALS = "clinical_trials"
    PMC = "pmc"
    LOCAL = "local"
    PREPRINT = "preprint"
    GUIDELINE = "guideline"
    OTHER = "other"


class ExclusionReason(str, Enum):
    """Standardized reasons for excluding records during screening."""

    # Hard filters
    LANGUAGE_NOT_SUPPORTED = "language_not_supported"
    YEAR_OUT_OF_RANGE = "year_out_of_range"
    STUDY_TYPE_EXCLUDED = "study_type_excluded"
    NO_ABSTRACT = "no_abstract"

    # PICO-related
    POPULATION_MISMATCH = "population_mismatch"
    INTERVENTION_MISMATCH = "intervention_mismatch"
    OUTCOME_MISMATCH = "outcome_mismatch"
    PICO_MISMATCH = "pico_mismatch"

    # Evidence gate failures (deterministic validation)
    # Refs: PRISMA 2020, GRADE indirectness domain
    POPULATION_EXCLUDED = "population_excluded"  # Population in exclusion criteria
    POPULATION_NOT_IN_SCOPE = "population_not_in_scope"  # Population not mentioned
    COMPARATOR_INDIRECT = "comparator_indirect"  # Wrong comparator context
    STUDY_TYPE_MISMATCH = "study_type_mismatch"  # Design doesn't match PICO

    # Quality/relevance
    LOW_RELEVANCE_SCORE = "low_relevance_score"
    DUPLICATE = "duplicate"
    RETRACTED = "retracted"

    # Content issues
    NO_ORIGINAL_DATA = "no_original_data"
    ANIMAL_STUDY = "animal_study"
    IN_VITRO_ONLY = "in_vitro_only"

    # Other
    MANUAL_EXCLUSION = "manual_exclusion"
    OTHER = "other"


class RoB2Domain(str, Enum):
    """Risk of Bias 2.0 domains for RCT assessment.

    Refs: https://methods.cochrane.org/bias/resources/rob-2-revised-cochrane-risk-bias-tool-randomized-trials
    """

    RANDOMIZATION = "randomization_process"
    DEVIATIONS = "deviations_from_intended_interventions"
    MISSING_DATA = "missing_outcome_data"
    MEASUREMENT = "measurement_of_outcome"
    SELECTION = "selection_of_reported_result"


class ROBINSIDomain(str, Enum):
    """ROBINS-I domains for non-randomized studies of interventions.

    HIGH-3 fix: ROBINS-I for observational studies.
    Refs: https://methods.cochrane.org/bias/resources/robins-i-tool
    """

    CONFOUNDING = "bias_due_to_confounding"
    SELECTION = "bias_in_selection_of_participants"
    CLASSIFICATION = "bias_in_classification_of_interventions"
    DEVIATIONS = "bias_due_to_deviations_from_intended_interventions"
    MISSING_DATA = "bias_due_to_missing_data"
    MEASUREMENT = "bias_in_measurement_of_outcomes"
    SELECTION_REPORTED = "bias_in_selection_of_reported_result"


class RoB2Judgment(str, Enum):
    """Risk of Bias 2.0 judgment levels."""

    LOW = "low"
    SOME_CONCERNS = "some_concerns"
    HIGH = "high"


class ROBINSIJudgment(str, Enum):
    """ROBINS-I judgment levels for non-randomized studies.

    HIGH-3 fix: Different judgment levels for observational studies.
    Refs: https://methods.cochrane.org/bias/resources/robins-i-tool
    """

    LOW = "low"
    MODERATE = "moderate"
    SERIOUS = "serious"
    CRITICAL = "critical"
    NO_INFORMATION = "no_information"


class GRADECertainty(str, Enum):
    """GRADE certainty of evidence levels."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class VerificationStatus(str, Enum):
    """Status of verification checks.

    Unified taxonomy used across verifier, publisher, metrics, and reports.

    - VERIFIED: Source fully entails the claim (high confidence)
    - PARTIAL: Source provides some support but incomplete
    - UNVERIFIABLE: Cannot verify (source unavailable, ambiguous)
    - CONTRADICTED: Source contradicts the claim
    - ERROR: Verification process failed (technical error)
    """

    VERIFIED = "verified"
    PARTIAL = "partial"
    UNVERIFIABLE = "unverifiable"
    CONTRADICTED = "contradicted"
    ERROR = "error"


class RunStatus(str, Enum):
    """Status of a CDR run.

    CRITICAL: These statuses have distinct semantic meanings:

    Technical statuses:
    - PENDING: Run not yet started
    - RUNNING: Run in progress
    - WAITING_HITL: Waiting for human-in-the-loop
    - COMPLETED: Run finished without crashes (technical success)
    - FAILED: Run crashed or had unrecoverable error
    - CANCELLED: Run was cancelled

    Scientific statuses (results may be honest but insufficient):
    - INSUFFICIENT_EVIDENCE: No evidence found or evidence below threshold
    - UNPUBLISHABLE: Evidence found but doesn't meet publication standards
                     (e.g., claims without snippets, RoB2 invalid)

    A run can be COMPLETED technically but INSUFFICIENT_EVIDENCE scientifically.
    This is honest and correct - not a failure.
    """

    PENDING = "pending"
    RUNNING = "running"
    WAITING_HITL = "waiting_hitl"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # Scientific outcome statuses (distinct from technical success)
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNPUBLISHABLE = "unpublishable"
    # FIX 7: Partial publishability for mixed evidence scenarios
    # Some sub-PICOs have valid evidence, others don't
    PARTIALLY_PUBLISHABLE = "partially_publishable"


class GraphNode(str, Enum):
    """Names of nodes in the CDR graph.

    CRITICAL: These MUST match exactly the nodes defined in orchestration/graph.py.
    Any mismatch breaks the workflow contract.
    """

    PARSE_QUESTION = "parse_question"
    PLAN_SEARCH = "plan_search"
    RETRIEVE = "retrieve"
    DEDUPLICATE = "deduplicate"
    SCREEN = "screen"
    PARSE_DOCS = "parse_docs"
    EXTRACT_DATA = "extract_data"
    ASSESS_ROB2 = "assess_rob2"
    SYNTHESIZE = "synthesize"
    COMPOSE = "compose"  # HIGH-1: Compositional inference (A+B⇒C)
    CRITIQUE = "critique"
    VERIFY = "verify"
    PUBLISH = "publish"


class CritiqueDimension(str, Enum):
    """Dimensions evaluated by the Skeptic agent."""

    INTERNAL_VALIDITY = "internal_validity"
    EXTERNAL_VALIDITY = "external_validity"
    STATISTICAL_ISSUES = "statistical_issues"
    MISSING_EVIDENCE = "missing_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    OVERSTATEMENT = "overstatement"
    CONFOUNDERS = "confounders"
    SEARCH_BIAS = "search_bias"


class CritiqueSeverity(str, Enum):
    """Severity levels for critique findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Section(str, Enum):
    """Standard scientific paper sections (IMRAD + extras)."""

    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    SUPPLEMENTARY = "supplementary"
    FULL_TEXT = "full_text"  # For PMC full-text retrieval (HIGH-2)
    UNKNOWN = "unknown"


class OutcomeMeasureType(str, Enum):
    """Types of statistical outcome measures."""

    RISK_RATIO = "RR"
    ODDS_RATIO = "OR"
    HAZARD_RATIO = "HR"
    MEAN_DIFFERENCE = "MD"
    STANDARDIZED_MEAN_DIFFERENCE = "SMD"
    RISK_DIFFERENCE = "RD"
    NUMBER_NEEDED_TO_TREAT = "NNT"
    CORRELATION = "r"
    PERCENTAGE = "percent"
    COUNT = "count"
    OTHER = "other"


class ComparatorSource(str, Enum):
    """Source of comparator inference in PICO.

    Per PRISMA 2020 and ICTRP standards, comparators must be explicitly tracked:
    - user_specified: User explicitly stated the comparator in the question
    - assumed_from_question: Inferred from efficacy question (e.g., "vs placebo/no treatment")
    - inferred_from_evidence: Extracted from trial registry metadata or study arms
    - heuristic: Detected via NLP keywords (lowest confidence)

    Refs: WHO ICTRP data set, ClinicalTrials.gov arm types, PRISMA 2020
    """

    USER_SPECIFIED = "user_specified"
    ASSUMED_FROM_QUESTION = "assumed_from_question"
    INFERRED_FROM_EVIDENCE = "inferred_from_evidence"
    HEURISTIC = "heuristic"
    NOT_APPLICABLE = "not_applicable"  # For non-comparative questions


class TherapeuticContext(str, Enum):
    """Therapeutic context for evidence claims.

    Prevents mixing incompatible therapeutic scenarios in the same claim.
    Per GRADE handbook: claims must be specific to therapeutic context.

    Refs: GRADE Handbook Section 5, Cochrane Handbook Section 11
    """

    # Monotherapy contexts
    MONOTHERAPY = "monotherapy"  # Drug alone vs placebo/no treatment
    MONOTHERAPY_VS_ACTIVE = "monotherapy_vs_active"  # Drug A vs Drug B

    # Combination/add-on contexts
    ADD_ON = "add_on"  # Drug added to existing therapy vs existing therapy alone
    COMBINATION = "combination"  # Multiple drugs together

    # Specific common patterns
    ASPIRIN_MONOTHERAPY = "aspirin_monotherapy"
    ASPIRIN_PLUS_ANTICOAGULANT = "aspirin_plus_anticoagulant"
    DOAC_VS_ASPIRIN = "doac_vs_aspirin"
    DOAC_VS_WARFARIN = "doac_vs_warfarin"

    # Head-to-head and other
    HEAD_TO_HEAD = "head_to_head"  # Active comparator, superiority/non-inferiority
    PREVENTION = "prevention"  # Primary/secondary prevention context
    TREATMENT = "treatment"  # Acute treatment context

    # Fallback
    UNCLASSIFIED = "unclassified"  # Could not determine context
