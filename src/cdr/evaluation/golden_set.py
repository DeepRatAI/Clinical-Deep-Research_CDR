"""
CDR Golden Set - Benchmark Clinical Questions

5 clinical questions with expected outcomes for evaluation.
Based on well-established medical knowledge for reproducible testing.

Each question includes:
    - PICO components
    - Expected evidence level
    - Expected claim patterns
    - Minimum quality thresholds

References:
    - capacidad.md: Hypothesis examples from real medical discoveries
    - CDR Architecture: Quality gates and DoD levels
"""

from dataclasses import dataclass, field
from enum import Enum


class ExpectedEvidenceLevel(str, Enum):
    """Expected quality of available evidence."""

    HIGH = "high"  # Multiple RCTs, clear answer
    MODERATE = "moderate"  # Mix of RCTs and observational
    LOW = "low"  # Mostly observational, indirect
    VERY_LOW = "very_low"  # Case reports, expert opinion
    NONE = "none"  # No evidence expected


@dataclass
class GoldenSetQuestion:
    """A benchmark question with expected outcomes."""

    id: str
    question: str
    population: str
    intervention: str
    comparator: str
    outcome: str

    # Expected outcomes
    expected_evidence_level: ExpectedEvidenceLevel = ExpectedEvidenceLevel.MODERATE
    expected_min_studies: int = 3  # Minimum studies to find
    expected_claim_patterns: list[str] = field(default_factory=list)  # Key terms expected in claims

    # Quality thresholds for this question
    min_verification_coverage: float = 0.8
    min_snippet_coverage: float = 1.0

    # Whether composition (A+B⇒C) is expected
    composition_expected: bool = False
    composition_hint: str = ""  # Hint about what composition might look like


# Golden Set Questions
GOLDEN_SET: list[GoldenSetQuestion] = [
    # Question 1: Well-established treatment question (HIGH evidence)
    GoldenSetQuestion(
        id="GS-001",
        question="Is aspirin effective for secondary prevention of cardiovascular events?",
        population="Adults with established cardiovascular disease",
        intervention="Low-dose aspirin (75-100mg daily)",
        comparator="Placebo or no treatment",
        outcome="Major adverse cardiovascular events (MACE)",
        expected_evidence_level=ExpectedEvidenceLevel.HIGH,
        expected_min_studies=10,
        expected_claim_patterns=[
            "aspirin",
            "cardiovascular",
            "secondary prevention",
            "MACE",
            "risk reduction",
        ],
        min_verification_coverage=0.9,
        composition_expected=False,
    ),
    # Question 2: Combination therapy (MODERATE evidence, composition expected)
    GoldenSetQuestion(
        id="GS-002",
        question="Does metformin combined with GLP-1 agonists improve outcomes beyond monotherapy?",
        population="Adults with type 2 diabetes inadequately controlled on metformin",
        intervention="Metformin plus GLP-1 receptor agonist",
        comparator="Metformin monotherapy",
        outcome="HbA1c reduction and cardiovascular events",
        expected_evidence_level=ExpectedEvidenceLevel.MODERATE,
        expected_min_studies=5,
        expected_claim_patterns=[
            "metformin",
            "GLP-1",
            "HbA1c",
            "glycemic control",
            "cardiovascular",
        ],
        composition_expected=True,
        composition_hint="GLP-1 effects + metformin mechanisms may suggest synergy for CV protection",
    ),
    # Question 3: Negative/uncertain evidence (LOW evidence)
    GoldenSetQuestion(
        id="GS-003",
        question="Is vitamin D supplementation effective for preventing respiratory infections?",
        population="General adult population",
        intervention="Vitamin D supplementation (any dose)",
        comparator="Placebo",
        outcome="Incidence of acute respiratory tract infections",
        expected_evidence_level=ExpectedEvidenceLevel.LOW,
        expected_min_studies=5,
        expected_claim_patterns=[
            "vitamin D",
            "respiratory",
            "infection",
            "inconsistent",
            "heterogeneity",
        ],
        min_verification_coverage=0.7,  # Lower threshold for uncertain evidence
        composition_expected=False,
    ),
    # Question 4: Diagnostic question (MODERATE evidence)
    GoldenSetQuestion(
        id="GS-004",
        question="What is the accuracy of high-sensitivity troponin for diagnosing acute MI?",
        population="Adults presenting with chest pain",
        intervention="High-sensitivity cardiac troponin T or I",
        comparator="Standard troponin or clinical diagnosis",
        outcome="Diagnostic accuracy (sensitivity, specificity) for acute MI",
        expected_evidence_level=ExpectedEvidenceLevel.MODERATE,
        expected_min_studies=8,
        expected_claim_patterns=[
            "troponin",
            "sensitivity",
            "specificity",
            "acute myocardial infarction",
            "NPV",
        ],
        composition_expected=False,
    ),
    # Question 5: Emerging/compositional question (MODERATE, composition expected)
    GoldenSetQuestion(
        id="GS-005",
        question="Can anti-inflammatory therapies reduce cardiovascular events in patients with elevated CRP?",
        population="Patients with established CVD and elevated hs-CRP (>2mg/L)",
        intervention="Anti-inflammatory agents (colchicine, IL-1β inhibitors)",
        comparator="Standard therapy without anti-inflammatory",
        outcome="Major adverse cardiovascular events",
        expected_evidence_level=ExpectedEvidenceLevel.MODERATE,
        expected_min_studies=3,
        expected_claim_patterns=[
            "inflammation",
            "CRP",
            "cardiovascular",
            "colchicine",
            "canakinumab",
            "IL-1",
        ],
        composition_expected=True,
        composition_hint="Inflammation → atherosclerosis + anti-inflammatory → reduced CRP may compose to CV benefit",
    ),
]


def get_golden_set() -> list[GoldenSetQuestion]:
    """Return the golden set of benchmark questions."""
    return GOLDEN_SET


def get_question_by_id(question_id: str) -> GoldenSetQuestion | None:
    """Get a specific golden set question by ID."""
    for q in GOLDEN_SET:
        if q.id == question_id:
            return q
    return None


def validate_against_golden_set(
    question_id: str,
    claims_count: int,
    snippets_count: int,
    verification_coverage: float,
    studies_found: int,
    hypotheses_count: int = 0,
) -> dict[str, bool | str]:
    """
    Validate run results against golden set expectations.

    Args:
        question_id: Golden set question ID
        claims_count: Number of claims generated
        snippets_count: Number of snippets retrieved
        verification_coverage: Verification coverage percentage
        studies_found: Number of studies found
        hypotheses_count: Number of hypotheses composed

    Returns:
        Dictionary with validation results
    """
    question = get_question_by_id(question_id)
    if not question:
        return {"valid": False, "error": f"Unknown question ID: {question_id}"}

    results = {
        "valid": True,
        "question_id": question_id,
        "checks": {},
    }

    # Check minimum studies
    min_studies_ok = studies_found >= question.expected_min_studies
    results["checks"]["min_studies"] = {
        "expected": question.expected_min_studies,
        "actual": studies_found,
        "passed": min_studies_ok,
    }
    if not min_studies_ok:
        results["valid"] = False

    # Check verification coverage
    verification_ok = verification_coverage >= question.min_verification_coverage
    results["checks"]["verification_coverage"] = {
        "expected": question.min_verification_coverage,
        "actual": verification_coverage,
        "passed": verification_ok,
    }
    if not verification_ok:
        results["valid"] = False

    # Check composition if expected
    if question.composition_expected:
        composition_ok = hypotheses_count > 0
        results["checks"]["composition"] = {
            "expected": True,
            "actual": hypotheses_count > 0,
            "hypotheses_count": hypotheses_count,
            "passed": composition_ok,
        }
        # Note: composition is a soft expectation, not a hard failure

    return results
