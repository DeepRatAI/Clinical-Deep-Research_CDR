"""
Publishable Harness - P0-09

This module defines "anchored" queries that MUST result in PUBLISHABLE status.
These are positive controls for the DoD3 gate system.

If any of these queries fail to produce PUBLISHABLE, it indicates a systematic
problem in the pipeline that must be fixed.

Two types of controls:
1. ID-Anchored: Uses specific NCT/PMID identifiers (tests retrieval + validation)
2. Free-Text: Uses natural language with trial names (tests ranking + validation)

Refs:
- DoD3 Contract: "At least 2 publishable + 2 unpublishable controls"
- PRISMA 2020: Reproducible search strategies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ExpectedStatus(str, Enum):
    """Expected status for a harness query."""

    PUBLISHABLE = "publishable"
    PARTIALLY_PUBLISHABLE = "partially_publishable"
    UNPUBLISHABLE = "unpublishable"


@dataclass
class HarnessQuery:
    """A single query in the publishable harness."""

    query_id: str
    name: str
    description: str

    # The actual query text
    question: str

    # Expected outcome
    expected_status: ExpectedStatus
    expected_blockers: int  # Expected number of support_integrity_blockers

    # Validation criteria
    min_claims: int = 1
    min_verification_coverage: float = 0.0  # 0.0 = no requirement

    # Anchoring identifiers (optional - for ID-anchored queries)
    anchor_pmids: list[str] = field(default_factory=list)
    anchor_nct_ids: list[str] = field(default_factory=list)

    # Trial metadata for validation
    trial_name: str | None = None
    trial_intervention: str | None = None
    trial_comparator: str | None = None
    trial_primary_outcome: str | None = None

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "name": self.name,
            "description": self.description,
            "question": self.question,
            "expected_status": self.expected_status.value,
            "expected_blockers": self.expected_blockers,
            "min_claims": self.min_claims,
            "anchor_pmids": self.anchor_pmids,
            "anchor_nct_ids": self.anchor_nct_ids,
            "trial_name": self.trial_name,
        }


@dataclass
class HarnessResult:
    """Result of running a single harness query."""

    query_id: str
    actual_status: str
    actual_blockers: int
    actual_claims: int
    actual_verification_coverage: float

    # Pass/fail determination
    status_passed: bool
    blockers_passed: bool
    claims_passed: bool

    # Overall
    passed: bool

    # Details for debugging
    run_id: str | None = None
    gate_report_summary: dict = field(default_factory=dict)
    status_reason: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "passed": self.passed,
            "actual_status": self.actual_status,
            "actual_blockers": self.actual_blockers,
            "actual_claims": self.actual_claims,
            "status_passed": self.status_passed,
            "blockers_passed": self.blockers_passed,
            "claims_passed": self.claims_passed,
            "run_id": self.run_id,
            "status_reason": self.status_reason,
            "errors": self.errors,
        }


# =============================================================================
# PUBLISHABLE CONTROLS (MUST PASS)
# =============================================================================

# These queries are designed to retrieve well-known, high-quality RCT evidence
# that should definitely pass DoD3 gates.

PUBLISHABLE_CONTROLS = [
    # =========================================================================
    # PC-01: JUPITER Trial (ID-Anchored)
    # =========================================================================
    # JUPITER (PMID: 18997196, NCT00239681) is a landmark RCT showing
    # rosuvastatin reduces cardiovascular events in patients with elevated CRP.
    HarnessQuery(
        query_id="PC-01",
        name="JUPITER (ID-Anchored)",
        description=(
            "Rosuvastatin vs placebo for cardiovascular events in elevated CRP. "
            "Anchored to specific PMID/NCT to ensure direct evidence retrieval."
        ),
        question=(
            "What is the effect of rosuvastatin 20mg versus placebo on major cardiovascular "
            "events in adults with elevated high-sensitivity C-reactive protein (hs-CRP ≥ 2mg/L) "
            "and LDL cholesterol below 130 mg/dL? "
            "Reference: JUPITER trial (NCT00239681, PMID 18997196)."
        ),
        expected_status=ExpectedStatus.PUBLISHABLE,
        expected_blockers=0,
        min_claims=1,
        min_verification_coverage=0.5,
        anchor_pmids=["18997196"],
        anchor_nct_ids=["NCT00239681"],
        trial_name="JUPITER",
        trial_intervention="rosuvastatin 20mg",
        trial_comparator="placebo",
        trial_primary_outcome="major cardiovascular events",
    ),
    # =========================================================================
    # PC-02: EMPEROR-Preserved (ID-Anchored)
    # =========================================================================
    # EMPEROR-Preserved (PMID: 34449189, NCT03057951) showed empagliflozin
    # reduces hospitalization for heart failure in HFpEF patients.
    HarnessQuery(
        query_id="PC-02",
        name="EMPEROR-Preserved (ID-Anchored)",
        description=(
            "Empagliflozin vs placebo for heart failure hospitalization in HFpEF. "
            "Anchored to specific PMID/NCT."
        ),
        question=(
            "What is the effect of empagliflozin 10mg versus placebo on hospitalization "
            "for heart failure in patients with heart failure with preserved ejection fraction "
            "(HFpEF, LVEF > 40%)? "
            "Reference: EMPEROR-Preserved trial (NCT03057951, PMID 34449189)."
        ),
        expected_status=ExpectedStatus.PUBLISHABLE,
        expected_blockers=0,
        min_claims=1,
        min_verification_coverage=0.5,
        anchor_pmids=["34449189"],
        anchor_nct_ids=["NCT03057951"],
        trial_name="EMPEROR-Preserved",
        trial_intervention="empagliflozin 10mg",
        trial_comparator="placebo",
        trial_primary_outcome="hospitalization for heart failure",
    ),
    # =========================================================================
    # PC-03: DAPA-HF (Free-Text)
    # =========================================================================
    # DAPA-HF is a well-known trial - this tests free-text retrieval ranking.
    HarnessQuery(
        query_id="PC-03",
        name="DAPA-HF (Free-Text)",
        description=(
            "Dapagliflozin vs placebo for heart failure outcomes. "
            "Free-text query to test retrieval ranking without explicit anchors."
        ),
        question=(
            "What is the effect of dapagliflozin 10mg versus placebo on cardiovascular death "
            "or worsening heart failure in patients with heart failure with reduced ejection "
            "fraction (HFrEF)? Focus on DAPA-HF trial evidence."
        ),
        expected_status=ExpectedStatus.PUBLISHABLE,
        expected_blockers=0,
        min_claims=1,
        anchor_pmids=["31535829"],  # DAPA-HF primary paper
        anchor_nct_ids=["NCT03036124"],
        trial_name="DAPA-HF",
        trial_intervention="dapagliflozin 10mg",
        trial_comparator="placebo",
        trial_primary_outcome="cardiovascular death or worsening heart failure",
    ),
]

# =============================================================================
# UNPUBLISHABLE CONTROLS (MUST FAIL)
# =============================================================================

# These queries are designed to trigger specific gate failures.

UNPUBLISHABLE_CONTROLS = [
    # =========================================================================
    # UC-01: ASPREE (Population Mismatch)
    # =========================================================================
    # ASPREE studied healthy elderly WITHOUT cardiovascular disease.
    # Querying for "with atrial fibrillation" should fail PICO population gate.
    HarnessQuery(
        query_id="UC-01",
        name="ASPREE Population Mismatch",
        description=(
            "Query for AF population against ASPREE which excludes AF. "
            "Should fail population match gate."
        ),
        question=(
            "What is the effect of aspirin versus placebo on stroke prevention "
            "in elderly patients with atrial fibrillation? "
            "Reference: ASPREE trial."
        ),
        expected_status=ExpectedStatus.UNPUBLISHABLE,
        expected_blockers=1,  # At least 1 population mismatch blocker
        min_claims=0,
        trial_name="ASPREE",
    ),
    # =========================================================================
    # UC-02: Observational Evidence for Therapeutic Question
    # =========================================================================
    # Query asking for RCT evidence but likely to retrieve observational data.
    HarnessQuery(
        query_id="UC-02",
        name="Study Type Mismatch",
        description=(
            "Therapeutic question requiring RCT evidence where observational "
            "studies dominate. Should fail study type gate."
        ),
        question=(
            "What is the effect of low-dose aspirin versus no aspirin on "
            "cardiovascular events in patients with peripheral artery disease "
            "based on randomized controlled trial evidence only?"
        ),
        expected_status=ExpectedStatus.UNPUBLISHABLE,
        expected_blockers=1,
        min_claims=0,
    ),
]

# =============================================================================
# BORDERLINE CONTROLS (PARTIALLY PUBLISHABLE)
# =============================================================================

BORDERLINE_CONTROLS = [
    # =========================================================================
    # BC-01: Mixed Evidence Quality
    # =========================================================================
    HarnessQuery(
        query_id="BC-01",
        name="Mixed Evidence (Partial)",
        description=(
            "Query with some RCT evidence and some observational. "
            "Should be partially publishable with stratification."
        ),
        question=(
            "What is the effect of SGLT2 inhibitors versus placebo on "
            "cardiovascular outcomes in patients with type 2 diabetes? "
            "Consider both RCT and real-world evidence."
        ),
        expected_status=ExpectedStatus.PARTIALLY_PUBLISHABLE,
        expected_blockers=0,  # Blockers only in excluded evidence
        min_claims=1,
    ),
]


# =============================================================================
# HARNESS RUNNER
# =============================================================================


def evaluate_harness_result(
    query: HarnessQuery,
    run_status: str,
    blocker_count: int,
    claim_count: int,
    verification_coverage: float,
    run_id: str | None = None,
    status_reason: str = "",
    gate_report_summary: dict | None = None,
) -> HarnessResult:
    """
    Evaluate a single harness result against expected criteria.

    Args:
        query: The harness query specification
        run_status: Actual status from the run (e.g., "publishable", "unpublishable")
        blocker_count: Number of support_integrity_blockers
        claim_count: Number of claims generated
        verification_coverage: Fraction of claims verified
        run_id: Optional run ID for reference
        status_reason: Status reason from gate_report
        gate_report_summary: Summary from gate_report

    Returns:
        HarnessResult with pass/fail determination
    """
    # Normalize status
    run_status_normalized = run_status.lower().replace("_", "")
    expected_normalized = query.expected_status.value.lower().replace("_", "")

    # Check individual criteria
    status_passed = run_status_normalized == expected_normalized

    # For publishable: blockers MUST be 0
    # For unpublishable: blockers MUST be > 0
    if query.expected_status == ExpectedStatus.PUBLISHABLE:
        blockers_passed = blocker_count == 0
    elif query.expected_status == ExpectedStatus.UNPUBLISHABLE:
        blockers_passed = blocker_count >= query.expected_blockers
    else:  # PARTIALLY_PUBLISHABLE
        blockers_passed = True  # More flexible for partial

    claims_passed = claim_count >= query.min_claims

    # Overall pass
    passed = status_passed and blockers_passed and claims_passed

    return HarnessResult(
        query_id=query.query_id,
        actual_status=run_status,
        actual_blockers=blocker_count,
        actual_claims=claim_count,
        actual_verification_coverage=verification_coverage,
        status_passed=status_passed,
        blockers_passed=blockers_passed,
        claims_passed=claims_passed,
        passed=passed,
        run_id=run_id,
        gate_report_summary=gate_report_summary or {},
        status_reason=status_reason,
        errors=[],
    )


def get_all_harness_queries() -> list[HarnessQuery]:
    """Get all harness queries for CI/testing."""
    return PUBLISHABLE_CONTROLS + UNPUBLISHABLE_CONTROLS + BORDERLINE_CONTROLS


def get_publishable_queries() -> list[HarnessQuery]:
    """Get only publishable control queries."""
    return PUBLISHABLE_CONTROLS


def get_unpublishable_queries() -> list[HarnessQuery]:
    """Get only unpublishable control queries."""
    return UNPUBLISHABLE_CONTROLS
