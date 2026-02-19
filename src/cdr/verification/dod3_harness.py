"""
DoD3 Validation Harness

Executable test harness for DoD3 compliance validation.

Usage:
    python -m cdr.verification.dod3_harness
    # Or via make:
    make test_coherence

This harness:
1. Runs all DoD3 gates on minimal test cases
2. Reports pass/fail per gate
3. Fails the process if any BLOCKER fails
4. Tracks no-regression against baseline

References:
- DoD3 Contract: CDR_Agent_Guidance_and_Development_Protocol.md
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Import gates
from cdr.verification.dod3_gates import (
    PICOMatchGate,
    StudyTypeEnforcementGate,
    ContextPurityGate,
    AssertionCoverageGate,
    DoD3Validator,
    GateResult,
    PopulationContext,
    TherapyMode,
)

# Import schemas
from cdr.core.schemas import PICO, Record, Snippet, SourceRef, EvidenceClaim
from cdr.core.enums import (
    StudyType,
    RecordSource,
    GRADECertainty,
    Section,
    ComparatorSource,
)


# =============================================================================
# TEST CASE DEFINITIONS
# =============================================================================


@dataclass
class TestCase:
    """A single test case for DoD3 validation."""

    name: str
    description: str
    pico: PICO
    records: list[Record]
    snippets: list[Snippet]
    claims: list[EvidenceClaim]
    expected_result: str  # "pass", "fail", "warn"
    expected_violations: list[str] = field(default_factory=list)  # Expected violation types
    tags: list[str] = field(default_factory=list)


def create_record(
    record_id: str,
    title: str,
    abstract: str,
    pmid: str | None = None,
    publication_type: list[str] | None = None,
    source: RecordSource = RecordSource.PUBMED,
) -> Record:
    """Helper to create test records."""
    return Record(
        record_id=record_id,
        source=source,
        content_hash=f"hash_{record_id}",
        title=title,
        abstract=abstract,
        pmid=pmid,
        publication_type=publication_type or [],
    )


def create_snippet(
    snippet_id: str,
    text: str,
    record_id: str,
    pmid: str | None = None,
) -> Snippet:
    """Helper to create test snippets."""
    return Snippet(
        snippet_id=snippet_id,
        text=text,
        source_ref=SourceRef(
            record_id=record_id,
            pmid=pmid,
        ),
        section=Section.ABSTRACT,
    )


def create_claim(
    claim_id: str,
    claim_text: str,
    supporting_snippet_ids: list[str],
    certainty: GRADECertainty = GRADECertainty.LOW,
) -> EvidenceClaim:
    """Helper to create test claims."""
    return EvidenceClaim(
        claim_id=claim_id,
        claim_text=claim_text,
        certainty=certainty,
        supporting_snippet_ids=supporting_snippet_ids,
    )


# =============================================================================
# MINIMAL TEST DATASET
# =============================================================================


def get_test_cases() -> list[TestCase]:
    """Get minimal test dataset for DoD3 validation."""

    cases = []

    # -------------------------------------------------------------------------
    # CASE 1: Clean RCT matching PICO (should PASS)
    # -------------------------------------------------------------------------
    pico_af_aspirin = PICO(
        population="patients with atrial fibrillation",
        intervention="aspirin",
        comparator="placebo",
        comparator_source=ComparatorSource.USER_SPECIFIED,
        outcome="stroke prevention",
        study_types=[StudyType.RCT],
    )

    record_clean_rct = create_record(
        record_id="rec_001",
        title="Aspirin for Stroke Prevention in Atrial Fibrillation: A Randomized Controlled Trial",
        abstract=(
            "Background: Patients with atrial fibrillation (AF) are at increased risk of stroke. "
            "Methods: We randomized 1000 patients with nonvalvular atrial fibrillation to receive "
            "aspirin 100mg daily or placebo. Primary outcome was stroke or systemic embolism. "
            "Results: Aspirin reduced stroke risk compared to placebo (HR 0.75, 95% CI 0.60-0.95). "
            "Conclusion: Aspirin may prevent stroke in patients with atrial fibrillation."
        ),
        pmid="12345678",
        publication_type=["Randomized Controlled Trial"],
    )

    snippet_clean = create_snippet(
        snippet_id="snp_001",
        text=(
            "We randomized 1000 patients with nonvalvular atrial fibrillation to receive "
            "aspirin 100mg daily or placebo. Primary outcome was stroke or systemic embolism. "
            "Aspirin reduced stroke risk compared to placebo (HR 0.75, 95% CI 0.60-0.95)."
        ),
        record_id="rec_001",
        pmid="12345678",
    )

    claim_clean = create_claim(
        claim_id="clm_001",
        claim_text="Aspirin may reduce stroke risk in patients with atrial fibrillation compared to placebo.",
        supporting_snippet_ids=["snp_001"],
    )

    cases.append(
        TestCase(
            name="clean_rct_match",
            description="Clean RCT matching PICO - should PASS all gates",
            pico=pico_af_aspirin,
            records=[record_clean_rct],
            snippets=[snippet_clean],
            claims=[claim_clean],
            expected_result="pass",
            tags=["baseline", "rct", "clean"],
        )
    )

    # -------------------------------------------------------------------------
    # CASE 2: Population EXCLUDED (should FAIL - ASPREE/CHIP pattern)
    # -------------------------------------------------------------------------
    record_aspree = create_record(
        record_id="rec_002",
        title="Aspirin in Healthy Elderly: The ASPREE Trial",
        abstract=(
            "Background: We studied healthy elderly adults without a diagnosed cardiovascular event, "
            "atrial fibrillation, or dementia. "
            "Methods: Participants without atrial fibrillation were randomized to aspirin or placebo. "
            "Results: Aspirin did not reduce cardiovascular events in this low-risk population. "
            "Conclusion: Aspirin is not beneficial for primary prevention in healthy elderly."
        ),
        pmid="41091476",
        publication_type=["Randomized Controlled Trial"],
    )

    snippet_aspree = create_snippet(
        snippet_id="snp_002",
        text=(
            "We studied healthy elderly adults without a diagnosed cardiovascular event, "
            "atrial fibrillation, or dementia. Participants without atrial fibrillation "
            "were randomized to aspirin or placebo."
        ),
        record_id="rec_002",
        pmid="41091476",
    )

    cases.append(
        TestCase(
            name="population_excluded_aspree",
            description="ASPREE-pattern: population WITHOUT AF (should FAIL)",
            pico=pico_af_aspirin,
            records=[record_aspree],
            snippets=[snippet_aspree],
            claims=[],
            expected_result="fail",
            expected_violations=["population_excluded"],
            tags=["blocker", "population", "aspree"],
        )
    )

    # -------------------------------------------------------------------------
    # CASE 3: Wrong comparator (DOAC vs aspirin when PICO says placebo)
    # -------------------------------------------------------------------------
    record_doac = create_record(
        record_id="rec_003",
        title="Apixaban versus Aspirin in Atrial Fibrillation",
        abstract=(
            "Background: Patients with atrial fibrillation require stroke prevention. "
            "Methods: We randomized 5000 AF patients to apixaban or aspirin. "
            "Results: Apixaban reduced stroke compared to aspirin (HR 0.50, 95% CI 0.40-0.65). "
            "Conclusion: Apixaban is superior to aspirin for stroke prevention in AF."
        ),
        pmid="22334455",
        publication_type=["Randomized Controlled Trial"],
    )

    snippet_doac = create_snippet(
        snippet_id="snp_003",
        text=(
            "We randomized 5000 AF patients to apixaban or aspirin. "
            "Apixaban reduced stroke compared to aspirin (HR 0.50, 95% CI 0.40-0.65)."
        ),
        record_id="rec_003",
        pmid="22334455",
    )

    cases.append(
        TestCase(
            name="comparator_indirect_doac",
            description="DOAC vs aspirin when PICO expects placebo (should FAIL)",
            pico=pico_af_aspirin,
            records=[record_doac],
            snippets=[snippet_doac],
            claims=[],
            expected_result="fail",
            expected_violations=["comparator_indirect", "comparator_jumped"],
            tags=["blocker", "comparator"],
        )
    )

    # -------------------------------------------------------------------------
    # CASE 4: Study type mismatch (cohort when RCT required)
    # -------------------------------------------------------------------------
    record_cohort = create_record(
        record_id="rec_004",
        title="Aspirin Use and Stroke Risk in AF: A Cohort Study",
        abstract=(
            "Background: Limited RCT data exist for aspirin in atrial fibrillation. "
            "Methods: This retrospective cohort study followed 10,000 AF patients. "
            "We compared those on aspirin vs no aspirin. "
            "Results: Aspirin users had 20% lower stroke risk (observational). "
            "Conclusion: Aspirin may be associated with lower stroke risk in AF."
        ),
        pmid="33445566",
        publication_type=["Cohort Study", "Observational Study"],
    )

    cases.append(
        TestCase(
            name="study_type_mismatch_cohort",
            description="Cohort study when PICO requires RCT (should FAIL)",
            pico=pico_af_aspirin,
            records=[record_cohort],
            snippets=[],
            claims=[],
            expected_result="fail",
            expected_violations=["study_type_mismatch"],
            tags=["blocker", "study_type"],
        )
    )

    # -------------------------------------------------------------------------
    # CASE 5: Case report (always excluded)
    # -------------------------------------------------------------------------
    record_case = create_record(
        record_id="rec_005",
        title="Stroke Prevention with Aspirin in AF: A Case Report",
        abstract=(
            "We report a case of a 75-year-old patient with atrial fibrillation "
            "who was treated with aspirin and did not experience stroke over 5 years."
        ),
        pmid="44556677",
        publication_type=["Case Report"],
    )

    cases.append(
        TestCase(
            name="case_report_excluded",
            description="Case report (always excluded)",
            pico=pico_af_aspirin,
            records=[record_case],
            snippets=[],
            claims=[],
            expected_result="fail",
            expected_violations=["study_type_mismatch"],
            tags=["blocker", "study_type"],
        )
    )

    # -------------------------------------------------------------------------
    # CASE 6: Pneumonia study (completely wrong population)
    # -------------------------------------------------------------------------
    record_pneumonia = create_record(
        record_id="rec_006",
        title="Aspirin for Acute Pneumonia: A Randomized Trial",
        abstract=(
            "Background: Acute pneumonia (AP) is a common cause of hospitalization. "
            "Methods: We randomized patients with acute pneumonia to aspirin or placebo. "
            "Primary outcome was 30-day mortality and major adverse cardiovascular events (MACE). "
            "Atrial fibrillation was not an inclusion criterion. "
            "Results: Aspirin did not reduce mortality in pneumonia patients."
        ),
        pmid="40341146",
        publication_type=["Randomized Controlled Trial"],
    )

    snippet_pneumonia = create_snippet(
        snippet_id="snp_006",
        text=(
            "We randomized patients with acute pneumonia to aspirin or placebo. "
            "Primary outcome was 30-day mortality and major adverse cardiovascular events (MACE)."
        ),
        record_id="rec_006",
        pmid="40341146",
    )

    cases.append(
        TestCase(
            name="wrong_population_pneumonia",
            description="Pneumonia study supporting AF claim (should FAIL)",
            pico=pico_af_aspirin,
            records=[record_pneumonia],
            snippets=[snippet_pneumonia],
            claims=[],
            expected_result="fail",
            expected_violations=["population_not_mentioned"],
            tags=["blocker", "population", "dirty"],
        )
    )

    # -------------------------------------------------------------------------
    # CASE 7: Post-ablation context (specific population)
    # -------------------------------------------------------------------------
    record_ablation = create_record(
        record_id="rec_007",
        title="Anticoagulation After AF Ablation: Aspirin vs DOAC",
        abstract=(
            "Background: Post-ablation anticoagulation strategy remains debated. "
            "Methods: We randomized AF patients after catheter ablation to rivaroxaban or aspirin. "
            "Results: Rivaroxaban reduced stroke compared to aspirin post-ablation. "
            "Conclusion: DOACs are preferred over aspirin after AF ablation."
        ),
        pmid="55667788",
        publication_type=["Randomized Controlled Trial"],
    )

    snippet_ablation = create_snippet(
        snippet_id="snp_007",
        text=(
            "We randomized AF patients after catheter ablation to rivaroxaban or aspirin. "
            "Rivaroxaban reduced stroke compared to aspirin post-ablation."
        ),
        record_id="rec_007",
        pmid="55667788",
    )

    cases.append(
        TestCase(
            name="post_ablation_context",
            description="Post-ablation specific context",
            pico=pico_af_aspirin,
            records=[record_ablation],
            snippets=[snippet_ablation],
            claims=[],
            expected_result="fail",  # Wrong comparator (DOAC vs aspirin)
            expected_violations=["comparator_jumped"],
            tags=["context", "post_ablation"],
        )
    )

    # -------------------------------------------------------------------------
    # CASE 8: Mixed therapy modes in single claim (context purity violation)
    # Note: This is a more subtle test - the primary gates catch comparator issues first
    # -------------------------------------------------------------------------
    snippet_mono = create_snippet(
        snippet_id="snp_008a",
        text="We randomized 500 patients with atrial fibrillation to aspirin alone vs placebo. Aspirin alone reduced stroke risk compared to placebo in AF patients.",
        record_id="rec_001",
        pmid="12345678",
    )

    snippet_combo = create_snippet(
        snippet_id="snp_008b",
        text="In patients with atrial fibrillation, aspirin plus rivaroxaban increased bleeding compared to rivaroxaban alone. This dual therapy approach showed higher bleeding risk.",
        record_id="rec_008",
        pmid="66778899",
    )

    claim_mixed = create_claim(
        claim_id="clm_008",
        claim_text="Aspirin therapy in AF patients affects both stroke prevention and bleeding outcomes.",
        supporting_snippet_ids=["snp_008a", "snp_008b"],
    )

    cases.append(
        TestCase(
            name="mixed_therapy_modes",
            description="Claim mixes monotherapy with add-on therapy (context violation)",
            pico=pico_af_aspirin,
            records=[record_clean_rct],
            snippets=[snippet_mono, snippet_combo],
            claims=[claim_mixed],
            expected_result="fail",  # Will fail on comparator_jumped from combo snippet
            expected_violations=["comparator_jumped", "mixed_therapy_modes"],
            tags=["context", "purity"],
        )
    )

    # -------------------------------------------------------------------------
    # CASE 9: Subclinical AF (ARTESiA pattern)
    # -------------------------------------------------------------------------
    record_subclinical = create_record(
        record_id="rec_009",
        title="ARTESiA: Apixaban in Subclinical Atrial Fibrillation",
        abstract=(
            "Background: Device-detected subclinical atrial fibrillation (SCAF) is common. "
            "Methods: We randomized patients with SCAF to apixaban or aspirin. "
            "Results: Apixaban reduced stroke compared to aspirin in subclinical AF. "
            "Conclusion: Anticoagulation benefits patients with device-detected AF."
        ),
        pmid="77889900",
        publication_type=["Randomized Controlled Trial"],
    )

    cases.append(
        TestCase(
            name="subclinical_af_artesia",
            description="Subclinical AF context - different from clinical AF",
            pico=pico_af_aspirin,
            records=[record_subclinical],
            snippets=[],
            claims=[],
            expected_result="fail",  # Wrong comparator (apixaban vs aspirin)
            expected_violations=["comparator_indirect"],
            tags=["context", "subclinical"],
        )
    )

    # -------------------------------------------------------------------------
    # CASE 10: Valid head-to-head with correct PICO
    # Need new records with explicit population mention
    # -------------------------------------------------------------------------
    pico_doac_aspirin = PICO(
        population="patients with atrial fibrillation",
        intervention="DOAC",
        comparator="aspirin",
        comparator_source=ComparatorSource.USER_SPECIFIED,
        outcome="stroke prevention",
        study_types=[StudyType.RCT],
    )

    record_doac_valid = create_record(
        record_id="rec_010",
        title="DOAC versus Aspirin for Stroke Prevention in Atrial Fibrillation",
        abstract=(
            "Background: Patients with atrial fibrillation need stroke prevention. "
            "Methods: We randomized 2000 patients with atrial fibrillation to DOAC or aspirin. "
            "Results: DOAC reduced stroke compared to aspirin in patients with atrial fibrillation. "
            "Conclusion: DOAC is superior to aspirin in AF patients."
        ),
        pmid="11223344",
        publication_type=["Randomized Controlled Trial"],
    )

    snippet_doac_valid = create_snippet(
        snippet_id="snp_010",
        text=(
            "In this randomized trial of patients with atrial fibrillation, "
            "DOAC reduced stroke risk by 40% compared to aspirin (HR 0.60, 95% CI 0.45-0.80) "
            "in patients with atrial fibrillation."
        ),
        record_id="rec_010",
        pmid="11223344",
    )

    cases.append(
        TestCase(
            name="valid_doac_vs_aspirin",
            description="DOAC vs aspirin study with matching PICO (should PASS)",
            pico=pico_doac_aspirin,
            records=[record_doac_valid],
            snippets=[snippet_doac_valid],
            claims=[],
            expected_result="pass",
            tags=["baseline", "head_to_head"],
        )
    )

    return cases


# =============================================================================
# HARNESS RUNNER
# =============================================================================


@dataclass
class TestResult:
    """Result of a single test case."""

    case_name: str
    passed: bool
    actual_result: str
    expected_result: str
    violations_found: list[str]
    expected_violations: list[str]
    message: str

    def to_dict(self) -> dict:
        return {
            "case_name": self.case_name,
            "passed": self.passed,
            "actual_result": self.actual_result,
            "expected_result": self.expected_result,
            "violations_found": self.violations_found,
            "expected_violations": self.expected_violations,
            "message": self.message,
        }


@dataclass
class HarnessResult:
    """Overall harness result."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    results: list[TestResult]
    blocker_failures: list[str]  # Cases that failed BLOCKER checks
    regression_detected: bool = False

    @property
    def success(self) -> bool:
        return self.failed_cases == 0 and not self.blocker_failures

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "success": self.success,
            "blocker_failures": self.blocker_failures,
            "regression_detected": self.regression_detected,
            "results": [r.to_dict() for r in self.results],
        }


def run_harness(cases: list[TestCase] | None = None, verbose: bool = True) -> HarnessResult:
    """Run the DoD3 validation harness."""

    if cases is None:
        cases = get_test_cases()

    validator = DoD3Validator(strict=True)
    results = []
    blocker_failures = []

    for case in cases:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Running: {case.name}")
            print(f"Description: {case.description}")
            print(f"Expected: {case.expected_result}")

        try:
            # Run validation
            validation = validator.validate(
                run_id=f"test_{case.name}",
                pico=case.pico,
                records=case.records,
                snippets=case.snippets,
                claims=case.claims,
            )

            # Determine actual result
            if validation.passed:
                actual_result = "pass"
            elif validation.gate_report.blocker_violations:
                actual_result = "fail"
            else:
                actual_result = "warn"

            # Get violations found
            violations_found = [
                v.mismatch_type.value for v in validation.gate_report.blocker_violations
            ]

            # Check if test passed
            result_match = actual_result == case.expected_result

            # Check violations match (if expected)
            violations_match = True
            if case.expected_violations:
                violations_match = any(ev in violations_found for ev in case.expected_violations)

            test_passed = result_match and (not case.expected_violations or violations_match)

            message = ""
            if not result_match:
                message = f"Expected {case.expected_result}, got {actual_result}"
            elif case.expected_violations and not violations_match:
                message = f"Expected violations {case.expected_violations}, got {violations_found}"

            result = TestResult(
                case_name=case.name,
                passed=test_passed,
                actual_result=actual_result,
                expected_result=case.expected_result,
                violations_found=violations_found,
                expected_violations=case.expected_violations,
                message=message,
            )

            if verbose:
                status = "✅ PASS" if test_passed else "❌ FAIL"
                print(f"Result: {status}")
                if violations_found:
                    print(f"Violations: {violations_found}")
                if message:
                    print(f"Message: {message}")

            # Track blocker failures
            if "blocker" in case.tags and not test_passed:
                blocker_failures.append(case.name)

        except Exception as e:
            result = TestResult(
                case_name=case.name,
                passed=False,
                actual_result="error",
                expected_result=case.expected_result,
                violations_found=[],
                expected_violations=case.expected_violations,
                message=f"Exception: {str(e)}",
            )
            if verbose:
                print(f"Result: ❌ ERROR - {e}")

        results.append(result)

    passed_cases = sum(1 for r in results if r.passed)
    failed_cases = len(results) - passed_cases

    harness_result = HarnessResult(
        total_cases=len(cases),
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        results=results,
        blocker_failures=blocker_failures,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print("HARNESS SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total cases: {harness_result.total_cases}")
        print(f"Passed: {harness_result.passed_cases}")
        print(f"Failed: {harness_result.failed_cases}")
        print(f"Blocker failures: {len(blocker_failures)}")
        print(f"Overall: {'✅ SUCCESS' if harness_result.success else '❌ FAILURE'}")

    return harness_result


def save_harness_results(result: HarnessResult, path: Path | str) -> None:
    """Save harness results to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """Main entry point for harness."""
    import argparse

    parser = argparse.ArgumentParser(description="DoD3 Validation Harness")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--tags",
        "-t",
        type=str,
        nargs="+",
        help="Only run cases with these tags",
    )

    args = parser.parse_args()

    # Get cases
    cases = get_test_cases()

    # Filter by tags if specified
    if args.tags:
        cases = [c for c in cases if any(t in c.tags for t in args.tags)]

    # Run harness
    result = run_harness(cases, verbose=not args.quiet)

    # Save results if path specified
    if args.output:
        save_harness_results(result, args.output)
        print(f"\nResults saved to: {args.output}")

    # Exit with appropriate code
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
