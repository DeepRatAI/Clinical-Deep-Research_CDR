#!/usr/bin/env python3
"""
CDR Semantic Coherence Evaluation Harness

Structural validation of semantic coherence without running full pipeline.
Tests pass/fail based on structure, not prose quality.

DoD Criteria:
- PICO.comparator_source populated correctly
- Therapeutic context tags on claims
- No invalid leaps in conclusions
- Context purity (no mixing)

Usage:
    python -m cdr.evaluation.semantic_harness
    pytest tests/evaluation/test_semantic_harness.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from cdr.core.enums import ComparatorSource, TherapeuticContext, GRADECertainty
from cdr.core.schemas import PICO, EvidenceClaim
from cdr.verification.assertion_gate import AssertionGate, AssertionGateResult


class CheckResult(str, Enum):
    """Result of an individual check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class HarnessCheck:
    """A single check result."""

    name: str
    result: CheckResult
    expected: str
    actual: str
    detail: str = ""


@dataclass
class TestCaseResult:
    """Result of running a test case."""

    case_id: str
    case_name: str
    passed: bool
    checks: list[HarnessCheck] = field(default_factory=list)
    assertion_gate_result: AssertionGateResult | None = None

    @property
    def failed_checks(self) -> list[HarnessCheck]:
        return [c for c in self.checks if c.result == CheckResult.FAIL]

    @property
    def warning_checks(self) -> list[HarnessCheck]:
        return [c for c in self.checks if c.result == CheckResult.WARN]

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "case_name": self.case_name,
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name,
                    "result": c.result.value,
                    "expected": c.expected,
                    "actual": c.actual,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "assertion_gate": self.assertion_gate_result.to_dict()
            if self.assertion_gate_result
            else None,
        }


@dataclass
class HarnessResult:
    """Overall harness execution result."""

    passed: bool
    total_cases: int
    passed_cases: int
    failed_cases: int
    case_results: list[TestCaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "summary": f"{self.passed_cases}/{self.total_cases} passed",
            "cases": [r.to_dict() for r in self.case_results],
        }

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("CDR SEMANTIC COHERENCE HARNESS - RESULTS")
        print("=" * 60)
        print(f"Overall: {'PASS ✓' if self.passed else 'FAIL ✗'}")
        print(f"Cases: {self.passed_cases}/{self.total_cases} passed")
        print()

        for result in self.case_results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  [{status}] {result.case_id}: {result.case_name}")
            if not result.passed:
                for check in result.failed_checks:
                    print(
                        f"       ├─ {check.name}: expected '{check.expected}', got '{check.actual}'"
                    )
            for check in result.warning_checks:
                print(f"       ├─ ⚠ {check.name}: {check.detail}")

        print("=" * 60)


class SemanticCoherenceHarness:
    """
    Harness for testing semantic coherence of CDR outputs.

    Tests structural properties, not prose quality.
    """

    def __init__(self, strict_assertion_gate: bool = True):
        """Initialize harness."""
        self.assertion_gate = AssertionGate(strict_mode=strict_assertion_gate)

    def run_all_tests(self) -> HarnessResult:
        """Run all built-in test cases."""
        test_cases = self._get_test_cases()
        results = []

        for case in test_cases:
            result = self._run_test_case(case)
            results.append(result)

        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count

        return HarnessResult(
            passed=failed_count == 0,
            total_cases=len(results),
            passed_cases=passed_count,
            failed_cases=failed_count,
            case_results=results,
        )

    def run_on_export(self, export_path: Path | str) -> TestCaseResult:
        """
        Run harness on an exported JSON file.

        Args:
            export_path: Path to exported CDR JSON

        Returns:
            TestCaseResult
        """
        export_path = Path(export_path)
        with open(export_path) as f:
            data = json.load(f)

        case = self._export_to_test_case(data, export_path.stem)
        return self._run_test_case(case)

    def _get_test_cases(self) -> list[dict]:
        """Get built-in test cases for semantic coherence."""
        return [
            # Case 1: DOAC vs Aspirin - the problematic case
            {
                "case_id": "TC001",
                "case_name": "DOAC vs Aspirin head-to-head (P1 P2 P3 scenario)",
                "pico": {
                    "population": "Patients with atrial fibrillation",
                    "intervention": "Aspirin",
                    "comparator": "placebo or no treatment",  # What user question implies
                    "comparator_source": "assumed_from_question",
                    "outcome": "Stroke prevention",
                },
                "claims": [
                    {
                        "claim_id": "claim_001",
                        "claim_text": "In patients with subclinical AF, apixaban was superior to aspirin for stroke prevention",
                        "certainty": "moderate",
                        "therapeutic_context": "doac_vs_aspirin",
                        "supporting_snippet_ids": ["rec_001_snip_0"],
                    },
                    {
                        "claim_id": "claim_002",
                        "claim_text": "DOAC therapy showed lower stroke rates compared to aspirin in AF patients",
                        "certainty": "moderate",
                        "therapeutic_context": "doac_vs_aspirin",
                        "supporting_snippet_ids": ["rec_002_snip_0"],
                    },
                ],
                "conclusion": "Aspirin may reduce the risk of stroke in patients with AF.",
                "expected_checks": {
                    "comparator_source_valid": True,
                    "therapeutic_context_present": True,
                    "no_invalid_leap": False,  # Should FAIL - invalid leap
                    "context_purity": True,
                },
            },
            # Case 2: Proper comparative conclusion
            {
                "case_id": "TC002",
                "case_name": "DOAC vs Aspirin with correct comparative conclusion",
                "pico": {
                    "population": "Patients with atrial fibrillation",
                    "intervention": "Aspirin",
                    "comparator": "DOACs",
                    "comparator_source": "user_specified",
                    "outcome": "Stroke prevention",
                },
                "claims": [
                    {
                        "claim_id": "claim_001",
                        "claim_text": "DOACs showed superior efficacy compared to aspirin for stroke prevention",
                        "certainty": "moderate",
                        "therapeutic_context": "doac_vs_aspirin",
                        "supporting_snippet_ids": ["rec_001_snip_0"],
                    },
                ],
                "conclusion": "In head-to-head trials comparing DOACs to aspirin, DOACs demonstrated lower stroke rates.",
                "expected_checks": {
                    "comparator_source_valid": True,
                    "therapeutic_context_present": True,
                    "no_invalid_leap": True,  # Should PASS - comparative claim
                    "context_purity": True,
                },
            },
            # Case 3: Add-on therapy context
            {
                "case_id": "TC003",
                "case_name": "Aspirin + Anticoagulant add-on scenario",
                "pico": {
                    "population": "Patients on anticoagulation",
                    "intervention": "Aspirin added to anticoagulant",
                    "comparator": "Anticoagulant alone",
                    "comparator_source": "user_specified",
                    "outcome": "Cardiovascular events and bleeding",
                },
                "claims": [
                    {
                        "claim_id": "claim_001",
                        "claim_text": "Adding aspirin to anticoagulation increases bleeding risk without clear benefit",
                        "certainty": "low",
                        "therapeutic_context": "aspirin_plus_anticoagulant",
                        "supporting_snippet_ids": ["rec_001_snip_0"],
                    },
                ],
                "conclusion": "Adding aspirin to anticoagulant therapy may increase bleeding without additional cardiovascular benefit.",
                "expected_checks": {
                    "comparator_source_valid": True,
                    "therapeutic_context_present": True,
                    "no_invalid_leap": True,
                    "context_purity": True,
                },
            },
            # Case 4: Context mixing (should warn)
            {
                "case_id": "TC004",
                "case_name": "Mixed contexts - monotherapy and combination",
                "pico": {
                    "population": "AF patients",
                    "intervention": "Aspirin",
                    "comparator": None,
                    "comparator_source": "not_applicable",
                    "outcome": "Various outcomes",
                },
                "claims": [
                    {
                        "claim_id": "claim_001",
                        "claim_text": "Aspirin monotherapy has limited efficacy in AF",
                        "certainty": "low",
                        "therapeutic_context": "aspirin_monotherapy",
                        "supporting_snippet_ids": ["rec_001_snip_0"],
                    },
                    {
                        "claim_id": "claim_002",
                        "claim_text": "Aspirin plus warfarin increases bleeding",
                        "certainty": "moderate",
                        "therapeutic_context": "aspirin_plus_anticoagulant",
                        "supporting_snippet_ids": ["rec_002_snip_0"],
                    },
                ],
                "conclusion": "Aspirin use in AF depends on the clinical context.",
                "expected_checks": {
                    "comparator_source_valid": True,
                    "therapeutic_context_present": True,
                    "no_invalid_leap": True,
                    "context_purity": False,  # Should WARN - mixed contexts
                },
            },
            # Case 5: Null comparator with comparative evidence (P1 scenario)
            {
                "case_id": "TC005",
                "case_name": "Null comparator despite comparative evidence",
                "pico": {
                    "population": "AF patients",
                    "intervention": "Aspirin",
                    "comparator": None,  # Problem: null but evidence is comparative
                    "comparator_source": "not_applicable",
                    "outcome": "Stroke",
                },
                "claims": [
                    {
                        "claim_id": "claim_001",
                        "claim_text": "Apixaban superior to aspirin",
                        "certainty": "high",
                        "therapeutic_context": "doac_vs_aspirin",
                        "supporting_snippet_ids": ["rec_001_snip_0"],
                    },
                    {
                        "claim_id": "claim_002",
                        "claim_text": "Edoxaban showed better outcomes vs aspirin",
                        "certainty": "moderate",
                        "therapeutic_context": "doac_vs_aspirin",
                        "supporting_snippet_ids": ["rec_002_snip_0"],
                    },
                ],
                "conclusion": "Evidence supports DOAC use over aspirin in AF.",
                "expected_checks": {
                    "comparator_source_valid": False,  # Should WARN - null comparator with comparative evidence
                    "therapeutic_context_present": True,
                    "no_invalid_leap": True,  # Conclusion is correctly comparative
                    "context_purity": True,
                },
            },
        ]

    def _run_test_case(self, case: dict) -> TestCaseResult:
        """Run a single test case."""
        checks = []

        # Parse PICO
        pico_data = case["pico"]
        pico = PICO(
            population=pico_data["population"],
            intervention=pico_data["intervention"],
            comparator=pico_data.get("comparator"),
            comparator_source=ComparatorSource(
                pico_data.get("comparator_source", "not_applicable")
            ),
            outcome=pico_data["outcome"],
        )

        # Parse claims
        claims = []
        for claim_data in case["claims"]:
            claims.append(
                EvidenceClaim(
                    claim_id=claim_data["claim_id"],
                    claim_text=claim_data["claim_text"],
                    certainty=GRADECertainty(claim_data["certainty"]),
                    therapeutic_context=TherapeuticContext(
                        claim_data.get("therapeutic_context", "unclassified")
                    ),
                    supporting_snippet_ids=claim_data["supporting_snippet_ids"],
                )
            )

        conclusion = case["conclusion"]
        expected = case["expected_checks"]

        # === CHECK 1: comparator_source valid ===
        comparator_valid = (
            pico.comparator_source != ComparatorSource.NOT_APPLICABLE or pico.comparator is None
        )
        # Special case: if evidence is comparative, PICO should have comparator
        comparative_claims = sum(
            1
            for c in claims
            if c.therapeutic_context
            in {
                TherapeuticContext.DOAC_VS_ASPIRIN,
                TherapeuticContext.HEAD_TO_HEAD,
                TherapeuticContext.DOAC_VS_WARFARIN,
            }
        )
        if comparative_claims > len(claims) * 0.5 and pico.comparator is None:
            comparator_valid = False

        checks.append(
            HarnessCheck(
                name="comparator_source_valid",
                result=CheckResult.PASS
                if comparator_valid == expected.get("comparator_source_valid", True)
                else CheckResult.FAIL,
                expected=str(expected.get("comparator_source_valid", True)),
                actual=str(comparator_valid),
                detail=f"PICO.comparator={pico.comparator}, source={pico.comparator_source.value}",
            )
        )

        # === CHECK 2: therapeutic_context present ===
        all_have_context = all(
            c.therapeutic_context != TherapeuticContext.UNCLASSIFIED for c in claims
        )
        checks.append(
            HarnessCheck(
                name="therapeutic_context_present",
                result=CheckResult.PASS
                if all_have_context == expected.get("therapeutic_context_present", True)
                else CheckResult.FAIL,
                expected=str(expected.get("therapeutic_context_present", True)),
                actual=str(all_have_context),
                detail=f"Contexts: {[c.therapeutic_context.value for c in claims]}",
            )
        )

        # === CHECK 3: no_invalid_leap (via AssertionGate) ===
        gate_result = self.assertion_gate.validate(conclusion, claims, pico)
        has_invalid_leap = any(
            v.violation_type.value == "invalid_leap" for v in gate_result.violations
        )
        no_leap_expected = expected.get("no_invalid_leap", True)

        checks.append(
            HarnessCheck(
                name="no_invalid_leap",
                result=CheckResult.PASS
                if (not has_invalid_leap) == no_leap_expected
                else CheckResult.FAIL,
                expected=f"no_invalid_leap={no_leap_expected}",
                actual=f"has_leap={has_invalid_leap}",
                detail=f"Gate violations: {len(gate_result.violations)}",
            )
        )

        # === CHECK 4: context_purity ===
        unique_contexts = {
            c.therapeutic_context
            for c in claims
            if c.therapeutic_context != TherapeuticContext.UNCLASSIFIED
        }
        # Check for problematic combinations
        has_mixing = (
            TherapeuticContext.ASPIRIN_MONOTHERAPY in unique_contexts
            and TherapeuticContext.ASPIRIN_PLUS_ANTICOAGULANT in unique_contexts
        )
        context_pure = not has_mixing
        context_purity_expected = expected.get("context_purity", True)

        check_result = (
            CheckResult.PASS if context_pure == context_purity_expected else CheckResult.WARN
        )
        checks.append(
            HarnessCheck(
                name="context_purity",
                result=check_result,
                expected=f"pure={context_purity_expected}",
                actual=f"pure={context_pure}",
                detail=f"Contexts: {[c.value for c in unique_contexts]}",
            )
        )

        # Determine overall pass/fail
        failed = any(c.result == CheckResult.FAIL for c in checks)

        return TestCaseResult(
            case_id=case["case_id"],
            case_name=case["case_name"],
            passed=not failed,
            checks=checks,
            assertion_gate_result=gate_result,
        )

    def _export_to_test_case(self, data: dict, case_id: str) -> dict:
        """Convert an exported JSON to a test case format."""
        pico = data.get("pico", {})
        claims_data = data.get("claims", [])
        answer = data.get("answer", "")

        return {
            "case_id": case_id,
            "case_name": f"Export: {case_id}",
            "pico": {
                "population": pico.get("population", "Unknown"),
                "intervention": pico.get("intervention", "Unknown"),
                "comparator": pico.get("comparator"),
                "comparator_source": pico.get("comparator_source", "not_applicable"),
                "outcome": pico.get("outcome", "Unknown"),
            },
            "claims": [
                {
                    "claim_id": c.get("claim_id", f"claim_{i}"),
                    "claim_text": c.get("claim_text", c.get("text", "")),
                    "certainty": c.get("certainty", "low"),
                    "therapeutic_context": c.get("therapeutic_context", "unclassified"),
                    "supporting_snippet_ids": c.get("supporting_snippet_ids", []),
                }
                for i, c in enumerate(claims_data)
            ],
            "conclusion": answer,
            "expected_checks": {
                # For real exports, we check against DoD criteria
                "comparator_source_valid": True,
                "therapeutic_context_present": True,
                "no_invalid_leap": True,
                "context_purity": True,
            },
        }


def run_harness() -> HarnessResult:
    """Run the semantic coherence harness."""
    harness = SemanticCoherenceHarness(strict_assertion_gate=True)
    return harness.run_all_tests()


if __name__ == "__main__":
    result = run_harness()
    result.print_summary()

    # Exit with appropriate code
    import sys

    sys.exit(0 if result.passed else 1)
