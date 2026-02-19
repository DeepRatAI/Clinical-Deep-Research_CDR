"""
Evidence Gates Test Harness

Unit tests for deterministic evidence validation gates.
Tests population matching, comparator alignment, study type consistency,
and deduplication.

Run with:
    python -m pytest tests/test_evidence_gates.py -v
    # Or directly:
    python tests/test_evidence_gates.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cdr.core.enums import ComparatorSource, RecordSource, StudyType
from cdr.core.schemas import PICO, Record
from cdr.verification.evidence_gates import (
    ComparatorAlignmentGate,
    DeduplicationGate,
    EvidenceValidator,
    GateResult,
    PopulationMatchGate,
    StudyTypeConsistencyGate,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


def make_pico(
    population: str = "patients with atrial fibrillation",
    intervention: str = "aspirin",
    comparator: str | None = "placebo or no treatment",
    outcome: str = "stroke prevention",
    study_types: list[StudyType] | None = None,
) -> PICO:
    """Create a PICO for testing."""
    return PICO(
        population=population,
        intervention=intervention,
        comparator=comparator,
        comparator_source=ComparatorSource.USER_SPECIFIED
        if comparator
        else ComparatorSource.NOT_APPLICABLE,
        outcome=outcome,
        study_types=study_types if study_types is not None else [StudyType.RCT],
    )


def make_record(
    record_id: str = "test_001",
    pmid: str = "12345678",
    title: str = "Test Study",
    abstract: str = "A study of aspirin in patients with atrial fibrillation.",
    publication_type: list[str] | None = None,
) -> Record:
    """Create a Record for testing."""
    return Record(
        record_id=record_id,
        pmid=pmid,
        source=RecordSource.PUBMED,
        content_hash=f"hash_{record_id}",
        title=title,
        abstract=abstract,
        publication_type=publication_type or [],
    )


# =============================================================================
# POPULATION MATCH GATE TESTS
# =============================================================================


class TestPopulationMatchGate:
    """Tests for PopulationMatchGate."""

    def test_pass_when_population_mentioned(self):
        """Should PASS when PICO population is mentioned in abstract."""
        gate = PopulationMatchGate(strict=False)
        pico = make_pico(population="patients with atrial fibrillation")
        record = make_record(
            abstract="We studied patients with atrial fibrillation receiving aspirin."
        )

        result = gate.check_record(record, pico)

        assert result.result == GateResult.PASS
        assert not result.violations

    def test_fail_when_population_excluded(self):
        """Should FAIL when PICO population appears in exclusion criteria."""
        gate = PopulationMatchGate(strict=False)
        pico = make_pico(population="patients with atrial fibrillation")
        record = make_record(
            abstract=(
                "This trial enrolled patients with ACS. "
                "Exclusion criteria: patients with atrial fibrillation, "
                "history of stroke, or bleeding disorders."
            )
        )

        result = gate.check_record(record, pico)

        assert result.result == GateResult.FAIL
        assert len(result.violations) == 1
        assert result.violations[0].mismatch_type.value == "population_excluded"

    def test_warn_when_population_not_mentioned(self):
        """Should WARN when PICO population not mentioned (non-strict)."""
        gate = PopulationMatchGate(strict=False)
        pico = make_pico(population="patients with atrial fibrillation")
        record = make_record(
            abstract="This study examined elderly adults receiving aspirin for primary prevention."
        )

        result = gate.check_record(record, pico)

        assert result.result == GateResult.WARN
        assert len(result.violations) == 1
        assert result.violations[0].mismatch_type.value == "population_not_mentioned"

    def test_fail_when_population_not_mentioned_strict(self):
        """Should FAIL when PICO population not mentioned (strict mode)."""
        gate = PopulationMatchGate(strict=True)
        pico = make_pico(population="patients with atrial fibrillation")
        record = make_record(
            abstract="This study examined elderly adults receiving aspirin for primary prevention."
        )

        result = gate.check_record(record, pico)

        assert result.result == GateResult.FAIL

    def test_population_synonyms(self):
        """Should recognize population synonyms (AF, afib, etc.)."""
        gate = PopulationMatchGate(strict=False)
        pico = make_pico(population="patients with atrial fibrillation")
        record = make_record(abstract="AF patients were randomized to aspirin or placebo.")

        result = gate.check_record(record, pico)

        assert result.result == GateResult.PASS


# =============================================================================
# COMPARATOR ALIGNMENT GATE TESTS
# =============================================================================


class TestComparatorAlignmentGate:
    """Tests for ComparatorAlignmentGate."""

    def test_pass_when_placebo_comparator_found(self):
        """Should PASS when PICO expects placebo and evidence has placebo."""
        gate = ComparatorAlignmentGate(strict=False)
        pico = make_pico(comparator="placebo")
        record = make_record(
            abstract="Patients were randomized to aspirin 100mg daily versus placebo."
        )

        result = gate.check_record(record, pico)

        # Should not fail - placebo matches
        assert result.result in (GateResult.PASS, GateResult.WARN)

    def test_fail_when_active_comparator_vs_placebo_expected(self):
        """Should FAIL when PICO expects placebo but evidence is head-to-head."""
        gate = ComparatorAlignmentGate(strict=False)
        pico = make_pico(comparator="placebo")
        record = make_record(abstract="This trial compared apixaban versus aspirin in AF patients.")

        result = gate.check_record(record, pico)

        assert result.result == GateResult.FAIL
        assert len(result.violations) >= 1
        assert result.violations[0].mismatch_type.value == "comparator_indirect"

    def test_skip_when_no_pico_comparator(self):
        """Should SKIP when PICO has no comparator specified."""
        gate = ComparatorAlignmentGate(strict=False)
        pico = make_pico(comparator=None)
        record = make_record(abstract="Any study text here.")

        result = gate.check_record(record, pico)

        assert result.result == GateResult.SKIP

    def test_warn_when_no_comparator_found(self):
        """Should WARN when no explicit comparator found in evidence."""
        gate = ComparatorAlignmentGate(strict=False)
        pico = make_pico(comparator="placebo")
        record = make_record(
            abstract="We studied aspirin effects on stroke prevention in AF patients."
        )

        result = gate.check_record(record, pico)

        assert result.result == GateResult.WARN
        assert result.violations[0].mismatch_type.value == "comparator_missing"


# =============================================================================
# STUDY TYPE CONSISTENCY GATE TESTS
# =============================================================================


class TestStudyTypeConsistencyGate:
    """Tests for StudyTypeConsistencyGate."""

    def test_pass_when_rct_expected_and_found(self):
        """Should PASS when PICO requires RCT and evidence is RCT."""
        gate = StudyTypeConsistencyGate(strict=False)
        pico = make_pico(study_types=[StudyType.RCT])
        record = make_record(
            abstract="In this randomized controlled trial, patients were...",
            publication_type=["Randomized Controlled Trial"],
        )

        result = gate.check_record(record, pico)

        assert result.result == GateResult.PASS

    def test_fail_when_rct_expected_but_observational(self):
        """Should FAIL by default when PICO requires RCT but evidence is observational.

        The study_type_strict parameter defaults to True for clinical validity.
        """
        gate = StudyTypeConsistencyGate(strict=False, study_type_strict=True)
        pico = make_pico(study_types=[StudyType.RCT])
        record = make_record(
            abstract="This cohort study followed patients for 5 years...",
            publication_type=["Cohort Study"],
        )

        result = gate.check_record(record, pico)

        # Now FAILS by default because study_type_strict=True
        assert result.result == GateResult.FAIL
        assert result.violations[0].mismatch_type.value == "study_type_mismatch"

    def test_warn_when_study_type_strict_false(self):
        """Should WARN when study_type_strict=False and type mismatch."""
        gate = StudyTypeConsistencyGate(strict=False, study_type_strict=False)
        pico = make_pico(study_types=[StudyType.RCT])
        record = make_record(
            abstract="This cohort study followed patients for 5 years...",
            publication_type=["Cohort Study"],
        )

        result = gate.check_record(record, pico)

        assert result.result == GateResult.WARN
        assert result.violations[0].mismatch_type.value == "study_type_mismatch"

    def test_fail_when_strict_and_type_mismatch(self):
        """Should FAIL in strict mode when study type doesn't match."""
        gate = StudyTypeConsistencyGate(strict=True)
        pico = make_pico(study_types=[StudyType.RCT])
        record = make_record(
            abstract="This retrospective cohort analysis...",
            publication_type=["Observational Study"],
        )

        result = gate.check_record(record, pico)

        assert result.result == GateResult.FAIL

    def test_skip_when_no_study_type_restriction(self):
        """Should SKIP when PICO has no study type requirements."""
        gate = StudyTypeConsistencyGate(strict=False)
        pico = make_pico(study_types=[])
        record = make_record(abstract="Any study type here.")

        result = gate.check_record(record, pico)

        assert result.result == GateResult.SKIP


# =============================================================================
# EVIDENCE VALIDATOR (MASTER) TESTS
# =============================================================================


class TestEvidenceValidator:
    """Tests for the master EvidenceValidator."""

    def test_validates_all_gates(self):
        """Should run all gates and aggregate results."""
        validator = EvidenceValidator(strict=False)
        pico = make_pico(
            population="patients with atrial fibrillation",
            comparator="placebo",
            study_types=[StudyType.RCT],
        )
        record = make_record(
            abstract="This RCT compared aspirin to placebo in AF patients.",
            publication_type=["Randomized Controlled Trial"],
        )

        result = validator.validate_record(record, pico)

        assert result.is_in_scope
        assert len(result.gate_results) >= 3  # pop, comp, study_type

    def test_fails_on_population_exclusion(self):
        """Should fail overall when population is excluded."""
        validator = EvidenceValidator(strict=False)
        pico = make_pico(population="patients with atrial fibrillation")
        record = make_record(
            abstract="Excluded: patients with atrial fibrillation, stroke history."
        )

        result = validator.validate_record(record, pico)

        assert not result.is_in_scope
        assert result.overall_result == GateResult.FAIL
        assert result.degraded_reason is not None

    def test_real_world_aspree_case(self):
        """
        ASPREE substudy case: Should FAIL for AF PICO because ASPREE
        enrolled adults WITHOUT atrial fibrillation.
        """
        validator = EvidenceValidator(strict=False)
        pico = make_pico(
            population="patients with atrial fibrillation",
            intervention="aspirin",
            comparator="placebo",
        )
        # Simulating PMID 41091476 - ASPREE substudy abstract
        record = make_record(
            pmid="41091476",
            title="Aspirin and Cardiovascular Events in Older Adults",
            abstract=(
                "The ASPREE trial enrolled healthy older adults aged 70 years or older. "
                "Exclusion criteria included: atrial fibrillation, prior cardiovascular "
                "disease, dementia, or life expectancy less than 5 years. "
                "Participants were randomized to aspirin 100mg or placebo."
            ),
        )

        result = validator.validate_record(record, pico)

        # Should FAIL because AF is in exclusion criteria
        assert not result.is_in_scope
        assert result.overall_result == GateResult.FAIL
        assert "atrial fibrillation" in result.degraded_reason.lower()

    def test_real_world_doac_vs_aspirin_case(self):
        """
        DOAC vs Aspirin case: Should FAIL when PICO expects placebo
        but evidence is DOAC vs Aspirin head-to-head.
        """
        validator = EvidenceValidator(strict=False)
        pico = make_pico(
            population="patients with atrial fibrillation",
            intervention="aspirin",
            comparator="placebo or no treatment",
        )
        record = make_record(
            title="Apixaban versus Aspirin in Patients with Atrial Fibrillation",
            abstract=(
                "This randomized trial compared apixaban 5mg twice daily versus "
                "aspirin 81mg once daily in patients with atrial fibrillation "
                "and at least one additional risk factor for stroke."
            ),
        )

        result = validator.validate_record(record, pico)

        # Should FAIL because comparator is apixaban, not placebo
        assert not result.is_in_scope
        assert result.overall_result == GateResult.FAIL


# =============================================================================
# MAIN
# =============================================================================


def run_tests():
    """Run all tests and print results."""
    import traceback

    test_classes = [
        TestPopulationMatchGate,
        TestComparatorAlignmentGate,
        TestStudyTypeConsistencyGate,
        TestEvidenceValidator,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{'=' * 60}")
        print(f"Running {test_class.__name__}")
        print("=" * 60)

        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
            except Exception as e:
                print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                failed_tests.append((test_class.__name__, method_name, traceback.format_exc()))

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed_tests}/{total_tests} passed")
    print("=" * 60)

    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
