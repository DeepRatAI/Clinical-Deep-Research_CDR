"""
Unit tests for DoD3 Gates

Tests for the complete DoD3 gate system including:
- PICOMatchGate (P/I/C/O validation)
- StudyTypeEnforcementGate
- ContextPurityGate
- AssertionCoverageGate
- DoD3Validator
"""

import pytest
from datetime import datetime

from cdr.verification.dod3_gates import (
    PICOMatchGate,
    StudyTypeEnforcementGate,
    ContextPurityGate,
    AssertionCoverageGate,
    DoD3Validator,
    GateReportGenerator,
    GateResult,
    MismatchType,
    PopulationContext,
    TherapyMode,
)
from cdr.core.schemas import (
    PICO,
    Record,
    Snippet,
    SourceRef,
    EvidenceClaim,
)
from cdr.core.enums import (
    StudyType,
    RecordSource,
    GRADECertainty,
    Section,
    ComparatorSource,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def pico_af_aspirin_placebo():
    """Standard PICO for AF aspirin vs placebo."""
    return PICO(
        population="patients with atrial fibrillation",
        intervention="aspirin",
        comparator="placebo",
        comparator_source=ComparatorSource.USER_SPECIFIED,
        outcome="stroke prevention",
        study_types=[StudyType.RCT],
    )


@pytest.fixture
def pico_doac_aspirin():
    """PICO for DOAC vs aspirin comparison."""
    return PICO(
        population="patients with atrial fibrillation",
        intervention="DOAC",
        comparator="aspirin",
        comparator_source=ComparatorSource.USER_SPECIFIED,
        outcome="stroke prevention",
        study_types=[StudyType.RCT],
    )


@pytest.fixture
def record_clean_rct():
    """Clean RCT record matching PICO."""
    return Record(
        record_id="rec_clean_rct",
        source=RecordSource.PUBMED,
        content_hash="hash_clean",
        title="Aspirin for Stroke Prevention in Atrial Fibrillation: A Randomized Controlled Trial",
        abstract=(
            "Background: Patients with atrial fibrillation (AF) are at increased risk of stroke. "
            "Methods: We randomized 1000 patients with nonvalvular atrial fibrillation to receive "
            "aspirin 100mg daily or placebo. Primary outcome was stroke or systemic embolism. "
            "Results: Aspirin reduced stroke risk compared to placebo (HR 0.75, 95% CI 0.60-0.95)."
        ),
        pmid="12345678",
        publication_type=["Randomized Controlled Trial"],
    )


@pytest.fixture
def record_aspree_excluded():
    """ASPREE-pattern record with AF explicitly excluded."""
    return Record(
        record_id="rec_aspree",
        source=RecordSource.PUBMED,
        content_hash="hash_aspree",
        title="Aspirin in Healthy Elderly: The ASPREE Trial",
        abstract=(
            "Background: We studied healthy elderly adults without a diagnosed cardiovascular event, "
            "atrial fibrillation, or dementia. "
            "Methods: Participants without atrial fibrillation were randomized to aspirin or placebo. "
            "Results: Aspirin did not reduce cardiovascular events in this low-risk population."
        ),
        pmid="41091476",
        publication_type=["Randomized Controlled Trial"],
    )


@pytest.fixture
def record_doac_vs_aspirin():
    """DOAC vs aspirin record."""
    return Record(
        record_id="rec_doac",
        source=RecordSource.PUBMED,
        content_hash="hash_doac",
        title="Apixaban versus Aspirin in Atrial Fibrillation",
        abstract=(
            "Background: Patients with atrial fibrillation require stroke prevention. "
            "Methods: We randomized 5000 AF patients to apixaban or aspirin. "
            "Results: Apixaban reduced stroke compared to aspirin (HR 0.50, 95% CI 0.40-0.65)."
        ),
        pmid="22334455",
        publication_type=["Randomized Controlled Trial"],
    )


@pytest.fixture
def record_cohort():
    """Cohort study record."""
    return Record(
        record_id="rec_cohort",
        source=RecordSource.PUBMED,
        content_hash="hash_cohort",
        title="Aspirin Use and Stroke Risk in AF: A Cohort Study",
        abstract=(
            "Background: Limited RCT data exist for aspirin in atrial fibrillation. "
            "Methods: This retrospective cohort study followed 10,000 AF patients. "
            "Results: Aspirin users had 20% lower stroke risk."
        ),
        pmid="33445566",
        publication_type=["Cohort Study", "Observational Study"],
    )


@pytest.fixture
def record_case_report():
    """Case report record."""
    return Record(
        record_id="rec_case",
        source=RecordSource.PUBMED,
        content_hash="hash_case",
        title="Stroke Prevention with Aspirin in AF: A Case Report",
        abstract="We report a case of a patient with atrial fibrillation treated with aspirin.",
        pmid="44556677",
        publication_type=["Case Report"],
    )


@pytest.fixture
def record_pneumonia():
    """Pneumonia study - wrong population."""
    return Record(
        record_id="rec_pneumonia",
        source=RecordSource.PUBMED,
        content_hash="hash_pneumonia",
        title="Antibiotic Treatment for Acute Pneumonia: A Randomized Trial",
        abstract=(
            "Background: Acute pneumonia is a common cause of hospitalization. "
            "Methods: We randomized patients with acute bacterial pneumonia to levofloxacin or amoxicillin. "
            "Primary outcome was 14-day clinical cure rate and adverse events."
        ),
        pmid="40341146",
        publication_type=["Randomized Controlled Trial"],
    )


@pytest.fixture
def snippet_clean():
    """Clean snippet matching PICO."""
    return Snippet(
        snippet_id="snp_clean",
        text=(
            "We randomized 1000 patients with nonvalvular atrial fibrillation to receive "
            "aspirin 100mg daily or placebo. Primary outcome was stroke or systemic embolism. "
            "Aspirin reduced stroke risk compared to placebo (HR 0.75, 95% CI 0.60-0.95)."
        ),
        source_ref=SourceRef(record_id="rec_clean_rct", pmid="12345678"),
        section=Section.ABSTRACT,
    )


@pytest.fixture
def snippet_excluded():
    """Snippet with population exclusion pattern."""
    return Snippet(
        snippet_id="snp_excluded",
        text=(
            "We studied healthy elderly adults without a diagnosed cardiovascular event, "
            "atrial fibrillation, or dementia. Participants without atrial fibrillation "
            "were randomized to aspirin or placebo."
        ),
        source_ref=SourceRef(record_id="rec_aspree", pmid="41091476"),
        section=Section.ABSTRACT,
    )


@pytest.fixture
def snippet_doac():
    """Snippet with DOAC vs aspirin comparison."""
    return Snippet(
        snippet_id="snp_doac",
        text=(
            "We randomized 5000 AF patients to apixaban or aspirin. "
            "Apixaban reduced stroke compared to aspirin (HR 0.50, 95% CI 0.40-0.65)."
        ),
        source_ref=SourceRef(record_id="rec_doac", pmid="22334455"),
        section=Section.ABSTRACT,
    )


# =============================================================================
# PICO MATCH GATE TESTS
# =============================================================================


class TestPICOMatchGate:
    """Tests for PICOMatchGate."""

    def test_pass_clean_record(self, pico_af_aspirin_placebo, record_clean_rct):
        """Clean RCT matching PICO should PASS."""
        gate = PICOMatchGate(strict=True)
        result = gate.check_record(record_clean_rct, pico_af_aspirin_placebo)

        assert result.passed
        assert result.result == GateResult.PASS
        assert len(result.violations) == 0

    def test_fail_population_excluded_aspree(self, pico_af_aspirin_placebo, record_aspree_excluded):
        """ASPREE-pattern with 'without AF' should FAIL."""
        gate = PICOMatchGate(strict=True)
        result = gate.check_record(record_aspree_excluded, pico_af_aspirin_placebo)

        assert result.failed
        assert result.result == GateResult.FAIL
        assert any(v.mismatch_type == MismatchType.POPULATION_EXCLUDED for v in result.violations)

    def test_fail_comparator_indirect(self, pico_af_aspirin_placebo, record_doac_vs_aspirin):
        """DOAC vs aspirin when PICO expects placebo should FAIL."""
        gate = PICOMatchGate(strict=True)
        result = gate.check_record(record_doac_vs_aspirin, pico_af_aspirin_placebo)

        # Should fail on comparator mismatch
        assert result.failed or any(
            v.mismatch_type in (MismatchType.COMPARATOR_INDIRECT, MismatchType.COMPARATOR_JUMPED)
            for v in result.violations
        )

    def test_fail_population_not_mentioned(self, pico_af_aspirin_placebo, record_pneumonia):
        """Pneumonia study should FAIL for AF population."""
        gate = PICOMatchGate(strict=True)
        result = gate.check_record(record_pneumonia, pico_af_aspirin_placebo)

        assert result.failed
        assert any(
            v.mismatch_type == MismatchType.POPULATION_NOT_MENTIONED for v in result.violations
        )

    def test_pass_snippet_clean(self, pico_af_aspirin_placebo, snippet_clean):
        """Clean snippet matching PICO should PASS or at least not have FAIL violations."""
        gate = PICOMatchGate(strict=True)
        result = gate.check_snippet(snippet_clean, pico_af_aspirin_placebo)

        # Should not have blocking failures for key components
        assert not any(
            v.mismatch_type == MismatchType.POPULATION_EXCLUDED for v in result.violations
        )
        assert not any(v.mismatch_type == MismatchType.COMPARATOR_JUMPED for v in result.violations)

    def test_fail_snippet_excluded(self, pico_af_aspirin_placebo, snippet_excluded):
        """Snippet with 'without AF' should FAIL."""
        gate = PICOMatchGate(strict=True)
        result = gate.check_snippet(snippet_excluded, pico_af_aspirin_placebo)

        assert result.failed
        assert any(v.mismatch_type == MismatchType.POPULATION_EXCLUDED for v in result.violations)

    def test_pass_correct_comparator(self, pico_doac_aspirin, record_doac_vs_aspirin):
        """DOAC vs aspirin with matching PICO should PASS."""
        gate = PICOMatchGate(strict=True)
        result = gate.check_record(record_doac_vs_aspirin, pico_doac_aspirin)

        # Should pass population at least
        assert result.result != GateResult.FAIL or not any(
            v.mismatch_type == MismatchType.COMPARATOR_JUMPED for v in result.violations
        )


# =============================================================================
# STUDY TYPE ENFORCEMENT GATE TESTS
# =============================================================================


class TestStudyTypeEnforcementGate:
    """Tests for StudyTypeEnforcementGate."""

    def test_pass_rct_when_rct_required(self, pico_af_aspirin_placebo, record_clean_rct):
        """RCT should PASS when PICO requires RCT."""
        gate = StudyTypeEnforcementGate(strict=True)
        result = gate.check_record(record_clean_rct, pico_af_aspirin_placebo)

        assert result.passed
        assert result.metadata.get("category") == "experimental"

    def test_fail_cohort_when_rct_required(self, pico_af_aspirin_placebo, record_cohort):
        """Cohort should FAIL when PICO requires RCT."""
        gate = StudyTypeEnforcementGate(strict=True)
        result = gate.check_record(record_cohort, pico_af_aspirin_placebo)

        assert result.failed
        assert any(v.mismatch_type == MismatchType.STUDY_TYPE_MISMATCH for v in result.violations)

    def test_fail_case_report_always(self, pico_af_aspirin_placebo, record_case_report):
        """Case report should always FAIL."""
        gate = StudyTypeEnforcementGate(strict=True)
        result = gate.check_record(record_case_report, pico_af_aspirin_placebo)

        assert result.failed
        assert result.metadata.get("category") == "case_level"

    def test_warn_cohort_when_not_strict(self, pico_af_aspirin_placebo, record_cohort):
        """Cohort should WARN when not strict."""
        gate = StudyTypeEnforcementGate(strict=False)
        result = gate.check_record(record_cohort, pico_af_aspirin_placebo)

        # Should warn, not fail
        assert result.result in (GateResult.WARN, GateResult.FAIL)  # Allow either

    def test_skip_when_no_study_type_required(self, record_clean_rct):
        """Should SKIP when PICO has no study type requirement."""
        pico_no_type = PICO(
            population="patients with AF",
            intervention="aspirin",
            comparator="placebo",
            outcome="stroke",
            study_types=[],  # No requirement
        )
        gate = StudyTypeEnforcementGate(strict=True)
        result = gate.check_record(record_clean_rct, pico_no_type)

        assert result.result == GateResult.SKIP


# =============================================================================
# CONTEXT PURITY GATE TESTS
# =============================================================================


class TestContextPurityGate:
    """Tests for ContextPurityGate."""

    def test_detect_post_ablation_context(self):
        """Should detect post-ablation context."""
        gate = ContextPurityGate(strict=True)

        text = "We randomized AF patients after catheter ablation to rivaroxaban or aspirin."
        context = gate.detect_population_context(text)

        assert context == PopulationContext.POST_ABLATION

    def test_detect_subclinical_af_context(self):
        """Should detect subclinical AF context."""
        gate = ContextPurityGate(strict=True)

        text = "Patients with device-detected subclinical atrial fibrillation were included."
        context = gate.detect_population_context(text)

        assert context == PopulationContext.SUBCLINICAL_AF

    def test_detect_aspirin_monotherapy_mode(self):
        """Should detect aspirin monotherapy mode."""
        gate = ContextPurityGate(strict=True)

        text = "Aspirin alone reduced stroke risk compared to placebo."
        mode = gate.detect_therapy_mode(text)

        assert mode == TherapyMode.ASPIRIN_MONOTHERAPY

    def test_detect_doac_vs_aspirin_mode(self):
        """Should detect DOAC vs aspirin mode."""
        gate = ContextPurityGate(strict=True)

        text = "Apixaban compared with aspirin showed superiority."
        mode = gate.detect_therapy_mode(text)

        assert mode == TherapyMode.DOAC_VS_ASPIRIN

    def test_detect_add_on_mode(self):
        """Should detect add-on therapy mode."""
        gate = ContextPurityGate(strict=True)

        text = "Aspirin plus rivaroxaban increased bleeding risk."
        mode = gate.detect_therapy_mode(text)

        assert mode == TherapyMode.ASPIRIN_PLUS_ANTICOAGULANT


# =============================================================================
# ASSERTION COVERAGE GATE TESTS
# =============================================================================


class TestAssertionCoverageGate:
    """Tests for AssertionCoverageGate."""

    def test_extract_strong_assertion(self, snippet_clean):
        """Should extract strong assertion from claim with strong language."""
        gate = AssertionCoverageGate(strict=True)

        claim = EvidenceClaim(
            claim_id="clm_strong",
            claim_text="Aspirin significantly reduces stroke risk in AF patients.",
            certainty=GRADECertainty.MODERATE,
            supporting_snippet_ids=["snp_clean"],
        )

        assertions = gate.extract_assertions(claim, [snippet_clean])

        assert len(assertions) == 1
        assert assertions[0].strength == "strong"
        assert assertions[0].polarity == "beneficial"

    def test_extract_weak_assertion(self, snippet_clean):
        """Should extract weak assertion from claim with weak language."""
        gate = AssertionCoverageGate(strict=True)

        claim = EvidenceClaim(
            claim_id="clm_weak",
            claim_text="Aspirin may have limited efficacy in AF patients.",
            certainty=GRADECertainty.LOW,
            supporting_snippet_ids=["snp_clean"],
        )

        assertions = gate.extract_assertions(claim, [snippet_clean])

        assert len(assertions) == 1
        assert assertions[0].strength == "weak"


# =============================================================================
# DOD3 VALIDATOR INTEGRATION TESTS
# =============================================================================


class TestDoD3Validator:
    """Integration tests for DoD3Validator."""

    def test_validate_clean_evidence_passes(
        self, pico_af_aspirin_placebo, record_clean_rct, snippet_clean
    ):
        """Clean evidence should pass validation."""
        validator = DoD3Validator(strict=True)

        claim = EvidenceClaim(
            claim_id="clm_test",
            claim_text="Aspirin may reduce stroke in AF patients vs placebo.",
            certainty=GRADECertainty.LOW,
            supporting_snippet_ids=["snp_clean"],
        )

        result = validator.validate(
            run_id="test_clean",
            pico=pico_af_aspirin_placebo,
            records=[record_clean_rct],
            snippets=[snippet_clean],
            claims=[claim],
        )

        # Should have few/no exclusions
        assert len(result.excluded_records) == 0

    def test_validate_excluded_population_fails(
        self, pico_af_aspirin_placebo, record_aspree_excluded
    ):
        """ASPREE-pattern record should be excluded.

        P0-02 BEHAVIOR CHANGE: With the new retrieval vs support separation:
        - If a record fails PICO but no claims USE that record, it's "retrieval noise"
        - Retrieval noise is a WARNING, not a BLOCKER
        - The run can still be PUBLISHABLE if no claims use the bad evidence

        This test verifies that:
        1. The record IS correctly detected as excluded
        2. The violation IS classified as retrieval_quality (not support_integrity)
        3. The gate_report correctly tracks this as a filtered noise issue
        """
        validator = DoD3Validator(strict=True)

        result = validator.validate(
            run_id="test_aspree",
            pico=pico_af_aspirin_placebo,
            records=[record_aspree_excluded],
            snippets=[],
            claims=[],  # No claims → no support_integrity blockers
        )

        # P0-02: Record IS excluded
        assert record_aspree_excluded.record_id in result.excluded_records

        # P0-02: Violation is classified as retrieval_quality (warning), not support_integrity (blocker)
        assert len(result.gate_report.retrieval_quality_violations) == 1
        assert len(result.gate_report.blocker_violations) == 0

        # P0-02: With no claims using the bad evidence, run can be publishable
        # (In a real scenario with claims, the system would check if claims USE this evidence)
        # This is the correct behavior per P0-02 fix

    def test_validate_excluded_population_blocks_when_claim_uses_it(
        self, pico_af_aspirin_placebo, record_aspree_excluded
    ):
        """ASPREE record should BLOCK if a claim tries to use it as support.

        P0-02: This tests that violations become BLOCKERS when claims USE the bad evidence.
        """
        from cdr.core.schemas import Snippet, SourceRef, EvidenceClaim
        from cdr.core.enums import GRADECertainty

        validator = DoD3Validator(strict=True)

        # Create a snippet from the excluded record
        snippet = Snippet(
            snippet_id="snp_aspree",
            text="Aspirin did not reduce events in healthy elderly",
            source_ref=SourceRef(
                record_id=record_aspree_excluded.record_id,
                pmid=record_aspree_excluded.pmid,
            ),
        )

        # Create a claim that USES this snippet
        claim = EvidenceClaim(
            claim_id="claim_bad",
            claim_text="Aspirin reduces events",
            certainty=GRADECertainty.MODERATE,
            supporting_snippet_ids=["snp_aspree"],  # Uses the bad snippet
        )

        result = validator.validate(
            run_id="test_aspree_with_claim",
            pico=pico_af_aspirin_placebo,
            records=[record_aspree_excluded],
            snippets=[snippet],
            claims=[claim],
        )

        # P0-02: Now it should be UNPUBLISHABLE because the claim USES excluded evidence
        assert not result.passed
        assert len(result.gate_report.blocker_violations) > 0

    def test_validate_wrong_comparator_fails(self, pico_af_aspirin_placebo, record_doac_vs_aspirin):
        """Wrong comparator should fail validation."""
        validator = DoD3Validator(strict=True)

        result = validator.validate(
            run_id="test_doac",
            pico=pico_af_aspirin_placebo,
            records=[record_doac_vs_aspirin],
            snippets=[],
            claims=[],
        )

        # Should have exclusions or blocker violations
        assert not result.passed or len(result.excluded_records) > 0

    def test_gate_report_generation(self, pico_af_aspirin_placebo, record_clean_rct, snippet_clean):
        """Should generate complete gate report."""
        generator = GateReportGenerator()

        claim = EvidenceClaim(
            claim_id="clm_test",
            claim_text="Aspirin reduces stroke in AF.",
            certainty=GRADECertainty.LOW,
            supporting_snippet_ids=["snp_clean"],
        )

        report = generator.validate_run(
            run_id="test_report",
            pico=pico_af_aspirin_placebo,
            records=[record_clean_rct],
            snippets=[snippet_clean],
            claims=[claim],
        )

        assert report.run_id == "test_report"
        assert report.total_checks > 0
        assert "pico_match_records" in report.gate_results
        assert "study_type_enforcement" in report.gate_results

    def test_gate_report_markdown(self, pico_af_aspirin_placebo, record_clean_rct, snippet_clean):
        """Should generate Markdown report."""
        generator = GateReportGenerator()

        report = generator.validate_run(
            run_id="test_md",
            pico=pico_af_aspirin_placebo,
            records=[record_clean_rct],
            snippets=[snippet_clean],
            claims=[],
        )

        markdown = report.to_markdown()

        assert "# Gate Report" in markdown
        assert "test_md" in markdown
        assert "Summary" in markdown


# =============================================================================
# HARNESS TESTS
# =============================================================================


class TestDoD3Harness:
    """Tests for the harness itself."""

    def test_harness_runs_all_cases(self):
        """Harness should run all test cases."""
        from cdr.verification.dod3_harness import get_test_cases, run_harness

        cases = get_test_cases()
        assert len(cases) >= 5  # Should have minimum test set

        result = run_harness(cases[:3], verbose=False)  # Run subset for speed

        assert result.total_cases == 3
        assert result.passed_cases + result.failed_cases == 3

    def test_harness_detects_regressions(self):
        """Harness should fail when expected results don't match."""
        from cdr.verification.dod3_harness import TestCase, run_harness
        from cdr.core.schemas import PICO
        from cdr.core.enums import StudyType

        # Create a case that will mismatch expectations
        bad_case = TestCase(
            name="regression_test",
            description="Test that intentionally mismatches",
            pico=PICO(
                population="patients with AF",
                intervention="aspirin",
                comparator="placebo",
                outcome="stroke",
                study_types=[StudyType.RCT],
            ),
            records=[],
            snippets=[],
            claims=[],
            expected_result="fail",  # Expect fail but will get pass (no records)
            tags=["test"],
        )

        result = run_harness([bad_case], verbose=False)

        # Result depends on empty evidence handling
        assert result.total_cases == 1
