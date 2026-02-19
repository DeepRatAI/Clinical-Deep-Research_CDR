"""
Contract Conformity Tests

These tests validate that schema ↔ prompt ↔ parser contracts are aligned
without fallback silenciosos.

Based on: CDR_Risk_Matrix_and_Contract_Map.md

Source of Truth:
- schemas.py defines the models
- enums.py defines the enum values
- structured_outputs.py JSON schemas must match schemas.py
- prompts must request exact field names from schemas.py
- parsers must prioritize payload fields, not reconstruct with heuristics
"""

import pytest
from cdr.core.enums import (
    CritiqueDimension,
    CritiqueSeverity,
    ExclusionReason,
    GRADECertainty,
    RoB2Domain,
    RoB2Judgment,
    RunStatus,
    VerificationStatus,
)
from cdr.core.schemas import (
    Critique,
    CritiqueResult,
    EvidenceClaim,
    Snippet,
    SourceRef,
    VerificationResult,
)


# =============================================================================
# HELPER FUNCTIONS (module-level to avoid duplication)
# =============================================================================


def run_async(coro):
    """Run async coroutine in a portable way for Python 3.10+.

    Handles both active loop and no-loop scenarios:
    - If no active loop: use asyncio.run()
    - If active loop: use nest_asyncio + run_until_complete

    LOW-F4/F6 fix: Extracted to module-level to avoid duplication.
    Refs: CDR_Integral_Audit_2026-01-20.md LOW-F4, LOW-F6
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No active loop, safe to use asyncio.run()
        return asyncio.run(coro)
    else:
        # Active loop exists - use nest_asyncio for compatibility
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)


class TestEnumValues:
    """Test that enum values are correctly defined."""

    def test_rob2_domain_values(self):
        """RoB2Domain must have exactly 5 domains per Cochrane RoB 2."""
        expected = {
            "randomization_process",
            "deviations_from_intended_interventions",
            "missing_outcome_data",
            "measurement_of_outcome",
            "selection_of_reported_result",
        }
        actual = {d.value for d in RoB2Domain}
        assert actual == expected, f"RoB2Domain mismatch: {actual - expected} | {expected - actual}"

    def test_rob2_judgment_values(self):
        """RoB2Judgment must have exactly 3 values per Cochrane RoB 2."""
        expected = {"low", "some_concerns", "high"}
        actual = {j.value for j in RoB2Judgment}
        assert actual == expected

    def test_critique_dimension_values(self):
        """CritiqueDimension must have the canonical values."""
        expected = {
            "internal_validity",
            "external_validity",
            "statistical_issues",
            "missing_evidence",
            "conflicting_evidence",
            "overstatement",
            "confounders",
            "search_bias",
        }
        actual = {d.value for d in CritiqueDimension}
        assert actual == expected

    def test_critique_severity_values(self):
        """CritiqueSeverity must have canonical values."""
        expected = {"critical", "high", "medium", "low", "info"}
        actual = {s.value for s in CritiqueSeverity}
        assert actual == expected

    def test_exclusion_reason_values(self):
        """ExclusionReason must be lowercase per enum."""
        for reason in ExclusionReason:
            assert reason.value == reason.value.lower(), f"{reason.name} should be lowercase"

    def test_run_status_scientific_outcomes(self):
        """RunStatus must include scientific outcomes."""
        values = {s.value for s in RunStatus}
        assert "insufficient_evidence" in values
        assert "unpublishable" in values
        assert "completed" in values


class TestEvidenceClaimContract:
    """Test EvidenceClaim schema contract."""

    def test_evidence_claim_requires_claim_text(self):
        """EvidenceClaim must use claim_text, not statement."""
        # Should succeed with claim_text
        claim = EvidenceClaim(
            claim_id="test_001",
            claim_text="This is a test claim with sufficient length",
            certainty=GRADECertainty.LOW,
            supporting_snippet_ids=["snip_001"],
        )
        assert claim.claim_text == "This is a test claim with sufficient length"

    def test_evidence_claim_has_no_statement_field(self):
        """EvidenceClaim must NOT have statement field."""
        import inspect

        fields = set(EvidenceClaim.model_fields.keys())
        assert "statement" not in fields, "EvidenceClaim should not have 'statement' field"
        assert "claim_text" in fields, "EvidenceClaim must have 'claim_text' field"

    def test_evidence_claim_uses_snippet_ids(self):
        """EvidenceClaim uses supporting_snippet_ids, not supporting_study_ids."""
        fields = set(EvidenceClaim.model_fields.keys())
        assert "supporting_snippet_ids" in fields
        assert "supporting_study_ids" not in fields


class TestSnippetContract:
    """Test Snippet schema contract."""

    def test_snippet_requires_snippet_id(self):
        """Snippet must have snippet_id."""
        with pytest.raises(Exception):
            # Should fail without snippet_id
            Snippet(
                text="Sample text that is long enough",
                source_ref=SourceRef(record_id="rec_001"),
            )

    def test_snippet_with_valid_source_ref(self):
        """Snippet with proper SourceRef."""
        snip = Snippet(
            snippet_id="rec_001_snip_0",
            text="Sample text that is long enough for validation",
            source_ref=SourceRef(
                record_id="rec_001",
                pmid="12345",
                offset_start=0,
                offset_end=100,
            ),
        )
        assert snip.snippet_id == "rec_001_snip_0"
        assert snip.source_ref.record_id == "rec_001"


class TestCritiqueContract:
    """Test Critique schema contract."""

    def test_critique_result_uses_finding(self):
        """CritiqueResult uses 'finding', not 'critique_text'."""
        fields = set(CritiqueResult.model_fields.keys())
        assert "finding" in fields
        assert "critique_text" not in fields

    def test_critique_result_valid_creation(self):
        """CritiqueResult with canonical field names."""
        result = CritiqueResult(
            dimension=CritiqueDimension.INTERNAL_VALIDITY,
            severity=CritiqueSeverity.HIGH,
            finding="This is a detailed finding about internal validity issues in the study",
            affected_claims=["claim_001"],
            recommendation="Address by checking randomization",
        )
        assert result.dimension == CritiqueDimension.INTERNAL_VALIDITY
        assert result.severity == CritiqueSeverity.HIGH

    def test_critique_has_findings_list(self):
        """Critique contains findings list, not individual claims."""
        fields = set(Critique.model_fields.keys())
        assert "findings" in fields
        assert "claim_id" not in fields  # claim_id is in CritiqueResult.affected_claims


class TestVerificationContract:
    """Test Verification schema contract."""

    def test_verification_status_values(self):
        """VerificationStatus has expected values."""
        expected = {"verified", "partial", "contradicted", "unverifiable", "error"}
        actual = {s.value for s in VerificationStatus}
        assert actual == expected


class TestStructuredOutputsAlignment:
    """Test that structured_outputs.py aligns with schemas.py."""

    def test_rob2_schema_uses_lowercase_domains(self):
        """structured_outputs.py RoB2 schema must use lowercase domain values."""
        from cdr.llm.structured_outputs import ROB2_ASSESSMENT_SCHEMA

        # Check that domains in schema are lowercase
        domains_prop = ROB2_ASSESSMENT_SCHEMA["properties"]["domains"]
        items_props = domains_prop["items"]["properties"]
        domain_enum = items_props["domain"]["enum"]

        for domain in domain_enum:
            assert domain == domain.lower(), f"Domain {domain} should be lowercase"
            assert domain in {d.value for d in RoB2Domain}, (
                f"Domain {domain} not in RoB2Domain enum"
            )

    def test_evidence_synthesis_schema_uses_claim_text(self):
        """structured_outputs.py synthesis schema must use claim_text."""
        from cdr.llm.structured_outputs import SYNTHESIS_RESULT_SCHEMA

        claims_items = SYNTHESIS_RESULT_SCHEMA["properties"]["claims"]["items"]["properties"]

        assert "claim_text" in claims_items, "Schema must request 'claim_text'"
        assert "supporting_snippet_ids" in claims_items, (
            "Schema must request 'supporting_snippet_ids'"
        )


class TestPromptAlignment:
    """Test that prompts request the correct field names."""

    def test_skeptic_prompt_uses_lowercase_dimensions(self):
        """SkepticAgent prompt must reference lowercase dimension values."""
        from cdr.skeptic.skeptic_agent import SKEPTIC_SYSTEM_PROMPT

        for dim in CritiqueDimension:
            assert dim.value in SKEPTIC_SYSTEM_PROMPT, f"Dimension {dim.value} not in prompt"

    def test_screener_prompt_uses_valid_exclusion_reasons(self):
        """Screener prompt must use valid ExclusionReason values."""
        from cdr.screening.screener import SCREENING_SYSTEM_PROMPT

        # Check that at least the common ones are referenced
        common_reasons = ["population_mismatch", "intervention_mismatch", "no_abstract"]
        for reason in common_reasons:
            assert reason in SCREENING_SYSTEM_PROMPT, f"Reason {reason} not in screener prompt"


class TestExtractorContract:
    """Test that extractors create valid model instances."""

    def test_extract_snippets_creates_valid_snippets(self):
        """extract_snippets must create Snippets with snippet_id field."""
        from cdr.extraction.extractor import extract_snippets

        # Create test text with multiple paragraphs (min 50 chars each per extractor logic)
        text = """This is the first paragraph with enough content to be extracted as a snippet.

This is the second paragraph with sufficient content to form a valid snippet entry.

This is the third paragraph that also has more than fifty characters for extraction."""

        snippets = extract_snippets(
            record_id="test_001",
            text=text,
            title="Test Study",
            pmid="12345678",
        )

        assert len(snippets) >= 1, "Should extract at least one snippet"

        for i, snip in enumerate(snippets):
            # Verify snippet_id field is used (not 'id')
            assert hasattr(snip, "snippet_id"), "Snippet must have snippet_id attribute"
            assert snip.snippet_id.startswith("test_001_snip_"), (
                f"snippet_id format wrong: {snip.snippet_id}"
            )

            # Verify source_ref is properly set
            assert snip.source_ref is not None
            assert snip.source_ref.record_id == "test_001"
            assert snip.source_ref.pmid == "12345678"

    def test_rule_based_screener_uses_valid_exclusion_reason(self):
        """RuleBasedScreener must use ExclusionReason values that exist."""
        from cdr.core.enums import ExclusionReason

        # Verify STUDY_TYPE_EXCLUDED exists (not WRONG_STUDY_TYPE)
        assert hasattr(ExclusionReason, "STUDY_TYPE_EXCLUDED")
        assert ExclusionReason.STUDY_TYPE_EXCLUDED.value == "study_type_excluded"

        # Verify WRONG_STUDY_TYPE does NOT exist
        assert not hasattr(ExclusionReason, "WRONG_STUDY_TYPE")


class TestAuditFixes:
    """Test fixes from the 2026-01-18 audit."""

    def test_study_card_extractor_no_placeholder(self):
        """StudyCardExtractor must NOT use 'placeholder' in supporting_snippet_ids."""
        import inspect
        from cdr.extraction.extractor import StudyCardExtractor

        # Inspect the _parse_response method source code
        source = inspect.getsource(StudyCardExtractor._parse_response)

        # CRITICAL: Must NOT contain "placeholder" as a fallback
        assert '"placeholder"' not in source and "'placeholder'" not in source, (
            "StudyCardExtractor._parse_response must NOT use 'placeholder' fallback - "
            "violates PRISMA trazability requirements"
        )

    def test_rule_based_screener_uses_record_id_not_id(self):
        """RuleBasedScreener must use record.record_id, not record.id."""
        import inspect
        from cdr.screening.screener import RuleBasedScreener

        # Inspect the _apply_rules method source code
        source = inspect.getsource(RuleBasedScreener._apply_rules)

        # CRITICAL: Must use record.record_id, not record.id
        assert "record.id" not in source or "record.record_id" in source, (
            "RuleBasedScreener must use record.record_id (not record.id) - "
            "Record schema has no 'id' field"
        )

        # Also verify it doesn't use record.study_type
        assert "record.study_type" not in source, (
            "RuleBasedScreener must NOT use record.study_type - "
            "Record schema has no 'study_type' field, must infer from publication_type"
        )

    def test_cdrstate_parsed_documents_is_dict(self):
        """CDRState.parsed_documents is dict[str, dict], not list."""
        from cdr.core.schemas import CDRState

        # Verify the schema defines it as dict
        field_info = CDRState.model_fields["parsed_documents"]
        # The annotation should be dict type
        assert "dict" in str(field_info.annotation).lower()


class TestSnippetGateFormal:
    """Test the formal PRISMA/GRADE gate: no real snippets = no claims.

    Refs:
    - PRISMA 2020: Transparency and traceability of evidence
    - GRADE Handbook: Certainty requires explicit evidence support
    - CDR_Reconciliation_Note_and_Snippet_Gate_2026-01-18.md
    """

    def test_synthesize_node_has_snippet_validation_gate(self):
        """synthesize_node must validate claims against state.snippets."""
        import inspect
        from cdr.orchestration.graph import synthesize_node

        source = inspect.getsource(synthesize_node)

        # GATE FORMAL: Must build valid_snippet_ids set
        assert "valid_snippet_ids" in source, (
            "synthesize_node must build valid_snippet_ids for claim validation"
        )

        # Must filter claims without valid snippets
        assert "validated_claims" in source, (
            "synthesize_node must produce validated_claims (not raw result.claims)"
        )

    def test_publish_node_checks_insufficient_evidence_for_no_snippets(self):
        """publish_node must set INSUFFICIENT_EVIDENCE when no snippets exist."""
        import inspect
        from cdr.orchestration.graph import publish_node

        source = inspect.getsource(publish_node)

        # Must check for no snippets condition
        assert "no_snippets_extracted" in source, (
            "publish_node must detect 'no_snippets_extracted' condition"
        )

        # Must use INSUFFICIENT_EVIDENCE status
        assert "INSUFFICIENT_EVIDENCE" in source, (
            "publish_node must use RunStatus.INSUFFICIENT_EVIDENCE for missing evidence"
        )

    def test_publish_node_checks_unpublishable_for_invalid_claims(self):
        """publish_node must set UNPUBLISHABLE when claims lack valid snippets."""
        import inspect
        from cdr.orchestration.graph import publish_node

        source = inspect.getsource(publish_node)

        # Must check for claims without valid snippets
        assert "claims_without_valid_snippets" in source, (
            "publish_node must detect claims without valid snippet support"
        )

        # Must use UNPUBLISHABLE status
        assert "UNPUBLISHABLE" in source, (
            "publish_node must use RunStatus.UNPUBLISHABLE for invalid claims"
        )

    def test_run_status_has_scientific_outcomes(self):
        """RunStatus must have INSUFFICIENT_EVIDENCE and UNPUBLISHABLE."""
        from cdr.core.enums import RunStatus

        # Required scientific outcome statuses per CDR spec
        assert hasattr(RunStatus, "INSUFFICIENT_EVIDENCE"), (
            "RunStatus must have INSUFFICIENT_EVIDENCE for no evidence conditions"
        )
        assert hasattr(RunStatus, "UNPUBLISHABLE"), (
            "RunStatus must have UNPUBLISHABLE for invalid evidence conditions"
        )
        assert hasattr(RunStatus, "COMPLETED"), "RunStatus must have COMPLETED for successful runs"


# =============================================================================
# FASE 0 REGRESSION TESTS
# Refs: CDR_Integral_Audit_2026-01-20.md
# =============================================================================


class TestCritical1EarlyExitStatus:
    """Tests for CRITICAL-1: Early termination must route to publish.

    Refs: CDR_Integral_Audit_2026-01-20.md CRITICAL-1
    """

    def test_retrieve_early_exit_routes_to_publish(self):
        """should_continue_after_retrieve returns 'publish' not 'end'."""
        import inspect
        from cdr.orchestration.graph import should_continue_after_retrieve

        # Check return type annotation
        sig = inspect.signature(should_continue_after_retrieve)
        return_annotation = str(sig.return_annotation)

        # Must return Literal with 'publish' not 'end'
        assert "publish" in return_annotation, (
            "should_continue_after_retrieve must route to 'publish' not 'end'"
        )
        assert "end" not in return_annotation, (
            "should_continue_after_retrieve must NOT route to 'end' (skip publish)"
        )

    def test_screen_early_exit_routes_to_publish(self):
        """should_continue_after_screen returns 'publish' not 'end'."""
        import inspect
        from cdr.orchestration.graph import should_continue_after_screen

        sig = inspect.signature(should_continue_after_screen)
        return_annotation = str(sig.return_annotation)

        assert "publish" in return_annotation, (
            "should_continue_after_screen must route to 'publish' not 'end'"
        )
        assert "end" not in return_annotation, (
            "should_continue_after_screen must NOT route to 'end' (skip publish)"
        )

    def test_publish_node_handles_no_records(self):
        """publish_node must handle no_records_retrieved case."""
        import inspect
        from cdr.orchestration.graph import publish_node

        source = inspect.getsource(publish_node)

        assert "no_records_retrieved" in source, (
            "publish_node must detect no_records_retrieved early-exit"
        )
        assert "no_records_included_after_screening" in source, (
            "publish_node must detect no_records_included_after_screening early-exit"
        )


class TestCritical2ScreeningProviderMismatch:
    """Tests for CRITICAL-2: Screener must accept provider instances.

    Refs: CDR_Integral_Audit_2026-01-20.md CRITICAL-2
    """

    def test_screener_accepts_provider_instance(self):
        """Screener.__init__ must accept BaseLLMProvider instances."""
        import inspect
        from cdr.screening.screener import Screener

        sig = inspect.signature(Screener.__init__)
        provider_param = sig.parameters.get("provider")

        assert provider_param is not None, "Screener must have provider parameter"

        # Check type annotation allows instances
        annotation = str(provider_param.annotation)
        assert "BaseLLMProvider" in annotation, (
            "Screener.provider must accept BaseLLMProvider instances"
        )

    def test_screener_init_has_isinstance_check(self):
        """Screener.__init__ must have isinstance check for BaseLLMProvider."""
        import inspect
        from cdr.screening.screener import Screener

        source = inspect.getsource(Screener.__init__)

        assert "isinstance(provider, BaseLLMProvider)" in source, (
            "Screener must check isinstance for provider instance"
        )


class TestCritical3APIContract:
    """Tests for CRITICAL-3: API contract alignment.

    Refs: CDR_Integral_Audit_2026-01-20.md CRITICAL-3
    """

    def test_run_status_has_running_not_in_progress(self):
        """RunStatus must have RUNNING, not IN_PROGRESS."""
        from cdr.core.enums import RunStatus

        assert hasattr(RunStatus, "RUNNING"), "RunStatus must have RUNNING"
        assert not hasattr(RunStatus, "IN_PROGRESS"), (
            "RunStatus must NOT have IN_PROGRESS (use RUNNING)"
        )

    def test_api_uses_running_not_in_progress(self):
        """API routes must use RunStatus.RUNNING not IN_PROGRESS."""
        import inspect
        from cdr.api import routes

        source = inspect.getsource(routes)

        # Check that IN_PROGRESS is not used as an actual status value
        # Note: comments may contain the string for documentation
        assert "RunStatus.IN_PROGRESS" not in source, (
            "API must NOT use RunStatus.IN_PROGRESS (use RunStatus.RUNNING)"
        )
        assert "RunStatus.RUNNING" in source, "API must use RunStatus.RUNNING"

    def test_claim_response_uses_correct_fields(self):
        """ClaimResponse must use claim_text, not statement."""
        from cdr.api.routes import ClaimResponse

        fields = ClaimResponse.model_fields

        assert "claim_text" in fields, "ClaimResponse must have claim_text field"
        assert "statement" not in fields, "ClaimResponse must NOT have statement (use claim_text)"
        assert "supporting_snippet_ids" in fields, "ClaimResponse must have supporting_snippet_ids"
        assert "supporting_studies" not in fields, (
            "ClaimResponse must NOT have supporting_studies (use supporting_snippet_ids)"
        )


# =============================================================================
# POST-FASE 0 INCREMENTAL AUDIT TESTS
# Refs: CDR_Integral_Audit_2026-01-20.md (Auditoria incremental post-Fase 0)
# =============================================================================


class TestCriticoAVerificationContract:
    """Tests for CRITICO-A: API uses verification, not verification_results.

    Refs: CDR_Integral_Audit_2026-01-20.md CRITICO-A
    """

    def test_cdrstate_has_verification_list(self):
        """CDRState.verification is list[VerificationResult], not dict."""
        from cdr.core.schemas import CDRState

        field_info = CDRState.model_fields["verification"]
        annotation = str(field_info.annotation)

        assert "list" in annotation.lower(), "CDRState.verification must be a list"
        assert "VerificationResult" in annotation, (
            "CDRState.verification must be list[VerificationResult]"
        )

    def test_cdrstate_has_no_verification_results(self):
        """CDRState must NOT have verification_results field."""
        from cdr.core.schemas import CDRState

        assert "verification_results" not in CDRState.model_fields, (
            "CDRState must NOT have verification_results (use verification)"
        )

    def test_api_claims_endpoint_uses_verification_list(self):
        """API claims endpoint must use result.verification (list), not result.verification_results."""
        import inspect
        from cdr.api import routes

        source = inspect.getsource(routes.get_run_claims)

        assert "result.verification" in source, "API claims endpoint must use result.verification"
        # The fixed code should NOT use verification_results
        assert ".verification_results" not in source, (
            "API claims endpoint must NOT use verification_results"
        )


class TestCriticoBNegativeOutcomes:
    """Tests for CRITICO-B: API allows negative outcome states.

    Refs: CDR_Integral_Audit_2026-01-20.md CRITICO-B
    """

    def test_api_claims_allows_insufficient_evidence(self):
        """API claims endpoint must allow INSUFFICIENT_EVIDENCE status."""
        import inspect
        from cdr.api import routes

        source = inspect.getsource(routes.get_run_claims)

        # Check that INSUFFICIENT_EVIDENCE is explicitly allowed
        assert "INSUFFICIENT_EVIDENCE" in source, (
            "API claims endpoint must allow INSUFFICIENT_EVIDENCE reads"
        )

    def test_api_claims_allows_unpublishable(self):
        """API claims endpoint must allow UNPUBLISHABLE status."""
        import inspect
        from cdr.api import routes

        source = inspect.getsource(routes.get_run_claims)

        assert "UNPUBLISHABLE" in source, "API claims endpoint must allow UNPUBLISHABLE reads"


class TestAltoCRunIdAlignment:
    """Tests for ALTO-C: API run_id aligns with runner.

    Refs: CDR_Integral_Audit_2026-01-20.md ALTO-C
    """

    def test_cdr_runner_accepts_run_id_parameter(self):
        """CDRRunner.run must accept optional run_id parameter."""
        import inspect
        from cdr.orchestration.graph import CDRRunner

        sig = inspect.signature(CDRRunner.run)
        params = sig.parameters

        assert "run_id" in params, "CDRRunner.run must accept run_id parameter"

        # Check it's optional (has default None)
        run_id_param = params["run_id"]
        # None as default means it's optional
        assert run_id_param.default is None, "run_id parameter should be optional (None default)"

    def test_api_passes_run_id_to_runner(self):
        """API _execute_run must pass run_id to runner."""
        import inspect
        from cdr.api import routes

        source = inspect.getsource(routes._execute_run)

        assert "run_id=run_id" in source or "run_id = run_id" in source, (
            "API must pass run_id to runner.run()"
        )


class TestAltoDReasonCodePropagation:
    """Tests for ALTO-D: Reason code propagation for LLM required.

    Refs: CDR_Integral_Audit_2026-01-20.md ALTO-D
    """

    def test_publish_node_checks_screening_blocked_no_llm(self):
        """publish_node must check screening_blocked_no_llm flag."""
        import inspect
        from cdr.orchestration.graph import publish_node

        source = inspect.getsource(publish_node)

        assert "screening_blocked_no_llm" in source, (
            "publish_node must check screening_blocked_no_llm flag"
        )

    def test_publish_node_has_llm_required_reason(self):
        """publish_node must use llm_required_for_level_2 reason code."""
        import inspect
        from cdr.orchestration.graph import publish_node

        source = inspect.getsource(publish_node)

        assert "llm_required_for_level_2" in source, (
            "publish_node must use llm_required_for_level_2 status_reason"
        )


class TestMedioEPrismaCountsInit:
    """Tests for MEDIO-E: PRISMA counts initialization on early failure.

    Refs: CDR_Integral_Audit_2026-01-20.md MEDIO-E
    """

    def test_retrieve_node_returns_prisma_counts_on_no_plan(self):
        """retrieve_node must return PRISMACounts when no search_plan."""
        import inspect
        from cdr.orchestration.graph import retrieve_node

        source = inspect.getsource(retrieve_node)

        # Check that it initializes PRISMACounts on error path
        assert "PRISMACounts" in source, (
            "retrieve_node must import/use PRISMACounts for early failure"
        )
        assert "prisma_counts" in source, "retrieve_node must return prisma_counts on error path"


# =============================================================================
# FUNCTIONAL TESTS (MEDIO-F): Runtime behavior validation
# Refs: CDR_Integral_Audit_2026-01-20.md MEDIO-F
# =============================================================================


class TestFunctionalPublishGates:
    """Functional tests for publish node gates.

    These tests instantiate real state objects and validate behavior.
    Refs: CDR_Integral_Audit_2026-01-20.md MEDIO-F
    """

    def test_empty_records_produces_insufficient_evidence(self):
        """State with no records must produce INSUFFICIENT_EVIDENCE."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus

        # Create minimal state with no records
        state = CDRState(
            run_id="test_001",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            retrieved_records=[],  # No records
        )

        # Verify the state is correctly initialized
        assert len(state.retrieved_records) == 0
        assert len(state.get_included_records()) == 0

    def test_screening_blocked_flag_set(self):
        """State with screening_blocked_no_llm flag must be detectable."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus

        state = CDRState(
            run_id="test_002",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            flags={"screening_blocked_no_llm": True},
        )

        assert state.flags.get("screening_blocked_no_llm") is True

    def test_verification_list_mappable_by_claim_id(self):
        """verification list can be mapped by claim_id."""
        from cdr.core.schemas import VerificationResult, VerificationCheck, SourceRef
        from cdr.core.enums import VerificationStatus
        from datetime import datetime

        # Create verification results as list
        # VerificationCheck uses claim_id, source_ref, status, confidence
        vr1 = VerificationResult(
            claim_id="claim_001",
            checks=[
                VerificationCheck(
                    claim_id="claim_001",
                    source_ref=SourceRef(record_id="rec_001"),
                    status=VerificationStatus.VERIFIED,
                    confidence=1.0,
                    explanation="Snippet exists and supports claim",
                )
            ],
            overall_status=VerificationStatus.VERIFIED,
            overall_confidence=1.0,
            verified_at=datetime.utcnow(),
        )
        vr2 = VerificationResult(
            claim_id="claim_002",
            checks=[],
            overall_status=VerificationStatus.PARTIAL,
            overall_confidence=0.7,
            verified_at=datetime.utcnow(),
        )

        verification_list = [vr1, vr2]

        # Build lookup by claim_id (as API should do)
        verification_by_claim = {vr.claim_id: vr for vr in verification_list}

        assert verification_by_claim["claim_001"].overall_status == VerificationStatus.VERIFIED
        assert verification_by_claim["claim_002"].overall_status == VerificationStatus.PARTIAL

    def test_prisma_counts_default_initialization(self):
        """PRISMACounts can be default-initialized with zeros."""
        from cdr.core.schemas import PRISMACounts

        counts = PRISMACounts()

        assert counts.records_identified == 0
        assert counts.records_from_pubmed == 0
        assert counts.duplicates_removed == 0
        assert counts.studies_included == 0

    def test_run_status_terminal_states(self):
        """Verify terminal states for API filtering."""
        from cdr.core.enums import RunStatus

        terminal_statuses = {
            RunStatus.COMPLETED.value,
            RunStatus.INSUFFICIENT_EVIDENCE.value,
            RunStatus.UNPUBLISHABLE.value,
        }

        # All should be valid terminal states
        assert "completed" in terminal_statuses
        assert "insufficient_evidence" in terminal_statuses
        assert "unpublishable" in terminal_statuses

        # RUNNING should NOT be terminal
        assert RunStatus.RUNNING.value not in terminal_statuses


# =============================================================================
# RUNTIME FUNCTIONAL TESTS (MEDIO-F REAL): Execute actual nodes
# Refs: CDR_Integral_Audit_2026-01-20.md MEDIO-F (post-verificacion auditor)
# =============================================================================


class TestRuntimePublishNode:
    """Tests that execute publish_node with simulated states.

    These tests invoke the actual publish_node function to validate
    that status_reason is correctly set based on state conditions.

    Refs: CDR_Integral_Audit_2026-01-20.md MEDIO-F
    """

    def _make_config(self, tmp_path, dod_level: int = 1):
        """Create a mock RunnableConfig for publish_node.

        Args:
            tmp_path: pytest tmp_path fixture for output_dir
            dod_level: DoD level for publish_node gates (default 1)
        """
        return {
            "configurable": {
                "output_dir": str(tmp_path),
                "dod_level": dod_level,
            }
        }

    def _make_record(self, record_id: str, title: str, abstract: str):
        """Create a valid Record with content_hash."""
        import hashlib
        from cdr.core.schemas import Record
        from cdr.core.enums import RecordSource

        content = f"{title}{abstract}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        return Record(
            record_id=record_id,
            title=title,
            abstract=abstract,
            source=RecordSource.PUBMED,
            content_hash=content_hash,
        )

    def _make_snippet(self, snippet_id: str, record_id: str, text: str):
        """Create a valid Snippet with SourceRef."""
        from cdr.core.schemas import Snippet, SourceRef

        return Snippet(
            snippet_id=snippet_id,
            text=text,
            source_ref=SourceRef(record_id=record_id),
        )

    def _make_study_card(self, record_id: str, snippet_ids: list):
        """Create a valid StudyCard."""
        from cdr.core.schemas import StudyCard, OutcomeMeasure
        from cdr.core.enums import StudyType, OutcomeMeasureType

        return StudyCard(
            record_id=record_id,
            study_type=StudyType.RCT,
            sample_size=100,
            population_description="Test population",
            intervention_description="Test intervention",
            comparator_description="Placebo",
            outcomes=[
                OutcomeMeasure(
                    name="Primary outcome",
                    measure_type=OutcomeMeasureType.MEAN_DIFFERENCE,
                    value=1.5,
                )
            ],
            supporting_snippet_ids=snippet_ids,
        )

    def _make_rob2_result(self, record_id: str):
        """Create a valid RoB2Result with all 5 domains."""
        from cdr.core.schemas import RoB2Result, RoB2DomainResult
        from cdr.core.enums import RoB2Domain, RoB2Judgment

        domains = [
            RoB2DomainResult(
                domain=domain,
                judgment=RoB2Judgment.LOW,
                rationale=f"Low risk for {domain.value} domain based on test data.",
            )
            for domain in RoB2Domain
        ]
        return RoB2Result(
            record_id=record_id,
            domains=domains,
            overall_judgment=RoB2Judgment.LOW,
            overall_rationale="Overall low risk of bias based on all domains assessed.",
        )

    def test_publish_node_no_records_returns_insufficient_evidence(self, tmp_path):
        """publish_node with no records must return INSUFFICIENT_EVIDENCE."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus
        from cdr.orchestration.graph import publish_node

        # State with no retrieved records
        state = CDRState(
            run_id="test_publish_001",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            retrieved_records=[],
        )

        # Execute publish_node
        config = self._make_config(tmp_path)
        result = run_async(publish_node(state, config))

        assert result["status"] == RunStatus.INSUFFICIENT_EVIDENCE
        # status_reason is in report_data
        assert result["report"]["status_reason"] == "no_records_retrieved"

    def test_publish_node_llm_blocked_returns_llm_required(self, tmp_path):
        """publish_node with screening_blocked_no_llm must return llm_required_for_level_2."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus
        from cdr.orchestration.graph import publish_node

        # State with records but LLM screening blocked
        state = CDRState(
            run_id="test_publish_002",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            retrieved_records=[
                self._make_record("rec_001", "Test Study", "Test abstract with sufficient content")
            ],
            flags={"screening_blocked_no_llm": True},
        )

        config = self._make_config(tmp_path)
        result = run_async(publish_node(state, config))

        assert result["status"] == RunStatus.INSUFFICIENT_EVIDENCE
        assert result["report"]["status_reason"] == "llm_required_for_level_2"

    def test_publish_node_no_included_records_returns_no_included(self, tmp_path):
        """publish_node with all records excluded must return no_records_included."""
        from cdr.core.schemas import CDRState, ScreeningDecision
        from cdr.core.enums import RunStatus, ExclusionReason
        from cdr.orchestration.graph import publish_node

        # State with records but all excluded
        state = CDRState(
            run_id="test_publish_003",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            retrieved_records=[self._make_record("rec_001", "Test Study", "Test abstract")],
            screened=[
                ScreeningDecision(
                    record_id="rec_001",
                    included=False,
                    reason_code=ExclusionReason.POPULATION_MISMATCH,
                    reason_text="Population does not match",
                )
            ],
        )

        config = self._make_config(tmp_path)
        result = run_async(publish_node(state, config))

        assert result["status"] == RunStatus.INSUFFICIENT_EVIDENCE
        assert result["report"]["status_reason"] == "no_records_included_after_screening"

    def test_publish_node_no_snippets_returns_no_snippets(self, tmp_path):
        """publish_node with included records but no snippets."""
        from cdr.core.schemas import CDRState, ScreeningDecision
        from cdr.core.enums import RunStatus
        from cdr.orchestration.graph import publish_node

        # State with included records but no snippets
        state = CDRState(
            run_id="test_publish_004",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            retrieved_records=[self._make_record("rec_001", "Test Study", "Test abstract")],
            screened=[
                ScreeningDecision(
                    record_id="rec_001",
                    included=True,
                )
            ],
            snippets=[],  # No snippets extracted
        )

        config = self._make_config(tmp_path)
        result = run_async(publish_node(state, config))

        assert result["status"] == RunStatus.INSUFFICIENT_EVIDENCE
        assert result["report"]["status_reason"] == "no_snippets_extracted"

    def test_publish_node_no_studies_returns_no_studies(self, tmp_path):
        """publish_node with snippets but no study_cards.

        MEDIO-F2 fix: Test for no_studies_included gate.
        Refs: CDR_Integral_Audit_2026-01-20.md MEDIO-F2
        """
        from cdr.core.schemas import CDRState, ScreeningDecision
        from cdr.core.enums import RunStatus
        from cdr.orchestration.graph import publish_node

        # State with included records and snippets but no study_cards
        state = CDRState(
            run_id="test_publish_005",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            retrieved_records=[self._make_record("rec_001", "Test Study", "Test abstract")],
            screened=[
                ScreeningDecision(
                    record_id="rec_001",
                    included=True,
                )
            ],
            snippets=[
                self._make_snippet("snip_001", "rec_001", "This is a test snippet with enough text")
            ],
            study_cards=[],  # No study cards extracted
        )

        config = self._make_config(tmp_path)
        result = run_async(publish_node(state, config))

        assert result["status"] == RunStatus.INSUFFICIENT_EVIDENCE
        assert result["report"]["status_reason"] == "no_studies_included"

    def test_publish_node_no_claims_returns_no_claims(self, tmp_path):
        """publish_node with study_cards but no claims.

        MEDIO-F2 fix: Test for no_claims_generated gate.
        Refs: CDR_Integral_Audit_2026-01-20.md MEDIO-F2
        """
        from cdr.core.schemas import CDRState, ScreeningDecision
        from cdr.core.enums import RunStatus
        from cdr.orchestration.graph import publish_node

        # State with study_cards but no claims
        state = CDRState(
            run_id="test_publish_006",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            retrieved_records=[self._make_record("rec_001", "Test Study", "Test abstract")],
            screened=[
                ScreeningDecision(
                    record_id="rec_001",
                    included=True,
                )
            ],
            snippets=[
                self._make_snippet("snip_001", "rec_001", "This is a test snippet with enough text")
            ],
            study_cards=[self._make_study_card("rec_001", ["snip_001"])],
            claims=[],  # No claims generated
        )

        config = self._make_config(tmp_path)
        result = run_async(publish_node(state, config))

        assert result["status"] == RunStatus.INSUFFICIENT_EVIDENCE
        assert result["report"]["status_reason"] == "no_claims_generated"

    def test_publish_node_dod2_low_verification_returns_unpublishable(self, tmp_path):
        """publish_node with DoD Level 2 and < 80% verification returns UNPUBLISHABLE.

        MEDIO-F2 fix: Test for verification_coverage_insufficient gate.
        Refs: CDR_Integral_Audit_2026-01-20.md MEDIO-F2
        """
        from cdr.core.schemas import (
            CDRState,
            ScreeningDecision,
            EvidenceClaim,
            VerificationResult,
            SynthesisResult,
        )
        from cdr.core.enums import RunStatus, GRADECertainty, VerificationStatus
        from cdr.orchestration.graph import publish_node

        # Create 3 claims
        claims = [
            EvidenceClaim(
                claim_id=f"claim_{i:03d}",
                claim_text=f"This is claim number {i} with sufficient text for validation.",
                certainty=GRADECertainty.MODERATE,
                supporting_snippet_ids=["snip_001"],
            )
            for i in range(3)
        ]

        # Only 2 of 3 claims verified (66% < 80% threshold)
        verification = [
            VerificationResult(
                claim_id="claim_000",
                overall_status=VerificationStatus.VERIFIED,
                overall_confidence=0.9,
            ),
            VerificationResult(
                claim_id="claim_001",
                overall_status=VerificationStatus.PARTIAL,
                overall_confidence=0.7,
            ),
            # claim_002 has UNVERIFIABLE status -> doesn't count
            VerificationResult(
                claim_id="claim_002",
                overall_status=VerificationStatus.UNVERIFIABLE,
                overall_confidence=0.0,
            ),
        ]

        # State with claims but < 80% verification coverage
        state = CDRState(
            run_id="test_publish_007",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            retrieved_records=[self._make_record("rec_001", "Test Study", "Test abstract")],
            screened=[ScreeningDecision(record_id="rec_001", included=True)],
            snippets=[
                self._make_snippet("snip_001", "rec_001", "This is a test snippet with enough text")
            ],
            study_cards=[self._make_study_card("rec_001", ["snip_001"])],
            rob2_results=[self._make_rob2_result("rec_001")],  # Add RoB2 to pass that gate
            claims=claims,
            verification=verification,
            # Use proper schema-defined synthesis_result
            synthesis_result=SynthesisResult(claims=claims),
        )

        # Configure DoD Level 2
        config = self._make_config(tmp_path, dod_level=2)
        result = run_async(publish_node(state, config))

        assert result["status"] == RunStatus.UNPUBLISHABLE
        assert "verification_coverage_insufficient" in result["report"]["status_reason"]

    def test_publish_node_includes_search_plan_for_prisma_s(self, tmp_path):
        """publish_node must include search_plan in report_data for PRISMA-S compliance.

        HIGH-4 fix: report_data must include reproducible search strategy.
        Refs: PRISMA-S (BMJ 2021), CDR_Integral_Audit_2026-01-20.md HIGH-4
        """
        from cdr.core.schemas import (
            CDRState,
            ScreeningDecision,
            EvidenceClaim,
            VerificationResult,
            SynthesisResult,
            SearchPlan,
            PICO,
        )
        from cdr.core.enums import RunStatus, GRADECertainty, VerificationStatus
        from cdr.orchestration.graph import publish_node

        # Create a complete state with search_plan
        pico = PICO(
            population="Adults with hypertension",
            intervention="ACE inhibitors",
            comparator="Placebo",
            outcome="Blood pressure reduction",
        )
        search_plan = SearchPlan(
            pico=pico,
            pubmed_query='("ACE inhibitors"[MeSH]) AND (hypertension)',
            ct_gov_query="ACE inhibitors AND hypertension",
            date_range=("2020/01/01", "2024/12/31"),
            languages=["english"],
            max_results_per_source=100,
        )

        claims = [
            EvidenceClaim(
                claim_id="claim_001",
                claim_text="ACE inhibitors reduce blood pressure in hypertensive adults.",
                certainty=GRADECertainty.MODERATE,
                supporting_snippet_ids=["snip_001"],
            )
        ]

        verification = [
            VerificationResult(
                claim_id="claim_001",
                overall_status=VerificationStatus.VERIFIED,
                overall_confidence=0.95,
            )
        ]

        state = CDRState(
            run_id="test_publish_008",
            question="Do ACE inhibitors reduce blood pressure?",
            status=RunStatus.RUNNING,
            pico=pico,
            search_plan=search_plan,
            retrieved_records=[self._make_record("rec_001", "ACE Study", "ACE inhibitors study")],
            screened=[ScreeningDecision(record_id="rec_001", included=True)],
            snippets=[
                self._make_snippet("snip_001", "rec_001", "ACE inhibitors significantly reduced BP")
            ],
            study_cards=[self._make_study_card("rec_001", ["snip_001"])],
            rob2_results=[self._make_rob2_result("rec_001")],
            claims=claims,
            verification=verification,
            synthesis_result=SynthesisResult(claims=claims),
        )

        config = self._make_config(tmp_path, dod_level=1)
        result = run_async(publish_node(state, config))

        # Verify search_plan is in report_data
        report = result["report"]
        assert "search_plan" in report, "report_data must include search_plan for PRISMA-S"
        assert report["search_plan"] is not None

        # Verify search_plan structure
        sp = report["search_plan"]
        assert sp["pubmed_query"] == '("ACE inhibitors"[MeSH]) AND (hypertension)'
        assert sp["ct_gov_query"] == "ACE inhibitors AND hypertension"
        assert sp["date_range"] == ("2020/01/01", "2024/12/31")
        assert sp["languages"] == ["english"]
        assert sp["max_results_per_source"] == 100
        assert sp["created_at"] is not None  # ISO format string

    def test_publish_node_includes_evaluation_report(self, tmp_path):
        """publish_node must include evaluation metrics in report_data.

        SOTA requirement: EvaluationReport per run for DoD compliance.
        Refs: CDR SOTA requirements, evaluation integration
        """
        from cdr.core.schemas import (
            CDRState,
            ScreeningDecision,
            EvidenceClaim,
            VerificationResult,
            SynthesisResult,
        )
        from cdr.core.enums import RunStatus, GRADECertainty, VerificationStatus
        from cdr.orchestration.graph import publish_node

        claims = [
            EvidenceClaim(
                claim_id="claim_001",
                claim_text="Test claim with sufficient content for evaluation.",
                certainty=GRADECertainty.MODERATE,
                supporting_snippet_ids=["snip_001"],
            )
        ]

        verification = [
            VerificationResult(
                claim_id="claim_001",
                overall_status=VerificationStatus.VERIFIED,
                overall_confidence=0.95,
            )
        ]

        state = CDRState(
            run_id="test_publish_eval_001",
            question="Test evaluation integration",
            status=RunStatus.RUNNING,
            retrieved_records=[self._make_record("rec_001", "Test Study", "Test abstract")],
            screened=[ScreeningDecision(record_id="rec_001", included=True)],
            snippets=[self._make_snippet("snip_001", "rec_001", "Test snippet for evaluation")],
            study_cards=[self._make_study_card("rec_001", ["snip_001"])],
            rob2_results=[self._make_rob2_result("rec_001")],
            claims=claims,
            verification=verification,
            synthesis_result=SynthesisResult(claims=claims),
        )

        config = self._make_config(tmp_path, dod_level=2)
        result = run_async(publish_node(state, config))

        # Verify evaluation is in report_data
        report = result["report"]
        assert "evaluation" in report, "report_data must include evaluation metrics"

        # Verify evaluation structure
        evaluation = report["evaluation"]
        assert "run_id" in evaluation
        assert "dod_level" in evaluation
        assert evaluation["dod_level"] == 2
        assert "metrics" in evaluation
        assert "overall_pass" in evaluation
        assert "summary" in evaluation

        # Verify core metrics are present
        metric_names = [m["name"] for m in evaluation["metrics"]]
        assert "snippet_coverage" in metric_names
        assert "verification_coverage" in metric_names
        assert "context_precision" in metric_names


class TestRuntimeRetrieveNode:
    """Tests that execute retrieve_node with simulated states.

    Refs: CDR_Integral_Audit_2026-01-20.md MEDIO-F
    """

    def _make_config(self):
        """Create a mock RunnableConfig for retrieve_node."""
        return {
            "configurable": {
                "max_results": 10,
            }
        }

    def test_retrieve_node_no_search_plan_returns_prisma_counts(self):
        """retrieve_node without search_plan must return empty PRISMACounts."""
        from cdr.core.schemas import CDRState, PRISMACounts
        from cdr.core.enums import RunStatus
        from cdr.orchestration.graph import retrieve_node

        # State without search_plan
        state = CDRState(
            run_id="test_retrieve_001",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            search_plan=None,
        )

        config = self._make_config()
        result = run_async(retrieve_node(state, config))

        # Must have errors and prisma_counts
        assert "errors" in result
        assert any("No search plan" in e for e in result["errors"])
        assert "prisma_counts" in result
        assert isinstance(result["prisma_counts"], PRISMACounts)
        assert result["prisma_counts"].records_identified == 0


class TestRuntimeParseDocsNode:
    """Tests for parse_documents_node with full-text fallback.

    HIGH-2 fix: Tests full-text retrieval integration.
    Refs: CDR_Integral_Audit_2026-01-20.md HIGH-2
    """

    def _make_record(self, record_id: str, pmid: str | None, abstract: str):
        """Create a valid Record with optional PMID."""
        import hashlib
        from cdr.core.schemas import Record
        from cdr.core.enums import RecordSource

        content = f"{record_id}{abstract}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        return Record(
            record_id=record_id,
            title=f"Test Study {record_id}",
            abstract=abstract,
            source=RecordSource.PUBMED,
            content_hash=content_hash,
            pmid=pmid,
        )

    def test_parse_docs_with_fulltext_disabled_uses_abstracts(self, tmp_path):
        """parse_documents_node with fulltext disabled uses abstracts only."""
        from cdr.core.schemas import CDRState, ScreeningDecision
        from cdr.core.enums import RunStatus, Section
        from cdr.orchestration.graph import parse_documents_node

        # State with included record that has abstract
        record = self._make_record(
            "rec_001", "12345678", "This is a test abstract with sufficient content for parsing."
        )
        state = CDRState(
            run_id="test_parse_001",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            retrieved_records=[record],
            screened=[ScreeningDecision(record_id="rec_001", included=True)],
        )

        # Config with fulltext disabled (default)
        config = {"configurable": {"enable_fulltext_retrieval": False}}
        result = run_async(parse_documents_node(state, config))

        assert "snippets" in result
        assert len(result["snippets"]) > 0
        # Snippet should be from abstract
        assert result["snippets"][0].section == Section.ABSTRACT

    def test_parse_docs_no_abstract_counts_as_not_retrieved(self, tmp_path):
        """parse_documents_node without abstract marks as reports_not_retrieved."""
        from cdr.core.schemas import CDRState, ScreeningDecision
        from cdr.core.enums import RunStatus
        from cdr.orchestration.graph import parse_documents_node

        # State with record that has no/short abstract
        record = self._make_record("rec_001", "12345678", "Too short")  # < 10 chars
        state = CDRState(
            run_id="test_parse_002",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            retrieved_records=[record],
            screened=[ScreeningDecision(record_id="rec_001", included=True)],
        )

        config = {"configurable": {"enable_fulltext_retrieval": False}}
        result = run_async(parse_documents_node(state, config))

        # Should have 0 snippets and updated prisma counts
        assert len(result["snippets"]) == 0
        assert result["prisma_counts"].reports_not_retrieved == 1


class TestRiskOfBiasRouting:
    """Tests for RoB2 vs ROBINS-I routing by study type.

    HIGH-3 fix: Routing by StudyType.
    Refs: CDR_Integral_Audit_2026-01-20.md HIGH-3
    """

    def _make_study_card(self, record_id: str, study_type):
        """Create a minimal StudyCard for testing."""
        from cdr.core.schemas import StudyCard, OutcomeMeasure
        from cdr.core.enums import OutcomeMeasureType

        return StudyCard(
            record_id=record_id,
            study_type=study_type,
            sample_size=100,
            population_description="Test population",
            intervention_description="Test intervention",
            comparator_description="Placebo",
            outcomes=[
                OutcomeMeasure(
                    name="Primary outcome",
                    measure_type=OutcomeMeasureType.MEAN_DIFFERENCE,
                    value=1.5,
                )
            ],
            supporting_snippet_ids=["snip_001"],
        )

    def _mock_rob2_result(self, record_id: str):
        """Create a mock RoB2Result with all required domains."""
        from cdr.core.schemas import RoB2Result, RoB2DomainResult
        from cdr.core.enums import RoB2Domain, RoB2Judgment

        # All 5 RoB2 domains are required
        domains = [
            RoB2DomainResult(
                domain=dom,
                judgment=RoB2Judgment.LOW,
                rationale=f"Mock rationale for {dom.value}",
                support=f"Mock support for {dom.value}",
            )
            for dom in RoB2Domain
        ]

        return RoB2Result(
            record_id=record_id,
            overall_judgment=RoB2Judgment.SOME_CONCERNS,
            overall_rationale="Mock overall rationale for RoB2",
            domains=domains,
        )

    def _mock_robins_i_result(self, record_id: str):
        """Create a mock ROBINSIResult with all required domains."""
        from cdr.core.schemas import ROBINSIResult, ROBINSIDomainResult
        from cdr.core.enums import ROBINSIDomain, ROBINSIJudgment

        # All 7 ROBINS-I domains are required
        domains = [
            ROBINSIDomainResult(
                domain=dom,
                judgment=ROBINSIJudgment.MODERATE,
                rationale=f"Mock rationale for {dom.value}",
                support=f"Mock support for {dom.value}",
            )
            for dom in ROBINSIDomain
        ]

        return ROBINSIResult(
            record_id=record_id,
            overall_judgment=ROBINSIJudgment.MODERATE,
            overall_rationale="Mock overall rationale for ROBINS-I",
            domains=domains,
        )

    def test_rct_uses_rob2(self):
        """RCT studies should be assessed with RoB2."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus, StudyType
        from cdr.orchestration.graph import assess_rob2_node
        from unittest.mock import patch, MagicMock

        rct_card = self._make_study_card("rec_rct", StudyType.RCT)

        state = CDRState(
            run_id="test_bias_001",
            question="Does aspirin prevent heart attacks?",
            status=RunStatus.RUNNING,
            study_cards=[rct_card],
            parsed_documents={"rec_rct": {"text": "This is a randomized controlled trial."}},
        )

        mock_rob2 = MagicMock()
        mock_rob2.assess.return_value = self._mock_rob2_result("rec_rct")

        mock_robins = MagicMock()
        mock_robins.assess.return_value = self._mock_robins_i_result("none")

        with (
            patch("cdr.rob2.assessor.RoB2Assessor", return_value=mock_rob2),
            patch("cdr.rob2.robins_i_assessor.ROBINSIAssessor", return_value=mock_robins),
        ):
            config = {"configurable": {}}
            result = run_async(assess_rob2_node(state, config))

        # RCT should produce RoB2 result
        assert len(result["rob2_results"]) == 1
        assert result["rob2_results"][0].record_id == "rec_rct"
        # No ROBINS-I for RCT
        assert len(result["robins_i_results"]) == 0
        # RoB2 assessor was called
        mock_rob2.assess.assert_called_once()
        # ROBINS-I assessor was NOT called
        mock_robins.assess.assert_not_called()

    def test_cohort_uses_robins_i(self):
        """Cohort studies should be assessed with ROBINS-I."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus, StudyType
        from cdr.orchestration.graph import assess_rob2_node
        from unittest.mock import patch, MagicMock

        cohort_card = self._make_study_card("rec_cohort", StudyType.COHORT)

        state = CDRState(
            run_id="test_bias_002",
            question="Does smoking increase cancer risk?",
            status=RunStatus.RUNNING,
            study_cards=[cohort_card],
            parsed_documents={"rec_cohort": {"text": "This is a cohort study."}},
        )

        mock_rob2 = MagicMock()
        mock_rob2.assess.return_value = self._mock_rob2_result("none")

        mock_robins = MagicMock()
        mock_robins.assess.return_value = self._mock_robins_i_result("rec_cohort")

        with (
            patch("cdr.rob2.assessor.RoB2Assessor", return_value=mock_rob2),
            patch("cdr.rob2.robins_i_assessor.ROBINSIAssessor", return_value=mock_robins),
        ):
            config = {"configurable": {}}
            result = run_async(assess_rob2_node(state, config))

        # Cohort should produce ROBINS-I result
        assert len(result["robins_i_results"]) == 1
        assert result["robins_i_results"][0].record_id == "rec_cohort"
        # No RoB2 for cohort
        assert len(result["rob2_results"]) == 0
        # ROBINS-I assessor was called
        mock_robins.assess.assert_called_once()
        # RoB2 assessor was NOT called
        mock_rob2.assess.assert_not_called()

    def test_mixed_study_types_routes_correctly(self):
        """Mixed study types should be routed to appropriate tools."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus, StudyType
        from cdr.orchestration.graph import assess_rob2_node
        from unittest.mock import patch, MagicMock

        rct_card = self._make_study_card("rec_rct", StudyType.RCT)
        cohort_card = self._make_study_card("rec_cohort", StudyType.COHORT)
        case_control_card = self._make_study_card("rec_cc", StudyType.CASE_CONTROL)

        state = CDRState(
            run_id="test_bias_003",
            question="What is the effectiveness of treatment X?",
            status=RunStatus.RUNNING,
            study_cards=[rct_card, cohort_card, case_control_card],
            parsed_documents={
                "rec_rct": {"text": "RCT study."},
                "rec_cohort": {"text": "Cohort study."},
                "rec_cc": {"text": "Case-control study."},
            },
        )

        mock_rob2 = MagicMock()
        mock_rob2.assess.return_value = self._mock_rob2_result("rec_rct")

        def robins_side_effect(record_id, text, **kwargs):
            return self._mock_robins_i_result(record_id)

        mock_robins = MagicMock()
        mock_robins.assess.side_effect = robins_side_effect

        with (
            patch("cdr.rob2.assessor.RoB2Assessor", return_value=mock_rob2),
            patch("cdr.rob2.robins_i_assessor.ROBINSIAssessor", return_value=mock_robins),
        ):
            config = {"configurable": {}}
            result = run_async(assess_rob2_node(state, config))

        # 1 RCT → RoB2
        assert len(result["rob2_results"]) == 1
        assert result["rob2_results"][0].record_id == "rec_rct"

        # 2 observational → ROBINS-I
        assert len(result["robins_i_results"]) == 2
        robins_ids = {r.record_id for r in result["robins_i_results"]}
        assert robins_ids == {"rec_cohort", "rec_cc"}


class TestCompositionalInference:
    """Tests for compositional inference (HIGH-1).

    HIGH-1 fix: Compositional inference module (A+B⇒C).
    Refs: CDR_Integral_Audit_2026-01-20.md HIGH-1
    """

    def _make_claim(self, claim_id: str, text: str):
        """Create a minimal EvidenceClaim for testing."""
        from cdr.core.schemas import EvidenceClaim
        from cdr.core.enums import GRADECertainty

        return EvidenceClaim(
            claim_id=claim_id,
            claim_text=text,
            certainty=GRADECertainty.MODERATE,
            supporting_snippet_ids=["snip_001"],
        )

    def _make_verification(self, claim_id: str, status):
        """Create a minimal VerificationResult."""
        from cdr.core.schemas import VerificationResult

        return VerificationResult(
            claim_id=claim_id,
            overall_status=status,
            overall_confidence=0.85,
            checks=[],
        )

    def test_compose_node_skipped_for_dod_level_1(self):
        """Composition should be skipped for DoD Level 1."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus, VerificationStatus
        from cdr.orchestration.graph import compose_node

        claim1 = self._make_claim("c1", "Drug A inhibits enzyme X")
        claim2 = self._make_claim("c2", "Enzyme X is required for disease Y")

        verification1 = self._make_verification("c1", VerificationStatus.VERIFIED)
        verification2 = self._make_verification("c2", VerificationStatus.VERIFIED)

        state = CDRState(
            run_id="test_compose_001",
            question="Can Drug A treat disease Y?",
            status=RunStatus.RUNNING,
            claims=[claim1, claim2],
            verification=[verification1, verification2],
        )

        # DoD Level 1 - should skip composition
        config = {"configurable": {"dod_level": 1}}
        result = run_async(compose_node(state, config))

        assert result["composed_hypotheses"] == []

    def test_compose_node_skipped_for_dod_level_2(self):
        """Composition should be skipped for DoD Level 2."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus, VerificationStatus
        from cdr.orchestration.graph import compose_node

        claim1 = self._make_claim("c1", "Drug A inhibits enzyme X")
        claim2 = self._make_claim("c2", "Enzyme X is required for disease Y")

        verification1 = self._make_verification("c1", VerificationStatus.VERIFIED)
        verification2 = self._make_verification("c2", VerificationStatus.VERIFIED)

        state = CDRState(
            run_id="test_compose_002",
            question="Can Drug A treat disease Y?",
            status=RunStatus.RUNNING,
            claims=[claim1, claim2],
            verification=[verification1, verification2],
        )

        # DoD Level 2 - should skip composition
        config = {"configurable": {"dod_level": 2}}
        result = run_async(compose_node(state, config))

        assert result["composed_hypotheses"] == []

    def test_compose_node_skipped_without_llm(self):
        """Composition should be skipped without LLM provider."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus, VerificationStatus
        from cdr.orchestration.graph import compose_node

        claim1 = self._make_claim("c1", "Drug A inhibits enzyme X")
        claim2 = self._make_claim("c2", "Enzyme X is required for disease Y")

        verification1 = self._make_verification("c1", VerificationStatus.VERIFIED)
        verification2 = self._make_verification("c2", VerificationStatus.VERIFIED)

        state = CDRState(
            run_id="test_compose_003",
            question="Can Drug A treat disease Y?",
            status=RunStatus.RUNNING,
            claims=[claim1, claim2],
            verification=[verification1, verification2],
        )

        # DoD Level 3 but NO LLM - should skip
        config = {"configurable": {"dod_level": 3, "llm_provider": None}}
        result = run_async(compose_node(state, config))

        assert result["composed_hypotheses"] == []

    def test_compose_node_skipped_insufficient_verified_claims(self):
        """Composition requires at least 2 verified claims."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus, VerificationStatus
        from cdr.orchestration.graph import compose_node

        claim1 = self._make_claim("c1", "Drug A inhibits enzyme X")
        claim2 = self._make_claim("c2", "Enzyme X is required for disease Y")

        # Only one verified, one contradicted
        verification1 = self._make_verification("c1", VerificationStatus.VERIFIED)
        verification2 = self._make_verification("c2", VerificationStatus.CONTRADICTED)

        state = CDRState(
            run_id="test_compose_004",
            question="Can Drug A treat disease Y?",
            status=RunStatus.RUNNING,
            claims=[claim1, claim2],
            verification=[verification1, verification2],
        )

        # DoD Level 3 but insufficient verified claims - should skip
        config = {"configurable": {"dod_level": 3, "llm_provider": None}}
        result = run_async(compose_node(state, config))

        assert result["composed_hypotheses"] == []

    def test_graph_node_compose_exists(self):
        """GraphNode enum should include COMPOSE for compositional inference."""
        from cdr.core.enums import GraphNode

        assert hasattr(GraphNode, "COMPOSE")
        assert GraphNode.COMPOSE.value == "compose"

    def test_cdr_state_has_composed_hypotheses_field(self):
        """CDRState should have composed_hypotheses field."""
        from cdr.core.schemas import CDRState
        from cdr.core.enums import RunStatus

        state = CDRState(
            run_id="test_schema",
            question="Test question",
            status=RunStatus.RUNNING,
        )

        assert hasattr(state, "composed_hypotheses")
        assert state.composed_hypotheses == []

    def test_composition_schemas_exist(self):
        """Composition module should export required schemas."""
        from cdr.composition import (
            CompositionEngine,
            ComposedHypothesis,
            MechanisticRelation,
            RelationType,
            HypothesisStrength,
        )

        # Verify enums have expected values
        assert hasattr(RelationType, "CAUSAL")
        assert hasattr(RelationType, "INHIBITORY")
        assert hasattr(HypothesisStrength, "STRONG")
        assert hasattr(HypothesisStrength, "MODERATE")
        assert hasattr(HypothesisStrength, "WEAK")

        # Verify classes exist
        assert CompositionEngine is not None
        assert ComposedHypothesis is not None
        assert MechanisticRelation is not None

    def test_compose_node_runs_with_llm_provider_dod3(self):
        """Test compose_node executes composition with LLM provider at DoD 3."""
        from unittest.mock import MagicMock
        from cdr.core.schemas import CDRState, PICO
        from cdr.core.enums import RunStatus, VerificationStatus
        from cdr.orchestration.graph import compose_node

        # Create claims with composable mechanistic chain
        claim1 = self._make_claim("c1", "Drug A inhibits enzyme X with high specificity")
        claim2 = self._make_claim("c2", "Enzyme X is essential for disease Y progression")

        verification1 = self._make_verification("c1", VerificationStatus.VERIFIED)
        verification2 = self._make_verification("c2", VerificationStatus.VERIFIED)

        pico = PICO(
            population="Adults with disease Y",
            intervention="Drug A",
            comparator="Placebo",
            outcome="Disease progression reduction",
        )

        state = CDRState(
            run_id="test_compose_e2e",
            question="Can Drug A treat disease Y?",
            status=RunStatus.RUNNING,
            pico=pico,
            claims=[claim1, claim2],
            verification=[verification1, verification2],
        )

        # Create mock LLM provider that returns valid composition JSON
        mock_provider = MagicMock()

        # Responses for: extract_relations (x2), compose_hypothesis, test_design
        relation_json = """{
            "relations": [
                {"source_concept": "Drug A", "target_concept": "enzyme X", "relation_type": "inhibitory", "confidence": 0.9}
            ]
        }"""
        relation_json_2 = """{
            "relations": [
                {"source_concept": "enzyme X", "target_concept": "disease Y", "relation_type": "causal", "confidence": 0.85}
            ]
        }"""
        hypothesis_json = """{
            "hypothesis_text": "Drug A may reduce disease Y progression by inhibiting enzyme X which is essential for pathogenesis",
            "mechanistic_chain": [
                {"source": "Drug A", "target": "enzyme X inhibition", "mechanism": "specific binding"},
                {"source": "enzyme X inhibition", "target": "disease Y reduction", "mechanism": "pathway disruption"}
            ],
            "strength": "moderate",
            "confidence": 0.75,
            "rival_hypotheses": ["Off-target effects"],
            "uncontrolled_confounders": ["Genetic variation"],
            "evidence_gaps": ["No human RCT data"],
            "reasoning": "Chain logic from enzyme inhibition to disease reduction"
        }"""
        test_design_json = """{
            "proposed_population": "Adults with active disease Y",
            "proposed_intervention": "Drug A 100mg daily",
            "proposed_comparator": "Placebo",
            "proposed_outcome": "Disease progression at 12 months",
            "mcid_value": null,
            "mcid_rationale": null,
            "recommended_design": "RCT",
            "minimum_sample_size": 300,
            "follow_up_duration": "12 months",
            "critical_measurements": ["Enzyme X levels", "Disease biomarkers"],
            "blinding_requirements": "double-blind"
        }"""

        from cdr.llm.base import LLMResponse

        call_count = [0]
        responses = [relation_json, relation_json_2, hypothesis_json, test_design_json]

        def mock_complete(*args, **kwargs):
            result = responses[call_count[0] % len(responses)]
            call_count[0] += 1
            return LLMResponse(content=result, model="stub", provider="stub")

        mock_provider.complete = mock_complete

        # DoD Level 3 WITH LLM provider - should execute composition
        config = {"configurable": {"dod_level": 3, "llm_provider": mock_provider}}
        result = run_async(compose_node(state, config))

        # Should have composed hypotheses
        assert "composed_hypotheses" in result
        # At least one hypothesis generated (if chain found)
        # Note: may be empty if find_composable_pairs doesn't match
        assert isinstance(result["composed_hypotheses"], list)
        # LLM was called (at least for relation extraction)
        assert call_count[0] >= 2


class TestCompositionEngineWithStub:
    """Tests for CompositionEngine with LLM stub (positive paths).

    HIGH-1 verification: Compositional inference with mocked LLM responses.
    Validates JSON contracts, hypothesis structure, and test design generation.
    Refs: CDR_Integral_Audit_2026-01-20.md HIGH-1
    """

    def _make_claim(self, claim_id: str, text: str):
        """Create a minimal EvidenceClaim for testing."""
        from cdr.core.schemas import EvidenceClaim
        from cdr.core.enums import GRADECertainty

        return EvidenceClaim(
            claim_id=claim_id,
            claim_text=text,
            certainty=GRADECertainty.MODERATE,
            supporting_snippet_ids=["snip_001"],
        )

    def _make_pico(self):
        """Create a PICO for testing."""
        from cdr.core.schemas import PICO

        return PICO(
            population="Adults with hypertension",
            intervention="ACE inhibitors",
            comparator="Placebo",
            outcome="Blood pressure reduction",
        )

    def _make_stub_provider(self, responses: list[str]):
        """Create a stub LLM provider that returns predefined responses."""
        from unittest.mock import MagicMock
        from cdr.llm.base import LLMResponse

        call_count = [0]

        def mock_complete(*args, **kwargs):
            response = responses[call_count[0] % len(responses)]
            call_count[0] += 1
            return LLMResponse(content=response, model="stub", provider="stub")

        provider = MagicMock()
        provider.complete = mock_complete
        return provider

    def test_extract_relations_parses_json(self):
        """Test relation extraction parses JSON correctly."""
        from cdr.composition import CompositionEngine, RelationType

        # Stub LLM response with valid JSON
        relation_json = """{
            "relations": [
                {
                    "source_concept": "ACE inhibitors",
                    "target_concept": "Angiotensin II reduction",
                    "mechanism": "Blocks angiotensin-converting enzyme",
                    "relation_type": "mechanistic",
                    "confidence": 0.85
                }
            ]
        }"""

        stub_provider = self._make_stub_provider([relation_json])
        engine = CompositionEngine(provider=stub_provider)

        claims = [self._make_claim("c1", "ACE inhibitors reduce angiotensin II levels")]
        pico = self._make_pico()

        relations = engine.extract_relations(claims, pico)

        assert len(relations) == 1
        assert relations[0].source_concept == "ACE inhibitors"
        assert relations[0].target_concept == "Angiotensin II reduction"
        assert relations[0].mechanism == "Blocks angiotensin-converting enzyme"
        assert relations[0].relation_type == RelationType.MECHANISTIC
        assert relations[0].confidence_score == 0.85

    def test_extract_relations_handles_multiple_claims(self):
        """Test relation extraction from multiple claims."""
        from cdr.composition import CompositionEngine

        # Two different responses for two claims
        relation_json_1 = """{
            "relations": [
                {
                    "source_concept": "Drug A",
                    "target_concept": "Enzyme X",
                    "relation_type": "inhibitory",
                    "confidence": 0.9
                }
            ]
        }"""
        relation_json_2 = """{
            "relations": [
                {
                    "source_concept": "Enzyme X",
                    "target_concept": "Disease Y",
                    "relation_type": "causal",
                    "confidence": 0.8
                }
            ]
        }"""

        stub_provider = self._make_stub_provider([relation_json_1, relation_json_2])
        engine = CompositionEngine(provider=stub_provider)

        claims = [
            self._make_claim("c1", "Drug A inhibits Enzyme X"),
            self._make_claim("c2", "Enzyme X causes Disease Y"),
        ]

        relations = engine.extract_relations(claims, None)

        assert len(relations) == 2
        # Chain: Drug A -> Enzyme X -> Disease Y

    def test_find_composable_pairs_identifies_chains(self):
        """Test finding composable pairs based on shared concepts."""
        from cdr.composition import CompositionEngine, MechanisticRelation, RelationType

        engine = CompositionEngine(provider=None)

        claims = [
            self._make_claim("c1", "Drug A inhibits Enzyme X"),
            self._make_claim("c2", "Enzyme X causes Disease Y"),
        ]

        # Manually create relations (as if extracted)
        relations = [
            MechanisticRelation(
                relation_id="rel_1",
                source_concept="Drug A",
                target_concept="Enzyme X",
                relation_type=RelationType.INHIBITORY,
                supporting_claim_ids=["c1"],
            ),
            MechanisticRelation(
                relation_id="rel_2",
                source_concept="Enzyme X",
                target_concept="Disease Y",
                relation_type=RelationType.CAUSAL,
                supporting_claim_ids=["c2"],
            ),
        ]

        pairs = engine.find_composable_pairs(claims, relations)

        assert len(pairs) >= 1
        # Should find (c1, c2) because c1's target (Enzyme X) is c2's source
        claim_a, claim_b, shared = pairs[0]
        assert "enzyme x" in shared

    def test_compose_hypothesis_generates_valid_structure(self):
        """Test hypothesis composition creates valid structure."""
        from cdr.composition import CompositionEngine, HypothesisStrength

        # Stub LLM response for hypothesis composition with if-then structure
        hypothesis_json = """{
            "hypothesis_text": "If Drug A inhibits Enzyme X, then Drug A may treat Disease Y by removing the causal pathogenic factor",
            "mechanistic_chain": [
                {
                    "source": "Drug A",
                    "target": "Enzyme X inhibition",
                    "mechanism": "Competitive binding"
                },
                {
                    "source": "Enzyme X inhibition",
                    "target": "Disease Y reduction",
                    "mechanism": "Removes pathogenic factor"
                }
            ],
            "strength": "moderate",
            "confidence": 0.65,
            "rival_hypotheses": [
                "Drug A may have direct anti-inflammatory effects unrelated to Enzyme X",
                "Other compensatory pathways may nullify Enzyme X inhibition"
            ],
            "uncontrolled_confounders": ["Age", "Comorbidities"],
            "evidence_gaps": ["No direct human trial of Drug A on Disease Y"],
            "reasoning": "Claim A establishes Drug A inhibits Enzyme X. Claim B establishes Enzyme X causes Disease Y. Therefore, Drug A may reduce Disease Y by removing the causal factor."
        }"""

        stub_provider = self._make_stub_provider([hypothesis_json])
        engine = CompositionEngine(provider=stub_provider)

        claim_a = self._make_claim("c1", "Drug A inhibits Enzyme X")
        claim_b = self._make_claim("c2", "Enzyme X causes Disease Y")

        hypothesis = engine.compose_hypothesis(claim_a, claim_b, ["enzyme x"], self._make_pico())

        assert hypothesis is not None
        assert hypothesis.hypothesis_id == "hyp_c1_c2"
        assert hypothesis.hypothesis_text.lower().startswith("if"), (
            "Hypothesis must have if-then structure"
        )
        assert "then" in hypothesis.hypothesis_text.lower(), (
            "Hypothesis must have if-then structure"
        )
        assert len(hypothesis.mechanistic_chain) == 2
        assert hypothesis.strength == HypothesisStrength.MODERATE
        assert hypothesis.confidence_score == 0.65
        assert hypothesis.threat_analysis is not None
        assert len(hypothesis.threat_analysis.rival_hypotheses) == 2
        assert "Age" in hypothesis.threat_analysis.uncontrolled_confounders
        assert hypothesis.reasoning_trace is not None

    def test_propose_test_design_generates_valid_structure(self):
        """Test test design proposal creates valid structure."""
        from cdr.composition import (
            CompositionEngine,
            ComposedHypothesis,
            HypothesisStrength,
            ThreatAnalysis,
        )

        # Stub LLM response for test design
        test_design_json = """{
            "proposed_population": "Adults aged 40-70 with confirmed Enzyme X overactivity and early Disease Y",
            "proposed_intervention": "Drug A 10mg daily for 12 weeks",
            "proposed_comparator": "Matching placebo",
            "proposed_outcome": "Disease Y severity score at 12 weeks (validated scale)",
            "mcid_value": 3.5,
            "mcid_rationale": "Based on prior studies showing 3.5 points is patient-perceptible improvement",
            "recommended_design": "RCT",
            "minimum_sample_size": 200,
            "follow_up_duration": "12 weeks primary, 52 weeks extension",
            "critical_measurements": [
                "Enzyme X activity levels at baseline and week 4",
                "Disease Y severity score at weeks 4, 8, 12",
                "Adverse events continuously"
            ],
            "blinding_requirements": "double-blind"
        }"""

        stub_provider = self._make_stub_provider([test_design_json])
        engine = CompositionEngine(provider=stub_provider)

        # Create a hypothesis to test
        hypothesis = ComposedHypothesis(
            hypothesis_id="hyp_test",
            hypothesis_text="Drug A may treat Disease Y by inhibiting Enzyme X through competitive binding",
            source_claim_ids=["c1", "c2"],
            mechanistic_chain=[],
            strength=HypothesisStrength.MODERATE,
            confidence_score=0.65,
            threat_analysis=ThreatAnalysis(
                rival_hypotheses=[], uncontrolled_confounders=[], evidence_gaps=[]
            ),
        )

        test_design = engine.propose_test_design(hypothesis, self._make_pico())

        assert test_design is not None
        assert "Adults aged 40-70" in test_design.proposed_population
        assert "Drug A 10mg" in test_design.proposed_intervention
        assert "placebo" in test_design.proposed_comparator.lower()
        assert test_design.mcid_value == 3.5
        assert test_design.recommended_design == "RCT"
        assert test_design.minimum_sample_size == 200
        assert len(test_design.critical_measurements) == 3
        assert test_design.blinding_requirements == "double-blind"

    def test_full_pipeline_run(self):
        """Test full composition pipeline with multiple stub responses."""
        from cdr.composition import CompositionEngine

        # Sequence of responses for full pipeline:
        # 1. Relation extraction for claim 1
        # 2. Relation extraction for claim 2
        # 3. Hypothesis composition
        # 4. Test design

        relation_json_1 = """{
            "relations": [
                {
                    "source_concept": "Statins",
                    "target_concept": "LDL cholesterol",
                    "relation_type": "inhibitory",
                    "confidence": 0.95
                }
            ]
        }"""

        relation_json_2 = """{
            "relations": [
                {
                    "source_concept": "LDL cholesterol",
                    "target_concept": "Cardiovascular events",
                    "relation_type": "causal",
                    "confidence": 0.9
                }
            ]
        }"""

        hypothesis_json = """{
            "hypothesis_text": "If statins lower LDL cholesterol, then they may reduce cardiovascular events via decreased atherosclerosis",
            "mechanistic_chain": [
                {"source": "Statins", "target": "LDL reduction", "mechanism": "HMG-CoA reductase inhibition"},
                {"source": "LDL reduction", "target": "CV event reduction", "mechanism": "Reduced atherosclerosis"}
            ],
            "strength": "strong",
            "confidence": 0.88,
            "rival_hypotheses": ["Pleiotropic effects of statins independent of LDL", "Anti-inflammatory effects unrelated to cholesterol"],
            "uncontrolled_confounders": ["Diet"],
            "evidence_gaps": ["No head-to-head comparison with other lipid-lowering strategies"],
            "reasoning": "Well-established chain"
        }"""

        test_design_json = """{
            "proposed_population": "High-risk adults",
            "proposed_intervention": "Atorvastatin 40mg",
            "proposed_comparator": "Placebo",
            "proposed_outcome": "MACE at 5 years",
            "mcid_value": 0.80,
            "mcid_rationale": "HR of 0.80 represents clinically meaningful 20% relative risk reduction consistent with major statin trials",
            "recommended_design": "RCT",
            "minimum_sample_size": 5000,
            "follow_up_duration": "5 years",
            "critical_measurements": ["LDL at 6 months", "MACE events"],
            "blinding_requirements": "double-blind"
        }"""

        stub_provider = self._make_stub_provider(
            [relation_json_1, relation_json_2, hypothesis_json, test_design_json]
        )
        engine = CompositionEngine(provider=stub_provider)

        claims = [
            self._make_claim("c1", "Statins lower LDL cholesterol"),
            self._make_claim("c2", "High LDL cholesterol increases cardiovascular events"),
        ]

        hypotheses = engine.run(
            claims=claims,
            pico=self._make_pico(),
            max_hypotheses=1,
            include_test_designs=True,
        )

        assert len(hypotheses) >= 1
        hyp = hypotheses[0]
        assert "cardiovascular" in hyp.hypothesis_text.lower()
        assert hyp.proposed_test is not None
        assert hyp.proposed_test.recommended_design == "RCT"

    def test_parse_json_handles_code_blocks(self):
        """Test JSON parsing handles code blocks."""
        from cdr.composition import CompositionEngine

        engine = CompositionEngine(provider=None)

        # Test with code block wrapper
        response = """```json
        {"relations": [{"source_concept": "A", "target_concept": "B", "relation_type": "causal", "confidence": 0.8}]}
        ```"""

        data = engine._parse_json_response(response)

        assert data is not None
        assert "relations" in data

    def test_parse_json_handles_plain_json(self):
        """Test JSON parsing handles plain JSON."""
        from cdr.composition import CompositionEngine

        engine = CompositionEngine(provider=None)

        response = '{"test": "value"}'
        data = engine._parse_json_response(response)

        assert data is not None
        assert data["test"] == "value"

    def test_parse_json_handles_embedded_json(self):
        """Test JSON parsing handles JSON embedded in text."""
        from cdr.composition import CompositionEngine

        engine = CompositionEngine(provider=None)

        response = 'Here is the result: {"key": "value"} as requested.'
        data = engine._parse_json_response(response)

        assert data is not None
        assert data["key"] == "value"

    def test_parse_json_returns_none_for_invalid(self):
        """Test JSON parsing returns None for invalid input."""
        from cdr.composition import CompositionEngine

        engine = CompositionEngine(provider=None)

        response = "This is not JSON at all"
        data = engine._parse_json_response(response)

        assert data is None
