"""
Unit tests for DoD3 Enforcement Layer.

Tests the hard exclusion logic that ensures:
- Out-of-scope evidence cannot support claims
- Claims with excluded evidence are degraded
- Orphan claims (no valid evidence) are removed
- Hypotheses without structural support are suppressed

Refs: DoD3 Contract, CDR_DOD3_b3142335 audit
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from cdr.verification.dod3_enforcement import (
    DoD3Enforcer,
    EnforcementResult,
    ExcludedEvidence,
    ExclusionReason,
    DegradedClaim,
    SuppressedHypothesis,
    SubPICODecomposer,
    HypothesisGate,
    GateReportRenderer,
)


# =============================================================================
# FIXTURES
# =============================================================================


def make_mock_pico():
    """Create a mock PICO for testing."""
    pico = MagicMock()
    pico.population = "patients with atrial fibrillation"
    pico.intervention = "aspirin"
    pico.comparator = "placebo"
    pico.outcome = "stroke prevention"
    pico.study_types = ["rct"]
    return pico


def make_mock_record(record_id: str, pmid: str = None, title: str = "Test Record"):
    """Create a mock Record."""
    record = MagicMock()
    record.record_id = record_id
    record.pmid = pmid or f"1234567{record_id[-1]}"
    record.title = title
    return record


def make_mock_snippet(
    snippet_id: str, record_id: str, pmid: str = None, text: str = "Test snippet"
):
    """Create a mock Snippet."""
    snippet = MagicMock()
    snippet.snippet_id = snippet_id
    snippet.text = text
    snippet.source_ref = MagicMock()
    snippet.source_ref.record_id = record_id
    snippet.source_ref.pmid = pmid or f"1234567{record_id[-1]}"
    return snippet


def make_mock_claim(claim_id: str, snippet_ids: list, text: str = "Test claim"):
    """Create a mock Claim."""
    claim = MagicMock()
    claim.claim_id = claim_id
    claim.claim_text = text
    claim.supporting_snippet_ids = snippet_ids
    return claim


def make_mock_hypothesis(
    hyp_id: str, claim_ids: list, text: str = "Test hypothesis", confidence: float = 0.7
):
    """Create a mock Hypothesis."""
    return {
        "hypothesis_id": hyp_id,
        "hypothesis_text": text,
        "source_claim_ids": claim_ids,
        "confidence_score": confidence,
    }


# =============================================================================
# TEST DoD3Enforcer
# =============================================================================


class TestDoD3Enforcer:
    """Tests for DoD3Enforcer class."""

    def test_enforce_no_violations_all_valid(self):
        """When no violations, all evidence should be valid."""
        pico = make_mock_pico()
        records = [make_mock_record("rec_001"), make_mock_record("rec_002")]
        snippets = [
            make_mock_snippet("rec_001_snip_1", "rec_001"),
            make_mock_snippet("rec_002_snip_1", "rec_002"),
        ]
        claims = [
            make_mock_claim("claim_001", ["rec_001_snip_1", "rec_002_snip_1"]),
        ]

        enforcer = DoD3Enforcer(strict=True)
        result = enforcer.enforce(
            run_id="test_run",
            pico=pico,
            records=records,
            snippets=snippets,
            claims=claims,
            gate_violations=[],
            is_unpublishable=False,
        )

        assert len(result.excluded_records) == 0
        assert len(result.excluded_snippets) == 0
        assert "rec_001" in result.valid_record_ids
        assert "rec_002" in result.valid_record_ids
        assert "rec_001_snip_1" in result.valid_snippet_ids
        assert "rec_002_snip_1" in result.valid_snippet_ids
        assert "claim_001" in result.valid_claim_ids
        assert len(result.orphan_claims) == 0

    def test_enforce_excludes_record_with_violation(self):
        """Record with gate violation should be excluded."""
        pico = make_mock_pico()
        records = [make_mock_record("rec_001"), make_mock_record("rec_002")]
        snippets = [
            make_mock_snippet("rec_001_snip_1", "rec_001"),
            make_mock_snippet("rec_002_snip_1", "rec_002"),
        ]
        claims = [make_mock_claim("claim_001", ["rec_001_snip_1", "rec_002_snip_1"])]

        violations = [
            {
                "result": "fail",
                "record_id": "rec_001",
                "snippet_id": None,
                "pmid": "12345671",
                "mismatch_type": "population_excluded",
                "message": "Population without AF",
                "pico_component": "P",
            }
        ]

        enforcer = DoD3Enforcer(strict=True)
        result = enforcer.enforce(
            run_id="test_run",
            pico=pico,
            records=records,
            snippets=snippets,
            claims=claims,
            gate_violations=violations,
            is_unpublishable=True,
        )

        # Record excluded
        assert len(result.excluded_records) == 1
        assert result.excluded_records[0].evidence_id == "rec_001"
        assert result.excluded_records[0].reason == ExclusionReason.POPULATION_EXCLUDED

        # Record's snippets also excluded
        assert "rec_001_snip_1" not in result.valid_snippet_ids
        assert "rec_002_snip_1" in result.valid_snippet_ids

        # Claim degraded but not orphan (still has one valid snippet)
        assert len(result.degraded_claims) == 1
        assert result.degraded_claims[0].claim_id == "claim_001"
        assert result.degraded_claims[0].is_orphan is False
        assert "rec_001_snip_1" in result.degraded_claims[0].removed_snippet_ids

    def test_enforce_orphans_claim_with_all_excluded_snippets(self):
        """Claim becomes orphan when ALL its snippets are excluded."""
        pico = make_mock_pico()
        records = [make_mock_record("rec_001")]
        snippets = [make_mock_snippet("rec_001_snip_1", "rec_001")]
        claims = [make_mock_claim("claim_001", ["rec_001_snip_1"])]

        violations = [
            {
                "result": "fail",
                "record_id": "rec_001",
                "snippet_id": None,
                "pmid": "12345671",
                "mismatch_type": "population_excluded",
                "message": "Out of scope",
            }
        ]

        enforcer = DoD3Enforcer(strict=True, min_snippets_per_claim=1)
        result = enforcer.enforce(
            run_id="test_run",
            pico=pico,
            records=records,
            snippets=snippets,
            claims=claims,
            gate_violations=violations,
            is_unpublishable=True,
        )

        # Claim is orphaned
        assert "claim_001" in result.orphan_claims
        assert len(result.degraded_claims) == 1
        assert result.degraded_claims[0].is_orphan is True

    def test_enforce_suppresses_hypotheses_when_unpublishable(self):
        """Hypotheses should be suppressed when run is unpublishable (strict mode)."""
        pico = make_mock_pico()
        records = [make_mock_record("rec_001")]
        snippets = [make_mock_snippet("rec_001_snip_1", "rec_001")]
        claims = [make_mock_claim("claim_001", ["rec_001_snip_1"])]
        hypotheses = [make_mock_hypothesis("hyp_001", ["claim_001"])]

        enforcer = DoD3Enforcer(
            strict=True,
            suppress_hypotheses_on_unpublishable=True,
        )
        result = enforcer.enforce(
            run_id="test_run",
            pico=pico,
            records=records,
            snippets=snippets,
            claims=claims,
            hypotheses=hypotheses,
            gate_violations=[],
            is_unpublishable=True,
        )

        # Hypothesis suppressed due to unpublishable status
        assert len(result.suppressed_hypotheses) == 1
        assert result.suppressed_hypotheses[0].hypothesis_id == "hyp_001"
        assert "unpublishable" in result.suppressed_hypotheses[0].reason.lower()

    def test_enforce_suppresses_hypothesis_with_orphan_claims(self):
        """Hypothesis with all orphan source claims should be suppressed."""
        pico = make_mock_pico()
        records = [make_mock_record("rec_001")]
        snippets = [make_mock_snippet("rec_001_snip_1", "rec_001")]
        claims = [make_mock_claim("claim_001", ["rec_001_snip_1"])]
        hypotheses = [make_mock_hypothesis("hyp_001", ["claim_001"])]

        violations = [
            {
                "result": "fail",
                "record_id": "rec_001",
                "mismatch_type": "population_excluded",
                "message": "Out of scope",
            }
        ]

        enforcer = DoD3Enforcer(
            strict=True,
            suppress_hypotheses_on_unpublishable=False,  # Don't suppress just for unpublishable
        )
        result = enforcer.enforce(
            run_id="test_run",
            pico=pico,
            records=records,
            snippets=snippets,
            claims=claims,
            hypotheses=hypotheses,
            gate_violations=violations,
            is_unpublishable=False,
        )

        # Claim orphaned
        assert "claim_001" in result.orphan_claims

        # Hypothesis suppressed because source claim is orphan
        assert len(result.suppressed_hypotheses) == 1
        assert "orphan" in result.suppressed_hypotheses[0].reason.lower()


class TestDoD3EnforcerApplyMethods:
    """Tests for apply_to_claims and apply_to_hypotheses methods."""

    def test_apply_to_claims_filters_orphans(self):
        """apply_to_claims should exclude orphan claims."""
        pico = make_mock_pico()
        records = [make_mock_record("rec_001"), make_mock_record("rec_002")]
        snippets = [
            make_mock_snippet("rec_001_snip_1", "rec_001"),
            make_mock_snippet("rec_002_snip_1", "rec_002"),
        ]
        claims = [
            make_mock_claim("claim_001", ["rec_001_snip_1"]),
            make_mock_claim("claim_002", ["rec_002_snip_1"]),
        ]

        violations = [
            {
                "result": "fail",
                "record_id": "rec_001",
                "mismatch_type": "population_excluded",
                "message": "Out of scope",
            }
        ]

        enforcer = DoD3Enforcer(strict=True)
        result = enforcer.enforce(
            run_id="test_run",
            pico=pico,
            records=records,
            snippets=snippets,
            claims=claims,
            gate_violations=violations,
            is_unpublishable=True,
        )

        # Apply to claims
        valid_claims = enforcer.apply_to_claims(claims, result)

        # claim_001 orphaned (rec_001 excluded), claim_002 valid
        assert len(valid_claims) == 1
        assert valid_claims[0].claim_id == "claim_002"

    def test_apply_to_claims_updates_snippet_ids(self):
        """apply_to_claims should update supporting_snippet_ids to valid only."""
        pico = make_mock_pico()
        records = [make_mock_record("rec_001"), make_mock_record("rec_002")]
        snippets = [
            make_mock_snippet("rec_001_snip_1", "rec_001"),
            make_mock_snippet("rec_002_snip_1", "rec_002"),
        ]
        claims = [
            make_mock_claim("claim_001", ["rec_001_snip_1", "rec_002_snip_1"]),
        ]

        violations = [
            {
                "result": "fail",
                "record_id": "rec_001",
                "mismatch_type": "population_excluded",
                "message": "Out of scope",
            }
        ]

        enforcer = DoD3Enforcer(strict=True)
        result = enforcer.enforce(
            run_id="test_run",
            pico=pico,
            records=records,
            snippets=snippets,
            claims=claims,
            gate_violations=violations,
            is_unpublishable=True,
        )

        valid_claims = enforcer.apply_to_claims(claims, result)

        # Claim still valid but with filtered snippets
        assert len(valid_claims) == 1
        assert valid_claims[0].supporting_snippet_ids == ["rec_002_snip_1"]


# =============================================================================
# TEST SubPICODecomposer
# =============================================================================


class TestSubPICODecomposer:
    """Tests for SubPICODecomposer class."""

    def test_decompose_finds_doac_vs_aspirin(self):
        """Should detect DOAC vs aspirin comparison in snippets."""
        pico = make_mock_pico()  # intervention=aspirin, comparator=placebo

        snippets = [
            make_mock_snippet("s1", "r1", text="Patients were randomized to apixaban or aspirin."),
            make_mock_snippet("s2", "r1", text="Apixaban versus aspirin reduced stroke risk."),
        ]

        decomposer = SubPICODecomposer()
        sub_picos = decomposer.decompose(pico, snippets)

        assert len(sub_picos) >= 1
        # Should find aspirin vs doac comparison
        found_doac = any("doac" in sp.intervention or "doac" in sp.comparator for sp in sub_picos)
        found_aspirin = any(
            "aspirin" in sp.intervention or "aspirin" in sp.comparator for sp in sub_picos
        )
        assert found_doac or found_aspirin

    def test_decompose_no_comparisons(self):
        """Empty result when no comparisons found."""
        pico = make_mock_pico()
        snippets = [
            make_mock_snippet("s1", "r1", text="This study examined patient outcomes."),
        ]

        decomposer = SubPICODecomposer()
        sub_picos = decomposer.decompose(pico, snippets)

        assert len(sub_picos) == 0

    def test_generate_sectioned_conclusion(self):
        """Should generate sectioned conclusion with caveat when no direct evidence."""
        pico = make_mock_pico()  # comparator=placebo

        snippets = [
            make_mock_snippet("s1", "r1", text="Apixaban vs aspirin showed benefit."),
        ]

        decomposer = SubPICODecomposer()
        sub_picos = decomposer.decompose(pico, snippets)

        result = decomposer.generate_sectioned_conclusion(sub_picos, pico, [])

        # Should indicate no direct evidence for placebo comparison
        assert result["has_direct_evidence"] is False or result.get("caveat") is not None


# =============================================================================
# TEST HypothesisGate
# =============================================================================


class TestHypothesisGate:
    """Tests for HypothesisGate class."""

    def test_hypothesis_passes_with_valid_claims(self):
        """Hypothesis should pass when source claims are valid."""
        hypothesis = make_mock_hypothesis("hyp_001", ["claim_001"])

        gate = HypothesisGate(strict=True)
        passed, reason = gate.check_hypothesis(
            hypothesis=hypothesis,
            valid_claim_ids={"claim_001"},
            orphan_claim_ids=[],
            is_unpublishable=False,
        )

        assert passed is True
        assert "passes" in reason.lower()

    def test_hypothesis_fails_when_unpublishable_strict(self):
        """Hypothesis should fail in strict mode when run is unpublishable."""
        hypothesis = make_mock_hypothesis("hyp_001", ["claim_001"])

        gate = HypothesisGate(strict=True)
        passed, reason = gate.check_hypothesis(
            hypothesis=hypothesis,
            valid_claim_ids={"claim_001"},
            orphan_claim_ids=[],
            is_unpublishable=True,
        )

        assert passed is False
        assert "unpublishable" in reason.lower()

    def test_hypothesis_fails_when_all_claims_orphan(self):
        """Hypothesis should fail when all source claims are orphan."""
        hypothesis = make_mock_hypothesis("hyp_001", ["claim_001", "claim_002"])

        gate = HypothesisGate(strict=True)
        passed, reason = gate.check_hypothesis(
            hypothesis=hypothesis,
            valid_claim_ids=set(),
            orphan_claim_ids=["claim_001", "claim_002"],
            is_unpublishable=False,
        )

        assert passed is False
        assert "orphan" in reason.lower()

    def test_high_confidence_needs_multiple_claims(self):
        """High-confidence hypothesis (>70%) needs at least 2 valid source claims."""
        hypothesis = make_mock_hypothesis("hyp_001", ["claim_001"], confidence=0.85)

        gate = HypothesisGate(strict=True)
        passed, reason = gate.check_hypothesis(
            hypothesis=hypothesis,
            valid_claim_ids={"claim_001"},
            orphan_claim_ids=[],
            is_unpublishable=False,
        )

        assert passed is False
        assert "2 valid source claims" in reason or "≥2" in reason


# =============================================================================
# TEST GateReportRenderer
# =============================================================================


class TestGateReportRenderer:
    """Tests for GateReportRenderer class."""

    def test_render_html_includes_status(self):
        """HTML should include status prominently."""
        renderer = GateReportRenderer()

        gate_report = {
            "overall_status": "unpublishable",
            "generated_at": "2026-01-21T12:00:00",
            "summary": {"total_checks": 4, "passed": 2, "warned": 1, "failed": 1},
            "gate_results": {},
            "blocker_violations": [],
        }

        html = renderer.render_html(gate_report)

        assert "UNPUBLISHABLE" in html
        assert "DoD3 Gate Report" in html
        assert "Total Checks" in html

    def test_render_html_includes_blocker_violations(self):
        """HTML should include blocker violations."""
        renderer = GateReportRenderer()

        gate_report = {
            "overall_status": "unpublishable",
            "generated_at": "2026-01-21T12:00:00",
            "summary": {"total_checks": 4, "passed": 2, "warned": 0, "failed": 2},
            "gate_results": {},
            "blocker_violations": [
                {
                    "gate": "pico_match",
                    "mismatch_type": "population_excluded",
                    "record_id": "rec_001",
                    "pmid": "12345678",
                    "message": "Population without AF",
                },
            ],
        }

        html = renderer.render_html(gate_report)

        assert "Blocker Violations" in html
        assert "population_excluded" in html
        assert "12345678" in html

    def test_render_html_includes_enforcement_summary(self):
        """HTML should include enforcement summary when provided."""
        renderer = GateReportRenderer()

        gate_report = {
            "overall_status": "unpublishable",
            "generated_at": "2026-01-21T12:00:00",
            "summary": {"total_checks": 4, "passed": 2, "warned": 0, "failed": 2},
            "gate_results": {},
            "blocker_violations": [],
        }

        enforcement = EnforcementResult(
            run_id="test",
            excluded_records=[
                ExcludedEvidence(
                    evidence_type="record",
                    evidence_id="rec_001",
                    pmid="12345678",
                    reason=ExclusionReason.POPULATION_EXCLUDED,
                    detail="Test",
                )
            ],
            excluded_snippets=[],
            orphan_claims=["claim_001"],
        )

        html = renderer.render_html(gate_report, enforcement)

        assert "Enforcement Applied" in html
        assert "Records Excluded: 1" in html
        assert "Claims Orphaned: 1" in html

    def test_render_markdown_basic(self):
        """Markdown render should include key information."""
        renderer = GateReportRenderer()

        gate_report = {
            "overall_status": "publishable",
            "generated_at": "2026-01-21T12:00:00",
            "summary": {"total_checks": 4, "passed": 4, "warned": 0, "failed": 0},
            "gate_results": {},
            "blocker_violations": [],
        }

        md = renderer.render_markdown(gate_report)

        assert "## DoD3 Gate Report" in md
        assert "PUBLISHABLE" in md
        assert "Total Checks: 4" in md


# =============================================================================
# TEST EnforcementResult
# =============================================================================


class TestEnforcementResult:
    """Tests for EnforcementResult dataclass."""

    def test_to_dict_complete(self):
        """to_dict should include all fields."""
        result = EnforcementResult(
            run_id="test_run",
            excluded_records=[
                ExcludedEvidence(
                    evidence_type="record",
                    evidence_id="rec_001",
                    pmid="12345678",
                    reason=ExclusionReason.POPULATION_EXCLUDED,
                    detail="Test exclusion",
                )
            ],
            excluded_snippets=[],
            degraded_claims=[
                DegradedClaim(
                    claim_id="claim_001",
                    original_text="Original",
                    degraded_text="Degraded",
                    reason="Evidence reduced",
                    removed_snippet_ids=["s1"],
                    remaining_snippet_ids=["s2"],
                )
            ],
            orphan_claims=["claim_002"],
            suppressed_hypotheses=[
                SuppressedHypothesis(
                    hypothesis_id="hyp_001",
                    hypothesis_text="Test hyp",
                    reason="Suppressed",
                    source_claim_ids=["claim_001"],
                    orphan_claim_ids=["claim_001"],
                )
            ],
            valid_record_ids={"rec_002"},
            valid_snippet_ids={"s2"},
            valid_claim_ids={"claim_003"},
        )

        d = result.to_dict()

        assert d["run_id"] == "test_run"
        assert len(d["excluded_records"]) == 1
        assert d["excluded_records"][0]["reason"] == "population_excluded"
        assert len(d["degraded_claims"]) == 1
        assert d["orphan_claims"] == ["claim_002"]
        assert len(d["suppressed_hypotheses"]) == 1
        assert d["summary"]["total_excluded_records"] == 1
        assert d["summary"]["total_orphan_claims"] == 1

    def test_has_exclusions_property(self):
        """has_exclusions should return True when there are exclusions."""
        result = EnforcementResult()
        assert result.has_exclusions is False

        result.excluded_records.append(
            ExcludedEvidence("record", "r1", None, ExclusionReason.POPULATION_EXCLUDED, "Test")
        )
        assert result.has_exclusions is True
