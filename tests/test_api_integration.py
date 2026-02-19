"""
Tests for CDR API endpoints.

API integration tests using FastAPI TestClient.
Refs: CDR_Integral_Audit_2026-01-20.md (API contract validation)

Tests:
1. Health endpoint
2. Run lifecycle (create, get status, list)
3. Detail endpoints (claims, snippets, studies, prisma, search-plan, hypotheses)
4. Error handling (404 for non-existent runs)
5. Contract validation (response schema conformity)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cdr.api.routes import router, _runs
from cdr.core.enums import RunStatus


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def app():
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clear_runs():
    """Clear runs before each test."""
    _runs.clear()
    yield
    _runs.clear()


# =============================================================================
# MOCK STATE BUILDERS
#
# These create mock CDRState-like objects that match what routes.py expects.
# Using dataclasses instead of real schemas to avoid import complexity.
# =============================================================================


class MockVerificationStatus(Enum):
    """Mock verification status."""

    VERIFIED = "VERIFIED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class MockCertaintyLevel(Enum):
    """Mock certainty level."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class MockSection(Enum):
    """Mock section."""

    ABSTRACT = "abstract"
    METHODS = "methods"


class MockStudyType(Enum):
    """Mock study type."""

    RCT = "rct"
    COHORT = "cohort"


@dataclass
class MockPICO:
    """Mock PICO."""

    population: str = "Adults"
    intervention: str = "Treatment A"
    comparator: str | None = "Placebo"
    outcome: str = "Primary outcome"
    study_types: list | None = None


@dataclass
class MockSearchPlan:
    """Mock search plan."""

    pubmed_query: str = "treatment A condition X"
    ct_gov_query: str = "treatment A"
    date_range: tuple | None = ("2020/01/01", "2024/01/01")
    languages: list | None = None
    max_results_per_source: int = 100
    created_at: Any = None


@dataclass
class MockPRISMACounts:
    """Mock PRISMA counts."""

    records_identified: int = 100
    records_screened: int = 90
    # Backend uses 'records_excluded', API response uses 'records_excluded_screening'
    records_excluded: int = 72
    reports_assessed: int = 18
    reports_not_retrieved: int = 2
    studies_included: int = 15
    exclusion_reasons: dict | None = None


@dataclass
class MockRecord:
    """Mock retrieved record for bibliographic metadata."""

    record_id: str = "rec_001"
    title: str = "Test Study Title"
    authors: list | None = field(default_factory=lambda: ["Author A", "Author B"])
    year: int | None = 2024
    journal: str | None = "Test Journal"
    url: str | None = None


@dataclass
class MockSourceRef:
    """Mock source reference for snippets."""

    record_id: str = "rec_001"
    pmid: str | None = "12345"
    doi: str | None = None
    nct_id: str | None = None
    title: str = "Test Study"
    authors: list | None = None
    publication_year: int | None = 2024
    journal: str | None = "Test Journal"
    url: str | None = None
    section: MockSection = MockSection.ABSTRACT
    offset_start: int | None = 0  # Position in text
    offset_end: int | None = 50  # Position in text


@dataclass
class MockSnippet:
    """Mock snippet."""

    snippet_id: str = "rec_001_snip_0"
    text: str = "Evidence snippet text"
    source_ref: MockSourceRef = field(default_factory=MockSourceRef)
    section: MockSection = MockSection.ABSTRACT
    offset_start: int | None = 0
    offset_end: int | None = 50
    relevance_score: float | None = 0.95


@dataclass
class MockOutcome:
    """Mock outcome measure."""

    name: str = "Primary outcome"
    value: float | None = 0.5
    unit: str | None = "mean difference"
    effect_size: float | None = None
    ci_lower: float | None = 0.2  # Routes expects ci_lower/ci_upper for confidence_interval
    ci_upper: float | None = 0.8
    confidence_interval: tuple | None = None
    p_value: float | None = None
    is_significant: bool | None = None
    direction: str | None = None


@dataclass
class MockStudyCard:
    """Mock study card."""

    study_id: str = "study_001"
    record_id: str = "rec_001"
    title: str = "Test Study"
    study_type: MockStudyType = MockStudyType.RCT
    # Routes expects *_extracted field names
    population_extracted: str | None = "Adults"
    intervention_extracted: str | None = "Treatment A"
    comparator_extracted: str | None = "Placebo"
    outcomes: list = field(default_factory=lambda: [MockOutcome()])
    sample_size: int | None = 100
    follow_up_duration: str | None = None


@dataclass
class MockGradeRationale:
    """Mock GRADE rationale as dict-like."""

    risk_of_bias: str = "Low"
    inconsistency: str = "None"
    indirectness: str = "None"
    imprecision: str = "None"
    publication_bias: str = "None"


@dataclass
class MockVerificationResult:
    """Mock verification result."""

    claim_id: str = "claim_001"
    overall_status: MockVerificationStatus = MockVerificationStatus.VERIFIED


@dataclass
class MockEvidenceClaim:
    """Mock evidence claim."""

    claim_id: str = "claim_001"
    claim_text: str = "Treatment A is effective"
    certainty: MockCertaintyLevel = MockCertaintyLevel.MODERATE
    supporting_snippet_ids: list = field(default_factory=lambda: ["rec_001_snip_0"])
    grade_rationale: dict | None = None


def make_mock_hypothesis() -> dict:
    """Create a mock hypothesis dict (backend expects dicts, not objects)."""
    return {
        "hypothesis_id": "hyp_001",
        "claim_a_id": "claim_001",
        "claim_b_id": "claim_002",
        "hypothesis_text": "Combined treatment may enhance efficacy",
        "mechanism": "Synergistic action",
        "rival_hypotheses": None,
        "threats_to_validity": None,
        "mcid": None,
        "test_design": None,
        "confidence": 0.8,
    }


@dataclass
class MockCDRState:
    """Mock CDR state that mimics real CDRState for API testing."""

    run_id: str = "test-run"
    question: str = "Does treatment A work?"
    status: RunStatus = RunStatus.COMPLETED
    pico: MockPICO | None = field(default_factory=MockPICO)
    search_plan: MockSearchPlan | None = field(default_factory=MockSearchPlan)
    prisma_counts: MockPRISMACounts | None = field(default_factory=MockPRISMACounts)
    retrieved_records: list | None = None  # MockRecord list for bibliographic metadata
    snippets: list | None = None
    study_cards: list | None = None
    claims: list | None = None
    verification: list | None = None
    composed_hypotheses: list | None = None
    status_reason: str | None = None
    report_data: Any = None


def make_complete_state(run_id: str = "test-run") -> MockCDRState:
    """Create a complete mock CDRState for testing."""
    record = MockRecord()  # Bibliographic metadata
    snippet = MockSnippet()
    study = MockStudyCard()
    claim = MockEvidenceClaim()
    verification = MockVerificationResult()

    return MockCDRState(
        run_id=run_id,
        retrieved_records=[record],  # Needed for snippets/studies endpoints
        snippets=[snippet],
        study_cards=[study],
        claims=[claim],
        verification=[verification],
    )


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client):
        """Health check returns OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "cdr"

    def test_health_includes_version(self, client):
        """Health check includes version."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data


# =============================================================================
# METRICS ENDPOINT TESTS
# =============================================================================


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_returns_dict(self, client):
        """Metrics endpoint returns dictionary."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        assert isinstance(response.json(), dict)


# =============================================================================
# RUN LIST ENDPOINT TESTS
# =============================================================================


class TestRunListEndpoint:
    """Tests for GET /runs endpoint."""

    def test_list_runs_empty(self, client):
        """Empty runs list returns empty array."""
        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_runs_with_data(self, client):
        """List runs returns stored runs."""
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "research_question": "Test question",
            "progress": {"current_node": "completed", "percentage": 100},
            "errors": [],
            "result": make_complete_state("test-run"),
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T01:00:00Z",
        }

        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["run_id"] == "test-run"
        assert data[0]["status"] == "completed"

    def test_list_runs_sorted_by_updated(self, client):
        """Runs are sorted by updated_at descending."""
        _runs["old-run"] = {
            "run_id": "old-run",
            "status": RunStatus.COMPLETED.value,
            "result": make_complete_state("old-run"),
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T01:00:00Z",
        }
        _runs["new-run"] = {
            "run_id": "new-run",
            "status": RunStatus.COMPLETED.value,
            "result": make_complete_state("new-run"),
            "created_at": "2024-01-02T00:00:00Z",
            "updated_at": "2024-01-02T01:00:00Z",
        }

        response = client.get("/api/v1/runs")
        data = response.json()
        assert data[0]["run_id"] == "new-run"
        assert data[1]["run_id"] == "old-run"


# =============================================================================
# RUN STATUS ENDPOINT TESTS
# =============================================================================


class TestRunStatusEndpoint:
    """Tests for GET /runs/{run_id} endpoint."""

    def test_get_status_not_found(self, client):
        """Non-existent run returns 404."""
        response = client.get("/api/v1/runs/nonexistent")
        assert response.status_code == 404

    def test_get_status_found(self, client):
        """Existing run returns status."""
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.RUNNING.value,
            "research_question": "Test question",
            "progress": {"current_node": "retrieve", "percentage": 20},
            "errors": [],
            "result": None,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }

        response = client.get("/api/v1/runs/test-run")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "test-run"
        assert data["status"] == "running"
        assert data["progress"]["percentage"] == 20


# =============================================================================
# CLAIMS ENDPOINT TESTS
# =============================================================================


class TestClaimsEndpoint:
    """Tests for GET /runs/{run_id}/claims endpoint."""

    def test_claims_not_found(self, client):
        """Non-existent run returns 404."""
        response = client.get("/api/v1/runs/nonexistent/claims")
        assert response.status_code == 404

    def test_claims_not_completed(self, client):
        """Non-terminal run (PENDING) returns empty list, not error."""
        # NOTE: RUNNING status now returns 200 to allow viewing intermediate results
        # Only PENDING returns empty list (run hasn't started)
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.PENDING.value,  # Changed from RUNNING
            "research_question": "Test",
            "progress": {},
            "errors": [],
            "result": None,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/claims")
        # PENDING runs now return 200 with empty list (more user-friendly)
        assert response.status_code == 200
        assert response.json() == []

    def test_claims_completed_run(self, client):
        """Completed run returns claims."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "research_question": "Test",
            "progress": {},
            "errors": [],
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/claims")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["claim_id"] == "claim_001"
        assert data[0]["claim_text"] == "Treatment A is effective"
        assert data[0]["certainty"] == "moderate"

    def test_claims_allows_insufficient_evidence(self, client):
        """INSUFFICIENT_EVIDENCE status allows claim retrieval."""
        state = MockCDRState(
            run_id="neg-run",
            status=RunStatus.INSUFFICIENT_EVIDENCE,
            claims=[],
        )
        _runs["neg-run"] = {
            "run_id": "neg-run",
            "status": RunStatus.INSUFFICIENT_EVIDENCE.value,
            "research_question": "Test",
            "progress": {},
            "errors": [],
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/neg-run/claims")
        assert response.status_code == 200
        assert response.json() == []

    def test_claims_includes_verification_status(self, client):
        """Claims include verification_status when available."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/claims")
        data = response.json()[0]
        assert data["verification_status"] == "VERIFIED"


# =============================================================================
# SNIPPETS ENDPOINT TESTS
# =============================================================================


class TestSnippetsEndpoint:
    """Tests for GET /runs/{run_id}/snippets endpoint."""

    def test_snippets_not_found(self, client):
        """Non-existent run returns 404."""
        response = client.get("/api/v1/runs/nonexistent/snippets")
        assert response.status_code == 404

    def test_snippets_returns_data(self, client):
        """Returns snippets for run with result."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/snippets")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["snippet_id"] == "rec_001_snip_0"
        assert data[0]["record_id"] == "rec_001"

    def test_snippets_empty_result(self, client):
        """Returns empty list when no snippets."""
        state = MockCDRState(run_id="test-run", snippets=None)
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/snippets")
        assert response.status_code == 200
        assert response.json() == []


# =============================================================================
# STUDIES ENDPOINT TESTS
# =============================================================================


class TestStudiesEndpoint:
    """Tests for GET /runs/{run_id}/studies endpoint."""

    def test_studies_not_found(self, client):
        """Non-existent run returns 404."""
        response = client.get("/api/v1/runs/nonexistent/studies")
        assert response.status_code == 404

    def test_studies_returns_data(self, client):
        """Returns study cards for run with result."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/studies")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["record_id"] == "rec_001"
        assert data[0]["study_type"] == "rct"


# =============================================================================
# PRISMA ENDPOINT TESTS
# =============================================================================


class TestPrismaEndpoint:
    """Tests for GET /runs/{run_id}/prisma endpoint."""

    def test_prisma_not_found(self, client):
        """Non-existent run returns 404."""
        response = client.get("/api/v1/runs/nonexistent/prisma")
        assert response.status_code == 404

    def test_prisma_returns_data(self, client):
        """Returns PRISMA counts for run with result."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/prisma")
        assert response.status_code == 200
        data = response.json()
        assert data["records_identified"] == 100
        assert data["studies_included"] == 15

    def test_prisma_default_counts(self, client):
        """Returns default counts when no prisma data."""
        state = MockCDRState(run_id="test-run", prisma_counts=None)
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/prisma")
        assert response.status_code == 200
        data = response.json()
        assert data["records_identified"] == 0


# =============================================================================
# SEARCH PLAN ENDPOINT TESTS
# =============================================================================


class TestSearchPlanEndpoint:
    """Tests for GET /runs/{run_id}/search-plan endpoint."""

    def test_search_plan_not_found(self, client):
        """Non-existent run returns 404."""
        response = client.get("/api/v1/runs/nonexistent/search-plan")
        assert response.status_code == 404

    def test_search_plan_returns_data(self, client):
        """Returns search plan for run with result."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/search-plan")
        assert response.status_code == 200
        data = response.json()
        assert data["pubmed_query"] == "treatment A condition X"

    def test_search_plan_not_available(self, client):
        """Returns 400 when search plan not available."""
        state = MockCDRState(run_id="test-run", search_plan=None)
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.RUNNING.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/search-plan")
        assert response.status_code == 400


# =============================================================================
# HYPOTHESES ENDPOINT TESTS
# =============================================================================


class TestHypothesesEndpoint:
    """Tests for GET /runs/{run_id}/hypotheses endpoint."""

    def test_hypotheses_not_found(self, client):
        """Non-existent run returns 404."""
        response = client.get("/api/v1/runs/nonexistent/hypotheses")
        assert response.status_code == 404

    def test_hypotheses_empty(self, client):
        """Returns empty list when no hypotheses."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/hypotheses")
        assert response.status_code == 200
        assert response.json() == []

    def test_hypotheses_with_data(self, client):
        """Returns hypotheses when present."""
        state = make_complete_state("test-run")
        state.composed_hypotheses = [make_mock_hypothesis()]
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/hypotheses")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["hypothesis_id"] == "hyp_001"


# =============================================================================
# PICO ENDPOINT TESTS
# =============================================================================


class TestPicoEndpoint:
    """Tests for GET /runs/{run_id}/pico endpoint."""

    def test_pico_not_found(self, client):
        """Non-existent run returns 404."""
        response = client.get("/api/v1/runs/nonexistent/pico")
        assert response.status_code == 404

    def test_pico_returns_data(self, client):
        """Returns PICO for run with result."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/pico")
        assert response.status_code == 200
        data = response.json()
        assert data["population"] == "Adults"
        assert data["intervention"] == "Treatment A"

    def test_pico_not_available(self, client):
        """Returns 400 when PICO not available."""
        state = MockCDRState(run_id="test-run", pico=None)
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.RUNNING.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/pico")
        assert response.status_code == 400


# =============================================================================
# RUN DETAIL ENDPOINT TESTS
# =============================================================================


class TestRunDetailEndpoint:
    """Tests for GET /runs/{run_id}/detail endpoint."""

    def test_detail_not_found(self, client):
        """Non-existent run returns 404."""
        response = client.get("/api/v1/runs/nonexistent/detail")
        assert response.status_code == 404

    def test_detail_returns_complete_data(self, client):
        """Returns full detail for run with result."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "research_question": "Test question",
            "result": state,
            "errors": [],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T01:00:00Z",
        }

        response = client.get("/api/v1/runs/test-run/detail")
        assert response.status_code == 200
        data = response.json()

        assert data["run_id"] == "test-run"
        assert data["status"] == "completed"
        assert data["claims_count"] == 1
        assert data["snippets_count"] == 1
        assert data["studies_count"] == 1

    def test_detail_with_no_result(self, client):
        """Returns detail even without result."""
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.PENDING.value,
            "result": None,
            "errors": [],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }

        response = client.get("/api/v1/runs/test-run/detail")
        assert response.status_code == 200
        data = response.json()
        assert data["claims_count"] == 0


# =============================================================================
# CREATE RUN ENDPOINT TESTS
# =============================================================================


class TestCreateRunEndpoint:
    """Tests for POST /runs endpoint."""

    def test_create_run_validation(self, client):
        """Validates request parameters."""
        # Question too short
        response = client.post(
            "/api/v1/runs",
            json={"research_question": "short"},
        )
        assert response.status_code == 422

    def test_create_run_returns_accepted(self, client):
        """Valid request returns 202 with run_id."""
        with patch("cdr.api.routes._execute_run", new=AsyncMock()):
            response = client.post(
                "/api/v1/runs",
                json={
                    "research_question": "What is the efficacy of treatment X for condition Y in adults?",
                    "max_results": 50,
                },
            )

        assert response.status_code == 202
        data = response.json()
        assert "run_id" in data
        assert data["status"] == "pending"

    def test_create_run_adds_to_runs_dict(self, client):
        """Created run is added to _runs."""
        with patch("cdr.api.routes._execute_run", new=AsyncMock()):
            response = client.post(
                "/api/v1/runs",
                json={
                    "research_question": "What is the efficacy of treatment X for condition Y in adults?",
                },
            )

        run_id = response.json()["run_id"]
        assert run_id in _runs
        assert _runs[run_id]["status"] == "pending"


# =============================================================================
# CANCEL RUN ENDPOINT TESTS
# =============================================================================


class TestCancelRunEndpoint:
    """Tests for DELETE /runs/{run_id} endpoint."""

    def test_cancel_not_found(self, client):
        """Non-existent run returns 404."""
        response = client.delete("/api/v1/runs/nonexistent")
        assert response.status_code == 404

    def test_cancel_running_run(self, client):
        """Running run can be cancelled."""
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.RUNNING.value,
            "errors": [],
        }

        response = client.delete("/api/v1/runs/test-run")
        assert response.status_code == 204
        assert _runs["test-run"]["status"] == "cancelled"

    def test_cancel_completed_fails(self, client):
        """Completed run cannot be cancelled."""
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "errors": [],
        }

        response = client.delete("/api/v1/runs/test-run")
        assert response.status_code == 400


# =============================================================================
# RESPONSE SCHEMA VALIDATION TESTS
# =============================================================================


class TestResponseSchemas:
    """Tests to validate response schemas match expected contracts."""

    def test_claim_response_has_required_fields(self, client):
        """ClaimResponse includes all required fields."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/claims")
        assert response.status_code == 200
        data = response.json()[0]

        # Check required fields per audit
        assert "claim_id" in data
        assert "claim_text" in data
        assert "certainty" in data
        assert "supporting_snippet_ids" in data
        assert "verification_status" in data

    def test_snippet_response_has_source_ref_fields(self, client):
        """SnippetResponse includes source_ref fields for traceability."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/snippets")
        assert response.status_code == 200
        data = response.json()[0]

        assert "record_id" in data
        assert "pmid" in data
        assert "section" in data

    def test_study_card_response_has_outcomes(self, client):
        """StudyCardResponse includes outcomes."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/studies")
        assert response.status_code == 200
        data = response.json()[0]

        assert "outcomes" in data
        assert len(data["outcomes"]) > 0
        assert data["outcomes"][0]["name"] == "Primary outcome"

    def test_run_detail_has_all_counts(self, client):
        """RunDetailResponse includes all count fields."""
        state = make_complete_state("test-run")
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED.value,
            "result": state,
            "errors": [],
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/detail")
        assert response.status_code == 200
        data = response.json()

        assert "claims_count" in data
        assert "snippets_count" in data
        assert "studies_count" in data
        assert "hypotheses_count" in data
        assert "verification_coverage" in data


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling across endpoints."""

    def test_404_includes_detail(self, client):
        """404 responses include detail message."""
        response = client.get("/api/v1/runs/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "nonexistent" in data["detail"]

    def test_400_for_invalid_pico(self, client):
        """400 responses include detail message for missing PICO."""
        # Use PICO endpoint which still returns 400 for missing data
        _runs["test-run"] = {
            "run_id": "test-run",
            "status": RunStatus.RUNNING.value,
            "result": MockCDRState(pico=None),  # No PICO available
            "errors": [],
            "created_at": "",
            "updated_at": "",
        }

        response = client.get("/api/v1/runs/test-run/pico")
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_422_for_invalid_request(self, client):
        """422 for invalid request body."""
        response = client.post(
            "/api/v1/runs",
            json={"research_question": "x"},  # Too short
        )
        assert response.status_code == 422


# =============================================================================
# RUNSTORE INTEGRATION TESTS
#
# These tests validate that the API correctly reads from RunStore when
# configured, falling back to in-memory _runs when not.
# Refs: CDR_Integral_Audit_2026-01-20.md (API + RunStore integration)
# =============================================================================


class TestAPIRunStoreIntegration:
    """Tests for API + RunStore integration.

    Validates that API endpoints correctly read from persisted state.
    """

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary database path."""
        return tmp_path / "test_api.db"

    @pytest.fixture
    def run_store(self, temp_db):
        """Create RunStore instance."""
        from cdr.storage.run_store import RunStore

        return RunStore(db_path=temp_db)

    @pytest.fixture
    def client_with_store(self, app, run_store):
        """Create test client with RunStore configured."""
        from cdr.api.routes import configure_run_store

        configure_run_store(run_store)
        with TestClient(app) as c:
            yield c
        # Reset after test
        configure_run_store(None)

    def test_list_runs_includes_persisted(self, client_with_store, run_store):
        """List runs endpoint includes runs from RunStore."""
        # Create run in store
        run_store.create_run("persisted-run-1", {"population": "Test"})
        run_store.update_run_status(
            "persisted-run-1",
            RunStatus.COMPLETED,
            current_node="done",
        )

        response = client_with_store.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()

        # Find our persisted run
        run_ids = [r["run_id"] for r in data]
        assert "persisted-run-1" in run_ids

        # Verify data is correct
        persisted = next(r for r in data if r["run_id"] == "persisted-run-1")
        assert persisted["status"] == "completed"

    def test_get_run_status_from_store(self, client_with_store, run_store):
        """Get run status reads from RunStore."""
        run_store.create_run("store-run", {"population": "Adults"})
        run_store.update_run_status(
            "store-run",
            RunStatus.RUNNING,
            current_node="retrieve",
        )

        response = client_with_store.get("/api/v1/runs/store-run")
        assert response.status_code == 200
        data = response.json()

        assert data["run_id"] == "store-run"
        assert data["status"] == "running"

    def test_404_when_not_in_store_or_memory(self, client_with_store):
        """Returns 404 when run is in neither store nor memory."""
        response = client_with_store.get("/api/v1/runs/ghost-run")
        assert response.status_code == 404

    def test_memory_fallback_when_not_in_store(self, client_with_store, run_store):
        """Falls back to in-memory when not in store."""
        # Add to in-memory only
        _runs["memory-only-run"] = {
            "run_id": "memory-only-run",
            "status": RunStatus.PENDING.value,
            "progress": {"current_node": "pending", "percentage": 0},
            "errors": [],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }

        response = client_with_store.get("/api/v1/runs/memory-only-run")
        assert response.status_code == 200
        assert response.json()["status"] == "pending"

    def test_list_runs_merges_store_and_memory(self, client_with_store, run_store):
        """List runs returns both store and memory runs without duplicates."""
        # Create in store
        run_store.create_run("store-run", {"population": "A"})

        # Create in memory
        _runs["memory-run"] = {
            "run_id": "memory-run",
            "status": RunStatus.PENDING.value,
            "result": None,
            "created_at": "2024-01-02T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
        }

        response = client_with_store.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()

        run_ids = [r["run_id"] for r in data]
        assert "store-run" in run_ids
        assert "memory-run" in run_ids
        # No duplicates
        assert len(run_ids) == len(set(run_ids))

    def test_store_priority_over_memory(self, client_with_store, run_store):
        """Store data takes priority when same run_id exists in both."""
        run_id = "dual-run"

        # Create in store with completed status
        run_store.create_run(run_id, {"population": "Store"})
        run_store.update_run_status(run_id, RunStatus.COMPLETED)

        # Create in memory with different status (should be ignored)
        _runs[run_id] = {
            "run_id": run_id,
            "status": RunStatus.PENDING.value,
            "progress": {},
            "errors": [],
            "created_at": "",
            "updated_at": "",
        }

        response = client_with_store.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        # Store value takes priority
        assert response.json()["status"] == "completed"

    def test_persisted_records_accessible(self, client_with_store, run_store):
        """Records persisted in store are accessible via API."""
        run_store.create_run("records-run", {"population": "Test"})
        run_store.add_records(
            "records-run",
            [
                {"id": "rec-1", "source": "pubmed", "title": "Study 1"},
                {"id": "rec-2", "source": "pubmed", "title": "Study 2"},
            ],
        )

        # Verify via direct store access (API doesn't expose records yet)
        records = run_store.get_records("records-run")
        assert len(records) == 2

        # Run exists in API
        response = client_with_store.get("/api/v1/runs/records-run")
        assert response.status_code == 200

    def test_cancel_persisted_run(self, client_with_store, run_store):
        """Persisted run can be cancelled via RunStore."""
        run_id = "cancel-store-run"
        run_store.create_run(run_id, {"population": "Test"})
        run_store.update_run_status(run_id, RunStatus.RUNNING)

        # Verify it's running
        run = run_store.get_run(run_id)
        assert run["status"] == "running"

        # Cancel via API
        response = client_with_store.delete(f"/api/v1/runs/{run_id}")
        assert response.status_code == 204

        # Verify status changed
        run = run_store.get_run(run_id)
        assert run["status"] == "cancelled"
        assert run["error_message"] == "Cancelled by user"

    def test_cancel_completed_persisted_run_fails(self, client_with_store, run_store):
        """Completed persisted run cannot be cancelled."""
        run_id = "completed-store-run"
        run_store.create_run(run_id, {"population": "Test"})
        run_store.update_run_status(run_id, RunStatus.COMPLETED)

        response = client_with_store.delete(f"/api/v1/runs/{run_id}")
        assert response.status_code == 400
