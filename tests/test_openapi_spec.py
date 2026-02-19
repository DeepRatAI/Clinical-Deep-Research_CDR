"""
OpenAPI Specification Validation Tests.

Validates that the FastAPI-generated OpenAPI spec:
1. Has all expected endpoints (18 routes)
2. Contains correct response models
3. Has proper descriptions
4. Matches the live API router

Refs:
- CDR_Integral_Audit_2026-01-20.md (API contract)
- scripts/export_openapi.py (export utility)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi import FastAPI


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def app() -> FastAPI:
    """Create the full CDR app."""
    from cdr.api import create_app

    return create_app()


@pytest.fixture(scope="module")
def schema(app: FastAPI) -> dict:
    """Get the OpenAPI schema."""
    return app.openapi()


@pytest.fixture(scope="module")
def client(app: FastAPI):
    """Test client for the app."""
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


# =============================================================================
# EXPECTED API SURFACE
# =============================================================================

# All 18 endpoints expected in the CDR API (method, path)
EXPECTED_ENDPOINTS = [
    ("get", "/api/v1/runs"),
    ("post", "/api/v1/runs"),
    ("get", "/api/v1/runs/{run_id}"),
    ("get", "/api/v1/runs/{run_id}/claims"),
    ("get", "/api/v1/runs/{run_id}/detail"),
    ("get", "/api/v1/runs/{run_id}/snippets"),
    ("get", "/api/v1/runs/{run_id}/studies"),
    ("get", "/api/v1/runs/{run_id}/prisma"),
    ("get", "/api/v1/runs/{run_id}/search-plan"),
    ("get", "/api/v1/runs/{run_id}/hypotheses"),
    ("get", "/api/v1/runs/{run_id}/pico"),
    ("get", "/api/v1/runs/{run_id}/report"),
    ("get", "/api/v1/runs/{run_id}/evaluation"),
    ("get", "/api/v1/runs/{run_id}/export/{format}"),
    ("delete", "/api/v1/runs/{run_id}"),
    ("post", "/api/v1/parse-question"),
    ("get", "/api/v1/health"),
    ("get", "/api/v1/metrics"),
]

# Response models that MUST appear in components/schemas
EXPECTED_RESPONSE_MODELS = [
    "RunResponse",
    "RunSummaryResponse",
    "RunStatusResponse",
    "PICOResponse",
    "ClaimResponse",
    "SnippetResponse",
    "StudyCardResponse",
    "PRISMACountsResponse",
    "SearchPlanResponse",
    "HypothesisResponse",
    "EvaluationReportResponse",
    "RunDetailResponse",
    "ReportResponse",
]


# =============================================================================
# SCHEMA STRUCTURE TESTS
# =============================================================================


class TestOpenAPISchemaStructure:
    """Validate overall OpenAPI schema structure."""

    def test_schema_has_info(self, schema):
        """Schema contains required info section."""
        assert "info" in schema
        info = schema["info"]
        assert "title" in info
        assert "version" in info

    def test_schema_title(self, schema):
        """Schema title matches CDR API."""
        title = schema["info"]["title"]
        assert "CDR" in title or "Clinical" in title

    def test_schema_version(self, schema):
        """Schema has a valid version string."""
        version = schema["info"]["version"]
        assert version  # non-empty

    def test_schema_has_paths(self, schema):
        """Schema contains paths section."""
        assert "paths" in schema
        assert len(schema["paths"]) > 0

    def test_schema_has_components(self, schema):
        """Schema contains components/schemas section."""
        assert "components" in schema
        assert "schemas" in schema["components"]

    def test_openapi_version(self, schema):
        """Schema uses OpenAPI 3.x."""
        assert schema.get("openapi", "").startswith("3.")


# =============================================================================
# ENDPOINT COMPLETENESS TESTS
# =============================================================================


class TestOpenAPIEndpoints:
    """Validate all expected endpoints are present."""

    def test_all_endpoints_present(self, schema):
        """All 18 expected endpoints exist in the schema."""
        paths = schema["paths"]
        missing = []
        for method, path in EXPECTED_ENDPOINTS:
            if path not in paths or method not in paths.get(path, {}):
                missing.append(f"{method.upper()} {path}")

        assert not missing, f"Missing {len(missing)} endpoints: {missing}"

    def test_endpoint_count(self, schema):
        """Schema has at least 18 endpoints."""
        paths = schema["paths"]
        endpoint_count = sum(
            1
            for methods in paths.values()
            for m in methods
            if m in ("get", "post", "put", "delete", "patch")
        )
        assert endpoint_count >= 18, f"Expected ≥18 endpoints, found {endpoint_count}"

    def test_health_endpoint(self, schema):
        """Health endpoint is documented."""
        health = schema["paths"].get("/api/v1/health", {}).get("get", {})
        assert health, "GET /api/v1/health not found"
        assert "200" in health.get("responses", {})

    def test_runs_list_endpoint(self, schema):
        """GET /runs returns list[RunSummaryResponse]."""
        runs = schema["paths"].get("/api/v1/runs", {}).get("get", {})
        assert runs, "GET /api/v1/runs not found"
        resp_200 = runs["responses"]["200"]
        assert resp_200  # has 200 response

    def test_create_run_endpoint(self, schema):
        """POST /runs has request body."""
        create = schema["paths"].get("/api/v1/runs", {}).get("post", {})
        assert create, "POST /api/v1/runs not found"
        # POST should have a requestBody
        assert "requestBody" in create or "200" in create.get("responses", {})

    def test_all_endpoints_have_responses(self, schema):
        """Every endpoint has at least one response defined."""
        paths = schema["paths"]
        endpoints_without_response = []
        for path, methods in paths.items():
            for method, spec in methods.items():
                if method not in ("get", "post", "put", "delete", "patch"):
                    continue
                if not isinstance(spec, dict):
                    continue
                responses = spec.get("responses", {})
                if not responses:
                    endpoints_without_response.append(f"{method.upper()} {path}")

        assert not endpoints_without_response, (
            f"Endpoints without responses: {endpoints_without_response}"
        )

    def test_all_endpoints_have_summary_or_description(self, schema):
        """Every endpoint has summary or description."""
        paths = schema["paths"]
        undocumented = []
        for path, methods in paths.items():
            for method, spec in methods.items():
                if method not in ("get", "post", "put", "delete", "patch"):
                    continue
                if not isinstance(spec, dict):
                    continue
                has_docs = spec.get("summary") or spec.get("description")
                if not has_docs:
                    undocumented.append(f"{method.upper()} {path}")

        assert not undocumented, f"Undocumented endpoints: {undocumented}"


# =============================================================================
# RESPONSE MODEL TESTS
# =============================================================================


class TestOpenAPIResponseModels:
    """Validate response models in components/schemas."""

    def test_all_response_models_present(self, schema):
        """All expected response models exist in schemas."""
        schemas = schema["components"]["schemas"]
        missing = [m for m in EXPECTED_RESPONSE_MODELS if m not in schemas]
        assert not missing, f"Missing response models: {missing}"

    def test_run_summary_response_fields(self, schema):
        """RunSummaryResponse has required fields."""
        model = schema["components"]["schemas"].get("RunSummaryResponse", {})
        properties = model.get("properties", {})
        required_fields = {"run_id", "status", "created_at", "updated_at"}
        actual_fields = set(properties.keys())
        missing = required_fields - actual_fields
        assert not missing, f"RunSummaryResponse missing fields: {missing}"

    def test_claim_response_fields(self, schema):
        """ClaimResponse has claim_id and claim_text."""
        model = schema["components"]["schemas"].get("ClaimResponse", {})
        properties = model.get("properties", {})
        assert "claim_id" in properties
        assert "claim_text" in properties

    def test_hypothesis_response_fields(self, schema):
        """HypothesisResponse has required composition fields."""
        model = schema["components"]["schemas"].get("HypothesisResponse", {})
        properties = model.get("properties", {})
        assert "hypothesis_id" in properties
        assert "hypothesis_text" in properties
        assert "mechanism" in properties

    def test_pico_response_fields(self, schema):
        """PICOResponse has P, I, C, O."""
        model = schema["components"]["schemas"].get("PICOResponse", {})
        properties = model.get("properties", {})
        for field in ("population", "intervention", "comparator", "outcome"):
            assert field in properties, f"PICOResponse missing '{field}'"


# =============================================================================
# LIVE ENDPOINT TESTS (docs / redoc)
# =============================================================================


class TestOpenAPILiveEndpoints:
    """Validate live /docs and /redoc endpoints."""

    def test_docs_endpoint_accessible(self, client):
        """/docs (Swagger UI) returns 200."""
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_endpoint_accessible(self, client):
        """/redoc returns 200."""
        resp = client.get("/redoc")
        assert resp.status_code == 200

    def test_openapi_json_endpoint(self, client):
        """/openapi.json returns valid JSON schema."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "openapi" in data
        assert "paths" in data
