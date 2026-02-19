"""
CDR API Routes

FastAPI routes for the Clinical Deep Research system.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from cdr.core.enums import RunStatus
from cdr.core.schemas import CDRState, PICO
from cdr.observability.tracer import tracer
from cdr.observability.metrics import metrics
from cdr.storage.run_store import RunStore

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class RunRequest(BaseModel):
    """Request to start a new CDR run."""

    research_question: str = Field(
        ...,
        description="The clinical research question to investigate",
        min_length=10,
        max_length=2000,
    )
    max_results: int = Field(
        default=100,
        description="Maximum records to retrieve per source",
        ge=10,
        le=500,
    )
    output_formats: list[str] = Field(
        default=["markdown", "json"],
        description="Output formats for the report",
    )
    model: str | None = Field(
        default=None,
        description="LLM model to use. If None, uses provider-specific default (Gemini: gemini-2.5-flash, Groq: llama-3.3-70b-versatile)",
    )
    dod_level: int = Field(
        default=1,
        description="Definition of Done level (1=Basic, 2=Full, 3=Research)",
        ge=1,
        le=3,
    )


class RunResponse(BaseModel):
    """Response after starting a run."""

    run_id: str
    status: str
    message: str


class RunStatusResponse(BaseModel):
    """Response for run status check."""

    run_id: str
    status: str
    progress: dict[str, Any]
    errors: list[str]
    report_path: str | None = None


class PICOResponse(BaseModel):
    """Parsed PICO response."""

    population: str
    intervention: str
    comparator: str | None
    outcome: str
    study_types: list[str] | None


class ClaimResponse(BaseModel):
    """Evidence claim response.

    Refs: EvidenceClaim schema, CDR_Integral_Audit_2026-01-20.md CRITICAL-3
    """

    claim_id: str
    claim_text: str  # Correct field from EvidenceClaim
    certainty: str
    supporting_snippet_ids: list[str]  # Correct field from EvidenceClaim
    verification_status: str | None
    grade_rationale: dict[str, str] | None = None  # Optional for GRADE traceability


class ReportResponse(BaseModel):
    """Report metadata response."""

    run_id: str
    generated_at: str
    formats_available: list[str]
    download_urls: dict[str, str]


class RunSummaryResponse(BaseModel):
    """Summary of a run for listing."""

    run_id: str
    status: str
    status_reason: str | None = None
    dod_level: int = 0
    claims_count: int = 0
    verification_coverage: float = 0.0
    created_at: str
    updated_at: str


class SnippetResponse(BaseModel):
    """Evidence snippet response."""

    snippet_id: str
    record_id: str
    pmid: str | None = None
    doi: str | None = None
    nct_id: str | None = None
    title: str
    authors: list[str] = []
    publication_year: int | None = None
    journal: str | None = None
    url: str | None = None
    section: str
    text: str
    offset_start: int | None = None
    offset_end: int | None = None
    relevance_score: float | None = None


class StudyCardResponse(BaseModel):
    """Study card response."""

    study_id: str
    record_id: str
    title: str
    study_type: str
    population: str | None = None
    intervention: str | None = None
    comparator: str | None = None
    outcomes: list[dict[str, Any]] = []
    sample_size: int | None = None
    follow_up_duration: str | None = None


class PRISMACountsResponse(BaseModel):
    """PRISMA flow counts response."""

    records_identified: int = 0
    records_screened: int = 0
    records_excluded_screening: int = 0
    reports_assessed: int = 0
    reports_not_retrieved: int = 0
    studies_included: int = 0
    exclusion_reasons: dict[str, int] = {}


class SearchPlanResponse(BaseModel):
    """Search strategy response (PRISMA-S compliant)."""

    pubmed_query: str
    ct_gov_query: str
    date_range: list[str] | None = None
    languages: list[str] = []
    max_results_per_source: int = 100
    created_at: str | None = None


class HypothesisResponse(BaseModel):
    """Composed hypothesis response."""

    hypothesis_id: str
    claim_a_id: str
    claim_b_id: str
    hypothesis_text: str
    mechanism: str
    rival_hypotheses: list[str] = []
    threats_to_validity: list[str] = []
    mcid: str | None = None
    test_design: str | None = None
    confidence: float = 0.0


class EvaluationDimensionResponse(BaseModel):
    """Single evaluation dimension response."""

    name: str
    score: float
    grade: str
    rationale: str


class EvaluationReportResponse(BaseModel):
    """Full evaluation report response."""

    run_id: str
    overall_score: float
    overall_grade: str
    dimensions: list[EvaluationDimensionResponse]
    strengths: list[str] = []
    weaknesses: list[str] = []
    recommendations: list[str] = []
    created_at: str | None = None


class RunDetailResponse(BaseModel):
    """Full run detail response."""

    run_id: str
    status: str
    status_reason: str | None = None
    dod_level: int = 0
    pico: PICOResponse | None = None
    search_plan: SearchPlanResponse | None = None
    prisma_counts: PRISMACountsResponse | None = None
    claims_count: int = 0
    snippets_count: int = 0
    studies_count: int = 0
    hypotheses_count: int = 0
    verification_coverage: float = 0.0
    errors: list[str] = []
    created_at: str
    updated_at: str


# =============================================================================
# IN-MEMORY STATE (for demo; use Redis/DB in production)
# =============================================================================

from datetime import datetime

_runs: dict[str, dict] = {}


# =============================================================================
# DEPENDENCY INJECTION FOR PERSISTENCE
# =============================================================================

# Global run store instance (set via configure_run_store)
_run_store: Optional[RunStore] = None


def configure_run_store(store: Optional[RunStore]) -> None:
    """Configure the global RunStore for persistence.

    Call this at application startup to enable SQLite persistence.
    If not configured, falls back to in-memory _runs dict.

    Args:
        store: RunStore instance or None to disable persistence
    """
    global _run_store
    _run_store = store


def get_run_store() -> Optional[RunStore]:
    """Dependency to get the current RunStore.

    Returns:
        RunStore if configured, None otherwise
    """
    return _run_store


def _get_run(run_id: str) -> Optional[dict]:
    """Get run from store or in-memory cache.

    Tries RunStore first if configured, falls back to _runs.

    Args:
        run_id: Run identifier

    Returns:
        Run data dict or None if not found
    """
    # Try persistent store first
    if _run_store is not None:
        run = _run_store.get_run(run_id)
        if run is not None:
            return run

    # Fall back to in-memory
    return _runs.get(run_id)


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(prefix="/api/v1", tags=["cdr"])


# =============================================================================
# LIST RUNS ENDPOINT
# =============================================================================


@router.get("/runs", response_model=list[RunSummaryResponse])
async def list_runs() -> list[RunSummaryResponse]:
    """List all CDR runs.

    Returns:
        List of run summaries (from RunStore + in-memory)
    """
    summaries = []
    seen_run_ids = set()

    # First, get runs from persistent store if configured
    if _run_store is not None:
        for run in _run_store.list_runs():
            run_id = run["run_id"]
            seen_run_ids.add(run_id)
            summaries.append(
                RunSummaryResponse(
                    run_id=run_id,
                    status=run["status"],
                    status_reason=run.get("status_reason"),
                    dod_level=run.get("dod_level", 0) or 0,
                    claims_count=run.get("claims_count", 0) or 0,
                    verification_coverage=run.get("verification_coverage", 0.0) or 0.0,
                    created_at=run.get("created_at", ""),
                    updated_at=run.get("updated_at", ""),
                )
            )

    # Then, add in-memory runs not in persistent store
    for run_id, run in _runs.items():
        if run_id in seen_run_ids:
            continue

        result = run.get("result")

        # FIX-B1: Always get dod_level from run dict (set at creation time)
        # Not from result.report_data which may not exist
        dod_level = run.get("dod_level", 1)  # Default to 1, not 0

        # Calculate metrics from result if available
        claims_count = 0
        verification_coverage = 0.0
        status_reason = None

        if result:
            claims_count = len(result.claims) if result.claims else 0
            # Calculate verification coverage
            if result.verification and claims_count > 0:
                verified_count = sum(
                    1
                    for v in result.verification
                    if v.overall_status.value in ("VERIFIED", "PARTIAL")
                )
                verification_coverage = verified_count / claims_count
            status_reason = getattr(result, "status_reason", None)

        summaries.append(
            RunSummaryResponse(
                run_id=run_id,
                status=run["status"],
                status_reason=status_reason,
                dod_level=dod_level,
                claims_count=claims_count,
                verification_coverage=verification_coverage,
                created_at=run.get("created_at", ""),
                updated_at=run.get("updated_at", ""),
            )
        )

    # Sort by updated_at descending
    summaries.sort(key=lambda x: x.updated_at, reverse=True)
    return summaries


@router.post("/runs", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    request: RunRequest,
    background_tasks: BackgroundTasks,
) -> RunResponse:
    """Start a new CDR systematic review run.

    This endpoint initiates an asynchronous systematic review process.
    The run will execute in the background and can be monitored via the
    status endpoint.

    Args:
        request: Run configuration
        background_tasks: FastAPI background task handler

    Returns:
        Run ID and initial status
    """
    with tracer.start_span("api.create_run") as span:
        run_id = str(uuid4())[:8]
        span.set_attribute("run_id", run_id)

        now = datetime.utcnow().isoformat() + "Z"

        # Initialize run state
        _runs[run_id] = {
            "run_id": run_id,
            "status": RunStatus.PENDING.value,
            "research_question": request.research_question,
            "progress": {"current_node": "pending", "percentage": 0},
            "errors": [],
            "result": None,
            "created_at": now,
            "updated_at": now,
            "dod_level": request.dod_level,
            "model": request.model,
        }

        # Schedule background execution
        background_tasks.add_task(
            _execute_run,
            run_id,
            request.research_question,
            request.max_results,
            request.output_formats,
            request.model,
            request.dod_level,
        )

        metrics.counter("cdr.api.runs_created")

        return RunResponse(
            run_id=run_id,
            status=RunStatus.PENDING.value,
            message=f"Run {run_id} created and queued for execution",
        )


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str) -> RunStatusResponse:
    """Get the status of a CDR run.

    Args:
        run_id: The run ID to check

    Returns:
        Current run status and progress
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    return RunStatusResponse(
        run_id=run_id,
        status=run["status"],
        progress=run.get("progress", {}),
        errors=run.get("errors", []),
        report_path=run.get("report_path"),
    )


@router.get("/runs/{run_id}/claims", response_model=list[ClaimResponse])
async def get_run_claims(run_id: str) -> list[ClaimResponse]:
    """Get the evidence claims from a completed run.

    Args:
        run_id: The run ID

    Returns:
        List of evidence claims with verification status
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    # FIX-B4: Allow reading claims for ALL states to enable viewing partial/intermediate results
    # PENDING returns empty list (hasn't started), RUNNING returns intermediate data
    # Terminal states (COMPLETED, INSUFFICIENT_EVIDENCE, UNPUBLISHABLE) return final data
    # This is more user-friendly than returning errors
    # Refs: CDR_Integral_Audit_2026-01-20.md CRITICO-B

    result = run.get("result")
    if not result or not result.claims:
        return []

    # CRITICO-A fix: CDRState.verification is a list[VerificationResult], not a dict
    # Build lookup by claim_id for efficient access
    # Refs: CDR_Integral_Audit_2026-01-20.md CRITICO-A
    verification_by_claim: dict = {}
    if result.verification:
        for vr in result.verification:
            verification_by_claim[vr.claim_id] = vr

    claims = []
    for claim in result.claims:
        ver_result = verification_by_claim.get(claim.claim_id)
        claims.append(
            ClaimResponse(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,  # Correct field from EvidenceClaim
                certainty=claim.certainty.value,
                supporting_snippet_ids=claim.supporting_snippet_ids,  # Correct field
                verification_status=ver_result.overall_status.value if ver_result else None,
                grade_rationale=claim.grade_rationale,  # GRADE traceability
            )
        )

    return claims


@router.get("/runs/{run_id}/detail", response_model=RunDetailResponse)
async def get_run_detail(run_id: str) -> RunDetailResponse:
    """Get full detail of a CDR run.

    Args:
        run_id: The run ID

    Returns:
        Full run detail including counts and summaries
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    result = run.get("result")

    # Build response with available data
    pico_response = None
    search_plan_response = None
    prisma_response = None
    claims_count = 0
    snippets_count = 0
    studies_count = 0
    hypotheses_count = 0
    verification_coverage = 0.0
    # Get dod_level from run (set at creation time) - default to 1 if missing
    dod_level = run.get("dod_level", 1)
    status_reason = None

    if result:
        # PICO
        if result.pico:
            pico_response = PICOResponse(
                population=result.pico.population,
                intervention=result.pico.intervention,
                comparator=result.pico.comparator,
                outcome=result.pico.outcome,
                study_types=[str(st) for st in result.pico.study_types]
                if result.pico.study_types
                else None,
            )

        # Search plan
        if result.search_plan:
            sp = result.search_plan
            search_plan_response = SearchPlanResponse(
                pubmed_query=sp.pubmed_query,
                ct_gov_query=sp.ct_gov_query,
                date_range=list(sp.date_range) if sp.date_range else None,
                languages=sp.languages or [],
                max_results_per_source=sp.max_results_per_source,
                created_at=sp.created_at.isoformat()
                if hasattr(sp, "created_at") and sp.created_at
                else None,
            )

        # PRISMA counts
        if result.prisma_counts:
            pc = result.prisma_counts
            prisma_response = PRISMACountsResponse(
                records_identified=pc.records_identified,
                records_screened=pc.records_screened,
                # Schema uses 'records_excluded', API response uses 'records_excluded_screening'
                records_excluded_screening=pc.records_excluded,
                reports_assessed=pc.reports_assessed,
                reports_not_retrieved=pc.reports_not_retrieved,
                studies_included=pc.studies_included,
                exclusion_reasons=pc.exclusion_reasons or {},
            )

        # Counts
        claims_count = len(result.claims) if result.claims else 0
        snippets_count = len(result.snippets) if result.snippets else 0
        studies_count = len(result.study_cards) if result.study_cards else 0
        hypotheses_count = (
            len(result.composed_hypotheses)
            if hasattr(result, "composed_hypotheses") and result.composed_hypotheses
            else 0
        )

        # Verification coverage
        if result.verification and claims_count > 0:
            verified_count = sum(
                1 for v in result.verification if v.overall_status.value in ("VERIFIED", "PARTIAL")
            )
            verification_coverage = verified_count / claims_count

        # DoD level and status reason - use run's dod_level as primary source
        dod_level = run.get("dod_level", 1)  # Use stored dod_level from run creation
        if hasattr(result, "report_data") and result.report_data:
            # Override with report data if available
            report_dod = getattr(result.report_data, "dod_level", None)
            if report_dod is not None:
                dod_level = report_dod
        status_reason = getattr(result, "status_reason", None)

    return RunDetailResponse(
        run_id=run_id,
        status=run["status"],
        status_reason=status_reason,
        dod_level=dod_level,
        pico=pico_response,
        search_plan=search_plan_response,
        prisma_counts=prisma_response,
        claims_count=claims_count,
        snippets_count=snippets_count,
        studies_count=studies_count,
        hypotheses_count=hypotheses_count,
        verification_coverage=verification_coverage,
        errors=run.get("errors", []),
        created_at=run.get("created_at", ""),
        updated_at=run.get("updated_at", ""),
    )


@router.get("/runs/{run_id}/snippets", response_model=list[SnippetResponse])
async def get_run_snippets(run_id: str) -> list[SnippetResponse]:
    """Get all snippets from a run.

    Args:
        run_id: The run ID

    Returns:
        List of evidence snippets
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    result = run.get("result")

    if not result or not result.snippets:
        return []

    # Build record lookup for bibliographic metadata
    records_by_id: dict[str, Any] = {}
    if result.retrieved_records:
        for rec in result.retrieved_records:
            records_by_id[rec.record_id] = rec

    snippets = []
    for snippet in result.snippets:
        sr = snippet.source_ref
        # Get bibliographic info from parent Record
        record = records_by_id.get(sr.record_id)

        snippets.append(
            SnippetResponse(
                snippet_id=snippet.snippet_id,
                record_id=sr.record_id,
                pmid=sr.pmid,
                doi=sr.doi,
                nct_id=sr.nct_id,
                # Bibliographic fields come from Record, not SourceRef
                title=record.title if record else "Unknown",
                authors=list(record.authors) if record and record.authors else [],
                publication_year=record.year if record else None,
                journal=record.journal if record else None,
                url=record.url if record else None,
                section=snippet.section.value
                if hasattr(snippet.section, "value")
                else str(snippet.section),
                text=snippet.text,
                # Offsets are in SourceRef, not Snippet directly
                offset_start=sr.offset_start,
                offset_end=sr.offset_end,
                # relevance_score not currently tracked in Snippet schema
                relevance_score=None,
            )
        )

    return snippets


@router.get("/runs/{run_id}/studies", response_model=list[StudyCardResponse])
async def get_run_studies(run_id: str) -> list[StudyCardResponse]:
    """Get all study cards from a run.

    Args:
        run_id: The run ID

    Returns:
        List of study cards
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    result = run.get("result")

    if not result or not result.study_cards:
        return []

    # Build record lookup for title
    records_by_id: dict[str, Any] = {}
    if result.retrieved_records:
        for rec in result.retrieved_records:
            records_by_id[rec.record_id] = rec

    studies = []
    for card in result.study_cards:
        outcomes = []
        if card.outcomes:
            for o in card.outcomes:
                # Build confidence_interval string from ci_lower/ci_upper
                ci_str = None
                if o.ci_lower is not None and o.ci_upper is not None:
                    ci_str = f"[{o.ci_lower}, {o.ci_upper}]"

                outcomes.append(
                    {
                        "name": o.name,
                        "value": o.value,
                        "unit": None,  # Not in OutcomeMeasure schema
                        "effect_size": o.value,  # Use value as effect_size
                        "confidence_interval": ci_str,
                        "p_value": o.p_value,
                        "is_significant": o.p_value < 0.05 if o.p_value is not None else None,
                        "direction": None,  # Not in OutcomeMeasure schema
                    }
                )

        # Get title from parent Record
        record = records_by_id.get(card.record_id)
        title = record.title if record else "Unknown Study"

        studies.append(
            StudyCardResponse(
                study_id=card.record_id,  # Use record_id as study_id
                record_id=card.record_id,
                title=title,
                study_type=card.study_type.value
                if hasattr(card.study_type, "value")
                else str(card.study_type),
                population=card.population_extracted,  # Correct field name
                intervention=card.intervention_extracted,  # Correct field name
                comparator=card.comparator_extracted,  # Correct field name
                outcomes=outcomes,
                sample_size=card.sample_size,
                follow_up_duration=card.follow_up_duration,
            )
        )

    return studies


@router.get("/runs/{run_id}/prisma", response_model=PRISMACountsResponse)
async def get_run_prisma(run_id: str) -> PRISMACountsResponse:
    """Get PRISMA flow counts for a run.

    Args:
        run_id: The run ID

    Returns:
        PRISMA flow counts
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    result = run.get("result")

    if not result or not result.prisma_counts:
        return PRISMACountsResponse()

    pc = result.prisma_counts
    return PRISMACountsResponse(
        records_identified=pc.records_identified,
        records_screened=pc.records_screened,
        # Schema uses 'records_excluded', API response uses 'records_excluded_screening'
        records_excluded_screening=pc.records_excluded,
        reports_assessed=pc.reports_assessed,
        reports_not_retrieved=pc.reports_not_retrieved,
        studies_included=pc.studies_included,
        exclusion_reasons=pc.exclusion_reasons or {},
    )


@router.get("/runs/{run_id}/search-plan", response_model=SearchPlanResponse)
async def get_run_search_plan(run_id: str) -> SearchPlanResponse:
    """Get search strategy (PRISMA-S) for a run.

    Args:
        run_id: The run ID

    Returns:
        Search plan details
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    result = run.get("result")

    if not result or not result.search_plan:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Search plan not yet available for run {run_id}",
        )

    sp = result.search_plan
    return SearchPlanResponse(
        pubmed_query=sp.pubmed_query,
        ct_gov_query=sp.ct_gov_query,
        date_range=list(sp.date_range) if sp.date_range else None,
        languages=sp.languages or [],
        max_results_per_source=sp.max_results_per_source,
        created_at=sp.created_at.isoformat()
        if hasattr(sp, "created_at") and sp.created_at
        else None,
    )


@router.get("/runs/{run_id}/hypotheses", response_model=list[HypothesisResponse])
async def get_run_hypotheses(run_id: str) -> list[HypothesisResponse]:
    """Get composed hypotheses from a run (DoD Level 3).

    Args:
        run_id: The run ID

    Returns:
        List of composed hypotheses
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    result = run.get("result")

    if not result or not hasattr(result, "composed_hypotheses") or not result.composed_hypotheses:
        return []

    hypotheses = []
    for h in result.composed_hypotheses:
        # composed_hypotheses is list[dict], not objects
        hypotheses.append(
            HypothesisResponse(
                hypothesis_id=h.get("hypothesis_id", ""),
                claim_a_id=h.get("claim_a_id", ""),
                claim_b_id=h.get("claim_b_id", ""),
                hypothesis_text=h.get("hypothesis_text", ""),
                mechanism=h.get("mechanism"),
                rival_hypotheses=h.get("rival_hypotheses") or [],
                threats_to_validity=h.get("threats_to_validity") or [],
                mcid=h.get("mcid"),
                test_design=h.get("test_design"),
                confidence=h.get("confidence"),
            )
        )

    return hypotheses


@router.get("/runs/{run_id}/pico", response_model=PICOResponse)
async def get_run_pico(run_id: str) -> PICOResponse:
    """Get the parsed PICO for a run.

    Args:
        run_id: The run ID

    Returns:
        Parsed PICO components
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    result = run.get("result")

    if not result or not result.pico:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"PICO not yet available for run {run_id}",
        )

    pico = result.pico
    return PICOResponse(
        population=pico.population,
        intervention=pico.intervention,
        comparator=pico.comparator,
        outcome=pico.outcome,
        study_types=[str(st) for st in pico.study_types] if pico.study_types else None,
    )


@router.get("/runs/{run_id}/report", response_model=ReportResponse)
async def get_run_report(run_id: str) -> ReportResponse:
    """Get report metadata for export.

    Args:
        run_id: The run ID

    Returns:
        Report metadata with download URLs
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    result = run.get("result")
    report_path = run.get("report_path")

    # Build download URLs based on available formats
    formats_available = []
    download_urls = {}

    if result and result.report and isinstance(result.report, dict):
        report_data = result.report
        file_path = report_data.get("file_path")
        if file_path:
            formats_available.append("json")
            download_urls["json"] = f"/api/v1/runs/{run_id}/export/json"
    elif report_path:
        formats_available.append("json")
        download_urls["json"] = f"/api/v1/runs/{run_id}/export/json"

    # Always offer markdown if we have claims
    if result and result.claims:
        formats_available.append("markdown")
        download_urls["markdown"] = f"/api/v1/runs/{run_id}/export/markdown"

    # Always offer PDF if we have claims (rendered from markdown)
    if result and result.claims:
        formats_available.append("pdf")
        download_urls["pdf"] = f"/api/v1/runs/{run_id}/export/pdf"

    return ReportResponse(
        run_id=run_id,
        generated_at=run.get("updated_at", ""),
        formats_available=formats_available,
        download_urls=download_urls,
    )


@router.get("/runs/{run_id}/evaluation", response_model=EvaluationReportResponse)
async def get_run_evaluation(run_id: str) -> EvaluationReportResponse:
    """Get evaluation report for a run.

    Args:
        run_id: The run ID

    Returns:
        Evaluation report with scores and grades
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    # Check RunStore for persisted evaluation
    store = get_run_store()
    if store:
        eval_data = store.get_evaluation(run_id)
        if eval_data:
            report = eval_data["report"]
            return EvaluationReportResponse(
                run_id=run_id,
                overall_score=eval_data["overall_score"] or report.get("overall_score", 0.0),
                overall_grade=eval_data["grade"] or report.get("overall_grade", "F"),
                dimensions=[
                    EvaluationDimensionResponse(
                        name=d.get("name", "Unknown"),
                        score=d.get("score", 0.0),
                        grade=d.get("grade", "F"),
                        rationale=d.get("rationale", ""),
                    )
                    for d in report.get("dimensions", [])
                ],
                strengths=report.get("strengths", []),
                weaknesses=report.get("weaknesses", []),
                recommendations=report.get("recommendations", []),
                created_at=eval_data["created_at"],
            )

    # Check in-memory result for evaluation_report
    result = run.get("result")
    if result and hasattr(result, "evaluation_report") and result.evaluation_report:
        report = result.evaluation_report
        return EvaluationReportResponse(
            run_id=run_id,
            overall_score=report.overall_score,
            overall_grade=report.overall_grade,
            dimensions=[
                EvaluationDimensionResponse(
                    name=d.name,
                    score=d.score,
                    grade=d.grade,
                    rationale=d.rationale,
                )
                for d in report.dimensions
            ],
            strengths=report.strengths or [],
            weaknesses=report.weaknesses or [],
            recommendations=report.recommendations or [],
            created_at=report.created_at.isoformat() if hasattr(report, "created_at") else None,
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Evaluation not yet available for run {run_id}",
    )


@router.get("/runs/{run_id}/export/{format}")
async def export_run(run_id: str, format: str):
    """Export run results in specified format.

    Args:
        run_id: The run ID
        format: Export format (json or markdown)

    Returns:
        Exported content as file download
    """
    from fastapi.responses import Response
    import json

    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    result = run.get("result")
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run {run_id} has no results to export",
        )

    if format == "json":
        # Build comprehensive JSON export
        # Extract DoD3 enforcement fields from result.report (if available)
        dod3_fields = {}
        if hasattr(result, "report") and result.report:
            dod3_fields = {
                "gate_report": result.report.get("gate_report"),
                "dod3_enforcement": result.report.get("dod3_enforcement"),
                "dod3_excluded_records": result.report.get("dod3_excluded_records", []),
                "dod3_excluded_snippets": result.report.get("dod3_excluded_snippets", []),
                "dod3_suppressed_hypotheses": result.report.get("dod3_suppressed_hypotheses", []),
            }

        export_data = {
            "run_id": run_id,
            "status": run.get("status"),
            "created_at": run.get("created_at"),
            "updated_at": run.get("updated_at"),
            "pico": None,
            "search_plan": None,
            "prisma_counts": None,
            "claims": [],
            "snippets": [],
            "study_cards": [],
            "hypotheses": [],
            "answer": result.answer if hasattr(result, "answer") else None,
            # DoD3 ENFORCEMENT AUDIT TRAIL
            **dod3_fields,
        }

        # PICO
        if result.pico:
            # Clean comparator: handle literal "null" string from LLM
            comparator = result.pico.comparator
            if comparator in ("null", "None", ""):
                comparator = None

            # Clean study_types: extract enum .value to avoid "StudyType.RCT" prefix
            study_types_clean = None
            if result.pico.study_types:
                study_types_clean = [
                    st.value if hasattr(st, "value") else str(st).replace("StudyType.", "")
                    for st in result.pico.study_types
                ]

            # Extract comparator_source (new field for semantic coherence)
            comparator_source = None
            if hasattr(result.pico, "comparator_source") and result.pico.comparator_source:
                comparator_source = (
                    result.pico.comparator_source.value
                    if hasattr(result.pico.comparator_source, "value")
                    else str(result.pico.comparator_source)
                )

            export_data["pico"] = {
                "population": result.pico.population,
                "intervention": result.pico.intervention,
                "comparator": comparator,
                "comparator_source": comparator_source,  # NEW: Semantic coherence tracking
                "outcome": result.pico.outcome,
                "study_types": study_types_clean,
            }

        # Search Plan
        if result.search_plan:
            export_data["search_plan"] = {
                "pubmed_query": result.search_plan.pubmed_query,
                "ct_gov_query": result.search_plan.ct_gov_query,
            }

        # PRISMA Counts
        if result.prisma_counts:
            pc = result.prisma_counts
            export_data["prisma_counts"] = {
                "records_identified": pc.records_identified,
                "records_screened": pc.records_screened,
                "records_excluded": pc.records_excluded,
                "reports_assessed": pc.reports_assessed,
                "studies_included": pc.studies_included,
            }

        # Claims
        for claim in result.claims or []:
            # Extract therapeutic_context (new field for semantic coherence)
            therapeutic_context = None
            if hasattr(claim, "therapeutic_context") and claim.therapeutic_context:
                therapeutic_context = (
                    claim.therapeutic_context.value
                    if hasattr(claim.therapeutic_context, "value")
                    else str(claim.therapeutic_context)
                )

            export_data["claims"].append(
                {
                    "claim_id": claim.claim_id,
                    "claim_text": claim.claim_text,
                    "certainty": claim.certainty.value
                    if hasattr(claim.certainty, "value")
                    else str(claim.certainty),
                    "therapeutic_context": therapeutic_context,  # NEW: Semantic coherence tracking
                    "supporting_snippet_ids": list(claim.supporting_snippet_ids),
                    "grade_rationale": dict(claim.grade_rationale)
                    if hasattr(claim, "grade_rationale") and claim.grade_rationale
                    else None,
                }
            )

        # Snippets
        for snippet in result.snippets or []:
            sr = snippet.source_ref
            export_data["snippets"].append(
                {
                    "snippet_id": snippet.snippet_id,
                    "record_id": sr.record_id,
                    "pmid": sr.pmid,
                    "doi": sr.doi,
                    "text": snippet.text,
                    "section": snippet.section.value
                    if hasattr(snippet.section, "value")
                    else str(snippet.section),
                }
            )

        # Study Cards
        # CRITICAL: Filter out records excluded by DoD3 enforcement
        # Refs: DoD3_ENFORCEMENT_VALIDATION.txt, CDR_Integral_Audit_2026-01-20.md
        excluded_record_ids = set()
        if export_data.get("dod3_excluded_records"):
            excluded_record_ids = {rec["record_id"] for rec in export_data["dod3_excluded_records"]}

        records_by_id = {rec.record_id: rec for rec in (result.retrieved_records or [])}
        for card in result.study_cards or []:
            # HARD GATE: Do not include records that failed DoD3 validation
            if card.record_id in excluded_record_ids:
                continue  # Skip excluded records

            record = records_by_id.get(card.record_id)
            export_data["study_cards"].append(
                {
                    "study_id": card.record_id,
                    "title": record.title if record else "Unknown",
                    "study_type": card.study_type.value
                    if hasattr(card.study_type, "value")
                    else str(card.study_type),
                    "sample_size": card.sample_size,
                }
            )

        # Hypotheses
        # CRITICAL: Filter out suppressed hypotheses (from DoD3 enforcement)
        # Refs: DoD3_ENFORCEMENT_VALIDATION.txt
        suppressed_hyp_ids = set()
        if export_data.get("dod3_suppressed_hypotheses"):
            suppressed_hyp_ids = {
                h["hypothesis_id"] for h in export_data["dod3_suppressed_hypotheses"]
            }

        for hyp in result.composed_hypotheses or []:
            # Handle both Pydantic objects and dicts
            if hasattr(hyp, "confidence_score"):
                confidence = hyp.confidence_score
                hyp_id = hyp.hypothesis_id
                hyp_text = hyp.hypothesis_text
                strength = (
                    hyp.strength.value if hasattr(hyp.strength, "value") else str(hyp.strength)
                )
                source_ids = hyp.source_claim_ids
                reasoning = hyp.reasoning_trace
            else:
                # Dict fallback
                confidence = hyp.get("confidence_score", hyp.get("confidence", 0.5))
                hyp_id = hyp.get("hypothesis_id", "")
                hyp_text = hyp.get("hypothesis_text", "")
                strength = hyp.get("strength", "weak")
                source_ids = hyp.get("source_claim_ids", [])
                reasoning = hyp.get("reasoning_trace")

            # HARD GATE: Do not include hypotheses that were suppressed by DoD3
            if hyp_id in suppressed_hyp_ids:
                continue  # Skip suppressed hypotheses

            export_data["hypotheses"].append(
                {
                    "hypothesis_id": hyp_id,
                    "hypothesis_text": hyp_text,
                    "confidence": confidence,
                    "strength": strength,
                    "source_claim_ids": source_ids,
                    "reasoning": reasoning,
                }
            )

        content = json.dumps(export_data, indent=2, default=str)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="cdr-report-{run_id}.json"'},
        )

    elif format == "markdown":
        # Build markdown report
        lines = [
            f"# CDR Report: {run_id}",
            "",
            f"**Status**: {run.get('status')}",
            f"**Created**: {run.get('created_at')}",
            "",
        ]

        # PICO
        if result.pico:
            comparator_md = (
                result.pico.comparator or "Not specified (single-arm evidence synthesis)"
            )
            lines.extend(
                [
                    "## PICO",
                    "",
                    f"- **Population**: {result.pico.population}",
                    f"- **Intervention**: {result.pico.intervention}",
                    f"- **Comparator**: {comparator_md}",
                    f"- **Outcome**: {result.pico.outcome}",
                    "",
                ]
            )

        # Claims
        if result.claims:
            lines.extend(["## Evidence Claims", ""])
            for i, claim in enumerate(result.claims, 1):
                certainty = (
                    claim.certainty.value
                    if hasattr(claim.certainty, "value")
                    else str(claim.certainty)
                )
                lines.extend(
                    [
                        f"### Claim {i}",
                        "",
                        f"> {claim.claim_text}",
                        "",
                        f"**Certainty**: {certainty}",
                        f"**Supporting Evidence**: {len(claim.supporting_snippet_ids)} snippets",
                        "",
                    ]
                )

        # Summary stats
        lines.extend(
            [
                "## Summary Statistics",
                "",
                f"- **Studies Included**: {len(result.study_cards or [])}",
                f"- **Evidence Snippets**: {len(result.snippets or [])}",
                f"- **Claims Generated**: {len(result.claims or [])}",
                f"- **Composed Hypotheses**: {len(result.composed_hypotheses or [])}",
                "",
            ]
        )

        # Answer
        if result.answer:
            lines.extend(
                [
                    "## Answer",
                    "",
                    result.answer,
                    "",
                ]
            )

        content = "\n".join(lines)
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="cdr-report-{run_id}.md"'},
        )

    elif format == "pdf":
        # Build comprehensive PDF from HTML representation
        from weasyprint import HTML, CSS
        import html as html_escape_module

        def escape_html(text: str) -> str:
            """Escape HTML special characters."""
            if text is None:
                return ""
            return html_escape_module.escape(str(text))

        # Build snippet lookup for claim references
        snippet_map = {}
        if result.snippets:
            for snip in result.snippets:
                snippet_map[snip.snippet_id] = snip

        # Professional CSS - no emojis, SOTA presentation
        css_styles = """
            @page {
                size: A4;
                margin: 2cm 1.5cm;
                @top-center { content: "CDR Clinical Evidence Report"; font-size: 9pt; color: #718096; }
                @bottom-center { content: "Page " counter(page) " of " counter(pages); font-size: 9pt; color: #718096; }
            }
            body {
                font-family: 'Helvetica Neue', 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
                line-height: 1.5;
                color: #1a202c;
            }
            h1 {
                font-size: 22pt;
                color: #1a365d;
                border-bottom: 3px solid #2b6cb0;
                padding-bottom: 10px;
                margin-bottom: 8px;
            }
            h2 {
                font-size: 14pt;
                color: #2c5282;
                margin-top: 24px;
                margin-bottom: 12px;
                border-bottom: 1px solid #cbd5e0;
                padding-bottom: 6px;
                page-break-after: avoid;
            }
            h3 {
                font-size: 11pt;
                color: #2d3748;
                margin-top: 16px;
                margin-bottom: 8px;
            }
            h4 {
                font-size: 10pt;
                color: #4a5568;
                margin: 10px 0 6px 0;
            }
            .meta {
                color: #718096;
                font-size: 9pt;
                margin-bottom: 16px;
            }
            .section-box {
                background: #f7fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 14px;
                margin: 12px 0;
            }
            .pico-box {
                background: #ebf8ff;
                border-left: 4px solid #3182ce;
                padding: 12px 14px;
                margin: 10px 0;
            }
            .pico-item { margin: 4px 0; }
            .pico-label { font-weight: 600; color: #2c5282; display: inline-block; width: 90px; }
            .search-box {
                background: #faf5ff;
                border-left: 4px solid #805ad5;
                padding: 12px 14px;
                margin: 10px 0;
            }
            .query-label { font-weight: 600; color: #553c9a; }
            .query-text { font-family: 'Consolas', 'Monaco', monospace; font-size: 9pt; background: #f0e7ff; padding: 2px 6px; border-radius: 3px; }
            .stats-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 12px 0;
            }
            .stat-box {
                background: #edf2f7;
                padding: 10px 16px;
                border-radius: 6px;
                text-align: center;
                flex: 1;
                min-width: 100px;
            }
            .stat-value { font-size: 20pt; font-weight: 700; color: #2b6cb0; }
            .stat-label { color: #4a5568; font-size: 8pt; text-transform: uppercase; letter-spacing: 0.5px; }
            .claim-box {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-left: 4px solid #48bb78;
                padding: 14px;
                margin: 14px 0;
                page-break-inside: avoid;
            }
            .claim-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
            .claim-id { font-weight: 600; color: #2d3748; }
            .claim-text { font-size: 10.5pt; color: #1a202c; margin: 8px 0; line-height: 1.5; }
            .certainty {
                display: inline-block;
                padding: 3px 10px;
                border-radius: 10px;
                font-size: 8pt;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .certainty-high { background: #c6f6d5; color: #22543d; }
            .certainty-moderate { background: #fefcbf; color: #744210; }
            .certainty-low { background: #fed7aa; color: #9c4221; }
            .certainty-very_low { background: #fed7d7; color: #742a2a; }
            .evidence-refs { margin-top: 10px; padding-top: 8px; border-top: 1px dashed #e2e8f0; }
            .evidence-ref {
                font-size: 8.5pt;
                color: #4a5568;
                margin: 4px 0;
                padding-left: 10px;
                border-left: 2px solid #cbd5e0;
            }
            .snippet-box {
                background: #fffaf0;
                border: 1px solid #fbd38d;
                padding: 12px;
                margin: 10px 0;
                page-break-inside: avoid;
            }
            .snippet-header { display: flex; justify-content: space-between; margin-bottom: 6px; }
            .snippet-source { font-weight: 600; color: #744210; font-size: 9pt; }
            .snippet-id { color: #a0aec0; font-size: 8pt; font-family: monospace; }
            .snippet-text { font-size: 9pt; color: #4a5568; line-height: 1.4; }
            .snippet-meta { margin-top: 6px; font-size: 8pt; color: #718096; }
            .snippet-meta a { color: #3182ce; text-decoration: none; }
            .study-table {
                width: 100%;
                border-collapse: collapse;
                margin: 12px 0;
                font-size: 9pt;
            }
            .study-table th {
                background: #2c5282;
                color: white;
                padding: 8px 10px;
                text-align: left;
                font-weight: 600;
            }
            .study-table td {
                padding: 8px 10px;
                border-bottom: 1px solid #e2e8f0;
                vertical-align: top;
            }
            .study-table tr:nth-child(even) { background: #f7fafc; }
            .study-table tr:hover { background: #edf2f7; }
            .hypothesis-box {
                background: #f0fff4;
                border: 1px solid #9ae6b4;
                padding: 12px;
                margin: 10px 0;
            }
            .hypothesis-text { font-size: 10pt; color: #22543d; }
            .confidence-bar {
                background: #e2e8f0;
                height: 8px;
                border-radius: 4px;
                margin-top: 6px;
                overflow: hidden;
            }
            .confidence-fill { background: #48bb78; height: 100%; }
            .answer-box {
                background: linear-gradient(135deg, #ebf8ff 0%, #f0fff4 100%);
                border: 2px solid #38a169;
                padding: 18px;
                margin: 16px 0;
                border-radius: 8px;
            }
            .answer-title { font-weight: 600; color: #22543d; margin-bottom: 10px; font-size: 11pt; }
            .answer-text { font-size: 10.5pt; line-height: 1.6; color: #1a202c; }
            .footer {
                margin-top: 30px;
                padding-top: 12px;
                border-top: 1px solid #e2e8f0;
                color: #a0aec0;
                font-size: 8pt;
                text-align: center;
            }
            .toc { margin: 16px 0; }
            .toc-item { margin: 4px 0; color: #3182ce; }
            .page-break { page-break-before: always; }
        """

        # Build HTML document
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<meta charset='utf-8'>",
            f"<style>{css_styles}</style>",
            "</head><body>",
        ]

        # Title and metadata
        created_date = run.get("created_at", "")[:10] if run.get("created_at") else "N/A"
        html_parts.extend(
            [
                "<h1>CDR Clinical Evidence Report</h1>",
                f"<div class='meta'>",
                f"<strong>Run ID:</strong> {run_id} &nbsp;|&nbsp; ",
                f"<strong>Status:</strong> {run.get('status', 'N/A')} &nbsp;|&nbsp; ",
                f"<strong>Date:</strong> {created_date}",
                "</div>",
            ]
        )

        # Table of Contents
        html_parts.extend(
            [
                "<div class='section-box'>",
                "<h3>Contents</h3>",
                "<div class='toc'>",
                "<div class='toc-item'>1. Research Question (PICO Framework)</div>",
                "<div class='toc-item'>2. Search Strategy</div>",
                "<div class='toc-item'>3. PRISMA Flow Summary</div>",
                "<div class='toc-item'>4. Evidence Claims</div>",
                "<div class='toc-item'>5. Included Studies</div>",
                "<div class='toc-item'>6. Evidence Snippets</div>",
                "<div class='toc-item'>7. Composed Hypotheses</div>",
                "<div class='toc-item'>8. Clinical Conclusion</div>",
                "</div></div>",
            ]
        )

        # 1. PICO Section
        html_parts.append("<h2>1. Research Question (PICO Framework)</h2>")
        if result.pico:
            comparator_val = result.pico.comparator
            if not comparator_val or comparator_val == "null" or comparator_val == "None":
                comparator_display = "Not specified (single-arm evidence synthesis)"
            else:
                comparator_display = escape_html(comparator_val)

            study_types_str = (
                ", ".join(
                    str(st).replace("StudyType.", "") for st in (result.pico.study_types or [])
                )
                or "Not specified"
            )

            html_parts.extend(
                [
                    "<div class='pico-box'>",
                    f"<div class='pico-item'><span class='pico-label'>Population:</span> {escape_html(result.pico.population)}</div>",
                    f"<div class='pico-item'><span class='pico-label'>Intervention:</span> {escape_html(result.pico.intervention)}</div>",
                    f"<div class='pico-item'><span class='pico-label'>Comparator:</span> {comparator_display}</div>",
                    f"<div class='pico-item'><span class='pico-label'>Outcome:</span> {escape_html(result.pico.outcome)}</div>",
                    f"<div class='pico-item'><span class='pico-label'>Study Types:</span> {escape_html(study_types_str)}</div>",
                    "</div>",
                ]
            )
        else:
            html_parts.append("<p>PICO data not available.</p>")

        # 2. Search Strategy
        html_parts.append("<h2>2. Search Strategy</h2>")
        # Fix: Use result.search_plan instead of run.get() to access the actual data
        if result.search_plan:
            sp = result.search_plan
            html_parts.append("<div class='search-box'>")
            pubmed_q = (
                sp.pubmed_query
                if hasattr(sp, "pubmed_query")
                else (sp.get("pubmed_query") if isinstance(sp, dict) else None)
            )
            ct_q = (
                sp.ct_gov_query
                if hasattr(sp, "ct_gov_query")
                else (sp.get("ct_gov_query") if isinstance(sp, dict) else None)
            )
            if pubmed_q:
                html_parts.append(
                    f"<div><span class='query-label'>PubMed Query:</span> "
                    f"<span class='query-text'>{escape_html(pubmed_q)}</span></div>"
                )
            if ct_q:
                html_parts.append(
                    f"<div style='margin-top: 6px;'><span class='query-label'>ClinicalTrials.gov:</span> "
                    f"<span class='query-text'>{escape_html(ct_q)}</span></div>"
                )
            html_parts.append("</div>")
        else:
            html_parts.append("<p>Search strategy not available.</p>")

        # 3. PRISMA Flow
        html_parts.append("<h2>3. PRISMA Flow Summary</h2>")
        if result.prisma_counts:
            pc = result.prisma_counts
            html_parts.extend(
                [
                    "<div class='stats-grid'>",
                    f"<div class='stat-box'><div class='stat-value'>{pc.records_identified}</div><div class='stat-label'>Identified</div></div>",
                    f"<div class='stat-box'><div class='stat-value'>{pc.records_screened}</div><div class='stat-label'>Screened</div></div>",
                    f"<div class='stat-box'><div class='stat-value'>{pc.records_excluded}</div><div class='stat-label'>Excluded</div></div>",
                    f"<div class='stat-box'><div class='stat-value'>{pc.reports_assessed}</div><div class='stat-label'>Assessed</div></div>",
                    f"<div class='stat-box'><div class='stat-value'>{pc.studies_included}</div><div class='stat-label'>Included</div></div>",
                    "</div>",
                ]
            )
        else:
            html_parts.append("<p>PRISMA counts not available.</p>")

        # 4. Evidence Claims (with supporting snippets)
        html_parts.append("<h2>4. Evidence Claims</h2>")
        if result.claims:
            for i, claim in enumerate(result.claims, 1):
                certainty = (
                    claim.certainty.value
                    if hasattr(claim.certainty, "value")
                    else str(claim.certainty)
                )
                certainty_lower = certainty.lower().replace(" ", "_")
                if "high" in certainty_lower and "very" not in certainty_lower:
                    certainty_class = "high"
                elif "moderate" in certainty_lower:
                    certainty_class = "moderate"
                elif "very" in certainty_lower and "low" in certainty_lower:
                    certainty_class = "very_low"
                else:
                    certainty_class = "low"

                html_parts.extend(
                    [
                        "<div class='claim-box'>",
                        "<div class='claim-header'>",
                        f"<span class='claim-id'>Claim {i} ({claim.claim_id})</span>",
                        f"<span class='certainty certainty-{certainty_class}'>{certainty.upper()}</span>",
                        "</div>",
                        f"<div class='claim-text'>{escape_html(claim.claim_text)}</div>",
                    ]
                )

                # Add supporting evidence references
                if claim.supporting_snippet_ids:
                    html_parts.append(
                        "<div class='evidence-refs'><strong>Supporting Evidence:</strong>"
                    )
                    for snip_id in claim.supporting_snippet_ids[:5]:  # Limit to first 5
                        snip = snippet_map.get(snip_id)
                        if snip:
                            # Access pmid through source_ref
                            pmid = (
                                snip.source_ref.pmid
                                if hasattr(snip, "source_ref") and snip.source_ref
                                else None
                            )
                            source_info = f"PMID: {pmid}" if pmid else snip.snippet_id
                            text_preview = (
                                escape_html((snip.text or "")[:150]) + "..."
                                if len(snip.text or "") > 150
                                else escape_html(snip.text or "")
                            )
                            html_parts.append(
                                f"<div class='evidence-ref'><strong>{source_info}</strong>: {text_preview}</div>"
                            )
                        else:
                            html_parts.append(f"<div class='evidence-ref'>{snip_id}</div>")
                    if len(claim.supporting_snippet_ids) > 5:
                        html_parts.append(
                            f"<div class='evidence-ref'>... and {len(claim.supporting_snippet_ids) - 5} more snippets</div>"
                        )
                    html_parts.append("</div>")

                html_parts.append("</div>")
        else:
            html_parts.append("<p>No claims generated.</p>")

        # 5. Included Studies
        html_parts.append("<div class='page-break'></div>")
        html_parts.append("<h2>5. Included Studies</h2>")
        if result.study_cards:
            html_parts.extend(
                [
                    "<table class='study-table'>",
                    "<thead><tr>",
                    "<th>Study ID</th>",
                    "<th>Design</th>",
                    "<th>Sample Size</th>",
                    "</tr></thead>",
                    "<tbody>",
                ]
            )
            for card in result.study_cards:
                # Use record_id instead of study_id, study_type instead of design
                study_id = escape_html(card.record_id or "N/A")
                design = (
                    card.study_type.value
                    if hasattr(card.study_type, "value")
                    else str(card.study_type)
                    if card.study_type
                    else "Not specified"
                )
                sample = str(card.sample_size) if card.sample_size else "N/A"
                html_parts.append(
                    f"<tr><td>{study_id}</td><td>{escape_html(design)}</td><td>{sample}</td></tr>"
                )
            html_parts.extend(["</tbody></table>"])
        else:
            html_parts.append("<p>No study cards available.</p>")

        # 6. Evidence Snippets (full detail)
        html_parts.append("<div class='page-break'></div>")
        html_parts.append("<h2>6. Evidence Snippets</h2>")
        html_parts.append(
            f"<p><em>Total: {len(result.snippets or [])} snippets extracted from included studies.</em></p>"
        )
        if result.snippets:
            for snip in result.snippets:
                # Access identifiers through source_ref
                src_ref = snip.source_ref if hasattr(snip, "source_ref") else None
                pmid = src_ref.pmid if src_ref else None
                doi = src_ref.doi if src_ref else None
                record_id = src_ref.record_id if src_ref else None
                section_val = (
                    snip.section.value
                    if hasattr(snip.section, "value")
                    else str(snip.section)
                    if snip.section
                    else "N/A"
                )

                pmid_display = f"PMID: {pmid}" if pmid else ""
                source_display = pmid_display or record_id or snip.snippet_id

                html_parts.extend(
                    [
                        "<div class='snippet-box'>",
                        "<div class='snippet-header'>",
                        f"<span class='snippet-source'>{escape_html(source_display)}</span>",
                        f"<span class='snippet-id'>{escape_html(snip.snippet_id)}</span>",
                        "</div>",
                        f"<div class='snippet-text'>{escape_html(snip.text)}</div>",
                        "<div class='snippet-meta'>",
                        f"<strong>Section:</strong> {escape_html(section_val)}",
                    ]
                )
                if doi:
                    html_parts.append(
                        f" | <strong>DOI:</strong> <a href='https://doi.org/{doi}'>{escape_html(doi)}</a>"
                    )
                html_parts.extend(["</div>", "</div>"])
        else:
            html_parts.append("<p>No snippets extracted.</p>")

        # 7. Composed Hypotheses
        html_parts.append("<div class='page-break'></div>")
        html_parts.append("<h2>7. Composed Hypotheses</h2>")
        hypotheses = result.composed_hypotheses or []
        if hypotheses:
            for i, hyp in enumerate(hypotheses, 1):
                # Fix: Use 'hypothesis_text' which is the correct field name in the model
                hyp_text = (
                    hyp.hypothesis_text
                    if hasattr(hyp, "hypothesis_text")
                    else str(hyp.get("hypothesis_text", "N/A") if isinstance(hyp, dict) else hyp)
                )
                # Fix: Use 'confidence_score' which is the actual field name in ComposedHypothesis schema
                confidence = (
                    hyp.confidence_score
                    if hasattr(hyp, "confidence_score")
                    else (
                        hyp.get("confidence_score", hyp.get("confidence", 0))
                        if isinstance(hyp, dict)
                        else 0
                    )
                )
                confidence_pct = int(float(confidence) * 100) if confidence else 0

                html_parts.extend(
                    [
                        "<div class='hypothesis-box'>",
                        f"<h4>Hypothesis {i}</h4>",
                        f"<div class='hypothesis-text'>{escape_html(hyp_text)}</div>",
                        f"<div style='margin-top: 8px; font-size: 9pt; color: #4a5568;'>Confidence: {confidence_pct}%</div>",
                        "<div class='confidence-bar'>",
                        f"<div class='confidence-fill' style='width: {confidence_pct}%;'></div>",
                        "</div>",
                        "</div>",
                    ]
                )
        else:
            html_parts.append("<p>No hypotheses composed.</p>")

        # 8. Clinical Conclusion
        html_parts.append("<h2>8. Clinical Conclusion</h2>")
        if result.answer:
            html_parts.extend(
                [
                    "<div class='answer-box'>",
                    "<div class='answer-title'>Evidence Synthesis</div>",
                    f"<div class='answer-text'>{escape_html(result.answer)}</div>",
                    "</div>",
                ]
            )
        else:
            html_parts.append("<p>No conclusion generated.</p>")

        # 9. Gate Report (DoD3) - CRITICAL for auditable unpublishable runs
        # Refs: DoD3 Contract, PRISMA 2020 Transparency
        # This section appears when status != publishable AND gates were run
        run_status = result.status.value if hasattr(result.status, "value") else str(result.status)
        # Extract from result.report if available (preferred), then fall back to direct attributes
        gate_report = None
        dod3_enforcement = None
        if hasattr(result, "report") and result.report:
            gate_report = result.report.get("gate_report")
            dod3_enforcement = result.report.get("dod3_enforcement")
        else:
            gate_report = getattr(result, "gate_report", None)
            dod3_enforcement = getattr(result, "dod3_enforcement", None)

        if run_status in ("unpublishable", "insufficient_evidence") or gate_report:
            html_parts.append("<div class='page-break'></div>")
            html_parts.append("<h2>9. DoD3 Gate Report</h2>")

            if gate_report:
                # Status box
                overall_status = gate_report.get("overall_status", "unknown").upper()
                status_color = "#22543d" if overall_status == "PUBLISHABLE" else "#742a2a"

                html_parts.extend(
                    [
                        f"<div style='background: #fef3c7; border: 2px solid #d69e2e; padding: 16px; margin: 16px 0; border-radius: 8px;'>",
                        f"<div style='display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 12px;'>",
                        f"<div><strong>Status:</strong> <span style='color: {status_color}; font-weight: bold;'>{overall_status}</span></div>",
                        f"<div><strong>Generated:</strong> {str(gate_report.get('generated_at', 'N/A'))[:19]}</div>",
                        f"</div>",
                    ]
                )

                # Summary counts
                summary = gate_report.get("summary", {})
                html_parts.extend(
                    [
                        f"<div style='display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0;'>",
                        f"<div style='padding: 8px 16px; background: #fff; border-radius: 6px;'>",
                        f"<strong>Total Checks:</strong> {summary.get('total_checks', 0)}</div>",
                        f"<div style='padding: 8px 16px; background: #c6f6d5; border-radius: 6px;'>",
                        f"<strong>Passed:</strong> {summary.get('passed', 0)}</div>",
                        f"<div style='padding: 8px 16px; background: #fefcbf; border-radius: 6px;'>",
                        f"<strong>Warned:</strong> {summary.get('warned', 0)}</div>",
                        f"<div style='padding: 8px 16px; background: #fed7d7; border-radius: 6px;'>",
                        f"<strong>Failed:</strong> {summary.get('failed', 0)}</div>",
                        f"</div>",
                    ]
                )

                # Gate Results Table
                gate_results = gate_report.get("gate_results", {})
                if gate_results:
                    html_parts.extend(
                        [
                            "<div style='margin-top: 16px;'><strong>Gate Results:</strong></div>",
                            "<table style='width: 100%; margin-top: 8px; border-collapse: collapse; font-size: 9pt;'>",
                            "<tr style='background: #edf2f7;'><th style='padding: 6px; text-align: left;'>Gate</th>",
                            "<th style='padding: 6px;'>Result</th><th style='padding: 6px;'>Details</th></tr>",
                        ]
                    )

                    for gate_name, gate_data in gate_results.items():
                        gate_result = gate_data.get("result", "unknown")
                        result_icon = (
                            "✓"
                            if gate_result == "pass"
                            else ("⚠" if gate_result == "warn" else "✗")
                        )
                        result_color = (
                            "#22543d"
                            if gate_result == "pass"
                            else ("#744210" if gate_result == "warn" else "#742a2a")
                        )

                        metadata = gate_data.get("metadata", {})
                        details = ", ".join(f"{k}: {v}" for k, v in list(metadata.items())[:3])[:80]

                        html_parts.extend(
                            [
                                f"<tr>",
                                f"<td style='padding: 6px; border-bottom: 1px solid #e2e8f0;'>{escape_html(gate_name.replace('_', ' ').title())}</td>",
                                f"<td style='padding: 6px; border-bottom: 1px solid #e2e8f0; text-align: center; color: {result_color};'>{result_icon} {gate_result.upper()}</td>",
                                f"<td style='padding: 6px; border-bottom: 1px solid #e2e8f0; font-size: 8pt; color: #718096;'>{escape_html(details)}</td>",
                                f"</tr>",
                            ]
                        )

                    html_parts.append("</table>")

                # Blocker Violations
                blockers = gate_report.get("blocker_violations", [])
                if blockers:
                    html_parts.extend(
                        [
                            f"<div style='margin-top: 16px;'><strong style='color: #742a2a;'>Blocker Violations ({len(blockers)}):</strong></div>",
                            "<div style='margin-top: 8px;'>",
                        ]
                    )

                    for i, v in enumerate(blockers[:10], 1):
                        gate_v = v.get("gate", v.get("gate_name", "unknown"))
                        mismatch = v.get("mismatch_type", "unknown")
                        if hasattr(mismatch, "value"):
                            mismatch = mismatch.value
                        record_id = v.get("record_id", "N/A")
                        pmid = v.get("pmid", "N/A")
                        message = str(v.get("message", ""))[:150]

                        html_parts.extend(
                            [
                                f"<div style='background: #fff5f5; border-left: 3px solid #c53030; padding: 8px; margin: 4px 0; font-size: 9pt;'>",
                                f"<strong>{i}. [{escape_html(str(gate_v))}]</strong> {escape_html(str(mismatch))}<br>",
                                f"<span style='color: #718096;'>Record: {escape_html(str(record_id))} | PMID: {escape_html(str(pmid))}</span><br>",
                                f"<span>{escape_html(message)}</span>",
                                f"</div>",
                            ]
                        )

                    if len(blockers) > 10:
                        html_parts.append(
                            f"<div style='font-size: 8pt; color: #a0aec0;'>... and {len(blockers) - 10} more violations</div>"
                        )

                    html_parts.append("</div>")

                # Enforcement Summary
                if dod3_enforcement:
                    exc_records = len(dod3_enforcement.get("excluded_records", []))
                    exc_snippets = len(dod3_enforcement.get("excluded_snippets", []))
                    deg_claims = len(dod3_enforcement.get("degraded_claims", []))
                    orphan_claims = len(dod3_enforcement.get("orphan_claims", []))
                    sup_hyp = len(dod3_enforcement.get("suppressed_hypotheses", []))

                    html_parts.extend(
                        [
                            f"<div style='margin-top: 16px; padding-top: 12px; border-top: 1px dashed #d69e2e;'>",
                            f"<strong>Enforcement Applied:</strong>",
                            f"<ul style='margin: 8px 0; padding-left: 20px; font-size: 9pt;'>",
                            f"<li>Records Excluded: {exc_records}</li>",
                            f"<li>Snippets Excluded: {exc_snippets}</li>",
                            f"<li>Claims Degraded: {deg_claims}</li>",
                            f"<li>Claims Orphaned: {orphan_claims}</li>",
                            f"<li>Hypotheses Suppressed: {sup_hyp}</li>",
                            f"</ul>",
                            f"</div>",
                        ]
                    )

                html_parts.append("</div>")  # Close main gate report box
            else:
                # CRITICAL: If unpublishable but no gate report, this is a DoD3 compliance violation
                # Generate a minimal gate report explaining the gap
                html_parts.extend(
                    [
                        f"<div style='background: #fed7d7; border: 2px solid #c53030; padding: 16px; margin: 16px 0; border-radius: 8px;'>",
                        f"<p><strong>⚠️ AUDIT GAP</strong></p>",
                        f"<p>Run status: <strong>{run_status}</strong></p>",
                        f"<p>Gate Report not available. This indicates a gap in the DoD3 audit trail.</p>",
                        f"<p>For unpublishable runs, a complete Gate Report is required per DoD3 Contract.</p>",
                        f"<p><strong>Status Reason:</strong> {result.report.get('status_reason', 'Unknown') if hasattr(result, 'report') and result.report else 'Status reason not captured'}</p>",
                        f"</div>",
                    ]
                )

        # Footer
        from datetime import datetime

        gen_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        html_parts.extend(
            [
                "<div class='footer'>",
                f"Generated by CDR (Clinical Deep Research) | {gen_time}<br>",
                "This report is for research purposes. Clinical decisions should be validated by qualified professionals.",
                "</div>",
                "</body></html>",
            ]
        )

        html_content = "\n".join(html_parts)

        # Generate PDF using WeasyPrint
        pdf_bytes = HTML(string=html_content).write_pdf()

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="cdr-report-{run_id}.pdf"'},
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported export format: {format}. Supported: json, markdown",
        )


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_run(run_id: str) -> None:
    """Cancel a running CDR job.

    Args:
        run_id: The run ID to cancel
    """
    run = _get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    terminal_statuses = (
        RunStatus.COMPLETED.value,
        RunStatus.FAILED.value,
        RunStatus.CANCELLED.value,
        "completed",
        "failed",
        "cancelled",
    )
    if run["status"] in terminal_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel run in status: {run['status']}",
        )

    # Try in-memory first (for backward compatibility)
    if run_id in _runs:
        _runs[run_id]["status"] = RunStatus.CANCELLED.value
        _runs[run_id]["errors"].append("Cancelled by user")
        return

    # Try RunStore if configured
    if _run_store is not None:
        _run_store.update_run_status(
            run_id=run_id,
            status=RunStatus.CANCELLED,
            error_message="Cancelled by user",
        )
        return

    # This shouldn't happen if _get_run returned something
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Run found but could not be cancelled",
    )


@router.post("/parse-question", response_model=PICOResponse)
async def parse_question(question: str) -> PICOResponse:
    """Parse a research question into PICO components.

    This is a utility endpoint for testing PICO extraction
    without starting a full run.

    Args:
        question: The research question to parse

    Returns:
        Parsed PICO components
    """
    from cdr.interface.question_parser import QuestionParser
    from cdr.llm.factory import create_provider

    try:
        llm = create_provider("openai")
        parser = QuestionParser(llm)
        pico = await parser.parse(question)

        return PICOResponse(
            population=pico.population,
            intervention=pico.intervention,
            comparator=pico.comparator,
            outcome=pico.outcome,
            study_types=[str(st) for st in pico.study_types] if pico.study_types else None,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse question: {e!s}",
        ) from e


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Health status and version info
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "cdr",
    }


@router.get("/metrics")
async def get_metrics() -> dict:
    """Get current metrics.

    Returns:
        Dictionary of metric values
    """
    return metrics.get_all()


# =============================================================================
# BACKGROUND EXECUTION
# =============================================================================


async def _execute_run(
    run_id: str,
    question: str,
    max_results: int,
    formats: list[str],
    model: str,
    dod_level: int = 1,
) -> None:
    """Execute a CDR run in the background.

    Args:
        run_id: Run identifier
        question: Research question
        max_results: Maximum records per source
        formats: Output formats
        model: LLM model to use
        dod_level: Definition of Done level (1-3)
    """
    run = _runs[run_id]
    run["status"] = RunStatus.RUNNING.value  # Correct: RUNNING not IN_PROGRESS
    run["updated_at"] = datetime.utcnow().isoformat() + "Z"

    try:
        from cdr.llm.factory import create_provider_with_fallback
        from cdr.orchestration.graph import CDRRunner

        # Create LLM provider with automatic fallback
        # Priority: HuggingFace > Groq > OpenAI > Anthropic
        # If HuggingFace credits are depleted (402), tries Groq, etc.
        # CRITICAL FIX: Pass the model from the API request to override provider defaults
        llm = create_provider_with_fallback(model=model)
        effective_model = model or llm.model  # Use requested model or fallback to provider default

        print(f"[API] Using LLM provider: {llm.__class__.__name__}, model: {effective_model}")

        # Create and run pipeline with DoD level
        runner = CDRRunner(llm, model=effective_model, dod_level=dod_level)

        # Update progress callback (would need graph hooks in real impl)
        run["progress"]["current_node"] = "parse_question"
        run["progress"]["percentage"] = 5

        # ALTO-C fix: Pass run_id to runner for traceability alignment
        # Refs: CDR_Integral_Audit_2026-01-20.md ALTO-C
        result = await runner.run(
            research_question=question,
            max_results=max_results,
            formats=formats,
            dod_level=dod_level,
            run_id=run_id,
        )

        # Store result
        run["result"] = result
        run["status"] = result.status.value
        run["updated_at"] = datetime.utcnow().isoformat() + "Z"
        # CDRState has 'report' dict, not 'report_path'
        # Extract file_path from report if available
        if result.report and isinstance(result.report, dict):
            run["report_path"] = result.report.get("file_path")
        else:
            run["report_path"] = None
        run["progress"]["percentage"] = 100
        run["progress"]["current_node"] = "completed"

        metrics.counter("cdr.api.runs_completed")
        print(f"[API] Run {run_id} completed with status: {result.status.value}")

    except Exception as e:
        import traceback

        error_detail = f"{e.__class__.__name__}: {str(e)}"
        print(f"[API] Run {run_id} FAILED: {error_detail}")
        print(f"[API] Traceback: {traceback.format_exc()}")
        run["status"] = RunStatus.FAILED.value
        run["errors"].append(error_detail)
        run["progress"]["current_node"] = "failed"
        run["updated_at"] = datetime.utcnow().isoformat() + "Z"

        metrics.counter("cdr.api.runs_failed")


# =============================================================================
# APP FACTORY
# =============================================================================


def create_app():
    """Create FastAPI application.

    Returns:
        Configured FastAPI app
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Clinical Deep Research (CDR) API",
        description="Evidence-based systematic review automation",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include router
    app.include_router(router)

    @app.on_event("startup")
    async def startup():
        """Initialize on startup."""
        metrics.counter("cdr.api.startup")

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup on shutdown."""
        pass

    return app


# Create app instance for uvicorn/gunicorn
app = create_app()


# For direct execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
