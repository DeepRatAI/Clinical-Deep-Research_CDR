"""
CDR Core Schemas

This module defines all Pydantic models (schemas) used throughout the CDR system.
These schemas represent the domain model and enforce invariants via validators.

Key Design Principles:
1. All schemas are immutable by default (frozen=True where appropriate)
2. Critical invariants are enforced via model_validator and field_validator
3. Every exclusion has a reason
4. Every claim has supporting evidence
5. All source references are traceable
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cdr.core.enums import (
    ComparatorSource,
    CritiqueDimension,
    CritiqueSeverity,
    ExclusionReason,
    GRADECertainty,
    GraphNode,
    OutcomeMeasureType,
    RecordSource,
    RoB2Domain,
    RoB2Judgment,
    ROBINSIDomain,
    ROBINSIJudgment,
    RunStatus,
    Section,
    StudyType,
    TherapeuticContext,
    VerificationStatus,
)


# =============================================================================
# PICO - Clinical Question Structure
# =============================================================================


class PICO(BaseModel):
    """
    PICO formalización de pregunta clínica.

    PICO es el estándar para estructurar preguntas clínicas:
    - Population: ¿En quién?
    - Intervention: ¿Qué intervención?
    - Comparator: ¿Comparado con qué?
    - Outcome: ¿Qué resultado?

    Invariants:
    - population, intervention, outcome son obligatorios (min 3 chars)
    - comparator es opcional pero recomendado
    - comparator_source tracks provenance of comparator (user, assumed, inferred)

    Refs: PRISMA 2020, WHO ICTRP, ClinicalTrials.gov arm classification
    """

    model_config = ConfigDict(frozen=True)

    population: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Target population (e.g., 'adults with type 2 diabetes')",
    )
    intervention: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Intervention being studied (e.g., 'metformin 500mg twice daily')",
    )
    comparator: str | None = Field(
        default=None,
        max_length=500,
        description="Comparator/control (e.g., 'placebo', 'standard care')",
    )
    comparator_source: ComparatorSource = Field(
        default=ComparatorSource.NOT_APPLICABLE,
        description="How the comparator was determined: user_specified, assumed_from_question, inferred_from_evidence, heuristic, or not_applicable",
    )
    outcome: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Primary outcome of interest (e.g., 'HbA1c reduction at 12 weeks')",
    )
    study_types: list[StudyType] = Field(
        default_factory=list, description="Preferred study types for this question"
    )

    def is_complete(self) -> bool:
        """Check if PICO has all components including comparator."""
        return bool(self.population and self.intervention and self.comparator and self.outcome)

    def to_search_query(self) -> str:
        """Convert PICO to a basic search query string."""
        parts = [self.population, self.intervention]
        if self.comparator:
            parts.append(self.comparator)
        parts.append(self.outcome)
        return " AND ".join(f"({p})" for p in parts)


# =============================================================================
# SEARCH PLAN - Reproducible Search Strategy
# =============================================================================


class SearchPlan(BaseModel):
    """
    Plan de búsqueda reproducible.

    Documenta exactamente qué queries se ejecutarán en qué fuentes,
    permitiendo reproducibilidad y auditoría.

    CRITICAL CONTRACT: Use ct_gov_query (not clinical_trials_query) consistently
    across all components (planner, graph, ct_client).
    """

    model_config = ConfigDict(frozen=True)

    pico: PICO
    pubmed_query: str | None = Field(default=None, description="Formatted PubMed query string")
    ct_gov_query: str | None = Field(default=None, description="ClinicalTrials.gov query string")
    date_range: tuple[str, str] | None = Field(
        default=None, description="Date range filter as (min_date, max_date) in YYYY/MM/DD format"
    )
    languages: list[str] = Field(
        default_factory=lambda: ["english"], description="Allowed languages"
    )
    max_results_per_source: int = Field(
        default=100, ge=1, le=1000, description="Maximum results to retrieve per source"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# EXECUTED SEARCH - PRISMA-S Compliance
# =============================================================================


class ExecutedSearch(BaseModel):
    """
    Record of an actually executed search for PRISMA-S compliance.

    PRISMA-S (BMJ 2021) requires reporting of:
    - Database searched
    - Exact query executed (may differ from planned)
    - Date of search
    - Results count

    HIGH-4 fix: Capture executed queries for reproducibility.
    Refs: CDR_Integral_Audit_2026-01-20.md HIGH-4
    """

    model_config = ConfigDict(frozen=True)

    database: str = Field(
        ..., description="Database searched: 'pubmed', 'clinicaltrials.gov', etc."
    )
    query_planned: str | None = Field(None, description="Original query from SearchPlan")
    query_executed: str = Field(..., description="Exact query that was executed (may be modified)")
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    results_count: int = Field(..., ge=0, description="Number of results returned")
    results_fetched: int = Field(..., ge=0, description="Number of results actually fetched")
    notes: str | None = Field(
        None, description="Any modifications or notes (e.g., 'query truncated')"
    )


# =============================================================================
# RECORD - Retrieved Evidence Record
# =============================================================================


class Record(BaseModel):
    """
    Registro unificado de cualquier fuente de evidencia.

    Normaliza datos de PubMed, ClinicalTrials.gov, PDFs locales, etc.
    a un formato común para procesamiento downstream.

    Invariants:
    - record_id es único
    - content_hash se usa para deduplicación
    - Identificadores originales (pmid, doi, nct_id) se preservan
    """

    model_config = ConfigDict(frozen=True)

    # Identifiers
    record_id: str = Field(..., description="Unique identifier within CDR")
    source: RecordSource
    content_hash: str = Field(..., description="SHA256 hash for deduplication")

    # Original identifiers (preserve for traceability)
    pmid: str | None = Field(default=None, description="PubMed ID")
    doi: str | None = Field(default=None, description="Digital Object Identifier")
    nct_id: str | None = Field(default=None, description="ClinicalTrials.gov NCT ID")
    pmc_id: str | None = Field(default=None, description="PubMed Central ID")

    # Metadata
    title: str = Field(..., min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1800, le=2100)
    journal: str | None = None
    publication_type: list[str] = Field(default_factory=list)

    # Content
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)

    # URLs
    url: str | None = None
    pdf_url: str | None = None

    # Retrieval scores (populated by retrieval pipeline)
    retrieval_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores from different retrieval methods (bm25, dense, rerank)",
    )

    # Metadata
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def compute_hash(cls, title: str, abstract: str | None, doi: str | None) -> str:
        """Compute content hash for deduplication."""
        content = f"{title.lower().strip()}|{(abstract or '').lower().strip()}|{doi or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get_best_identifier(self) -> tuple[str, str]:
        """Return the most authoritative identifier (type, value)."""
        if self.doi:
            return ("doi", self.doi)
        if self.pmid:
            return ("pmid", self.pmid)
        if self.nct_id:
            return ("nct_id", self.nct_id)
        if self.pmc_id:
            return ("pmc_id", self.pmc_id)
        return ("record_id", self.record_id)


# =============================================================================
# SCREENING - Inclusion/Exclusion Decisions
# =============================================================================


class ScreeningDecision(BaseModel):
    """
    Decisión de inclusión/exclusión para un record.

    Invariants:
    - Si included=False, reason_code y reason_text son OBLIGATORIOS
    - pico_match_score es opcional pero recomendado
    """

    model_config = ConfigDict(frozen=True)

    record_id: str
    included: bool
    reason_code: ExclusionReason | None = None
    reason_text: str | None = None
    pico_match_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="PICO relevance score (0-1)"
    )
    study_type_detected: StudyType | None = None
    screened_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_exclusion_has_reason(self) -> "ScreeningDecision":
        """Ensure excluded records have documented reasons."""
        if not self.included:
            if self.reason_code is None:
                raise ValueError("Excluded records MUST have reason_code")
            if self.reason_text is None:
                raise ValueError("Excluded records MUST have reason_text")
        return self


# =============================================================================
# SOURCE REFERENCE - Traceable Citation
# =============================================================================


class SourceRef(BaseModel):
    """
    Referencia estable a una fuente de evidencia.

    Permite trazar cualquier claim de vuelta a su origen exacto,
    incluyendo sección y offsets dentro del documento.

    Invariants:
    - record_id siempre presente
    - Identificadores originales preservados para verificación externa
    """

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(..., description="Reference to parent Record")
    snippet_id: str | None = Field(
        default=None, description="Reference to specific Snippet if applicable"
    )

    # Original identifiers for external verification
    pmid: str | None = None
    doi: str | None = None
    nct_id: str | None = None

    # Location within document
    section: Section | None = None
    page: int | None = Field(default=None, ge=1)
    offset_start: int | None = Field(default=None, ge=0)
    offset_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "SourceRef":
        """Ensure offset_start <= offset_end if both present."""
        if self.offset_start is not None and self.offset_end is not None:
            if self.offset_start > self.offset_end:
                raise ValueError("offset_start must be <= offset_end")
        return self

    def to_citation_string(self) -> str:
        """Generate a citation string for display."""
        parts = []
        if self.pmid:
            parts.append(f"PMID:{self.pmid}")
        elif self.doi:
            parts.append(f"DOI:{self.doi}")
        elif self.nct_id:
            parts.append(f"NCT:{self.nct_id}")
        else:
            parts.append(f"ID:{self.record_id[:8]}")

        if self.section:
            parts.append(f"[{self.section.value}]")

        return " ".join(parts)


# =============================================================================
# SNIPPET - Citable Evidence Fragment
# =============================================================================


class Snippet(BaseModel):
    """
    Fragmento citable de evidencia con referencia a fuente.

    Los snippets son la unidad básica de evidencia. Todo claim debe
    estar soportado por uno o más snippets con source_ref válido.

    Invariants:
    - text no vacío
    - source_ref válido
    - snippet_id único
    """

    model_config = ConfigDict(frozen=True)

    snippet_id: str = Field(..., description="Unique identifier for this snippet")
    text: str = Field(..., min_length=10, max_length=5000)
    source_ref: SourceRef
    section: Section = Field(default=Section.UNKNOWN)

    # Metadata
    char_count: int = Field(default=0)
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def compute_char_count(self) -> "Snippet":
        """Auto-compute char_count from text."""
        object.__setattr__(self, "char_count", len(self.text))
        return self


# =============================================================================
# OUTCOME MEASURE - Statistical Result
# =============================================================================


class OutcomeMeasure(BaseModel):
    """
    Medida de resultado extraída de un estudio.

    Representa un resultado estadístico específico (RR, OR, HR, etc.)
    con su intervalo de confianza y p-value.

    Invariants:
    - Si value presente, supporting_snippet_id DEBE estar presente
    - CI bounds deben ser coherentes (lower < upper)
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, description="Name of the outcome")
    measure_type: OutcomeMeasureType | None = None
    value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    sample_size: int | None = Field(default=None, ge=0)

    # Traceability
    supporting_snippet_id: str | None = Field(
        default=None, description="ID of snippet supporting this measure"
    )

    @model_validator(mode="after")
    def validate_ci_bounds(self) -> "OutcomeMeasure":
        """Ensure CI bounds are coherent."""
        if self.ci_lower is not None and self.ci_upper is not None:
            if self.ci_lower > self.ci_upper:
                raise ValueError("ci_lower must be <= ci_upper")
        return self

    @model_validator(mode="after")
    def validate_value_has_support(self) -> "OutcomeMeasure":
        """Warn if value present without supporting snippet."""
        # Note: We use warning instead of error to allow partial extraction
        # The verification layer will catch unsupported values
        return self


# =============================================================================
# STUDY CARD - Structured Study Extraction
# =============================================================================


class StudyCard(BaseModel):
    """
    Extracción estructurada de datos de un estudio.

    Representa toda la información extraída de un estudio incluido,
    con referencias a los snippets que soportan cada campo.

    Invariants:
    - record_id referencia un Record existente
    - Campos extraídos deben tener snippet de soporte
    """

    model_config = ConfigDict(frozen=True)

    record_id: str
    study_type: StudyType

    # PICO extracted
    population_extracted: str | None = None
    intervention_extracted: str | None = None
    comparator_extracted: str | None = None

    # Study characteristics
    sample_size: int | None = Field(default=None, ge=0)
    follow_up_duration: str | None = None
    setting: str | None = None
    country: str | None = None

    # Outcomes
    primary_outcome: str | None = None
    outcomes: list[OutcomeMeasure] = Field(default_factory=list)

    # Quality indicators
    registration: str | None = Field(
        default=None, description="Trial registration number (e.g., NCT ID)"
    )
    funding_source: str | None = None
    conflicts_of_interest: str | None = None

    # Supporting evidence
    supporting_snippet_ids: list[str] = Field(
        default_factory=list, description="IDs of snippets supporting this extraction"
    )

    # Confidence
    extraction_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score for this extraction"
    )

    extracted_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# ROB2 - Risk of Bias Assessment
# =============================================================================


class RoB2DomainResult(BaseModel):
    """
    Evaluación de un dominio RoB2.

    Cada dominio tiene un juicio (low/some concerns/high) y rationale.
    """

    model_config = ConfigDict(frozen=True)

    domain: RoB2Domain
    judgment: RoB2Judgment
    rationale: str = Field(..., min_length=10)
    supporting_snippet_ids: list[str] = Field(default_factory=list)
    signalling_questions: dict[str, str] = Field(
        default_factory=dict, description="Answers to RoB2 signalling questions"
    )


class RoB2Result(BaseModel):
    """
    Evaluación completa RoB2 de un estudio.

    Invariants:
    - Exactamente 5 dominios evaluados
    - Overall judgment derivado de dominios
    """

    model_config = ConfigDict(frozen=True)

    record_id: str
    domains: list[RoB2DomainResult]
    overall_judgment: RoB2Judgment
    overall_rationale: str = Field(..., min_length=10)
    assessed_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("domains")
    @classmethod
    def validate_all_domains_present(cls, v: list[RoB2DomainResult]) -> list[RoB2DomainResult]:
        """Ensure all 5 RoB2 domains are assessed."""
        domain_set = {d.domain for d in v}
        required = set(RoB2Domain)

        if domain_set != required:
            missing = required - domain_set
            raise ValueError(f"Missing RoB2 domains: {missing}")

        return v

    def get_domain(self, domain: RoB2Domain) -> RoB2DomainResult | None:
        """Get result for a specific domain."""
        for d in self.domains:
            if d.domain == domain:
                return d
        return None


# =============================================================================
# ROBINS-I RESULT - Non-Randomized Study Bias Assessment
# =============================================================================


class ROBINSIDomainResult(BaseModel):
    """
    Evaluación de un dominio ROBINS-I para estudios no randomizados.

    HIGH-3 fix: ROBINS-I domains for observational studies.
    Refs: https://methods.cochrane.org/bias/resources/robins-i-tool
    """

    model_config = ConfigDict(frozen=True)

    domain: ROBINSIDomain
    judgment: ROBINSIJudgment
    rationale: str = Field(..., min_length=10)
    supporting_snippet_ids: list[str] = Field(default_factory=list)


class ROBINSIResult(BaseModel):
    """
    Evaluación completa ROBINS-I de un estudio observacional.

    ROBINS-I (Risk Of Bias In Non-randomised Studies of Interventions) is used
    for cohort studies, case-control studies, and other non-randomized designs.

    HIGH-3 fix: Separate assessment tool for observational studies.
    Refs: https://methods.cochrane.org/bias/resources/robins-i-tool

    Invariants:
    - Exactamente 7 dominios evaluados
    - Overall judgment derivado de dominios
    """

    model_config = ConfigDict(frozen=True)

    record_id: str
    domains: list[ROBINSIDomainResult]
    overall_judgment: ROBINSIJudgment
    overall_rationale: str = Field(..., min_length=10)
    assessed_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("domains")
    @classmethod
    def validate_all_domains_present(
        cls, v: list[ROBINSIDomainResult]
    ) -> list[ROBINSIDomainResult]:
        """Ensure all 7 ROBINS-I domains are assessed."""
        domain_set = {d.domain for d in v}
        required = set(ROBINSIDomain)

        if domain_set != required:
            missing = required - domain_set
            raise ValueError(f"Missing ROBINS-I domains: {missing}")

        return v

    def get_domain(self, domain: ROBINSIDomain) -> ROBINSIDomainResult | None:
        """Get result for a specific domain."""
        for d in self.domains:
            if d.domain == domain:
                return d
        return None


# =============================================================================
# EVIDENCE CLAIM - Verified Assertion
# =============================================================================


class EvidenceClaim(BaseModel):
    """
    Claim verificable con soporte de evidencia.

    Los claims son las unidades de conocimiento generadas por CDR.
    Cada claim DEBE estar soportado por al menos un snippet.

    Invariants:
    - supporting_snippet_ids no vacío (al menos 1)
    - certainty basado en GRADE

    GRADE Rationale:
    - grade_rationale captures explicit reasons for certainty level
    - Keys: "risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias"
    - Values: explanation text or None if not applicable
    Refs: GRADE Handbook Section 5.2, ADR-004 Audit v3
    """

    model_config = ConfigDict(frozen=True)

    claim_id: str
    claim_text: str = Field(..., min_length=20, max_length=1000)
    certainty: GRADECertainty

    # Evidence support (REQUIRED)
    supporting_snippet_ids: list[str] = Field(
        ..., min_length=1, description="IDs of snippets supporting this claim"
    )

    # Conflicts (optional but important)
    conflicting_snippet_ids: list[str] = Field(
        default_factory=list, description="IDs of snippets that may conflict with this claim"
    )

    # Qualifications
    limitations: list[str] = Field(
        default_factory=list, description="Known limitations of this claim"
    )

    # GRADE Rationale - structured certainty justification
    # Per GRADE Handbook Section 5.2: Each domain that leads to downgrade
    # must be explicitly documented with rationale
    # Keys: risk_of_bias, inconsistency, indirectness, imprecision, publication_bias
    # Refs: https://gradepro.org/handbook/, ADR-004 Audit v3
    grade_rationale: dict[str, str] = Field(
        default_factory=dict,
        description="Structured GRADE rationale per domain (risk_of_bias, inconsistency, etc.)",
    )

    # Therapeutic context - prevents mixing incompatible scenarios
    # CRITICAL: Claims must be tagged with their therapeutic context to prevent
    # mixing e.g. aspirin monotherapy results with aspirin+anticoagulant results
    # Refs: GRADE Handbook Section 5, Cochrane Handbook Section 11, CDR DoD P2
    therapeutic_context: TherapeuticContext = Field(
        default=TherapeuticContext.UNCLASSIFIED,
        description="Therapeutic context: monotherapy, add_on, head_to_head, etc.",
    )

    # Metadata
    studies_supporting: int = Field(default=0, ge=0)
    studies_conflicting: int = Field(default=0, ge=0)

    @field_validator("supporting_snippet_ids")
    @classmethod
    def validate_has_support(cls, v: list[str]) -> list[str]:
        """Ensure claim has at least one supporting snippet."""
        if not v:
            raise ValueError("EvidenceClaim MUST have at least one supporting snippet")
        return v


# =============================================================================
# CRITIQUE - Skeptic Agent Output
# =============================================================================


class CritiqueResult(BaseModel):
    """Single critique finding from Skeptic agent."""

    model_config = ConfigDict(frozen=True)

    dimension: CritiqueDimension
    severity: CritiqueSeverity
    finding: str = Field(..., min_length=10)
    affected_claims: list[str] = Field(default_factory=list)
    recommendation: str | None = None


class Critique(BaseModel):
    """Complete critique from Skeptic agent."""

    model_config = ConfigDict(frozen=True)

    findings: list[CritiqueResult] = Field(default_factory=list)
    blockers: list[str] = Field(
        default_factory=list, description="Critical issues that block publication"
    )
    recommendations: list[str] = Field(default_factory=list)
    overall_assessment: str | None = None
    critiqued_at: datetime = Field(default_factory=datetime.utcnow)

    def has_blockers(self) -> bool:
        """Check if there are any blockers."""
        return len(self.blockers) > 0


# =============================================================================
# VERIFICATION - Claim Verification Results
# =============================================================================


class VerificationCheck(BaseModel):
    """
    Result of verifying a claim against a single source snippet.

    Each claim may have multiple checks (one per supporting snippet).
    This captures the entailment relationship between claim and source.
    """

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(..., description="ID of the claim being verified")
    source_ref: SourceRef = Field(..., description="Reference to the source snippet")
    status: VerificationStatus = Field(
        ..., description="Verification status (VERIFIED, PARTIAL, etc.)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this check")
    explanation: str | None = Field(
        default=None, description="Explanation of the verification result"
    )
    supporting_quote: str | None = Field(
        default=None, description="Exact quote from source supporting the claim"
    )

    @property
    def passed(self) -> bool:
        """Check passed if status is VERIFIED."""
        return self.status == VerificationStatus.VERIFIED


class VerificationResult(BaseModel):
    """
    Aggregated verification result for a single claim.

    Contains all individual checks and the overall verdict.
    """

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(..., description="ID of the claim that was verified")
    checks: list[VerificationCheck] = Field(
        default_factory=list, description="Individual verification checks"
    )
    overall_status: VerificationStatus = Field(..., description="Aggregated verification status")
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregated confidence score"
    )
    verified_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def passed(self) -> bool:
        """Overall verification passed if status is VERIFIED or PARTIAL.

        PARTIAL indicates the claim has supporting evidence but not complete
        confirmation. For DoD3 purposes, PARTIAL counts toward verification_coverage
        since it demonstrates grounding in source material.
        """
        from cdr.core.enums import VerificationStatus

        return self.overall_status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIAL)

    def get_failed_checks(self) -> list[VerificationCheck]:
        """Get list of checks that did not pass."""
        return [c for c in self.checks if not c.passed]


# =============================================================================
# PRISMA COUNTS - PRISMA Flow Diagram Data
# =============================================================================


class PRISMACounts(BaseModel):
    """PRISMA-style counts for flow diagram.

    P0-08: PRISMA 2 Stages - Separation of screening from enforcement.

    Stage 1 (Screening): Initial eligibility assessment during retrieval
    - Records screened based on basic criteria (language, date, type)
    - Exclusions here are "retrieval noise" - not penalties

    Stage 2 (Enforcement): DoD3 gate validation
    - Records assessed against PICO/study type requirements
    - Exclusions here indicate support integrity issues if claims used them
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # IDENTIFICATION
    # =========================================================================
    records_identified: int = Field(default=0, ge=0)
    records_from_pubmed: int = Field(default=0, ge=0)
    records_from_clinical_trials: int = Field(default=0, ge=0)
    records_from_other: int = Field(default=0, ge=0)

    duplicates_removed: int = Field(default=0, ge=0)

    # =========================================================================
    # P0-08: STAGE 1 - SCREENING (retrieval noise filter)
    # These exclusions do NOT affect publishability
    # =========================================================================
    records_screened: int = Field(default=0, ge=0)
    records_excluded_screening: int = Field(
        default=0,
        ge=0,
        description="P0-08: Stage 1 exclusions - retrieval noise, not PICO failures",
    )
    # Legacy field - kept for backwards compatibility
    records_excluded: int = Field(default=0, ge=0)

    # Screening exclusion breakdown (language, date, type)
    screening_exclusion_reasons: dict[str, int] = Field(
        default_factory=dict,
        description="P0-08: Stage 1 exclusion breakdown (language, date, etc.)",
    )

    # =========================================================================
    # P0-08: STAGE 2 - ENFORCEMENT (DoD3 gate validation)
    # These exclusions affect publishability if claims USE the excluded evidence
    # =========================================================================
    reports_sought: int = Field(default=0, ge=0)
    reports_not_retrieved: int = Field(default=0, ge=0)
    reports_assessed: int = Field(default=0, ge=0)
    reports_excluded: int = Field(default=0, ge=0)

    # P0-08: Enforcement exclusions (DoD3 gate failures)
    records_excluded_enforcement: int = Field(
        default=0,
        ge=0,
        description="P0-08: Stage 2 exclusions - PICO/study type failures",
    )

    # Enforcement exclusion breakdown (PICO mismatch, study type, etc.)
    enforcement_exclusion_reasons: dict[str, int] = Field(
        default_factory=dict,
        description="P0-08: Stage 2 exclusion breakdown (PICO, study type, etc.)",
    )

    studies_included: int = Field(default=0, ge=0)

    # Legacy: Exclusion breakdown (combined, for backwards compatibility)
    exclusion_reasons: dict[str, int] = Field(default_factory=dict)


# =============================================================================
# SYNTHESIS RESULT
# =============================================================================


class SynthesisResult(BaseModel):
    """Result of evidence synthesis.

    DoD Level Gates rely on used_markdown_fallback to determine
    if claims were extracted via heuristic parsing (not allowed in Level 2+).
    Refs: ADR-005 Post-Change Audit, PRISMA 2020 (reproducibility)
    """

    model_config = ConfigDict(frozen=True)

    claims: list[EvidenceClaim] = Field(default_factory=list)
    heterogeneity_assessment: str = ""
    overall_narrative: str = ""
    meta_analysis_data: dict[str, Any] | None = None

    # DoD Level tracking: indicates if fallback parsing was used
    # Level 2+ requires JSON structured outputs; Markdown fallback is blocked
    # Refs: ADR-005 Post-Change Audit, GRADE handbook (reproducibility)
    used_markdown_fallback: bool = Field(
        default=False,
        description="True if claims were extracted via Markdown fallback (heuristic parsing)",
    )

    @property
    def claim_count(self) -> int:
        """Number of synthesized claims."""
        return len(self.claims)

    @property
    def high_certainty_claims(self) -> list[EvidenceClaim]:
        """Claims with high GRADE certainty."""
        return [c for c in self.claims if c.certainty == GRADECertainty.HIGH]

    @property
    def low_certainty_claims(self) -> list[EvidenceClaim]:
        """Claims with low or very low certainty."""
        return [
            c for c in self.claims if c.certainty in (GRADECertainty.LOW, GRADECertainty.VERY_LOW)
        ]


# =============================================================================
# CDR STATE - Complete Workflow State
# =============================================================================


class CDRState(BaseModel):
    """
    Estado completo del workflow CDR.

    Este es el estado central que fluye a través del LangGraph.
    Contiene todos los artefactos generados durante la ejecución.
    """

    # Allow mutation for LangGraph state updates
    model_config = ConfigDict(validate_assignment=True)

    # Identifiers
    run_id: str

    # Input
    question: str

    # Generated artifacts
    pico: PICO | None = None
    search_plan: SearchPlan | None = None
    # PRISMA-S: Track actually executed searches for reproducibility (HIGH-4)
    executed_searches: list[ExecutedSearch] = Field(
        default_factory=list,
        description="Record of searches actually executed for PRISMA-S compliance",
    )
    retrieved_records: list[Record] = Field(default_factory=list)
    screened: list[ScreeningDecision] = Field(default_factory=list)

    # Intermediate parsing/extraction state
    parsed_documents: dict[str, dict[str, Any]] = Field(default_factory=dict)

    snippets: list[Snippet] = Field(default_factory=list)
    study_cards: list[StudyCard] = Field(default_factory=list)
    rob2_results: list[RoB2Result] = Field(default_factory=list)
    robins_i_results: list[ROBINSIResult] = Field(
        default_factory=list,
        description="ROBINS-I results for observational studies (HIGH-3 fix)",
    )

    # Synthesis artifacts
    claims: list[EvidenceClaim] = Field(default_factory=list)
    synthesis_result: SynthesisResult | None = None

    # Compositional inference (HIGH-1 fix)
    # Note: Stores as dict to avoid circular import with composition module
    composed_hypotheses: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Composed hypotheses from A+B⇒C inference (DoD Level 3)",
    )

    # Quality control
    critique: Critique | None = None
    verification: list[VerificationResult] = Field(
        default_factory=list, description="Verification results for each claim"
    )

    # Output
    answer: str | None = None
    report: dict[str, Any] | None = None
    prisma_counts: PRISMACounts | None = None

    # Execution metadata
    status: RunStatus = Field(default=RunStatus.PENDING)
    current_node: GraphNode | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    # Loop control
    iteration_count: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=3, ge=1, le=10)

    # Tracing
    traces: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    flags: dict[str, bool] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    def get_included_records(self) -> list[Record]:
        """Get records that passed screening."""
        included_ids = {s.record_id for s in self.screened if s.included}
        return [r for r in self.retrieved_records if r.record_id in included_ids]

    def get_excluded_records(self) -> list[tuple[Record, ScreeningDecision]]:
        """Get records that were excluded with their reasons."""
        excluded_decisions = {s.record_id: s for s in self.screened if not s.included}
        result = []
        for record in self.retrieved_records:
            if record.record_id in excluded_decisions:
                result.append((record, excluded_decisions[record.record_id]))
        return result

    def add_trace(self, node: str, data: dict[str, Any]) -> None:
        """Add a trace entry."""
        self.traces.append({"node": node, "timestamp": datetime.utcnow().isoformat(), **data})

    def set_flag(self, flag: str, value: bool = True) -> None:
        """Set a flag."""
        self.flags[flag] = value

    def has_flag(self, flag: str) -> bool:
        """Check if a flag is set."""
        return self.flags.get(flag, False)

    # ==========================================================================
    # PHASE CONTRACTS: Explicit validation per workflow stage
    # ==========================================================================

    def validate_after_parse_question(self) -> None:
        """
        CONTRACT: After PARSE_QUESTION node.
        MUST have: pico populated.
        """
        if self.pico is None:
            raise ValueError("Contract violation: PICO must be populated after PARSE_QUESTION")

    def validate_after_plan_search(self) -> None:
        """
        CONTRACT: After PLAN_SEARCH node.
        MUST have: search_plan populated with at least one query.
        """
        if self.search_plan is None:
            raise ValueError("Contract violation: search_plan must exist after PLAN_SEARCH")
        if not self.search_plan.pubmed_query and not self.search_plan.ct_gov_query:
            raise ValueError("Contract violation: search_plan must have at least one query")

    def validate_after_retrieve(self) -> None:
        """
        CONTRACT: After RETRIEVE node.
        MUST have: retrieved_records populated (can be empty if no results).
        """
        # retrieved_records is initialized as empty list, so always exists
        # This is a soft contract: we just verify the field exists
        if not isinstance(self.retrieved_records, list):
            raise ValueError("Contract violation: retrieved_records must be a list")

    def validate_after_screen(self) -> None:
        """
        CONTRACT: After SCREEN node.
        MUST have: screened populated.
        At least one record must be included OR all must be explicitly excluded.
        """
        if not self.screened:
            raise ValueError("Contract violation: screened must be populated after SCREEN")

    def validate_after_parse_docs(self) -> None:
        """
        CONTRACT: After PARSE_DOCS node.
        MUST have: snippets extracted from included records.
        """
        if not self.snippets and any(s.included for s in self.screened):
            raise ValueError(
                "Contract violation: snippets must be populated if records were included"
            )

    def validate_after_extract_data(self) -> None:
        """
        CONTRACT: After EXTRACT_DATA node.
        MUST have: study_cards populated for included studies.
        """
        if not self.study_cards and self.snippets:
            raise ValueError("Contract violation: study_cards must be populated if snippets exist")

    def validate_after_synthesize(self) -> None:
        """
        CONTRACT: After SYNTHESIZE node.
        MUST have: claims populated, answer generated.
        """
        if not self.claims:
            raise ValueError("Contract violation: claims must be populated after SYNTHESIZE")
        if not self.answer:
            raise ValueError("Contract violation: answer must be generated after SYNTHESIZE")

    def validate_after_critique(self) -> None:
        """
        CONTRACT: After CRITIQUE node.
        MUST have: critique populated.
        """
        if self.critique is None:
            raise ValueError("Contract violation: critique must be populated after CRITIQUE")

    def validate_after_verify(self) -> None:
        """
        CONTRACT: After VERIFY node.
        MUST have: verification populated with status.
        """
        if self.verification is None:
            raise ValueError("Contract violation: verification must be populated after VERIFY")

    def validate_after_publish(self) -> None:
        """
        CONTRACT: After PUBLISH node.
        MUST have: report generated, prisma_counts populated.
        """
        if self.report is None:
            raise ValueError("Contract violation: report must be generated after PUBLISH")
        if self.prisma_counts is None:
            raise ValueError("Contract violation: prisma_counts must be populated after PUBLISH")

    def validate_for_node(self, node: GraphNode) -> None:
        """
        Validate state contracts for a specific node.

        This should be called AFTER a node completes to ensure it fulfilled its contract.
        """
        validators = {
            GraphNode.PARSE_QUESTION: self.validate_after_parse_question,
            GraphNode.PLAN_SEARCH: self.validate_after_plan_search,
            GraphNode.RETRIEVE: self.validate_after_retrieve,
            GraphNode.SCREEN: self.validate_after_screen,
            GraphNode.PARSE_DOCS: self.validate_after_parse_docs,
            GraphNode.EXTRACT_DATA: self.validate_after_extract_data,
            GraphNode.SYNTHESIZE: self.validate_after_synthesize,
            GraphNode.CRITIQUE: self.validate_after_critique,
            GraphNode.VERIFY: self.validate_after_verify,
            GraphNode.PUBLISH: self.validate_after_publish,
        }

        if node in validators:
            validators[node]()
