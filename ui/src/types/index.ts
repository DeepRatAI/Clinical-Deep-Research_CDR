/**
 * CDR UI Types
 * 
 * TypeScript types mirroring Python schemas from cdr.core.schemas
 * Ref: src/cdr/core/schemas.py
 */

// =============================================================================
// ENUMS
// =============================================================================

export type RunStatus = 
  | 'PENDING'
  | 'RUNNING'
  | 'COMPLETED'
  | 'FAILED'
  | 'INSUFFICIENT_EVIDENCE'

export type CertaintyLevel =
  | 'HIGH'
  | 'MODERATE'
  | 'LOW'
  | 'VERY_LOW'

export type VerificationStatus =
  | 'VERIFIED'
  | 'PARTIAL'
  | 'REFUTED'
  | 'UNVERIFIABLE'

export type StudyType =
  | 'RCT'
  | 'QUASI_EXPERIMENTAL'
  | 'COHORT'
  | 'CASE_CONTROL'
  | 'CROSS_SECTIONAL'
  | 'CASE_SERIES'
  | 'CASE_REPORT'
  | 'SYSTEMATIC_REVIEW'
  | 'META_ANALYSIS'
  | 'OTHER'

export type RoB2Judgment =
  | 'LOW'
  | 'SOME_CONCERNS'
  | 'HIGH'

export type RoB2Domain =
  | 'RANDOMIZATION'
  | 'DEVIATIONS'
  | 'MISSING_DATA'
  | 'MEASUREMENT'
  | 'SELECTION'

export type Section =
  | 'ABSTRACT'
  | 'INTRODUCTION'
  | 'METHODS'
  | 'RESULTS'
  | 'DISCUSSION'
  | 'CONCLUSION'
  | 'FULL_TEXT'
  | 'OTHER'

export type RecordSource =
  | 'PUBMED'
  | 'CT_GOV'
  | 'SEMANTIC_SCHOLAR'
  | 'MANUAL'
  | 'OTHER'

// =============================================================================
// CORE SCHEMAS
// =============================================================================

export interface PICO {
  population: string
  intervention: string
  comparator: string | null
  outcome: string
  study_types: StudyType[] | null
}

export interface SearchPlan {
  pico: PICO
  pubmed_query: string
  ct_gov_query: string
  date_range: [string, string] | null
  languages: string[]
  max_results_per_source: number
  created_at: string
}

export interface Record {
  id: string
  pmid: string | null
  doi: string | null
  nct_id: string | null
  title: string
  abstract: string | null
  authors: string[]
  publication_date: string | null
  journal: string | null
  source: RecordSource
  url: string | null
  mesh_terms: string[]
  study_type: StudyType | null
  full_text_available: boolean
}

export interface SourceRef {
  record_id: string
  pmid: string | null
  doi: string | null
  nct_id: string | null
  title: string
  authors: string[]
  publication_year: number | null
  journal: string | null
  url: string | null
}

export interface Snippet {
  snippet_id: string
  source_ref: SourceRef
  section: Section
  text: string
  offset_start: number | null
  offset_end: number | null
  relevance_score: number | null
}

export interface OutcomeMeasure {
  name: string
  value: number | null
  unit: string | null
  effect_size: number | null
  confidence_interval: [number, number] | null
  p_value: number | null
  is_significant: boolean | null
  direction: 'positive' | 'negative' | 'neutral' | null
}

export interface RoB2DomainAssessment {
  domain: RoB2Domain
  judgment: RoB2Judgment
  rationale: string
}

export interface RoB2Result {
  study_id: string
  study_type: StudyType
  domains: RoB2DomainAssessment[]
  overall_judgment: RoB2Judgment
  overall_rationale: string
}

export interface StudyCard {
  study_id: string
  record_id: string
  title: string
  study_type: StudyType
  population: string | null
  intervention: string | null
  comparator: string | null
  outcomes: OutcomeMeasure[]
  supporting_snippet_ids: string[]
  sample_size: number | null
  follow_up_duration: string | null
}

export interface GRATERationale {
  risk_of_bias: string | null
  inconsistency: string | null
  indirectness: string | null
  imprecision: string | null
  publication_bias: string | null
}

export interface EvidenceClaim {
  claim_id: string
  claim_text: string
  certainty: CertaintyLevel
  supporting_snippet_ids: string[]
  grade_rationale: GRATERationale | null
  verification_status: VerificationStatus | null
  verification_score: number | null
}

export interface PRISMACounts {
  records_identified: number
  records_screened: number
  records_excluded_screening: number
  reports_assessed: number
  reports_not_retrieved: number
  studies_included: number
  exclusion_reasons: { [reason: string]: number }
}

export interface ComposedHypothesis {
  hypothesis_id: string
  claim_a_id: string
  claim_b_id: string
  hypothesis_text: string
  mechanism: string
  rival_hypotheses: string[]
  threats_to_validity: string[]
  mcid: string | null
  test_design: string | null
  confidence: number
}

// =============================================================================
// RUN & REPORT
// =============================================================================

export interface RunSummary {
  run_id: string
  status: RunStatus
  status_reason: string | null
  dod_level: number
  claims_count: number
  verification_coverage: number
  created_at: string
  updated_at: string
}

export interface RunProgress {
  current_node: string
  nodes_completed: string[]
  total_nodes: number
  started_at: string
  elapsed_seconds: number
}

export interface RunDetail {
  run_id: string
  status: RunStatus
  status_reason: string | null
  dod_level: number
  progress: RunProgress | null
  pico: PICO | null
  search_plan: SearchPlan | null
  prisma_counts: PRISMACounts | null
  records: Record[]
  snippets: Snippet[]
  study_cards: StudyCard[]
  claims: EvidenceClaim[]
  rob2_results: RoB2Result[]
  composed_hypotheses: ComposedHypothesis[]
  errors: string[]
  created_at: string
  updated_at: string
}

export interface ReportMetadata {
  run_id: string
  generated_at: string
  formats_available: string[]
  download_urls: { [format: string]: string }
}

// =============================================================================
// EVALUATION TYPES (GRADE, Dimensions)
// =============================================================================

export interface EvaluationDimension {
  name: string
  score: number
  grade: string
  rationale?: string
}

export interface Evaluation {
  overall_score: number
  overall_grade: string
  dimensions: EvaluationDimension[]
  strengths: string[]
  weaknesses: string[]
  recommendations: string[]
}

// =============================================================================
// API REQUEST/RESPONSE
// =============================================================================

export interface CreateRunRequest {
  research_question: string
  max_results?: number
  output_formats?: string[]
  model?: string
  dod_level?: number
}

export interface CreateRunResponse {
  run_id: string
  status: string
  message: string
}

export interface ErrorResponse {
  detail: string
  status_code: number
}
