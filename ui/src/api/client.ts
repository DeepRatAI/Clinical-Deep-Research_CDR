/**
 * CDR API Client
 * 
 * Centralized API client for CDR backend communication
 */

import type {
  CreateRunRequest,
  CreateRunResponse,
  RunSummary,
  EvidenceClaim,
  Snippet,
  StudyCard,
  PRISMACounts,
  SearchPlan,
  ComposedHypothesis,
  ReportMetadata,
  PICO,
  Evaluation,
} from '../types'

import {
  transformSnippets,
  transformClaims,
  transformStudyCards,
  transformRunSummaries,
  transformHypotheses,
  type RawSnippetResponse,
  type RawClaimResponse,
  type RawStudyCardResponse,
  type RawRunSummary,
} from './transformers'

const API_BASE = '/api/v1'

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}))
    throw new ApiError(
      errorBody.detail || response.statusText,
      response.status,
      errorBody.detail
    )
  }
  return response.json()
}

// =============================================================================
// RUN DETAIL RESPONSE (from API)
// =============================================================================

export interface RunDetailApiResponse {
  run_id: string
  status: string
  status_reason: string | null
  dod_level: number
  pico: {
    population: string
    intervention: string
    comparator: string | null
    outcome: string
    study_types: string[] | null
  } | null
  search_plan: {
    pubmed_query: string
    ct_gov_query: string
    date_range: string[] | null
    languages: string[]
    max_results_per_source: number
    created_at: string | null
  } | null
  prisma_counts: {
    records_identified: number
    records_screened: number
    records_excluded_screening: number
    reports_assessed: number
    reports_not_retrieved: number
    studies_included: number
    exclusion_reasons: Record<string, number>
  } | null
  claims_count: number
  snippets_count: number
  studies_count: number
  hypotheses_count: number
  verification_coverage: number
  errors: string[]
  created_at: string
  updated_at: string
}

// =============================================================================
// RUNS API
// =============================================================================

/**
 * List all runs
 */
export async function listRuns(): Promise<RunSummary[]> {
  const response = await fetch(`${API_BASE}/runs`)
  const raw = await handleResponse<RawRunSummary[]>(response)
  return transformRunSummaries(raw)
}

/**
 * Get run detail by ID
 */
export async function getRunDetail(runId: string): Promise<RunDetailApiResponse> {
  const response = await fetch(`${API_BASE}/runs/${runId}/detail`)
  const raw = await handleResponse<RunDetailApiResponse>(response)
  // Normalize status in detail response
  return {
    ...raw,
    status: raw.status.toUpperCase(),
  }
}

/**
 * Get run status (basic)
 */
export async function getRunStatus(runId: string): Promise<{
  run_id: string
  status: string
  progress: Record<string, unknown>
  errors: string[]
  report_path: string | null
}> {
  const response = await fetch(`${API_BASE}/runs/${runId}`)
  return handleResponse(response)
}

/**
 * Create a new run
 */
export async function createRun(
  request: CreateRunRequest
): Promise<CreateRunResponse> {
  const response = await fetch(`${API_BASE}/runs`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
  return handleResponse<CreateRunResponse>(response)
}

/**
 * Get claims for a run
 */
export async function getRunClaims(runId: string): Promise<EvidenceClaim[]> {
  const response = await fetch(`${API_BASE}/runs/${runId}/claims`)
  const raw = await handleResponse<RawClaimResponse[]>(response)
  return transformClaims(raw)
}

/**
 * Get snippets for a run
 */
export async function getRunSnippets(runId: string): Promise<Snippet[]> {
  const response = await fetch(`${API_BASE}/runs/${runId}/snippets`)
  const raw = await handleResponse<RawSnippetResponse[]>(response)
  return transformSnippets(raw)
}

/**
 * Get studies for a run
 */
export async function getRunStudies(runId: string): Promise<StudyCard[]> {
  const response = await fetch(`${API_BASE}/runs/${runId}/studies`)
  const raw = await handleResponse<RawStudyCardResponse[]>(response)
  return transformStudyCards(raw)
}

/**
 * Get PRISMA counts for a run
 */
export async function getRunPrisma(runId: string): Promise<PRISMACounts> {
  const response = await fetch(`${API_BASE}/runs/${runId}/prisma`)
  return handleResponse<PRISMACounts>(response)
}

/**
 * Get search plan for a run
 */
export async function getRunSearchPlan(runId: string): Promise<SearchPlan> {
  const response = await fetch(`${API_BASE}/runs/${runId}/search-plan`)
  return handleResponse<SearchPlan>(response)
}

/**
 * Get hypotheses for a run
 */
export async function getRunHypotheses(runId: string): Promise<ComposedHypothesis[]> {
  const response = await fetch(`${API_BASE}/runs/${runId}/hypotheses`)
  const raw = await handleResponse<ComposedHypothesis[]>(response)
  return transformHypotheses(raw)
}

/**
 * Get PICO for a run
 */
export async function getRunPico(runId: string): Promise<PICO> {
  const response = await fetch(`${API_BASE}/runs/${runId}/pico`)
  return handleResponse<PICO>(response)
}

/**
 * Get report metadata for a run
 */
export async function getReport(runId: string): Promise<ReportMetadata> {
  const response = await fetch(`${API_BASE}/runs/${runId}/report`)
  return handleResponse<ReportMetadata>(response)
}

/**
 * Get evaluation for a run
 */
export async function getRunEvaluation(runId: string): Promise<Evaluation> {
  const response = await fetch(`${API_BASE}/runs/${runId}/evaluation`)
  return handleResponse<Evaluation>(response)
}

// =============================================================================
// HOOKS HELPERS
// =============================================================================

export const queryKeys = {
  runs: ['runs'] as const,
  run: (id: string) => ['runs', id] as const,
  runDetail: (id: string) => ['runs', id, 'detail'] as const,
  runClaims: (id: string) => ['runs', id, 'claims'] as const,
  runSnippets: (id: string) => ['runs', id, 'snippets'] as const,
  runStudies: (id: string) => ['runs', id, 'studies'] as const,
  runPrisma: (id: string) => ['runs', id, 'prisma'] as const,
  runSearchPlan: (id: string) => ['runs', id, 'search-plan'] as const,
  runHypotheses: (id: string) => ['runs', id, 'hypotheses'] as const,
  runEvaluation: (id: string) => ['runs', id, 'evaluation'] as const,
  report: (id: string) => ['runs', id, 'report'] as const,
}

export { ApiError }
