/**
 * API Response Transformers
 * 
 * Transform backend API responses to frontend-expected formats.
 * This layer handles structure mismatches between backend and frontend types.
 * 
 * Key transformations:
 * - Case normalization for enums (lowercase → UPPERCASE)
 * - Flat → nested object restructuring (SnippetResponse → Snippet)
 * - Adding missing fields with defaults
 */

import type {
  RunSummary,
  EvidenceClaim,
  Snippet,
  StudyCard,
  ComposedHypothesis,
  RunStatus,
  CertaintyLevel,
  VerificationStatus,
  Section,
  StudyType,
  SourceRef,
} from '../types'

// =============================================================================
// RAW API RESPONSE TYPES (what backend actually returns)
// =============================================================================

/** Raw snippet response from backend (flat structure) */
export interface RawSnippetResponse {
  snippet_id: string
  record_id: string
  pmid: string | null
  doi: string | null
  nct_id: string | null
  title: string
  authors: string[]
  publication_year: number | null
  journal: string | null
  url: string | null
  section: string
  text: string
  offset_start: number | null
  offset_end: number | null
  relevance_score: number | null
}

/** Raw claim response from backend */
export interface RawClaimResponse {
  claim_id: string
  claim_text: string
  certainty: string  // lowercase
  supporting_snippet_ids: string[]
  verification_status: string | null  // lowercase
  grade_rationale: Record<string, string> | null
}

/** Raw study card response from backend */
export interface RawStudyCardResponse {
  study_id: string
  record_id: string
  title: string
  study_type: string  // lowercase
  population: string | null
  intervention: string | null
  comparator: string | null
  outcomes: Array<{
    name: string
    value: number | null
    unit: string | null
    effect_size: number | null
    confidence_interval: string | null  // "[lower, upper]" as string
    p_value: number | null
    is_significant: boolean | null
    direction: string | null
  }>
  sample_size: number | null
  follow_up_duration: string | null
}

/** Raw run summary from backend */
export interface RawRunSummary {
  run_id: string
  status: string  // lowercase
  status_reason: string | null
  dod_level: number
  claims_count: number
  verification_coverage: number
  created_at: string
  updated_at: string
}

// =============================================================================
// CASE NORMALIZERS
// =============================================================================

/** Normalize run status to uppercase */
function normalizeRunStatus(status: string): RunStatus {
  const upper = status.toUpperCase()
  const validStatuses: RunStatus[] = ['PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'INSUFFICIENT_EVIDENCE']
  if (validStatuses.includes(upper as RunStatus)) {
    return upper as RunStatus
  }
  // Fallback for edge cases
  return 'RUNNING'
}

/** Normalize certainty level to uppercase */
function normalizeCertainty(certainty: string): CertaintyLevel {
  const upper = certainty.toUpperCase()
  const validLevels: CertaintyLevel[] = ['HIGH', 'MODERATE', 'LOW', 'VERY_LOW']
  if (validLevels.includes(upper as CertaintyLevel)) {
    return upper as CertaintyLevel
  }
  return 'LOW'
}

/** Normalize verification status to uppercase */
function normalizeVerificationStatus(status: string | null): VerificationStatus | null {
  if (!status) return null
  const upper = status.toUpperCase()
  const validStatuses: VerificationStatus[] = ['VERIFIED', 'PARTIAL', 'REFUTED', 'UNVERIFIABLE']
  if (validStatuses.includes(upper as VerificationStatus)) {
    return upper as VerificationStatus
  }
  return 'UNVERIFIABLE'
}

/** Normalize section to uppercase */
function normalizeSection(section: string): Section {
  const upper = section.toUpperCase()
  const validSections: Section[] = [
    'ABSTRACT', 'INTRODUCTION', 'METHODS', 'RESULTS', 
    'DISCUSSION', 'CONCLUSION', 'FULL_TEXT', 'OTHER'
  ]
  if (validSections.includes(upper as Section)) {
    return upper as Section
  }
  return 'OTHER'
}

/** Normalize study type to uppercase */
function normalizeStudyType(studyType: string): StudyType {
  // Map common lowercase/variant values to standard enum values
  const mappings: Record<string, StudyType> = {
    'rct': 'RCT',
    'randomized_controlled_trial': 'RCT',
    'quasi_experimental': 'QUASI_EXPERIMENTAL',
    'cohort': 'COHORT',
    'case_control': 'CASE_CONTROL',
    'cross_sectional': 'CROSS_SECTIONAL',
    'case_series': 'CASE_SERIES',
    'case_report': 'CASE_REPORT',
    'systematic_review': 'SYSTEMATIC_REVIEW',
    'meta_analysis': 'META_ANALYSIS',
    'other': 'OTHER',
  }
  
  const lower = studyType.toLowerCase()
  if (lower in mappings) {
    return mappings[lower]
  }
  
  // Try uppercase directly
  const upper = studyType.toUpperCase()
  const validTypes: StudyType[] = [
    'RCT', 'QUASI_EXPERIMENTAL', 'COHORT', 'CASE_CONTROL',
    'CROSS_SECTIONAL', 'CASE_SERIES', 'CASE_REPORT',
    'SYSTEMATIC_REVIEW', 'META_ANALYSIS', 'OTHER'
  ]
  if (validTypes.includes(upper as StudyType)) {
    return upper as StudyType
  }
  
  return 'OTHER'
}

// =============================================================================
// TRANSFORMERS
// =============================================================================

/**
 * Transform raw snippet response to frontend Snippet type.
 * Restructures flat fields into nested source_ref object.
 */
export function transformSnippet(raw: RawSnippetResponse): Snippet {
  const sourceRef: SourceRef = {
    record_id: raw.record_id,
    pmid: raw.pmid,
    doi: raw.doi,
    nct_id: raw.nct_id,
    title: raw.title,
    authors: raw.authors || [],
    publication_year: raw.publication_year,
    journal: raw.journal,
    url: raw.url,
  }

  return {
    snippet_id: raw.snippet_id,
    source_ref: sourceRef,
    section: normalizeSection(raw.section),
    text: raw.text,
    offset_start: raw.offset_start,
    offset_end: raw.offset_end,
    relevance_score: raw.relevance_score,
  }
}

/**
 * Transform array of raw snippets
 */
export function transformSnippets(raw: RawSnippetResponse[]): Snippet[] {
  return raw.map(transformSnippet)
}

/**
 * Transform raw claim response to frontend EvidenceClaim type.
 * Normalizes case and adds missing verification_score.
 */
export function transformClaim(raw: RawClaimResponse): EvidenceClaim {
  return {
    claim_id: raw.claim_id,
    claim_text: raw.claim_text,
    certainty: normalizeCertainty(raw.certainty),
    supporting_snippet_ids: raw.supporting_snippet_ids || [],
    grade_rationale: raw.grade_rationale ? {
      risk_of_bias: raw.grade_rationale.risk_of_bias ?? null,
      inconsistency: raw.grade_rationale.inconsistency ?? null,
      indirectness: raw.grade_rationale.indirectness ?? null,
      imprecision: raw.grade_rationale.imprecision ?? null,
      publication_bias: raw.grade_rationale.publication_bias ?? null,
    } : null,
    verification_status: normalizeVerificationStatus(raw.verification_status),
    verification_score: null,  // Not provided by backend, default to null
  }
}

/**
 * Transform array of raw claims
 */
export function transformClaims(raw: RawClaimResponse[]): EvidenceClaim[] {
  return raw.map(transformClaim)
}

/**
 * Transform raw study card response to frontend StudyCard type.
 * Normalizes study_type case and restructures outcomes.
 */
export function transformStudyCard(raw: RawStudyCardResponse): StudyCard {
  const outcomes = (raw.outcomes || []).map(o => ({
    name: o.name,
    value: o.value,
    unit: o.unit,
    effect_size: o.effect_size,
    // Parse confidence_interval string to tuple
    confidence_interval: parseConfidenceInterval(o.confidence_interval),
    p_value: o.p_value,
    is_significant: o.is_significant,
    direction: o.direction as 'positive' | 'negative' | 'neutral' | null,
  }))

  return {
    study_id: raw.study_id,
    record_id: raw.record_id,
    title: raw.title,
    study_type: normalizeStudyType(raw.study_type),
    population: raw.population,
    intervention: raw.intervention,
    comparator: raw.comparator,
    outcomes,
    supporting_snippet_ids: [],  // Not provided by backend
    sample_size: raw.sample_size,
    follow_up_duration: raw.follow_up_duration,
  }
}

/**
 * Parse confidence interval string "[lower, upper]" to tuple
 */
function parseConfidenceInterval(ci: string | null): [number, number] | null {
  if (!ci) return null
  
  // Handle string format "[1.2, 3.4]"
  const match = ci.match(/\[?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]?/)
  if (match) {
    const lower = parseFloat(match[1])
    const upper = parseFloat(match[2])
    if (!isNaN(lower) && !isNaN(upper)) {
      return [lower, upper]
    }
  }
  
  return null
}

/**
 * Transform array of raw study cards
 */
export function transformStudyCards(raw: RawStudyCardResponse[]): StudyCard[] {
  return raw.map(transformStudyCard)
}

/**
 * Transform raw run summary to frontend RunSummary type.
 * Normalizes status case.
 */
export function transformRunSummary(raw: RawRunSummary): RunSummary {
  return {
    run_id: raw.run_id,
    status: normalizeRunStatus(raw.status),
    status_reason: raw.status_reason,
    dod_level: raw.dod_level,
    claims_count: raw.claims_count,
    verification_coverage: raw.verification_coverage,
    created_at: raw.created_at,
    updated_at: raw.updated_at,
  }
}

/**
 * Transform array of raw run summaries
 */
export function transformRunSummaries(raw: RawRunSummary[]): RunSummary[] {
  return raw.map(transformRunSummary)
}

/**
 * Normalize hypotheses (already mostly correct, just ensure types)
 */
export function transformHypotheses(raw: ComposedHypothesis[]): ComposedHypothesis[] {
  return raw.map(h => ({
    hypothesis_id: h.hypothesis_id || '',
    claim_a_id: h.claim_a_id || '',
    claim_b_id: h.claim_b_id || '',
    hypothesis_text: h.hypothesis_text || '',
    mechanism: h.mechanism || '',
    rival_hypotheses: h.rival_hypotheses || [],
    threats_to_validity: h.threats_to_validity || [],
    mcid: h.mcid,
    test_design: h.test_design,
    confidence: h.confidence ?? 0,
  }))
}
