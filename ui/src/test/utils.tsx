/**
 * Test Utilities for CDR UI
 * 
 * Common test helpers, mock data, and render functions.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { render, type RenderOptions } from '@testing-library/react'
import type { ReactElement, ReactNode } from 'react'
import type { RunDetailApiResponse } from '../api/client'
import type { 
  EvidenceClaim, 
  Snippet, 
  PRISMACounts, 
  RunSummary,
  RunDetail,
  ComposedHypothesis,
  SourceRef,
} from '../types'

// -----------------------------------------------------------------------------
// Test Query Client
// -----------------------------------------------------------------------------

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
    },
  })
}

// -----------------------------------------------------------------------------
// Custom Render with Providers
// -----------------------------------------------------------------------------

interface WrapperProps {
  children: ReactNode
}

function AllTheProviders({ children }: WrapperProps) {
  const queryClient = createTestQueryClient()
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export function renderWithProviders(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) {
  return render(ui, { wrapper: AllTheProviders, ...options })
}

// -----------------------------------------------------------------------------
// Mock Data Factories
// -----------------------------------------------------------------------------

export function createMockSourceRef(overrides?: Partial<SourceRef>): SourceRef {
  return {
    record_id: 'rec-001',
    pmid: '12345678',
    doi: '10.1000/test',
    nct_id: null,
    title: 'Test Study Title',
    authors: ['Author A', 'Author B'],
    publication_year: 2024,
    journal: 'Test Journal',
    url: 'https://pubmed.ncbi.nlm.nih.gov/12345678',
    ...overrides,
  }
}

export function createMockSnippet(overrides?: Partial<Snippet>): Snippet {
  return {
    snippet_id: 'snip-001',
    source_ref: createMockSourceRef(),
    section: 'RESULTS',
    text: 'This is a test snippet with evidence text.',
    offset_start: 100,
    offset_end: 200,
    relevance_score: 0.85,
    ...overrides,
  }
}

export function createMockClaim(overrides?: Partial<EvidenceClaim>): EvidenceClaim {
  return {
    claim_id: 'claim-001',
    claim_text: 'Metformin reduces HbA1c more effectively than sulfonylureas.',
    certainty: 'HIGH',
    supporting_snippet_ids: ['snip-001', 'snip-002'],
    grade_rationale: {
      risk_of_bias: 'Low risk across all domains',
      inconsistency: 'Consistent results',
      indirectness: 'Direct evidence',
      imprecision: 'Narrow confidence intervals',
      publication_bias: 'No evidence of bias',
    },
    verification_status: 'VERIFIED',
    verification_score: 0.92,
    ...overrides,
  }
}

export function createMockPrismaCounts(overrides?: Partial<PRISMACounts>): PRISMACounts {
  return {
    records_identified: 500,
    records_screened: 450,
    records_excluded_screening: 380,
    reports_assessed: 70,
    reports_not_retrieved: 5,
    studies_included: 12,
    exclusion_reasons: {
      'Not RCT': 200,
      'Wrong population': 100,
      'No relevant outcome': 80,
    },
    ...overrides,
  }
}

export function createMockHypothesis(overrides?: Partial<ComposedHypothesis>): ComposedHypothesis {
  return {
    hypothesis_id: 'hyp-001',
    claim_a_id: 'claim-001',
    claim_b_id: 'claim-002',
    hypothesis_text: 'If metformin reduces HbA1c more than sulfonylureas, then it may also reduce cardiovascular events.',
    mechanism: 'AMPK activation leading to improved endothelial function',
    rival_hypotheses: [
      'Weight loss may be the mediating factor',
      'Baseline HbA1c differences confound results',
    ],
    threats_to_validity: [
      'Limited follow-up duration',
      'Heterogeneous populations',
    ],
    mcid: '0.5% HbA1c reduction',
    test_design: 'Prospective RCT with cardiovascular outcomes',
    confidence: 0.78,
    ...overrides,
  }
}

export function createMockRunSummary(overrides?: Partial<RunSummary>): RunSummary {
  return {
    run_id: 'run-test-001',
    status: 'COMPLETED',
    status_reason: null,
    dod_level: 2,
    claims_count: 5,
    verification_coverage: 0.85,
    created_at: '2026-01-20T10:00:00Z',
    updated_at: '2026-01-20T10:30:00Z',
    ...overrides,
  }
}

export function createMockRunDetail(overrides?: Partial<RunDetail>): RunDetail {
  return {
    run_id: 'run-test-001',
    status: 'COMPLETED',
    status_reason: null,
    dod_level: 2,
    progress: null,
    pico: {
      population: 'Adults with type 2 diabetes',
      intervention: 'Metformin',
      comparator: 'Sulfonylureas',
      outcome: 'HbA1c reduction',
      study_types: ['RCT'],
    },
    search_plan: null,
    prisma_counts: createMockPrismaCounts(),
    records: [],
    snippets: [createMockSnippet()],
    study_cards: [],
    claims: [createMockClaim()],
    rob2_results: [],
    composed_hypotheses: [createMockHypothesis()],
    errors: [],
    created_at: '2026-01-20T10:00:00Z',
    updated_at: '2026-01-20T10:30:00Z',
    ...overrides,
  }
}

/**
 * Factory for RunDetailApiResponse (API response format).
 * Use this for mocking getRunDetail() API calls.
 */
export function createMockRunDetailApiResponse(
  overrides?: Partial<RunDetailApiResponse>
): RunDetailApiResponse {
  const prismaCounts = createMockPrismaCounts()
  return {
    run_id: 'run-test-001',
    status: 'COMPLETED',
    status_reason: null,
    dod_level: 2,
    pico: {
      population: 'Adults with type 2 diabetes',
      intervention: 'Metformin',
      comparator: 'Sulfonylureas',
      outcome: 'HbA1c reduction',
      study_types: ['RCT'],
    },
    search_plan: null,
    prisma_counts: prismaCounts,
    claims_count: 1,
    snippets_count: 1,
    studies_count: prismaCounts.studies_included,
    hypotheses_count: 1,
    verification_coverage: 0.85,
    errors: [],
    created_at: '2026-01-20T10:00:00Z',
    updated_at: '2026-01-20T10:30:00Z',
    ...overrides,
  }
}

// -----------------------------------------------------------------------------
// API Mock Helpers
// -----------------------------------------------------------------------------

export const mockApiHandlers = {
  listRuns: () => [createMockRunSummary()],
  getRunDetail: () => createMockRunDetail(),
  getRunClaims: () => [createMockClaim()],
  getRunSnippets: () => [createMockSnippet()],
  getRunHypotheses: () => [createMockHypothesis()],
  getRunPrisma: () => createMockPrismaCounts(),
}
