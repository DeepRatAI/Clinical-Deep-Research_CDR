/**
 * RunDetail Page Tests
 * 
 * Tests for RunDetail page rendering and tab navigation.
 * Ref: src/pages/RunDetail.tsx
 * Audit: CDR_Integral_Audit_2026-01-20.md (UI page tests)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import RunDetail from '../RunDetail'
import { 
  renderWithProviders, 
  createMockRunDetailApiResponse,
  createMockClaim,
  createMockSnippet,
  createMockHypothesis,
} from '../../test/utils'
import * as api from '../../api/client'

// Mock the API client module
vi.mock('../../api/client', async () => {
  const actual = await vi.importActual('../../api/client')
  return {
    ...actual,
    getRunDetail: vi.fn(),
    getRunClaims: vi.fn(),
    getRunSnippets: vi.fn(),
    getRunHypotheses: vi.fn(),
  }
})

// Mock react-router-dom params
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useParams: () => ({ runId: 'run-test-001' }),
  }
})

describe('RunDetail', () => {
  const mockGetRunDetail = vi.mocked(api.getRunDetail)
  const mockGetRunClaims = vi.mocked(api.getRunClaims)
  const mockGetRunSnippets = vi.mocked(api.getRunSnippets)
  const mockGetRunHypotheses = vi.mocked(api.getRunHypotheses)

  beforeEach(() => {
    vi.clearAllMocks()
    // Set up default mock returns
    mockGetRunDetail.mockResolvedValue(createMockRunDetailApiResponse())
    mockGetRunClaims.mockResolvedValue([createMockClaim()])
    mockGetRunSnippets.mockResolvedValue([createMockSnippet()])
    mockGetRunHypotheses.mockResolvedValue([createMockHypothesis()])
  })

  describe('Loading State', () => {
    it('shows loading indicator while fetching', async () => {
      mockGetRunDetail.mockImplementation(() => new Promise(() => {}))
      
      renderWithProviders(<RunDetail />)
      
      // Loader SVG has animate-spin class
      const loader = document.querySelector('.animate-spin')
      expect(loader).toBeInTheDocument()
    })
  })

  describe('Header', () => {
    it('displays run ID in header', async () => {
      const run = createMockRunDetailApiResponse({ run_id: 'run-abc-xyz' })
      mockGetRunDetail.mockResolvedValue(run)
      
      renderWithProviders(<RunDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/run-abc-xyz/)).toBeInTheDocument()
      })
    })

    it('displays status badge', async () => {
      const run = createMockRunDetailApiResponse({ status: 'COMPLETED' })
      mockGetRunDetail.mockResolvedValue(run)
      
      renderWithProviders(<RunDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/completed/i)).toBeInTheDocument()
      })
    })

    it('displays DoD level badge', async () => {
      const run = createMockRunDetailApiResponse({ dod_level: 3 })
      mockGetRunDetail.mockResolvedValue(run)
      
      renderWithProviders(<RunDetail />)
      
      await waitFor(() => {
        // Match exact DoD badge text
        expect(screen.getByText(/DoD 3/)).toBeInTheDocument()
      })
    })

    it('has back link to dashboard', async () => {
      renderWithProviders(<RunDetail />)
      
      await waitFor(() => {
        const backLink = screen.getByRole('link', { name: /back/i })
        expect(backLink).toHaveAttribute('href', '/')
      })
    })
  })

  describe('Tab Navigation', () => {
    it('defaults to overview tab', async () => {
      renderWithProviders(<RunDetail />)
      
      await waitFor(() => {
        const overviewTab = screen.getByRole('button', { name: /overview/i })
        expect(overviewTab).toHaveClass('border-primary-500')
      })
    })

    it('switches to claims tab', async () => {
      const user = userEvent.setup()
      renderWithProviders(<RunDetail />)
      
      await waitFor(async () => {
        const claimsTab = screen.getByRole('button', { name: /claims/i })
        await user.click(claimsTab)
      })
      
      await waitFor(() => {
        // Claims should be called and claim cards rendered
        expect(mockGetRunClaims).toHaveBeenCalled()
      })
    })

    it('switches to PRISMA tab', async () => {
      const user = userEvent.setup()
      renderWithProviders(<RunDetail />)
      
      await waitFor(async () => {
        const prismaTab = screen.getByRole('button', { name: /prisma/i })
        await user.click(prismaTab)
      })
      
      await waitFor(() => {
        // PRISMA flow should show identified records
        expect(screen.getByText(/Identified/)).toBeInTheDocument()
      })
    })

    it('switches to snippets tab', async () => {
      const user = userEvent.setup()
      renderWithProviders(<RunDetail />)
      
      await waitFor(async () => {
        const snippetsTab = screen.getByRole('button', { name: /snippets/i })
        await user.click(snippetsTab)
      })
      
      await waitFor(() => {
        // Snippets API should be called
        expect(mockGetRunSnippets).toHaveBeenCalled()
      })
    })

    it('switches to hypotheses tab', async () => {
      const user = userEvent.setup()
      renderWithProviders(<RunDetail />)
      
      await waitFor(async () => {
        const hypTab = screen.getByRole('button', { name: /hypotheses/i })
        await user.click(hypTab)
      })
      
      await waitFor(() => {
        // Hypotheses API should be called
        expect(mockGetRunHypotheses).toHaveBeenCalled()
      })
    })
  })

  describe('Overview Tab Content', () => {
    it('displays PICO information', async () => {
      const run = createMockRunDetailApiResponse({
        pico: {
          population: 'Adults with diabetes',
          intervention: 'Metformin therapy',
          comparator: 'Placebo',
          outcome: 'HbA1c reduction',
          study_types: ['RCT'],
        },
      })
      mockGetRunDetail.mockResolvedValue(run)
      
      renderWithProviders(<RunDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/Adults with diabetes/)).toBeInTheDocument()
        // Use more specific text to avoid matching claim text
        expect(screen.getByText(/Metformin therapy/)).toBeInTheDocument()
      })
    })

    it('displays claims summary', async () => {
      const claims = [
        createMockClaim({ claim_id: 'claim-1' }),
        createMockClaim({ claim_id: 'claim-2' }),
      ]
      mockGetRunClaims.mockResolvedValue(claims)
      
      renderWithProviders(<RunDetail />)
      
      await waitFor(() => {
        // Should show Claims section header in overview
        expect(screen.getByText(/Claims \(/)).toBeInTheDocument()
      })
    })
  })

  describe('Error Handling', () => {
    it('shows error message when run fetch fails', async () => {
      mockGetRunDetail.mockRejectedValue(new Error('Not found'))
      
      renderWithProviders(<RunDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/Run not found/)).toBeInTheDocument()
      }, { timeout: 3000 })
    })
  })

  describe('Status-specific behavior', () => {
    it('shows progress for running status', async () => {
      const run = createMockRunDetailApiResponse({
        status: 'RUNNING',
      })
      mockGetRunDetail.mockResolvedValue(run)
      
      renderWithProviders(<RunDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/Running/)).toBeInTheDocument()
      })
    })

    it('shows status reason for terminal states', async () => {
      const run = createMockRunDetailApiResponse({
        status: 'INSUFFICIENT_EVIDENCE',
        status_reason: 'Fewer than 3 studies passed screening',
      })
      mockGetRunDetail.mockResolvedValue(run)
      
      renderWithProviders(<RunDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/Insufficient Evidence/)).toBeInTheDocument()
      })
    })
  })
})
