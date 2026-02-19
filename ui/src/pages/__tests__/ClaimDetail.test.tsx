/**
 * ClaimDetail Page Tests
 * 
 * Tests for ClaimDetail page rendering and GRADE display.
 * Ref: src/pages/ClaimDetail.tsx
 * Audit: CDR_Integral_Audit_2026-01-20.md (UI page tests)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ClaimDetail from '../ClaimDetail'
import { 
  renderWithProviders, 
  createMockRunDetailApiResponse,
  createMockClaim,
  createMockSnippet,
} from '../../test/utils'
import * as api from '../../api'

// Mock the API module
vi.mock('../../api', async () => {
  const actual = await vi.importActual('../../api')
  return {
    ...actual,
    getRunDetail: vi.fn(),
    getRunClaims: vi.fn(),
    getRunSnippets: vi.fn(),
  }
})

// Mock react-router-dom params
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useParams: () => ({ runId: 'run-test-001', claimId: 'claim-001' }),
  }
})

describe('ClaimDetail', () => {
  const mockGetRunDetail = vi.mocked(api.getRunDetail)
  const mockGetRunClaims = vi.mocked(api.getRunClaims)
  const mockGetRunSnippets = vi.mocked(api.getRunSnippets)

  const defaultClaim = createMockClaim({
    claim_id: 'claim-001',
    claim_text: 'Metformin significantly reduces HbA1c levels',
    certainty: 'HIGH',
    supporting_snippet_ids: ['snip-001', 'snip-002'],
    verification_status: 'VERIFIED',
    verification_score: 0.92,
    grade_rationale: {
      risk_of_bias: 'Low risk',
      inconsistency: 'Consistent results across studies',
      indirectness: 'Direct evidence',
      imprecision: 'Narrow confidence intervals',
      publication_bias: 'No evidence of bias',
    },
  })

  const defaultSnippets = [
    createMockSnippet({ snippet_id: 'snip-001', text: 'Evidence snippet 1' }),
    createMockSnippet({ snippet_id: 'snip-002', text: 'Evidence snippet 2' }),
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    mockGetRunDetail.mockResolvedValue(createMockRunDetailApiResponse())
    mockGetRunClaims.mockResolvedValue([defaultClaim])
    mockGetRunSnippets.mockResolvedValue(defaultSnippets)
  })

  describe('Loading State', () => {
    it('shows loading indicator while fetching', async () => {
      mockGetRunDetail.mockImplementation(() => new Promise(() => {}))
      
      renderWithProviders(<ClaimDetail />)
      
      // Loader SVG has animate-spin class
      const loader = document.querySelector('.animate-spin')
      expect(loader).toBeInTheDocument()
    })
  })

  describe('Claim Display', () => {
    it('displays claim text', async () => {
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/Metformin significantly reduces HbA1c/)).toBeInTheDocument()
      })
    })

    it('displays certainty badge', async () => {
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/HIGH/i)).toBeInTheDocument()
      })
    })

    it('displays verification status icon', async () => {
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        // VERIFIED status should show green checkmark
        const verifiedIcon = document.querySelector('.text-green-500')
        expect(verifiedIcon).toBeInTheDocument()
      })
    })

    it('displays verification score', async () => {
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        expect(screen.getByText('92%')).toBeInTheDocument()
      })
    })
  })

  describe('GRADE Rationale', () => {
    it('displays GRADE section collapsed by default', async () => {
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/GRADE/i)).toBeInTheDocument()
      })
    })

    it('expands GRADE details on click', async () => {
      const user = userEvent.setup()
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(async () => {
        const gradeToggle = screen.getByRole('button', { name: /grade/i })
        await user.click(gradeToggle)
      })
      
      await waitFor(() => {
        // GRADE domains should be visible
        expect(screen.getByText(/risk of bias/i)).toBeInTheDocument()
        expect(screen.getByText(/inconsistency/i)).toBeInTheDocument()
        expect(screen.getByText(/indirectness/i)).toBeInTheDocument()
        expect(screen.getByText(/imprecision/i)).toBeInTheDocument()
        expect(screen.getByText(/publication bias/i)).toBeInTheDocument()
      })
    })

    it('displays GRADE domain values', async () => {
      const user = userEvent.setup()
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(async () => {
        const gradeToggle = screen.getByRole('button', { name: /grade/i })
        await user.click(gradeToggle)
      })
      
      await waitFor(() => {
        expect(screen.getByText(/Low risk/)).toBeInTheDocument()
        expect(screen.getByText(/Consistent results/)).toBeInTheDocument()
      })
    })
  })

  describe('Supporting Snippets', () => {
    it('displays supporting snippets section', async () => {
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/Supporting Snippets/)).toBeInTheDocument()
      })
    })

    it('displays correct number of snippets', async () => {
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        // Match the specific header text "Supporting Snippets (2)"
        expect(screen.getByText(/Supporting Snippets \(2\)/)).toBeInTheDocument()
      })
    })

    it('renders snippet cards', async () => {
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/Evidence snippet 1/)).toBeInTheDocument()
        expect(screen.getByText(/Evidence snippet 2/)).toBeInTheDocument()
      })
    })
  })

  describe('Navigation', () => {
    it('has back link to run detail', async () => {
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        const backLink = screen.getByRole('link', { name: /back/i })
        expect(backLink).toHaveAttribute('href', '/runs/run-test-001')
      })
    })
  })

  describe('Not Found State', () => {
    it('shows error when claim not found', async () => {
      mockGetRunClaims.mockResolvedValue([
        createMockClaim({ claim_id: 'different-claim' }),
      ])
      
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        expect(screen.getByText(/claim not found/i)).toBeInTheDocument()
      })
    })

    it('shows link to go back to run', async () => {
      mockGetRunClaims.mockResolvedValue([])
      
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        const backLink = screen.getByRole('link', { name: /back to run/i })
        expect(backLink).toHaveAttribute('href', '/runs/run-test-001')
      })
    })
  })

  describe('Different Verification Statuses', () => {
    it('displays REFUTED status correctly', async () => {
      const refutedClaim = createMockClaim({
        claim_id: 'claim-001',
        verification_status: 'REFUTED',
      })
      mockGetRunClaims.mockResolvedValue([refutedClaim])
      
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        // REFUTED status should show red icon
        const refutedIcon = document.querySelector('.text-red-500')
        expect(refutedIcon).toBeInTheDocument()
      })
    })

    it('displays PARTIAL status correctly', async () => {
      const partialClaim = createMockClaim({
        claim_id: 'claim-001',
        verification_status: 'PARTIAL',
      })
      mockGetRunClaims.mockResolvedValue([partialClaim])
      
      renderWithProviders(<ClaimDetail />)
      
      await waitFor(() => {
        // PARTIAL status should show yellow icon
        const partialIcon = document.querySelector('.text-yellow-500')
        expect(partialIcon).toBeInTheDocument()
      })
    })
  })
})
