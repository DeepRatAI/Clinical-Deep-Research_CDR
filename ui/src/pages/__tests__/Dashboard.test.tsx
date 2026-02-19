/**
 * Dashboard Page Tests
 * 
 * Tests for Dashboard page rendering and interactions.
 * Ref: src/pages/Dashboard.tsx
 * Audit: CDR_Integral_Audit_2026-01-20.md (UI page tests)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Dashboard from '../Dashboard'
import { renderWithProviders, createMockRunSummary } from '../../test/utils'
import * as api from '../../api'

// Mock the API module
vi.mock('../../api', async () => {
  const actual = await vi.importActual('../../api')
  return {
    ...actual,
    listRuns: vi.fn(),
  }
})

describe('Dashboard', () => {
  const mockListRuns = vi.mocked(api.listRuns)

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Loading State', () => {
    it('shows loading indicator while fetching', async () => {
      // Never resolve to keep loading state
      mockListRuns.mockImplementation(() => new Promise(() => {}))
      
      renderWithProviders(<Dashboard />)
      
      // Refresh button should indicate loading
      const refreshBtn = screen.getByRole('button', { name: /refresh/i })
      expect(refreshBtn).toBeDisabled()
    })
  })

  describe('Empty State', () => {
    it('renders empty table when no runs exist', async () => {
      mockListRuns.mockResolvedValue([])
      
      renderWithProviders(<Dashboard />)
      
      await waitFor(() => {
        // Table headers should still be present
        expect(screen.getByText('Run ID')).toBeInTheDocument()
        expect(screen.getByText('Status')).toBeInTheDocument()
      })
    })
  })

  describe('Data Display', () => {
    it('renders run list correctly', async () => {
      const runs = [
        createMockRunSummary({
          run_id: 'run-abc-123',
          status: 'COMPLETED',
          dod_level: 3,
          claims_count: 5,
        }),
        createMockRunSummary({
          run_id: 'run-def-456',
          status: 'RUNNING',
          dod_level: 1,
          claims_count: 0,
        }),
      ]
      mockListRuns.mockResolvedValue(runs)
      
      renderWithProviders(<Dashboard />)
      
      await waitFor(() => {
        expect(screen.getByText(/run-abc-/)).toBeInTheDocument()
        expect(screen.getByText(/run-def-/)).toBeInTheDocument()
      })
    })

    it('displays status badges', async () => {
      const runs = [
        createMockRunSummary({ status: 'COMPLETED' }),
      ]
      mockListRuns.mockResolvedValue(runs)
      
      renderWithProviders(<Dashboard />)
      
      await waitFor(() => {
        expect(screen.getByText(/Completed/)).toBeInTheDocument()
      })
    })

    it('displays DoD level badges', async () => {
      const runs = [
        createMockRunSummary({ dod_level: 3 }),
      ]
      mockListRuns.mockResolvedValue(runs)
      
      renderWithProviders(<Dashboard />)
      
      await waitFor(() => {
        // Level 3 should show
        expect(screen.getByText(/3/)).toBeInTheDocument()
      })
    })

    it('displays claims count', async () => {
      const runs = [
        createMockRunSummary({ claims_count: 7 }),
      ]
      mockListRuns.mockResolvedValue(runs)
      
      renderWithProviders(<Dashboard />)
      
      await waitFor(() => {
        expect(screen.getByText('7')).toBeInTheDocument()
      })
    })

    it('displays verification coverage as percentage', async () => {
      const runs = [
        createMockRunSummary({ verification_coverage: 0.85 }),
      ]
      mockListRuns.mockResolvedValue(runs)
      
      renderWithProviders(<Dashboard />)
      
      await waitFor(() => {
        expect(screen.getByText('85%')).toBeInTheDocument()
      })
    })
  })

  describe('Navigation', () => {
    it('links run ID to detail page', async () => {
      const runs = [
        createMockRunSummary({ run_id: 'run-nav-test' }),
      ]
      mockListRuns.mockResolvedValue(runs)
      
      renderWithProviders(<Dashboard />)
      
      await waitFor(() => {
        const link = screen.getByRole('link', { name: /run-nav-/i })
        expect(link).toHaveAttribute('href', '/runs/run-nav-test')
      })
    })

    it('has New Run button linking to /runs/new', () => {
      mockListRuns.mockResolvedValue([])
      
      renderWithProviders(<Dashboard />)
      
      const newRunLink = screen.getByRole('link', { name: /new run/i })
      expect(newRunLink).toHaveAttribute('href', '/runs/new')
    })
  })

  describe('Refresh Functionality', () => {
    it('refetch button calls listRuns again', async () => {
      mockListRuns.mockResolvedValue([])
      const user = userEvent.setup()
      
      renderWithProviders(<Dashboard />)
      
      await waitFor(() => {
        expect(mockListRuns).toHaveBeenCalledTimes(1)
      })
      
      const refreshBtn = screen.getByRole('button', { name: /refresh/i })
      await user.click(refreshBtn)
      
      await waitFor(() => {
        // Called at least twice (initial + refresh)
        expect(mockListRuns.mock.calls.length).toBeGreaterThanOrEqual(2)
      })
    })
  })

  describe('Error Handling', () => {
    it('shows error message when fetch fails', async () => {
      mockListRuns.mockRejectedValue(new Error('Network error'))
      
      renderWithProviders(<Dashboard />)
      
      await waitFor(() => {
        expect(screen.getByText(/failed to load runs/i)).toBeInTheDocument()
      })
    })
  })
})
