/**
 * ClaimCard Component Tests
 * 
 * Tests for ClaimCard and ClaimsList components.
 * Ref: src/components/claims/ClaimCard.tsx
 */

import { describe, it, expect } from 'vitest'
import { screen } from '@testing-library/react'
import { ClaimCard, ClaimsList } from '../ClaimCard'
import { renderWithProviders, createMockClaim } from '../../../test/utils'

describe('ClaimCard', () => {
  const runId = 'run-test-001'

  it('renders claim text', () => {
    const claim = createMockClaim({ claim_text: 'Test claim for unit testing' })
    renderWithProviders(<ClaimCard claim={claim} runId={runId} />)
    
    expect(screen.getByText(/Test claim for unit testing/)).toBeInTheDocument()
  })

  it('displays certainty badge', () => {
    const claim = createMockClaim({ certainty: 'HIGH' })
    renderWithProviders(<ClaimCard claim={claim} runId={runId} />)
    
    expect(screen.getByText(/HIGH/i)).toBeInTheDocument()
  })

  it('displays verification score when present', () => {
    const claim = createMockClaim({ verification_score: 0.85 })
    renderWithProviders(<ClaimCard claim={claim} runId={runId} />)
    
    expect(screen.getByTestId('verification-coverage')).toHaveTextContent('coverage 0.85')
  })

  it('shows supporting snippet count', () => {
    const claim = createMockClaim({ 
      supporting_snippet_ids: ['snip-1', 'snip-2', 'snip-3'] 
    })
    renderWithProviders(<ClaimCard claim={claim} runId={runId} />)
    
    expect(screen.getByText(/3 supporting snippet\(s\)/)).toBeInTheDocument()
  })

  it('displays claim ID', () => {
    const claim = createMockClaim({ claim_id: 'claim-abc-123' })
    renderWithProviders(<ClaimCard claim={claim} runId={runId} />)
    
    expect(screen.getByText('claim-abc-123')).toBeInTheDocument()
  })

  it('links to claim detail page', () => {
    const claim = createMockClaim({ claim_id: 'claim-link-test' })
    renderWithProviders(<ClaimCard claim={claim} runId={runId} />)
    
    const link = screen.getByRole('link')
    expect(link).toHaveAttribute('href', `/runs/${runId}/claims/claim-link-test`)
  })

  it('renders verification icon for VERIFIED status', () => {
    const claim = createMockClaim({ verification_status: 'VERIFIED' })
    renderWithProviders(<ClaimCard claim={claim} runId={runId} />)
    
    // CheckCircle icon should be present (green-500 color class)
    const icon = document.querySelector('.text-green-500')
    expect(icon).toBeInTheDocument()
  })

  it('renders verification icon for REFUTED status', () => {
    const claim = createMockClaim({ verification_status: 'REFUTED' })
    renderWithProviders(<ClaimCard claim={claim} runId={runId} />)
    
    // XCircle icon should be present (red-500 color class)
    const icon = document.querySelector('.text-red-500')
    expect(icon).toBeInTheDocument()
  })

  it('handles null verification_score gracefully', () => {
    const claim = createMockClaim({ verification_score: null })
    renderWithProviders(<ClaimCard claim={claim} runId={runId} />)
    
    // Should not show percentage
    expect(screen.queryByText(/%/)).not.toBeInTheDocument()
  })
})

describe('ClaimsList', () => {
  const runId = 'run-test-001'

  it('renders multiple claims', () => {
    const claims = [
      createMockClaim({ claim_id: 'claim-1', claim_text: 'First claim' }),
      createMockClaim({ claim_id: 'claim-2', claim_text: 'Second claim' }),
      createMockClaim({ claim_id: 'claim-3', claim_text: 'Third claim' }),
    ]
    renderWithProviders(<ClaimsList claims={claims} runId={runId} />)
    
    expect(screen.getByText(/First claim/)).toBeInTheDocument()
    expect(screen.getByText(/Second claim/)).toBeInTheDocument()
    expect(screen.getByText(/Third claim/)).toBeInTheDocument()
  })

  it('shows empty state when no claims', () => {
    renderWithProviders(<ClaimsList claims={[]} runId={runId} />)
    
    expect(screen.getByText(/No claims extracted/)).toBeInTheDocument()
    expect(screen.getByText(/evidence may be insufficient/i)).toBeInTheDocument()
  })

  it('creates correct links for each claim', () => {
    const claims = [
      createMockClaim({ claim_id: 'claim-a' }),
      createMockClaim({ claim_id: 'claim-b' }),
    ]
    renderWithProviders(<ClaimsList claims={claims} runId={runId} />)
    
    const links = screen.getAllByRole('link')
    expect(links).toHaveLength(2)
    expect(links[0]).toHaveAttribute('href', `/runs/${runId}/claims/claim-a`)
    expect(links[1]).toHaveAttribute('href', `/runs/${runId}/claims/claim-b`)
  })
})
