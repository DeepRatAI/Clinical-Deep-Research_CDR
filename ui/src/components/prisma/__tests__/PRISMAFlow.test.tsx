/**
 * PRISMAFlow Component Tests
 * 
 * Tests for PRISMAFlow and PRISMACompact components.
 * Ref: src/components/prisma/PRISMAFlow.tsx
 */

import { describe, it, expect } from 'vitest'
import { screen } from '@testing-library/react'
import { PRISMAFlow, PRISMACompact } from '../PRISMAFlow'
import { renderWithProviders, createMockPrismaCounts } from '../../../test/utils'

describe('PRISMAFlow', () => {
  it('renders PRISMA Flow title', () => {
    const counts = createMockPrismaCounts()
    renderWithProviders(<PRISMAFlow counts={counts} />)
    
    expect(screen.getByText('PRISMA Flow')).toBeInTheDocument()
  })

  it('displays records identified count', () => {
    const counts = createMockPrismaCounts({ records_identified: 1250 })
    renderWithProviders(<PRISMAFlow counts={counts} />)
    
    expect(screen.getByText('1250')).toBeInTheDocument()
    expect(screen.getByText('Identified')).toBeInTheDocument()
  })

  it('displays records screened count', () => {
    const counts = createMockPrismaCounts({ records_screened: 800 })
    renderWithProviders(<PRISMAFlow counts={counts} />)
    
    expect(screen.getByText('800')).toBeInTheDocument()
    expect(screen.getByText('Screened')).toBeInTheDocument()
  })

  it('displays reports assessed count', () => {
    const counts = createMockPrismaCounts({ reports_assessed: 150 })
    renderWithProviders(<PRISMAFlow counts={counts} />)
    
    expect(screen.getByText('150')).toBeInTheDocument()
    expect(screen.getByText('Assessed')).toBeInTheDocument()
  })

  it('displays studies included count', () => {
    const counts = createMockPrismaCounts({ studies_included: 25 })
    renderWithProviders(<PRISMAFlow counts={counts} />)
    
    expect(screen.getByText('25')).toBeInTheDocument()
    expect(screen.getByText('Included')).toBeInTheDocument()
  })

  it('displays excluded screening count', () => {
    const counts = createMockPrismaCounts({ 
      records_identified: 1000, // Different from excluded to avoid duplicate
      records_excluded_screening: 500 
    })
    renderWithProviders(<PRISMAFlow counts={counts} />)
    
    // Use getAllByText since counts may appear in multiple places
    const elements = screen.getAllByText('500')
    expect(elements.length).toBeGreaterThan(0)
    expect(screen.getByText(/Excluded \(screening\):/)).toBeInTheDocument()
  })

  it('displays not retrieved count', () => {
    const counts = createMockPrismaCounts({ reports_not_retrieved: 10 })
    renderWithProviders(<PRISMAFlow counts={counts} />)
    
    expect(screen.getByText('10')).toBeInTheDocument()
    expect(screen.getByText(/Not retrieved:/)).toBeInTheDocument()
  })

  it('displays exclusion reasons', () => {
    const counts = createMockPrismaCounts({
      exclusion_reasons: {
        'Not RCT': 100,
        'Wrong population': 50,
        'No outcome data': 25,
      },
    })
    renderWithProviders(<PRISMAFlow counts={counts} />)
    
    expect(screen.getByText(/Not RCT:/)).toBeInTheDocument()
    expect(screen.getByText(/Wrong population:/)).toBeInTheDocument()
    expect(screen.getByText(/No outcome data:/)).toBeInTheDocument()
  })

  it('does not show exclusion reasons section when empty', () => {
    const counts = createMockPrismaCounts({ exclusion_reasons: {} })
    renderWithProviders(<PRISMAFlow counts={counts} />)
    
    expect(screen.queryByText('Exclusion reasons:')).not.toBeInTheDocument()
  })

  it('applies custom className', () => {
    const counts = createMockPrismaCounts()
    const { container } = renderWithProviders(
      <PRISMAFlow counts={counts} className="custom-class" />
    )
    
    expect(container.firstChild).toHaveClass('custom-class')
  })
})

describe('PRISMACompact', () => {
  it('displays all counts in compact format', () => {
    const counts = createMockPrismaCounts({
      records_identified: 500,
      records_screened: 450,
      reports_assessed: 70,
      studies_included: 12,
    })
    renderWithProviders(<PRISMACompact counts={counts} />)
    
    expect(screen.getByText('500')).toBeInTheDocument()
    expect(screen.getByText('450')).toBeInTheDocument()
    expect(screen.getByText('70')).toBeInTheDocument()
    expect(screen.getByText('12')).toBeInTheDocument()
  })

  it('highlights studies included with bold style', () => {
    const counts = createMockPrismaCounts({ studies_included: 15 })
    renderWithProviders(<PRISMACompact counts={counts} />)
    
    const includedElement = screen.getByText('15')
    expect(includedElement).toHaveClass('font-semibold')
  })
})
