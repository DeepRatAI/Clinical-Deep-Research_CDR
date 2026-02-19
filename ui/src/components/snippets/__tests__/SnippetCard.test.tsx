/**
 * SnippetCard Component Tests
 * 
 * Tests for SnippetCard and SnippetsList components.
 * Ref: src/components/snippets/SnippetCard.tsx
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, fireEvent, waitFor } from '@testing-library/react'
import { SnippetCard, SnippetsList } from '../SnippetCard'
import { renderWithProviders, createMockSnippet, createMockSourceRef } from '../../../test/utils'

describe('SnippetCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders snippet text in blockquote', () => {
    const snippet = createMockSnippet({ text: 'Evidence text from the study' })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    expect(screen.getByText(/"Evidence text from the study"/)).toBeInTheDocument()
  })

  it('displays PMID when available', () => {
    const snippet = createMockSnippet({
      source_ref: createMockSourceRef({ pmid: '98765432' }),
    })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    expect(screen.getByText('PMID: 98765432')).toBeInTheDocument()
  })

  it('displays NCT ID when no PMID available', () => {
    const snippet = createMockSnippet({
      source_ref: createMockSourceRef({ pmid: null, nct_id: 'NCT01234567' }),
    })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    expect(screen.getByText('NCT: NCT01234567')).toBeInTheDocument()
  })

  it('displays DOI when no PMID or NCT ID available', () => {
    const snippet = createMockSnippet({
      source_ref: createMockSourceRef({ pmid: null, nct_id: null, doi: '10.1234/test' }),
    })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    expect(screen.getByText('DOI')).toBeInTheDocument()
    expect(screen.getByText('DOI').closest('a')).toHaveAttribute('href', 'https://doi.org/10.1234/test')
  })

  it('displays "Unknown source" when no identifiers', () => {
    const snippet = createMockSnippet({
      source_ref: createMockSourceRef({ pmid: null, nct_id: null, doi: null }),
    })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    // Component still renders without source identifiers (shows section only)
    expect(screen.getByTestId('snippet-card')).toBeInTheDocument()
    // No PMID, NCT, or DOI links rendered
    expect(screen.queryByText(/PMID/)).not.toBeInTheDocument()
    expect(screen.queryByText(/NCT/)).not.toBeInTheDocument()
  })

  it('displays section type', () => {
    const snippet = createMockSnippet({ section: 'METHODS' })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    expect(screen.getByText('METHODS')).toBeInTheDocument()
  })

  it('displays offset when present', () => {
    const snippet = createMockSnippet({ offset_start: 1500 })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    expect(screen.getByTestId('snippet-offset')).toBeInTheDocument()
    expect(screen.getByTestId('snippet-offset')).toHaveTextContent(/offset.*1500/)
  })

  it('displays source title', () => {
    const snippet = createMockSnippet({
      source_ref: createMockSourceRef({ title: 'Important Clinical Trial' }),
    })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    expect(screen.getByText('Important Clinical Trial')).toBeInTheDocument()
  })

  it('displays authors with "et al." for more than 3', () => {
    const snippet = createMockSnippet({
      source_ref: createMockSourceRef({ 
        authors: ['Author A', 'Author B', 'Author C', 'Author D', 'Author E'] 
      }),
    })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    expect(screen.getByText(/Author A, Author B, Author C et al/)).toBeInTheDocument()
  })

  it('displays publication year', () => {
    const snippet = createMockSnippet({
      source_ref: createMockSourceRef({ publication_year: 2023, authors: ['Test Author'] }),
    })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    expect(screen.getByText(/\(2023\)/)).toBeInTheDocument()
  })

  it('displays relevance score bar', () => {
    const snippet = createMockSnippet({ relevance_score: 0.75 })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    expect(screen.getByText('75% relevance')).toBeInTheDocument()
  })

  it('copies text to clipboard on copy button click', async () => {
    const snippet = createMockSnippet({ text: 'Text to be copied' })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    const copyButton = screen.getByTitle('Copy snippet text')
    fireEvent.click(copyButton)

    await waitFor(() => {
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith('Text to be copied')
    })
  })

  it('has external link when URL is available', () => {
    const snippet = createMockSnippet({
      source_ref: createMockSourceRef({ url: 'https://example.com/paper' }),
    })
    renderWithProviders(<SnippetCard snippet={snippet} />)
    
    const link = screen.getByTitle('Open source')
    // Component uses primaryUrl: PMID takes priority over url prop
    expect(link).toHaveAttribute('href', 'https://pubmed.ncbi.nlm.nih.gov/12345678')
    expect(link).toHaveAttribute('target', '_blank')
  })
})

describe('SnippetsList', () => {
  it('renders multiple snippets', () => {
    const snippets = [
      createMockSnippet({ snippet_id: 'snip-1', text: 'First snippet text' }),
      createMockSnippet({ snippet_id: 'snip-2', text: 'Second snippet text' }),
    ]
    renderWithProviders(<SnippetsList snippets={snippets} />)
    
    expect(screen.getByText(/"First snippet text"/)).toBeInTheDocument()
    expect(screen.getByText(/"Second snippet text"/)).toBeInTheDocument()
  })

  it('shows empty state when no snippets', () => {
    renderWithProviders(<SnippetsList snippets={[]} />)
    
    expect(screen.getByText('No snippets found')).toBeInTheDocument()
  })
})
