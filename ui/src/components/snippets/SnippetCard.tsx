import type { Snippet } from '../../types'
import { FileText, ExternalLink, Copy, Check } from 'lucide-react'
import { useState } from 'react'

interface SnippetCardProps {
  snippet: Snippet
  onSourceClick?: () => void
}

/**
 * Build PubMed URL from PMID
 */
function buildPubMedUrl(pmid: string): string {
  return `https://pubmed.ncbi.nlm.nih.gov/${pmid}`
}

/**
 * Build DOI URL
 */
function buildDoiUrl(doi: string): string {
  // Remove any "doi:" prefix if present
  const cleanDoi = doi.replace(/^doi:/i, '').trim()
  return `https://doi.org/${cleanDoi}`
}

export function SnippetCard({ snippet, onSourceClick: _onSourceClick }: SnippetCardProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(snippet.text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Build source label and URL
  const pmid = snippet.source_ref.pmid
  const doi = snippet.source_ref.doi
  const nctId = snippet.source_ref.nct_id

  // Determine primary source URL (PubMed preferred, then DOI, then URL)
  const primaryUrl = pmid
    ? buildPubMedUrl(pmid)
    : doi
    ? buildDoiUrl(doi)
    : snippet.source_ref.url

  return (
    <div className="snippet-card" data-testid="snippet-card" data-snippet-id={snippet.snippet_id}>
      {/* Source header with attribution */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 text-sm flex-wrap">
          <FileText className="w-4 h-4 text-primary-500" />
          
          {/* PMID Link */}
          {pmid && (
            <a
              href={buildPubMedUrl(pmid)}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-primary-600 hover:text-primary-800 hover:underline"
              data-pmid={pmid}
            >
              PMID: {pmid}
            </a>
          )}
          
          {/* DOI Link */}
          {doi && (
            <>
              {pmid && <span className="text-gray-400">|</span>}
              <a
                href={buildDoiUrl(doi)}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-primary-600 hover:text-primary-800 hover:underline"
                data-doi={doi}
              >
                DOI
              </a>
            </>
          )}
          
          {/* NCT ID */}
          {nctId && (
            <>
              {(pmid || doi) && <span className="text-gray-400">|</span>}
              <a
                href={`https://clinicaltrials.gov/study/${nctId}`}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-primary-600 hover:text-primary-800 hover:underline"
                data-nctid={nctId}
              >
                NCT: {nctId}
              </a>
            </>
          )}
          
          {/* Section */}
          <span className="text-gray-400">|</span>
          <span className="text-gray-500" data-testid="snippet-section">{snippet.section}</span>
          
          {/* Offset badge/tooltip */}
          {snippet.offset_start !== null && snippet.offset_end !== null && (
            <>
              <span className="text-gray-400">|</span>
              <span 
                className="text-xs bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded"
                title={`Character offset in source document: ${snippet.offset_start}–${snippet.offset_end}`}
                data-testid="snippet-offset"
              >
                offset {snippet.offset_start}–{snippet.offset_end}
              </span>
            </>
          )}
          {snippet.offset_start !== null && snippet.offset_end === null && (
            <>
              <span className="text-gray-400">|</span>
              <span 
                className="text-xs bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded"
                title={`Character offset in source document: ${snippet.offset_start}`}
                data-testid="snippet-offset"
              >
                offset {snippet.offset_start}
              </span>
            </>
          )}
        </div>
        
        <div className="flex items-center gap-1">
          <button
            onClick={handleCopy}
            className="btn-ghost p-1 rounded"
            title="Copy snippet text"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-500" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
          {primaryUrl && (
            <a
              href={primaryUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-ghost p-1 rounded"
              title="Open source"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
          )}
        </div>
      </div>

      {/* Snippet text */}
      <blockquote className="text-sm text-gray-800 italic leading-relaxed" data-testid="snippet-text">
        "{snippet.text}"
      </blockquote>

      {/* Source title */}
      <div className="mt-3 pt-2 border-t border-gray-200">
        <p className="text-xs text-gray-500 truncate">
          {snippet.source_ref.title}
        </p>
        {snippet.source_ref.authors.length > 0 && (
          <p className="text-xs text-gray-400 mt-1 truncate">
            {snippet.source_ref.authors.slice(0, 3).join(', ')}
            {snippet.source_ref.authors.length > 3 && ' et al.'}
            {snippet.source_ref.publication_year && (
              <> ({snippet.source_ref.publication_year})</>
            )}
          </p>
        )}
      </div>

      {/* Relevance score if available */}
      {snippet.relevance_score !== null && (
        <div className="mt-2 flex items-center gap-2">
          <div className="flex-1 h-1 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-500 rounded-full"
              style={{ width: `${snippet.relevance_score * 100}%` }}
            />
          </div>
          <span className="text-xs text-gray-500">
            {(snippet.relevance_score * 100).toFixed(0)}% relevance
          </span>
        </div>
      )}
    </div>
  )
}

interface SnippetsListProps {
  snippets: Snippet[]
}

export function SnippetsList({ snippets }: SnippetsListProps) {
  if (snippets.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>No snippets found</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {snippets.map((snippet) => (
        <SnippetCard key={snippet.snippet_id} snippet={snippet} />
      ))}
    </div>
  )
}
