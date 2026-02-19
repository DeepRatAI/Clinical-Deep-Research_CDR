import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getRunDetail, getRunClaims, getRunSnippets, queryKeys } from '../api'
import { CertaintyBadge } from '../components/common/Badges'
import { SnippetCard } from '../components/snippets/SnippetCard'
import {
  ArrowLeft,
  CheckCircle,
  AlertCircle,
  XCircle,
  HelpCircle,
  Loader2,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import { useState } from 'react'
import type { EvidenceClaim, Snippet } from '../types'

export default function ClaimDetail() {
  const { runId, claimId } = useParams<{ runId: string; claimId: string }>()
  const [gradeExpanded, setGradeExpanded] = useState(false)

  // Fetch run detail for basic info
  const { data: runData, isLoading: runLoading, error: runError } = useQuery({
    queryKey: queryKeys.runDetail(runId!),
    queryFn: () => getRunDetail(runId!),
    retry: false,
    enabled: !!runId,
    staleTime: 30_000,
  })

  // Fetch claims from API
  const { data: claimsData, isLoading: claimsLoading } = useQuery({
    queryKey: queryKeys.runClaims(runId!),
    queryFn: () => getRunClaims(runId!),
    enabled: !!runId,
  })

  // Fetch snippets from API  
  const { data: snippetsData, isLoading: snippetsLoading } = useQuery({
    queryKey: queryKeys.runSnippets(runId!),
    queryFn: () => getRunSnippets(runId!),
    enabled: !!runId,
  })

  const isLoading = runLoading || claimsLoading || snippetsLoading

  // Use claims directly from API (already typed as EvidenceClaim[])
  const claims: EvidenceClaim[] = claimsData ?? []

  // Use snippets directly from API (already typed as Snippet[])
  const snippets: Snippet[] = snippetsData ?? []

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
      </div>
    )
  }

  // Find the specific claim
  const claim = claims.find((c) => c.claim_id === claimId)

  if (runError || !runData || !claim) {
    return (
      <div className="p-6">
        <div className="text-center py-12">
          <AlertCircle className="w-12 h-12 mx-auto text-red-400 mb-4" />
          <h2 className="text-lg font-medium text-gray-900">Claim not found</h2>
          <Link to={`/runs/${runId}`} className="btn btn-primary mt-4">
            Back to Run
          </Link>
        </div>
      </div>
    )
  }

  // Get supporting snippets
  const supportingSnippets = snippets.filter((s) =>
    claim.supporting_snippet_ids.includes(s.snippet_id)
  )

  const verificationIcons = {
    VERIFIED: { icon: CheckCircle, color: 'text-green-500', label: 'Verified' },
    PARTIAL: { icon: AlertCircle, color: 'text-yellow-500', label: 'Partially Verified' },
    REFUTED: { icon: XCircle, color: 'text-red-500', label: 'Refuted' },
    UNVERIFIABLE: { icon: HelpCircle, color: 'text-gray-400', label: 'Unverifiable' },
  }

  const verification = claim.verification_status
    ? verificationIcons[claim.verification_status]
    : null

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-gray-500 mb-6">
        <Link to="/" className="hover:text-gray-700">
          Dashboard
        </Link>
        <span>/</span>
        <Link to={`/runs/${runId}`} className="hover:text-gray-700">
          Run {runId?.slice(0, 8)}...
        </Link>
        <span>/</span>
        <span className="text-gray-900 font-medium">Claim {claimId}</span>
      </nav>

      {/* Back button */}
      <Link
        to={`/runs/${runId}`}
        className="inline-flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 mb-6"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Run
      </Link>

      {/* Claim header */}
      <div className="card mb-6">
        <div className="p-6">
          <div className="flex items-start justify-between gap-4 mb-4">
            <div className="flex items-center gap-3">
              <CertaintyBadge certainty={claim.certainty} />
              {verification && (
                <div className="flex items-center gap-1">
                  <verification.icon className={`w-5 h-5 ${verification.color}`} />
                  <span className="text-sm text-gray-600">
                    {verification.label}
                  </span>
                </div>
              )}
            </div>
            {claim.verification_score !== null && (
              <div className="flex items-center gap-2">
                <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green-500 rounded-full"
                    style={{ width: `${claim.verification_score * 100}%` }}
                  />
                </div>
                <span className="text-sm font-medium text-gray-900">
                  {(claim.verification_score * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>

          <p className="text-lg text-gray-900 leading-relaxed">{claim.claim_text}</p>

          <p className="mt-4 text-sm text-gray-500 font-mono">{claim.claim_id}</p>
        </div>
      </div>

      {/* GRADE Rationale */}
      {claim.grade_rationale && (
        <div className="card mb-6">
          <button
            onClick={() => setGradeExpanded(!gradeExpanded)}
            className="w-full card-header flex items-center justify-between cursor-pointer"
          >
            <h3 className="text-sm font-semibold text-gray-900">
              GRADE Rationale
            </h3>
            {gradeExpanded ? (
              <ChevronUp className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            )}
          </button>
          {gradeExpanded && (
            <div className="card-body space-y-3">
              {Object.entries(claim.grade_rationale).map(([key, value]) => (
                <div key={key}>
                  <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                    {key.replace(/_/g, ' ')}
                  </p>
                  <p className="text-sm text-gray-900">{value || '—'}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Supporting Snippets */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-gray-900 mb-4">
          Supporting Snippets ({supportingSnippets.length})
        </h3>
        {supportingSnippets.length > 0 ? (
          <div className="space-y-4">
            {supportingSnippets.map((snippet) => (
              <SnippetCard key={snippet.snippet_id} snippet={snippet} />
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500 card">
            <HelpCircle className="w-8 h-8 mx-auto mb-2 text-gray-300" />
            <p>No supporting snippets linked</p>
          </div>
        )}
      </div>

      {/* Missing snippets warning */}
      {claim.supporting_snippet_ids.length > supportingSnippets.length && (
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-700">
            <AlertCircle className="w-4 h-4 inline mr-1" />
            {claim.supporting_snippet_ids.length - supportingSnippets.length} linked
            snippet(s) could not be found
          </p>
        </div>
      )}
    </div>
  )
}
