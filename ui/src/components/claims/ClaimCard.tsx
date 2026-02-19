import { Link } from 'react-router-dom'
import type { EvidenceClaim } from '../../types'
import { CertaintyBadge } from '../common/Badges'
import { CheckCircle, AlertCircle, XCircle, HelpCircle, ChevronRight } from 'lucide-react'

interface ClaimCardProps {
  claim: EvidenceClaim
  runId: string
}

const verificationConfig = {
  VERIFIED: { 
    icon: CheckCircle, 
    color: 'text-green-500', 
    bgColor: 'bg-green-50',
    label: 'Verified' 
  },
  PARTIAL: { 
    icon: AlertCircle, 
    color: 'text-yellow-500', 
    bgColor: 'bg-yellow-50',
    label: 'Partial' 
  },
  REFUTED: { 
    icon: XCircle, 
    color: 'text-red-500', 
    bgColor: 'bg-red-50',
    label: 'Refuted' 
  },
  UNVERIFIABLE: { 
    icon: HelpCircle, 
    color: 'text-gray-400', 
    bgColor: 'bg-gray-50',
    label: 'Unverifiable' 
  },
}

export function ClaimCard({ claim, runId }: ClaimCardProps) {
  const verification = claim.verification_status 
    ? verificationConfig[claim.verification_status] 
    : null

  return (
    <Link
      to={`/runs/${runId}/claims/${claim.claim_id}`}
      className="block card hover:shadow-md transition-shadow"
      data-testid="claim-card"
      data-claim-id={claim.claim_id}
    >
      <div className="p-4">
        <div className="flex items-start justify-between gap-4">
          {/* Claim content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              {/* Certainty badge (GRADE) */}
              <CertaintyBadge certainty={claim.certainty} />
              
              {/* Verification status badge */}
              {verification && (
                <span 
                  className={`verification-badge inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${verification.bgColor} ${verification.color}`}
                  data-testid="verified"
                  data-verification-status={claim.verification_status}
                >
                  <verification.icon className="w-3.5 h-3.5" />
                  {verification.label}
                </span>
              )}
              
              {/* Verification score / coverage */}
              {claim.verification_score !== null && (
                <span 
                  className="text-xs text-gray-500"
                  data-testid="verification-coverage"
                >
                  coverage {claim.verification_score.toFixed(2)}
                </span>
              )}
            </div>
            
            <p className="text-sm text-gray-900 line-clamp-2" data-testid="claim-text">
              {claim.claim_text}
            </p>

            <div className="mt-2 flex items-center gap-2 text-xs text-gray-500">
              <span>
                {claim.supporting_snippet_ids.length} supporting snippet(s)
              </span>
              <span>•</span>
              <span className="font-mono">{claim.claim_id}</span>
            </div>
          </div>

          {/* Arrow */}
          <ChevronRight className="w-5 h-5 text-gray-400 flex-shrink-0" />
        </div>
      </div>
    </Link>
  )
}

interface ClaimsListProps {
  claims: EvidenceClaim[]
  runId: string
}

export function ClaimsList({ claims, runId }: ClaimsListProps) {
  if (claims.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <HelpCircle className="w-12 h-12 mx-auto mb-4 text-gray-300" />
        <p>No claims extracted</p>
        <p className="text-sm mt-1">
          The evidence may be insufficient to make claims
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-3" data-testid="claims-list">
      {claims.map((claim) => (
        <ClaimCard key={claim.claim_id} claim={claim} runId={runId} />
      ))}
    </div>
  )
}
