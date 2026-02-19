import type { PRISMACounts } from '../../types'
import { ArrowRight, FileText, Filter, CheckCircle, XCircle } from 'lucide-react'

interface PRISMAFlowProps {
  counts: PRISMACounts
  className?: string
}

export function PRISMAFlow({ counts, className = '' }: PRISMAFlowProps) {
  const steps = [
    {
      label: 'Identified',
      count: counts.records_identified,
      icon: FileText,
      description: 'Records from databases',
    },
    {
      label: 'Screened',
      count: counts.records_screened,
      icon: Filter,
      description: 'Title/abstract screening',
    },
    {
      label: 'Assessed',
      count: counts.reports_assessed,
      icon: FileText,
      description: 'Full-text eligibility',
    },
    {
      label: 'Included',
      count: counts.studies_included,
      icon: CheckCircle,
      description: 'Studies in synthesis',
    },
  ]

  return (
    <div className={`card ${className}`}>
      <div className="card-header">
        <h3 className="text-sm font-semibold text-gray-900">PRISMA Flow</h3>
      </div>
      <div className="card-body">
        {/* Flow diagram */}
        <div className="flex items-center justify-between">
          {steps.map((step, index) => (
            <div key={step.label} className="flex items-center">
              {/* Step box */}
              <div className="prisma-step prisma-step-complete">
                <step.icon className="w-5 h-5 text-gray-600 mb-1" />
                <span className="text-2xl font-bold text-gray-900">
                  {step.count}
                </span>
                <span className="text-xs text-gray-500">{step.label}</span>
              </div>
              
              {/* Arrow between steps */}
              {index < steps.length - 1 && (
                <ArrowRight className="prisma-arrow w-6 h-6" />
              )}
            </div>
          ))}
        </div>

        {/* Exclusion details */}
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <XCircle className="w-4 h-4 text-gray-400" />
              <span className="text-gray-500">Excluded (screening):</span>
              <span className="font-medium text-gray-900">
                {counts.records_excluded_screening}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <XCircle className="w-4 h-4 text-gray-400" />
              <span className="text-gray-500">Not retrieved:</span>
              <span className="font-medium text-gray-900">
                {counts.reports_not_retrieved}
              </span>
            </div>
          </div>

          {/* Exclusion reasons */}
          {Object.keys(counts.exclusion_reasons).length > 0 && (
            <div className="mt-3">
              <p className="text-xs text-gray-500 mb-2">Exclusion reasons:</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(counts.exclusion_reasons).map(([reason, count]) => (
                  <span
                    key={reason}
                    className="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 rounded text-xs"
                  >
                    {reason}: <strong>{count}</strong>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

interface PRISMACompactProps {
  counts: PRISMACounts
}

export function PRISMACompact({ counts }: PRISMACompactProps) {
  return (
    <div className="flex items-center gap-1 text-sm text-gray-600">
      <span>{counts.records_identified}</span>
      <ArrowRight className="w-3 h-3" />
      <span>{counts.records_screened}</span>
      <ArrowRight className="w-3 h-3" />
      <span>{counts.reports_assessed}</span>
      <ArrowRight className="w-3 h-3" />
      <span className="font-semibold text-gray-900">
        {counts.studies_included}
      </span>
    </div>
  )
}
