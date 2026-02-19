import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { 
  getRunDetail, 
  getRunClaims, 
  getRunSnippets, 
  getRunHypotheses,
  getRunEvaluation,
  getReport,
  queryKeys,
} from '../api/client'
import { StatusBadge, DoDLevelBadge } from '../components/common/Badges'
import { PRISMAFlow } from '../components/prisma/PRISMAFlow'
import { ClaimsList } from '../components/claims/ClaimCard'
import { SnippetsList } from '../components/snippets/SnippetCard'
import {
  ArrowLeft,
  FileText,
  Search,
  ClipboardList,
  BookOpen,
  Lightbulb,
  Download,
  Loader2,
  AlertCircle,
  AlertTriangle,
  BarChart3,
  CheckCircle,
  TrendingUp,
} from 'lucide-react'
import { useState, useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import type { RunStatus, PRISMACounts } from '../types'

type Tab = 'overview' | 'prisma' | 'claims' | 'snippets' | 'hypotheses' | 'evaluation' | 'report'

/**
 * Derive initial tab from URL path
 */
function getTabFromPath(pathname: string): Tab {
  if (pathname.endsWith('/claims')) return 'claims'
  if (pathname.endsWith('/snippets')) return 'snippets'
  if (pathname.endsWith('/hypotheses')) return 'hypotheses'
  if (pathname.endsWith('/evaluation')) return 'evaluation'
  if (pathname.endsWith('/prisma')) return 'prisma'
  if (pathname.endsWith('/report')) return 'report'
  return 'overview'
}

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>()
  const location = useLocation()
  const [activeTab, setActiveTab] = useState<Tab>(() => getTabFromPath(location.pathname))

  // Sync tab with URL path changes
  useEffect(() => {
    setActiveTab(getTabFromPath(location.pathname))
  }, [location.pathname])

  // Fetch run detail with auto-polling for active runs
  const { 
    data: run, 
    isLoading: runLoading, 
    error: runError 
  } = useQuery({
    queryKey: queryKeys.runDetail(runId!),
    queryFn: () => getRunDetail(runId!),
    enabled: !!runId,
    retry: 1,
    staleTime: 5_000, // 5 seconds
    // FIX-U1: Auto-refetch every 2 seconds while run is in progress
    refetchInterval: (query) => {
      const data = query.state.data
      const isActive = data?.status === 'PENDING' || data?.status === 'RUNNING'
      return isActive ? 2000 : false
    },
  })

  // Fetch claims (only when needed)
  const { 
    data: claims = [], 
    isLoading: claimsLoading 
  } = useQuery({
    queryKey: queryKeys.runClaims(runId!),
    queryFn: () => getRunClaims(runId!),
    enabled: !!runId && (activeTab === 'claims' || activeTab === 'overview'),
    retry: 1,
  })

  // Fetch snippets (only when needed)
  const { 
    data: snippets = [], 
    isLoading: snippetsLoading 
  } = useQuery({
    queryKey: queryKeys.runSnippets(runId!),
    queryFn: () => getRunSnippets(runId!),
    enabled: !!runId && activeTab === 'snippets',
    retry: 1,
  })

  // Fetch hypotheses (only when needed)
  const { 
    data: hypotheses = [], 
    isLoading: hypothesesLoading 
  } = useQuery({
    queryKey: queryKeys.runHypotheses(runId!),
    queryFn: () => getRunHypotheses(runId!),
    enabled: !!runId && (activeTab === 'hypotheses' || activeTab === 'overview'),
    retry: 1,
  })

  // Fetch evaluation (only when needed)
  const { 
    data: evaluation, 
    isLoading: evaluationLoading 
  } = useQuery({
    queryKey: queryKeys.runEvaluation(runId!),
    queryFn: () => getRunEvaluation(runId!),
    enabled: !!runId && (activeTab === 'evaluation' || activeTab === 'overview'),
    retry: 1,
  })

  const tabs = [
    { id: 'overview' as Tab, label: 'Overview', icon: FileText },
    { id: 'prisma' as Tab, label: 'PRISMA', icon: ClipboardList },
    { id: 'claims' as Tab, label: 'Claims', icon: BookOpen, count: run?.claims_count },
    { id: 'snippets' as Tab, label: 'Snippets', icon: Search, count: run?.snippets_count },
    { id: 'hypotheses' as Tab, label: 'Hypotheses', icon: Lightbulb, count: run?.hypotheses_count },
    { id: 'evaluation' as Tab, label: 'Evaluation', icon: BarChart3 },
    { id: 'report' as Tab, label: 'Report', icon: Download },
  ]

  // Export state
  const [isExporting, setIsExporting] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)

  /**
   * Handle export - fetch report metadata and download
   */
  const handleExport = async (format: 'markdown' | 'json' | 'pdf' = 'json') => {
    if (!runId) return
    
    setIsExporting(true)
    setExportError(null)

    try {
      const report = await getReport(runId)
      
      if (!report.download_urls || !report.download_urls[format]) {
        setExportError(`Format "${format}" is not available for this report`)
        return
      }

      // Trigger download
      const url = report.download_urls[format]
      const link = document.createElement('a')
      link.href = url
      link.download = `cdr-report-${runId}.${format === 'markdown' ? 'md' : format}`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    } catch (err) {
      console.error('Export failed:', err)
      setExportError(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setIsExporting(false)
    }
  }

  if (runLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-primary-500" />
      </div>
    )
  }

  if (runError || !run) {
    return (
      <div className="p-6">
        <div className="text-center py-12">
          <AlertCircle className="w-12 h-12 mx-auto text-red-400 mb-4" />
          <h2 className="text-lg font-medium text-gray-900">Run not found</h2>
          <p className="text-sm text-gray-500 mt-1">
            {runError instanceof Error ? runError.message : 'The requested run could not be loaded'}
          </p>
          <Link to="/" className="btn btn-primary mt-4">
            Back to Dashboard
          </Link>
        </div>
      </div>
    )
  }

  // Transform API response to component-compatible format
  const transformedPrismaCounts: PRISMACounts | null = run.prisma_counts ? {
    records_identified: run.prisma_counts.records_identified,
    records_screened: run.prisma_counts.records_screened,
    records_excluded_screening: run.prisma_counts.records_excluded_screening,
    reports_assessed: run.prisma_counts.reports_assessed,
    reports_not_retrieved: run.prisma_counts.reports_not_retrieved,
    studies_included: run.prisma_counts.studies_included,
    exclusion_reasons: run.prisma_counts.exclusion_reasons,
  } : null

  // Claims and snippets are already typed correctly from API fetchers
  const transformedClaims = claims
  const transformedSnippets = snippets

  return (
    <div className="min-h-screen">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="p-4">
          {/* Back + Run ID */}
          <div className="flex items-center gap-4 mb-4">
            <Link
              to="/"
              className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </Link>
            <span className="text-sm text-gray-400">|</span>
            <span className="font-mono text-sm text-gray-500">{run.run_id}</span>
          </div>

          {/* Status + DoD */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <StatusBadge status={run.status as RunStatus} />
              <DoDLevelBadge level={run.dod_level} />
              {run.status_reason && (
                <span className="text-sm text-gray-500">{run.status_reason}</span>
              )}
            </div>
            <button 
              className="btn btn-secondary"
              onClick={() => handleExport('json')}
              disabled={isExporting || run.status !== 'COMPLETED'}
              title={run.status !== 'COMPLETED' ? 'Export available after run completes' : 'Export report'}
            >
              {isExporting ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Download className="w-4 h-4 mr-2" />
              )}
              Export
            </button>
          </div>

          {/* Export error display */}
          {exportError && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
              <span className="text-sm text-red-700">{exportError}</span>
              <button 
                onClick={() => setExportError(null)} 
                className="ml-auto text-red-500 hover:text-red-700"
              >
                ×
              </button>
            </div>
          )}
        </div>

        {/* Tabs */}
        <div className="px-4 flex gap-1 border-t border-gray-100">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
              {tab.count !== undefined && tab.count > 0 && (
                <span className="bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full text-xs">
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      <div className="p-6">
        {/* FIX-U2: Show errors prominently if run failed */}
        {run.status === 'FAILED' && run.errors && run.errors.length > 0 && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-sm font-medium text-red-800">Run Failed</h3>
                <ul className="mt-2 text-sm text-red-700 list-disc list-inside space-y-1">
                  {run.errors.map((error, i) => (
                    <li key={i}>{error}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Show progress indicator for running runs */}
        {(run.status === 'RUNNING' || run.status === 'PENDING') && (
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center gap-3">
              <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
              <div>
                <h3 className="text-sm font-medium text-blue-800">
                  {run.status === 'PENDING' ? 'Starting...' : 'Processing...'}
                </h3>
                <p className="text-sm text-blue-600 mt-1">
                  This may take a few minutes. The page will update automatically.
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* PICO summary */}
            {run.pico && (
              <div className="card">
                <div className="card-header">
                  <h3 className="text-sm font-semibold text-gray-900">PICO</h3>
                </div>
                <div className="card-body grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                      Population
                    </p>
                    <p className="text-sm text-gray-900">{run.pico.population}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                      Intervention
                    </p>
                    <p className="text-sm text-gray-900">{run.pico.intervention}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                      Comparator
                    </p>
                    <p className="text-sm text-gray-900">
                      {run.pico.comparator || '—'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                      Outcome
                    </p>
                    <p className="text-sm text-gray-900">{run.pico.outcome}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Quick stats */}
            <div className="grid grid-cols-4 gap-4">
              <div className="card p-4 text-center">
                <p className="text-3xl font-bold text-gray-900">
                  {run.prisma_counts?.studies_included || 0}
                </p>
                <p className="text-sm text-gray-500">Studies Included</p>
              </div>
              <div className="card p-4 text-center">
                <p className="text-3xl font-bold text-gray-900">
                  {run.claims_count}
                </p>
                <p className="text-sm text-gray-500">Claims</p>
              </div>
              <div className="card p-4 text-center">
                <p className="text-3xl font-bold text-gray-900">
                  {run.snippets_count}
                </p>
                <p className="text-sm text-gray-500">Snippets</p>
              </div>
              <div className="card p-4 text-center">
                <p className="text-3xl font-bold text-gray-900">
                  {run.hypotheses_count}
                </p>
                <p className="text-sm text-gray-500">Hypotheses</p>
              </div>
            </div>

            {/* Verification coverage */}
            {run.verification_coverage > 0 && (
              <div className="card p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">
                    Verification Coverage
                  </span>
                  <span className="text-sm font-bold text-gray-900">
                    {(run.verification_coverage * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green-500 rounded-full transition-all"
                    style={{ width: `${run.verification_coverage * 100}%` }}
                  />
                </div>
              </div>
            )}

            {/* PRISMA mini */}
            {transformedPrismaCounts && <PRISMAFlow counts={transformedPrismaCounts} />}

            {/* Recent claims */}
            {transformedClaims.length > 0 && (
              <div className="card">
                <div className="card-header flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-900">
                    Claims ({transformedClaims.length})
                  </h3>
                  <button
                    onClick={() => setActiveTab('claims')}
                    className="text-sm text-primary-600 hover:text-primary-700"
                  >
                    View all
                  </button>
                </div>
                <div className="p-4">
                  <ClaimsList claims={transformedClaims.slice(0, 3)} runId={run.run_id} />
                </div>
              </div>
            )}

            {/* Hypotheses preview */}
            {hypotheses.length > 0 && (
              <div className="card" data-testid="hypotheses">
                <div className="card-header flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-900">
                    Hypotheses ({hypotheses.length})
                  </h3>
                  <button
                    onClick={() => setActiveTab('hypotheses')}
                    className="text-sm text-primary-600 hover:text-primary-700"
                  >
                    View all
                  </button>
                </div>
                <div className="p-4 space-y-3">
                  {hypotheses.slice(0, 2).map((h) => (
                    <div key={h.hypothesis_id} className="border-b border-gray-100 pb-3 last:border-0 last:pb-0 hypothesis">
                      <p className="text-sm text-gray-900 mb-2">{h.hypothesis_text}</p>
                      {h.rival_hypotheses.length > 0 && (
                        <div className="text-xs text-gray-500" data-testid="rival-hypotheses">
                          <span className="font-medium text-yellow-600">Rival alternatives: </span>
                          {h.rival_hypotheses.length} alternative explanation(s)
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Evaluation summary */}
            {evaluation && (
              <div className="card" data-testid="evaluation">
                <div className="card-header flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-900">
                    Evaluation Summary
                  </h3>
                  <button
                    onClick={() => setActiveTab('evaluation')}
                    className="text-sm text-primary-600 hover:text-primary-700"
                  >
                    View details
                  </button>
                </div>
                <div className="p-4">
                  {/* Overall Grade */}
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-sm text-gray-700">Overall</span>
                    <div className="flex items-center gap-3">
                      <span className="text-2xl font-bold text-primary-600">{evaluation.overall_grade}</span>
                      <span className="text-sm text-gray-500">{(evaluation.overall_score * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  {/* Dimension names preview */}
                  <div className="space-y-2">
                    {evaluation.dimensions.slice(0, 3).map((dim) => (
                      <div key={dim.name} className="flex items-center justify-between text-sm">
                        <span className="text-gray-600" data-testid="dimension-name">{dim.name.replace(/_/g, ' ')}</span>
                        <span className="text-gray-500">{dim.grade}</span>
                      </div>
                    ))}
                  </div>
                  {/* Strengths preview */}
                  {evaluation.strengths.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-gray-100" data-testid="strengths">
                      <p className="text-xs text-gray-500 uppercase mb-2">Key Strengths</p>
                      <ul className="text-sm text-gray-700 space-y-1">
                        {evaluation.strengths.slice(0, 2).map((s, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
                            <span>{s}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {/* Recommendations preview */}
                  {evaluation.recommendations.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-gray-100" data-testid="recommendations">
                      <p className="text-xs text-gray-500 uppercase mb-2">Recommendations</p>
                      <ul className="text-sm text-gray-700 space-y-1">
                        {evaluation.recommendations.slice(0, 2).map((r, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <TrendingUp className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
                            <span>{r}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'prisma' && transformedPrismaCounts && (
          <div className="space-y-6">
            <PRISMAFlow counts={transformedPrismaCounts} />

            {/* Search strategy */}
            {run.search_plan && (
              <div className="card">
                <div className="card-header">
                  <h3 className="text-sm font-semibold text-gray-900">
                    Search Strategy (PRISMA-S)
                  </h3>
                </div>
                <div className="card-body space-y-4">
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                      PubMed Query
                    </p>
                    <pre className="text-xs bg-gray-50 p-3 rounded-lg overflow-x-auto font-mono">
                      {run.search_plan.pubmed_query}
                    </pre>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                      ClinicalTrials.gov Query
                    </p>
                    <pre className="text-xs bg-gray-50 p-3 rounded-lg overflow-x-auto font-mono">
                      {run.search_plan.ct_gov_query}
                    </pre>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                        Date Range
                      </p>
                      <p className="text-sm text-gray-900">
                        {run.search_plan.date_range
                          ? `${run.search_plan.date_range[0]} to ${run.search_plan.date_range[1]}`
                          : 'All dates'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                        Languages
                      </p>
                      <p className="text-sm text-gray-900">
                        {run.search_plan.languages.join(', ') || 'All'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
                        Executed
                      </p>
                      <p className="text-sm text-gray-900">
                        {run.search_plan.created_at 
                          ? new Date(run.search_plan.created_at).toLocaleString()
                          : '—'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'claims' && (
          <div>
            {claimsLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-6 h-6 animate-spin text-primary-500" />
              </div>
            ) : (
              <ClaimsList claims={transformedClaims} runId={run.run_id} />
            )}
          </div>
        )}

        {activeTab === 'snippets' && (
          <div>
            {snippetsLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-6 h-6 animate-spin text-primary-500" />
              </div>
            ) : (
              <SnippetsList snippets={transformedSnippets} />
            )}
          </div>
        )}

        {activeTab === 'hypotheses' && (
          <div data-testid="hypotheses">
            {hypothesesLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-6 h-6 animate-spin text-primary-500" />
              </div>
            ) : hypotheses.length > 0 ? (
              <div className="space-y-4">
                {hypotheses.map((h) => (
                  <div key={h.hypothesis_id} className="card p-4 hypothesis" data-testid="hypothesis-card">
                    <div className="flex items-start justify-between mb-3">
                      <span className="text-xs font-mono text-gray-500">
                        {h.hypothesis_id}
                      </span>
                      <span className="badge badge-info">
                        {(h.confidence * 100).toFixed(0)}% confidence
                      </span>
                    </div>
                    <p className="text-sm text-gray-900 mb-3" data-testid="hypothesis-text">
                      {h.hypothesis_text}
                    </p>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-xs text-gray-500 uppercase mb-1">Mechanism</p>
                        <p className="text-gray-700">{h.mechanism}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 uppercase mb-1">Test Design</p>
                        <p className="text-gray-700">{h.test_design || '—'}</p>
                      </div>
                    </div>
                    {/* MCID Value */}
                    {h.mcid && (
                      <div className="mt-3 pt-3 border-t border-gray-100">
                        <p className="text-xs text-gray-500 uppercase mb-1">MCID</p>
                        <p className="text-sm text-gray-700 mcid" data-testid="mcid">{h.mcid}</p>
                      </div>
                    )}
                    {/* Rival Hypotheses Section */}
                    {h.rival_hypotheses.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-100" data-testid="rival-hypotheses">
                        <p className="text-xs text-gray-500 uppercase mb-2 flex items-center gap-1">
                          <AlertTriangle className="w-3.5 h-3.5 text-yellow-500" />
                          <span>Rival Hypotheses (Alternatives)</span>
                        </p>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {h.rival_hypotheses.map((r, i) => (
                            <li key={i} className="flex items-start gap-2">
                              <span className="text-yellow-500 mt-0.5">•</span>
                              <span>{r}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {/* Threats to Validity */}
                    {h.threats_to_validity.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-100">
                        <p className="text-xs text-gray-500 uppercase mb-1">Threats to Validity</p>
                        <ul className="text-sm text-gray-600 list-disc list-inside">
                          {h.threats_to_validity.map((t, i) => (
                            <li key={i}>{t}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <Lightbulb className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p>No compositional hypotheses</p>
                <p className="text-sm mt-1">
                  Requires DoD Level 3 and sufficient evidence
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'evaluation' && (
          <div data-testid="evaluation" className="evaluation">
            {evaluationLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-6 h-6 animate-spin text-primary-500" />
              </div>
            ) : evaluation ? (
              <div className="space-y-6">
                {/* Overall Score Card */}
                <div className="card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">Overall Evaluation</h3>
                    <div className="flex items-center gap-4">
                      <div className="text-center">
                        <span className="text-3xl font-bold text-primary-600" data-testid="overall-grade">
                          {evaluation.overall_grade}
                        </span>
                        <p className="text-xs text-gray-500">Grade</p>
                      </div>
                      <div className="text-center">
                        <span className="text-3xl font-bold text-gray-900" data-testid="overall-score">
                          {(evaluation.overall_score * 100).toFixed(0)}%
                        </span>
                        <p className="text-xs text-gray-500">Score</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Dimensions Grid */}
                <div className="card">
                  <div className="card-header">
                    <h3 className="text-sm font-semibold text-gray-900">Evaluation Dimensions</h3>
                  </div>
                  <div className="p-4 space-y-4" data-testid="evaluation-dimensions">
                    {evaluation.dimensions.map((dim) => (
                      <div key={dim.name} className="border-b border-gray-100 pb-3 last:border-0 last:pb-0">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-gray-700" data-testid="dimension-name">
                            {dim.name.replace(/_/g, ' ')}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className="badge badge-info" data-testid="dimension-grade">{dim.grade}</span>
                            <span className="text-sm text-gray-500" data-testid="dimension-score">
                              {(dim.score * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                        {/* Score bar */}
                        <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary-500 rounded-full transition-all"
                            style={{ width: `${dim.score * 100}%` }}
                          />
                        </div>
                        {dim.rationale && (
                          <p className="text-xs text-gray-500 mt-1">{dim.rationale}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Strengths */}
                {evaluation.strengths.length > 0 && (
                  <div className="card">
                    <div className="card-header">
                      <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        Strengths
                      </h3>
                    </div>
                    <div className="p-4" data-testid="strengths">
                      <ul className="space-y-2">
                        {evaluation.strengths.map((s, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                            <span className="text-green-500 mt-0.5">✓</span>
                            <span>{s}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}

                {/* Weaknesses */}
                {evaluation.weaknesses.length > 0 && (
                  <div className="card">
                    <div className="card-header">
                      <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                        <AlertCircle className="w-4 h-4 text-yellow-500" />
                        Weaknesses
                      </h3>
                    </div>
                    <div className="p-4" data-testid="weaknesses">
                      <ul className="space-y-2">
                        {evaluation.weaknesses.map((w, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                            <span className="text-yellow-500 mt-0.5">!</span>
                            <span>{w}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                {evaluation.recommendations.length > 0 && (
                  <div className="card">
                    <div className="card-header">
                      <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-blue-500" />
                        Recommendations
                      </h3>
                    </div>
                    <div className="p-4" data-testid="recommendations">
                      <ul className="space-y-2">
                        {evaluation.recommendations.map((r, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                            <span className="text-blue-500 mt-0.5">→</span>
                            <span>{r}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p>No evaluation data available</p>
                <p className="text-sm mt-1">
                  Evaluation is generated after run completion
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'report' && (
          <div className="card">
            <div className="card-header">
              <h3 className="text-sm font-semibold text-gray-900">
                Download Report
              </h3>
            </div>
            <div className="card-body space-y-3">
              {run.status !== 'COMPLETED' ? (
                <div className="text-center py-8 text-gray-500">
                  <Download className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                  <p>Reports available after run completes</p>
                  <p className="text-sm mt-1">
                    Current status: {run.status}
                  </p>
                </div>
              ) : (
                <>
                  <button 
                    className="btn btn-secondary w-full justify-start"
                    onClick={() => handleExport('markdown')}
                    disabled={isExporting}
                  >
                    {isExporting ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Download className="w-4 h-4 mr-2" />
                    )}
                    Download Markdown Report
                  </button>
                  <button 
                    className="btn btn-secondary w-full justify-start"
                    onClick={() => handleExport('json')}
                    disabled={isExporting}
                  >
                    {isExporting ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Download className="w-4 h-4 mr-2" />
                    )}
                    Download JSON Bundle
                  </button>
                  <button 
                    className="btn btn-secondary w-full justify-start"
                    onClick={() => handleExport('pdf')}
                    disabled={isExporting}
                  >
                    {isExporting ? (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    ) : (
                      <Download className="w-4 h-4 mr-2" />
                    )}
                    Download PDF Report
                  </button>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
