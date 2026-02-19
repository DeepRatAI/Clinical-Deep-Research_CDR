import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { listRuns, queryKeys } from '../api'
import { StatusBadge, DoDLevelBadge } from '../components/common/Badges'
import { PlusCircle, RefreshCw, AlertCircle, FileSearch } from 'lucide-react'
import type { RunSummary } from '../types'

export default function Dashboard() {
  // Fetch runs from real API with auto-refresh for active runs
  const { data: runs, isLoading, error, refetch } = useQuery<RunSummary[]>({
    queryKey: queryKeys.runs,
    queryFn: listRuns,
    staleTime: 5_000, // 5 seconds - shorter for responsive updates
    refetchOnWindowFocus: true,
    // FIX-U1: Auto-refetch every 3 seconds if any run is in progress
    refetchInterval: (query) => {
      const data = query.state.data as RunSummary[] | undefined
      const hasActiveRun = data?.some(
        run => run.status === 'PENDING' || run.status === 'RUNNING'
      )
      return hasActiveRun ? 3000 : false
    },
  })

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Research Runs</h1>
          <p className="text-sm text-gray-500 mt-1">
            Clinical evidence synthesis runs
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => refetch()}
            className="btn btn-secondary"
            disabled={isLoading}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <Link to="/runs/new" className="btn btn-primary">
            <PlusCircle className="w-4 h-4 mr-2" />
            New Run
          </Link>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-500" />
          <p className="text-sm text-red-700">
            Failed to load runs. Using demo data.
          </p>
        </div>
      )}

      {/* Runs table */}
      <div className="card">
        <div className="table-container">
          <table className="table">
            <thead className="table-header">
              <tr>
                <th className="table-header-cell">Run ID</th>
                <th className="table-header-cell">Status</th>
                <th className="table-header-cell">DoD Level</th>
                <th className="table-header-cell">Claims</th>
                <th className="table-header-cell">Verification</th>
                <th className="table-header-cell">Updated</th>
              </tr>
            </thead>
            <tbody className="table-body">
              {runs?.map((run) => (
                <tr key={run.run_id} className="hover:bg-gray-50">
                  <td className="table-cell">
                    <Link
                      to={`/runs/${run.run_id}`}
                      className="flex items-center gap-2 text-primary-600 hover:text-primary-800 font-mono"
                    >
                      <FileSearch className="w-4 h-4" />
                      {run.run_id.slice(0, 8)}...
                    </Link>
                  </td>
                  <td className="table-cell">
                    <StatusBadge status={run.status} />
                  </td>
                  <td className="table-cell">
                    <DoDLevelBadge level={run.dod_level} />
                  </td>
                  <td className="table-cell text-gray-900 font-medium">
                    {run.claims_count}
                  </td>
                  <td className="table-cell">
                    {run.verification_coverage > 0 ? (
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-green-500 rounded-full"
                            style={{ width: `${run.verification_coverage * 100}%` }}
                          />
                        </div>
                        <span className="text-sm text-gray-600">
                          {(run.verification_coverage * 100).toFixed(0)}%
                        </span>
                      </div>
                    ) : (
                      <span className="text-gray-400">—</span>
                    )}
                  </td>
                  <td className="table-cell text-gray-500">
                    {new Date(run.updated_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Empty state */}
        {(!runs || runs.length === 0) && !isLoading && (
          <div className="text-center py-12">
            <FileSearch className="w-12 h-12 mx-auto text-gray-300 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-1">
              No research runs yet
            </h3>
            <p className="text-sm text-gray-500 mb-4">
              Start your first evidence synthesis run
            </p>
            <Link to="/runs/new" className="btn btn-primary">
              <PlusCircle className="w-4 h-4 mr-2" />
              Create First Run
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}
