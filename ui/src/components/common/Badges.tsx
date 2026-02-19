import type { RunStatus, CertaintyLevel, RoB2Judgment } from '../../types'

interface StatusBadgeProps {
  status: RunStatus
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const variants: Record<RunStatus, string> = {
    PENDING: 'badge-gray',
    RUNNING: 'badge-info',
    COMPLETED: 'badge-success',
    FAILED: 'badge-error',
    INSUFFICIENT_EVIDENCE: 'badge-warning',
  }

  const labels: Record<RunStatus, string> = {
    PENDING: 'Pending',
    RUNNING: 'Running',
    COMPLETED: 'Completed',
    FAILED: 'Failed',
    INSUFFICIENT_EVIDENCE: 'Insufficient Evidence',
  }

  return (
    <span className={`badge ${variants[status]}`}>
      {labels[status]}
    </span>
  )
}

interface CertaintyBadgeProps {
  certainty: CertaintyLevel
}

export function CertaintyBadge({ certainty }: CertaintyBadgeProps) {
  const variants: Record<CertaintyLevel, string> = {
    HIGH: 'badge-success',
    MODERATE: 'badge-warning',
    LOW: 'badge-error',
    VERY_LOW: 'badge-gray',
  }

  return (
    <span className={`badge ${variants[certainty]}`}>
      {certainty.replace('_', ' ')}
    </span>
  )
}

interface RoBBadgeProps {
  judgment: RoB2Judgment
}

export function RoBBadge({ judgment }: RoBBadgeProps) {
  const variants: Record<RoB2Judgment, string> = {
    LOW: 'badge-success',
    SOME_CONCERNS: 'badge-warning',
    HIGH: 'badge-error',
  }

  const labels: Record<RoB2Judgment, string> = {
    LOW: 'Low Risk',
    SOME_CONCERNS: 'Some Concerns',
    HIGH: 'High Risk',
  }

  return (
    <span className={`badge ${variants[judgment]}`}>
      {labels[judgment]}
    </span>
  )
}

interface DoDLevelBadgeProps {
  level: number
}

export function DoDLevelBadge({ level }: DoDLevelBadgeProps) {
  const labels: Record<number, string> = {
    0: 'DoD 0 - Error',
    1: 'DoD 1 - Basic',
    2: 'DoD 2 - Full',
    3: 'DoD 3 - Research',
  }

  const variants: Record<number, string> = {
    0: 'badge-error',
    1: 'badge-gray',
    2: 'badge-info',
    3: 'badge-success',
  }

  return (
    <span className={`badge ${variants[level] || 'badge-gray'}`}>
      {labels[level] || `DoD ${level}`}
    </span>
  )
}
