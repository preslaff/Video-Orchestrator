'use client'

import { errorMessage } from '@/api/client'
import type { JobPublic } from '@/api/types'
import { useRetry, useStopJob } from '@/hooks/useJobs'
import { Button, ErrorBox } from '@/components/ui'

/**
 * Run control for a job: **Stop** a running segment (it ends at the next safe point,
 * status `cancelled` = "stopped"), **Retry** a failed one, **Resume** a stopped or an
 * orphaned one (persisted as running but not active in this server — a restart mid-run).
 * Resuming keeps finished work. ``onStarted`` reopens the event stream.
 */
export function RunControls({ job, onStarted }: { job: JobPublic; onStarted?: () => void }) {
  const retry = useRetry(job.id)
  const stop = useStopJob(job.id)
  const orphaned = job.status === 'running' && !job.active
  const resume = (confirmText?: string) => {
    if (confirmText && !confirm(confirmText)) return
    retry.mutate(undefined, { onSuccess: () => onStarted?.() })
  }
  return (
    <>
      {job.status === 'running' && job.active && (
        <Button disabled={stop.isPending || job.stopping} data-testid="stop-job"
          title="stop at the next safe point: finished work is kept; the ComfyUI step in progress is withdrawn, a Gemma call finishes first"
          onClick={() => stop.mutate()}>
          {job.stopping || stop.isPending ? 'Stopping…' : 'Stop'}
        </Button>
      )}
      {job.status === 'failed' && (
        <Button variant="primary" disabled={retry.isPending} title="resume from the persisted state: finished scenes are kept, the failed step runs again"
          onClick={() => resume()}>
          {retry.isPending ? 'Resuming…' : 'Retry'}
        </Button>
      )}
      {(job.status === 'cancelled' || orphaned) && (
        <Button variant="primary" disabled={retry.isPending} data-testid="resume-job"
          title="continue from the saved state: finished work is kept"
          onClick={() => resume(orphaned
            ? 'This job was running when the server stopped, and nothing is running it now. Resume it from its saved state?\n\nIf it is still running in a `vorch run` terminal, cancel.'
            : undefined)}>
          {retry.isPending ? 'Resuming…' : 'Resume'}
        </Button>
      )}
      {(retry.error || stop.error) && (
        <div className="basis-full"><ErrorBox message={errorMessage(retry.error ?? stop.error)} /></div>
      )}
    </>
  )
}
