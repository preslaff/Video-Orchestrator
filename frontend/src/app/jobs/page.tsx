'use client'

import Link from 'next/link'
import { useState } from 'react'
import { Trash2 } from 'lucide-react'
import { errorMessage } from '@/api/client'
import type { JobStatus } from '@/api/types'
import { useDeleteJob, useJobs } from '@/hooks/useJobs'
import { useGlobalEvents } from '@/hooks/useGlobalEvents'
import { Button, Card, Empty, ErrorBox, StatusBadge, fmtTime } from '@/components/ui'

const STATUSES: JobStatus[] = ['pending', 'running', 'needs_approval', 'done', 'failed', 'cancelled']
const PAGE = 25

export default function JobsPage() {
  const [status, setStatus] = useState<JobStatus | ''>('')
  const [offset, setOffset] = useState(0)
  const jobs = useJobs({ status: status || undefined, limit: PAGE, offset })
  const del = useDeleteJob()
  const live = useGlobalEvents()
  const total = jobs.data?.total ?? 0

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Jobs</h1>
        <div className="flex items-center gap-3 text-sm">
          <span className="text-xs text-neutral-500">{live.connected ? '● live' : '○ offline'}</span>
          <select value={status} onChange={(e) => { setStatus(e.target.value as JobStatus | ''); setOffset(0) }} className="rounded border border-neutral-300 bg-white px-2 py-1">
            <option value="">all statuses</option>
            {STATUSES.map((s) => <option key={s} value={s}>{s.replace('_', ' ')}</option>)}
          </select>
          <Link href="/jobs/new" className="rounded bg-neutral-900 px-3 py-1.5 text-sm font-medium text-white">New job</Link>
        </div>
      </div>
      <ErrorBox message={jobs.error ? errorMessage(jobs.error) : del.error ? errorMessage(del.error) : null} />
      <Card>
        {jobs.isLoading && <Empty>Loading…</Empty>}
        {jobs.data && jobs.data.items.length === 0 && <Empty>No jobs yet. Create one from “New job”.</Empty>}
        {jobs.data && jobs.data.items.length > 0 && (
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-neutral-500">
              <tr><th className="py-1 pr-3">status</th><th className="py-1 pr-3">idea</th><th className="py-1 pr-3">kind</th><th className="py-1 pr-3">profile</th><th className="py-1 pr-3">updated</th><th /></tr>
            </thead>
            <tbody>
              {jobs.data.items.map((j) => (
                <tr key={j.id} className="border-t border-neutral-100 hover:bg-neutral-50">
                  <td className="py-2 pr-3"><StatusBadge status={j.status} /></td>
                  <td className="py-2 pr-3"><Link href={`/jobs/${j.id}`} className="font-medium hover:underline">{j.idea || j.id}</Link></td>
                  <td className="py-2 pr-3 text-neutral-600">{j.kind}</td>
                  <td className="py-2 pr-3 text-neutral-600">{j.profile_id ?? '—'}</td>
                  <td className="py-2 pr-3 whitespace-nowrap text-neutral-500">{fmtTime(j.updated_at)}</td>
                  <td className="py-2 text-right">
                    <Button variant="ghost" title="delete job" disabled={j.status === 'running'} onClick={() => { if (confirm(`Delete job “${j.idea}”? Files on disk are kept.`)) del.mutate(j.id) }}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {total > PAGE && (
          <div className="mt-3 flex items-center justify-between text-sm text-neutral-600">
            <span>{offset + 1}–{Math.min(offset + PAGE, total)} of {total}</span>
            <div className="flex gap-2">
              <Button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</Button>
              <Button disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>Next</Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}
