'use client'

import Link from 'next/link'
import { Suspense, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Trash2 } from 'lucide-react'
import { errorMessage } from '@/api/client'
import type { JobStatus } from '@/api/types'
import { useDeleteJob, useJobs } from '@/hooks/useJobs'
import { useProjects } from '@/hooks/useSetup'
import { useGlobalEvents } from '@/hooks/useGlobalEvents'
import { Button, Card, Empty, ErrorBox, JobIdChip, StatusBadge, fmtTime } from '@/components/ui'

const STATUSES: JobStatus[] = ['pending', 'running', 'needs_approval', 'done', 'failed', 'stopped', 'cancelled']
const PAGE = 25

const UNASSIGNED = '__unassigned'

export default function JobsPage() {
  // useSearchParams needs a Suspense boundary for the static prerender
  return <Suspense fallback={<Empty>Loading…</Empty>}><JobsList /></Suspense>
}

/** Filters live in the URL (`?project_id=…`, `?unassigned=1`, `?status=…`) so project pages and
 * the Unassigned row can deep-link into the list; paging stays local. */
function JobsList() {
  const router = useRouter()
  const sp = useSearchParams()
  const status = (sp.get('status') ?? '') as JobStatus | ''
  const projectId = sp.get('project_id') ?? (sp.get('unassigned') ? UNASSIGNED : '')
  const [offset, setOffset] = useState(0)
  const setFilters = (next: { status?: string; project?: string }) => {
    const q = new URLSearchParams()
    const st = next.status ?? status
    const pr = next.project ?? projectId
    if (st) q.set('status', st)
    if (pr === UNASSIGNED) q.set('unassigned', '1')
    else if (pr) q.set('project_id', pr)
    setOffset(0)
    router.replace(q.size ? `/jobs?${q}` : '/jobs')
  }
  const jobs = useJobs({
    status: status || undefined,
    project_id: projectId && projectId !== UNASSIGNED ? projectId : undefined,
    unassigned: projectId === UNASSIGNED || undefined,
    limit: PAGE, offset,
  })
  const projects = useProjects()
  const projectName = (id: string | null | undefined) => projects.data?.find((p) => p.id === id)?.name
  const del = useDeleteJob()
  const live = useGlobalEvents()
  const total = jobs.data?.total ?? 0

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Jobs</h1>
        <div className="flex items-center gap-3 text-sm">
          <span className="text-xs text-neutral-500">{live.connected ? '● live' : '○ offline'}</span>
          <select value={projectId} onChange={(e) => setFilters({ project: e.target.value })} className="rounded border border-neutral-300 bg-white px-2 py-1">
            <option value="">all projects</option>
            {projects.data?.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            <option value={UNASSIGNED}>— unassigned —</option>
          </select>
          <select value={status} onChange={(e) => setFilters({ status: e.target.value })} className="rounded border border-neutral-300 bg-white px-2 py-1">
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
              <tr><th className="py-1 pr-3">status</th><th className="py-1 pr-3">id</th><th className="py-1 pr-3">idea</th><th className="py-1 pr-3">project</th><th className="py-1 pr-3">kind</th><th className="py-1 pr-3">profile</th><th className="py-1 pr-3">updated</th><th /></tr>
            </thead>
            <tbody>
              {jobs.data.items.map((j) => (
                <tr key={j.id} className="border-t border-neutral-100 hover:bg-neutral-50">
                  <td className="py-2 pr-3"><StatusBadge status={j.status} /></td>
                  <td className="py-2 pr-3"><JobIdChip id={j.id} /></td>
                  <td className="py-2 pr-3"><Link href={`/jobs/${j.id}`} className="font-medium hover:underline">{j.idea || j.id}</Link></td>
                  <td className="py-2 pr-3">
                    {j.project_id ? (
                      <Link href={`/projects/${j.project_id}`} className="rounded bg-neutral-100 px-1.5 py-0.5 text-xs text-neutral-700 hover:bg-neutral-200">
                        {projectName(j.project_id) ?? j.project_id.slice(0, 8)}
                      </Link>
                    ) : <span className="text-xs text-neutral-400">—</span>}
                  </td>
                  <td className="py-2 pr-3 text-neutral-600">{j.kind}</td>
                  <td className="py-2 pr-3 text-neutral-600" title={j.profile_id ?? undefined}>{j.profile_name ?? j.profile_id ?? '—'}</td>
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
