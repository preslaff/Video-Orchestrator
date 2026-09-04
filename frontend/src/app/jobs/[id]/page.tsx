'use client'

import { use } from 'react'
import Link from 'next/link'
import { errorMessage } from '@/api/client'
import { useDecisions, useExportStatus, useJob, useProvenance, useRunExport } from '@/hooks/useJobs'
import { useJobEvents } from '@/hooks/useJobEvents'
import { useProject } from '@/hooks/useSetup'
import { Button, Card, Collapsible, Empty, ErrorBox, StatusBadge, fmtTime } from '@/components/ui'
import { Timeline } from '@/components/Timeline'
import { DecisionLog } from '@/components/DecisionLog'
import { SceneCard, ScenePrompts } from '@/components/SceneGrid'
import { GatePanel } from '@/components/GatePanel'
import { ProgressPanel } from '@/components/ProgressPanel'

export default function JobPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const job = useJob(id)
  const decisions = useDecisions(id)
  const provenance = useProvenance(id)
  const exp = useExportStatus(id)
  const project = useProject(job.data?.project_id ?? undefined)
  const runExport = useRunExport(id)
  // Always subscribe while the page is open: a finished job costs one synthetic
  // `end`, and a later approve/regenerate reconnects (onStarted) or resumes (active).
  const live = useJobEvents(id, {
    enabled: !!job.data,
    active: job.data?.status === 'running' || job.data?.status === 'pending',
  })

  if (job.isLoading) return <Empty>Loading…</Empty>
  if (job.error || !job.data) return <ErrorBox message={errorMessage(job.error, 'Job not found')} />
  const j = job.data
  const profile = j.profile as { aspect?: string } | null

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/jobs" className="text-sm text-neutral-500 hover:underline">← jobs</Link>
        <StatusBadge status={j.status} />
        <h1 className="text-xl font-semibold">{j.idea}</h1>
        {j.project_id ? (
          <Link href={`/setup/projects/${j.project_id}`} className="rounded bg-neutral-100 px-2 py-0.5 text-xs text-neutral-700 hover:bg-neutral-200" title="project">
            📁 {project.data?.name ?? 'project'}
          </Link>
        ) : (
          <span className="rounded bg-neutral-50 px-2 py-0.5 text-xs text-neutral-400">no project</span>
        )}
        <span className="text-xs text-neutral-500">{j.kind} · {j.profile_id ?? 'default profile'} · {profile?.aspect ?? '16:9'} · seed {j.seed || 'random'}</span>
        {j.pending_gate && <span className="rounded bg-amber-100 px-2 py-0.5 text-xs text-amber-900">waiting for approval: {j.pending_gate}</span>}
      </div>

      <ProgressPanel job={j} events={live.events} connected={live.connected} />

      {(j.status === 'needs_approval' || j.status === 'done' || j.status === 'failed') && j.scenes.length > 0 && (
        <GatePanel key={`${j.id}-${j.status}-${j.updated_at}`} job={j} decisions={decisions.data} onStarted={live.reconnect} />
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Timeline" className="lg:col-span-1">
          <Timeline events={live.events} decisions={decisions.data} connected={live.connected} ended={live.ended} />
        </Card>
        <Card title="Export (handoff folder for Shotcut)" className="lg:col-span-2" actions={
          <Button disabled={runExport.isPending || j.status === 'running'} onClick={() => runExport.mutate(j.status !== 'done')} title={j.status !== 'done' ? 'job not done: exports what exists so far' : undefined}>
            {runExport.isPending ? 'Exporting…' : exp.data?.exported ? 'Re-export' : 'Export now'}
          </Button>
        }>
          <ErrorBox message={runExport.error ? errorMessage(runExport.error) : null} />
          {exp.data?.exported ? (
            <div className="space-y-1 text-sm">
              <p><span className="font-medium">folder:</span> <code className="rounded bg-neutral-100 px-1">{String(exp.data.path)}</code>{exp.data.stale ? <span className="ml-2 text-amber-700">(stale — job changed since)</span> : null}</p>
              <ul className="list-inside list-disc text-neutral-700">{(exp.data.files as string[]).map((f) => <li key={f}>{f}</li>)}</ul>
              {(exp.data.missing as string[])?.length > 0 && <p className="text-amber-700">missing: {(exp.data.missing as string[]).join(', ')}</p>}
            </div>
          ) : (
            <Empty>Not exported yet — happens automatically when the job finishes.</Empty>
          )}
        </Card>
      </div>

      <Card title={`Scenes (${j.scenes.length})`}>
        {j.scenes.length === 0 && <Empty>No storyboard yet.</Empty>}
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {j.scenes.map((s) => (
            <SceneCard key={s.index} job={j} scene={s}>
              <ScenePrompts scene={s} video={j.kind === 'video'} />
            </SceneCard>
          ))}
        </div>
      </Card>

      <div className="space-y-3">
        <Collapsible title="Decision log" count={decisions.data?.length ?? 0} hint="why each frame was accepted, retried or flagged · click to expand">
          <DecisionLog decisions={decisions.data} />
        </Collapsible>
        <Collapsible title="Provenance" count={provenance.data?.length ?? 0} hint="seed + prompt per generated file · click to expand">
          {!provenance.data?.length ? <Empty>No artifacts yet.</Empty> : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase text-neutral-500"><tr><th className="py-1 pr-3">scene</th><th className="py-1 pr-3">kind</th><th className="py-1 pr-3">seed</th><th className="py-1 pr-3">file</th><th className="py-1">prompt</th></tr></thead>
                <tbody>
                  {provenance.data.map((r, i) => (
                    <tr key={i} className="border-t border-neutral-100 align-top">
                      <td className="py-1 pr-3">{r.scene_index + 1}</td>
                      <td className="py-1 pr-3">{r.kind}</td>
                      <td className="py-1 pr-3 font-mono text-xs">{r.seed ?? (r.has_manifest ? '' : 'no manifest')}</td>
                      <td className="py-1 pr-3">{r.url ? <a href={r.url} target="_blank" rel="noreferrer" className="hover:underline">{r.filename}</a> : r.filename}</td>
                      <td className="py-1 text-xs text-neutral-600">{r.prompt}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Collapsible>
      </div>
      <p className="text-xs text-neutral-400">created {fmtTime(j.created_at)} · updated {fmtTime(j.updated_at)}</p>
    </div>
  )
}
