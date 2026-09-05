'use client'

import { use, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { errorMessage } from '@/api/client'
import { useDeleteAsset, useDeleteProject, useProfiles, useProject, useUpdateAssetLabel, useUpdateProject, useUploadAsset } from '@/hooks/useSetup'
import { useJobs } from '@/hooks/useJobs'
import { Button, Card, Empty, ErrorBox, JobIdChip, StatusBadge, fmtTime } from '@/components/ui'

const KINDS = ['master_face', 'reference', 'product', 'logo', 'other']

export default function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const router = useRouter()
  const project = useProject(id)
  const profiles = useProfiles()
  const jobs = useJobs({ project_id: id, limit: 50 })
  const update = useUpdateProject(id)
  const del = useDeleteProject()
  const upload = useUploadAsset(id)
  const delAsset = useDeleteAsset(id)
  const relabel = useUpdateAssetLabel(id)
  const [kind, setKind] = useState('master_face')
  const [label, setLabel] = useState('')
  const [file, setFile] = useState<File | null>(null)

  if (project.isLoading) return <Empty>Loading…</Empty>
  if (project.error || !project.data) return <ErrorBox message={errorMessage(project.error, 'Project not found')} />
  const p = project.data
  const inp = 'mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1 text-sm'

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/projects" className="text-sm text-neutral-500 hover:underline">← projects</Link>
        <h2 className="text-lg font-semibold">{p.name}</h2>
        <span className="font-mono text-xs text-neutral-500">{p.slug}</span>
        <Link href={`/jobs/new?project=${id}`} className="rounded bg-neutral-900 px-3 py-1.5 text-sm font-medium text-white">New job in this project</Link>
        <Button variant="danger" onClick={() => { if (confirm(`Delete project “${p.name}”? Jobs are kept (unlinked); asset files stay on disk.`)) del.mutate({ id }, { onSuccess: () => router.push('/projects') }) }}>Delete</Button>
      </div>
      <ErrorBox message={update.error ? errorMessage(update.error) : del.error ? errorMessage(del.error) : upload.error ? errorMessage(upload.error) : delAsset.error ? errorMessage(delAsset.error) : null} />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Settings">
          <form className="space-y-2" onSubmit={(e) => { e.preventDefault(); const f = new FormData(e.currentTarget); update.mutate({ name: String(f.get('name')), brand_notes: String(f.get('brand_notes')), style_notes: String(f.get('style_notes')), default_profile_id: String(f.get('default_profile_id')), export_root: String(f.get('export_root')) }) }}>
            <label className="block"><span className="text-xs text-neutral-500">name</span><input name="name" defaultValue={p.name} className={inp} /></label>
            <label className="block"><span className="text-xs text-neutral-500">default profile</span><select name="default_profile_id" defaultValue={p.default_profile_id} className={inp + ' bg-white'}>{profiles.data?.map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}</select></label>
            <label className="block"><span className="text-xs text-neutral-500">brand notes</span><textarea name="brand_notes" defaultValue={p.brand_notes} rows={2} className={inp} /></label>
            <label className="block"><span className="text-xs text-neutral-500">style notes</span><textarea name="style_notes" defaultValue={p.style_notes} rows={2} className={inp} /></label>
            <label className="block"><span className="text-xs text-neutral-500">export root (effective: {p.export_root_effective})</span><input name="export_root" defaultValue={p.export_root} className={inp + ' font-mono text-xs'} /></label>
            <div className="flex items-center gap-3"><Button type="submit" variant="primary" disabled={update.isPending}>Save</Button>{update.isSuccess && !update.isPending && <span className="text-sm text-green-700">Saved.</span>}</div>
          </form>
        </Card>
        <Card title={`Assets (${p.assets.length})`}>
          <form className="mb-3 grid gap-2 rounded border border-neutral-200 p-2 sm:grid-cols-[1fr_auto_auto_auto]" onSubmit={(e) => { e.preventDefault(); if (file) upload.mutate({ file, kind, label }, { onSuccess: () => { setFile(null); setLabel('') } }) }}>
            <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="text-sm" />
            <select value={kind} onChange={(e) => setKind(e.target.value)} className="rounded border border-neutral-300 bg-white px-2 py-1 text-sm">{KINDS.map((k) => <option key={k} value={k}>{k}</option>)}</select>
            <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="label" className="rounded border border-neutral-300 px-2 py-1 text-sm" />
            <Button type="submit" variant="primary" disabled={!file || upload.isPending}>{upload.isPending ? 'Uploading…' : 'Upload'}</Button>
          </form>
          {p.assets.length === 0 && <Empty>No assets. Add a <b>master_face</b> for identity, <b>reference</b>/<b>product</b> images for edit mode.</Empty>}
          {p.assets.some((a) => a.kind === 'master_face') && <p className="mb-2 text-xs text-neutral-500">Label faces by who and when (e.g. “Maria, age 8”) — the storyboard reads the labels to assign a face to each scene.</p>}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {p.assets.map((a) => (
              <div key={a.id} className="rounded border border-neutral-200 p-1">
                {/* eslint-disable-next-line @next/next/no-img-element -- served by FastAPI */}
                <img src={a.url} alt={a.name} className="aspect-square w-full rounded object-cover" />
                <div className="mt-1 flex items-center justify-between text-xs">
                  <span><span className="rounded bg-neutral-100 px-1">{a.kind}</span> <button type="button" className="hover:underline" title="edit label" onClick={() => { const v = prompt(`Label for “${a.name}”`, a.label || ''); if (v !== null) relabel.mutate({ assetId: a.id, label: v }) }}>{a.label || a.name}</button></span>
                  <button type="button" className="text-red-700 hover:underline" onClick={() => { if (confirm(`Remove asset “${a.name}”?`)) delAsset.mutate(a.id) }}>remove</button>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card title={`Jobs in this project (${jobs.data?.total ?? p.n_jobs})`} actions={jobs.data && jobs.data.total > 0 ? <Link href={`/jobs?project_id=${id}`} className="text-sm hover:underline">all jobs of this project →</Link> : null}>
        {jobs.data?.items.length === 0 && <Empty>No jobs yet — start one with “New job in this project”.</Empty>}
        {jobs.data && jobs.data.items.length > 0 && (
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-neutral-500"><tr><th className="py-1 pr-3">status</th><th className="py-1 pr-3">id</th><th className="py-1 pr-3">idea</th><th className="py-1 pr-3">kind</th><th className="py-1 pr-3">profile</th><th className="py-1">updated</th></tr></thead>
            <tbody>
              {jobs.data.items.map((j) => (
                <tr key={j.id} className="border-t border-neutral-100 hover:bg-neutral-50">
                  <td className="py-1 pr-3"><StatusBadge status={j.status} /></td>
                  <td className="py-1 pr-3"><JobIdChip id={j.id} /></td>
                  <td className="py-1 pr-3"><Link href={`/jobs/${j.id}`} className="font-medium hover:underline">{j.idea}</Link></td>
                  <td className="py-1 pr-3 text-neutral-600">{j.kind}</td>
                  <td className="py-1 pr-3 text-neutral-600" title={j.profile_id ?? undefined}>{j.profile_name ?? j.profile_id ?? '—'}</td>
                  <td className="py-1 whitespace-nowrap text-neutral-500">{fmtTime(j.updated_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  )
}
