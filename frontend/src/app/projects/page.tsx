'use client'

import Link from 'next/link'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { errorMessage } from '@/api/client'
import { useCreateProject, useProfiles, useProjects } from '@/hooks/useSetup'
import { useJobs } from '@/hooks/useJobs'
import { Button, Card, Empty, ErrorBox, fmtTime } from '@/components/ui'

/** Projects are the unit of daily work: a client/campaign with its master face, references,
 * default profile and jobs. Jobs without a project show up as a virtual "Unassigned" row. */
export default function ProjectsPage() {
  const router = useRouter()
  const projects = useProjects()
  const profiles = useProfiles()
  const unassigned = useJobs({ unassigned: true, limit: 1 })
  const create = useCreateProject()
  const [name, setName] = useState('')
  const [profileId, setProfileId] = useState('cinematic_16x9')
  const [exportRoot, setExportRoot] = useState('')
  const profileName = (id: string) => profiles.data?.find((p) => p.id === id)?.name ?? id
  const nUnassigned = unassigned.data?.total ?? 0
  const cell = 'py-2 pr-3'
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card title="Projects" className="lg:col-span-2">
        <ErrorBox message={projects.error ? errorMessage(projects.error) : null} />
        {projects.isLoading && <Empty>Loading…</Empty>}
        {projects.data?.length === 0 && nUnassigned === 0 && <Empty>No projects yet. A project groups jobs and holds the shared master face, references and export folder.</Empty>}
        {projects.data && (projects.data.length > 0 || nUnassigned > 0) && (
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-neutral-500"><tr><th className={cell}>project</th><th className={cell}>default profile</th><th className={cell}>assets</th><th className={cell}>jobs</th><th className="py-2">last activity</th></tr></thead>
            <tbody>
              {projects.data.map((p) => (
                <tr key={p.id} className="border-t border-neutral-100 hover:bg-neutral-50">
                  <td className={cell}>
                    <Link href={`/projects/${p.id}`} className="font-medium hover:underline">{p.name}</Link>
                    <span className="ml-2 font-mono text-xs text-neutral-400">{p.slug}</span>
                  </td>
                  <td className={cell} title={p.default_profile_id}>{profileName(p.default_profile_id)}</td>
                  <td className={cell}>{p.n_assets}</td>
                  <td className={cell}>{p.n_jobs > 0 ? <Link href={`/jobs?project_id=${p.id}`} className="hover:underline">{p.n_jobs}</Link> : <span className="text-neutral-400">0</span>}</td>
                  <td className="py-2 text-neutral-500">{fmtTime(p.last_job_at ?? p.updated_at)}</td>
                </tr>
              ))}
              {nUnassigned > 0 && (
                <tr className="border-t border-neutral-100 text-neutral-600">
                  <td className={cell}><Link href="/jobs?unassigned=1" className="font-medium hover:underline">Unassigned</Link><span className="ml-2 text-xs text-neutral-400">jobs without a project</span></td>
                  <td className={cell}>—</td>
                  <td className={cell}>—</td>
                  <td className={cell}><Link href="/jobs?unassigned=1" className="hover:underline">{nUnassigned}</Link></td>
                  <td className="py-2 text-neutral-500">{fmtTime(unassigned.data?.items[0]?.updated_at)}</td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </Card>
      <Card title="New project">
        <ErrorBox message={create.error ? errorMessage(create.error) : null} />
        <div className="space-y-2 text-sm">
          <label className="block"><span className="text-xs text-neutral-500">name</span><input value={name} onChange={(e) => setName(e.target.value)} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" placeholder="Autumn campaign" /></label>
          <label className="block"><span className="text-xs text-neutral-500">default profile</span><select value={profileId} onChange={(e) => setProfileId(e.target.value)} className="mt-0.5 block w-full rounded border border-neutral-300 bg-white px-2 py-1">{profiles.data?.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select></label>
          <label className="block"><span className="text-xs text-neutral-500">export root (empty = default)</span><input value={exportRoot} onChange={(e) => setExportRoot(e.target.value)} className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1 font-mono text-xs" placeholder="/mnt/deliveries" /></label>
          <Button variant="primary" disabled={!name.trim() || create.isPending} onClick={() => create.mutate({ name: name.trim(), default_profile_id: profileId, export_root: exportRoot }, { onSuccess: (p) => router.push(`/projects/${p.id}`) })}>Create</Button>
        </div>
        <p className="mt-3 text-xs text-neutral-500">Global settings (environment, workflows, profiles) live under <Link href="/setup" className="underline">Setup</Link>.</p>
      </Card>
    </div>
  )
}
