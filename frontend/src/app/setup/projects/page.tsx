'use client'

import Link from 'next/link'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { errorMessage } from '@/api/client'
import { useCreateProject, useProfiles, useProjects } from '@/hooks/useSetup'
import { Button, Card, Empty, ErrorBox, fmtTime } from '@/components/ui'

export default function ProjectsPage() {
  const router = useRouter()
  const projects = useProjects()
  const profiles = useProfiles()
  const create = useCreateProject()
  const [name, setName] = useState('')
  const [profileId, setProfileId] = useState('cinematic_16x9')
  const [exportRoot, setExportRoot] = useState('')
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card title="Projects" className="lg:col-span-2">
        <ErrorBox message={projects.error ? errorMessage(projects.error) : null} />
        {projects.isLoading && <Empty>Loading…</Empty>}
        {projects.data?.length === 0 && <Empty>No projects yet. A project groups jobs and holds the shared master face, references and export folder.</Empty>}
        {projects.data && projects.data.length > 0 && (
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-neutral-500"><tr><th className="py-1 pr-3">project</th><th className="py-1 pr-3">slug</th><th className="py-1 pr-3">default profile</th><th className="py-1 pr-3">assets</th><th className="py-1 pr-3">jobs</th><th className="py-1">updated</th></tr></thead>
            <tbody>
              {projects.data.map((p) => (
                <tr key={p.id} className="border-t border-neutral-100">
                  <td className="py-1 pr-3"><Link href={`/setup/projects/${p.id}`} className="font-medium hover:underline">{p.name}</Link></td>
                  <td className="py-1 pr-3 font-mono text-xs">{p.slug}</td>
                  <td className="py-1 pr-3">{p.default_profile_id}</td>
                  <td className="py-1 pr-3">{p.n_assets}</td>
                  <td className="py-1 pr-3">{p.n_jobs}</td>
                  <td className="py-1 text-neutral-500">{fmtTime(p.updated_at)}</td>
                </tr>
              ))}
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
          <Button variant="primary" disabled={!name.trim() || create.isPending} onClick={() => create.mutate({ name: name.trim(), default_profile_id: profileId, export_root: exportRoot }, { onSuccess: (p) => router.push(`/setup/projects/${p.id}`) })}>Create</Button>
        </div>
      </Card>
    </div>
  )
}
