'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { errorMessage } from '@/api/client'
import { useCloneProfile, useDeleteProfile, useProfiles } from '@/hooks/useSetup'
import { profileSummary } from '@/components/NewJobWizard'
import { Button, Card, Empty, ErrorBox } from '@/components/ui'

export default function ProfilesPage() {
  const router = useRouter()
  const profiles = useProfiles()
  const clone = useCloneProfile()
  const del = useDeleteProfile()
  return (
    <Card title="Output profiles" actions={<Link href="/setup/profiles/new" className="rounded bg-neutral-900 px-3 py-1.5 text-sm font-medium text-white">New profile</Link>}>
      <ErrorBox message={profiles.error ? errorMessage(profiles.error) : clone.error ? errorMessage(clone.error) : del.error ? errorMessage(del.error) : null} />
      {profiles.isLoading && <Empty>Loading…</Empty>}
      <div className="grid gap-3 md:grid-cols-2">
        {profiles.data?.map((p) => (
          <div key={p.id} className="flex flex-col gap-2 rounded border border-neutral-200 p-3">
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="font-medium">{p.name} <span className="text-xs font-normal text-neutral-500">{p.id}</span></div>
                <div className="text-sm text-neutral-600">{profileSummary(p)} · {p.purpose}</div>
                {(p.style?.keyframe_prefix || (p.post_steps?.length ?? 0) > 0) && (
                  <div className="text-xs text-neutral-500">{p.style?.keyframe_prefix ? `style: ${p.style.keyframe_prefix}` : ''}{(p.post_steps?.length ?? 0) > 0 ? ` · post: ${p.post_steps!.map((s) => s.workflow).join(' → ')}` : ''}</div>
                )}
              </div>
              <span className={`rounded px-2 py-0.5 text-xs ${p.builtin ? 'bg-neutral-200' : 'bg-blue-100 text-blue-800'}`}>{p.builtin ? 'built-in' : 'user'}</span>
            </div>
            <div className="flex gap-2">
              <Button onClick={() => { const name = prompt('Name for the copy', `${p.name} (copy)`); if (name) clone.mutate({ id: p.id, name }, { onSuccess: (n) => router.push(`/setup/profiles/${n.id}`) }) }}>Clone</Button>
              {!p.builtin && <Link href={`/setup/profiles/${p.id}`} className="rounded border border-neutral-300 px-3 py-1.5 text-sm font-medium">Edit</Link>}
              {!p.builtin && <Button variant="danger" onClick={() => { if (confirm(`Delete profile “${p.name}”?`)) del.mutate(p.id) }}>Delete</Button>}
              {p.builtin && <Link href={`/setup/profiles/${p.id}`} className="rounded px-3 py-1.5 text-sm text-neutral-600">View</Link>}
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}
