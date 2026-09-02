'use client'

import { use } from 'react'
import Link from 'next/link'
import { errorMessage } from '@/api/client'
import { useProfile } from '@/hooks/useSetup'
import { ProfileEditor } from '@/components/ProfileEditor'
import { Empty, ErrorBox } from '@/components/ui'

export default function ProfilePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const profile = useProfile(id)
  if (profile.isLoading) return <Empty>Loading…</Empty>
  if (profile.error || !profile.data) return <ErrorBox message={errorMessage(profile.error, 'Profile not found')} />
  const p = profile.data
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Link href="/setup/profiles" className="text-sm text-neutral-500 hover:underline">← profiles</Link>
        <h2 className="text-lg font-semibold">{p.name}</h2>
        {p.builtin && <span className="rounded bg-neutral-200 px-2 py-0.5 text-xs">built-in · read-only (clone to edit)</span>}
      </div>
      {p.builtin ? (
        <pre className="overflow-x-auto rounded border border-neutral-200 bg-white p-3 text-xs">{JSON.stringify(p, null, 2)}</pre>
      ) : (
        <ProfileEditor key={p.updated_at} profile={p} mode="edit" />
      )}
    </div>
  )
}
