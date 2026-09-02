'use client'

import { useState } from 'react'
import { useProfiles } from '@/hooks/useSetup'
import { ProfileEditor } from '@/components/ProfileEditor'
import { Empty } from '@/components/ui'

export default function NewProfilePage() {
  const profiles = useProfiles()
  const [baseId, setBaseId] = useState('cinematic_16x9')
  const base = profiles.data?.find((p) => p.id === baseId)
  return (
    <div className="space-y-4">
      <label className="block text-sm">
        <span className="text-neutral-600">Start from</span>
        <select value={baseId} onChange={(e) => setBaseId(e.target.value)} className="ml-2 rounded border border-neutral-300 bg-white px-2 py-1">
          {profiles.data?.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
      </label>
      {base ? <ProfileEditor key={base.id} profile={{ ...base, name: `${base.name} (copy)` }} mode="new" /> : <Empty>Loading…</Empty>}
    </div>
  )
}
