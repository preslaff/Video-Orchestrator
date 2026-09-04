'use client'

import { useState } from 'react'
import { errorMessage } from '@/api/client'
import type { AssemblyVersion } from '@/api/types'
import { useDeleteAssemblyVersion, useSelectAssemblyVersion } from '@/hooks/useJobs'
import { MediaThumb } from '@/components/MediaThumb'
import { Button, ErrorBox, fmtTime } from '@/components/ui'
import { fmtBytes, fmtSeconds, fmtSpeed } from '@/lib/assembly'

export function describeSpeeds(spec: unknown): string {
  const s = spec as { items?: { scene_index: number; target_seconds?: number | null; speed?: number }[]; smooth?: boolean } | undefined
  const parts = (s?.items ?? []).filter((i) => i.target_seconds != null).map((i) => `s${String(i.scene_index + 1).padStart(2, '0')} ${fmtSpeed(i.speed ?? 1)}`)
  if (parts.length && s?.smooth) parts.push('smooth')
  return parts.join(' · ')
}

export function describeCuts(spec: unknown): string {
  const cuts = (spec as { cuts?: { transition: string; duration: number }[] } | undefined)?.cuts ?? []
  if (cuts.length === 0) return 'single clip'
  return cuts.map((c) => (c.transition === 'none' ? 'cut' : `${c.transition} ${c.duration}s`)).join(' · ')
}

/** All kept assemblies of a job: play any, make one current (= exported), delete. */
export function AssemblyVersions({ jobId, versions, size, disabled }: { jobId: string; versions: AssemblyVersion[]; size: [number, number] | null; disabled?: boolean }) {
  const select = useSelectAssemblyVersion(jobId)
  const del = useDeleteAssemblyVersion(jobId)
  const [playing, setPlaying] = useState<number | null>(null)
  if (versions.length === 0) return null
  const busy = disabled || select.isPending || del.isPending
  const shown = playing != null ? versions.find((v) => v.version === playing) : null

  return (
    <div className="space-y-2">
      <div className="text-xs font-semibold uppercase tracking-wide text-neutral-500">Versions ({versions.length})</div>
      <ErrorBox message={select.error ? errorMessage(select.error) : del.error ? errorMessage(del.error) : null} />
      <ul className="divide-y divide-neutral-100 rounded border border-neutral-200 bg-white text-sm">
        {[...versions].reverse().map((v) => (
          <li key={v.version} data-testid={`version-${v.version}`} className={`flex flex-wrap items-center gap-3 px-3 py-2 ${v.current ? 'bg-neutral-50' : ''}`}>
            <span className="w-8 font-mono font-semibold">v{v.version}</span>
            <span className="flex-1 min-w-[12rem]">
              <span className="text-neutral-800">{describeCuts(v.spec)}</span>
              {describeSpeeds(v.spec) && <span className="ml-2 rounded bg-blue-50 px-1.5 py-0.5 text-xs text-blue-900">{describeSpeeds(v.spec)}</span>}
              <span className="ml-2 text-xs text-neutral-500">{v.n_clips} clip(s) · {fmtSeconds(v.duration)} · {fmtBytes(v.size_bytes)} · {fmtTime(v.created_at)}</span>
              {!v.exists && <span className="ml-2 text-xs text-amber-700">file missing</span>}
            </span>
            {v.current ? (
              <span className="rounded bg-green-100 px-1.5 py-0.5 text-xs text-green-800" title="this one is assembly.mp4 in the export folder">current · exported</span>
            ) : (
              <Button disabled={busy || !v.exists} onClick={() => select.mutate(v.version)} title="make this the assembly.mp4 that the export folder ships">Use for export</Button>
            )}
            <Button variant="ghost" disabled={!v.exists} onClick={() => setPlaying(playing === v.version ? null : v.version)}>{playing === v.version ? 'Hide' : 'Play'}</Button>
            <Button variant="danger" disabled={busy} onClick={() => { if (confirm(`Delete assembly v${v.version}?`)) del.mutate(v.version, { onSuccess: () => setPlaying(null) }) }}>Delete</Button>
          </li>
        ))}
      </ul>
      {shown && (
        <div className="max-w-md">
          <MediaThumb key={shown.url ?? shown.version} url={shown.url} video width={shown.width ?? size?.[0]} height={shown.height ?? size?.[1]} />
        </div>
      )}
    </div>
  )
}
