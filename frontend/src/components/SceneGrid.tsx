'use client'

import type { JobPublic, ScenePublic, VerdictOut } from '@/api/types'
import { MediaThumb } from '@/components/MediaThumb'

function profileSizes(job: JobPublic): { kf: [number, number] | null; clip: [number, number] | null } {
  const p = job.profile as { keyframe?: { width?: number; height?: number }; video?: { width?: number; height?: number } | null } | null
  const kf = p?.keyframe?.width && p.keyframe.height ? ([p.keyframe.width, p.keyframe.height] as [number, number]) : null
  const clip = p?.video?.width && p.video.height ? ([p.video.width, p.video.height] as [number, number]) : kf
  return { kf, clip }
}

export function VerdictChip({ v }: { v: VerdictOut | null | undefined }) {
  if (!v) return null
  const ok = v.decision === 'accept'
  return (
    <span className={`rounded px-1.5 py-0.5 text-xs ${ok ? 'bg-green-100 text-green-800' : 'bg-amber-100 text-amber-900'}`} title={v.fix_suggestion || undefined}>
      {v.decision} · {v.score}/10{v.needs_review ? ' · review' : ''}{v.framing_ok === false ? ' · framing' : ''}
    </span>
  )
}

export function SceneCard({ job, scene, children }: { job: JobPublic; scene: ScenePublic; children?: React.ReactNode }) {
  const { kf, clip } = profileSizes(job)
  const stills = job.kind === 'stills'
  return (
    <article className="flex flex-col gap-2 rounded-lg border border-neutral-200 bg-white p-3">
      <header className="flex items-center justify-between">
        <span className="font-semibold">Scene {scene.index + 1}</span>
        <span className="text-xs text-neutral-500">{scene.status}</span>
      </header>
      {(scene.setting || scene.mood) && <p className="text-xs text-neutral-500">{[scene.setting, scene.mood].filter(Boolean).join(' · ')}</p>}
      {scene.description && <p className="text-sm">{scene.description}</p>}

      {scene.variants.length > 1 ? (
        <div className="grid grid-cols-2 gap-2">
          {scene.variants.map((v, i) => (
            <div key={i} className={`rounded p-1 ${v.path === scene.keyframe_path ? 'ring-2 ring-neutral-900' : ''}`}>
              <MediaThumb url={v.swapped_url ?? v.url} width={kf?.[0]} height={kf?.[1]} alt={`variant ${i + 1}`} />
              <div className="mt-1 flex items-center justify-between text-xs text-neutral-500">
                <span>v{i + 1}{v.seed != null ? ` · seed ${v.seed}` : ''}</span>
                <VerdictChip v={v.verdict} />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <MediaThumb url={scene.swapped_url ?? scene.keyframe_url} width={kf?.[0]} height={kf?.[1]} alt="keyframe" />
      )}
      {!stills && <MediaThumb url={scene.video_url} video width={clip?.[0]} height={clip?.[1]} />}

      <div className="flex flex-wrap items-center gap-2">
        <VerdictChip v={scene.keyframe_verdict} />
        {scene.mode === 'edit' && <span className="rounded bg-violet-100 px-1.5 py-0.5 text-xs text-violet-800">edit · {scene.references.length} ref</span>}
        {scene.post_paths.length > 0 && <span className="rounded bg-neutral-100 px-1.5 py-0.5 text-xs">{scene.post_paths.length} post-step(s)</span>}
      </div>
      {children}
    </article>
  )
}

export function ScenePrompts({ scene, video }: { scene: ScenePublic; video: boolean }) {
  return (
    <div className="space-y-1 text-xs text-neutral-600">
      <p><span className="font-medium text-neutral-800">keyframe:</span> {scene.keyframe_prompt}</p>
      {scene.keyframe_negative && <p><span className="font-medium text-neutral-800">negative:</span> {scene.keyframe_negative}</p>}
      {video && <p><span className="font-medium text-neutral-800">animation:</span> {scene.animation_prompt}</p>}
    </div>
  )
}
