'use client'

import type { ScenePublic } from '@/api/types'

type Variant = ScenePublic['variants'][number]
type Clip = ScenePublic['clips'][number]

/** One entry of a strip, normalised so keyframes and clips render the same way. */
type Entry = {
  id: string
  url: string | null | undefined
  posterUrl: string | null | undefined   // clips have no still of their own: use the frame they came from
  active: boolean
  batch: number
  label: string
  title: string
  score: number | null
  user: boolean
}

function variantEntry(v: Variant): Entry {
  return {
    id: v.id, url: v.swapped_url ?? v.url, posterUrl: v.swapped_url ?? v.url, active: v.active, batch: v.batch,
    label: v.source === 'user' ? 'yours' : v.seed != null ? `seed ${v.seed}` : 'keyframe',
    title: [v.source === 'user' ? 'your image' : `seed ${v.seed ?? '—'}`, v.created_at].filter(Boolean).join(' · '),
    score: v.verdict?.score ?? null,
    user: v.source === 'user',
  }
}

function clipEntry(c: Clip, frameOf: (path: string | null | undefined) => string | null | undefined): Entry {
  return {
    id: c.id, url: c.url, posterUrl: frameOf(c.keyframe_path), active: c.active, batch: c.batch,
    label: c.seed != null ? `seed ${c.seed}` : 'clip',
    title: [`clip · seed ${c.seed ?? '—'}`, c.created_at].filter(Boolean).join(' · '),
    score: c.verdict?.score ?? null,
    user: false,
  }
}

/**
 * Every version a scene has of one kind of media, newest first, with the active one ringed.
 *
 * Clicking selects — a pointer change, applied at once, never a regeneration. ✕ drops the
 * version from the job; the file stays on disk. A clip's thumbnail is the keyframe it was
 * animated from, since a video has no still of its own to show at this size.
 */
export function VersionStrip({
  scene, kind, disabled = false, onSelect, onDelete,
}: {
  scene: ScenePublic
  kind: 'keyframe' | 'clip'
  disabled?: boolean
  onSelect: (id: string) => void
  onDelete: (id: string, label: string) => void
}) {
  const variants = scene.variants ?? []
  const frameOf = (path: string | null | undefined) => variants.find((v) => v.path === path)?.url ?? null
  const entries = (kind === 'keyframe'
    ? variants.map(variantEntry)
    : (scene.clips ?? []).map((c) => clipEntry(c, frameOf))
  ).slice().reverse()   // newest first
  if (entries.length <= 1) return null   // nothing to choose between

  return (
    <div className="flex gap-1 overflow-x-auto pb-1" data-testid={`versions-${kind}-${scene.index}`}>
      {entries.map((e, i) => (
        <div key={e.id} className="relative shrink-0">
          {i > 0 && entries[i - 1].batch !== e.batch && (
            // where one generation run ends and the previous begins
            <span className="absolute -left-0.5 top-0 h-full w-px bg-neutral-300" aria-hidden />
          )}
          <button type="button" disabled={disabled || e.active} aria-pressed={e.active} title={e.title}
            data-testid={`version-${kind}-${scene.index}-${e.id}`} onClick={() => onSelect(e.id)}
            className={`block w-20 overflow-hidden rounded border text-left ${e.active ? 'border-neutral-900 ring-2 ring-neutral-900' : 'border-neutral-300 hover:border-neutral-500'}`}>
            {e.posterUrl ? (
              // eslint-disable-next-line @next/next/no-img-element -- local media served by FastAPI
              <img src={e.posterUrl} alt="" loading="lazy" className="aspect-square w-full bg-neutral-100 object-cover" />
            ) : (
              <span className="flex aspect-square w-full items-center justify-center bg-neutral-100 text-[10px] text-neutral-400">
                {kind === 'clip' ? 'clip' : 'no image'}
              </span>
            )}
            <span className="flex items-center justify-between gap-1 px-1 py-0.5 text-[10px] text-neutral-600">
              <span className="truncate">{e.user ? '★ ' : ''}{e.label}</span>
              {e.score != null && <span className="shrink-0 text-neutral-500">{e.score}/10</span>}
            </span>
          </button>
          <button type="button" disabled={disabled} onClick={() => onDelete(e.id, e.label)}
            data-testid={`version-delete-${kind}-${scene.index}-${e.id}`}
            title="remove this version from the job (the file stays on disk)"
            className="absolute right-0.5 top-0.5 rounded bg-white/80 px-1 text-[10px] text-neutral-600 hover:bg-white hover:text-red-700">
            ✕
          </button>
        </div>
      ))}
    </div>
  )
}
