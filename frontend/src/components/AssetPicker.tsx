'use client'

import { useMemo, useState } from 'react'
import type { AssetOut } from '@/api/types'
import { filterAssets, topTags } from '@/lib/assetFilter'
import { ZoomButton } from '@/components/Lightbox'
import type { LightboxItem } from '@/components/Lightbox'

/** Above this many assets the picker adds search, tag filters and paging. */
export const SEARCH_THRESHOLD = 12
const PAGE = 48

/** Thumbnails of project assets to pick from. Single choice (with a "none" tile) or an
 * ordered multi-selection (click order = priority, shown as a badge). With ``onUpload``
 * a last tile adds a file from the computer (the caller turns it into a project asset).
 *
 * Large sets stay usable: a search box over label, caption, tags and sign text, chips for
 * the most common tags, the current selection pinned on top (so a filter never hides a
 * choice), a bounded scroll area rendering 48 results at a time, and small lazy thumbnails. */
export function AssetPicker({
  assets, selected, onChange, multi = false, none, testId, disabled = false, onUpload, uploading = false,
}: {
  assets: AssetOut[]
  selected: string[]
  onChange: (ids: string[]) => void
  multi?: boolean
  none?: string            // label of the "none" tile (single mode)
  testId?: string
  disabled?: boolean
  onUpload?: (file: File) => void
  uploading?: boolean
}) {
  const [query, setQuery] = useState('')
  const [tags, setTags] = useState<string[]>([])
  const [limit, setLimit] = useState(PAGE)
  const large = assets.length > SEARCH_THRESHOLD
  const chips = useMemo(() => (large ? topTags(assets) : []), [assets, large])
  // Newest first: a frame just saved from another scene, or a just-uploaded photo, is what the
  // person is looking for — stored order would bury it on page two of eighty.
  const ordered = useMemo(
    () => (large ? [...assets].sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? '')) : assets),
    [assets, large],
  )
  const matches = useMemo(() => (large ? filterAssets(ordered, query, tags) : ordered), [ordered, large, query, tags])
  const filtered = !!query || tags.length > 0
  const pinned = large ? selected.map((id) => assets.find((a) => a.id === id)).filter((a): a is AssetOut => !!a) : []
  const results = large ? matches.filter((a) => !selected.includes(a.id)) : ordered
  const shown = large ? results.slice(0, limit) : results
  const sid = (suffix: string) => (testId ? `${testId}-${suffix}` : undefined)

  const toggle = (id: string) => {
    if (disabled) return
    if (!multi) return onChange(selected[0] === id ? [] : [id])
    onChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id])
  }
  const toggleTag = (t: string) => { setTags((ts) => (ts.includes(t) ? ts.filter((x) => x !== t) : [...ts, t])); setLimit(PAGE) }

  const tile = (a: AssetOut, i: number, list: AssetOut[]) => {
    const pos = selected.indexOf(a.id)
    const on = pos >= 0
    // The caption is what makes picking a reference for scene 23 out of seventy
    // thumbnails a decision rather than a guess.
    const caption = a.note_caption || ''
    // ⤢ opens the original without touching the selection; ← / → step through the same tiles.
    const group: LightboxItem[] = list.map((x) => ({ url: x.url, caption: [x.label || x.name, x.note_caption].filter(Boolean).join(' — ') }))
    // The wrapper (needed so ⤢ is not a button inside a button) is the grid item now: `min-w-0`
    // lets it shrink to its column despite the nowrap label, and the tile must fill it with
    // `block w-full` — a button sizes to its content otherwise, and the w-full image then falls
    // back to the thumbnail's natural 320 px.
    return (
      <div key={a.id} className="group relative min-w-0">
      <button type="button" disabled={disabled} aria-pressed={on} onClick={() => toggle(a.id)}
        title={[`${a.kind}: ${a.label || a.name}`, caption].filter(Boolean).join('\n')}
        data-testid={sid(a.id)}
        className={`relative block w-full overflow-hidden rounded border text-left ${on ? 'border-neutral-900 ring-2 ring-neutral-900' : 'border-neutral-300 hover:border-neutral-500'}`}>
        {/* eslint-disable-next-line @next/next/no-img-element -- served by FastAPI */}
        <img src={a.thumb_url || a.url} alt={a.label || a.name} loading="lazy" className="aspect-square w-full object-cover" />
        {multi && on && <span className="absolute left-1 top-1 rounded bg-neutral-900 px-1 text-xs text-white">{pos + 1}</span>}
        {a.label_source === 'auto' && (
          <span className="absolute right-1 top-1 rounded bg-amber-500 px-1 text-[10px] font-medium text-white" title="suggested by Gemma, not yet reviewed">AI</span>
        )}
        <span className="block truncate px-1 pt-0.5 text-[11px] text-neutral-700">{a.label || a.name}</span>
        {caption && <span className="block truncate px-1 pb-0.5 text-[10px] text-neutral-500">{caption}</span>}
      </button>
      <ZoomButton items={group} index={i} className="right-1 top-6" testId={`${sid(a.id)}-zoom`} />
      </div>
    )
  }
  const grid = 'grid grid-cols-3 gap-1 sm:grid-cols-4 lg:grid-cols-6'

  return (
    <div data-testid={testId}>
      {large && (
        <div className="mb-1 space-y-1">
          <div className="flex items-center gap-2">
            <input type="search" value={query} onChange={(e) => { setQuery(e.target.value); setLimit(PAGE) }}
              placeholder="search labels, captions, tags, sign text" aria-label="search assets" data-testid={sid('search')}
              className="w-full rounded border border-neutral-300 px-2 py-1 text-xs" />
            <span className="shrink-0 text-xs text-neutral-500" data-testid={sid('count')}>{matches.length} of {assets.length} shown</span>
            {filtered && (
              // Without this, a filter left on from an earlier look reads as "the asset is gone".
              <button type="button" data-testid={sid('clear')} onClick={() => { setQuery(''); setTags([]); setLimit(PAGE) }}
                className="shrink-0 rounded border border-neutral-300 px-2 py-0.5 text-xs text-neutral-700 hover:bg-neutral-100">Clear</button>
            )}
          </div>
          {chips.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {chips.map((t) => (
                <button key={t} type="button" aria-pressed={tags.includes(t)} onClick={() => toggleTag(t)} data-testid={sid(`tag-${t}`)}
                  className={`rounded-full px-2 py-0.5 text-[11px] ${tags.includes(t) ? 'bg-neutral-900 text-white' : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200'}`}>
                  {t}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {pinned.length > 0 && (
        <div className="mb-1">
          <span className="text-[11px] text-neutral-500">selected</span>
          <div className={grid}>{pinned.map((a, i) => tile(a, i, pinned))}</div>
        </div>
      )}
      <div className={large ? 'max-h-80 overflow-y-auto rounded border border-neutral-100 p-1' : ''}>
        <div className={grid}>
          {!multi && none && (
            <button type="button" disabled={disabled} aria-pressed={selected.length === 0} onClick={() => onChange([])}
              className={`flex aspect-square items-center justify-center rounded border text-xs ${selected.length === 0 ? 'border-neutral-900 ring-2 ring-neutral-900' : 'border-neutral-300 text-neutral-500'}`}>
              {none}
            </button>
          )}
          {shown.map((a, i) => tile(a, i, shown))}
          {onUpload && (
            <label className={`flex aspect-square cursor-pointer flex-col items-center justify-center rounded border border-dashed text-xs ${disabled || uploading ? 'border-neutral-200 text-neutral-400' : 'border-neutral-400 text-neutral-600 hover:border-neutral-700'}`}
              title="add an image from your computer to the project">
              <span className="text-lg leading-none">+</span>
              {uploading ? 'Uploading…' : 'Upload…'}
              <input type="file" accept="image/*" className="sr-only" disabled={disabled || uploading} data-testid={sid('upload')}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) onUpload(f); e.target.value = '' }} />
            </label>
          )}
        </div>
        {large && results.length > limit && (
          <button type="button" className="mt-1 w-full rounded bg-neutral-100 py-1 text-xs text-neutral-700 hover:bg-neutral-200"
            onClick={() => setLimit((n) => n + PAGE)} data-testid={sid('more')}>
            Show more ({results.length - limit} more)
          </button>
        )}
        {large && matches.length === 0 && <p className="p-2 text-xs text-neutral-500">No assets match.</p>}
      </div>
    </div>
  )
}
