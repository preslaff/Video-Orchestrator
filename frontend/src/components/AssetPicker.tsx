'use client'

import type { AssetOut } from '@/api/types'

/** Thumbnails of project assets to pick from. Single choice (with a "none" tile) or an
 * ordered multi-selection (click order = priority, shown as a badge). With ``onUpload``
 * a last tile adds a file from the computer (the caller turns it into a project asset). */
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
  const toggle = (id: string) => {
    if (disabled) return
    if (!multi) return onChange(selected[0] === id ? [] : [id])
    onChange(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id])
  }
  return (
    <div className="grid grid-cols-3 gap-1 sm:grid-cols-4 lg:grid-cols-6" data-testid={testId}>
      {!multi && none && (
        <button type="button" disabled={disabled} aria-pressed={selected.length === 0} onClick={() => onChange([])}
          className={`flex aspect-square items-center justify-center rounded border text-xs ${selected.length === 0 ? 'border-neutral-900 ring-2 ring-neutral-900' : 'border-neutral-300 text-neutral-500'}`}>
          {none}
        </button>
      )}
      {assets.map((a) => {
        const pos = selected.indexOf(a.id)
        const on = pos >= 0
        // The caption is what makes picking a reference for scene 23 out of seventy
        // thumbnails a decision rather than a guess.
        const caption = a.note_caption || ''
        return (
          <button key={a.id} type="button" disabled={disabled} aria-pressed={on} onClick={() => toggle(a.id)}
            title={[`${a.kind}: ${a.label || a.name}`, caption].filter(Boolean).join('\n')}
            data-testid={testId ? `${testId}-${a.id}` : undefined}
            className={`relative overflow-hidden rounded border text-left ${on ? 'border-neutral-900 ring-2 ring-neutral-900' : 'border-neutral-300 hover:border-neutral-500'}`}>
            {/* eslint-disable-next-line @next/next/no-img-element -- served by FastAPI */}
            <img src={a.url} alt={a.label || a.name} className="aspect-square w-full object-cover" />
            {multi && on && <span className="absolute left-1 top-1 rounded bg-neutral-900 px-1 text-xs text-white">{pos + 1}</span>}
            {a.label_source === 'auto' && (
              <span className="absolute right-1 top-1 rounded bg-amber-500 px-1 text-[10px] font-medium text-white" title="suggested by Gemma, not yet reviewed">AI</span>
            )}
            <span className="block truncate px-1 pt-0.5 text-[11px] text-neutral-700">{a.label || a.name}</span>
            {caption && <span className="block truncate px-1 pb-0.5 text-[10px] text-neutral-500">{caption}</span>}
          </button>
        )
      })}
      {onUpload && (
        <label className={`flex aspect-square cursor-pointer flex-col items-center justify-center rounded border border-dashed text-xs ${disabled || uploading ? 'border-neutral-200 text-neutral-400' : 'border-neutral-400 text-neutral-600 hover:border-neutral-700'}`}
          title="add an image from your computer to the project">
          <span className="text-lg leading-none">+</span>
          {uploading ? 'Uploading…' : 'Upload…'}
          <input type="file" accept="image/*" className="sr-only" disabled={disabled || uploading} data-testid={testId ? `${testId}-upload` : undefined}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onUpload(f); e.target.value = '' }} />
        </label>
      )}
    </div>
  )
}
