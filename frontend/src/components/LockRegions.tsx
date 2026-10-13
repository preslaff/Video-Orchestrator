'use client'

import { useRef, useState } from 'react'
import type { LockedRegion } from '@/api/types'
import { Button } from './ui'

/** Where a drag started and where it is now, in fractions of the frame. */
interface Drag { x0: number; y0: number; x1: number; y1: number }

const MIN_SIZE = 0.01   // a stray click is not a rectangle

// A rectangle over a label does not need seventeen digits, and a tidy fraction survives a
// round trip through JSON unchanged. `|| 0` also turns a -0 from the clamp back into 0.
const round = (v: number) => Math.round(v * 1e4) / 1e4 || 0

function toRegion(d: Drag, matte = false): LockedRegion {
  return {
    x: round(Math.min(d.x0, d.x1)),
    y: round(Math.min(d.y0, d.y1)),
    w: round(Math.abs(d.x1 - d.x0)),
    h: round(Math.abs(d.y1 - d.y0)),
    matte,
  }
}

function style(r: LockedRegion) {
  return { left: `${r.x * 100}%`, top: `${r.y * 100}%`, width: `${r.w * 100}%`, height: `${r.h * 100}%` }
}

/**
 * Draw the parts of a keyframe the animator must not redraw.
 *
 * A video model rebuilds every frame from a latent compressed 8x in each direction, so printed
 * type below roughly 32 px of cap height cannot survive it — a brand label comes back as a
 * smear, wrong from the very first frame. Marking it here takes it out of the model's hands:
 * those pixels are pasted back from this frame onto every frame of the clip.
 *
 * Each rectangle can be switched to a **shape**, which segments the marked object inside the box
 * and locks that instead — for a label curved around a bottle, where the box would also paste
 * back the background around it. It costs a SAM2 pass per box, so it is off by default.
 */
export function LockRegions({
  url, regions, busy = false, onSave, onCancel, testId = 'locks',
}: {
  url: string
  regions: LockedRegion[]
  busy?: boolean
  onSave: (regions: LockedRegion[]) => void
  onCancel: () => void
  testId?: string
}) {
  const [draft, setDraft] = useState<LockedRegion[]>(regions)
  const [drag, setDrag] = useState<Drag | null>(null)
  const frame = useRef<HTMLDivElement>(null)

  /** Pointer position as a fraction of the frame, clamped to it. */
  const at = (e: { clientX: number; clientY: number }) => {
    const box = frame.current?.getBoundingClientRect()
    if (!box || !box.width || !box.height) return null
    const clamp = (v: number) => Math.min(1, Math.max(0, v))
    return { x: clamp((e.clientX - box.left) / box.width), y: clamp((e.clientY - box.top) / box.height) }
  }

  const start = (e: React.PointerEvent) => {
    if (busy) return
    const p = at(e)
    if (!p) return
    e.currentTarget.setPointerCapture?.(e.pointerId)
    setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y })
  }
  const move = (e: React.PointerEvent) => {
    if (!drag) return
    const p = at(e)
    if (p) setDrag({ ...drag, x1: p.x, y1: p.y })
  }
  const end = () => {
    if (!drag) return
    const r = toRegion(drag)
    setDrag(null)
    if (r.w >= MIN_SIZE && r.h >= MIN_SIZE) setDraft([...draft, r])
  }
  const toggleMatte = (i: number) =>
    setDraft(draft.map((r, j) => (j === i ? { ...r, matte: !r.matte } : r)))

  const live = drag ? toRegion(drag) : null
  return (
    <div className="space-y-2" data-testid={testId}>
      <p className="text-xs text-neutral-600">
        Drag over the label, the monogram — anything printed. Those pixels are taken from this
        frame and put back on every frame of the clip, following the shot as it moves, so the
        model never redraws them.
      </p>
      <div
        ref={frame}
        className="relative touch-none select-none overflow-hidden rounded border border-neutral-300"
        data-testid={`${testId}-frame`}
        onPointerDown={start}
        onPointerMove={move}
        onPointerUp={end}
        onPointerCancel={end}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={url} alt="the frame to mark" className="block w-full" draggable={false} />
        {draft.map((r, i) => (
          <div key={i}
            className={`absolute border-2 bg-amber-300/25 ${r.matte ? 'border-dashed border-sky-400' : 'border-amber-400'}`}
            style={style(r)}>
            <button
              type="button"
              className="absolute -right-2 -top-2 rounded-full bg-neutral-900 px-1.5 text-xs leading-tight text-white"
              data-testid={`${testId}-remove-${i}`}
              title="unmark this rectangle"
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => { e.stopPropagation(); setDraft(draft.filter((_, j) => j !== i)) }}
            >
              ✕
            </button>
            <button
              type="button"
              className={`absolute -left-2 -top-2 rounded-full px-1.5 text-xs leading-tight ${
                r.matte ? 'bg-sky-500 text-white' : 'bg-white text-neutral-700 ring-1 ring-neutral-400'}`}
              data-testid={`${testId}-matte-${i}`}
              title={r.matte
                ? 'locking the shape inside this box — click for the plain rectangle'
                : 'lock the shape inside this box instead of the rectangle (slower: one SAM2 pass)'}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => { e.stopPropagation(); toggleMatte(i) }}
            >
              ◓
            </button>
          </div>
        ))}
        {live && <div className="absolute border-2 border-amber-400 bg-amber-300/25" style={style(live)} />}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button disabled={busy} onClick={() => onSave(draft)} data-testid={`${testId}-save`}>
          {draft.length === 0 ? 'Remove the marks' : `Lock ${draft.length} region${draft.length > 1 ? 's' : ''}`}
        </Button>
        <button type="button" disabled={busy} data-testid={`${testId}-cancel`}
          className="text-xs text-neutral-700 underline-offset-2 hover:underline disabled:opacity-50"
          onClick={onCancel}>
          Cancel
        </button>
        {draft.some((r) => r.matte) && !busy && (
          <span className="text-xs text-neutral-500">
            {draft.filter((r) => r.matte).length} by shape — each needs a pass over the clip
          </span>
        )}
        {busy && (
          <span className="text-xs text-neutral-500">
            {draft.some((r) => r.matte)
              ? 'finding the shape in every frame, then putting the pixels back…'
              : 'putting the pixels back…'}
          </span>
        )}
      </div>
    </div>
  )
}
