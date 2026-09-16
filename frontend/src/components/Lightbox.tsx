'use client'

import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

/**
 * One lightbox for the whole app: any image or video preview opens its **original file** large.
 *
 * Previews show thumbnails (320 px asset thumbs, 80 px version tiles); judging a keyframe, a
 * reference photo's framing or a clip's motion needs the real thing. `useLightbox().open` takes a
 * group of items so a grid can be stepped through with ← / →.
 */

export type LightboxItem = { url: string; caption?: string; video?: boolean }
type OpenArgs = { items: LightboxItem[]; index?: number }
type Ctx = { open: (args: OpenArgs) => void }

// Outside a provider (isolated component tests) opening is a no-op rather than a crash.
const LightboxContext = createContext<Ctx>({ open: () => {} })

export const useLightbox = () => useContext(LightboxContext)

export function LightboxProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<{ items: LightboxItem[]; index: number } | null>(null)
  const open = useCallback(({ items, index = 0 }: OpenArgs) => {
    const usable = items.filter((i) => !!i.url)
    if (!usable.length) return
    // Opening pauses any small player on the page, so two copies of a clip never play at once.
    document.querySelectorAll('video').forEach((v) => v.pause())
    // Items without a file are skipped; keep pointing at the item that was clicked.
    const at = usable.indexOf(items[index])
    setState({ items: usable, index: at >= 0 ? at : 0 })
  }, [])
  return (
    <LightboxContext.Provider value={{ open }}>
      {children}
      {state && <LightboxView items={state.items} index={state.index}
        onIndex={(index) => setState((s) => (s ? { ...s, index } : s))} onClose={() => setState(null)} />}
    </LightboxContext.Provider>
  )
}

function LightboxView({ items, index, onIndex, onClose }: {
  items: LightboxItem[]; index: number; onIndex: (i: number) => void; onClose: () => void
}) {
  const item = items[index]
  const many = items.length > 1
  const closeRef = useRef<HTMLButtonElement>(null)
  const prev = index > 0 ? () => onIndex(index - 1) : undefined
  const next = index < items.length - 1 ? () => onIndex(index + 1) : undefined

  const count = items.length
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      else if (e.key === 'ArrowLeft' && index > 0) onIndex(index - 1)
      else if (e.key === 'ArrowRight' && index < count - 1) onIndex(index + 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, onIndex, index, count])
  useEffect(() => { closeRef.current?.focus() }, [])
  useEffect(() => {              // no page scroll behind the overlay
    const was = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = was }
  }, [])

  const nav = 'absolute top-1/2 -translate-y-1/2 rounded-full bg-white/15 px-3 py-2 text-2xl text-white hover:bg-white/30 disabled:opacity-20'
  return (
    <div role="dialog" aria-modal="true" aria-label={item.caption || 'preview'} data-testid="lightbox"
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/90 px-4 py-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <button ref={closeRef} type="button" onClick={onClose} aria-label="close" data-testid="lightbox-close"
        className="absolute right-3 top-3 rounded bg-white/15 px-2.5 py-1 text-lg text-white hover:bg-white/30">✕</button>
      {item.video ? (
        // key: a new clip must replace the player, not keep playing the old source
        <video key={item.url} src={item.url} controls autoPlay loop playsInline data-testid="lightbox-video"
          className="max-h-[85vh] max-w-full rounded bg-black" />
      ) : (
        // eslint-disable-next-line @next/next/no-img-element -- local media served by FastAPI
        <img key={item.url} src={item.url} alt={item.caption || ''} data-testid="lightbox-image"
          className="max-h-[85vh] max-w-full rounded object-contain" onClick={(e) => e.stopPropagation()} />
      )}
      <div className="mt-2 flex max-w-full flex-wrap items-center justify-center gap-3 text-sm text-neutral-200">
        {many && <span data-testid="lightbox-counter">{index + 1} / {items.length}</span>}
        {item.caption && <span className="max-w-3xl truncate" title={item.caption}>{item.caption}</span>}
        <a href={item.url} target="_blank" rel="noreferrer" className="text-neutral-400 underline-offset-2 hover:underline">open original ↗</a>
      </div>
      {many && (
        <>
          <button type="button" onClick={prev} disabled={!prev} aria-label="previous" data-testid="lightbox-prev" className={`${nav} left-2`}>‹</button>
          <button type="button" onClick={next} disabled={!next} aria-label="next" data-testid="lightbox-next" className={`${nav} right-2`}>›</button>
        </>
      )}
    </div>
  )
}

/** A preview that opens the lightbox when clicked — for images whose click means nothing else. */
export function Zoomable({ items, index = 0, className = '', children, testId }: {
  items: LightboxItem[]; index?: number; className?: string; children: ReactNode; testId?: string
}) {
  const { open } = useLightbox()
  return (
    <button type="button" onClick={() => open({ items, index })} data-testid={testId}
      title={items[index]?.caption ? `${items[index].caption} — click to enlarge` : 'click to enlarge'}
      className={`block cursor-zoom-in ${className}`}>
      {children}
    </button>
  )
}

/**
 * A corner ⤢ for tiles whose click already does something (select a version, pick an asset), and
 * for videos, whose click is play/pause. Place it inside a `relative` wrapper, as a sibling of the
 * tile — never inside another button.
 */
export function ZoomButton({ items, index = 0, always = false, className = '', testId }: {
  items: LightboxItem[]; index?: number; always?: boolean; className?: string; testId?: string
}) {
  const { open } = useLightbox()
  if (!items[index]?.url) return null
  return (
    <button type="button" onClick={(e) => { e.stopPropagation(); open({ items, index }) }} data-testid={testId}
      aria-label="enlarge" title="enlarge"
      className={`absolute z-[1] rounded bg-white/85 px-1 text-xs leading-5 text-neutral-700 shadow-sm hover:bg-white focus:opacity-100 ${always ? '' : 'opacity-100 sm:opacity-0 sm:group-hover:opacity-100'} ${className}`}>
      ⤢
    </button>
  )
}
