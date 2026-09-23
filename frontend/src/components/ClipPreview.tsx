'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { round2, snapToFrame } from '@/lib/assembly'

/**
 * Where the playhead sits and what the counter says, for a given mode.
 *
 * `speed` is the retime ("play in N seconds"): the position inside the clip is source time, but
 * the clock a viewer experiences is that divided by the speed — which is the number the "play in"
 * field asked for, and therefore the one the counter has to show.
 */
export function playhead(
  now: number, lo: number, hi: number, whole: boolean, duration: number, speed = 1,
) {
  const from = whole ? 0 : lo
  const to = whole ? duration : hi
  const span = Math.max(1e-6, to - from)
  const rate = whole || !speed ? 1 : speed        // the whole clip is shown as the file holds it
  return {
    at: Math.min(1, Math.max(0, (now - from) / span)),
    elapsed: Math.max(0, now - from) / rate,
    total: (to - from) / rate,
  }
}

/**
 * A clip with its kept span marked, played either whole or as it will be cut.
 *
 * Trimming by typing numbers is exact but blind: the question is always "does the cut land
 * before she turns her head?", and that is answered by watching, not by arithmetic. Both modes
 * play the same file — the span is a seek and a stop, so switching between what the timeline
 * will use and what the clip actually holds costs nothing and needs no render.
 */
export function ClipPreview({
  url, poster, duration, fps, start, end, speed = 1, testId = 'preview',
}: {
  url: string
  poster?: string | null
  duration: number
  fps: number
  /** The retime from "play in N seconds" (source length / target). 1 = natural. */
  speed?: number
  /** The kept span, already snapped; null means the clip's own edge. */
  start: number | null
  end: number | null
  testId?: string
}) {
  const ref = useRef<HTMLVideoElement>(null)
  const raf = useRef<number | null>(null)
  const [now, setNow] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [whole, setWhole] = useState(false)

  const lo = Math.max(0, snapToFrame(start ?? 0, fps))
  const hi = Math.min(duration, snapToFrame(end ?? duration, fps))
  const trimmed = start != null || end != null

  // `timeupdate` fires about four times a second, which reads as a stuttering counter. While the
  // clip is playing the position is sampled every frame instead.
  // Held in a ref so the loop can schedule itself without the callback depending on its own
  // identity (which would rebuild it every frame).
  const sampler = useRef<() => void>(() => {})
  useEffect(() => {
    sampler.current = () => {
      const v = ref.current
      if (!v) return
      setNow(v.currentTime)
      if (!whole && v.currentTime >= hi - 1e-3) {
        v.pause()
        v.currentTime = lo
        setNow(lo)
        return
      }
      raf.current = requestAnimationFrame(() => sampler.current())
    }
  })
  const sample = useCallback(() => sampler.current(), [])

  useEffect(() => () => { if (raf.current) cancelAnimationFrame(raf.current) }, [])

  // Play it at the speed the timeline will: a slow-motion clip that previews at natural speed is
  // not showing you the shot you are about to cut. The whole-clip mode stays natural, since that
  // mode exists to show what the file holds.
  useEffect(() => {
    const v = ref.current
    if (v) v.playbackRate = whole ? 1 : Math.min(4, Math.max(0.25, speed || 1))
  }, [speed, whole])

  // Retyping the span moves the picture to the new in-point, so the number in the field and the
  // frame on screen never tell different stories. Keyed on the in-point alone: reacting to
  // `playing` too would snap back to the start every time you pressed pause.
  const appliedLo = useRef(lo)
  useEffect(() => {
    if (appliedLo.current === lo) return
    appliedLo.current = lo
    const v = ref.current
    if (!v || whole) return
    v.currentTime = lo
    setNow(lo)
  }, [lo, whole])

  const toggle = () => {
    const v = ref.current
    if (!v) return
    if (playing) { v.pause(); return }
    if (!whole && (v.currentTime < lo || v.currentTime >= hi - 1e-3)) v.currentTime = lo
    if (whole && v.currentTime >= duration - 1e-3) v.currentTime = 0
    void v.play()
  }

  const seekTo = (fraction: number) => {
    const v = ref.current
    if (!v) return
    const from = whole ? 0 : lo
    const to = whole ? duration : hi
    v.currentTime = from + fraction * (to - from)
    setNow(v.currentTime)
  }

  const head = playhead(now, lo, hi, whole, duration, speed)
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / (duration || 1)) * 100))}%`

  return (
    <div className="space-y-1" data-testid={testId}>
      <video
        ref={ref} key={url} src={url} poster={poster ?? undefined} preload="metadata"
        className="w-full rounded bg-black object-contain" style={{ maxHeight: '13rem' }}
        onPlay={() => { setPlaying(true); raf.current = requestAnimationFrame(sample) }}
        onPause={() => { setPlaying(false); if (raf.current) cancelAnimationFrame(raf.current) }}
        onTimeUpdate={() => { if (!playing) setNow(ref.current?.currentTime ?? 0) }}
        onLoadedMetadata={() => { if (!whole && trimmed) { ref.current!.currentTime = lo; setNow(lo) } }}
      />

      {/* The whole clip, with the part the timeline keeps picked out of it. */}
      <div
        role="presentation" data-testid={`${testId}-bar`}
        className="relative h-3 cursor-pointer rounded bg-neutral-200"
        onClick={(e) => {
          const box = e.currentTarget.getBoundingClientRect()
          if (!box.width) return
          const x = (e.clientX - box.left) / box.width          // a click is in whole-clip space
          const t = x * duration
          const from = whole ? 0 : lo
          const to = whole ? duration : hi
          seekTo(Math.min(1, Math.max(0, (t - from) / Math.max(1e-6, to - from))))
        }}
      >
        <div className="absolute inset-y-0 rounded bg-amber-300"
          style={{ left: pct(lo), width: `${Math.max(0, ((hi - lo) / (duration || 1)) * 100)}%` }} />
        <div className="absolute inset-y-0 w-0.5 bg-neutral-900" style={{ left: pct(now) }} data-testid={`${testId}-head`} />
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs text-neutral-600">
        <button type="button" onClick={toggle} data-testid={`${testId}-play`}
          className="rounded border border-neutral-300 bg-white px-2 py-0.5 hover:bg-neutral-50">
          {playing ? '❚❚ pause' : '▶ play'}
        </button>
        <button type="button" data-testid={`${testId}-mode`} aria-pressed={whole}
          onClick={() => { const v = ref.current; v?.pause(); setWhole(!whole); if (v) { v.currentTime = !whole ? 0 : lo; setNow(v.currentTime) } }}
          className={`rounded border px-2 py-0.5 ${whole ? 'border-neutral-400 bg-neutral-100' : 'border-neutral-300 bg-white'}`}>
          {whole ? 'whole clip' : 'as cut'}
        </button>
        {/* Elapsed inside whichever span is playing, plus the absolute frame — which is the number
            you need when the cut has to land on a specific moment. */}
        <span data-testid={`${testId}-time`} className="font-mono tabular-nums">
          {round2(head.elapsed).toFixed(2)} / {round2(head.total).toFixed(2)}s · frame {Math.round(now * fps)}
        </span>
        {!whole && speed !== 1 && (
          <span data-testid={`${testId}-speed`} className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-900">
            {round2(speed)}×
          </span>
        )}
        {trimmed && !whole && <span className="text-neutral-400">cut from {round2(duration)}s</span>}
      </div>
    </div>
  )
}
