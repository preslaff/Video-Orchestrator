'use client'

import { useEffect, useState } from 'react'
import { PHASE_LABELS, describeStep, type JobEvent } from '@/api/events'
import type { JobPublic } from '@/api/types'
import { GOTO_SCENE } from '@/components/GatePanel'

const VIDEO_PHASES = ['storyboard', 'keyframes', 'videos', 'export']
const STILLS_PHASES = ['storyboard', 'keyframes', 'stills', 'export']

export interface Progress {
  phases: string[]
  current: string | null        // phase in progress (null when idle / paused / done)
  done: string[]                // phases completed
  percent: number               // 0..100 over scene units
  units: { done: number; total: number }
  step: JobEvent | null         // last announced step (only meaningful while running)
}

/** The last event matching ``pred``, walked backwards in place.
 *
 *  ``[...events].reverse().find(...)`` copies and reverses the whole array — three times per
 *  render here, and this panel re-renders every second while a job runs. On a page left open
 *  for hours that is thousands of entries copied every tick, which drives the browser's GC
 *  hard enough to spin its thread pool. */
function findLast(events: readonly JobEvent[], pred: (e: JobEvent) => boolean): JobEvent | undefined {
  for (let i = events.length - 1; i >= 0; i--) if (pred(events[i])) return events[i]
  return undefined
}

/** Derive the progress model from the persisted job + the live events. */
export function deriveProgress(job: JobPublic, events: readonly JobEvent[]): Progress {
  const phases = job.kind === 'stills' ? STILLS_PHASES : VIDEO_PHASES
  const n = job.scenes.length
  const kfDone = job.scenes.filter((s) => !!s.keyframe_path).length
  const finalDone = job.kind === 'stills' ? job.scenes.filter((s) => s.status === 'complete').length : job.scenes.filter((s) => !!s.video_path).length
  const done: string[] = []
  if (n > 0) done.push('storyboard')
  if (n > 0 && kfDone === n) done.push('keyframes')
  if (n > 0 && finalDone === n) done.push(job.kind === 'stills' ? 'stills' : 'videos')
  if (job.export) done.push('export')

  const running = job.status === 'running' || job.status === 'pending'
  let current: string | null = null
  if (running) {
    const lastPhase = findLast(events, (e) => e.type === 'phase')
    current = lastPhase ? String(lastPhase.phase) : n === 0 ? 'storyboard' : kfDone < n ? 'keyframes' : phases[2]
  }
  const lastStep = findLast(events, (e) => e.type === 'step') ?? null
  const perScene = job.kind === 'stills' ? 2 : 2
  const total = Math.max(1, n) * perScene
  const units = n === 0 ? 0 : kfDone + finalDone
  return { phases, current, done, percent: Math.round((units / total) * 100), units: { done: units, total }, step: running ? lastStep : null }
}

function useTick(active: boolean) {
  const [, setNow] = useState(0)
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [active])
}

function elapsed(from: number | undefined): string {
  if (!from) return ''
  const s = Math.max(0, Math.floor((Date.now() - from) / 1000))
  return s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s` : `${s}s`
}

export function ProgressPanel({ job, events, connected }: { job: JobPublic; events: JobEvent[]; connected: boolean }) {
  const p = deriveProgress(job, events)
  const orphaned = job.status === 'running' && !job.active
  const running = (job.status === 'running' && !orphaned) || job.status === 'pending'
  useTick(running)
  const paused = job.status === 'needs_approval'
  const failed = job.status === 'failed'
  const stopped = job.status === 'stopped'
  const lastError = failed ? findLast(events, (e) => e.type === 'error') : undefined

  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4 shadow-sm" aria-live="polite">
      <ol className="flex flex-wrap items-center gap-2 text-sm">
        {p.phases.map((ph, i) => {
          const state = p.done.includes(ph) ? 'done' : p.current === ph ? 'current' : paused && p.done.length === i ? 'gate' : 'todo'
          const cls = {
            done: 'bg-green-100 text-green-800',
            current: 'bg-blue-100 text-blue-900 ring-2 ring-blue-300',
            gate: 'bg-amber-100 text-amber-900',
            todo: 'bg-neutral-100 text-neutral-500',
          }[state]
          return (
            <li key={ph} className="flex items-center gap-2">
              <span className={`rounded-full px-3 py-1 ${cls}`} data-state={state}>
                {state === 'current' && <span className="mr-1 inline-block h-2 w-2 animate-pulse rounded-full bg-blue-600" />}
                {state === 'done' ? '✓ ' : ''}{PHASE_LABELS[ph]}{state === 'gate' ? ' · awaiting approval' : ''}
              </span>
              {i < p.phases.length - 1 && <span className="text-neutral-300">→</span>}
            </li>
          )
        })}
      </ol>

      <div className="mt-3 h-2 w-full overflow-hidden rounded bg-neutral-100">
        <div className={`h-full rounded transition-all ${failed ? 'bg-red-500' : 'bg-neutral-900'}`} style={{ width: `${p.percent}%` }} />
      </div>
      <div className="mt-1 flex flex-wrap items-center justify-between gap-2 text-xs text-neutral-500">
        <span>{p.units.done}/{p.units.total} steps · {p.percent}%</span>
        <span>{running ? (job.stopping ? 'stopping…' : connected ? '● live' : '○ reconnecting…') : job.status === 'done' ? 'finished' : paused ? 'paused at a gate' : failed ? 'failed' : stopped ? 'stopped' : orphaned ? 'not running' : job.status}</span>
      </div>

      {running && (
        <p className="mt-3 flex items-start gap-2 text-sm">
          <span className="mt-1 inline-block h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-900" />
          <span>
            {job.stopping && <span className="mr-1 font-medium">Stopping — finishing the current step.</span>}
            {p.step ? describeStep(p.step) : p.current === 'storyboard' ? describeStep({ type: 'step', step: 'storyboard', job_id: job.id }) : 'Working…'}
            {p.step?._at ? <span className="ml-2 font-mono text-xs text-neutral-500">{elapsed(p.step._at)}</span> : null}
          </span>
        </p>
      )}
      {failed && <p className="mt-3 text-sm text-red-700">Failed{lastError ? `: ${String(lastError.error)}` : ''}. Check the decision log and the API log, then Retry — finished scenes are kept.</p>}
      {stopped && <p className="mt-3 text-sm text-neutral-700">Stopped. Edit any scene below, then Resume — finished work is kept.</p>}
      {orphaned && <p className="mt-3 text-sm text-amber-800">Not running: the server restarted while this job ran. Resume continues from its saved state.</p>}

      {job.scenes.length > 0 && (
        <ul className="mt-3 flex flex-wrap gap-1.5">
          {job.scenes.map((s) => {
            const active = p.step && p.step.index === s.index
            const label = s.status === 'complete' ? '✓' : s.status === 'pending' ? '·' : s.status
            return (
              <li key={s.index}>
                <button type="button" data-testid={`goto-scene-${s.index}`}
                  title={`${s.description}\n\nclick to jump to this scene`}
                  onClick={() => {
                    // The gate panel may have this scene collapsed: ask it to open before scrolling.
                    window.dispatchEvent(new CustomEvent(GOTO_SCENE, { detail: s.index }))
                    document.getElementById(`scene-${s.index}`)?.scrollIntoView({ block: 'start', behavior: 'smooth' })
                  }}
                  className={`rounded px-2 py-0.5 text-xs ${active ? 'bg-blue-100 text-blue-900' : s.status === 'complete' ? 'bg-green-50 text-green-800' : 'bg-neutral-100 text-neutral-600'} hover:ring-1 hover:ring-neutral-400`}>
                  s{s.index + 1} {label}{s.variants.length > 1 ? ` (${s.variants.length})` : ''}
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
