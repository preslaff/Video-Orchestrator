'use client'

import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp } from 'lucide-react'
import { errorMessage } from '@/api/client'
import { describeStep, type JobEvent } from '@/api/events'
import type { AssemblyClip, AssemblySpecIn, AssemblyStatus as AssemblyStatusT, JobPublic } from '@/api/types'
import { useAssembly, useRunAssembly } from '@/hooks/useJobs'
import { MediaThumb } from '@/components/MediaThumb'
import { Button, Empty, ErrorBox, fmtTime } from '@/components/ui'
import { clampCut, defaultCuts, fmtBytes, fmtSeconds, groupTransitions, maxTransition, totalDuration, type CutDraft } from '@/lib/assembly'

type Row = { scene_index: number; selected: boolean; source: 'final' | 'raw' }

function clipSize(job: JobPublic): [number, number] | null {
  const v = (job.profile as { video?: { width?: number; height?: number } } | null)?.video
  return v?.width && v.height ? [v.width, v.height] : null
}

type Draft = { rows: Row[]; cuts: CutDraft[]; fps: number | '' }

/** Initial editor state from the last assembly spec (if any) or all clips in order. */
function seedDraft(status: AssemblyStatusT | undefined): Draft {
  if (!status) return { rows: [], cuts: [], fps: '' }
  const clips = status.clips
  const known = new Set(clips.map((c) => c.scene_index))
  const spec = status.record?.spec as { items?: { scene_index: number; source: 'final' | 'raw' }[]; cuts?: CutDraft[]; fps?: number | null } | undefined
  const rows: Row[] = []
  const used = new Set<number>()
  for (const it of spec?.items ?? []) {
    if (known.has(it.scene_index) && !used.has(it.scene_index)) {
      rows.push({ scene_index: it.scene_index, selected: true, source: it.source ?? 'final' })
      used.add(it.scene_index)
    }
  }
  for (const c of clips) if (!used.has(c.scene_index)) rows.push({ scene_index: c.scene_index, selected: !spec, source: 'final' })
  const n = rows.filter((r) => r.selected).length
  const cuts = defaultCuts(n).map((d, i) => spec?.cuts?.[i] ?? d)
  return { rows, cuts, fps: spec?.fps ?? '' }
}

export function AssemblePanel({ job, events, onStarted }: { job: JobPublic; events: JobEvent[]; onStarted?: () => void }) {
  const status = useAssembly(job.id)
  const run = useRunAssembly(job.id)
  const clips = useMemo(() => status.data?.clips ?? [], [status.data])
  const byIndex = useMemo(() => new Map(clips.map((c) => [c.scene_index, c])), [clips])
  // Editor state is seeded once from the first loaded status (key resets it if the job's clips change).
  const [draft, setDraft] = useState<Draft | null>(null)
  const seeded = draft ?? seedDraft(status.data)
  const rows = seeded.rows
  const cuts = seeded.cuts          // cut i sits between selected clip i and i+1 (position-keyed, survives reordering)
  const fps = seeded.fps
  const update = (patch: Partial<Draft>) => setDraft({ ...seeded, ...patch })
  const [encoder, setEncoder] = useState<'libx264' | 'h264_nvenc'>('libx264')

  const selected = rows.filter((r) => r.selected)
  const targetFps = fps || status.data?.profile_fps || clips[0]?.fps || 16
  const durations = selected.map((r) => byIndex.get(r.scene_index)?.duration ?? 0)
  const cutList: CutDraft[] = selected.slice(1).map((_, i) =>
    clampCut(cuts[i] ?? defaultCuts(2)[0], durations[i], durations[i + 1], targetFps))
  const total = totalDuration(durations, cutList)
  const groups = groupTransitions(status.data?.transitions ?? ['none', 'fade'])
  const active = !!status.data?.active
  const busy = active || run.isPending || job.status === 'running'
  const lastStep = [...events].reverse().find((e) => e.type === 'step' && e.step === 'assemble')
  const rec = status.data?.record ?? null

  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir
    if (j < 0 || j >= rows.length) return
    const copy = [...rows]
    ;[copy[i], copy[j]] = [copy[j], copy[i]]
    update({ rows: copy })
  }
  const toggle = (i: number) => update({ rows: rows.map((x, k) => (k === i ? { ...x, selected: !x.selected } : x)) })
  const setSource = (i: number, source: 'final' | 'raw') => update({ rows: rows.map((x, k) => (k === i ? { ...x, source } : x)) })
  const setCut = (pos: number, patch: Partial<CutDraft>) => {
    const next = [...cuts]
    while (next.length <= pos) next.push(defaultCuts(2)[0])
    next[pos] = { ...next[pos], ...patch }
    update({ cuts: next })
  }
  const setFps = (v: number | '') => update({ fps: v })

  const submit = () => {
    const spec: AssemblySpecIn = {
      items: selected.map((r) => ({ scene_index: r.scene_index, source: r.source })),
      cuts: cutList,
      fps: fps || null,
      width: null,
      height: null,
      crf: 18,
      encoder,
    }
    run.mutate({ spec, force: job.status !== 'done' }, { onSuccess: () => onStarted?.() })
  }

  if (status.isLoading) return <Empty>Loading…</Empty>
  if (status.error) return <ErrorBox message={errorMessage(status.error)} />
  if (!status.data?.available) return <Empty>No clips to assemble yet.</Empty>
  const size = clipSize(job)

  return (
    <div className="space-y-4">
      <p className="text-sm text-neutral-600">Tick the clips, order them, choose a transition for every cut. The result is re-encoded once (h264, same size and fps as the clips) and lands in the export folder as <code className="rounded bg-neutral-100 px-1">assembly.mp4</code>.</p>

      <ol className="space-y-2">
        {rows.map((r, i) => {
          const clip = byIndex.get(r.scene_index) as AssemblyClip | undefined
          const selPos = selected.findIndex((s) => s.scene_index === r.scene_index)
          const pos = selPos > 0 ? selPos - 1 : -1            // index of the cut before this clip
          const cut = pos >= 0 ? cutList[pos] : null
          const maxT = pos >= 0 ? maxTransition(durations[pos], durations[pos + 1], targetFps) : 0
          return (
            <li key={r.scene_index} className="space-y-2">
              {cut && (
                <div className="ml-10 flex flex-wrap items-center gap-2 rounded border border-dashed border-neutral-300 bg-neutral-50 px-3 py-1.5 text-sm" data-testid={`cut-${pos}`}>
                  <span className="text-xs uppercase text-neutral-500">cut</span>
                  <select value={cut.transition} onChange={(e) => setCut(pos, { transition: e.target.value })} className="rounded border border-neutral-300 bg-white px-2 py-1" aria-label="transition">
                    {groups.map((g) => (
                      <optgroup key={g.label} label={g.label}>{g.names.map((n) => <option key={n} value={n}>{n}</option>)}</optgroup>
                    ))}
                  </select>
                  {cut.transition !== 'none' && (
                    <label className="flex items-center gap-1 text-xs text-neutral-600">
                      <input type="number" min={0.1} max={maxT} step={0.1} value={cut.duration} aria-label="transition seconds"
                        onChange={(e) => setCut(pos, { duration: Number(e.target.value) })}
                        className="w-20 rounded border border-neutral-300 px-2 py-1 text-sm" />
                      s (max {maxT}s)
                    </label>
                  )}
                </div>
              )}
              <div className={`flex items-center gap-3 rounded border p-2 ${r.selected ? 'border-neutral-300 bg-white' : 'border-neutral-100 bg-neutral-50 opacity-60'}`}>
                <input type="checkbox" checked={r.selected} onChange={() => toggle(i)} aria-label={`include scene ${r.scene_index + 1}`} />
                <div className="w-28 shrink-0">
                  <MediaThumb url={clip?.url} video width={size?.[0]} height={size?.[1]} />
                </div>
                <div className="flex-1 text-sm">
                  <div className="font-medium">Scene {r.scene_index + 1} <span className="text-xs font-normal text-neutral-500">{fmtSeconds(clip?.duration)} · {clip?.width}×{clip?.height} @ {clip?.fps} fps</span></div>
                  {clip?.has_raw && (
                    <label className="text-xs text-neutral-600">source
                      <select value={r.source} onChange={(e) => setSource(i, e.target.value as 'final' | 'raw')} className="ml-1 rounded border border-neutral-300 bg-white px-1 py-0.5">
                        <option value="final">final (post-processed)</option>
                        <option value="raw">raw Wan output</option>
                      </select>
                    </label>
                  )}
                  {clip?.error && <div className="text-xs text-red-700">{clip.error}</div>}
                </div>
                <div className="flex flex-col gap-1">
                  <button type="button" onClick={() => move(i, -1)} disabled={i === 0} className="rounded border border-neutral-300 p-1 disabled:opacity-30" aria-label="move up"><ArrowUp className="h-3 w-3" /></button>
                  <button type="button" onClick={() => move(i, 1)} disabled={i === rows.length - 1} className="rounded border border-neutral-300 p-1 disabled:opacity-30" aria-label="move down"><ArrowDown className="h-3 w-3" /></button>
                </div>
              </div>
            </li>
          )
        })}
      </ol>

      <div className="flex flex-wrap items-center gap-4 text-sm">
        <span data-testid="total"><b>{selected.length}</b> clip(s) · <b>{fmtSeconds(total)}</b> total</span>
        <label className="flex items-center gap-1 text-neutral-600">fps
          <select value={fps} onChange={(e) => setFps(e.target.value === '' ? '' : Number(e.target.value))} className="rounded border border-neutral-300 bg-white px-2 py-1">
            <option value="">clips ({status.data.profile_fps ?? clips[0]?.fps ?? '?'})</option>
            {[24, 25, 30].map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-1 text-neutral-600">encoder
          <select value={encoder} onChange={(e) => setEncoder(e.target.value as 'libx264' | 'h264_nvenc')} className="rounded border border-neutral-300 bg-white px-2 py-1">
            <option value="libx264">libx264 (CPU)</option>
            <option value="h264_nvenc">h264_nvenc (GPU)</option>
          </select>
        </label>
        <Button data-testid="assemble" variant="primary" disabled={busy || selected.length === 0} onClick={submit}>
          {run.isPending || active ? 'Assembling…' : rec ? 'Re-assemble' : 'Assemble'}
        </Button>
      </div>
      <ErrorBox message={run.error ? errorMessage(run.error) : null} />

      {active && (
        <p className="flex items-center gap-2 text-sm">
          <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-900" />
          {lastStep ? describeStep(lastStep) : 'Assembling…'}
        </p>
      )}

      {rec && !active && (
        rec.ok ? (
          <div className="grid gap-3 md:grid-cols-[minmax(0,28rem)_1fr]">
            <MediaThumb key={rec.url ?? rec.created_at} url={rec.url} video width={rec.width ?? size?.[0]} height={rec.height ?? size?.[1]} />
            <div className="space-y-1 text-sm">
              <div className="font-medium">assembly.mp4 {rec.exported ? <span className="rounded bg-green-100 px-1.5 py-0.5 text-xs text-green-800">exported</span> : <span className="rounded bg-neutral-100 px-1.5 py-0.5 text-xs">not exported yet</span>}</div>
              <div className="text-neutral-600">{rec.n_clips} clip(s) · {fmtSeconds(rec.duration)} · {rec.width}×{rec.height} @ {rec.fps} fps · {fmtBytes(rec.size_bytes)}</div>
              <div className="text-xs text-neutral-500">{fmtTime(rec.created_at)}</div>
              <code className="block break-all rounded bg-neutral-100 px-2 py-1 text-xs">{rec.path}</code>
              {!rec.exists && <div className="text-xs text-amber-700">The file is no longer on disk — re-assemble.</div>}
            </div>
          </div>
        ) : (
          <ErrorBox message={`Last assembly failed: ${rec.error}`} />
        )
      )}
    </div>
  )
}
