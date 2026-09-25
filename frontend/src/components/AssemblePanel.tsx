'use client'

import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp } from 'lucide-react'
import { errorMessage } from '@/api/client'
import { describeStep, type JobEvent } from '@/api/events'
import type { AssemblyClip, AssemblySpecIn, AssemblyStatus as AssemblyStatusT, JobPublic, TextItem } from '@/api/types'
import { useAssembly, useRunAssembly, useSaveCut } from '@/hooks/useJobs'
import { ClipPreview } from '@/components/ClipPreview'
import { MediaThumb } from '@/components/MediaThumb'
import { AssemblyVersions } from '@/components/AssemblyVersions'
import { TextItems } from '@/components/TextItems'
import { Button, Empty, ErrorBox, fmtTime } from '@/components/ui'
import { clampCut, clampTarget, defaultCuts, effectiveSource, snapToFrame, fmtBytes, fmtSeconds, fmtSpeed, groupTransitions, maxTransition, retimed, round2, speedFor, totalDuration, MAX_SPEED, type CutDraft } from '@/lib/assembly'

// `uid` is the row's own identity: a scene may appear more than once (a repeat, or two takes of
// one shot joined), so `scene_index` no longer identifies a row. `target`/`start`/`end` hold raw
// field text — null means "not set".
type Row = {
  uid: string; scene_index: number; selected: boolean; source: 'final' | 'raw'
  target: string | null; clip_id: string | null; start: string | null; end: string | null
}
// Seeded rows take their identity from their position, so re-deriving the draft (which happens
// on every render until the first edit) yields the *same* ids: React keys stay put and anything
// remembering a row by id — an open preview — survives. Rows added later get a unique id of
// their own, in a separate namespace so the two can never collide.
let nextUid = 0
const newUid = () => `d${nextUid++}`
const seedUid = (position: number) => `s${position}`

function clipSize(job: JobPublic): [number, number] | null {
  const v = (job.profile as { video?: { width?: number; height?: number } } | null)?.video
  return v?.width && v.height ? [v.width, v.height] : null
}

type Draft = { rows: Row[]; cuts: CutDraft[]; fps: number | ''; smooth: boolean; audio: boolean; narration: boolean; texts: TextItem[]; titles: boolean }

/** Initial editor state from the last assembly spec (if any) or all clips in order. */
function seedDraft(status: AssemblyStatusT | undefined): Draft {
  if (!status) return { rows: [], cuts: [], fps: '', smooth: false, audio: false, narration: true, texts: [], titles: true }
  const clips = status.clips
  const known = new Set(clips.map((c) => c.scene_index))
  const spec = status.record?.spec as { items?: { scene_index: number; source: 'final' | 'raw'; target_seconds?: number | null; clip_id?: string | null; start?: number | null; end?: number | null }[]; cuts?: CutDraft[]; fps?: number | null; smooth?: boolean; audio?: boolean; narration?: boolean; texts?: TextItem[]; titles?: boolean } | undefined
  const num = (v: number | null | undefined) => (v != null ? String(v) : null)
  const rows: Row[] = []
  // Every saved item becomes its own row, repeats included — collapsing them by scene used to
  // throw away half of a timeline that deliberately played a shot twice.
  for (const it of spec?.items ?? []) {
    if (known.has(it.scene_index)) {
      rows.push({ uid: seedUid(rows.length), scene_index: it.scene_index, selected: true, source: it.source ?? 'final',
                  target: num(it.target_seconds), clip_id: it.clip_id ?? null, start: num(it.start), end: num(it.end) })
    }
  }
  const placed = new Set(rows.map((r) => r.scene_index))
  for (const c of clips) if (!placed.has(c.scene_index)) rows.push({ uid: seedUid(rows.length), scene_index: c.scene_index, selected: !spec, source: 'final', target: null, clip_id: null, start: null, end: null })
  const n = rows.filter((r) => r.selected).length
  const cuts = defaultCuts(n).map((d, i) => spec?.cuts?.[i] ?? d)
  return { rows, cuts, fps: spec?.fps ?? '', smooth: !!spec?.smooth, audio: !!spec?.audio, narration: spec?.narration !== false,
    texts: spec?.texts ?? [], titles: spec?.titles !== false }
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
  const smooth = seeded.smooth
  const audio = seeded.audio
  const narration = seeded.narration
  const texts = seeded.texts
  const titles = seeded.titles
  const update = (patch: Partial<Draft>) => setDraft({ ...seeded, ...patch })
  const [encoder, setEncoder] = useState<'libx264' | 'h264_nvenc'>('libx264')
  // What the assembly is mastered as. The clips are rendered small (a 16 GB card animates at
  // 480p); this is where they become a file you can deliver.
  const [delivery, setDelivery] = useState<string | null>(null)
  const [grain, setGrain] = useState(0)
  // One open preview at a time — two clips playing together is noise, not a comparison.
  const [preview, setPreview] = useState<string | null>(null)
  const saveCut = useSaveCut(job.id)
  /** Keep this row's trim as a real take, then point the row at it: the timeline plays exactly
   *  the same thing, but the cut is now a file rather than an instruction — so it survives
   *  export, reuse, and anyone else opening the job. */
  const keepCut = (i: number, makeActive: boolean) => {
    const r = rows[i]
    saveCut.mutate(
      { index: r.scene_index, clipId: r.clip_id, start: parseNum(r.start), end: parseNum(r.end), makeActive },
      { onSuccess: ({ takeId }: { takeId: string | null }) => update({ rows: rows.map((x, k) =>
          (k === i ? { ...x, clip_id: takeId ?? x.clip_id, start: null, end: null } : x)) }) },
    )
  }

  const selected = rows.filter((r) => r.selected)
  const targetFps = fps || status.data?.profile_fps || clips[0]?.fps || 16
  const parseNum = (raw: string | null): number | null => {
    if (raw == null || raw === '') return null
    const n = Number(raw)
    return Number.isFinite(n) && n >= 0 ? n : null
  }
  /** What this row actually plays: the trimmed span, else the whole clip. */
  const srcOf = (r: Row) => {
    const d = byIndex.get(r.scene_index)?.duration ?? 0
    const f = byIndex.get(r.scene_index)?.fps || targetFps
    return effectiveSource(d, parseNum(r.start), parseNum(r.end), f)
  }
  const sources = selected.map(srcOf)
  const parseTarget = (raw: string | null): number | null => {
    if (raw == null) return null
    const n = Number(raw)
    return Number.isFinite(n) && n > 0 ? n : null
  }
  const targets = selected.map((r, i) => { const t = parseTarget(r.target); return t == null ? null : clampTarget(sources[i], t) })
  const durations = retimed(sources, targets)   // retimed lengths drive cuts, limits and the total
  const cutList: CutDraft[] = selected.slice(1).map((_, i) =>
    clampCut(cuts[i] ?? defaultCuts(2)[0], durations[i], durations[i + 1], targetFps))
  const total = totalDuration(durations, cutList)
  const anySlow = targets.some((t, i) => t != null && speedFor(sources[i], t) < 1)
  // Whether there is any audio to keep decides whether 'keep audio' matters at all: with it
  // off, narration does not sit over the clip's own sound, it replaces it.
  const withAudio = clips.filter((c) => c.has_audio).length
  const anyClipAudio = withAudio > 0
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
  const setTarget = (i: number, target: string | null) => update({ rows: rows.map((x, k) => (k === i ? { ...x, target } : x)) })
  const setTrim = (i: number, which: 'start' | 'end', raw: string) =>
    update({ rows: rows.map((x, k) => (k === i ? { ...x, [which]: raw === '' ? null : raw } : x)) })
  const setTake = (i: number, clip_id: string | null) => update({ rows: rows.map((x, k) => (k === i ? { ...x, clip_id } : x)) })
  /** Place this clip again, right after itself — two takes of one shot, or a deliberate repeat.
   *  The copy gets its own identity, so editing one instance never touches the other. */
  const duplicate = (i: number) => {
    const copy = [...rows]
    copy.splice(i + 1, 0, { ...rows[i], uid: newUid(), selected: true })
    update({ rows: copy, cuts: [...cuts, defaultCuts(2)[0]] })
  }

  const submit = () => {
    const spec: AssemblySpecIn = {
      items: selected.map((r, i) => ({
        scene_index: r.scene_index, source: r.source, target_seconds: targets[i],
        clip_id: r.clip_id, start: parseNum(r.start), end: parseNum(r.end),
      })),
      cuts: cutList,
      fps: fps || null,
      width: null,
      height: null,
      crf: 18,
      encoder,
      smooth,
      audio,
      narration,
      duck: 6,
      texts,
      titles,
      delivery: target?.id ?? 'source',
      grain,
      max_mbps: null,
    }
    run.mutate({ spec, force: job.status !== 'done' }, { onSuccess: () => onStarted?.() })
  }

  if (status.isLoading) return <Empty>Loading…</Empty>
  if (status.error) return <ErrorBox message={errorMessage(status.error)} />
  if (!status.data?.available) return <Empty>No clips to assemble yet.</Empty>
  const size = clipSize(job)
  const deliveryTargets = status.data.deliveries ?? []
  const target = deliveryTargets.find((d) => d.id === (delivery ?? status.data?.delivery)) ?? deliveryTargets[0]
  const outW = target?.width ?? size?.[0] ?? clips[0]?.width ?? null
  const outH = target?.height ?? size?.[1] ?? clips[0]?.height ?? null
  const outFps = fps || target?.fps || status.data.profile_fps || clips[0]?.fps || null
  const upscale = outW && size?.[0] ? outW / size[0] : 1

  return (
    <div className="space-y-4">
      <p className="text-sm text-neutral-600">Tick the clips, order them, choose a transition for every cut. The result is re-encoded once and lands in the export folder as <code className="rounded bg-neutral-100 px-1">assembly.mp4</code>.</p>

      {deliveryTargets.length > 1 && (
        <div className="flex flex-wrap items-center gap-3 rounded border border-neutral-200 bg-neutral-50 p-2 text-sm">
          <label className="flex items-center gap-1 text-neutral-700">deliver as
            <select aria-label="delivery" value={target?.id ?? 'source'} onChange={(e) => setDelivery(e.target.value)}
              className="rounded border border-neutral-300 bg-white px-2 py-1">
              {deliveryTargets.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
            </select>
          </label>
          <span data-testid="delivery-format" className="text-neutral-600">
            {outW && outH ? `${outW}×${outH}` : 'the clips\' size'}{outFps ? ` @ ${outFps} fps` : ''}
          </span>
          {upscale > 1.05 && (
            <span className="text-xs text-amber-800" data-testid="delivery-upscale">
              upscaled {upscale.toFixed(1)}× from {size?.[0]}×{size?.[1]} — sharpness comes from the clips, not the scaler
            </span>
          )}
          {target?.note && <span className="text-xs text-neutral-500">{target.note}</span>}
          <label className="flex items-center gap-1 text-neutral-600" title="light film grain; an upscale reads plastic without it">
            grain
            <select aria-label="grain" value={grain} onChange={(e) => setGrain(Number(e.target.value))}
              className="rounded border border-neutral-300 bg-white px-2 py-1">
              <option value={0}>none</option>
              <option value={1}>light</option>
              <option value={2}>medium</option>
            </select>
          </label>
        </div>
      )}

      <ol className="space-y-2">
        {rows.map((r, i) => {
          const clip = byIndex.get(r.scene_index) as AssemblyClip | undefined
          // By identity, not by scene: two rows can share a scene and only one of them owns the
          // transition that follows it.
          const selPos = selected.findIndex((s) => s.uid === r.uid)
          const pos = selPos > 0 ? selPos - 1 : -1            // index of the cut before this clip
          const cut = pos >= 0 ? cutList[pos] : null
          const maxT = pos >= 0 ? maxTransition(durations[pos], durations[pos + 1], targetFps) : 0
          return (
            <li key={r.uid} className="space-y-2">
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
                <input type="checkbox" checked={r.selected} onChange={() => toggle(i)} aria-label={`include clip ${i + 1}`} />
                <div className="w-28 shrink-0">
                  <MediaThumb url={clip?.url} video width={size?.[0]} height={size?.[1]} caption={`Scene ${r.scene_index + 1} · clip`} />
                </div>
                <div className="flex-1 text-sm">
                  <div className="font-medium">Scene {r.scene_index + 1} <span className="text-xs font-normal text-neutral-500">{fmtSeconds(clip?.duration)} · {clip?.width}×{clip?.height} @ {clip?.fps} fps</span></div>
                  {r.selected && clip?.duration != null && (() => {
                    const cf = clip.fps || targetFps
                    const snap = (raw: string | null) => { const n = parseNum(raw); return n == null ? null : snapToFrame(n, cf) }
                    const lo = snap(r.start)
                    const hi = snap(r.end)
                    return (
                      <label className="mt-1 flex flex-wrap items-center gap-2 text-xs text-neutral-600">
                        use
                        <input type="number" aria-label={`clip ${i + 1} from`} min={0} max={round2(clip.duration!)} step={0.001}
                          value={r.start ?? ''} placeholder="0" onChange={(e) => setTrim(i, 'start', e.target.value)}
                          className="w-20 rounded border border-neutral-300 px-2 py-0.5 text-sm" />
                        to
                        <input type="number" aria-label={`clip ${i + 1} to`} min={0} max={round2(clip.duration!)} step={0.001}
                          value={r.end ?? ''} placeholder={String(round2(clip.duration!))} onChange={(e) => setTrim(i, 'end', e.target.value)}
                          className="w-20 rounded border border-neutral-300 px-2 py-0.5 text-sm" />
                        s
                        {/* A cut lands on a frame, so say which one — a field that silently moves
                            the number you typed reads as broken. */}
                        <span data-testid={`trim-${i}`} className="text-neutral-500">
                          {lo == null && hi == null
                            ? 'whole clip'
                            : `${round2(lo ?? 0)}–${round2(hi ?? clip.duration!)}s · frame ${Math.round((lo ?? 0) * cf)} · ${fmtSeconds(srcOf(r))}`}
                        </span>
                        {(r.start != null || r.end != null) && (
                          <button type="button" onClick={() => { setTrim(i, 'start', ''); setTrim(i, 'end', '') }}
                            className="text-neutral-500 underline">all</button>
                        )}
                        {(r.start != null || r.end != null) && (
                          <>
                            <button type="button" data-testid={`save-cut-${i}`} disabled={saveCut.isPending}
                              onClick={() => keepCut(i, false)}
                              title="render this span to a file and keep it as one of the scene's takes"
                              className="rounded border border-neutral-300 bg-white px-2 py-0.5 disabled:opacity-50">
                              {saveCut.isPending ? 'saving…' : 'save cut'}
                            </button>
                            <button type="button" data-testid={`save-cut-active-${i}`} disabled={saveCut.isPending}
                              onClick={() => keepCut(i, true)}
                              title="save it and make it this scene's clip, so the export folder carries the cut"
                              className="text-neutral-700 underline disabled:opacity-50">
                              + use as scene clip
                            </button>
                          </>
                        )}
                        <button type="button" data-testid={`preview-toggle-${i}`}
                          onClick={() => setPreview(preview === r.uid ? null : r.uid)}
                          className="text-neutral-700 underline">
                          {preview === r.uid ? 'hide' : 'preview'}
                        </button>
                      </label>
                    )
                  })()}
                  {preview === r.uid && clip?.url && clip.duration != null && (() => {
                    // The same two numbers the row shows: the trim decides what plays, "play in"
                    // decides how fast, and the preview has to obey both to be worth looking at.
                    const s = srcOf(r)
                    const want = parseTarget(r.target)
                    return (
                      <div className="mt-2 max-w-sm">
                        <ClipPreview url={clip.url} duration={clip.duration} fps={clip.fps || targetFps}
                          start={parseNum(r.start)} end={parseNum(r.end)}
                          speed={speedFor(s, want == null ? null : clampTarget(s, want))}
                          testId={`preview-${i}`} />
                      </div>
                    )
                  })()}
                  {r.selected && (clip?.takes?.length ?? 0) > 1 && (
                    <label className="mt-1 block text-xs text-neutral-600">take
                      <select value={r.clip_id ?? ''} onChange={(e) => setTake(i, e.target.value || null)}
                        aria-label={`clip ${i + 1} take`}
                        className="ml-1 rounded border border-neutral-300 bg-white px-1 py-0.5">
                        <option value="">current</option>
                        {clip!.takes!.map((tk, k) => (
                          <option key={tk.id} value={tk.id!}>take {k + 1}{tk.active ? ' (current)' : ''}</option>
                        ))}
                      </select>
                    </label>
                  )}
                  {r.selected && clip?.duration != null && (() => {
                    const src = srcOf(r)
                    const parsed = parseTarget(r.target)
                    const eff = parsed == null ? src : clampTarget(src, parsed)
                    const spd = speedFor(src, parsed == null ? null : eff)
                    return (
                      <label className="mt-1 flex flex-wrap items-center gap-2 text-xs text-neutral-600">
                        play in
                        <input type="number" aria-label={`clip ${i + 1} seconds`} min={round2(src / MAX_SPEED)} max={round2(src * MAX_SPEED)} step={0.1}
                          value={r.target ?? String(round2(src))} onChange={(e) => setTarget(i, e.target.value)}
                          onBlur={() => { if (r.target != null && parseTarget(r.target) == null) setTarget(i, null) }}
                          className="w-20 rounded border border-neutral-300 px-2 py-0.5 text-sm" />
                        s
                        <span data-testid={`speed-${i}`} className={`rounded px-1.5 py-0.5 ${spd === 1 ? 'bg-neutral-100' : spd < 1 ? 'bg-blue-100 text-blue-900' : 'bg-amber-100 text-amber-900'}`}>{fmtSpeed(spd)}</span>
                        {r.target != null && <button type="button" onClick={() => setTarget(i, null)} className="text-neutral-500 underline">natural</button>}
                      </label>
                    )
                  })()}
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
                  <button type="button" onClick={() => duplicate(i)} disabled={!clip?.duration}
                    data-testid={`duplicate-${i}`} title="place this clip again, so two takes can make one longer shot"
                    className="rounded border border-neutral-300 p-1 text-[10px] leading-none disabled:opacity-30"
                    aria-label={`duplicate clip ${i + 1}`}>⧉</button>
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
            <option value="">{target?.fps ? `delivery (${target.fps})` : `clips (${status.data.profile_fps ?? clips[0]?.fps ?? '?'})`}</option>
            {[24, 25, 30].map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-1 text-neutral-600" title="motion-compensated interpolation (ffmpeg minterpolate) for clips slower than 1×; several seconds per clip">
          <input type="checkbox" aria-label="smooth slow motion" checked={smooth} onChange={(e) => update({ smooth: e.target.checked })} />
          smooth slow motion{anySlow && !smooth ? <span className="text-xs text-amber-700">(off: frames are repeated)</span> : <span className="text-xs text-neutral-400">(slower render)</span>}
        </label>
        <label className="flex items-center gap-1 text-neutral-600" title="keep the clips' own audio. Wan clips are silent, so this changes nothing for them; an imported clip brings its own sound, and turning this off means the narration replaces it rather than sitting over it">
          <input type="checkbox" aria-label="keep audio" checked={audio} onChange={(e) => update({ audio: e.target.checked })} />
          keep audio
          {anyClipAudio
            ? (audio
                ? <span className="text-xs text-neutral-400">({withAudio} of {clips.length} clips have a track)</span>
                : <span className="text-xs text-amber-700">(off: their own audio is dropped{narration ? ', so only the voice is heard' : ''})</span>)
            : <span className="text-xs text-neutral-400">(no clip has a track)</span>}
        </label>
        <label className="flex items-center gap-1 text-neutral-600"
          title="lay each scene's spoken line over its clip, ducking the clip audio while it speaks">
          <input type="checkbox" aria-label="narration" checked={narration}
            onChange={(e) => update({ narration: e.target.checked })} />
          narration
        </label>
        <label className="flex items-center gap-1 text-neutral-600"
          title="draw the captions — scene captions and the ones below. Off renders a clean pass without deleting anything">
          <input type="checkbox" aria-label="titles" checked={titles}
            onChange={(e) => update({ titles: e.target.checked })} />
          titles{texts.length > 0 && <span className="text-xs text-neutral-400">({texts.length} on the film)</span>}
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

      <div className="mt-3 rounded border border-neutral-200 p-3">
        <div className="mb-2 flex items-baseline gap-2">
          <h3 className="text-sm font-medium">Captions on the film</h3>
          <span className="text-xs text-neutral-500">
            absolute times on the finished cut — a title card, an end card, credits. These do not
            follow a reorder or a trim; a caption that should stay with a shot belongs on the scene.
          </span>
        </div>
        <TextItems items={texts} onChange={(v: TextItem[]) => update({ texts: v })}
          videoUrl={status.data?.record?.url ?? undefined}
          effects={status.data?.text_effects} presets={status.data?.text_presets}
          fonts={status.data?.text_fonts}
          limit={total} busy={run.isPending} testId="timeline-texts" />
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
            <MediaThumb key={rec.url ?? rec.created_at} url={rec.url} video width={rec.width ?? size?.[0]} height={rec.height ?? size?.[1]} caption={`assembly v${rec.version}`} />
            <div className="space-y-1 text-sm">
              <div className="font-medium">assembly.mp4 <span className="font-mono text-xs text-neutral-500">v{rec.version}</span> {rec.exported ? <span className="rounded bg-green-100 px-1.5 py-0.5 text-xs text-green-800">exported</span> : <span className="rounded bg-neutral-100 px-1.5 py-0.5 text-xs">not exported yet</span>}</div>
              <div className="text-neutral-600">{rec.n_clips} clip(s) · {fmtSeconds(rec.duration)} · {rec.width}×{rec.height} @ {rec.fps} fps · {fmtBytes(rec.size_bytes)}</div>
              <div className="text-xs text-neutral-500">{fmtTime(rec.created_at)}</div>
              <code className="block break-all rounded bg-neutral-100 px-2 py-1 text-xs">{rec.path}</code>
              {!rec.exists && <div className="text-xs text-amber-700">The file is no longer on disk — re-assemble.</div>}
              {rec.exists && rec.stale && <div className="text-xs text-amber-700" data-testid="assembly-stale">Clips changed since this assembly — assemble again to include them.</div>}
            </div>
          </div>
        ) : (
          <ErrorBox message={`Last assembly failed: ${rec.error}`} />
        )
      )}

      {!active && (status.data.versions?.length ?? 0) > 0 && (
        <AssemblyVersions jobId={job.id} versions={status.data.versions} size={size} disabled={busy} />
      )}
    </div>
  )
}
