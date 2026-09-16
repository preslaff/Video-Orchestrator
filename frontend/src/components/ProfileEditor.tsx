'use client'

import { useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { errorMessage } from '@/api/client'
import type { OutputProfile } from '@/api/types'
import { useCreateProfile, useUpdateProfile, useWorkflows } from '@/hooks/useSetup'
import { Button, Card, ErrorBox } from '@/components/ui'
import { ASPECTS, keyframeSize, orientation, parseAspect, wanSize } from '@/lib/aspect'
import { estimateClip, fmtRange, vramRatio, vramWarning } from '@/lib/render'
import { clipSeconds, isWanLength, snapToWan } from '@/lib/wan'

type Draft = {
  name: string; purpose: OutputProfile['purpose']; kind: 'video' | 'stills'; aspect: string
  keyframe: { workflow: string; width: number; height: number; steps: number; cfg: number; unet: string }
  video: { workflow: string; width: number; height: number; length: number; fps: number; negative: string }
  stills: { variants: number }
  style: { keyframe_prefix: string; keyframe_negative: string; animation_suffix: string; animation_negative: string }
  face_swap: { enabled: boolean; workflow: string; restore_model: string }
  evaluate: { enabled: boolean; max_attempts: number; review_threshold: number }
  edit_workflow: string
  post_steps: { workflow: string; params: string }[]
}

const PURPOSES: OutputProfile['purpose'][] = ['cinematic', 'social_vertical', 'social_square', 'ad_broll', 'stills']

function toDraft(p: OutputProfile): Draft {
  return {
    name: p.name, purpose: p.purpose ?? 'cinematic', kind: p.kind ?? 'video', aspect: p.aspect ?? '16:9',
    keyframe: { workflow: p.keyframe?.workflow ?? 'keyframe', width: p.keyframe?.width ?? 1024, height: p.keyframe?.height ?? 576, steps: p.keyframe?.steps ?? 24, cfg: p.keyframe?.cfg ?? 5, unet: p.keyframe?.unet ?? '' },
    video: { workflow: p.video?.workflow ?? 'animate', width: p.video?.width ?? 832, height: p.video?.height ?? 480, length: p.video?.length ?? 49, fps: p.video?.fps ?? 16, negative: p.video?.negative ?? '' },
    stills: { variants: p.stills?.variants ?? 1 },
    style: { keyframe_prefix: p.style?.keyframe_prefix ?? '', keyframe_negative: p.style?.keyframe_negative ?? '', animation_suffix: p.style?.animation_suffix ?? 'cinematic', animation_negative: p.style?.animation_negative ?? '' },
    face_swap: { enabled: p.face_swap?.enabled ?? true, workflow: p.face_swap?.workflow ?? 'faceswap', restore_model: p.face_swap?.restore_model ?? 'GFPGANv1.4.pth' },
    evaluate: { enabled: p.evaluate?.enabled ?? true, max_attempts: p.evaluate?.max_attempts ?? 2, review_threshold: p.evaluate?.review_threshold ?? 4 },
    edit_workflow: p.edit_workflow ?? 'identity_edit',
    post_steps: (p.post_steps ?? []).map((s) => ({ workflow: s.workflow, params: JSON.stringify(s.params ?? {}) })),
  }
}

function toDoc(d: Draft): Record<string, unknown> {
  return {
    name: d.name, purpose: d.purpose, kind: d.kind, aspect: d.aspect,
    keyframe: { ...d.keyframe, unet: d.keyframe.unet || null },
    video: d.kind === 'video' ? d.video : null,
    stills: d.stills, style: d.style, face_swap: d.face_swap, evaluate: d.evaluate,
    edit_workflow: d.edit_workflow,
    post_steps: d.post_steps.filter((s) => s.workflow).map((s) => ({ workflow: s.workflow, params: s.params.trim() ? JSON.parse(s.params) : {} })),
  }
}

export function ProfileEditor({ profile, mode }: { profile: OutputProfile; mode: 'new' | 'edit' }) {
  const router = useRouter()
  const [d, setD] = useState<Draft>(() => toDraft(profile))
  const [customAspect, setCustomAspect] = useState(!ASPECTS.includes(profile.aspect ?? '16:9'))
  const workflows = useWorkflows()
  const create = useCreateProfile()
  const update = useUpdateProfile(profile.id)
  const byKind = (kind: string) => workflows.data?.workflows.filter((w) => w.kind === kind).map((w) => w.name) ?? []

  const set = <K extends keyof Draft>(k: K, v: Draft[K]) => setD((x) => ({ ...x, [k]: v }))
  /** What this size costs: VRAM headroom and how long one clip takes. The numbers come from
   *  measured runs; a legal size can still be one nobody wants to wait for. */
  const renderCost = (v: Draft['video']) => {
    const est = estimateClip({ width: v.width, height: v.height, length: v.length, workflow: v.workflow })
    const warning = vramWarning(v.width, v.height, v.length)
    const ratio = vramRatio(v.width, v.height, v.length)
    if (!est) return null
    return (
      <p className={`text-xs ${warning ? 'text-amber-800' : 'text-neutral-500'}`}>
        <b>{fmtRange(est)}</b> per clip on {est.where}
        {` · ${ratio.toFixed(1)}× the tested 16 GB budget`}
        {warning ? ` — ${warning.replace(/^[\d.]+× the tested( 16 GB)? budget( \(portrait\))? — /, '')}` : ''}
        {!est.measured && ' · estimated with the local rate (this workflow has no measured run)'}
      </p>
    )
  }

  const applyAspect = (aspect: string) => {
    const kf = keyframeSize(aspect)
    const wan = wanSize(aspect)
    setD((x) => ({
      ...x, aspect,
      keyframe: kf ? { ...x.keyframe, width: kf[0], height: kf[1] } : x.keyframe,
      video: wan ? { ...x.video, width: wan[0], height: wan[1] } : x.video,
    }))
  }

  const problems = useMemo(() => {
    const out: string[] = []
    if (!d.name.trim()) out.push('Name is required.')
    if (!parseAspect(d.aspect)) out.push('Aspect must look like 16:9.')
    const blanks: string[] = []
    if (!d.keyframe.width || !d.keyframe.height) blanks.push('keyframe size')
    if (!d.keyframe.steps || !d.keyframe.cfg) blanks.push('keyframe steps/cfg')
    if (d.kind === 'video' && (!d.video.width || !d.video.height)) blanks.push('clip size')
    if (d.kind === 'video' && (!d.video.length || !d.video.fps)) blanks.push('clip length/fps')
    if (d.kind === 'stills' && !d.stills.variants) blanks.push('variants per scene')
    if (blanks.length) out.push(`Fill in: ${blanks.join(', ')}.`)
    if (d.kind === 'video' && d.video.length > 0 && !isWanLength(d.video.length)) {
      out.push(`Wan length must be 4n+1 (nearest: ${snapToWan(d.video.length)}).`)
    }
    for (const [i, s] of d.post_steps.entries()) {
      if (s.params.trim()) { try { JSON.parse(s.params) } catch { out.push(`Post-step ${i + 1}: params must be JSON.`) } }
    }
    return out
  }, [d])

  const save = () => {
    let doc: Record<string, unknown>
    try { doc = toDoc(d) } catch (e) { return alert(String(e)) }
    if (mode === 'new') {
      const { name, ...overrides } = doc
      create.mutate({ name: String(name), base: profile.id, overrides }, { onSuccess: (p) => router.push(`/setup/profiles/${p.id}`) })
    } else {
      update.mutate(doc)
    }
  }
  const err = create.error ? errorMessage(create.error) : update.error ? errorMessage(update.error) : null
  // An emptied field stays empty (0) instead of snapping back to a default — otherwise clearing
  // it restores the old number and the next keystroke appends to it ("480" + "1248").
  // `problems` above refuses to save a profile with a field left blank.
  const num = (v: string) => (v.trim() === '' ? 0 : Number(v))
  /** 0 is "the user cleared this", which renders as an empty box, not a literal zero. */
  const shown = (n: number) => (n ? String(n) : '')
  const inp = 'mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1 text-sm'
  const sel = inp + ' bg-white'

  return (
    <div className="space-y-4">
      <ErrorBox message={err} />
      {problems.map((p) => <ErrorBox key={p} message={p} />)}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Identity & format">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="block sm:col-span-2"><span className="text-xs text-neutral-500">name</span><input value={d.name} onChange={(e) => set('name', e.target.value)} className={inp} /></label>
            <label className="block"><span className="text-xs text-neutral-500">purpose</span><select value={d.purpose} onChange={(e) => set('purpose', e.target.value as Draft['purpose'])} className={sel}>{PURPOSES.map((p) => <option key={p} value={p}>{p}</option>)}</select></label>
            <label className="block"><span className="text-xs text-neutral-500">kind</span><select value={d.kind} onChange={(e) => set('kind', e.target.value as 'video' | 'stills')} className={sel}><option value="video">video (keyframes → clips)</option><option value="stills">stills (keyframes only, N variants)</option></select></label>
            <label className="block"><span className="text-xs text-neutral-500">aspect</span>
              {customAspect ? <input value={d.aspect} onChange={(e) => applyAspect(e.target.value)} placeholder="21:9" className={inp} /> : (
                <select value={d.aspect} onChange={(e) => applyAspect(e.target.value)} className={sel}>{ASPECTS.map((a) => <option key={a} value={a}>{a} ({orientation(a)})</option>)}</select>
              )}
              <label className="mt-1 flex items-center gap-1 text-xs text-neutral-500"><input type="checkbox" checked={customAspect} onChange={(e) => setCustomAspect(e.target.checked)} /> custom W:H</label>
            </label>
            <p className="text-xs text-neutral-500 sm:self-end">keyframe {d.keyframe.width}×{d.keyframe.height}{d.kind === 'video' ? ` · clip ${d.video.width}×${d.video.height}` : ''} — sizes follow the aspect (16-aligned); edit below if needed.</p>
          </div>
        </Card>
        <Card title="Keyframe (Klein)">
          <div className="grid gap-3 sm:grid-cols-3">
            <label className="block sm:col-span-3"><span className="text-xs text-neutral-500">workflow</span><select value={d.keyframe.workflow} onChange={(e) => set('keyframe', { ...d.keyframe, workflow: e.target.value })} className={sel}>{[d.keyframe.workflow, ...byKind('keyframe')].filter((v, i, a) => a.indexOf(v) === i).map((w) => <option key={w} value={w}>{w}</option>)}</select></label>
            <label className="block"><span className="text-xs text-neutral-500">width</span><input type="number" step={16} value={shown(d.keyframe.width)} onChange={(e) => set('keyframe', { ...d.keyframe, width: num(e.target.value) })} className={inp} /></label>
            <label className="block"><span className="text-xs text-neutral-500">height</span><input type="number" step={16} value={shown(d.keyframe.height)} onChange={(e) => set('keyframe', { ...d.keyframe, height: num(e.target.value) })} className={inp} /></label>
            <label className="block"><span className="text-xs text-neutral-500">steps</span><input type="number" min={1} value={shown(d.keyframe.steps)} onChange={(e) => set('keyframe', { ...d.keyframe, steps: num(e.target.value) })} className={inp} /></label>
            <label className="block"><span className="text-xs text-neutral-500">cfg</span><input type="number" step={0.5} value={shown(d.keyframe.cfg)} onChange={(e) => set('keyframe', { ...d.keyframe, cfg: num(e.target.value) })} className={inp} /></label>
            <label className="block sm:col-span-2"><span className="text-xs text-neutral-500">unet (model file; empty = template default)</span><input value={d.keyframe.unet} onChange={(e) => set('keyframe', { ...d.keyframe, unet: e.target.value })} className={inp + ' font-mono text-xs'} /></label>
          </div>
        </Card>
        {d.kind === 'video' ? (
          <Card title="Video (Wan)">
            <div className="grid gap-3 sm:grid-cols-3">
              <label className="block sm:col-span-3"><span className="text-xs text-neutral-500">workflow</span><select value={d.video.workflow} onChange={(e) => set('video', { ...d.video, workflow: e.target.value })} className={sel}>{[d.video.workflow, ...byKind('animate')].filter((v, i, a) => a.indexOf(v) === i).map((w) => <option key={w} value={w}>{w}</option>)}</select></label>
              <label className="block"><span className="text-xs text-neutral-500">width</span><input type="number" step={16} value={shown(d.video.width)} onChange={(e) => set('video', { ...d.video, width: num(e.target.value) })} className={inp} /></label>
              <label className="block"><span className="text-xs text-neutral-500">height</span><input type="number" step={16} value={shown(d.video.height)} onChange={(e) => set('video', { ...d.video, height: num(e.target.value) })} className={inp} /></label>
              <label className="block"><span className="text-xs text-neutral-500">fps</span><input type="number" min={1} max={60} value={shown(d.video.fps)} onChange={(e) => set('video', { ...d.video, fps: num(e.target.value) })} className={inp} /></label>
              <label className="block"><span className="text-xs text-neutral-500">length (frames, 4n+1)</span><input type="number" min={1} value={shown(d.video.length)} onChange={(e) => set('video', { ...d.video, length: num(e.target.value) })} onBlur={() => d.video.length > 0 && set('video', { ...d.video, length: snapToWan(d.video.length) })} className={inp} /></label>
              <p className="text-xs text-neutral-500 sm:col-span-2 sm:self-end">≈ {clipSeconds(d.video.length, d.video.fps)} s per clip · 49 / 81 / 121 are typical.</p>
              <div className="sm:col-span-3" data-testid="render-cost">{renderCost(d.video)}</div>
              <label className="block sm:col-span-3"><span className="text-xs text-neutral-500">negative (empty = style default)</span><input value={d.video.negative} onChange={(e) => set('video', { ...d.video, negative: e.target.value })} className={inp} /></label>
            </div>
          </Card>
        ) : (
          <Card title="Stills">
            <label className="block"><span className="text-xs text-neutral-500">variants per scene (1–8)</span><input type="number" min={1} max={8} value={shown(d.stills.variants)} onChange={(e) => set('stills', { variants: num(e.target.value) })} className={inp} /></label>
          </Card>
        )}
        <Card title="Style vocabulary">
          <div className="grid gap-3">
            <label className="block"><span className="text-xs text-neutral-500">keyframe prefix (technique first: “impasto oil painting”, “phone photo, natural light”)</span><input value={d.style.keyframe_prefix} onChange={(e) => set('style', { ...d.style, keyframe_prefix: e.target.value })} className={inp} /></label>
            <label className="block"><span className="text-xs text-neutral-500">keyframe negative</span><input value={d.style.keyframe_negative} onChange={(e) => set('style', { ...d.style, keyframe_negative: e.target.value })} className={inp} /></label>
            <label className="block"><span className="text-xs text-neutral-500">animation suffix (how motion prompts end)</span><input value={d.style.animation_suffix} onChange={(e) => set('style', { ...d.style, animation_suffix: e.target.value })} className={inp} /></label>
            <label className="block"><span className="text-xs text-neutral-500">animation negative</span><input value={d.style.animation_negative} onChange={(e) => set('style', { ...d.style, animation_negative: e.target.value })} className={inp} /></label>
          </div>
        </Card>
        <Card title="Face swap & evaluation">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={d.face_swap.enabled} onChange={(e) => set('face_swap', { ...d.face_swap, enabled: e.target.checked })} /> impose the master face (ReActor)</label>
            <label className="block"><span className="text-xs text-neutral-500">face-swap workflow</span><select value={d.face_swap.workflow} onChange={(e) => set('face_swap', { ...d.face_swap, workflow: e.target.value })} className={sel}>{[d.face_swap.workflow, ...byKind('faceswap')].filter((v, i, a) => a.indexOf(v) === i).map((w) => <option key={w} value={w}>{w}</option>)}</select></label>
            <label className="block"><span className="text-xs text-neutral-500">restore model</span><input value={d.face_swap.restore_model} onChange={(e) => set('face_swap', { ...d.face_swap, restore_model: e.target.value })} className={inp + ' font-mono text-xs'} /></label>
            <label className="block"><span className="text-xs text-neutral-500">edit workflow (reference-locked keyframes)</span><select value={d.edit_workflow} onChange={(e) => set('edit_workflow', e.target.value)} className={sel}>{[d.edit_workflow, ...byKind('edit')].filter((v, i, a) => a.indexOf(v) === i).map((w) => <option key={w} value={w}>{w}</option>)}</select></label>
            <label className="flex items-center gap-2 text-sm sm:col-span-2"><input type="checkbox" checked={d.evaluate.enabled} onChange={(e) => set('evaluate', { ...d.evaluate, enabled: e.target.checked })} /> evaluate keyframes with Gemma vision (retry with the fix)</label>
            <label className="block"><span className="text-xs text-neutral-500">max attempts</span><input type="number" min={1} max={6} value={shown(d.evaluate.max_attempts)} onChange={(e) => set('evaluate', { ...d.evaluate, max_attempts: num(e.target.value) })} className={inp} /></label>
            <label className="block"><span className="text-xs text-neutral-500">review threshold (score below → flag, never block)</span><input type="number" min={0} max={10} value={d.evaluate.review_threshold} onChange={(e) => set('evaluate', { ...d.evaluate, review_threshold: num(e.target.value) })} className={inp} /></label>
          </div>
        </Card>
        <Card title="Post-process steps" actions={<Button onClick={() => set('post_steps', [...d.post_steps, { workflow: byKind('postprocess')[0] ?? '', params: '' }])}>+ add step</Button>}>
          {d.post_steps.length === 0 && <p className="text-sm text-neutral-500">None. Registered <b>postprocess</b> workflows (interpolation, upscale…) run in order after each clip / still. {byKind('postprocess').length === 0 && 'No postprocess workflows are registered yet — see Workflows.'}</p>}
          <div className="space-y-2">
            {d.post_steps.map((s, i) => (
              <div key={i} className="grid gap-2 sm:grid-cols-[1fr_2fr_auto]">
                <select value={s.workflow} onChange={(e) => set('post_steps', d.post_steps.map((x, j) => (j === i ? { ...x, workflow: e.target.value } : x)))} className={sel}>
                  <option value="">— workflow —</option>
                  {[s.workflow, ...byKind('postprocess')].filter((v, k, a) => v && a.indexOf(v) === k).map((w) => <option key={w} value={w}>{w}</option>)}
                </select>
                <input value={s.params} placeholder='{"multiplier": 2}' onChange={(e) => set('post_steps', d.post_steps.map((x, j) => (j === i ? { ...x, params: e.target.value } : x)))} className={inp + ' font-mono text-xs'} />
                <Button variant="ghost" onClick={() => set('post_steps', d.post_steps.filter((_, j) => j !== i))}>remove</Button>
              </div>
            ))}
          </div>
        </Card>
      </div>
      <div className="flex items-center gap-3">
        <Button variant="primary" disabled={problems.length > 0 || create.isPending || update.isPending} onClick={save}>{mode === 'new' ? 'Create profile' : 'Save profile'}</Button>
        {update.isSuccess && !update.isPending && <span className="text-sm text-green-700">Saved.</span>}
      </div>
    </div>
  )
}
