'use client'

import { useEffect, useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { errorDetails, errorMessage } from '@/api/client'
import type { JobSpec, OutputProfile } from '@/api/types'
import { useCreateJob } from '@/hooks/useJobs'
import { useEnv, useProfiles, useProject, useProjects, useWorkflows, type EnvReport } from '@/hooks/useSetup'
import { AssetPicker } from '@/components/AssetPicker'
import { EnvSummary } from '@/components/EnvSummary'
import { ZoomButton, Zoomable } from '@/components/Lightbox'
import { Button, Card, ErrorBox } from '@/components/ui'
import { ASPECTS, clipSeconds, clipSize, snapToWan, stillSize, type Aspect } from '@/lib/wan'
import { estimateClip, fmtDuration, fmtRange, vramWarning } from '@/lib/render'

const STEPS = ['Project', 'Profile', 'Brief', 'Review'] as const

/** The two sizes are different things — the still Klein renders, and the size Wan animates it at
 *  — so name them instead of showing "A → B", which reads as a conversion. */
export function profileSummary(p: OutputProfile): string {
  const kf = `still ${p.keyframe?.width ?? '?'}×${p.keyframe?.height ?? '?'}`
  if (p.kind === 'stills') return `stills · ${p.aspect} · ${kf} · ${p.stills?.variants ?? 1} variant(s)`
  const v = p.video
  if (!v) return `video · ${p.aspect} · ${kf}`
  return `video · ${p.aspect} · ${kf} · clip ${v.width}×${v.height} · ${v.length} frames @ ${v.fps} fps (~${clipSeconds(v.length, v.fps)}s)`
}

/** Aspect ratio of each thumbnail, measured in the browser.
 *
 *  The API does not carry asset dimensions, and thumbnails keep the original ratio — which is
 *  all that decides whether an image survives the centre crop. Nothing renders differently
 *  until a measurement arrives, so a headless environment simply shows no crop warning. */
function useThumbAspects(urls: string[]): Record<string, number> {
  const [ratios, setRatios] = useState<Record<string, number>>({})
  const key = urls.join('|')
  useEffect(() => {
    let alive = true
    for (const url of key ? key.split('|') : []) {
      const img = new window.Image()
      img.onload = () => {
        if (alive && img.naturalHeight) {
          setRatios((m) => (m[url] ? m : { ...m, [url]: img.naturalWidth / img.naturalHeight }))
        }
      }
      img.src = url
    }
    return () => { alive = false }
  }, [key])
  return ratios
}

export function NewJobWizard({ initialProjectId = '' }: { initialProjectId?: string } = {}) {
  const router = useRouter()
  const [step, setStep] = useState(initialProjectId ? 1 : 0)  // a preselected project skips straight to the profile step
  const [projectId, setProjectId] = useState<string>(initialProjectId)
  const [profileId, setProfileId] = useState<string>('')
  const [idea, setIdea] = useState('')
  const [style, setStyle] = useState('')
  const [nScenes, setNScenes] = useState<number | ''>('')
  const [seed, setSeed] = useState<number>(0)
  const [autonomy, setAutonomy] = useState<'semi' | 'full'>('semi')
  const [mode, setMode] = useState<'t2i' | 'edit'>('t2i')
  // "generate": vorch writes a storyboard and makes the keyframes. "animate": the keyframes
  // already exist as project assets — one scene each, and only the motion is written.
  const [source, setSource] = useState<'generate' | 'animate'>('generate')
  const [animateIds, setAnimateIds] = useState<string[]>([])
  // Output overrides: the model (a registered animate workflow, which also picks the backend)
  // and the size class. '' = whatever the profile says.
  const [clipWorkflow, setClipWorkflow] = useState('')
  const [clipAspect, setClipAspect] = useState<Aspect | ''>('')
  const [clipClass, setClipClass] = useState<'' | '480' | '720'>('')
  const [clipLength, setClipLength] = useState<number | ''>('')
  const [refIds, setRefIds] = useState<string[]>([])
  const [masterFace, setMasterFace] = useState('')
  const [faceSwap, setFaceSwap] = useState<'profile' | 'on' | 'off'>('profile')
  const [evaluate, setEvaluate] = useState<'profile' | 'on' | 'off'>('profile')
  const [force, setForce] = useState(false)

  const projects = useProjects()
  const project = useProject(projectId || undefined)
  const profiles = useProfiles()
  const effectiveProfileId = profileId || project.data?.default_profile_id || 'cinematic_16x9'
  const profile = profiles.data?.find((p) => p.id === effectiveProfileId)
  const env = useEnv(effectiveProfileId, false, step === 3)
  const animateWorkflows = useWorkflows('animate').data?.workflows ?? []
  // What this job will actually render at, once the overrides are folded in.
  const output = (() => {
    const v = profile?.video
    const aspect = (clipAspect || (profile?.aspect as Aspect) || '16:9') as Aspect
    const known = (ASPECTS as readonly string[]).includes(aspect)
    const cls = clipClass || '480'
    const [cw, ch] = clipClass || clipAspect ? clipSize(known ? aspect : '16:9', cls === '720' ? 720 : 480)
      : [v?.width ?? 0, v?.height ?? 0]
    const [sw, sh] = clipClass || clipAspect ? stillSize(known ? aspect : '16:9')
      : [profile?.keyframe?.width ?? 0, profile?.keyframe?.height ?? 0]
    const length = clipLength === '' ? (v?.length ?? 49) : Number(clipLength)
    return { aspect, width: cw, height: ch, still: [sw, sh] as [number, number], length,
             fps: v?.fps ?? 16, workflow: clipWorkflow || v?.workflow || 'animate' }
  })()
  const overrides = (() => {
    const o: Record<string, unknown> = {}
    const video: Record<string, unknown> = {}
    if (clipWorkflow) video.workflow = clipWorkflow
    if (clipLength !== '') video.length = snapToWan(Number(clipLength))
    if (clipAspect || clipClass) {
      o.aspect = output.aspect
      o.keyframe = { width: output.still[0], height: output.still[1] }
      video.width = output.width
      video.height = output.height
    }
    if (Object.keys(video).length) o.video = video
    return Object.keys(o).length ? o : null
  })()

  // What this job will cost in GPU time, from measured runs — the scene count is the multiplier
  // that turns a comfortable size into an overnight render.
  const renderCost = (() => {
    const v = profile?.video
    if (!v?.width || !v.height || !v.length) return null
    // The overridden size and model, not the profile's — that is what will actually render.
    const est = estimateClip({ width: output.width, height: output.height,
                               length: snapToWan(output.length), workflow: output.workflow })
    if (!est) return null
    const scenes = source === 'animate' ? (animateIds.length || null) : (nScenes === '' ? null : Number(nScenes))
    const warning = vramWarning(output.width, output.height, snapToWan(output.length))
    return (
      <span className={warning ? 'text-amber-800' : undefined}>
        {fmtRange(est)} per clip on {est.where}
        {scenes ? ` · ${fmtDuration(est.seconds * scenes)} for ${scenes} scenes` : ' · × the scene count'}
        {warning ? ` — ${warning}` : ''}
      </span>
    )
  })()
  const create = useCreateJob()

  const masterAsset = project.data?.assets.find((a) => a.kind === 'master_face')
  const swapOn = faceSwap === 'on' || (faceSwap === 'profile' && !!profile?.face_swap?.enabled)
  const refAssets = project.data?.assets.filter((a) => a.kind !== 'master_face') ?? []

  // Which of the picked images would lose pixels to the still's centre crop.
  const picked = animateIds.map((id) => refAssets.find((a) => a.id === id)).filter(Boolean) as typeof refAssets
  const thumbRatios = useThumbAspects(picked.map((a) => a.thumb_url || a.url))
  const croppedAssets = picked.filter((a) => {
    const r = thumbRatios[a.thumb_url || a.url]
    return r !== undefined && Math.abs(r - output.still[0] / output.still[1]) > 0.02
  })

  const problems = useMemo(() => {
    const out: string[] = []
    if (source === 'animate') {
      if (!projectId) out.push('Animating your own images needs a project.')
      if (animateIds.length === 0) out.push('Pick at least one image to animate.')
      // Their image *is* the keyframe; a swap would paste a face onto it.
      if (swapOn) out.push('Turn face swap off: it would paste a face onto your images.')
    } else {
      if (!idea.trim()) out.push('An idea is required.')
      if (swapOn && !masterFace && !masterAsset) out.push('Face swap is on but there is no master face (add one to the project or paste a path).')
      if (mode === 'edit' && refIds.length === 0) out.push('Edit mode needs at least one reference asset.')
    }
    return out
  }, [idea, swapOn, masterFace, masterAsset, mode, refIds, source, projectId, animateIds])

  const envFailures = env.data?.checks.filter((c) => c.status === 'fail') ?? []
  const blocked = problems.length > 0 || (envFailures.length > 0 && !force)
  const submitError = create.error ? errorMessage(create.error) : null
  const submitReport = create.error ? (errorDetails(create.error) as EnvReport | undefined) : undefined

  const spec: JobSpec = {
    idea: idea.trim(),
    style,
    autonomy,
    n_scenes: nScenes === '' ? null : Number(nScenes),
    seed,
    project_id: projectId || null,
    profile_id: profileId || null,
    profile_overrides: overrides,
    mode,
    reference_asset_ids: source === 'generate' && mode === 'edit' ? refIds : [],
    reference_images: [],
    keyframe_asset_ids: source === 'animate' ? animateIds : [],
    master_face: masterFace,
    face_swap: faceSwap === 'profile' ? null : faceSwap === 'on',
    evaluate: evaluate === 'profile' ? null : evaluate === 'on',
    force,
  }

  const submit = () => {
    create.mutate(spec, { onSuccess: (job) => router.push(`/jobs/${job.id}`) })
  }

  return (
    <div className="space-y-4">
      <ol className="flex gap-2 text-sm">
        {STEPS.map((s, i) => (
          <li key={s}>
            <button type="button" onClick={() => i < step && setStep(i)} className={`rounded px-3 py-1 ${i === step ? 'bg-neutral-900 text-white' : i < step ? 'bg-neutral-200' : 'bg-neutral-100 text-neutral-400'}`}>
              {i + 1}. {s}
            </button>
          </li>
        ))}
      </ol>

      {step === 0 && (
        <Card title="Project (optional)">
          <label className="block text-sm">
            <span className="text-neutral-600">Group this job under a project to reuse its master face, references and export folder.</span>
            <select data-testid="project" value={projectId} onChange={(e) => { setProjectId(e.target.value); setProfileId(''); setRefIds([]) }} className="mt-2 block w-full rounded border border-neutral-300 bg-white px-2 py-1">
              <option value="">— no project —</option>
              {projects.data?.map((p) => <option key={p.id} value={p.id}>{p.name} ({p.n_assets} asset{p.n_assets === 1 ? '' : 's'})</option>)}
            </select>
          </label>
          {project.data && (
            <div className="mt-3 flex items-start gap-4">
              {masterAsset ? (
                <Zoomable items={[{ url: masterAsset.url, caption: masterAsset.label || masterAsset.name }]} className="shrink-0">
                  {/* eslint-disable-next-line @next/next/no-img-element -- served by FastAPI */}
                  <img src={masterAsset.url} alt={masterAsset.label || masterAsset.name} className="h-24 w-24 rounded object-cover ring-1 ring-neutral-200" />
                </Zoomable>
              ) : (
                <div className="flex h-24 w-24 items-center justify-center rounded bg-neutral-100 text-center text-xs text-neutral-400">no master face</div>
              )}
              <div className="text-sm text-neutral-600">
                <p>default profile <b>{project.data.default_profile_id}</b></p>
                <p>master face {masterAsset ? <b>{masterAsset.label || masterAsset.name}</b> : <span className="text-amber-700">none — add one under Projects, or paste a path in step 3</span>}</p>
                <p>{refAssets.length} reference/product asset(s) for edit mode</p>
                {project.data.assets.filter((a) => a.kind === 'master_face').length > 1 && <p className="text-xs text-sky-800">{project.data.assets.filter((a) => a.kind === 'master_face').length} faces: the storyboard assigns one per scene from their labels (or none); adjust at the storyboard gate.</p>}
                <p>export → <code className="text-xs">{project.data.export_root_effective}</code></p>
              </div>
            </div>
          )}
        </Card>
      )}

      {step === 1 && (
        <Card title="Output profile">
          <div className="space-y-2">
            {profiles.data?.map((p) => (
              <label key={p.id} className={`flex cursor-pointer items-start gap-3 rounded border p-2 ${effectiveProfileId === p.id ? 'border-neutral-900 bg-neutral-50' : 'border-neutral-200'}`}>
                <input type="radio" name="profile" value={p.id} checked={effectiveProfileId === p.id} onChange={() => setProfileId(p.id)} className="mt-1" />
                <span>
                  <span className="font-medium">{p.name}</span> <span className="text-xs text-neutral-500">{p.id}{p.builtin ? ' · built-in' : ''}</span>
                  <br />
                  <span className="text-sm text-neutral-600">{profileSummary(p)} · {p.purpose}</span>
                </span>
              </label>
            ))}
          </div>
        </Card>
      )}

      {step === 2 && (
        <Card title="Brief">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="md:col-span-2">
              <span className="text-sm font-medium">Where do the keyframes come from?</span>
              <div className="mt-1 flex flex-wrap gap-2">
                {([['generate', 'Generate images', 'vorch writes a storyboard and makes every keyframe'],
                   ['animate', 'Animate my images', 'one scene per asset you pick; only the motion is written']] as const).map(([value, label, hint]) => (
                  <button key={value} type="button" data-testid={`source-${value}`} onClick={() => setSource(value)}
                    className={`flex-1 rounded border px-3 py-2 text-left text-sm ${source === value ? 'border-neutral-900 bg-neutral-900 text-white' : 'border-neutral-300 bg-white hover:border-neutral-500'}`}>
                    <span className="font-medium">{label}</span>
                    <span className={`block text-xs ${source === value ? 'text-neutral-300' : 'text-neutral-500'}`}>{hint}</span>
                  </button>
                ))}
              </div>
            </div>
            <label className="block text-sm md:col-span-2">
              <span className="font-medium">{source === 'animate' ? 'Direction (tone and pace — Gemma writes each motion from the picture)' : 'Idea'}</span>
              <textarea data-testid="idea" value={idea} onChange={(e) => setIdea(e.target.value)} rows={source === 'animate' ? 2 : 3} className="mt-1 block w-full rounded border border-neutral-300 px-2 py-1"
                placeholder={source === 'animate' ? 'intimate, slow, warm — a 15 second teaser' : 'A woman kneads bread at dawn in a stone kitchen…'} />
            </label>
            {source === 'animate' && (
              <div className="md:col-span-2">
                <span className="text-sm font-medium">Images to animate — click in the order they should play</span>
                {!projectId && <p className="text-xs text-amber-700">Pick a project in step 1 to choose its assets.</p>}
                {projectId && (
                  <AssetPicker assets={refAssets} selected={animateIds} multi onChange={setAnimateIds}
                    testId="animate" />
                )}
                <p className="mt-1 text-xs text-neutral-500">
                  {animateIds.length} selected · each becomes one scene, cropped to the profile&apos;s
                  {' '}{profile?.keyframe?.width}×{profile?.keyframe?.height} still
                </p>
              </div>
            )}
            <div className="md:col-span-2 rounded border border-neutral-200 bg-neutral-50 p-3">
              <span className="text-sm font-medium">Output</span>
              <div className="mt-1 grid gap-2 sm:grid-cols-4">
                <label className="block text-sm sm:col-span-2">
                  <span className="text-xs text-neutral-500">model</span>
                  <select data-testid="clip-workflow" value={clipWorkflow} onChange={(e) => setClipWorkflow(e.target.value)}
                    className="mt-0.5 block w-full rounded border border-neutral-300 bg-white px-2 py-1">
                    <option value="">profile default ({profile?.video?.workflow ?? 'animate'})</option>
                    {animateWorkflows.map((w) => (
                      <option key={w.name} value={w.name}>{w.name} — {w.backend}</option>
                    ))}
                  </select>
                </label>
                <label className="block text-sm">
                  <span className="text-xs text-neutral-500">aspect</span>
                  <select data-testid="clip-aspect" value={clipAspect} onChange={(e) => setClipAspect(e.target.value as Aspect | '')}
                    className="mt-0.5 block w-full rounded border border-neutral-300 bg-white px-2 py-1">
                    <option value="">profile ({profile?.aspect ?? '16:9'})</option>
                    {ASPECTS.map((a2) => <option key={a2} value={a2}>{a2}</option>)}
                  </select>
                </label>
                <label className="block text-sm">
                  <span className="text-xs text-neutral-500">size class</span>
                  <select data-testid="clip-class" value={clipClass} onChange={(e) => setClipClass(e.target.value as '' | '480' | '720')}
                    className="mt-0.5 block w-full rounded border border-neutral-300 bg-white px-2 py-1">
                    <option value="">profile</option>
                    <option value="480">480 — fits this GPU</option>
                    <option value="720">720 — cloud backend</option>
                  </select>
                </label>
                <label className="block text-sm">
                  <span className="text-xs text-neutral-500">frames (4n+1)</span>
                  <input type="number" data-testid="clip-length" min={5} step={4} value={clipLength}
                    onChange={(e) => setClipLength(e.target.value === '' ? '' : Number(e.target.value))}
                    placeholder={String(profile?.video?.length ?? 49)}
                    className="mt-0.5 block w-full rounded border border-neutral-300 px-2 py-1" />
                </label>
              </div>
              <p className="mt-1 text-xs text-neutral-600" data-testid="output-summary">
                still {output.still[0]}×{output.still[1]} → clip {output.width}×{output.height} ·{' '}
                {snapToWan(output.length)} frames @ {output.fps} fps (~{clipSeconds(snapToWan(output.length), output.fps)}s) · {output.workflow}
              </p>
              {source === 'animate' && croppedAssets.length > 0 && (
                <p className="text-xs text-amber-700" data-testid="crop-warning">
                  {croppedAssets.length} of your {animateIds.length} image(s) will be centre-cropped to{' '}
                  {output.still[0]}×{output.still[1]} — sides or top and bottom are lost.
                </p>
              )}
            </div>
            {source === 'generate' && <label className="block text-sm">
              <span className="font-medium">Style (free text, goes first in every keyframe prompt)</span>
              <input value={style} onChange={(e) => setStyle(e.target.value)} className="mt-1 block w-full rounded border border-neutral-300 px-2 py-1" placeholder="impasto oil painting / 35mm film still / phone photo" />
            </label>}
            <div className="grid grid-cols-3 gap-2">
              {source === 'generate' && <label className="block text-sm"><span className="font-medium">Scenes</span><input type="number" min={1} max={12} value={nScenes} onChange={(e) => setNScenes(e.target.value === '' ? '' : Number(e.target.value))} className="mt-1 block w-full rounded border border-neutral-300 px-2 py-1" placeholder="auto" /></label>}
              <label className="block text-sm"><span className="font-medium">Seed</span><input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} className="mt-1 block w-full rounded border border-neutral-300 px-2 py-1" /><span className="text-xs text-neutral-500">0 = random</span></label>
              <label className="block text-sm"><span className="font-medium">Autonomy</span><select value={autonomy} onChange={(e) => setAutonomy(e.target.value as 'semi' | 'full')} className="mt-1 block w-full rounded border border-neutral-300 bg-white px-2 py-1"><option value="semi">semi (gates)</option><option value="full">full</option></select></label>
            </div>
            {source === 'generate' && <label className="block text-sm">
              <span className="font-medium">Keyframe mode</span>
              <select data-testid="mode" value={mode} onChange={(e) => setMode(e.target.value as 't2i' | 'edit')} className="mt-1 block w-full rounded border border-neutral-300 bg-white px-2 py-1">
                <option value="t2i">text-to-image (new scenes)</option>
                <option value="edit">reference-locked edit (same composition, light edit)</option>
              </select>
            </label>}
            {source === 'generate' && mode === 'edit' && (
              <div className="text-sm md:col-span-2">
                <span className="font-medium">References (project assets)</span>
                <span className="ml-2 text-xs text-neutral-500">{refIds.length} selected · order = click order; the first is the primary reference</span>
                {!projectId && <p className="text-amber-700">Pick a project in step 1 to use its reference/product assets.</p>}
                {projectId && refAssets.length === 0 && <p className="text-amber-700">The selected project has no reference/product assets — add some under Projects.</p>}
                {refAssets.length > 0 && (
                  <div className="mt-2 grid grid-cols-3 gap-2 sm:grid-cols-4 lg:grid-cols-6">
                    {refAssets.map((a) => {
                      const pos = refIds.indexOf(a.id)
                      const selected = pos >= 0
                      return (
                        <div key={a.id} className="group relative min-w-0">
                        <button
                          type="button"
                          data-testid={`ref-${a.id}`}
                          aria-pressed={selected}
                          onClick={() => setRefIds(selected ? refIds.filter((x) => x !== a.id) : [...refIds, a.id])}
                          className={`relative block w-full overflow-hidden rounded border text-left ${selected ? 'border-neutral-900 ring-2 ring-neutral-900' : 'border-neutral-200 hover:border-neutral-400'}`}
                          title={a.label || a.name}
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element -- served by FastAPI */}
                          <img src={a.url} alt={a.label || a.name} className="aspect-square w-full bg-neutral-100 object-cover" />
                          {selected && <span className="absolute left-1 top-1 rounded-full bg-neutral-900 px-1.5 text-xs font-semibold text-white">{pos + 1}</span>}
                          <span className="block truncate px-1 py-0.5 text-xs text-neutral-700">
                            <span className="rounded bg-neutral-100 px-1">{a.kind}</span> {a.label || a.name}
                          </span>
                        </button>
                        <ZoomButton items={refAssets.map((x) => ({ url: x.url, caption: x.label || x.name }))} index={refAssets.indexOf(a)}
                          className="right-1 top-1" testId={`ref-${a.id}-zoom`} />
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )}
            <label className="block text-sm"><span className="font-medium">Face swap</span><select data-testid="face_swap" value={faceSwap} onChange={(e) => setFaceSwap(e.target.value as typeof faceSwap)} className="mt-1 block w-full rounded border border-neutral-300 bg-white px-2 py-1"><option value="profile">profile default ({profile?.face_swap?.enabled ? 'on' : 'off'})</option><option value="on">on</option><option value="off">off</option></select></label>
            <label className="block text-sm"><span className="font-medium">Evaluate keyframes (Gemma vision)</span><select value={evaluate} onChange={(e) => setEvaluate(e.target.value as typeof evaluate)} className="mt-1 block w-full rounded border border-neutral-300 bg-white px-2 py-1"><option value="profile">profile default ({profile?.evaluate?.enabled ? 'on' : 'off'})</option><option value="on">on</option><option value="off">off</option></select></label>
            {swapOn && (
              <label className="block text-sm md:col-span-2">
                <span className="font-medium">Master face path</span>
                <input data-testid="master_face" value={masterFace} onChange={(e) => setMasterFace(e.target.value)} className="mt-1 block w-full rounded border border-neutral-300 px-2 py-1 font-mono text-xs" placeholder={masterAsset ? `project asset: ${masterAsset.name}` : '/path/to/face.jpg'} />
              </label>
            )}
          </div>
        </Card>
      )}

      {step === 3 && (
        <Card title="Review & submit">
          <dl className="grid gap-x-6 gap-y-1 text-sm md:grid-cols-[10rem_1fr]">
            <dt className="text-neutral-500">{source === 'animate' ? 'direction' : 'idea'}</dt>
            <dd>{idea || (source === 'animate' ? <em className="text-neutral-500">none</em> : <em className="text-red-700">missing</em>)}</dd>
            <dt className="text-neutral-500">project</dt><dd>{project.data?.name ?? '—'}</dd>
            <dt className="text-neutral-500">profile</dt><dd>{profile ? `${profile.name} — ${profileSummary(profile)}` : effectiveProfileId}</dd>
            <dt className="text-neutral-500">scenes / seed</dt>
            <dd>{source === 'animate' ? `${animateIds.length} (your images)` : (nScenes || 'auto')} / {seed || 'random'} · {autonomy}</dd>
            <dt className="text-neutral-500">mode</dt>
            <dd>{source === 'animate' ? 'animate my images — no keyframes generated'
                 : `${mode}${mode === 'edit' ? ` (${refIds.length} ref)` : ''}`}</dd>
            <dt className="text-neutral-500">face swap</dt><dd>{swapOn ? `on (${masterFace || masterAsset?.name || 'no face!'})` : 'off'}</dd>
            {renderCost && <><dt className="text-neutral-500">render time</dt><dd data-testid="job-cost">{renderCost}</dd></>}
          </dl>
          <div className="mt-3 space-y-2">
            {problems.map((p) => <ErrorBox key={p} message={p} />)}
            <div className="rounded border border-neutral-200 p-2">
              <div className="mb-1 text-xs font-semibold uppercase text-neutral-500">Environment check {env.isFetching ? '(checking…)' : ''}</div>
              <EnvSummary report={env.data} />
              {envFailures.length > 0 && (
                <label className="mt-2 flex items-center gap-2 text-sm"><input data-testid="force" type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} /> submit anyway (force)</label>
              )}
            </div>
            <ErrorBox message={submitError} />
            {submitReport?.checks && <EnvSummary report={submitReport} />}
          </div>
        </Card>
      )}

      <div className="flex justify-between">
        <Button disabled={step === 0} onClick={() => setStep(step - 1)}>Back</Button>
        {step < 3 ? (
          <Button variant="primary" onClick={() => setStep(step + 1)}>Next</Button>
        ) : (
          <Button data-testid="submit" variant="primary" disabled={blocked || create.isPending} onClick={submit}>{create.isPending ? 'Submitting…' : 'Start job'}</Button>
        )}
      </div>
    </div>
  )
}
